import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import database
import rag_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    # Clear history on /start so the new session starts fresh
    database.clear_user_history(user_id)
    await update.message.reply_text(
        "Hello! I am a Mini-RAG Bot.\n\n"
        " Send me a PDF to add it to my knowledge base.\n"
        " Then use /ask <query> to ask questions about it!\n\n"
        "Use /help for full instructions."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        " *How to use me:*\n\n"
        "1. *Upload a PDF* to add it to my knowledge base.\n"
        "2. Use `/ask <your question>` to query the document.\n\n"
        "_Example:_ `/ask What is the refund policy?`\n\n"
        " Your last 3 interactions are remembered per session.\n"
        " Send /start to reset your session history."
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    doc_name = update.message.document.file_name

    if not doc_name.endswith('.pdf'):
        await update.message.reply_text(" Please upload a PDF file.")
        return

    await update.message.reply_text(f" Downloading and processing *{doc_name}*...", parse_mode='Markdown')

    # Ensure downloads directory exists
    os.makedirs("downloads", exist_ok=True)
    file_path = os.path.join("downloads", doc_name)

    await file.download_to_drive(file_path)

    try:
        chunks_created = rag_engine.process_pdf(file_path, doc_name)
        await update.message.reply_text(
            f" Successfully processed *{doc_name}* into {chunks_created} chunks and stored them in the database!",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f" Failed to process PDF: {str(e)}")
    finally:
        # Always clean up the downloaded file
        if os.path.exists(file_path):
            os.remove(file_path)

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text(
            "Please provide a query.\nExample: `/ask What is the company policy?`",
            parse_mode='Markdown'
        )
        return

    user_id = update.message.from_user.id

    # Load persistent history from DB
    history = database.get_user_history(user_id)
    history_context = " | ".join(history)

    await update.message.reply_text("🔍 Thinking...")

    try:
        answer = rag_engine.query_llm(query, history_context)

        # Persist updated history (keep last 3)
        history.append(f"Q: {query} A: {answer[:50]}...")
        if len(history) > 3:
            history.pop(0)
        database.save_user_history(user_id, history)

        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(
            f" Error: {str(e)}\n\nMake sure Ollama is running at the configured URL!"
        )

async def invalid_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        " *Invalid Query Pattern!*\n\n"
        "I only answer questions that start with the `/ask` command.\n"
        "Example: `/ask who is jehangir ayoub?`\n\n"
        "Use /help for full instructions."
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

if __name__ == '__main__':
    # Initialize the database (creates tables if they don't exist)
    database.init_db()

    MY_REAL_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not MY_REAL_TOKEN:
        print(" FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file!")
    else:
        app = Application.builder().token(MY_REAL_TOKEN).build()

        # Command handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("ask", ask_command))

        # Document handler
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

        # Catch-all for plain text (MUST be last)
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), invalid_query))

        print("✅ Bot is running! Strict mode active.")
        app.run_polling()
