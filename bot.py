import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import database
import rag_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dictionary to maintain last 3 interactions per user 
user_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am a Mini-RAG Bot. Send me a PDF, then use /ask <query> to ask questions about it!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = "📖 **How to use me:**\n\n1. **Upload a PDF** to add it to my knowledge base.\n2. Use `/ask <your question>` to query the document.\n\n*Example:* `/ask What is the refund policy?`"
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    doc_name = update.message.document.file_name
    
    if not doc_name.endswith('.pdf'):
        await update.message.reply_text(" Please upload a PDF file.")
        return

    await update.message.reply_text(f" Downloading and processing {doc_name}...")
    file_path = f"downloads_{doc_name}"
    await file.download_to_drive(file_path)
    
    chunks_created = rag_engine.process_pdf(file_path, doc_name)
    os.remove(file_path) # Clean up
    
    await update.message.reply_text(f" Successfully processed {doc_name} into {chunks_created} chunks and stored them in the local database!")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a query. Example: `/ask What is the company policy?`", parse_mode='Markdown')
        return

    user_id = update.message.from_user.id
    history = user_history.get(user_id, [])
    history_context = " | ".join(history)

    await update.message.reply_text(">>> Thinking...")
    
    try:
        answer = rag_engine.query_llm(query, history_context)
        
        # Update message history (keep last 3)
        history.append(f"Q: {query} A: {answer[:50]}...")
        if len(history) > 3:
            history.pop(0)
        user_history[user_id] = history

        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f" Error: {str(e)}. Make sure Ollama is running!")

# --- NEW HANDLER FOR INVALID QUERIES ---
async def invalid_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This triggers if a user sends plain text WITHOUT the /ask command
    msg = (
        " **Invalid Query Pattern!**\n\n"
        "I only answer questions that start with the `/ask` command.\n"
        "Example: `/ask who is jehangir ayoub?`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

if __name__ == '__main__':
    # Initialize the database
    database.init_db()

    # --- TOKEN CONFIGURATION ---
    MY_REAL_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not MY_REAL_TOKEN:
        print(" FATAL ERROR: TELEGRAM_BOT_TOKEN not found in .env file!")
    else:
        # Create the application
        app = Application.builder().token(MY_REAL_TOKEN).build()

        # 1. Register Command Handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("ask", ask_command))

        # 2. Register Document Handler
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

        # 3. Register Catch-all Handler (MUST BE LAST)
        # This catches any text that isn't a command or a document
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), invalid_query))

        print(">>>> Bot is running using .env token! Strict mode active<<<<.")
        app.run_polling()
