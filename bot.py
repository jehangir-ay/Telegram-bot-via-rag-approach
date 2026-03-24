import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import database
import rag_engine
from dotenv import load_dotenv

# Dictionary to maintain last 3 interactions per user 
user_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am a Mini-RAG Bot. Send me a PDF, then use /ask <query> to ask questions about it!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = "/ask <query> - Ask a question about the uploaded documents\n/help - Show this message\nSimply upload a PDF to add it to my knowledge base."
    await update.message.reply_text(help_text)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    doc_name = update.message.document.file_name
    
    if not doc_name.endswith('.pdf'):
        await update.message.reply_text("Please upload a PDF file.")
        return

    await update.message.reply_text(f"Downloading and processing {doc_name}...")
    file_path = f"downloads_{doc_name}"
    await file.download_to_drive(file_path)
    
    chunks_created = rag_engine.process_pdf(file_path, doc_name)
    os.remove(file_path) # Clean up
    
    await update.message.reply_text(f" Successfully processed {doc_name} into {chunks_created} chunks and stored them in the local database!")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please provide a query. Example: /ask What is the company policy?")
        return

    user_id = update.message.from_user.id
    history = user_history.get(user_id, [])
    history_context = " | ".join(history)

    await update.message.reply_text("🔍 Thinking...")
    
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
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("ask", ask_command))
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

        print(" Bot is running using .env token!")
        app.run_polling()
