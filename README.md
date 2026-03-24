

#  Mini-RAG Telegram Bot (Local LLM)

A lightweight Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about them via Telegram. This project was built as a Data Science assessment to demonstrate local LLM integration and efficient document retrieval[cite: 3, 7].

##  Quick Start Guide

### 1. Prerequisites
* **Python 3.10+**
* **Ollama**: [Download and install here](https://ollama.com/).
* **Telegram Bot Token**: Get one from [@BotFather](https://t.me/botfather).

### 2. Installation
Clone this repository and install the required libraries:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

pip install python-telegram-bot sentence-transformers PyPDF2 numpy requests python-dotenv
```

### 3. Setup Local LLM
Open your terminal and run:
```bash
ollama run llama3
```
*Note: Keep this terminal running in the background.*

### 4. Configuration
Create a `.env` file in the root directory:
```env
TELEGRAM_BOT_TOKEN=your_token_here
```

### 5. Run the Bot
```bash
python bot.py
```

---

##  How It Works (System Design)

The system follows a modular RAG architecture to ensure scalability and privacy:



1.  **Ingestion**: When a user uploads a PDF, the bot extracts text and splits it into chunks.
2.  **Embedding**: Text chunks are converted into 384-dimensional vectors using the `all-MiniLM-L6-v2` model.
3.  **Storage**: Vectors and text are stored in a local **SQLite** database for persistence.
4.  **Retrieval**: When a user sends `/ask <query>`, the query is embedded, and the system performs a **Cosine Similarity** search to find the top-k most relevant chunks
5.  **Generation**: The retrieved context is sent to the local **Llama 3** model (via Ollama) to generate a summarized, factual response.

---

##  Key Strategies & Methods

 **Chunking Strategy**: Uses **Fixed-Size Overlapping Chunking**. This ensures that context is not lost between page breaks or sentence splits
  **Semantic Search**: Unlike keyword search, this uses vector embeddings to understand the *meaning* of the user's question
  **Message History Awareness**: The bot maintains the last 3 interactions per user, allowing for natural follow-up questions
  **Basic Caching**: To improve efficiency, identical queries are handled using a cache to avoid redundant embedding calculations
  **Source Attribution**: Every response includes "Source Snippets" to show which part of the uploaded document was used

---

##  Advantages of this Approach

* **Data Privacy**: Everything runs locally on the host machine. No data is sent to external APIs like OpenAI.
* **Cost Effective**: Uses open-source models (Llama 3, MiniLM), meaning zero API costs
* **User Experience**: Features a simple Telegram interface with real-time "Thinking..." status and clear instructions
* **Efficiency**: The `all-MiniLM-L6-v2` model is extremely fast and lightweight, making it perfect for real-time bot interactions

---

## Bot Commands
* `/start` - Initialize the bot.
* `/help` - Show usage instructions.
* `/ask <query>` - Query the knowledge base
* **Upload PDF** - Automatically adds a document to the knowledge base

---
![t0](https://github.com/user-attachments/assets/900d69e7-b6eb-4bd4-a986-9c2b02f8bdaf)
![t1](https://github.com/user-attachments/assets/88602c7d-b9d7-45ad-a493-2c0062d217cf)


