import os
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import database
from dotenv import load_dotenv
from collections import OrderedDict

load_dotenv()

# --- Config from .env ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))

# Initialize local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# LRU-style cache with a max size so memory doesn't grow forever
query_cache = OrderedDict()

def _add_to_cache(key: str, value: str):
    """Add to cache, evicting oldest entry if over CACHE_MAX_SIZE."""
    if key in query_cache:
        query_cache.move_to_end(key)
    query_cache[key] = value
    if len(query_cache) > CACHE_MAX_SIZE:
        query_cache.popitem(last=False)

def _split_into_chunks(text: str, chunk_size: int) -> list:
    """
    Word-aware chunking: never cuts mid-word.
    Splits on whitespace boundaries closest to chunk_size characters.
    """
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1  # +1 for the space
        if current_len + word_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks

def process_pdf(file_path: str, doc_name: str) -> int:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        raise ValueError("Could not extract any text from the PDF. It may be scanned/image-based.")

    chunks = _split_into_chunks(text, CHUNK_SIZE)
    stored = 0
    for chunk in chunks:
        if len(chunk.strip()) > 10:
            embedding = embedder.encode(chunk)
            database.store_chunk(doc_name, chunk, embedding)
            stored += 1

    return stored

def query_llm(user_query: str, history_context: str = "") -> str:
    # Return cached answer if available
    cache_key = f"{history_context}|||{user_query}"
    if cache_key in query_cache:
        return query_cache[cache_key]

    # 1. Embed query
    query_embedding = embedder.encode(user_query)

    # 2. Retrieve top-k relevant chunks
    top_chunks = database.retrieve_top_k(query_embedding, k=3)

    if not top_chunks:
        return "I don't have any documents in my knowledge base yet. Please upload a PDF first."

    # 3. Build context string
    context = ""
    sources = set()
    for sim, doc_name, text in top_chunks:
        context += f"---\n{text}\n"
        sources.add(doc_name)

    source_list = ", ".join(sources)

    prompt = f"""You are a helpful assistant answering questions based strictly on the provided context.

Previous Conversation Context: {history_context}

Knowledge Context:
{context}

Question: {user_query}

STRICT RULES:
1. If the answer is not contained within the context above, reply: "I'm sorry, but I don't have information about that in my current database."
2. Do NOT use any outside knowledge.
3. Do NOT make up facts or hallucinate.
4. Keep the answer concise and based on the provided documents.

Answer:"""

    # 4. Call Ollama
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60  # prevent hanging indefinitely
        )
    except requests.exceptions.ConnectionError:
        return f" Could not connect to Ollama at `{OLLAMA_URL}`. Make sure it is running."
    except requests.exceptions.Timeout:
        return " The LLM took too long to respond. Please try again."

    if response.status_code == 200:
        answer = response.json().get("response", "").strip()
        final_answer = f"{answer}\n\n📄 *Sources: {source_list}*"
        _add_to_cache(cache_key, final_answer)
        return final_answer
    else:
        return f" Ollama returned an error (HTTP {response.status_code}). Check that the model `{OLLAMA_MODEL}` is available."
