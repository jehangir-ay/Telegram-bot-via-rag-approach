import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import database

# Initialize local embedding model 
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Basic caching dictionary 
query_cache = {}

def process_pdf(file_path, doc_name):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Split into chunks of approx 500 characters [cite: 23]
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in chunks:
        if len(chunk.strip()) > 10:
            embedding = embedder.encode(chunk)
            database.store_chunk(doc_name, chunk, embedding)
    
    return len(chunks)

def query_llm(user_query, history_context=""):
    # Check cache first 
    if user_query in query_cache:
        return query_cache[user_query]

    # 1. Embed user query
    query_embedding = embedder.encode(user_query)
    
    # 2. Retrieve top-k chunks [cite: 29]
    top_chunks = database.retrieve_top_k(query_embedding, k=3)
    
    if not top_chunks:
        return "I don't have any documents uploaded yet. Please upload a PDF first."

    # 3. Build context 
    context = ""
    sources = set()
    for sim, doc_name, text in top_chunks:
        context += f"---\n{text}\n"
        sources.add(doc_name)
    
    source_snippets = ", ".join(sources)

    prompt = f"""You are a helpful assistant answering questions based strictly on the provided context.
    Previous Conversation Context: {history_context}
    
    Knowledge Context:
    {context}
    
    Question: {user_query}
    You are a professional assistant. Use ONLY the following pieces of retrieved context to answer the user's question. 
    
    STRICT RULES:
    1. If the answer is not contained within the context below, simply state: "I'm sorry, but I don't have information about that in my current database."
    2. Do NOT use any outside knowledge.
    3. Do NOT make up facts or "hallucinate".
    4. Keep the answer concise and based on the provided documents.:"""

    # 4. Call local Ollama LLM [cite: 31, 66]
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    
    if response.status_code == 200:
        answer = response.json().get("response", "")
        # Append source snippets [cite: 49]
        final_answer = f"{answer}\n\n*Sources: {source_snippets}*"
        query_cache[user_query] = final_answer
        return final_answer
    else:
        return "Error connecting to local LLM."
