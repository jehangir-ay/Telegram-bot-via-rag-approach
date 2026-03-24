import sqlite3
import numpy as np
import json

DB_NAME = "rag_store.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            chunk_text TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

def store_chunk(doc_name, chunk_text, embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Convert numpy array embedding to bytes for SQLite BLOB storage
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    cursor.execute(
        "INSERT INTO documents (doc_name, chunk_text, embedding) VALUES (?, ?, ?)",
        (doc_name, chunk_text, embedding_bytes)
    )
    conn.commit()
    conn.close()

def retrieve_top_k(query_embedding, k=3):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT doc_name, chunk_text, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    results = []
    query_vec = np.array(query_embedding, dtype=np.float32)
    
    for row in rows:
        doc_name, chunk_text, emb_bytes = row
        doc_vec = np.frombuffer(emb_bytes, dtype=np.float32)
        # Calculate Cosine Similarity
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        results.append((similarity, doc_name, chunk_text))
    
    # Sort by highest similarity and return top k 
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]
