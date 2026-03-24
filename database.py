import sqlite3
import numpy as np
import json

DB_NAME = "rag_store.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Table for document chunks + embeddings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            chunk_text TEXT,
            embedding BLOB
        )
    ''')

    # Table for persistent user session history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            user_id INTEGER PRIMARY KEY,
            history TEXT NOT NULL DEFAULT '[]'
        )
    ''')

    conn.commit()
    conn.close()

def store_chunk(doc_name, chunk_text, embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
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
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        return []

    for row in rows:
        doc_name, chunk_text, emb_bytes = row
        doc_vec = np.frombuffer(emb_bytes, dtype=np.float32)
        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm == 0:
            continue
        similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
        results.append((similarity, doc_name, chunk_text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:k]

def get_user_history(user_id: int) -> list:
    """Load the last 3 interactions for a user from the DB."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT history FROM user_history WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []

def save_user_history(user_id: int, history: list):
    """Persist user history (last 3 interactions) to the DB."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_history (user_id, history) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET history = excluded.history",
        (user_id, json.dumps(history))
    )
    conn.commit()
    conn.close()

def clear_user_history(user_id: int):
    """Clear history for a user (called on /start)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
