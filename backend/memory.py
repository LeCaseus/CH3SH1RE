import sqlite3
import datetime

# stored in the data folder, gitignored
DB_PATH = "data/memory.db"


def init_db():
    # creates the database and table if they don't exist yet
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            user_input  TEXT,
            ai_response TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def save_memory(user_input: str, ai_response: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (timestamp, user_input, ai_response) VALUES (?, ?, ?)",
        (datetime.datetime.now().isoformat(), user_input, ai_response),
    )
    conn.commit()
    conn.close()


def get_recent_memories(limit: int = 6) -> list[dict]:
    # returns last N exchanges as role/content dicts ready for ask_llm
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT user_input, ai_response FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()

    # reverse to chronological order then format for llm
    messages = []
    for user_input, ai_response in reversed(rows):
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": ai_response})
    return messages


def search_memories(query: str, limit: int = 3) -> list[dict]:
    # simple keyword search — finds past conversations mentioning the query
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """SELECT user_input, ai_response FROM conversations
           WHERE user_input LIKE ? OR ai_response LIKE ?
           ORDER BY id DESC LIMIT ?""",
        (f"%{query}%", f"%{query}%", limit),
    )
    rows = c.fetchall()
    conn.close()

    messages = []
    for user_input, ai_response in reversed(rows):
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": ai_response})
    return messages


def clear_memories():
    # wipe all conversations — useful for a fresh start
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()
