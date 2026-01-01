import sqlite3
import bcrypt


DB_NAME = "users.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password BLOB
        )
    """)
    conn.commit()
    conn.close()


def create_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_pw)
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False


def check_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "SELECT password FROM users WHERE username = ?",
        (username,)
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        return False

    return bcrypt.checkpw(password.encode(), row[0])