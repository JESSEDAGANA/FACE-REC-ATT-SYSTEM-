import sqlite3
from contextlib import contextmanager
import os
from datetime import datetime

DATABASE = 'attendance.db'

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    user_id INTEGER UNIQUE NOT NULL)''')
        
        # Attendance table
        c.execute('''CREATE TABLE IF NOT EXISTS attendance
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    time TEXT NOT NULL,
                    date TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')
        
        conn.commit()

def add_user(username, user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, user_id) VALUES (?, ?)",
                 (username, user_id))
        conn.commit()
        return c.lastrowid

def add_attendance(user_id):
    with get_db() as conn:
        c = conn.cursor()
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        c.execute("INSERT INTO attendance (user_id, time, date) VALUES (?, ?, ?)",
                 (user_id, current_time, current_date))
        conn.commit()

def get_today_attendance():
    current_date = datetime.now().strftime("%Y-%m-%d")
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''SELECT u.username, u.user_id, a.time 
                    FROM attendance a
                    JOIN users u ON a.user_id = u.id
                    WHERE a.date = ?
                    ORDER BY a.time DESC''', (current_date,))
        return c.fetchall()

def get_all_users():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, user_id FROM users")
        return c.fetchall()

def get_user_by_id(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, user_id FROM users WHERE user_id = ?", (user_id,))
        return c.fetchone()

def user_exists(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        return c.fetchone() is not None