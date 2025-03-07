import os
import pymysql
import sqlite3
from dotenv import load_dotenv
load_dotenv()

def connect_to_sqlite_database():
    try:
        connection = sqlite3.connect("database.db")
        connection.row_factory = sqlite3.Row
        print("Database connection established successfully")
        return connection

    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def create_tables():
    connection = connect_to_sqlite_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            subject TEXT,
            body TEXT,
            metadata TEXT,
            email_account TEXT NOT NULL,
            message_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mail_id INTEGER NOT NULL,
            attachment_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mail_id) REFERENCES emails (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS
                   contractors_data (
                   id INTEGER PRIMARY KEY,
                   id_raks TEXT NOT NULL,
                   nazwa TEXT,
                   email TEXT,
                   skrot_raks TEXT,
                   formaty_plikow TEXT
                   )
    ''')
    connection.commit()
    connection.close()

def save_email(connection, sender, subject, body, metadata, email_account, message_id):
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO emails (sender, subject, body, metadata, email_account, message_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (sender, subject, body, metadata, email_account, message_id))
    connection.commit()
    return cursor.lastrowid

def save_email_attachment(connection, mail_id, attachment_name, file_path):
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO attachments (mail_id, attachment_name, file_path)
        VALUES (?, ?, ?)
    ''', (mail_id, attachment_name, file_path))
    connection.commit()

def connect_to_database():
    try:
        connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            cursorclass=pymysql.cursors.DictCursor,
        )
        print("Database connection established successfully")
        return connection

    except pymysql.Error as e:
        print(f"Database connection error: {e}")
        return None

def get_email_configurations():
    connection = connect_to_database()
    if not connection:
        return []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM email_konfiguracja WHERE login = 'dropshipping@progresja.eu'")
            email_configs = cursor.fetchall()
            return email_configs

    except pymysql.Error as e:
        print(f"Failed to fetch email configurations: {e}")
        return []
