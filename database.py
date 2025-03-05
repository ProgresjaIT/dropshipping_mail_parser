#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database connection and query functions.
"""

import os
import pymysql


def connect_to_database():
    """
    Connect to MySQL database using credentials from .env file.

    Returns:
        pymysql.connections.Connection: Database connection object or None if failed
    """
    try:
        connection = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        print("Database connection established successfully")
        return connection

    except pymysql.Error as e:
        print(f"Database connection error: {e}")
        return None


def get_email_configurations(connection):
    """
    Get all email configurations from the database.

    Args:
        connection (pymysql.connections.Connection): Database connection object

    Returns:
        list: List of email configurations as dictionaries
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM email_konfiguracja")
            email_configs = cursor.fetchall()
            return email_configs

    except pymysql.Error as e:
        print(f"Failed to fetch email configurations: {e}")
        return []
