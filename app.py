#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main application for fetching unread emails from IMAP accounts.
"""

import os
from dotenv import load_dotenv
import database
import functions

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
MESSAGES_DIR = os.path.join(BASE_DIR, "messages")


def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()

    # Create output directories if they don't exist
    os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
    os.makedirs(MESSAGES_DIR, exist_ok=True)

    # Connect to database
    connection = database.connect_to_database()
    if not connection:
        print("Failed to connect to database. Exiting.")
        return

    try:
        # Get email configurations
        email_configs = database.get_email_configurations(connection)
        print(f"Found {len(email_configs)} email configurations")

        # Process each email account
        new_emails_count = 0
        for config in email_configs:
            new_count = functions.process_email_account(
                config, MESSAGES_DIR, ATTACHMENTS_DIR
            )
            new_emails_count += new_count

        print(f"Total new unread emails found: {new_emails_count}")

    finally:
        # Close database connection
        connection.close()
        print("Database connection closed")


if __name__ == "__main__":
    main()
