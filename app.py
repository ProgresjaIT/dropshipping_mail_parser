import os
import database
import functions

print("Starting mail parser")
# Ensure database tables exist
database.create_tables()

print("Getting email configurations from database")
EMAIL_CONFIGS = database.get_email_configurations()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
os.makedirs(ATTACHMENTS_DIR, exist_ok=True)

for CONFIG in EMAIL_CONFIGS:
    print("Checking emails for acc:", CONFIG["email"])
    RAW_EMAILS = functions.get_unread_emails(CONFIG)
    if len(RAW_EMAILS) == 0:
        print("No new emails")
        continue

    print(f"Found {len(RAW_EMAILS)} new emails")

    # First save all emails to database
    DB_CONN = database.connect_to_sqlite_database()
    if not DB_CONN:
        print("Failed to connect to database")
        continue

    for EMAIL in RAW_EMAILS:
        print("Processing email:", EMAIL["meta"]["message_id"])

        # Save email first
        EMAIL_ID = database.save_email(
            DB_CONN,
            EMAIL["sender"],
            EMAIL["subject"],
            EMAIL["text"] or EMAIL["html"],
            str(EMAIL["meta"]),
            CONFIG["email"],
            EMAIL["meta"]["message_id"]
        )

        # Get attachments
        ATTACHMENTS = functions.get_email_attachments(EMAIL)
        for ATTACHMENT in ATTACHMENTS:
            ATTACHMENT_PATH = functions.save_attachment(ATTACHMENT, EMAIL_ID, ATTACHMENTS_DIR)
            if ATTACHMENT_PATH:
                database.save_email_attachment(DB_CONN, EMAIL_ID, ATTACHMENT["name"], ATTACHMENT_PATH)

        CONTRACTOR = functions.identify_contractor(EMAIL, ATTACHMENTS)
        if CONTRACTOR:
            print("\033[1;32m")  # Bold and green
            print("##########################")
            print("#                        #")
            print(f"#   KONTRAHENT FOUND DB    #")
            print(f"#   {CONTRACTOR['Nazwa']}".ljust(24) + "#")
            print("#                        #")
            print("##########################")
            print("\033[0m")  # Reset formatting

            ORDERS = functions.get_orders(EMAIL, ATTACHMENTS, CONTRACTOR)

    DB_CONN.close()

    # Clean old attachments - keep only last 10 emails worth
    functions.cleanup_old_attachments(ATTACHMENTS_DIR)
