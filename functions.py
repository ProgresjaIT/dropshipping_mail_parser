#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core functions for processing emails from IMAP accounts.
"""

import os
import imaplib
import email
from email.header import decode_header
from datetime import datetime


def process_email_account(config, messages_dir, attachments_dir):
    """
    Process a single email account configuration.
    
    Args:
        config (dict): Email account configuration
        messages_dir (str): Directory to save message content
        attachments_dir (str): Directory to save attachments
    
    Returns:
        int: Number of new unread emails found
    """
    print(f"\nProcessing account: {config['email']}")
    
    # Extract account details
    imap_server = config['imap']
    port = int(config.get('port', 993))
    username = config['login']
    password = config['haslo']
    email_address = config['email']
    
    # Create account directory
    account_dir = os.path.join(messages_dir, email_address.replace('@', '_at_'))
    os.makedirs(account_dir, exist_ok=True)
    
    # Connect to IMAP server
    try:
        mail = connect_to_imap(imap_server, port, username, password)
        if not mail:
            return 0
        
        # Search for unread messages
        message_ids = get_unread_messages(mail)
        if not message_ids:
            print("No new unread messages")
            mail.logout()
            return 0
            
        print(f"Found {len(message_ids)} unread messages")
        
        # Process each message
        for msg_id in message_ids:
            process_email_message(mail, msg_id, account_dir, attachments_dir, email_address)
            
        mail.logout()
        return len(message_ids)
        
    except Exception as e:
        print(f"Error processing account {email_address}: {e}")
        return 0


def connect_to_imap(server, port, username, password):
    """
    Connect to an IMAP server.
    
    Args:
        server (str): IMAP server address
        port (int): IMAP server port
        username (str): IMAP username
        password (str): IMAP password
    
    Returns:
        imaplib.IMAP4: IMAP connection object or None if failed
    """
    try:
        if port == 993:
            mail = imaplib.IMAP4_SSL(server, port)
        else:
            mail = imaplib.IMAP4(server, port)
            try:
                mail.starttls()
            except:
                print("Warning: STARTTLS failed")
                
        mail.login(username, password)
        mail.select('INBOX')
        return mail
        
    except Exception as e:
        print(f"Failed to connect to IMAP server {server}: {e}")
        return None


def get_unread_messages(mail):
    """
    Get IDs of unread messages.
    
    Args:
        mail (imaplib.IMAP4): IMAP connection object
    
    Returns:
        list: List of message IDs
    """
    status, messages = mail.search(None, 'UNSEEN')
    if status != 'OK':
        print(f"Search failed: {status}")
        return []
        
    return messages[0].split()


def process_email_message(mail, msg_id, account_dir, attachments_dir, email_address):
    """
    Process a single email message.
    
    Args:
        mail (imaplib.IMAP4): IMAP connection object
        msg_id (bytes): Message ID
        account_dir (str): Directory to save message content
        attachments_dir (str): Directory to save attachments
        email_address (str): Email account address
    """
    try:
        # Fetch message without changing its read status
        status, msg_data = mail.fetch(msg_id, '(RFC822)')
        if status != 'OK':
            print(f"Failed to fetch message {msg_id.decode()}")
            return
            
        # Parse email message
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        
        # Extract metadata
        subject = decode_email_subject(msg['Subject'])
        sender = msg.get('From', 'Unknown Sender')
        date = msg.get('Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Create message directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = ''.join(c if c.isalnum() else '_' for c in subject[:30])
        msg_dir = os.path.join(account_dir, f"{timestamp}_{safe_subject}")
        os.makedirs(msg_dir, exist_ok=True)
        
        # Save email content
        body = get_email_body(msg)
        with open(os.path.join(msg_dir, "message.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Subject: {subject}\n")
            f.write(f"From: {sender}\n")
            f.write(f"Date: {date}\n\n")
            f.write("--- BODY ---\n\n")
            f.write(body)
        
        # Save attachments
        attachments = save_attachments(msg, msg_dir)
        
        print(f"Saved message: {subject}")
        if attachments:
            print(f"Saved {len(attachments)} attachments")
            
    except Exception as e:
        print(f"Error processing message {msg_id.decode()}: {e}")


def decode_email_subject(subject):
    """
    Decode email subject with proper encoding.
    
    Args:
        subject (str): Email subject
    
    Returns:
        str: Decoded subject
    """
    if not subject:
        return "No Subject"
        
    decoded_subject = ""
    try:
        decoded_chunks = decode_header(subject)
        for chunk, encoding in decoded_chunks:
            if isinstance(chunk, bytes):
                decoded_subject += chunk.decode(encoding or 'utf-8', errors='replace')
            else:
                decoded_subject += str(chunk)
    except:
        decoded_subject = str(subject)
        
    return decoded_subject


def get_email_body(msg):
    """
    Extract the body from an email message.
    
    Args:
        msg (email.message.Message): Email message object
    
    Returns:
        str: Email body content
    """
    if msg.is_multipart():
        body = ""
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")
            
            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body += payload.decode(charset, errors='replace')
                    except:
                        body += "Cannot decode plain text content\n"
                        
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body += payload.decode(charset, errors='replace')
                    except:
                        body += "Cannot decode HTML content\n"
        return body
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or 'utf-8'
            return payload.decode(charset, errors='replace')
        except:
            return "Cannot decode message content\n"


def save_attachments(msg, msg_dir):
    """
    Save email attachments.
    
    Args:
        msg (email.message.Message): Email message object
        msg_dir (str): Directory to save attachments
    
    Returns:
        list: List of saved attachment paths
    """
    attachments = []
    
    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
            
        filename = part.get_filename()
        if not filename:
            continue
            
        # Decode filename if needed
        if decode_header(filename)[0][1] is not None:
            try:
                filename = decode_header(filename)[0][0].decode(decode_header(filename)[0][1])
            except:
                pass
        
        # Save attachment
        filepath = os.path.join(msg_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                f.write(part.get_payload(decode=True))
            attachments.append(filepath)
        except Exception as e:
            print(f"Error saving attachment {filename}: {e}")
    
    return attachments