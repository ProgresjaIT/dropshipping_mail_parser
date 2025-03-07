import os
import imaplib
import email
import json
import re
from email.header import decode_header
from datetime import datetime
import ocrmypdf
import database
import subprocess
import tempfile
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
import sys
# Load environment variables
load_dotenv()

def connect_to_imap(server, port, username, password):
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
        mail.select("INBOX")
        return mail

    except Exception as e:
        print(f"Failed to connect to IMAP server {server}: {e}")
        return None

def get_unread_messages(mail):
    status, messages = mail.search(None, "UNSEEN")
    if status != "OK":
        print(f"Search failed: {status}")
        return []
    return messages[0].split()

def decode_email_subject(subject):
    if not subject:
        return "No Subject"

    decoded_subject = ""
    try:
        decoded_chunks = decode_header(subject)
        for chunk, encoding in decoded_chunks:
            if isinstance(chunk, bytes):
                decoded_subject += chunk.decode(encoding or "utf-8", errors="replace")
            else:
                decoded_subject += str(chunk)
    except:
        decoded_subject = str(subject)

    return decoded_subject

def get_unread_emails(config):
    mail = connect_to_imap(config["imap"],
                        int(config.get("port", 993)),
                        config["login"],
                        config["haslo"])
    if not mail:
        return []

    message_ids = get_unread_messages(mail)
    mail.logout()

    if not message_ids:
        return []

    mail = connect_to_imap(config["imap"],
                        int(config.get("port", 993)),
                        config["login"],
                        config["haslo"])
    if not mail:
        return []

    messages = []
    for msg_id in message_ids:
        msg_dict = get_message(mail, msg_id)
        if msg_dict:
            messages.append(msg_dict)

    mail.logout()
    return messages

def get_message(mail, id):
    status, msg_data = mail.fetch(id, "(RFC822)")
    if status != "OK":
        print(f"Failed to fetch message {id.decode()}")
        return None

    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    text_content = ""
    html_content = ""
    attachments = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition") or "")

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachments.append({
                        "name": filename,
                        "data": part.get_payload(decode=True),
                        "content_type": part.get_content_type()
                    })
            else:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        decoded_content = payload.decode(charset, errors="replace")
                        if content_type == "text/plain":
                            text_content += decoded_content
                        elif content_type == "text/html":
                            html_content += decoded_content
                except:
                    continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    html_content = content
                else:
                    text_content = content
        except:
            pass

    return {
        "sender": msg.get("From", "Unknown Sender"),
        "subject": decode_email_subject(msg["Subject"]),
        "html": html_content,
        "text": text_content,
        "attachments": attachments,
        "meta": {
            "message_id": msg.get("Message-ID", ""),
            "to": msg.get("To", ""),
            "cc": msg.get("Cc", ""),
            "content_type": msg.get_content_type()
        }
    }

def get_email_attachments(email_dict):
    return email_dict.get("attachments", [])

def save_attachment(attachment, email_id, attachments_dir):
    try:
        _, ext = os.path.splitext(attachment["name"])
        if not ext:
            ext = '.unknown'
            
        filename = f"{email_id}_{len(os.listdir(attachments_dir))}{ext.lower()}"
        filepath = os.path.join(attachments_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(attachment["data"])
        return filepath
    except Exception as e:
        print(f"Error saving attachment {attachment['name']}: {e}")
        return None

def cleanup_old_attachments(attachments_dir):
    try:
        files = os.listdir(attachments_dir)
        if len(files) <= 10:
            return

        # Filter out files that don't match the expected format
        valid_files = []
        for file in files:
            try:
                if '_' in file:
                    int(file.split('_')[0])  # Test if it can be converted to int
                    valid_files.append(file)
            except ValueError:
                continue  # Skip files that don't have a number before the underscore

        if len(valid_files) <= 10:
            return

        valid_files.sort(key=lambda x: int(x.split('_')[0]))
        files_to_remove = valid_files[:-10]  # Keep the 10 newest files
        
        for file in files_to_remove:
            os.remove(os.path.join(attachments_dir, file))

    except Exception as e:
        print(f"Error cleaning up attachments: {e}")


def identify_contractor(EMAIL, ATTACHMENTS):
    """
    Identifies a contractor based on email attachments and sender information.

    Parameters:
    EMAIL (dict): Dictionary containing email information, including 'sender'
    ATTACHMENTS (list): List of attachment dictionaries from the email

    Returns:
    dict: Contractor information from database or None if not found
    """
    print(f"Attempting to identify contractor for email from: {EMAIL['sender']}")

    # Connect to MySQL database
    connection = database.connect_to_database()
    if not connection:
        print("Failed to connect to database")
        return None

    try:
        with connection.cursor() as cursor:
            # 1. First check if sender email matches Email field in kontrahenci table
            sender_email_address = EMAIL["sender"]
            # Extract just the email address from a field like "Name <email@domain.com>"
            import re

            match = re.search(r"<([^>]+)>", sender_email_address)
            if match:
                email_address = match.group(1)
            else:
                email_address = (
                    sender_email_address  # If no <> format, use whole string
                )

            # Search for contractor by email address
            cursor.execute(
                "SELECT * FROM kontrahenci WHERE Email LIKE %s", (f"%{email_address}%",)
            )
            contractor = cursor.fetchone()

            if contractor:
                print(f"Found contractor by email address: {contractor['Nazwa']}")
                return contractor

            # 2. If not found by email, try to find by company name in subject or content
            subject = EMAIL["subject"]
            content = EMAIL["text"] or EMAIL["html"] or ""

            # Get all contractors to check their names in content
            cursor.execute("SELECT * FROM kontrahenci")
            all_contractors = cursor.fetchall()

            for potential_contractor in all_contractors:
                contractor_name = potential_contractor["Nazwa"]
                if contractor_name and (
                    contractor_name.lower() in subject.lower()
                    or contractor_name.lower() in content.lower()
                ):
                    print(f"Found contractor by name in content: {contractor_name}")
                    return potential_contractor

            # 3. If we have attachments, try to find contractor by filenames
            if ATTACHMENTS and len(ATTACHMENTS) > 0:
                for attachment in ATTACHMENTS:
                    file_name = attachment.get("name", "")

                    # Also check allowed file formats for this contractor
                    for potential_contractor in all_contractors:
                        contractor_name = potential_contractor["Nazwa"]
                        raks_shortcut = potential_contractor["Skrot_Raks"]
                        allowed_formats = potential_contractor["Formaty_Plikow"]

                        # Check if contractor name or RAKS shortcut is in filename
                        name_match = (
                            contractor_name
                            and contractor_name.lower() in file_name.lower()
                        ) or (
                            raks_shortcut and raks_shortcut.lower() in file_name.lower()
                        )

                        # Check if file extension matches allowed formats for this contractor
                        format_match = False
                        if allowed_formats:
                            file_ext = os.path.splitext(file_name)[1].lower()
                            allowed_formats_list = allowed_formats.lower().split(",")
                            format_match = any(
                                ext.strip() in file_ext for ext in allowed_formats_list
                            )

                        if name_match or format_match:
                            print(
                                f"Found contractor by filename or format: {contractor_name}"
                            )
                            return potential_contractor

            print("Failed to identify contractor")
            return None

    except Exception as e:
        print(f"Error while identifying contractor: {e}")
        return None
    finally:
        connection.close()

# Initialize LangChain with OpenAI
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define output schemas for different extraction tasks
class OrderInfo(BaseModel):
    order_number: Optional[str] = Field(description="Order number/identifier")
    order_date: Optional[str] = Field(description="Date when the order was placed")
    payment_terms: Optional[str] = Field(description="Payment terms or due date")
    notes: Optional[str] = Field(description="Any additional notes or comments about the order")

class OrderItem(BaseModel):
    contractor_product_code: str = Field(description="Product code used by the contractor (Index_Obcy)")
    quantity: str = Field(description="Quantity of the product ordered")
    dimensions: Optional[str] = Field(description="Dimensions of the product if available (e.g., length x width)")

class ShippingInfo(BaseModel):
    company: Optional[str] = Field(description="Company name for shipping")
    contact_person: Optional[str] = Field(description="Contact person name")
    phone: Optional[str] = Field(description="Phone number")
    street: Optional[str] = Field(description="Street address")
    building_number: Optional[str] = Field(description="Building number")
    apartment_number: Optional[str] = Field(description="Apartment or suite number")
    postal_code: Optional[str] = Field(description="Postal/ZIP code")
    city: Optional[str] = Field(description="City")
    country: Optional[str] = Field(description="Country")
    eu_code: Optional[str] = Field(description="EU country code if applicable")

def wybierz_zalacznik_zamowienia(ATTACHMENTS, CONTRACTOR):
    """
    Selects the appropriate attachment containing the order based on format and name.
    
    Parameters:
    ATTACHMENTS (list): List of attachment dictionaries from the email
    CONTRACTOR (dict): Information about the contractor
    
    Returns:
    dict: Selected attachment containing the order or None
    """
    if not ATTACHMENTS or len(ATTACHMENTS) == 0:
        print("No attachments found")
        return None
    
    # Get allowed file formats for this contractor
    allowed_formats = CONTRACTOR.get("Formaty_Plikow", "").lower().split(",") if CONTRACTOR.get("Formaty_Plikow") else []
    allowed_formats = [fmt.strip() for fmt in allowed_formats]
    
    # If no specific formats are defined, use common document formats
    if not allowed_formats:
        allowed_formats = [".pdf", ".xls", ".xlsx", ".csv", ".doc", ".docx"]
    
    # First, try to find attachments with allowed formats
    for attachment in ATTACHMENTS:
        file_name = attachment.get("name", "")
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Check if extension is in allowed formats
        if any(ext in file_ext for ext in allowed_formats):
            # Check if filename contains keywords related to orders
            order_keywords = ["order", "zamówienie", "zamowienie", "bestellung", "commande", "pedido", "ordine"]
            if any(keyword.lower() in file_name.lower() for keyword in order_keywords):
                print(f"Selected attachment by order keyword and format: {file_name}")
                return attachment
    
    # If no attachment with order keywords found, just return the first one with allowed format
    for attachment in ATTACHMENTS:
        file_name = attachment.get("name", "")
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if any(ext in file_ext for ext in allowed_formats):
            print(f"Selected first attachment with allowed format: {file_name}")
            return attachment
    
    # If still no match, return the first attachment
    if ATTACHMENTS:
        print(f"No attachment with allowed format found, using first attachment: {ATTACHMENTS[0].get('name', '')}")
        return ATTACHMENTS[0]
    
    return None

def przetworz_zalacznik(attachment):
    """
    Processes the attachment based on its type and extracts content.
    
    Parameters:
    attachment (dict): Dictionary with information about the attachment
    
    Returns:
    str/dict: Processed text or data from the attachment
    """
    if not attachment:
        return None
    
    file_name = attachment.get("name", "")
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # Create a temporary file to save the attachment
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
        temp_file.write(attachment.get("data", b""))
        temp_path = temp_file.name
    
    try:
        # Process based on file type
        if file_ext in [".pdf"]:
            # First try OCR with ocrmypdf
            ocr_output = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
            # try:
            ocrmypdf.ocr(temp_path, ocr_output, force_ocr=True)
                # Extract text from OCR'd PDF
            # text = subprocess.check_output(["pdftotext", ocr_output, "-"]).decode("utf-8", errors="replace")
            # except Exception as e:
            # print(f"OCRmyPDF failed: {e}, falling back to pytesseract")
            # Fallback to pytesseract
            text = ""
            try:
                images = convert_from_path(temp_path)
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n\n"
            except Exception as inner_e:
                print(f"Pytesseract fallback failed: {inner_e}")
                return None
            print(text)
            return text
            
        elif file_ext in [".xls", ".xlsx"]:
            # Process Excel files
            try:
                df_dict = pd.read_excel(temp_path, sheet_name=None)
                result = {}
                
                # Convert each sheet to a dictionary
                for sheet_name, df in df_dict.items():
                    result[sheet_name] = df.to_dict(orient="records")
                
                return result
            except Exception as e:
                print(f"Excel processing failed: {e}")
                return None
                
        elif file_ext in [".csv"]:
            # Process CSV files
            try:
                df = pd.read_csv(temp_path)
                return df.to_dict(orient="records")
            except Exception as e:
                print(f"CSV processing failed: {e}")
                try:
                    # Try with different encoding and delimiter
                    df = pd.read_csv(temp_path, encoding="latin1", sep=";")
                    return df.to_dict(orient="records")
                except Exception as e2:
                    print(f"Alternative CSV processing failed: {e2}")
                    return None
                    
        elif file_ext in [".doc", ".docx"]:
            # Process Word documents
            try:
                # Use textract if available, otherwise fallback
                import textract
                text = textract.process(temp_path).decode("utf-8", errors="replace")
                return text
            except ImportError:
                print("textract not available, trying docx2txt")
                try:
                    import docx2txt
                    text = docx2txt.process(temp_path)
                    return text
                except ImportError:
                    print("docx2txt not available, falling back to basic text extraction")
                    with open(temp_path, "r", errors="replace") as f:
                        return f.read()
        else:
            # For other file types, try to read as text
            try:
                with open(temp_path, "r", errors="replace") as f:
                    return f.read()
            except Exception as e:
                print(f"Failed to read file as text: {e}")
                return None
    
    except Exception as e:
        print(f"Error processing attachment {file_name}: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_path)
            if 'ocr_output' in locals():
                os.unlink(ocr_output)
        except:
            pass

def wyodrebnij_informacje_zamowienia(content, EMAIL, CONTRACTOR):
    """
    Extracts main order information from content.
    
    Parameters:
    content: Processed text or data from the attachment
    EMAIL (dict): Dictionary with email information
    CONTRACTOR (dict): Information about the contractor
    
    Returns:
    dict: Dictionary with main order information
    """
    # Create a prompt for extracting order information
    order_info_prompt = ChatPromptTemplate.from_template(
        """
    You are an expert in extracting order information from documents in multiple languages.
    
    Extract the following information from the provided text:
    - Order number/identifier which is called "LIEFERSCHEIN"
    - Order date
    - Total price
    - Currency
    - Payment terms
    - Any notes or comments about the order
    
    The text may be in various languages including English, Polish, German, French, Italian, or Spanish.
    
    Text:
    {content}
    
    Additional context:
    - Email subject: {subject}
    - Contractor name: {contractor_name}
    
    Return the extracted information in JSON format.
    """
    )

    # Create a parser for the output
    parser = JsonOutputParser(pydantic_object=OrderInfo)

    # Create the chain
    chain = order_info_prompt | llm | parser

    # Process the content
    if isinstance(content, str):
        # For text content
        text_to_process = content
    elif isinstance(content, dict):
        # For structured content (e.g., from Excel)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    elif isinstance(content, list):
        # For list content (e.g., from CSV)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        print(f"Unsupported content type: {type(content)}")
        return {}

    # Limit text length to avoid token limits
    max_length = 15000  # Adjust based on model's context window
    if len(text_to_process) > max_length:
        text_to_process = text_to_process[:max_length]
    print(text_to_process)
    try:
        # Run the extraction
        result = chain.invoke({
            "content": text_to_process,
            "subject": EMAIL.get("subject", ""),
            "contractor_name": CONTRACTOR.get("Nazwa", "")
        })

        # Handle both Pydantic object and dictionary responses
        if isinstance(result, dict):
            # It's a dictionary
            order_info = {
                "NR_Zamowienia": result.get("order_number"),
                "Data_Wyslania": result.get("order_date"),
                "Cena_Zamowienia": result.get("total_price"),
                "Waluta": result.get("currency"),
                "Termin_Platnosci": result.get("payment_terms"),
                "Uwagi": result.get("notes")
            }
        else:
            # It's a Pydantic object
            order_info = {
                "NR_Zamowienia": getattr(result, "order_number", None),
                "Data_Wyslania": getattr(result, "order_date", None),
                "Cena_Zamowienia": getattr(result, "total_price", None),
                "Waluta": getattr(result, "currency", None),
                "Termin_Platnosci": getattr(result, "payment_terms", None),
                "Uwagi": getattr(result, "notes", None)
            }

        return order_info

    except Exception as e:
        print(f"Error extracting order information: {e}")
        return {}

def wyodrebnij_pozycje_zamowienia(content, CONTRACTOR):
    """
    Extracts information about all products from the order.
    
    Parameters:
    content: Processed text or data from the attachment
    CONTRACTOR (dict): Information about the contractor
    
    Returns:
    list: List of dictionaries with information about individual items
    """
    # Create a prompt for extracting order items
    order_items_prompt = ChatPromptTemplate.from_template("""
    You are an expert in extracting product information from order documents in multiple languages.
    
    Extract all product items from the provided text. For each product, identify:
    - Product code used by the contractor (Index_Obcy)
    - Quantity ordered
    - Dimensions (if available)
    - Price per item (if available)
    
    The text may be in various languages including English, Polish, German, French, Italian, or Spanish.
    
    Text:
    {content}
    
    Additional context:
    - Contractor name: {contractor_name}
    
    Return the extracted information as a list of JSON objects, one for each product.
    Each object should have the fields: contractor_product_code, quantity, dimensions (optional), price (optional).
    """)
    
    # Create a parser for the output
    class OrderItems(BaseModel):
        items: List[OrderItem]
    
    parser = JsonOutputParser(pydantic_object=OrderItems)
    
    # Create the chain
    chain = order_items_prompt | llm | parser
    
    # Process the content
    if isinstance(content, str):
        # For text content
        text_to_process = content
    elif isinstance(content, dict):
        # For structured content (e.g., from Excel)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    elif isinstance(content, list):
        # For list content (e.g., from CSV)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        print(f"Unsupported content type: {type(content)}")
        return []
    
    # Limit text length to avoid token limits
    max_length = 15000  # Adjust based on model's context window
    if len(text_to_process) > max_length:
        text_to_process = text_to_process[:max_length]
    
    try:
        # Run the extraction
        result = chain.invoke({
            "content": text_to_process,
            "contractor_name": CONTRACTOR.get("Nazwa", "")
        })
        
        # Handle different response types
        extracted_items = []
        if isinstance(result, dict) and "items" in result:
            # It's a dictionary with items key
            for item in result["items"]:
                if isinstance(item, dict):
                    extracted_items.append({
                        "Index_Obcy": item.get("contractor_product_code"),
                        "Ilosc": item.get("quantity"),
                        "Wymiary": item.get("dimensions"),
                        "Cena": item.get("price")
                    })
                else:
                    extracted_items.append({
                        "Index_Obcy": getattr(item, "contractor_product_code", None),
                        "Ilosc": getattr(item, "quantity", None),
                        "Wymiary": getattr(item, "dimensions", None),
                        "Cena": getattr(item, "price", None)
                    })
        elif isinstance(result, list):
            # It's a list of items
            for item in result:
                if isinstance(item, dict):
                    extracted_items.append({
                        "Index_Obcy": item.get("contractor_product_code"),
                        "Ilosc": item.get("quantity"),
                        "Wymiary": item.get("dimensions"),
                        "Cena": item.get("price")
                    })
                else:
                    extracted_items.append({
                        "Index_Obcy": getattr(item, "contractor_product_code", None),
                        "Ilosc": getattr(item, "quantity", None),
                        "Wymiary": getattr(item, "dimensions", None),
                        "Cena": getattr(item, "price", None)
                    })
        elif hasattr(result, "items"):
            # It's a Pydantic object with items attribute
            for item in result.items:
                if isinstance(item, dict):
                    extracted_items.append({
                        "Index_Obcy": item.get("contractor_product_code"),
                        "Ilosc": item.get("quantity"),
                        "Wymiary": item.get("dimensions"),
                        "Cena": item.get("price")
                    })
                else:
                    extracted_items.append({
                        "Index_Obcy": getattr(item, "contractor_product_code", None),
                        "Ilosc": getattr(item, "quantity", None),
                        "Wymiary": getattr(item, "dimensions", None),
                        "Cena": getattr(item, "price", None)
                    })
                    
        return extracted_items
    
    except Exception as e:
        print(f"Error extracting order items: {e}")
        return []

def mapuj_produkty_kontrahenta(extracted_items, ID_Kontrahent):
    """
    Maps contractor product codes to our internal codes.
    
    Parameters:
    extracted_items (list): List of dictionaries with product information
    ID_Kontrahent (int): Contractor ID from the database
    
    Returns:
    list: List of dictionaries with completed product information
    """
    if not extracted_items:
        return []
    
    # Connect to database
    connection = database.connect_to_database()
    if not connection:
        print("Failed to connect to database")
        return extracted_items  # Return original items if can't connect
    
    mapped_items = []
    
    try:
        with connection.cursor() as cursor:
            for item in extracted_items:
                contractor_code = item.get("Index_Obcy")
                if not contractor_code:
                    continue
                
                # Look up mapping in database
                cursor.execute(
                    """
                    SELECT * FROM kontrahenci_indeksy 
                    WHERE ID_Kontrahent = %s AND Indeks_Kontrahent = %s
                    """, 
                    (ID_Kontrahent, contractor_code)
                )
                
                mapping = cursor.fetchone()
                
                if mapping:
                    # Use mapped values
                    mapped_item = {
                        "Index_Obcy": contractor_code,
                        "Index_Raks": mapping.get("Indeks_Progresja"),
                        "Dlugosc": mapping.get("Dlugosc") or item.get("Wymiary", "").split("x")[0] if item.get("Wymiary") else "",
                        "Wysokosc": mapping.get("Wysokosc") or item.get("Wymiary", "").split("x")[1] if item.get("Wymiary") and "x" in item.get("Wymiary") else "",
                        "Ilosc": item.get("Ilosc", ""),
                        "Cena_Pozycji": mapping.get("Cena") or item.get("Cena", ""),
                        "Usluga_Transportowa": "0"  # Default value
                    }
                else:
                    # No mapping found, use original values and mark for manual check
                    dimensions = item.get("Wymiary", "").split("x") if item.get("Wymiary") and "x" in item.get("Wymiary") else ["", ""]
                    mapped_item = {
                        "Index_Obcy": contractor_code,
                        "Index_Raks": "",  # Empty, needs manual mapping
                        "Dlugosc": dimensions[0] if len(dimensions) > 0 else "",
                        "Wysokosc": dimensions[1] if len(dimensions) > 1 else "",
                        "Ilosc": item.get("Ilosc", ""),
                        "Cena_Pozycji": item.get("Cena", ""),
                        "Usluga_Transportowa": "0"  # Default value
                    }
                
                mapped_items.append(mapped_item)
    
    except Exception as e:
        print(f"Error mapping contractor products: {e}")
        # Return original items if mapping fails
        return extracted_items
    finally:
        connection.close()
    
    return mapped_items

def wyodrebnij_dane_wysylkowe(content, EMAIL):
    """
    Extracts shipping address information.
    
    Parameters:
    content: Processed text or data from the attachment
    EMAIL (dict): Dictionary with email information
    
    Returns:
    dict: Dictionary with shipping address information
    """
    # Create a prompt for extracting shipping information
    shipping_info_prompt = ChatPromptTemplate.from_template("""
    You are an expert in extracting shipping information from order documents in multiple languages.
    
    Extract the following shipping information from the provided text:
    - Company name
    - Contact person name
    - Phone number
    - Street address
    - Building number
    - Apartment or suite number (if applicable)
    - Postal/ZIP code
    - City
    - Country
    - EU country code (if applicable)
    
    The text may be in various languages including English, Polish, German, French, Italian, or Spanish.
    
    Text:
    {content}
    
    Additional context:
    - Email subject: {subject}
    - Email sender: {sender}
    
    Return the extracted information in JSON format.
    """)
    
    # Create a parser for the output
    parser = JsonOutputParser(pydantic_object=ShippingInfo)
    
    # Create the chain
    chain = shipping_info_prompt | llm | parser
    
    # Process the content
    if isinstance(content, str):
        # For text content
        text_to_process = content
    elif isinstance(content, dict):
        # For structured content (e.g., from Excel)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    elif isinstance(content, list):
        # For list content (e.g., from CSV)
        text_to_process = json.dumps(content, ensure_ascii=False, indent=2)
    else:
        print(f"Unsupported content type: {type(content)}")
        return {}
    
    # Limit text length to avoid token limits
    max_length = 15000  # Adjust based on model's context window
    if len(text_to_process) > max_length:
        text_to_process = text_to_process[:max_length]
    
    try:
        # Run the extraction
        result = chain.invoke({
            "content": text_to_process,
            "subject": EMAIL.get("subject", ""),
            "sender": EMAIL.get("sender", "")
        })
        
        # Handle both Pydantic object and dictionary responses
        if isinstance(result, dict):
            # It's a dictionary
            shipping_info = {
                "Firma": result.get("company"),
                "Osoba": result.get("contact_person"),
                "Telefon": result.get("phone"),
                "Ulica": result.get("street"),
                "Nr_Domu": result.get("building_number"),
                "Nr_Lokalu": result.get("apartment_number"),
                "Kod_Pocztowy": result.get("postal_code"),
                "Miejscowosc": result.get("city"),
                "Kraj": result.get("country"),
                "EU_CODE": result.get("eu_code")
            }
        else:
            # It's a Pydantic object
            shipping_info = {
                "Firma": getattr(result, "company", None),
                "Osoba": getattr(result, "contact_person", None),
                "Telefon": getattr(result, "phone", None),
                "Ulica": getattr(result, "street", None),
                "Nr_Domu": getattr(result, "building_number", None),
                "Nr_Lokalu": getattr(result, "apartment_number", None),
                "Kod_Pocztowy": getattr(result, "postal_code", None),
                "Miejscowosc": getattr(result, "city", None),
                "Kraj": getattr(result, "country", None),
                "EU_CODE": getattr(result, "eu_code", None)
            }
        
        return shipping_info
    
    except Exception as e:
        print(f"Error extracting shipping information: {e}")
        return {}

def waliduj_dane_zamowienia(order_data):
    """
    Checks if all required order data has been correctly extracted.
    
    Parameters:
    order_data (dict): Dictionary with order data
    
    Returns:
    bool: True if data is complete and valid, False otherwise
    """
    print(order_data)
    # Check if order has basic information
    if not order_data.get("zamowienie"):
        print("Missing order information")
        return False
    
    # Check if there are any order items
    if not order_data.get("pozycje") or len(order_data.get("pozycje", [])) == 0:
        print("No order items found")
        return False
    
    # Check if shipping information is present
    if not order_data.get("wysylka"):
        print("Missing shipping information")
        return False
    
    # Check for critical fields in order
    critical_order_fields = ["NR_Zamowienia"]
    for field in critical_order_fields:
        if not order_data["zamowienie"].get(field):
            print(f"Missing critical order field: {field}")
            return False
    
    # Check for critical fields in shipping
    # critical_shipping_fields = ["Firma", "Miejscowosc", "Kraj"]
    # for field in critical_shipping_fields:
    #     if not order_data["wysylka"].get(field):
    #         print(f"Missing critical shipping field: {field}")
    #         return False
    
    # Check if all order items have required fields
    for item in order_data["pozycje"]:
        if not item.get("Index_Obcy") or not item.get("Ilosc"):
            print("Order item missing required fields")
            return False
    
    # Validate numeric values
    try:
        # Check if price is a valid number
        price = order_data["zamowienie"].get("Cena_Zamowienia", "0")
        if price and not isinstance(price, (int, float)):
            float(price.replace(",", "."))
        
        # Check if quantities are valid numbers
        for item in order_data["pozycje"]:
            quantity = item.get("Ilosc", "0")
            if quantity and not isinstance(quantity, (int, float)):
                float(quantity.replace(",", "."))
    except ValueError:
        print("Invalid numeric value found")
        return False
    
    return True

def get_orders(EMAIL, ATTACHMENTS, CONTRACTOR):
    """
    Extracts order information from email and attachments.
    
    Parameters:
    EMAIL (dict): Dictionary containing email information, including 'sender', 'subject', 'text', 'html'
    ATTACHMENTS (list): List of dictionaries with attachments from the email
    CONTRACTOR (dict): Information about the contractor identified earlier
    
    Returns:
    dict: Dictionary containing information about the order, its items, and shipping address
    """
    print(f"Processing order from contractor: {CONTRACTOR['Nazwa']}")
    
    # Prepare dictionary for order data
    order_data = {
        "zamowienie": {},
        "pozycje": [],
        "wysylka": {}
    }
    print("ATTACHMENTS:",ATTACHMENTS) 
    print("CONTRACTOR:",CONTRACTOR)
    sys.exit(0) 
    # 1. Find and select the appropriate attachment with the order
    order_attachment = wybierz_zalacznik_zamowienia(ATTACHMENTS, CONTRACTOR)
    if not order_attachment:
        print("No appropriate order attachment found")
        return None
        
    # 2. Process the attachment and extract text/data
    attachment_content = przetworz_zalacznik(order_attachment)
    if not attachment_content:
        print("Failed to process attachment content")
        return None
    
    # 3. Extract main order information (number, date, price, etc.)
    order_data["zamowienie"] = wyodrebnij_informacje_zamowienia(
        attachment_content, 
        EMAIL,
        CONTRACTOR
    )
    
    # 4. Extract order items (products)
    extracted_items = wyodrebnij_pozycje_zamowienia(
        attachment_content,
        CONTRACTOR
    )
    
    # 5. Map contractor product codes to our codes
    order_data["pozycje"] = mapuj_produkty_kontrahenta(
        extracted_items,
        CONTRACTOR["ID_Kontrahent"]
    )
    
    # 6. Extract shipping address data
    order_data["wysylka"] = wyodrebnij_dane_wysylkowe(
        attachment_content,
        EMAIL
    )
    
    # 7. Validate collected data
    if waliduj_dane_zamowienia(order_data):
        print(f"Successfully processed order: {order_data['zamowienie'].get('NR_Zamowienia', 'no number')}")
        return order_data
    else:
        print("Order data validation failed")
        return None
