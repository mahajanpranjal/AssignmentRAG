import os
import logging
import time
import boto3
import json
import tempfile
from pathlib import Path
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "pdf_documents"

# Initialize Mistral Client (Replace with your API key)
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

# Ensure S3 Client Setup
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION,
    )

# Ensure Temp Directory Exists
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def extract_links_from_text(text):
    import re
    url_pattern = r'https?://[^\s)]+'
    return re.findall(url_pattern, text)

def extract_tables_from_response(ocr_response):
    tables = []
    for page in ocr_response.get("pages", []):
        for table in page.get("tables", []):
            tables.append(table.get("markdown", ""))
    return tables

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns = []
    for page in ocr_response.pages:
        markdowns.append(page.markdown)
    return "\n\n".join(markdowns)

def process_pdf_with_mistral(uploaded_pdf):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_pdf_path = Path(tmp_file.name)
            temp_pdf_path.write_bytes(uploaded_pdf)

        uploaded_file = client.files.upload(
            file={"file_name": temp_pdf_path.stem, "content": temp_pdf_path.read_bytes()},
            purpose="ocr",
        )

        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )

        response_dict = json.loads(pdf_response.model_dump_json())
        extracted_text = response_dict["pages"][0]["markdown"]
        extracted_links = extract_links_from_text(extracted_text)
        extracted_tables = extract_tables_from_response(response_dict)
        combined_markdown = get_combined_markdown(pdf_response)

        return {
            "combined_markdown": combined_markdown,
            "extracted_links": extracted_links,
            "extracted_tables": extracted_tables
        }

    except Exception as e:
        return {"error": str(e)}

# Upload Markdown file to S3
def upload_to_s3(content: str, filename: str):
    s3_client = get_s3_client()
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{S3_PREFIX}/{filename}",
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )
        return filename
    except Exception as e:
        raise Exception(f"Failed to upload {filename} to S3: {str(e)}")

# Process PDF with Mistral and Upload to S3
def process_and_store_pdf(file_path: Path):
    with open(file_path, "rb") as file:
        pdf_data = file.read()
    
    processed_data = process_pdf_with_mistral(pdf_data)
    if "error" in processed_data:
        raise Exception(processed_data["error"])
    
    markdown_content = processed_data["combined_markdown"]
    filename = f"{file_path.stem}.md"
    upload_to_s3(markdown_content, filename)
    
    return {"message": "File processed and uploaded successfully", "filename": filename}
