import os
import logging
import time
import boto3
import tempfile
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from markitdown import MarkItDown

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "pdf_documents"

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

IMAGE_RESOLUTION_SCALE = 2.0

def docling_pdf_to_markdown(input_pdf_path: Path, output_dir: Path, embed_images=True):
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        start_time = time.time()
        conv_res = doc_converter.convert(input_pdf_path)
        if conv_res is None:
            return None, None, None, "Failed to convert PDF"

        output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = input_pdf_path.stem

        # Extract images & tables
        images, tables = save_images_and_tables(conv_res, output_dir, doc_filename)

        # Choose image embedding mode
        image_mode = ImageRefMode.EMBEDDED if embed_images else ImageRefMode.REFERENCED

        # Save markdown
        markdown_file = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(markdown_file, image_mode=image_mode)

        end_time = time.time() - start_time
        logger.info(f"Document converted in {end_time:.2f} seconds.")

        return markdown_file, images, tables, None

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return None, None, None, str(e)

def save_images_and_tables(conv_res, output_dir: Path, doc_filename: str):
    table_images = []
    extracted_tables = []
    
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_filename = output_dir / f"{doc_filename}-table-{len(table_images) + 1}.png"
            with table_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
            table_images.append(str(table_filename))
            extracted_tables.append(element.get_table_data())

        if isinstance(element, PictureItem):
            picture_filename = output_dir / f"{doc_filename}-picture-{len(table_images) + 1}.png"
            with picture_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
            table_images.append(str(picture_filename))

    return table_images, extracted_tables

# Convert parsed text to Markdown format
def convert_to_markdown(text, filename):
    return f"# Document: {filename}\n\n{text}\n"

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

# Process PDF and Upload to S3
def process_pdf(file_path: Path):
    output_dir = TEMP_DIR / file_path.stem
    markdown_file, images, tables, error = docling_pdf_to_markdown(file_path, output_dir)
    if error:
        raise Exception(error)

    # Read Markdown content
    with open(markdown_file, "r", encoding="utf-8") as md_file:
        markdown_content = md_file.read()

    # Upload Markdown to S3
    filename = upload_to_s3(markdown_content, markdown_file.name)

    return {"message": "File processed and uploaded successfully", "filename": filename}