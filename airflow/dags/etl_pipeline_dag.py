import requests
from bs4 import BeautifulSoup
import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import re
from collections import Counter
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pinecone

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# S3 configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = os.getenv("S3_PREFIX")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")

# Define base URLs
BASE_URL_NORMAL = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-{quarter}-quarter-fiscal-{year}"
BASE_URL_FOURTH = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-{year}"

QUARTERS = ['first', 'second', 'third', 'fourth']
YEARS = range(2021, 2026)

COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./local_chromadb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Create DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'nvidia_reports_processor',
    default_args=default_args,
    description='Process NVIDIA quarterly reports',
    schedule_interval=None,
    catchup=False
)

def get_s3_client():
    """Create and return an S3 client."""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=S3_REGION
        )
        return s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}", exc_info=True)
        raise

def check_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return url
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error checking URL {url}: {e}")
        return None

def scrape_nvidia_reports():
    logger.info("Starting to scrape NVIDIA reports")
    all_links = [
        check_url(BASE_URL_FOURTH.format(year=year)) if quarter == 'fourth' else
        check_url(BASE_URL_NORMAL.format(quarter=quarter, year=year))
        for year in YEARS for quarter in QUARTERS
    ]
    valid_links = [link for link in all_links if link]
    logger.info(f"Found {len(valid_links)} valid links.")
    return valid_links

def extract_data_from_link(url):
    logger.info(f"Extracting data from: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find('div', class_='main-content')
        if not main_content:
            logger.warning(f"No 'main-content' div found in {url}")
            return "", [], []

        text = main_content.get_text(separator='\n\n', strip=True)
        images = [img.get('src') for img in main_content.find_all('img') if img.get('src') and not img.get('src').startswith('data:')]
        tables = [
            [[th.get_text(strip=True) for th in table.find_all('th')]] +
            [[td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find_all('tr') if tr.find_all('td')]
            for table in main_content.find_all('table')
        ]
        return text, images, tables
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error extracting data from {url}: {e}", exc_info=True)
        return "", [], []
    except Exception as e:
        logger.error(f"Error extracting data from {url}: {e}", exc_info=True)
        return "", [], []

def convert_to_markdown(text, images, tables, url):
    logger.info(f"Converting data to markdown for {url}")
    report_name = url.split('/')[-2] if url.split('/')[-1] == 'default.aspx' else url.split('/')[-1]
    markdown_data = f"# NVIDIA Report: {report_name}\n\nSource: {url}\n\n## Content\n\n{text}\n\n"

    if images:
        markdown_data += "## Images\n\n" + "\n".join([f"![Image]({img})" for img in images]) + "\n\n"

    if tables:
        markdown_data += "## Tables\n\n" + "\n".join(
            [f"### Table {i + 1}\n\n" + "\n".join(['| ' + ' | '.join(row) + ' |' for row in table]) for i, table in
             enumerate(tables)]
        )

    logger.info(f"Successfully converted data to markdown for {url}")
    return markdown_data

def generate_filename(url):
   # Initialize quarter and year variables
    quarter = None
    year = None
    
    # Check if the URL contains a quarter and fiscal year
    for part in QUARTERS:
        if part in url:
            quarter = part
            break
    
    # Specifically handle the case for 'fourth_quarter_and_fiscal_X'
    if 'fourth_quarter_and_fiscal' in url:
        quarter = 'fourth'
    
    # Extract the year from the URL, which is typically a 4-digit number
    year_match = re.search(r'(\d{4})', url)
    if year_match:
        year = year_match.group(1)
    
    # Ensure both quarter and year are found
    if quarter and year:
        # Generate and return the formatted filename
        return f"nvidia_{quarter}_quarter_fiscal_{year}.md"
    
    filename = url.split('/')[-1] if 'default.aspx' in url else url.split('/')[-2]
    return f"nvidia_report_{filename.replace('-', '_')}.md"

def save_markdown_to_s3(content, filename):
    s3_client = get_s3_client()
    try:
        content_bytes = content.encode('utf-8')
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{S3_PREFIX}/{filename}",
            Body=content_bytes,
            ContentType='text/markdown'
        )
        logger.info(f"Uploaded {filename} to S3")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}", exc_info=True)
        return False

def section_based_chunking(markdown_text):
    sections = re.split(r'(?=^#+ )', markdown_text, flags=re.MULTILINE)
    return [section.strip() for section in sections if section.strip()]

def table_based_chunking(markdown_text):
    tables = re.findall(r'(\|[^\n]+\|\n)+', markdown_text)
    return tables

def sliding_window_chunking(text, window_size=500, stride=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = ' '.join(words[i:i+window_size])
        chunks.append(chunk)
    return chunks

def chunk_data_task(**kwargs):
    s3_client = get_s3_client()
    uploaded_files = kwargs['ti'].xcom_pull(key='uploaded_files', task_ids='upload_to_s3_task')

    if not uploaded_files:
        logger.warning("No files found in S3 bucket")
        return "No files found in S3 bucket"

    all_chunks = []

    # Chunking logic
    for i, file_key in enumerate(uploaded_files):
        try:
            file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}/{file_key}")
            text_content = file_content['Body'].read().decode('utf-8')
            filename = file_key.replace('.md', '')
            year, quarter = None, None

            year, quarter = None, None
            if "nvidia_" in filename and "_quarter_fiscal_" in filename:
                try:
                    # Extract the year and quarter directly from the filename
                    year_match = re.search(r'(\d{4})$', filename)  # Looks for 4-digit year at the end of the filename
                    quarter_match = re.search(r'(first|second|third|fourth)', filename)  # Matches the quarter in the filename
                    
                    if year_match and quarter_match:
                        year = year_match.group(1)
                        quarter = quarter_match.group(1)
                        logger.info(f"Extracted quarter: {quarter} for file: {filename}") 
                    else:
                        logger.info(f"Could not extract quarter/year from {filename}")  
                except IndexError:
                    logger.info(f"Error parsing year or quarter from filename: {filename}")
                    year, quarter = None, None
            else:
                logger.info(f"Filename does not follow expected naming convention: {filename}")
                year, quarter = None, None

            # Apply different chunking strategies based on file index
            if i < 7:
                chunks = section_based_chunking(text_content)
                chunk_type = "section"
            elif 7 <= i < 14:
                chunks = table_based_chunking(text_content)
                chunk_type = "table"
            else:
                chunks = sliding_window_chunking(text_content)
                chunk_type = "sliding"

            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "year": year,
                        "quarter": quarter,
                        "chunk_type": chunk_type,
                        "chunk_index": j
                    }
                })

            logger.info(f"Chunking completed for {file_key} using {chunk_type} chunking")

        except Exception as e:
            logger.error(f"Error processing {file_key}: {e}", exc_info=True)

    kwargs['ti'].xcom_push(key='all_chunks', value=all_chunks)
    return f"Chunking Completed for {len(uploaded_files)} files"

def create_tfidf_embeddings(chunks):
    texts = [chunk['text'] for chunk in chunks]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        tfidf_vector = tfidf_matrix[i].toarray()[0]
        top_features = sorted(zip(feature_names, tfidf_vector), key=lambda x: x[1], reverse=True)[:100]
        
        embedding = {word: score for word, score in top_features}
        embeddings.append({
            "embedding": embedding,
            "metadata": chunk['metadata'],
            "text": chunk['text']
        })
    
    return embeddings


def tfidf_embedding_storage(**kwargs):
    all_chunks = kwargs['ti'].xcom_pull(key='all_chunks', task_ids='chunk_data_task')
    
    if not all_chunks:
        logger.warning("No chunks found for TF-IDF embedding")
        return "No chunks found for TF-IDF embedding"
    
    embeddings = create_tfidf_embeddings(all_chunks)
    
    # Store embeddings in Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=100, metric="cosine")
    
    index = pinecone.Index(PINECONE_INDEX_NAME)
    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        vectors = [
            (
                f"{emb['metadata']['filename']}_{emb['metadata']['chunk_type']}_{emb['metadata']['chunk_index']}",
                list(emb['embedding'].values()),
                {**emb['metadata'], 'text': emb['text']}
            )
            for emb in batch
        ]
        index.upsert(vectors=vectors)
    
    return f"Stored {len(embeddings)} TF-IDF embeddings in Pinecone"


# Sentence Transformer Embedding task for ChromaDB storage
def get_or_create_collection():
    """
    Get or create a ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )    

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=embedding_func
        )
        logger.info(f"Using existing collection: {COLLECTION_NAME}")

    except:
        collection = client.create_collection(
            name=COLLECTION_NAME, 
            embedding_function=embedding_func, 
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created new collection: {COLLECTION_NAME}")
    
    return collection

def sentence_transformer_embedding_storage(**kwargs):
    all_chunks = kwargs['ti'].xcom_pull(key='all_chunks', task_ids='chunk_data_task')
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    embeddings = embeddings.cpu().numpy()

    # Store embeddings in ChromaDB
    collection = get_or_create_collection()

    ids = [
        f"chunk_{chunk['metadata']['filename']}_{chunk['metadata']['chunk_type']}_{chunk['metadata']['chunk_index']}"
        for chunk in all_chunks
    ]
    metadatas = [
        {"filename": chunk['metadata']['filename'], "chunk_type": chunk['metadata']['chunk_type'], "chunk_index": chunk['metadata']['chunk_index']}
        for chunk in all_chunks
    ]

    collection.add(
        embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
    )

    count = collection.count()

    return f"Stored {count} sentence transformer embeddings in ChromaDB"

def scrape_links_task(**kwargs):
    scraped_links = scrape_nvidia_reports()
    kwargs['ti'].xcom_push(key='scraped_links', value=scraped_links)
    return len(scraped_links)

def extract_and_convert_task(**kwargs):
    scraped_links = kwargs['ti'].xcom_pull(key='scraped_links', task_ids='scrape_links_task')
    if not scraped_links:
        return "No links to process"

    markdown_data_list = []
    for link in scraped_links:
        text, images, tables = extract_data_from_link(link)
        if text:
            filename = generate_filename(link)
            if filename: #Only process if filename is valid
                markdown_data = convert_to_markdown(text, images, tables, link)
                markdown_data_list.append({
                    'content': markdown_data,
                    'filename': filename,
                    'source_url': link
                })
            else:
                logger.warning(f"Skipping {link} due to filename generation failure.")
        else:
            logger.warning(f"No text extracted from {link}, skipping.")

    kwargs['ti'].xcom_push(key='markdown_data_list', value=markdown_data_list)
    return len(markdown_data_list)

def upload_to_s3_task(**kwargs):
    markdown_data_list = kwargs['ti'].xcom_pull(key='markdown_data_list', task_ids='extract_and_convert_task')
    if not markdown_data_list:
        return "No data to upload"

    uploaded_files = []

    for item in markdown_data_list:
        if save_markdown_to_s3(item['content'], item['filename']):
            uploaded_files.append(item['filename'])

    kwargs['ti'].xcom_push(key='uploaded_files', value=uploaded_files)
    return uploaded_files

with DAG('embedding_storage_dag', default_args=default_args, schedule_interval=None) as dag:
    scrape_task = PythonOperator(
        task_id='scrape_links_task',
        python_callable=scrape_links_task,
        provide_context=True,
    )

    extract_convert_task = PythonOperator(
        task_id='extract_and_convert_task',
        python_callable=extract_and_convert_task,
        provide_context=True,
    )

    upload_task = PythonOperator(
        task_id='upload_to_s3_task',
        python_callable=upload_to_s3_task,
        provide_context=True,
    )

    chunk_task = PythonOperator(
        task_id='chunk_data_task',
        python_callable=chunk_data_task,
        provide_context=True,
    )

    tfidf_embedding_storage_task = PythonOperator(
        task_id='tfidf_embedding_storage_task',
        python_callable=tfidf_embedding_storage,
        provide_context=True,
    )

    sentence_transformer_embedding_storage_task = PythonOperator(
        task_id='sentence_transformer_embedding_storage_task',
        python_callable=sentence_transformer_embedding_storage,
        provide_context=True,
    )

    # Define task dependencies
    scrape_task >> extract_convert_task >> upload_task >> chunk_task
    chunk_task >> tfidf_embedding_storage_task
    chunk_task >> sentence_transformer_embedding_storage_task