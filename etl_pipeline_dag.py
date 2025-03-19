import requests
from bs4 import BeautifulSoup
import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Define base URLs
BASE_URL_NORMAL = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-{quarter}-quarter-fiscal-{year}"
BASE_URL_FOURTH = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-{year}"

QUARTERS = ['first', 'second', 'third', 'fourth']
YEARS = range(2021, 2026)

# S3 configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")
S3_PREFIX = "nvidia_reports"  

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
    'nvidia_reports_scraper',
    default_args=default_args,
    description='Scrape NVIDIA quarterly reports',
    schedule_interval=None,
)

def get_s3_client():
    """Create and return an S3 client."""
    try:
        return boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=S3_REGION)
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}", exc_info=True)
        raise

def check_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return url
    except Exception as e:
        logger.error(f"Error checking URL {url}: {e}", exc_info=True)
    return None

def scrape_nvidia_reports():
    logger.info("Starting to scrape NVIDIA reports")
    all_links = [
        check_url(BASE_URL_FOURTH.format(year=year)) if quarter == 'fourth' else
        check_url(BASE_URL_NORMAL.format(quarter=quarter, year=year))
        for year in YEARS for quarter in QUARTERS
    ]
    return [link for link in all_links if link]

def extract_data_from_link(url):
    logger.info(f"Extracting data from: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text = soup.find('div', class_='main-content').get_text(separator='\n\n', strip=True)
        images = [img.get('src') for img in soup.find_all('img') if img.get('src') and not img.get('src').startswith('data:')]
        tables = [
            [[th.get_text(strip=True) for th in table.find_all('th')]] + 
            [[td.get_text(strip=True) for td in tr.find_all('td')] for tr in table.find_all('tr') if tr.find_all('td')]
            for table in soup.find_all('table')
        ]
        return text, images, tables
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
            [f"### Table {i+1}\n\n" + "\n".join(['| ' + ' | '.join(row) + ' |' for row in table]) for i, table in enumerate(tables)]
        )

    logger.info(f"Successfully converted data to markdown for {url}")
    return markdown_data

def generate_filename(url):
    filename = url.split('/')[-2] if url.split('/')[-1] == 'default.aspx' else url.split('/')[-1]
    return f"nvidia_report_{filename.replace('-', '_')}.md"

## Airflow DAG tasks

def scrape_links_task(**kwargs):
    scraped_links = scrape_nvidia_reports()
    kwargs['ti'].xcom_push(key='scraped_links', value=scraped_links)
    return len(scraped_links)

def extract_and_convert_task(**kwargs):
    scraped_links = kwargs['ti'].xcom_pull(key='scraped_links', task_ids='scrape_links_task')
    if not scraped_links:
        return "No links to process"
    
    markdown_data_list = [
        {
            'content': convert_to_markdown(*extract_data_from_link(link), link),
            'filename': generate_filename(link),
            'source_url': link
        }
        for link in scraped_links
    ]
    
    kwargs['ti'].xcom_push(key='markdown_data_list', value=markdown_data_list)
    return len(markdown_data_list)

def upload_to_s3_task(**kwargs):
    markdown_data_list = kwargs['ti'].xcom_pull(key='markdown_data_list', task_ids='extract_and_convert_task')
    if not markdown_data_list:
        return "No data to upload"
    
    s3_client = get_s3_client()
    uploaded_files = []
    failed_files = []
    
    for item in markdown_data_list:
        try:
            content_bytes = item['content'].encode('utf-8')
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{S3_PREFIX}/{item['filename']}",
                Body=content_bytes,
                ContentType='text/markdown',
                Metadata={'source_url': item['source_url']}
            )
            uploaded_files.append(item['filename'])
        except Exception as e:
            logger.error(f"Failed to upload {item['filename']}: {e}", exc_info=True)
            failed_files.append(item['filename'])
    
    summary = {
        'uploaded_count': len(uploaded_files),
        'failed_count': len(failed_files),
        'uploaded_files': uploaded_files,
        'failed_files': failed_files
    }
    kwargs['ti'].xcom_push(key='s3_upload_summary', value=summary)
    return f"Uploaded {len(uploaded_files)} files to S3, Failed: {len(failed_files)}"

# Define Chunking tasks

def sentence_chunk_task(**kwargs):
    s3_client = get_s3_client()
    markdown_data_list = kwargs['ti'].xcom_pull(key='markdown_data_list', task_ids='upload_to_s3_task')

    if not markdown_data_list:
        return "No data to chunk"
    
    # Apply Sentence Chunking to files 1-7
    sentence_chunked_files = markdown_data_list[:7]
    for item in sentence_chunked_files:
        file_key = f"{S3_PREFIX}/{item['filename']}"
        file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        text_content = file_content['Body'].read().decode('utf-8')
        
        # Perform Sentence Chunking
        recursive_token_chunker = RecursiveTokenChunker(
            chunk_size=400,
            chunk_overlap=0,
            length_function=openai_token_count,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        sentence_chunks = recursive_token_chunker.split_text(text_content)
        save_chunks_to_json(sentence_chunks, "sentence_chunking")

    return "Sentence Chunking Completed"

def semantic_chunk_task(**kwargs):
    s3_client = get_s3_client()
    markdown_data_list = kwargs['ti'].xcom_pull(key='markdown_data_list', task_ids='upload_to_s3_task')

    if not markdown_data_list:
        return "No data to chunk"

    # Apply Semantic Chunking to files 8-14
    semantic_chunked_files = markdown_data_list[7:14]
    for item in semantic_chunked_files:
        file_key = f"{S3_PREFIX}/{item['filename']}"
        file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        text_content = file_content['Body'].read().decode('utf-8')

        api_key = os.environ.get("OPENAI_API_KEY")
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
        kamradt_chunker = KamradtModifiedChunker(
            avg_chunk_size=300,
            min_chunk_size=50,
            embedding_function=embedding_function
        )
        semantic_chunks = kamradt_chunker.split_text(text_content)
        save_chunks_to_json(semantic_chunks, "semantic_chunking")

    return "Semantic Chunking Completed"

def paragraph_chunk_task(**kwargs):
    s3_client = get_s3_client()
    markdown_data_list = kwargs['ti'].xcom_pull(key='markdown_data_list', task_ids='upload_to_s3_task')

    if not markdown_data_list:
        return "No data to chunk"
    
    # Apply Paragraph Chunking to files 15-20
    paragraph_chunked_files = markdown_data_list[14:20]
    for item in paragraph_chunked_files:
        file_key = f"{S3_PREFIX}/{item['filename']}"
        file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        text_content = file_content['Body'].read().decode('utf-8')

        recursive_token_overlap_chunker = RecursiveTokenChunker(
            chunk_size=400,
            chunk_overlap=200,
            length_function=openai_token_count,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        paragraph_chunks = recursive_token_overlap_chunker.split_text(text_content)
        save_chunks_to_json(paragraph_chunks, "paragraph_chunking")

    return "Paragraph Chunking Completed"

# Assign tasks
scrape_links = PythonOperator(
    task_id='scrape_links_task',
    python_callable=scrape_links_task,
    provide_context=True,
    dag=dag,
)

extract_and_convert = PythonOperator(
    task_id='extract_and_convert_task',
    python_callable=extract_and_convert_task,
    provide_context=True,
    dag=dag,
)

upload_to_s3 = PythonOperator(
    task_id='upload_to_s3_task',
    python_callable=upload_to_s3_task,
    provide_context=True,
    dag=dag,
)

sentence_chunk = PythonOperator(
    task_id='sentence_chunk_task',
    python_callable=sentence_chunk_task,
    provide_context=True,
    dag=dag,
)

semantic_chunk = PythonOperator(
    task_id='semantic_chunk_task',
    python_callable=semantic_chunk_task,
    provide_context=True,
    dag=dag,
)

paragraph_chunk = PythonOperator(
    task_id='paragraph_chunk_task',
    python_callable=paragraph_chunk_task,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
scrape_links >> extract_and_convert >> upload_to_s3
upload_to_s3 >> sentence_chunk
upload_to_s3 >> semantic_chunk
upload_to_s3 >> paragraph_chunk
