import requests
from bs4 import BeautifulSoup
import boto3
import logging
import os
from dotenv import load_dotenv
import re
from collections import Counter
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
from chromadb.config import Settings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        print(f"Error creating S3 client: {e}")  # Use print for visibility
        raise

def check_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return url
    except requests.exceptions.RequestException as e:
        print(f"Error checking URL {url}: {e}")  # Use print for visibility
        return None

def scrape_nvidia_reports():
    print("Starting to scrape NVIDIA reports")  # Use print for visibility
    all_links = [
        check_url(BASE_URL_FOURTH.format(year=year)) if quarter == 'fourth' else
        check_url(BASE_URL_NORMAL.format(quarter=quarter, year=year))
        for year in YEARS for quarter in QUARTERS
    ]
    valid_links = [link for link in all_links if link]
    print(f"Found {len(valid_links)} valid links.")  # Use print for visibility
    return valid_links

def extract_data_from_link(url):
    print(f"Extracting data from: {url}")  # Use print for visibility
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content = soup.find('div', class_='main-content')
        if not main_content:
            print(f"No 'main-content' div found in {url}")  # Use print for visibility
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
        print(f"HTTP error extracting data from {url}: {e}")  # Use print for visibility
        return "", [], []
    except Exception as e:
        print(f"Error extracting data from {url}: {e}")  # Use print for visibility
        return "", [], []

def convert_to_markdown(text, images, tables, url):
    print(f"Converting data to markdown for {url}")  # Use print for visibility
    report_name = url.split('/')[-2] if url.split('/')[-1] == 'default.aspx' else url.split('/')[-1]
    markdown_data = f"# NVIDIA Report: {report_name}\n\nSource: {url}\n\n## Content\n\n{text}\n\n"

    if images:
        markdown_data += "## Images\n\n" + "\n".join([f"![Image]({img})" for img in images]) + "\n\n"

    if tables:
        markdown_data += "## Tables\n\n" + "\n".join(
            [f"### Table {i + 1}\n\n" + "\n".join(['| ' + ' | '.join(row) + ' |' for row in table]) for i, table in
             enumerate(tables)]
        )

    print(f"Successfully converted data to markdown for {url}")  # Use print for visibility
    return markdown_data

def generate_filename(url):
    """
    Generates the filename based on the URL format. 
    It extracts the quarter and fiscal year from the URL and formats it as: 
    nvidia_<quarter>_quarter_fiscal_<year>.md
    """
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
    
    # Fallback logic if no valid quarter or year is found
    # This can happen if the URL structure is unexpected
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
        print(f"Uploaded {filename} to S3")  # Use print for visibility
        return True
    except Exception as e:
        print(f"Failed to upload {filename}: {e}")  # Use print for visibility
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

def chunk_data(uploaded_files):
    s3_client = get_s3_client()

    if not uploaded_files:
        print("No files found in S3 bucket")  # Use print for visibility
        return []  # Return empty list, so downstream functions don't break

    all_chunks = []

    # Chunking logic
    for i, file_key in enumerate(uploaded_files):
        try:
            file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}/{file_key}")
            text_content = file_content['Body'].read().decode('utf-8')
            filename = file_key.replace('.md', '')

            # Ensure you correctly extract the year and quarter here based on the new naming
            year, quarter = None, None
            if "nvidia_" in filename and "_quarter_fiscal_" in filename:
                try:
                    # Extract the year and quarter directly from the filename
                    year_match = re.search(r'(\d{4})$', filename)  # Looks for 4-digit year at the end of the filename
                    quarter_match = re.search(r'(first|second|third|fourth)', filename)  # Matches the quarter in the filename
                    
                    if year_match and quarter_match:
                        year = year_match.group(1)
                        quarter = quarter_match.group(1)
                        print(f"Extracted quarter: {quarter} for file: {filename}")  # Print the extracted quarter
                    else:
                        print(f"Could not extract quarter/year from {filename}")  # In case of failure
                except IndexError:
                    print(f"Error parsing year or quarter from filename: {filename}")
                    year, quarter = None, None
            else:
                print(f"Filename does not follow expected naming convention: {filename}")
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

            print(f"Chunking completed for {file_key} using {chunk_type} chunking")  # Use print for visibility

        except Exception as e:
            print(f"Error processing {file_key}: {e}")  # Use print for visibility

    return all_chunks


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

def embedding_data(all_chunks):
    if not all_chunks:
        print("No chunks found for embedding")  # Use print for visibility
        return []  # Returning empty list, so downstream doesn't break.
    embeddings = create_tfidf_embeddings(all_chunks)
    return embeddings

def store_in_chromadb(embeddings):
    client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIRECTORY))
    collection = client.create_collection(name="nvidia_reports")

    for embedding in embeddings:
        try:
            embedding_values = list(embedding['embedding'].values())
            if not embedding_values:
                print(f"Skipping embedding due to empty embedding values.")  # Use print for visibility
                continue

            collection.add(
                embeddings=[embedding_values],
                documents=[embedding['text']],
                metadatas=[embedding['metadata']],
                ids=[f"{embedding['metadata']['filename']}_{embedding['metadata']['chunk_type']}_{embedding['metadata']['chunk_index']}"]
            )
        except Exception as e:
            print(f"Error adding embedding to ChromaDB: {e}")  # Use print for visibility

    return "Stored embeddings in ChromaDB"

def filter_zero_vectors(vectors):
    """
    Filters out vectors that contain only zeros.
    """
    non_zero_vectors = []
    for vector in vectors:
        if np.any(np.array(vector[1]) != 0):  # Check if any element in the vector is non-zero
            non_zero_vectors.append(vector)
        else:
            print(f"Vector {vector[0]} contains only zeros and will not be upserted.")  # Log for debugging
    return non_zero_vectors

def store_in_pinecone(embeddings):
    print("Storing embeddings in Pinecone")

    # Corrected usage of Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=100,  # Ensure this matches the dimensionality of your embeddings
            metric="cosine",  # Using cosine similarity metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"  # Change this to match your Pinecone environment
            )
        )

    # Connect to the Pinecone index
    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        
        # Prepare vectors for Pinecone upsert
        vectors = [
            (
                f"{emb['metadata']['filename']}_{emb['metadata']['chunk_type']}_{emb['metadata']['chunk_index']}",
                [float(value) for value in emb['embedding'].values()],  # Convert numpy.float64 to Python float
                {**emb['metadata'], 'text': emb['text']}
            )
            for emb in batch
        ]

        # Filter out zero vectors before upsert
        vectors = filter_zero_vectors(vectors)
        
        if vectors:  # Only upsert non-zero vectors
            try:
                index.upsert(vectors=vectors)
                print(f"Upserted a batch of {len(vectors)} vectors to Pinecone.")
            except Exception as e:
                print(f"Error upserting vectors to Pinecone: {e}")
        else:
            print("No valid vectors to upsert in this batch.")

    return "Stored embeddings in Pinecone"



def storage_data(embeddings):
    if not embeddings:
        print("No embeddings found for storage")  # Use print for visibility
        return "No embeddings found for storage"

    ##chroma_result = store_in_chromadb(embeddings)
    pinecone_result = store_in_pinecone(embeddings)

    ##return f"{chroma_result}\n{pinecone_result}"
    return f"{pinecone_result}"


def print_first_two_embeddings(embeddings):
    for i, embedding in enumerate(embeddings[:2]):  # Print only the first 2 embeddings
        print(f"Embedding {i + 1}:")
        print(f"Metadata: {embedding['metadata']}")
        print(f"Embedding: {embedding['embedding']}")
        print("-" * 40)


def scrape_links_task():
    return scrape_nvidia_reports()

def extract_and_convert_task(scraped_links):
    if not scraped_links:
        return []

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
                print(f"Skipping {link} due to filename generation failure.")  # Use print for visibility
        else:
            print(f"No text extracted from {link}, skipping.")  # Use print for visibility

    return markdown_data_list

def upload_to_s3_task(markdown_data_list):
    if not markdown_data_list:
        return []

    uploaded_files = []

    for item in markdown_data_list:
        if save_markdown_to_s3(item['content'], item['filename']):
            uploaded_files.append(item['filename'])

    return uploaded_files

def chunk_data_task(uploaded_files):
    return chunk_data(uploaded_files)

def embedding_task(all_chunks):
    return embedding_data(all_chunks)

def storage_task(embeddings):
    return storage_data(embeddings)

def main():
    # 1. Scrape Links
    scraped_links = scrape_links_task()
    print(f"Scraped {len(scraped_links)} links.")  # Use print for visibility

    # 2. Extract and Convert
    markdown_data_list = extract_and_convert_task(scraped_links)
    print(f"Extracted and converted {len(markdown_data_list)} reports.")  # Use print for visibility

    # 3. Upload to S3
    uploaded_files = upload_to_s3_task(markdown_data_list)
    print(f"Uploaded {len(uploaded_files)} files to S3.")  # Use print for visibility

    # 4. Chunk Data
    all_chunks = chunk_data_task(uploaded_files)
    print(f"Created {len(all_chunks)} chunks.")  # Use print for visibility

    # 5. Embed Data
    embeddings = embedding_task(all_chunks)
    print(f"Created {len(embeddings)} embeddings.")  # Use print for visibility

    print_first_two_embeddings(embeddings)

    # 6. Store Data
    storage_result = storage_task(embeddings)
    print(storage_result)  # Use print for visibility

if __name__ == "__main__":
    main()
