import os
import logging
import json
import tempfile
import fitz
import base64
import boto3
import openai
import chromadb
import re
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker
)
from chunking_evaluation.utils import openai_token_count
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import APIRouter
from openai import OpenAI

router = APIRouter()

load_dotenv()  
 
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("GPT4o_API_KEY")
openai.api_key = OPENAI_API_KEY

# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "pdf_documents"
 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=PINECONE_API_KEY)
 
# ChromaDB configuration
COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./local_chromadb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"SentenceTransformer model '{EMBEDDING_MODEL}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    embedding_model = None

app = FastAPI()
app.include_router(router)

API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

class QueryPayload(BaseModel):
    question: str
    vector_db: str
    quarters: list

class ProcessChunkPayload(BaseModel):
    filename: str
    chunking_strategy: str
    vector_db: str
    quarter: str
 
# Ensure S3 Client Setup
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION,
    )

def upload_to_s3(content: str, filename: str):
    s3_client = get_s3_client()
    
    # Debugging logs
    logger.info(f"Uploading {filename} to S3 Bucket: {S3_BUCKET_NAME} in Region: {S3_REGION}")
    
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is None. Check your environment variables.")
        raise Exception("S3_BUCKET_NAME is not set. Please check your environment variables.")
 
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{S3_PREFIX}/{filename}",
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info(f"Successfully uploaded {filename} to S3.")
        return filename
    except Exception as e:
        logger.error(f"Failed to upload {filename} to S3: {str(e)}")
        raise Exception(f"Failed to upload {filename} to S3: {str(e)}")
    
 
def extract_text_pymupdf(pdf_file_io: BytesIO):
    try:
        doc = fitz.open(stream=pdf_file_io, filetype="pdf")
        text_data = ""
        images = []
        tables = []
 
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_data += f"### Page {page_num + 1}\n\n"
            text_data += page.get_text("text") + "\n\n"
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(BytesIO(image_bytes))
                img_filename = f"image_{page_num + 1}_{img_index + 1}.png"
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({
                    "filename": img_filename,
                    "base64": img_base64
                })
            
            table = page.get_text("dict")
            tables.append(table)
        
        markdown_content = "# Extracted Data from PDF\n\n" + text_data + "\n"
        
        markdown_content += "## Extracted Images\n"
        for img in images:
            markdown_content += f"![{img['filename']}](data:image/png;base64,{img['base64']})\n"
        
        markdown_content += "\n## Extracted Tables\n"
        for table in tables:
            markdown_content += "### Table\n"
            for block in table["blocks"]:
                if block['type'] == 0:
                    for line in block["lines"]:
                        line_text = " | ".join([span["text"] for span in line["spans"]])
                        markdown_content += line_text + "\n"
        
        return markdown_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
def process_pdf_with_docling(input_pdf_path: Path):
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        conv_res = doc_converter.convert(input_pdf_path)
        if conv_res is None:
            return None
        output_filename = Path(f"{input_pdf_path.stem}_output.md")
        
        conv_res.document.save_as_markdown(output_filename) 

        # Read the saved markdown file
        with open(output_filename, 'r') as f:
            markdown_content = f.read()
        
        os.remove(output_filename)
        
        return markdown_content
    except Exception as e:
        raise Exception(f"Docling error: {str(e)}")

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

        with open(temp_pdf_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file, 
                purpose="ocr"
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
 
@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), parser: str = Form(...)):
    try:
        pdf_bytes = file.file.read()
        markdown_content = None
        
        if parser == "pymupdf":
            markdown_content = extract_text_pymupdf(BytesIO(pdf_bytes))
            logger.info("PyMuPDF parser used successfully")
        
        elif parser == "mistral_ocr":
            logger.info("Using Mistral OCR parser...")
            extracted_data = process_pdf_with_mistral(pdf_bytes)
            
            logger.info(f"Mistral OCR response: {json.dumps(extracted_data)[:200]}...")
            
            if "error" in extracted_data:
                logger.error(f"Mistral OCR error: {extracted_data['error']}")
                raise HTTPException(status_code=500, detail=f"Mistral OCR error: {extracted_data['error']}")
            
            if not extracted_data.get("pages"):
                logger.error("No 'pages' field in Mistral OCR response")
                raise HTTPException(status_code=500, detail="Invalid Mistral OCR response format: no pages found")
            
            markdown_content = extracted_data.get("pages", [{}])[0].get("markdown", "")
            logger.info(f"Extracted markdown content length: {len(markdown_content) if markdown_content else 0}")
        
        elif parser == "docling":
            logger.info("Using Docling parser...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                pdf_path = Path(tmp_file.name)
            markdown_content = process_pdf_with_docling(pdf_path)
            logger.info(f"Docling parser generated content length: {len(markdown_content) if markdown_content else 0}")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid processing method")
 
        if not markdown_content:
            logger.error("Markdown content is None or empty")
            raise HTTPException(status_code=500, detail="Failed to generate markdown content")
        
        filename = f"{file.filename}.md"
        logger.info(f"Uploading {filename} to S3, content length: {len(markdown_content)}")
        upload_to_s3(markdown_content, filename)
        logger.info(f"Successfully uploaded {filename} to S3")
 
        return {"message": "File processed and uploaded successfully", "filename": filename}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
 
 
def create_sentence_transformer_embeddings(chunks):
    """Creates embeddings for the given chunks using SentenceTransformer."""
    global embedding_model
    if embedding_model is None:
        raise ValueError("Embedding model not initialized.")

    texts = [chunk['text'] for chunk in chunks]
    
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings)
 
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "embedding": embeddings[i].tolist(),  
            "metadata": chunk["metadata"],
            "text": chunk["text"][:300] 
        })
 
    return result
 
def fetch_files_from_s3():
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    return [obj["Key"].split("/")[-1] for obj in response.get("Contents", [])]
 
def chunk_Recursive_document(text):
    chunker = RecursiveTokenChunker(
        chunk_size=400,
        chunk_overlap=200,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    chunks = chunker.split_text(text)

    return chunks
 
def chunk_Token_document(text):
    fixed_token_chunker = FixedTokenChunker(
    chunk_size=400,  
    chunk_overlap=0,  
    encoding_name="cl100k_base"  
)
  
    chunks = fixed_token_chunker.split_text(text)
   
    return chunks
 
def chunk_Semantics_document(text):
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("GPT4o_API_KEY"),
        model_name="text-embedding-3-small"
    )
    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=300,   
        min_chunk_size=50,      
        embedding_function=embedding_function  
    )
    
    chunks = kamradt_chunker.split_text(text)

    return chunks
 
def chunk_data(uploaded_files, chunking_strategy):
    s3_client = get_s3_client()
    all_chunks = []
    for file_key in uploaded_files:
        try:
            file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}/{file_key}")
            text_content = file_content['Body'].read().decode('utf-8')
            filename = file_key.replace('.md', '')
 
            if chunking_strategy == "Recursive":
                chunks = chunk_Recursive_document(text_content)
            elif chunking_strategy == "Token":
                chunks = chunk_Token_document(text_content)
            else:
                chunks = chunk_Semantics_document(text_content)
 
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_type": chunking_strategy,
                        "chunk_index": j
                    }
                })
        except Exception as e:
            print(f"Error processing {file_key}: {e}")
 
    return all_chunks


def get_or_create_collection():

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"Using existing collection: {COLLECTION_NAME}")

    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {COLLECTION_NAME}")

    return collection


def store_in_chromadb(all_chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

    embeddings = embeddings.cpu().numpy() 
 
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
    print(f"Total documents in collection: {count}")
 
    return f"Stored {count} embeddings in ChromaDB"


def store_in_pinecone(embeddings):
    print("⚡ Storing embeddings in Pinecone")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 50  
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]

        vectors = [
            (
                f"{emb['metadata']['filename']}_{emb['metadata']['chunk_type']}_{emb['metadata']['chunk_index']}",
                emb['embedding'],
                {**emb['metadata'], 'text': emb['text'][:300]}  
            )
            for emb in batch
        ]

        vectors = filter_zero_vectors(vectors)

        if vectors:
            try:
                index.upsert(vectors=vectors)
                print(f"✅ Uploaded batch of {len(vectors)} vectors to Pinecone")
            except Exception as e:
                print(f"❌ Error uploading batch to Pinecone: {e}")
        else:
            print("⚠️ No valid vectors to upload in this batch.")

    return "📦 Stored embeddings in Pinecone"


def filter_zero_vectors(vectors):

    non_zero_vectors = []
    for vector in vectors:
        if np.any(np.array(vector[1]) != 0):  
            non_zero_vectors.append(vector)
        else:
            print(f"Vector {vector[0]} contains only zeros and will not be upserted.")  
    return non_zero_vectors

def create_query_vector(question: str) -> np.ndarray:
    global embedding_model  
    if embedding_model is None:
        raise ValueError("Embedding model not initialized.")

    try:
        embedding = embedding_model.encode(question)
        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding for question: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")

def query_chromadb(question: str, quarters: list):
    try:
        collection = get_or_create_collection()

        where_filter = None
        if quarters and len(quarters) > 0:
            where_filter = {"quarter": {"$in": quarters}}

        results = collection.query(
            query_texts=[question],
            n_results=5,
            where=where_filter
        )

        if not results or 'ids' not in results or len(results['ids']) == 0 or len(results['ids'][0]) == 0:
            logger.warning("No results found in ChromaDB for the given query and filters")
            return []

        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return []


def query_pinecone(question: str, quarters: list):
    try:
        if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
            logger.error("Pinecone API key or index name not configured")
            return []

        query_vector = create_query_vector(question)

        vector_dim = len(query_vector)
        expected_dim = 384 
        
        if vector_dim != expected_dim:
            logger.warning(f"Query vector dimension {vector_dim} does not match expected dimension {expected_dim}")
        
        filter_dict = {}
        if quarters and len(quarters) > 0:
            filter_dict = {"quarter": {"$in": quarters}}

        pc = Pinecone(api_key=PINECONE_API_KEY)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist")
            return []
            
        index = pc.Index(PINECONE_INDEX_NAME)

        results = index.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )

        if not results.matches:
            logger.warning("No matches found in Pinecone for the given query and filters")
            return []
            
        retrieved_chunks = [match.metadata for match in results.matches]
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []


@app.post("/query")
def query_documents(payload: QueryPayload):
    try:
        question = payload.question
        vector_db = payload.vector_db
        quarters = payload.quarters

        logger.info(f"Query request: question='{question}', vector_db='{vector_db}', quarters={quarters}")

        if isinstance(quarters, str):
            quarters = [quarters]
        elif not quarters:
            quarters = []

        retrieved_chunks = []
        try:
            if vector_db == "pinecone":
                retrieved_chunks = query_pinecone(question, quarters)
            else:
                retrieved_chunks = query_chromadb(question, quarters)
        except Exception as query_error:
            logger.error(f"Error in vector DB query: {query_error}")

        if not retrieved_chunks:
            logger.warning(f"No chunks found for question: {question}")
            return {"answer": "No relevant information found.", "context": []}

        context_texts = []
        for chunk in retrieved_chunks:
            chunk_text = None
            if isinstance(chunk, dict) and 'text' in chunk:
                chunk_text = chunk['text']
            elif isinstance(chunk, str):
                chunk_text = chunk
                
            if chunk_text:
                context_texts.append(chunk_text)

        context = "\n".join(context_texts)

        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not configured")
            return {
                "answer": "Could not generate an answer due to missing OpenAI API configuration.",
                "context": context_texts
            }

        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": f"Answer the following question based on the context provided:\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"}
                ],
                max_tokens=150,
                temperature=0.5,
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer: {answer[:50]}...")
        except Exception as openai_error:
            logger.error(f"Error generating answer with OpenAI: {openai_error}")
            return {"answer": "Failed to generate an answer. Please try again later.", "context": context_texts}

        return {"answer": answer, "context": context_texts}

    except Exception as e:
        logger.error(f"Unhandled error in query_documents: {e}")
        return {"answer": "An error occurred while processing your request.", "context": []}

@app.post("/process_chunks")
async def process_chunks(payload: ProcessChunkPayload):
    """Process a file, chunk it, and store embeddings in a vector database."""
    try:
        filename = payload.filename
        chunking_strategy = payload.chunking_strategy
        vector_db = payload.vector_db
        quarter = payload.quarter

        if not filename or not chunking_strategy or not vector_db or not quarter:
            raise HTTPException(status_code=400, detail="Missing required parameters.")

        uploaded_files = [filename] 
        chunks = chunk_data(uploaded_files, chunking_strategy)
        logger.info(f"Generated {len(chunks)} chunks from {filename} using {chunking_strategy} strategy.")

        for chunk in chunks:
            chunk['metadata']['quarter'] = quarter  

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated.")

        embeddings = create_sentence_transformer_embeddings(chunks)
        logger.info(f"Created {len(embeddings)} embeddings.")

        if vector_db == "pinecone":
            storage_result = store_in_pinecone(embeddings)
        else:
            storage_result = store_in_chromadb(embeddings)

        logger.info(storage_result)
        return {"message": "File processed and embeddings stored successfully!"}

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    try:
        files = fetch_files_from_s3()
        return {"files": files}
    except Exception as e:
        logger.error(f"Error fetching files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AnswerRequest(BaseModel):
    question: str
    year: str
    quarter: str

client = OpenAI(api_key=os.getenv("GPT4o_API_KEY"))  

@app.post("/generate_answer")
def generate_answer(payload: AnswerRequest):
    try:
        query = payload.question
        selected_year = payload.year
        selected_quarter = payload.quarter.upper()

        year_match = re.search(r"\b(20\d{2})\b", query)
        quarter_match = re.search(r"\bQ([1-4])\b", query, re.IGNORECASE)

        if not year_match or not quarter_match:
            return {"answer": "No data found. Please include a valid year and quarter in your query."}

        query_year = year_match.group(1)
        query_quarter = f"Q{quarter_match.group(1)}"

        if query_year != selected_year or query_quarter.upper() != selected_quarter:
            return {"answer": "No data found. Mismatched year or quarter in query and selection."}

        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_vector = embedding_response.data[0].embedding

        index = pc.Index(PINECONE_INDEX_NAME)
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter={"year": {"$eq": query_year}, "quarter": {"$eq": query_quarter}}
        )

        matches = [
            m for m in results.matches
            if m.metadata.get("year") == query_year and m.metadata.get("quarter") == query_quarter
        ]

        if not matches:
            return {"answer": "No relevant information found."}

        context = "\n".join([m.metadata.get("text", "") for m in matches])

        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that answers financial questions based only on the provided context."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"}
            ]
        )

        answer = chat_response.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="RAG query failed.")
