from fastapi import FastAPI, UploadFile, Request
import shutil
from ingest import ingest_documents, create_vector_store, create_vector_store_with_retry
from rag_bot import create_rag_bot, ask_question
from upload import upload_document_to_s3, s3_client
import os
import boto3
import logging
from langchain_community.vectorstores import FAISS
from document_processor import process_document
from langchain_openai import OpenAIEmbeddings
import tempfile
from config import S3_BUCKET_NAME
import faiss

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

app = FastAPI()
s3_client = boto3.client('s3')
vector_store = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the path where FAISS index is stored
index_path = "./faiss_indexes/latest.index"  # Change as needed

# Global vector store variable
vector_store = None

# Load existing FAISS index if available
if os.path.exists(index_path):
    vector_store = FAISS.load_local(index_path, OpenAIEmbeddings(openai_api_key=openai_api_key))
    print(f"✅ Loaded existing vector store from: {index_path}")
else:
    print("⚠️ No existing FAISS index found. Upload a document first.")


@app.on_event("startup")
async def startup_event():
    global vector_store
    index_name = "latest.index"  # Change as needed
    index_path = f"./faiss_indexes/{index_name}"

    try:
        # Download the FAISS index from S3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as temp_file:
            s3_client.download_file(S3_BUCKET_NAME, index_name, temp_file.name)
            temp_file_path = temp_file.name

        vector_store = FAISS.load_local(temp_file_path, OpenAIEmbeddings(openai_api_key=openai_api_key))
        os.remove(temp_file_path)
        print(f"✅ Loaded existing vector store from S3: {index_name}")
    except Exception as e:
        print(f"⚠️ No existing FAISS index found in S3: {e}. Upload a document first.")

@app.on_event("shutdown")
async def shutdown_event():
    # Perform any necessary cleanup here
    pass

@app.post("/upload/")
async def upload_document(file: UploadFile):
    global vector_store  # Ensure we're modifying the global vector store

    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Uploading {file.filename} to S3...")
        with open(temp_file_path, "rb") as buffer:
            s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, file.filename)
        logger.info("Upload successful!")

        # Process the file for vector storage
        logger.info(f"Processing file {file.filename} for vector store...")
        chunks = ingest_documents(temp_file_path)
        vector_store = create_vector_store(chunks, index_name=f"{file.filename}.index")

        return {"message": "File uploaded & indexed successfully!", "index_name": f"{file.filename}.index"}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        return {"message": f"Failed to upload document: {e}"}

'''@app.post("/upload/")
async def upload_document(file: UploadFile):
    global vector_store
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = ingest_documents(file.filename)
   # vector_store = create_vector_store(chunks)
    vector_store = create_vector_store_with_retry(chunks, api_key=openai_api_key)
    return {"message": "Document uploaded and indexed!"}'''

'''@app.get("/ask/")
async def ask(query: str):
    if vector_store is None:
        return {"message": "No documents available. Please upload a document first."}

    qa_chain = create_rag_bot(vector_store)
    answer, sources = ask_question(qa_chain, query)
    return {"answer": answer, "sources": [s.metadata['source'] for s in sources]}
'''

'''@app.get("/ask/")
async def ask(request: Request):
    if vector_store is None:
        return {"message": "No documents available. Please upload a document first."}

    data = await request.json()
    query = data.get("query")
    if not query:
        return {"message": "Query not provided in the request."}

    qa_chain = create_rag_bot(vector_store)
    answer, sources = ask_question(qa_chain, query)
    unique_sources = list({s.metadata['source'] for s in sources})
    return {"answer": answer, "sources": unique_sources}'''


@app.get("/ask/")
async def ask(request: Request):
    global vector_store  # Ensure we use the global variable

    if vector_store is None:
        return {"message": "No documents available. Please upload a document first."}

    data = await request.json()
    query = data.get("query")
    if not query:
        return {"message": "Query not provided in the request."}

    # Create the QA chain
    qa_chain = create_rag_bot(vector_store)
    answer, sources = ask_question(qa_chain, query)

    # Extract unique source documents
    unique_sources = list({s.metadata.get('source', 'Unknown') for s in sources})

    return {"answer": answer, "sources": unique_sources}
