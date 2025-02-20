from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import time
import logging
from openai import OpenAIError
import boto3
import tempfile
from config import S3_BUCKET_NAME
from io import BytesIO
import faiss

load_dotenv()

s3_client = boto3.client('s3')

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

def ingest_documents(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

'''def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store'''

def create_vector_store_with_retry(chunks, api_key, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            embeddings = OpenAIEmbeddings(api_key=api_key)
            vector_store = FAISS.from_documents(chunks, embeddings)
            return vector_store
        except OpenAIError as e:
            retries += 1
            wait_time = 2 ** retries  # Exponential backoff
            logging.warning(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded. Please try again later.")


def create_vector_store(chunks, index_name="vector_store.index"):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS index to a temporary file
    index_temp_file = os.path.join(tempfile.gettempdir(), index_name)
    faiss.write_index(vector_store.index, index_temp_file)

    # Upload the FAISS index to S3
    with open(index_temp_file, "rb") as buffer:
        s3_client.upload_fileobj(buffer, S3_BUCKET_NAME, index_name)

    # Clean up the temporary file
    os.remove(index_temp_file)

    return {"message": "Vector store created successfully and uploaded to S3", "index_name": index_name}