from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pytesseract
from langchain.schema import Document
import os
from PIL import Image
from dotenv import load_dotenv


# Load API Key
load_dotenv()  # Load variables from .env file
openai_api_key = os.getenv('OPENAI_API_KEY')

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os

def create_vector_embeddings(information_dir):
    files = os.listdir(information_dir)
    pdf_paths = []
    image_paths = []
    for file in files:
        full_path = os.path.join(information_dir, file)
        if file.lower().endswith('.pdf'):
            pdf_paths.append(full_path)
        elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(full_path)

    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    

    Chroma(embedding_function=embedding_function, persist_directory="./chroma_db").delete_collection()
    
    # Create a new Chroma collection
    vectorstore = Chroma(embedding_function=embedding_function, persist_directory="./chroma_db")

    # Process PDF documents
    docs = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(documents=splits)

    # Process images
    for path in image_paths:
        image = Image.open(path)
        extracted_text = pytesseract.image_to_string(image)
        splits = text_splitter.split_text(extracted_text)
        vectorstore.add_texts(texts=splits)

    return vectorstore