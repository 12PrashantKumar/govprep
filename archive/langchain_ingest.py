import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def ingest_exam_materials(file_path:str):
    print(f"Reading {file_path}....")
    loader = TextLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap =200)
    chunks  = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # This creates the connection point: a folder named "govprep_langchain_db"
    Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = "./govprep_lanchain_db"

    )
    print("Ingestion complete. Database updated.")

if __name__ == "__main__":

    ingest_exam_materials("data/plity_ncert_ch_01.pdf")
    