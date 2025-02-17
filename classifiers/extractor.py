import re
from langchain.document_loaders import PyPDFLoader
import json
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import os


filename = "peta_pdf"
file_path = os.path.join(os.getcwd(), filename)


def split_by_paragraph(text):
    paragraphs = re.split(r'\n\s*\n+', text.strip()) 
    return [p.strip() for p in paragraphs if p.strip()] 


def clean_text(text):
    return re.sub(r'[\n\t\r]+', ' ', text)


def save_chunks(chunks, filename="preprocessed_chunks.json"):
    chunk_dicts = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(filename, 'w') as f:
        json.dump(chunk_dicts, f)


def extracter_ttp():
    loader = PyPDFLoader("peta_pdf.pdf")
    docs = loader.load()

    paragraph_chunks = []
    for doc in docs:
        cleaned_text = clean_text(doc.page_content)
        paragraphs = split_by_paragraph(cleaned_text)
        for para in paragraphs:
            paragraph_chunks.append({"page_content": para, "metadata": doc.metadata})

    from langchain.schema import Document
    chunks = [Document(page_content=para["page_content"], metadata=para["metadata"]) for para in paragraph_chunks]
    save_chunks(chunks)

    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

    vectorstore = Chroma(persist_directory="./chroma_db_pedition", embedding_function=embedding_function)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks] 

    for i in tqdm(range(0, len(texts), 10), desc="Storing documents in ChromaDB", unit="batch"):
        vectorstore.add_texts(texts=texts[i:i+10], metadatas=metadatas[i:i+10]) 
    vectorstore.persist()
