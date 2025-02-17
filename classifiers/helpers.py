import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
# from nltk.tokenize import sent_tokenize
from langchain.schema import Document
# import nltk
# nltk.download('punkt')
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np


embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
# rerank_model_name = "BAAI/bge-reranker-large" # USE THIS IF NEEDED A PRECISE RESULT
rerank_model_name = "BAAI/bge-reranker-base"  # USE THIS IF NEEDED A FASTER RESULT
tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)


def load_chunks(filename="preprocessed_chunks.json"):
    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {filename} was not found at {file_path}")

    with open(file_path, 'r') as f:
        chunk_dicts = json.load(f)

    return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunk_dicts]


def calculate_confidence(scores):
    probabilities = torch.sigmoid(torch.tensor(scores)).numpy()
    return float(np.max(probabilities))


# INCLUDES SENTENCE RERANKER FOR MINIMAL INPUT TOKENS
# def rerank_chunks(query, chunks, top_k=3):
#     sentences = []
#     for chunk in chunks:
#         try:
#             chunk_sentences = sent_tokenize(chunk.page_content)
#         except:
#             chunk_sentences = chunk.page_content.split('. ')
        
#         for sent in chunk_sentences:
#             sentences.append(Document(
#                 page_content=sent.strip(),
#                 metadata=chunk.metadata  
#             ))

#     pairs = [[query, doc.page_content] for doc in sentences]
#     inputs = tokenizer(pairs, padding=True, truncation=True, 
#                       return_tensors="pt", max_length=512)

#     with torch.no_grad():
#         scores = model(**inputs).logits.view(-1).float()

#     sorted_indices = scores.argsort(descending=True)
#     top_sentences = [sentences[i] for i in sorted_indices[:top_k]]
#     top_scores = scores[sorted_indices[:top_k]].numpy()

#     return top_sentences, top_scores


# NO SENTENCE RETREIVAL
def rerank_chunks(query, chunks, top_k=3):
    pairs = [[query, chunk.page_content] for chunk in chunks]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).float()

    sorted_indices = scores.argsort(descending=True)
    top_chunks = [chunks[i] for i in sorted_indices[:top_k]]
    top_scores = scores[sorted_indices[:top_k]].numpy()  # Extract scores for top results

    return top_chunks, top_scores 


def format_chunks(chunks):
    return "\n\n".join([f"Page {c.metadata['page']}: {c.page_content}" for c in chunks])


# def get_threshold_response():
#     return "I'm not entirely confident about this answer. Would you like to rephrase or ask about another topic?"


