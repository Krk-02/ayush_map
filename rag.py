import requests
import io
import fitz  # pip install PyMuPDF
import os
import voyageai
from typing import List
import chromadb
from chromadb.config import Settings
from chunking_evaluation.chunking import (
    ClusterSemanticChunker
)
from chunking_evaluation.utils import openai_token_count
import openai
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
API_KEY = os.getenv("API_KEY")
app = FastAPI()
# Configure API key for Voyage AI
voyageai.api_key = os.getenv("VOYAGEAI_API_KEY")
openai.api_key = os.getenv("OPEN_AI_KEY")
vo = voyageai.Client()
def voyage_embedding_function(texts: List[str]) -> List[List[float]]:
    resp = vo.embed(texts, model="voyage-3.5-lite", input_type="document")
    return resp.embeddings

def extract_text_fitz(pdf_url: str) -> str:
    resp = requests.get(pdf_url)
    resp.raise_for_status()
    pdf_bytes = io.BytesIO(resp.content)

    # Read with fitz
    full_text = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            full_text.append(page.get_text())
        return "\n".join(full_text)

# Initialize a persistent ChromaDB client (local folder "./chroma_db")
client = chromadb.PersistentClient()

# Get or create a collection for embeddings
collection = client.get_or_create_collection(
    name="issac",
    metadata={"hnsw:space": "cosine"}
)
def voyage_token_count(texts):
    # texts can be a single string or a list of strings
    if isinstance(texts, str):
        texts = [texts]
    return vo.count_tokens(texts) 
def embed_in_batches(chunks: list[str] ,batch_size: int = 500, model: str = "voyage-3.5-lite", input_type: str = "document"):
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        resp = vo.embed(batch, model=model, input_type=input_type)
        all_embeddings.extend(resp.embeddings)
    return all_embeddings

def cluster_chunk_and_store(url,document: str, doc_id: str, max_chunk_size: int = 800, batch_size: int = 500):
    """
    Semantic cluster chunking and embedding storage using Voyage AI + ChromaDB.
    """
    chunker = ClusterSemanticChunker(
        embedding_function=voyage_embedding_function,
        max_chunk_size=max_chunk_size,
        length_function=voyage_token_count  # use a token-count function if desired
    )
    segments = chunker.split_text(document)
    
    ids = [f"{doc_id}_seg{idx}" for idx in range(len(segments))]
    
    embeddings = embed_in_batches(segments,batch_size=batch_size)
    metadatas = [{"source": url, "chunk_index": idx} for idx in range(len(segments))]
    collection.add(ids=ids, documents=segments, embeddings=embeddings, metadatas=metadatas)
    return segments
def format_retrieved_context(results, max_length=12000):
    """Intelligently formats retrieved documents into context"""
    seen_chunks = set()
    ordered_chunks = []
    total_length = 0
    
    # Process in order of relevance (already sorted by ChromaDB)
    for doc, metadata, score in zip(results['documents'][0], 
                                  results['metadatas'][0], 
                                  results['distances'][0]):
        # Skip duplicates and low-quality matches
        if doc in seen_chunks or score > 0.5:  # Adjust threshold as needed
            continue
            
        seen_chunks.add(doc)
        chunk_length = len(doc)
        
        # Stop before exceeding token limits
        if total_length + chunk_length > max_length:
            break
            
        ordered_chunks.append({
            'text': doc,
            'source': metadata.get('source', 'unknown'),
            'chunk_idx': metadata.get('chunk_index', -1)
        })
        total_length += chunk_length
    
    # Reconstruct in original document order
    ordered_chunks.sort(key=lambda x: x['chunk_idx'])
    
    # Format with metadata headers
    context_parts = []
    for chunk in ordered_chunks:
        context_parts.append(
            f"【Source: {chunk['source']} | Segment {chunk['chunk_idx']}】\n"
            f"{chunk['text']}\n\n"
        )
    
    return "".join(context_parts).strip()
def orchastrator(url,questions):
    # url = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"

    text2 = extract_text_fitz(url)
    # print("\n=== PyMuPDF Extracted Text ===")
    print(len(text2))
    luster_chunk_and_store(url,text2,f"{url}_{len(text2)}")
    print("done")
    # questions = ["According to Newton, what are the three laws of motion and how do they apply in celestial mechanics?"]
    answers=[]
    for question in questions : 
        query_embedding = vo.embed(
                texts=[question],
                model="voyage-3.5-lite",
                input_type="query"
            ).embeddings[0]
            
            # Retrieve relevant context
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"source": url},  # Filter by source
            include=["documents", "metadatas", "distances"]
        )
# Filter by distance threshold
        context = format_retrieved_context(results)
        print(context)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert AI assistant
                                   Answer based on the provided context and ONLY about what is asked in the question. 
                                   Be concise(2-4 sentences). If the context contains partial information, fill in the gaps and answer.
                                   Only say 'I don't know' if the question is completely irrelevant to the document,
                                   or if the question is unsafe and intrusive say 'Irrelavent question'"""
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        answers.append(answer)
    print(answers)
    return answers

prompt = input("Enter the query: ")
orchastrator("https://iris.who.int/bitstream/handle/10665/365543/9789240064935-eng.pdf?sequence=1",[prompt])
