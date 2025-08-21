import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads .env

# ---- Config ----
CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "support_knowledge")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # swap to llama-3.3-70b-versatile if desired
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Embedding model ----
_EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    id: str

def get_embedder():
    return SentenceTransformer(_EMBED_MODEL_NAME, trust_remote_code=True)

def get_chroma():
    # Persistent local DB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # If you wanted Chroma to auto-embed, you could pass an embedding function in collection creation.
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    return client, col

def simple_chunk(text: str, max_words: int = 220) -> List[str]:
    # Clean and chunk by words; good enough for FAQs
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return [c for c in chunks if c]

def upsert_documents(docs: List[Dict[str, Any]]):
    """
    docs: list of dicts with keys: id(str), text(str), metadata(dict)
    """
    embedder = get_embedder()
    _, collection = get_chroma()

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    vectors = embedder.encode(texts, convert_to_numpy=True).tolist()
    # Add/update
    # If ids exist, Chroma will append new entries unless you manage duplicates yourself.
    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

def query_similar(query: str, k: int = 4) -> List[RetrievedChunk]:
    embedder = get_embedder()
    _, collection = get_chroma()
    q_vec = embedder.encode([query], convert_to_numpy=True).tolist()
    res = collection.query(query_embeddings=q_vec, n_results=k)
    # Build results
    out = []
    for docs, metas, ids in zip(res.get("documents",[[]])[0], res.get("metadatas",[[]])[0], res.get("ids",[[]])[0]):
        out.append(RetrievedChunk(text=docs, metadata=metas or {}, id=ids))
    return out

def build_system_prompt():
    return (
        "You are an intelligent, empathetic, and professional AI assistant for customer support. "
        "Your primary goal is to assist the user based only on the provided context and past conversation history. "
        "Always remember and use previous interactions in this session to give coherent, personalized, and natural responses. "
        "If the user repeats a question, politely acknowledge and provide a consistent answer. "
        "If the user changes the topic, smoothly adapt and connect with earlier details where relevant. "
        "You can summarize or highlight important points from earlier in the conversation to show continuity. "
        "Reason step by step internally before answering, but keep responses simple, clear, and conversational for the user. "
        "Be concise but helpful, and show empathy in tone. "
        "If the required information is not in the context, politely say you don’t have that info instead of guessing."
    )


def make_rag_prompt(user_query: str, contexts: List[RetrievedChunk]) -> List[Dict[str, str]]:
    context_block = "\n\n".join(
        [f"[Doc #{i+1}] {c.text}" for i, c in enumerate(contexts)]
    )
    system = build_system_prompt()
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User question: {user_query}\n\nContext:\n{context_block}"},
    ]
    return msgs

def call_groq(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Put it in your .env file.")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()

def rag_answer(query: str, k: int = 4) -> Dict[str, Any]:
    hits = query_similar(query, k=k)
    messages = make_rag_prompt(query, hits)
    answer = call_groq(messages)
    return {
        "answer": answer,
        "sources": [
            {"id": h.id, "metadata": h.metadata} for h in hits
        ],
    }

def prepare_docs_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert CSV/records with fields: id, question, answer, category -> chunked docs
    """
    out = []
    for r in rows:
        base_text = f"Q: {r['question']}\nA: {r['answer']}"
        chunks = simple_chunk(base_text, max_words=180)
        for ci, ch in enumerate(chunks):
            out.append({
                "id": f"{r['id']}-{ci}",
                "text": ch,
                "metadata": {
                    "source_id": r["id"],
                    "category": r.get("category", "general"),
                    "part": ci
                }
            })
    return out
