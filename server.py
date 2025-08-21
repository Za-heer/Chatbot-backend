from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_utils import rag_answer, get_chroma

load_dotenv()

app = FastAPI(title="Groq Llama3 RAG Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    query: str
    top_k: int = 4

@app.on_event("startup")
def _startup():
    # Warm up Chroma (and optionally print stats)
    _, col = get_chroma()
    print(f"Chroma collection loaded. Current count = {col.count()}")

@app.post("/chat")
async def chat(req: ChatRequest):
    result = rag_answer(req.query, k=req.top_k)
    return result

