from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import ask_question

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    provider: str = "openai"

@app.get("/")
def root():
    return {"message": "Legal RAG API running"}

@app.post("/ask")
def ask(req: QueryRequest):
    response = ask_question(req.question, req.provider)
    return response