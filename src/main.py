# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import answer_question

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = answer_question(request.query)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
