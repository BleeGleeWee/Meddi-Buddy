from fastapi import FastAPI, Query
from rag_pipeline import load_pipeline, ask_question

app = FastAPI(title="Medical RAG API")

# ✅ load once
chain = load_pipeline()


@app.get("/")
def home():
    return {"message": "Medical RAG API is running ✅"}


@app.post("/chat")
def chat(query: str = Query(..., description="User query")):
    try:
        result = ask_question(chain, query)
        return result
    except Exception as e:
        return {"error": str(e)}
