import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline.rag import RAGPipeline  # Import from the module

app = FastAPI()

# Initialize RAGPipeline outside the endpoint
try:
    output_dir = os.getenv("OUTPUT_DIR", "./output")  # Default to './output'
    rag_pipeline = RAGPipeline(output_dir=output_dir)
except Exception as e:
    logging.error(f"Failed to initialize RAGPipeline: {e}", exc_info=True)
    # Consider a more graceful shutdown or error handling if the RAG pipeline fails to initialize
    raise  # Re-raise the exception to prevent the app from starting in a broken state


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        answer = rag_pipeline.get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
