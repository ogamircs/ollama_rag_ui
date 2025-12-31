"""FastAPI application for PDF RAG with Ollama."""
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine

# Configuration
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Initialize app and RAG engine
app = FastAPI(title="PDF RAG with Ollama", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG engine
rag_engine = RAGEngine(
    ollama_base_url=OLLAMA_BASE_URL,
    model_name=MODEL_NAME,
    embedding_model=EMBEDDING_MODEL
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    doc_id: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list


class DocumentInfo(BaseModel):
    """Response model for document info."""
    doc_id: str
    filename: str
    chunks: int
    characters: int


# Store document metadata
documents_metadata = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    return FileResponse("templates/index.html")


@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
) -> DocumentInfo:
    """Upload and process a PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Generate unique document ID
    doc_id = str(uuid.uuid4())[:8]

    # Save file
    file_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Process with RAG engine
        result = await rag_engine.add_document(
            str(file_path),
            doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Store metadata
        documents_metadata[doc_id] = {
            "filename": file.filename,
            "path": str(file_path),
            "chunks": result["chunks"],
            "characters": result["characters"]
        }

        return DocumentInfo(
            doc_id=doc_id,
            filename=file.filename,
            chunks=result["chunks"],
            characters=result["characters"]
        )
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the RAG system."""
    try:
        # Use selected model if provided
        if request.model:
            rag_engine.model_name = request.model

        result = await rag_engine.query(
            question=request.message,
            doc_id=request.doc_id
        )
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents() -> list:
    """List all uploaded documents."""
    doc_ids = rag_engine.list_documents()
    return [
        {
            "doc_id": doc_id,
            "filename": documents_metadata.get(doc_id, {}).get("filename", "Unknown"),
            "chunks": documents_metadata.get(doc_id, {}).get("chunks", 0)
        }
        for doc_id in doc_ids
    ]


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    success = rag_engine.delete_document(doc_id)
    if success:
        # Remove file
        if doc_id in documents_metadata:
            path = documents_metadata[doc_id].get("path")
            if path and os.path.exists(path):
                os.remove(path)
            del documents_metadata[doc_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/api/models")
async def list_models():
    """List available Ollama models."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [
                {
                    "name": m["name"],
                    "size": m.get("size", 0),
                    "modified": m.get("modified_at", "")
                }
                for m in data.get("models", [])
            ]
            return {"models": models, "current": rag_engine.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": rag_engine.model_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
