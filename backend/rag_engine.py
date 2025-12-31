"""RAG Engine for PDF processing and querying with Ollama using FAISS."""
import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import faiss
import httpx
from PyPDF2 import PdfReader


class RAGEngine:
    """Handles PDF processing, embedding, and RAG queries using FAISS."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        embedding_model: str = "nomic-embed-text",
        persist_dir: str = "./vector_db"
    ):
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.embedding_dim = 768  # nomic-embed-text dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[str] = []
        self.metadata: List[Dict] = []

        self._load_index()

    def _load_index(self):
        """Load existing index from disk if available."""
        index_path = self.persist_dir / "faiss.index"
        docs_path = self.persist_dir / "documents.pkl"

        if index_path.exists() and docs_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.metadata = []

    def _save_index(self):
        """Save index to disk."""
        index_path = self.persist_dir / "faiss.index"
        docs_path = self.persist_dir / "documents.pkl"

        faiss.write_index(self.index, str(index_path))
        with open(docs_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # Try to end at a sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - chunk_overlap

        return [c for c in chunks if c]

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            # Normalize for cosine similarity
            arr = np.array(embedding, dtype=np.float32)
            arr = arr / np.linalg.norm(arr)
            return arr

    async def add_document(
        self,
        pdf_path: str,
        doc_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> dict:
        """Process and add a PDF document to the vector store."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            raise ValueError("Could not extract text from PDF")

        # Chunk text with provided parameters
        chunks = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Remove existing chunks for this document
        self._remove_document(doc_id)

        # Get embeddings and add to FAISS
        embeddings = []
        for chunk in chunks:
            embedding = await self.get_embedding(chunk)
            embeddings.append(embedding)
            self.documents.append(chunk)
            self.metadata.append({"doc_id": doc_id, "chunk_index": len(self.documents) - 1})

        # Add to FAISS index
        embeddings_array = np.vstack(embeddings)
        self.index.add(embeddings_array)

        self._save_index()

        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "characters": len(text)
        }

    def _remove_document(self, doc_id: str):
        """Remove all chunks for a document. Rebuilds the index."""
        if not self.metadata:
            return

        # Find indices to keep
        indices_to_keep = [i for i, m in enumerate(self.metadata) if m["doc_id"] != doc_id]

        if len(indices_to_keep) == len(self.metadata):
            return  # Nothing to remove

        if not indices_to_keep:
            # Remove everything
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.metadata = []
            self._save_index()
            return

        # Rebuild index with remaining documents
        # We need to re-fetch embeddings or store them - for simplicity, rebuild
        new_documents = [self.documents[i] for i in indices_to_keep]
        new_metadata = [{"doc_id": self.metadata[i]["doc_id"], "chunk_index": j}
                       for j, i in enumerate(indices_to_keep)]

        # Reconstruct vectors from index
        old_vectors = np.vstack([self.index.reconstruct(i) for i in indices_to_keep])

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(old_vectors)
        self.documents = new_documents
        self.metadata = new_metadata
        self._save_index()

    async def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        n_results: int = 5
    ) -> dict:
        """Query the RAG system with a question."""
        if self.index.ntotal == 0:
            return {
                "answer": "No documents found. Please upload a PDF first.",
                "sources": []
            }

        # Get embedding for the question
        question_embedding = await self.get_embedding(question)
        question_embedding = question_embedding.reshape(1, -1)

        # Search in FAISS
        k = min(n_results * 3, self.index.ntotal)  # Get more results to filter
        distances, indices = self.index.search(question_embedding, k)

        # Filter by doc_id if specified
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if doc_id is None or meta["doc_id"] == doc_id:
                results.append({
                    "text": self.documents[idx],
                    "distance": float(dist),
                    "doc_id": meta["doc_id"]
                })
            if len(results) >= n_results:
                break

        if not results:
            return {
                "answer": "No relevant content found in the selected document.",
                "sources": []
            }

        # Build context from retrieved chunks
        context = "\n\n---\n\n".join([r["text"] for r in results])

        # Generate answer using Ollama
        answer = await self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "distance": r["distance"]
                }
                for r in results
            ]
        }

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using Ollama."""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer: /no_think"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]

    def list_documents(self) -> List[str]:
        """List all unique document IDs."""
        doc_ids = set()
        for meta in self.metadata:
            doc_ids.add(meta["doc_id"])
        return list(doc_ids)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the collection."""
        if doc_id not in [m["doc_id"] for m in self.metadata]:
            return False
        self._remove_document(doc_id)
        return True
