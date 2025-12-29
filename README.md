# PDF RAG Chat with Ollama

A local Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using Ollama's local LLM models. Upload PDFs through a web interface and ask questions about their content.

## Features

- **PDF Upload**: Drag-and-drop or click to upload PDF documents
- **Local Processing**: All data stays on your machine - no external API calls
- **Vector Search**: FAISS-powered semantic search for relevant document chunks
- **Chat Interface**: Clean, modern UI for conversing with your documents
- **Source Citations**: See which parts of the document were used to generate answers
- **Multi-Document Support**: Upload and manage multiple PDFs
- **Docker Support**: Easy deployment with Docker Compose

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Browser                              │
│                    (http://localhost:8000)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│                         (app.py)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   /upload   │  │   /chat     │  │   /documents            │  │
│  │   endpoint  │  │   endpoint  │  │   endpoint              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Engine                                  │
│                  (backend/rag_engine.py)                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PDF Processing Pipeline                                  │   │
│  │  1. Extract text (PyPDF2)                                │   │
│  │  2. Chunk text (500 chars, 50 overlap)                   │   │
│  │  3. Generate embeddings (Ollama)                         │   │
│  │  4. Store in FAISS index                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Query Pipeline                                           │   │
│  │  1. Embed question (Ollama)                              │   │
│  │  2. Search FAISS for similar chunks                      │   │
│  │  3. Build context from top-k results                     │   │
│  │  4. Generate answer with LLM (Ollama)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────┐
│      FAISS Vector DB      │   │           Ollama              │
│      (vector_db/)         │   │    (http://localhost:11434)   │
│                           │   │                               │
│  - faiss.index            │   │  Models:                      │
│  - documents.pkl          │   │  - qwen3:8b (chat)            │
│                           │   │  - nomic-embed-text (embed)   │
└───────────────────────────┘   └───────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python) |
| Vector Store | FAISS |
| LLM | Qwen3:8b via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| PDF Parsing | PyPDF2 |
| Frontend | HTML/CSS/JavaScript |
| Containerization | Docker |

## Prerequisites

- Python 3.10+ (3.12 recommended)
- [Ollama](https://ollama.ai/) installed and running
- ~6GB disk space for models

## Installation

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ogamircs/ollama_rag_ui.git
   cd ollama_rag_ui
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Pull required Ollama models**
   ```bash
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to http://localhost:8000

### Option 2: Docker (with GPU)

```bash
docker-compose up --build
```

### Option 3: Docker (CPU only)

```bash
docker-compose -f docker-compose.cpu.yml up --build
```

After starting with Docker, pull the models:
```bash
docker exec -it ollama ollama pull qwen3:8b
docker exec -it ollama ollama pull nomic-embed-text
```

## Usage

1. **Upload a PDF**: Click the upload area or drag and drop a PDF file
2. **Wait for processing**: The document will be chunked and embedded
3. **Ask questions**: Type your question in the chat input
4. **View sources**: Click "View sources" to see which document sections were used

## Configuration

Environment variables (can be set in `.env` or passed to Docker):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MODEL_NAME` | `qwen3:8b` | Chat model to use |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model to use |

## Project Structure

```
ollama_rag_ui/
├── app.py                 # FastAPI application
├── backend/
│   ├── __init__.py
│   └── rag_engine.py      # RAG logic (PDF processing, FAISS, Ollama)
├── templates/
│   └── index.html         # Web UI
├── static/                # Static assets
├── uploads/               # Uploaded PDFs (gitignored)
├── vector_db/             # FAISS index storage (gitignored)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker setup with GPU support
└── docker-compose.cpu.yml # Docker setup for CPU-only systems
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve web UI |
| `POST` | `/api/upload` | Upload and process a PDF |
| `POST` | `/api/chat` | Send a question and get an answer |
| `GET` | `/api/documents` | List all uploaded documents |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document |
| `GET` | `/api/health` | Health check |

## How It Works

### Document Ingestion
1. PDF is uploaded via the web interface
2. Text is extracted using PyPDF2
3. Text is split into overlapping chunks (500 chars, 50 char overlap)
4. Each chunk is embedded using `nomic-embed-text` model
5. Embeddings are normalized and stored in a FAISS index
6. Document metadata is persisted to disk

### Query Processing
1. User question is embedded using the same embedding model
2. FAISS performs similarity search to find top-k relevant chunks
3. Retrieved chunks are assembled into a context
4. Context + question are sent to `qwen3:8b` for answer generation
5. Answer and source citations are returned to the UI

## Troubleshooting

**Ollama connection refused**
- Ensure Ollama is running: `ollama serve`
- Check the URL in environment variables

**Model not found**
- Pull the required models: `ollama pull qwen3:8b && ollama pull nomic-embed-text`

**Out of memory**
- Try a smaller model like `qwen3:4b` or `llama3.2:3b`
- Reduce chunk size in `rag_engine.py`

**PDF text extraction fails**
- Some PDFs are image-based and require OCR (not currently supported)
- Try a different PDF or use a tool to convert images to text first

## License

MIT
