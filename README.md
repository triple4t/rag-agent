# RAG Assistant

Production-ready RAG (Retrieval-Augmented Generation) Assistant application with a modern React frontend and FastAPI backend.

## Overview

A complete RAG system that allows users to upload PDF documents and ask questions about their content. The system uses hybrid search (vector + keyword), reranking, and LangGraph orchestration to provide accurate, source-cited answers.

## Architecture

```
rag-agent/
├── backend/          # FastAPI backend with LangGraph
├── frontend/         # React + TypeScript frontend
└── INTEGRATION.md    # Integration guide
```

## Features

- **Document Management**: Upload, view, and delete PDF documents
- **Hybrid Search**: Vector + keyword search with RRF fusion
- **Reranking**: Cohere API or local CrossEncoder for better results
- **Query Routing**: Intelligent routing between RAG and general conversation
- **Source Citations**: Answers include source documents with relevance scores
- **Quality Scoring**: Automatic quality assessment of generated answers
- **Modern UI**: ChatGPT-style clean interface
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- ChromaDB (vector database, embedded - no separate service needed)
- OpenAI API key
- Azure API key (for OCR)
- Cohere API key (optional, for reranking)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run backend
make run
# or
uvicorn main:app --reload
```

Backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend
npm install

# Configure environment variables
cp .env.example .env.local
# Edit .env.local with your backend URL (default: http://localhost:8000/api/v1)

# Run frontend
npm run dev
```

Frontend will be available at `http://localhost:8080`

## Project Structure

### Backend

- `app/api/` - API routes and middleware
- `app/agents/` - LangGraph agents (answer generation, quality reflection)
- `app/graph/` - LangGraph workflows (router, RAG)
- `app/search/` - Search components (hybrid search, reranking)
- `app/utils/` - Utilities (document loading, metrics)
- `app/optimization/` - Performance optimization (caching, cost tracking)

### Frontend

- `src/components/` - React components
- `src/hooks/` - Custom React hooks
- `src/lib/` - API client and utilities
- `src/pages/` - Page components
- `src/types/` - TypeScript type definitions

## API Endpoints

### Documents
- `GET /api/v1/documents` - List all documents
- `POST /api/v1/documents/upload` - Upload PDF documents
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document

### Queries
- `POST /api/v1/queries` - Submit query to RAG system

### Health
- `GET /health` - Health check
- `GET /health/ready` - Readiness check

## Configuration

### Backend Environment Variables

Key variables (see `backend/.env.example`):
- `OPENAI_API_KEY` - Required
- `AZURE_API_KEY` - Required for OCR
- `AZURE_MISTRAL_OCR_ENDPOINT` - Required for OCR
- `COHERE_API_KEY` - Optional, for reranking
- `CORS_ORIGINS` - Frontend URLs

### Frontend Environment Variables

- `VITE_API_URL` - Backend API URL (default: `http://localhost:8000/api/v1`)

## Development

### Backend

```bash
cd backend
make run          # Start development server
make test         # Run tests
make lint         # Lint code
make format       # Format code
```

### Frontend

```bash
cd frontend
npm run dev       # Start development server
npm run build     # Build for production
npm run lint      # Lint code
```

## Docker

### Backend

```bash
cd backend
docker-compose up -d
```

### Manual Build

```bash
docker build -t rag-assistant-backend .
docker run -p 8000:8000 --env-file .env rag-assistant-backend
```

## Documentation

- `backend/README.md` - Backend documentation
- `frontend/README.md` - Frontend documentation
- `INTEGRATION.md` - Frontend-backend integration guide
- API Docs: `http://localhost:8000/docs` (Swagger UI)

## Technology Stack

### Backend
- FastAPI
- LangGraph & LangChain
- OpenAI (GPT-4o, embeddings)
- Azure Document Intelligence (OCR)
- ChromaDB (vector database, embedded)
- Cohere (reranking)

### Frontend
- React 18
- TypeScript
- Vite
- Tailwind CSS
- shadcn-ui
- React Query

## License

MIT

