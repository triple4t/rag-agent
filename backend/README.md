# RAG System Backend

Production-ready FastAPI backend for RAG System with Hybrid Search, Reranking, and LangGraph orchestration.

## Features

- **FastAPI** - Modern, fast web framework
- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - LLM integration
- **Hybrid Search** - Vector + Keyword search with RRF fusion
- **Reranking** - Cohere API or local CrossEncoder
- **Azure Document Intelligence** - OCR for scanned documents
- **OpenAI Embeddings** - text-embedding-3-small
- **Qdrant** - Vector database
- **Redis** - Caching support
- **Async/Await** - Full async support
- **Docker** - Containerized deployment
- **Type Safety** - Full type hints with Pydantic
- **API Documentation** - Auto-generated OpenAPI/Swagger docs

## Architecture

The backend follows a clean architecture pattern:

```
app/
├── api/          # API routes and middleware
├── agents/       # LangGraph agents
├── core/         # Core infrastructure
├── graph/        # LangGraph workflows
├── search/       # Search components
├── utils/        # Utilities
└── optimization/ # Performance optimization
```

## Quick Start

### Prerequisites

- Python 3.11+
- Qdrant (via Docker or local installation)
- Redis (optional, for caching)
- OpenAI API key
- Azure API key (for OCR)

### Installation

1. Clone the repository and navigate to the backend directory:
```bash
cd rag-agent/backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start Qdrant (if not using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

6. Run the development server:
```bash
make run
# Or
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Docker Setup

### Using Docker Compose

1. Build and start services:
```bash
docker-compose up -d
```

2. View logs:
```bash
docker-compose logs -f backend
```

### Manual Docker Build

```bash
docker build -t rag-system-backend .
docker run -p 8000:8000 --env-file .env rag-system-backend
```

## API Endpoints

### Health Check
- `GET /health` - Health check
- `GET /health/ready` - Readiness check

### Documents
- `POST /api/v1/documents/upload` - Upload and process PDFs
- `GET /api/v1/documents` - List loaded documents
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Remove document

### Queries
- `POST /api/v1/queries` - Submit query to RAG system
- `GET /api/v1/queries/{id}` - Get query result (future)

### Metrics
- `GET /api/v1/metrics` - Get system metrics
- `GET /api/v1/metrics/success-criteria` - Check success criteria status
- `GET /api/v1/metrics/cost` - Get cost report
- `GET /api/v1/metrics/report` - Get full metrics report

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## LangGraph Workflow

The RAG workflow consists of four stages:

1. **Hybrid Search** - Vector + Keyword search with RRF fusion
2. **Reranking** - Reorder results by relevance
3. **Answer Generation** - LLM generates answer from context
4. **Quality Reflection** - Score answer quality

## Success Metrics

The system tracks the following metrics (from roadmap lines 96-100):

- **MRR@5** (Mean Reciprocal Rank) ≥ 0.7
- **Quality Score** ≥ 0.75
- **Latency** < 2s p95
- **Cost per Query** < $0.01

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint
```

## Environment Variables

See `.env.example` for all available environment variables.

Key variables:
- `OPENAI_API_KEY` - OpenAI API key (required)
- `AZURE_API_KEY` - Azure API key (required)
- `AZURE_MISTRAL_OCR_ENDPOINT` - Azure OCR endpoint (required)
- `AZURE_MISTRAL_MODEL` - Azure OCR model (default: mistral-document-ai-2505)
- `COHERE_API_KEY` - Cohere API key (optional, for reranking)
- `LANGCHAIN_TRACING_V2` - Enable LangSmith tracing
- `LANGCHAIN_API_KEY` - LangSmith API key (optional)
- `QDRANT_URL` - Qdrant URL (default: localhost)
- `REDIS_URL` - Redis URL (optional, for caching)

## Production Deployment

1. Set `ENVIRONMENT=production` in `.env`
2. Configure proper `SECRET_KEY`
3. Set up Qdrant and Redis
4. Configure CORS origins
5. Build Docker image:
```bash
docker build -t rag-system-backend .
```
6. Deploy using your preferred platform (AWS, GCP, Azure, etc.)

## Troubleshooting

### Document Processing Issues
- Ensure Azure API key and endpoint are correct
- Check file format (PDF only)
- Verify OCR model name

### Search Issues
- Verify Qdrant is running and accessible
- Check embedding model configuration
- Ensure documents are loaded

### API Errors
- Check API keys are set correctly
- Verify environment variables
- Check logs for detailed error messages

## License

MIT

## Support

For issues and questions, please open an issue on GitHub.

