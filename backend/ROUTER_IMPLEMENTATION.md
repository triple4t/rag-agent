# Router Implementation - General + RAG Query Routing

## Overview

The system now implements an intelligent router pattern that classifies queries and routes them to either:
- **General Conversation Agent**: For general knowledge questions, casual conversation, or questions not related to uploaded documents
- **RAG Agent**: For questions about uploaded documents, specific content, or information that requires document search

## Architecture

Based on the [LangChain Router Pattern](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base) and routing concepts from the Medium article.

### Flow

```
User Query
    ↓
Router (Classification)
    ↓
    ├─→ General Route → General LLM → Response
    └─→ RAG Route → Hybrid Search → Rerank → Answer Generation → Response
```

## Key Components

### 1. Router Graph (`app/graph/router_graph.py`)

- **Classification Node**: Uses structured LLM output to classify queries
- **General Agent**: Handles conversational queries without document search
- **RAG Agent**: Handles document-based queries with full RAG pipeline

### 2. Router State (`app/graph/router_state.py`)

Tracks:
- Query classification (route, confidence, reasoning)
- Answer and quality metrics
- Sources (with filenames for RAG queries)
- Error handling

### 3. Query Classification Logic

**Routes to RAG when:**
- Query references uploaded documents
- Questions about specific content or data
- Requests for summaries or analysis of documents
- Questions that need document search

**Routes to General when:**
- General knowledge questions
- Conversational queries
- Questions not related to uploaded content
- No documents available (automatic fallback)

## Features

### ✅ Source Display Enhancement

Sources now show:
- **Filename** (e.g., "resume.pdf") instead of generic "Source 1"
- **Chunk index** (which chunk from the document)
- **Relevance score**
- **Content preview**

### ✅ Smart Routing

- Automatically detects if query needs document search
- Falls back to general conversation if no documents uploaded
- Handles errors gracefully

### ✅ Backward Compatible

- Existing RAG functionality preserved
- All existing endpoints work
- No breaking changes to API

## Example Queries

### General Route Examples:
- "What is machine learning?"
- "How does Python work?"
- "Tell me a joke"
- "Explain quantum computing"

### RAG Route Examples:
- "What does the document say about X?"
- "Summarize the uploaded PDF"
- "What are the key points in the document?"
- "Find information about authentication in the documents"

## Testing

Test both routes:

```bash
# General query (no documents needed)
curl -X POST "http://localhost:8000/api/v1/queries" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is artificial intelligence?"}'

# RAG query (requires documents)
curl -X POST "http://localhost:8000/api/v1/queries" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about authentication?"}'
```

## Benefits

1. **Better UX**: Users can ask general questions even without documents
2. **Intelligent Routing**: System automatically determines best approach
3. **Clear Sources**: Filenames make it clear which document provided information
4. **Scalable**: Easy to add more routes (e.g., web search, database queries)

