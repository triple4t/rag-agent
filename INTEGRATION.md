# Frontend-Backend Integration Guide

This document describes the integration between the frontend and backend of the RAG Assistant application.

## Integration Status

✅ **Fully Integrated** - The frontend is now connected to the backend with production-ready practices.

## Key Integration Points

### 1. API Client (`src/lib/api.ts`)

A production-ready API client that handles:
- ✅ All backend endpoints (documents, queries)
- ✅ Proper error handling with custom `APIError` class
- ✅ Network error detection
- ✅ Type-safe responses
- ✅ Environment variable configuration

### 2. Type Definitions (`src/types/index.ts`)

TypeScript types that match the backend response structure:
- ✅ `Document` - Matches backend `DocumentResponse`
- ✅ `QueryResponse` - Matches backend query response
- ✅ `Source` - Matches backend source structure
- ✅ `Message` - Frontend chat message format

### 3. Custom Hook (`src/hooks/useRAG.ts`)

React hook that manages:
- ✅ Document CRUD operations
- ✅ Query submission
- ✅ State management
- ✅ Error handling with user feedback
- ✅ Loading states

### 4. Environment Configuration

- ✅ `.env.local` - Local development configuration (gitignored)
- ✅ `.env.example` - Template for environment variables
- ✅ `VITE_API_URL` - Backend API URL configuration

## API Endpoints Used

### Documents API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/documents` | List all documents |
| POST | `/api/v1/documents/upload` | Upload PDF files |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete document |

### Queries API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/queries` | Submit query to RAG system |

## Data Flow

### Document Upload Flow

```
User selects files
  ↓
UploadArea component
  ↓
useRAG.uploadDocuments()
  ↓
documentsApi.upload()
  ↓
Backend POST /documents/upload
  ↓
Backend processes PDFs
  ↓
Response with document metadata
  ↓
Frontend updates document list
  ↓
UI shows success message
```

### Query Flow

```
User types question
  ↓
ChatInput component
  ↓
useRAG.sendQuery()
  ↓
queriesApi.submit()
  ↓
Backend POST /queries
  ↓
Backend processes with RAG
  ↓
Response with answer + sources
  ↓
Frontend displays message
  ↓
User sees answer with sources
```

## Error Handling

### Network Errors
- Detected when fetch fails
- Shows: "Unable to connect to the server"
- User-friendly toast notification

### HTTP Errors
- 4xx errors: Client errors (bad request, not found, etc.)
- 5xx errors: Server errors
- Error message from backend displayed to user

### Validation Errors
- File type validation (PDF only)
- Empty query validation
- User-friendly error messages

## Environment Variables

### Development

```env
VITE_API_URL=http://localhost:8000/api/v1
```

### Production

```env
VITE_API_URL=https://api.yourdomain.com/api/v1
```

## Testing the Integration

### 1. Start Backend

```bash
cd backend
make run
# Backend should be running on http://localhost:8000
```

### 2. Start Frontend

```bash
cd frontend
npm run dev
# Frontend should be running on http://localhost:8080
```

### 3. Test Document Upload

1. Open frontend in browser
2. Click or drag PDF files to upload area
3. Verify documents appear in sidebar
4. Check backend logs for processing

### 4. Test Query

1. Upload at least one document
2. Type a question in the chat input
3. Press Enter to send
4. Verify response appears with sources
5. Check backend logs for query processing

## Troubleshooting

### Frontend can't connect to backend

1. **Check backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify `.env.local`**:
   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   ```

3. **Check CORS settings** in backend:
   - Backend should allow `http://localhost:8080`
   - Check `CORS_ORIGINS` in backend config

### Documents not uploading

1. **Check file format**: Only PDF files are accepted
2. **Check file size**: Backend may have size limits
3. **Check backend logs**: Look for error messages
4. **Check browser console**: Look for network errors

### Queries not working

1. **Check documents are uploaded**: Need at least one document
2. **Check backend logs**: Look for query processing errors
3. **Check browser console**: Look for API errors
4. **Verify API key**: Backend needs OpenAI API key

## Production Checklist

- [ ] Set `VITE_API_URL` to production backend URL
- [ ] Configure CORS on backend for production domain
- [ ] Test all API endpoints
- [ ] Verify error handling works correctly
- [ ] Test with multiple documents
- [ ] Test query responses
- [ ] Verify source citations work
- [ ] Test on mobile devices
- [ ] Check performance and loading states

## Next Steps

1. **Add Authentication** (if needed)
   - Add auth tokens to API requests
   - Handle 401/403 errors

2. **Add Caching** (optional)
   - Cache document list
   - Cache query responses

3. **Add Real-time Updates** (optional)
   - WebSocket for live updates
   - Server-sent events for progress

4. **Add Analytics** (optional)
   - Track user queries
   - Track document usage

## Support

For issues:
1. Check backend logs
2. Check browser console
3. Check network tab in DevTools
4. Verify environment variables
5. Check CORS configuration

