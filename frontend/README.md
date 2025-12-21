# RAG Assistant Frontend

Production-ready React frontend for the RAG Assistant application, built with Vite, TypeScript, React, and shadcn-ui.

## Features

- **Modern Stack**: Vite + React + TypeScript
- **UI Components**: shadcn-ui component library
- **State Management**: React hooks with custom useRAG hook
- **API Integration**: Production-ready API client with error handling
- **Responsive Design**: Mobile-first design with sidebar drawer
- **Type Safety**: Full TypeScript support
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Prerequisites

- Node.js 18+ (recommended: use [nvm](https://github.com/nvm-sh/nvm))
- npm, yarn, or bun
- Backend server running (see backend README)

## Quick Start

### 1. Install Dependencies

```bash
npm install
# or
yarn install
# or
bun install
```

### 2. Configure Environment Variables

Create a `.env.local` file in the frontend directory:

```bash
# Copy the example file
cp .env.example .env.local
```

Edit `.env.local` and set your backend API URL:

```env
# For local development
VITE_API_URL=http://localhost:8000/api/v1

# For production (replace with your production backend URL)
# VITE_API_URL=https://your-backend-domain.com/api/v1
```

**Important**: The `.env.local` file is gitignored and should not be committed. Use `.env.example` as a template.

### 3. Start Development Server

```bash
npm run dev
# or
yarn dev
# or
bun dev
```

The application will be available at `http://localhost:8080` (or the port specified in `vite.config.ts`).

### 4. Build for Production

```bash
npm run build
# or
yarn build
# or
bun run build
```

The production build will be in the `dist` directory.

### 5. Preview Production Build

```bash
npm run preview
# or
yarn preview
# or
bun run preview
```

## Project Structure

```
src/
├── components/          # React components
│   ├── ui/             # shadcn-ui components
│   ├── Avatar.tsx
│   ├── ChatInput.tsx
│   ├── ChatInterface.tsx
│   ├── ChatMessage.tsx
│   ├── DocumentCard.tsx
│   ├── Sidebar.tsx
│   └── ...
├── hooks/              # Custom React hooks
│   ├── useRAG.ts       # Main RAG functionality hook
│   └── ...
├── lib/                # Utility libraries
│   ├── api.ts          # API client
│   └── utils.ts
├── pages/              # Page components
│   ├── Index.tsx
│   └── NotFound.tsx
├── types/              # TypeScript type definitions
│   └── index.ts
└── main.tsx            # Application entry point
```

## API Integration

The frontend communicates with the backend through a production-ready API client located in `src/lib/api.ts`.

### API Endpoints

- **Documents**
  - `GET /api/v1/documents` - List all documents
  - `POST /api/v1/documents/upload` - Upload PDF documents
  - `GET /api/v1/documents/{id}` - Get document details
  - `DELETE /api/v1/documents/{id}` - Delete document

- **Queries**
  - `POST /api/v1/queries` - Submit query to RAG system

### Error Handling

The API client includes comprehensive error handling:
- Network errors (connection failures)
- HTTP errors (4xx, 5xx responses)
- Custom error messages from backend
- User-friendly toast notifications

### Environment Variables

- `VITE_API_URL` - Backend API base URL (default: `http://localhost:8000/api/v1`)

**Note**: Vite requires the `VITE_` prefix for environment variables to be exposed to the client.

## Development

### Code Quality

```bash
# Lint code
npm run lint
# or
yarn lint
```

### Type Checking

TypeScript type checking is integrated with the build process. Run:

```bash
# Check types
npx tsc --noEmit
```

## Backend Integration

### Prerequisites

1. Ensure the backend server is running (see `../backend/README.md`)
2. Backend should be accessible at the URL specified in `VITE_API_URL`
3. CORS should be properly configured on the backend

### Testing the Connection

1. Start the backend server:
   ```bash
   cd ../backend
   make run
   # or
   uvicorn main:app --reload
   ```

2. Start the frontend:
   ```bash
   npm run dev
   ```

3. Open `http://localhost:8080` in your browser

4. Try uploading a PDF document and asking a question

## Production Deployment

### Build Configuration

The production build is optimized with:
- Code splitting
- Tree shaking
- Minification
- Asset optimization

### Environment Variables for Production

Set `VITE_API_URL` to your production backend URL:

```env
VITE_API_URL=https://api.yourdomain.com/api/v1
```

### Deployment Options

- **Vercel**: Connect your repository and deploy
- **Netlify**: Connect your repository and deploy
- **Docker**: Build a Docker image (see Dockerfile if available)
- **Static Hosting**: Upload the `dist` folder to any static hosting service

### CORS Configuration

Ensure your backend CORS settings allow requests from your frontend domain:

```python
# In backend configuration
CORS_ORIGINS = [
    "http://localhost:8080",  # Development
    "https://your-frontend-domain.com",  # Production
]
```

## Troubleshooting

### Backend Connection Issues

1. **Check if backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify API URL in `.env.local`**:
   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   ```

3. **Check browser console** for network errors

4. **Verify CORS settings** on the backend

### Build Issues

1. **Clear node_modules and reinstall**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Check Node.js version** (should be 18+):
   ```bash
   node --version
   ```

### Type Errors

1. **Regenerate types** if backend API changed
2. **Check `src/types/index.ts`** matches backend response structure
3. **Run type check**: `npx tsc --noEmit`

## License

MIT

## Support

For issues and questions, please refer to the main project repository.
