/**
 * Production-ready API client for RAG Assistant
 * Handles all backend communication with proper error handling and type safety
 */

import type { Document, QueryResponse } from '@/types';

// Get API base URL from environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Custom error class for API errors
export class APIError extends Error {
    constructor(
        message: string,
        public statusCode?: number,
        public response?: any
    ) {
        super(message);
        this.name = 'APIError';
    }
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;

    const config: RequestInit = {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    };

    try {
        const response = await fetch(url, config);

        // Handle non-JSON responses (like 204 No Content)
        if (response.status === 204) {
            return {} as T;
        }

        const data = await response.json();

        if (!response.ok) {
            throw new APIError(
                data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`,
                response.status,
                data
            );
        }

        return data;
    } catch (error) {
        if (error instanceof APIError) {
            throw error;
        }

        // Network errors or other fetch errors
        if (error instanceof TypeError && error.message.includes('fetch')) {
            throw new APIError(
                'Network error: Unable to connect to the server. Please check if the backend is running.',
                0,
                error
            );
        }

        throw new APIError(
            error instanceof Error ? error.message : 'An unexpected error occurred',
            0,
            error
        );
    }
}

/**
 * Documents API
 */
export const documentsApi = {
    /**
     * Upload documents (PDFs and images)
     */
    async upload(files: File[]): Promise<Document[]> {
        const formData = new FormData();
        files.forEach((file) => {
            // Accept both PDFs and images
            const fileName = file.name.toLowerCase();
            const isPDF = fileName.endsWith('.pdf');
            const isImage = fileName.match(/\.(png|jpg|jpeg|gif|webp|bmp|svg|heic|heif)$/);
            
            if (!isPDF && !isImage) {
                throw new APIError(`File ${file.name} is not a supported file type. Please upload PDF or image files.`, 400);
            }
            formData.append('files', file);
        });

        const response = await fetch(`${API_BASE_URL}/documents/upload`, {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - browser will set it with boundary
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new APIError(
                error.detail || `Failed to upload documents: ${response.statusText}`,
                response.status,
                error
            );
        }

        const data = await response.json();
        return data.map((doc: any) => ({
            id: doc.id,
            name: doc.filename,
            size: doc.file_size,
            chunks: doc.total_chunks,
            extractionMethod: doc.extraction_method,
            isScanned: doc.is_scanned,
            uploadedAt: new Date(), // Backend doesn't return timestamp, use current time
        }));
    },

    /**
     * List all documents
     */
    async list(): Promise<Document[]> {
        const data = await fetchAPI<any[]>('/documents');
        return data.map((doc) => ({
            id: doc.id,
            name: doc.filename,
            size: doc.file_size,
            chunks: doc.total_chunks,
            extractionMethod: doc.extraction_method,
            isScanned: doc.is_scanned,
            uploadedAt: new Date(), // Backend doesn't return timestamp
        }));
    },

    /**
     * Get document by ID
     */
    async get(id: string): Promise<Document> {
        const doc = await fetchAPI<any>(`/documents/${id}`);
        return {
            id: doc.id,
            name: doc.filename,
            size: doc.file_size,
            chunks: doc.total_chunks,
            extractionMethod: doc.extraction_method,
            isScanned: doc.is_scanned,
            uploadedAt: new Date(),
        };
    },

    /**
     * Delete document
     */
    async delete(id: string): Promise<void> {
        await fetchAPI(`/documents/${id}`, {
            method: 'DELETE',
        });
    },
};

/**
 * Queries API
 */
export const queriesApi = {
    /**
     * Submit a query to the RAG system
     */
    async submit(
        query: string, 
        conversationHistory?: Array<{role: string, content: string}>,
        images?: string[]
    ): Promise<QueryResponse> {
        const payload: any = { query };
        if (conversationHistory && conversationHistory.length > 0) {
            payload.conversation_history = conversationHistory;
        }
        if (images && images.length > 0) {
            payload.images = images;
        }
        const data = await fetchAPI<QueryResponse>('/queries', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
        return data;
    },
};

/**
 * Health check API
 */
export const healthApi = {
    /**
     * Check if backend is healthy
     */
    async check(): Promise<{ status: string }> {
        const baseUrl = API_BASE_URL.replace('/api/v1', '');
        const response = await fetch(`${baseUrl}/health`);
        if (!response.ok) {
            throw new APIError('Backend health check failed', response.status);
        }
        return response.json();
    },
};

