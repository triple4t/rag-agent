/**
 * Custom hook for RAG functionality
 * Handles document management and query submission with proper error handling
 */
import { useState, useCallback, useEffect } from 'react';
import type { Document, Message, QueryResponse, Source } from '@/types';
import { toast } from 'sonner';
import { documentsApi, queriesApi, APIError } from '@/lib/api';

export const useRAG = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetchingDocuments, setIsFetchingDocuments] = useState(false);

  /**
   * Fetch all documents from the backend
   */
  const fetchDocuments = useCallback(async () => {
    setIsFetchingDocuments(true);
    try {
      const docs = await documentsApi.list();
      setDocuments(docs);
    } catch (error) {
      console.error('Error fetching documents:', error);
      if (error instanceof APIError) {
        if (error.statusCode !== 0) {
          // Only show error if it's not a network error (network errors are handled elsewhere)
          toast.error(`Failed to fetch documents: ${error.message}`);
        }
      }
      // Don't clear documents on error - keep existing state
    } finally {
      setIsFetchingDocuments(false);
    }
  }, []);

  /**
   * Upload PDF documents to the backend
   */
  const uploadDocuments = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    setIsUploading(true);
    try {
      const newDocs = await documentsApi.upload(files);
      setDocuments((prev) => [...prev, ...newDocs]);
      toast.success(
        `${files.length} document${files.length > 1 ? 's' : ''} uploaded successfully`
      );
    } catch (error) {
      console.error('Error uploading documents:', error);
      if (error instanceof APIError) {
        toast.error(`Upload failed: ${error.message}`);
      } else {
        toast.error('An unexpected error occurred while uploading documents');
      }
      throw error; // Re-throw to let component handle it
    } finally {
      setIsUploading(false);
    }
  }, []);

  /**
   * Delete a document from the backend
   */
  const deleteDocument = useCallback(async (id: string) => {
    try {
      await documentsApi.delete(id);
      setDocuments((prev) => prev.filter((doc) => doc.id !== id));
      toast.success('Document deleted successfully');
    } catch (error) {
      console.error('Error deleting document:', error);
      if (error instanceof APIError) {
        toast.error(`Failed to delete document: ${error.message}`);
      } else {
        toast.error('An unexpected error occurred while deleting the document');
      }
      throw error;
    }
  }, []);

  /**
   * Send a query to the RAG system
   */
  const sendQuery = useCallback(async (query: string) => {
    if (!query.trim()) return;

    // Add user message immediately
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response: QueryResponse = await queriesApi.submit(query.trim());

      // Transform backend sources to frontend format
      const sources: Source[] = response.sources.map((source) => ({
        doc_id: source.doc_id,
        filename: source.filename,
        chunk_idx: source.chunk_idx,
        content: source.content,
        score: source.score,
      }));

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.answer || 'No answer generated',
        timestamp: new Date(),
        sources,
        qualityScore: response.quality_score,
        latency: response.latency,
        cost: response.cost,
        error: response.error || undefined,
        queryId: response.query_id,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Show warning if quality score is low
      if (response.quality_score < 0.75) {
        toast.warning('Response quality may be lower than expected');
      }
    } catch (error) {
      console.error('Error sending query:', error);
      
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: error instanceof APIError
          ? `Error: ${error.message}`
          : 'An unexpected error occurred while processing your query. Please try again.',
        timestamp: new Date(),
        error: error instanceof APIError ? error.message : 'Unknown error',
      };

      setMessages((prev) => [...prev, errorMessage]);

      if (error instanceof APIError) {
        if (error.statusCode === 0) {
          // Network error
          toast.error('Unable to connect to the server. Please check if the backend is running.');
        } else {
          toast.error(`Query failed: ${error.message}`);
        }
      } else {
        toast.error('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return {
    documents,
    messages,
    isUploading,
    isLoading,
    isFetchingDocuments,
    fetchDocuments,
    uploadDocuments,
    deleteDocument,
    sendQuery,
  };
};
