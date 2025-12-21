/**
 * Document interface matching backend response
 */
export interface Document {
  id: string;
  name: string;
  size: number;
  chunks: number;
  extractionMethod?: string;
  isScanned?: boolean;
  uploadedAt: Date;
}

/**
 * Chat message interface
 */
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
  qualityScore?: number;
  latency?: number;
  cost?: number;
  error?: string;
  queryId?: string;
}

/**
 * Source document interface matching backend response
 */
export interface Source {
  doc_id: string;
  filename?: string;
  chunk_idx: number;
  content: string;
  score: number;
}

/**
 * Query response interface matching backend API
 */
export interface QueryResponse {
  query_id: string;
  query: string;
  answer: string;
  quality_score: number;
  reasoning: string;
  sources: Source[];
  latency: number;
  cost: number;
  error?: string | null;
}
