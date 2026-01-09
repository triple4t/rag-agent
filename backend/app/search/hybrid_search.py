"""Production Hybrid Search Engine with OpenAI Embeddings and ChromaDB (LangChain Integration)."""

import logging
from typing import List, Dict, Optional
import os

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from rank_bm25 import BM25Okapi
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.exceptions import SearchError
from app.core.logging import get_logger
from app.utils.document_loader import Document

logger = get_logger(__name__)
settings = get_settings()


class HybridSearchEngine:
    """Production-grade hybrid search engine with ChromaDB using LangChain integration."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.collection_name = settings.QDRANT_COLLECTION_NAME  # Reusing config name

        # Initialize OpenAI embeddings using LangChain
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )

        logger.info(
            "initializing_embeddings",
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )

        # Initialize ChromaDB using LangChain integration (per docs)
        try:
            # Use persistent storage in .chroma directory
            chroma_path = os.path.join(os.getcwd(), ".chroma")
            os.makedirs(chroma_path, exist_ok=True)
            
            # Create Chroma vector store with persistence (per LangChain docs)
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=chroma_path,
            )
            
            logger.info("chromadb_initialized", path=chroma_path, collection=self.collection_name)
        except Exception as e:
            logger.error("chromadb_initialization_failed", error=str(e))
            raise SearchError(f"ChromaDB initialization failed: {str(e)}")

        # Initialize BM25 for keyword search
        self.corpus = []
        self.chunk_map = []
        self.bm25 = None

        self._setup_vector_db()
        self._setup_keyword_search()

    def _setup_vector_db(self):
        """Initialize ChromaDB vector database using LangChain."""
        # Convert our Document format to LangChain Document format
        langchain_docs = []
        doc_ids = []

        for doc in self.documents:
            chunk_pages = doc.metadata.get("chunk_pages", [])
            for i, chunk in enumerate(doc.chunks):
                chunk_id = f"{doc.id}_{i}"
                doc_ids.append(chunk_id)
                
                # Get page number for this chunk
                page_number = chunk_pages[i] if i < len(chunk_pages) else None
                
                # Filter metadata to only include simple types that ChromaDB accepts
                # ChromaDB only accepts: str, int, float, bool, None (not lists, dicts, etc.)
                filtered_metadata = {}
                for key, value in doc.metadata.items():
                    # Skip complex types (lists, dicts) - ChromaDB doesn't support them
                    if key == "chunk_pages":  # Skip the list itself
                        continue
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                
                # Create LangChain Document with metadata
                langchain_docs.append(
                    LangChainDocument(
                        page_content=chunk,
                        metadata={
                            "doc_id": doc.id,
                            "chunk_idx": i,
                            "page_number": page_number,  # int or None - ChromaDB compatible
                            **filtered_metadata,
                        }
                    )
                )

        if not langchain_docs:
            logger.warning("no_chunks_to_index")
            return

        logger.info("adding_documents_to_chromadb", total_chunks=len(langchain_docs))

        try:
            # Clear existing collection if re-indexing (delete all documents)
            try:
                # Get all existing IDs and delete them
                existing_ids = self.vector_store.get(include=[])['ids']
                if existing_ids:
                    self.vector_store.delete(ids=existing_ids)
                    logger.debug("cleared_existing_collection", deleted_count=len(existing_ids))
            except Exception as e:
                logger.debug("no_existing_collection_to_clear", error=str(e))
            
            # Add documents using LangChain API (per docs)
            # Persistence is automatic when persist_directory is set
            self.vector_store.add_documents(documents=langchain_docs, ids=doc_ids)
            
            logger.info("vector_db_initialized", total_points=len(langchain_docs))
        except Exception as e:
            logger.error("chromadb_add_failed", error=str(e), exc_info=True)
            raise SearchError(f"Failed to add documents to ChromaDB: {str(e)}")

    def _setup_keyword_search(self):
        """Initialize BM25 keyword search."""
        for doc in self.documents:
            for i, chunk in enumerate(doc.chunks):
                self.corpus.append(chunk)
                self.chunk_map.append({"doc_id": doc.id, "chunk_idx": i})

        if not self.corpus:
            logger.warning("no_corpus_for_bm25")
            return

        tokenized_corpus = [chunk.split() for chunk in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("bm25_initialized", total_chunks=len(self.corpus))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def vector_search(self, query: str, k: int = None) -> List[Dict]:
        """Vector similarity search using ChromaDB with LangChain (per docs)."""
        k = k or settings.TOP_K_SEARCH

        try:
            # Use LangChain's similarity_search_with_score (per docs)
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
            )

            # Format results
            hits = []
            for doc, score in results:
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                metadata = doc.metadata
                hits.append({
                    "score": similarity_score,
                    "content": doc.page_content,
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_idx": metadata.get("chunk_idx", 0),
                    "metadata": {k: v for k, v in metadata.items() if k not in ["doc_id", "chunk_idx"]},
                    "search_type": "vector",
                })

            return hits

        except Exception as e:
            logger.error("vector_search_failed", error=str(e), exc_info=True)
            raise SearchError(f"Vector search failed: {str(e)}")

    def keyword_search(self, query: str, k: int = None) -> List[Dict]:
        """BM25 keyword search."""
        k = k or settings.TOP_K_SEARCH

        if not self.bm25:
            return []

        try:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk_info = self.chunk_map[idx]
                    doc = next(
                        (
                            d
                            for d in self.documents
                            if d.id == chunk_info["doc_id"]
                        ),
                        None,
                    )
                    doc_metadata = doc.metadata if doc else {}
                    
                    # Get page number for this chunk
                    chunk_pages = doc_metadata.get("chunk_pages", [])
                    page_number = (
                        chunk_pages[chunk_info["chunk_idx"]]
                        if chunk_info["chunk_idx"] < len(chunk_pages)
                        else None
                    )
                    
                    # Create metadata with page_number
                    chunk_metadata = {**doc_metadata, "page_number": page_number}
                    
                    results.append(
                        {
                            "score": float(scores[idx]),
                            "content": self.corpus[idx],
                            "doc_id": chunk_info["doc_id"],
                            "chunk_idx": chunk_info["chunk_idx"],
                            "metadata": chunk_metadata,
                            "search_type": "keyword",
                        }
                    )

            return results

        except Exception as e:
            logger.error("keyword_search_failed", error=str(e))
            raise SearchError(f"Keyword search failed: {str(e)}")

    def hybrid_search(
        self,
        query: str,
        k: int = None,
        vector_weight: float = None,
        keyword_weight: float = None,
    ) -> List[Dict]:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        k = k or settings.TOP_K_SEARCH
        vector_weight = vector_weight or settings.VECTOR_WEIGHT
        keyword_weight = keyword_weight or settings.KEYWORD_WEIGHT
        rrf_k = settings.RRF_K

        # Get results from both methods
        vector_results = self.vector_search(query, k=k)
        keyword_results = self.keyword_search(query, k=k)

        # RRF fusion
        fusion_scores = {}

        for rank, result in enumerate(vector_results):
            key = f"{result['doc_id']}_{result['chunk_idx']}"
            fusion_scores[key] = fusion_scores.get(key, 0) + vector_weight / (
                rank + rrf_k
            )

        for rank, result in enumerate(keyword_results):
            key = f"{result['doc_id']}_{result['chunk_idx']}"
            fusion_scores[key] = fusion_scores.get(key, 0) + keyword_weight / (
                rank + rrf_k
            )

        # Combine results
        all_results_map = {}
        for result in vector_results + keyword_results:
            key = f"{result['doc_id']}_{result['chunk_idx']}"
            if key not in all_results_map:
                all_results_map[key] = result

        # Sort by fusion score
        sorted_results = sorted(
            [(fusion_scores[key], all_results_map[key]) for key in fusion_scores],
            key=lambda x: x[0],
            reverse=True,
        )

        # Normalize fusion scores to 0-1 range for better readability
        # This makes scores more interpretable (0.0 = worst, 1.0 = best)
        results = []
        if sorted_results:
            max_fusion_score = sorted_results[0][0]  # Highest score (first after sorting)
            if max_fusion_score > 0:
                for original_fusion_score, result in sorted_results[:k]:
                    normalized_score = original_fusion_score / max_fusion_score
                    results.append({
                        **result,
                        "score": normalized_score,
                        "fusion_score": original_fusion_score,  # Keep original for debugging
                    })
            else:
                # All scores are 0, use original scores
                for original_fusion_score, result in sorted_results[:k]:
                    results.append({
                        **result,
                        "score": original_fusion_score,
                        "fusion_score": original_fusion_score,
                    })
        
        return results
