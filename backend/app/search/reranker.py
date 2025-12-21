"""Reranking Agent using Cohere API or Local CrossEncoder."""

from typing import List, Dict

from app.config import get_settings
from app.core.exceptions import RerankingError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Try to import Cohere
try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# Try to import CrossEncoder
try:
    from sentence_transformers import CrossEncoder

    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False


class RerankerAgent:
    """Rerank search results using Cohere API or local CrossEncoder."""

    def __init__(self, use_cohere: bool = True):
        self.use_cohere = (
            use_cohere
            and COHERE_AVAILABLE
            and settings.COHERE_API_KEY
        )

        if self.use_cohere:
            self.client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
            logger.info("using_cohere_reranker")
        elif CROSSENCODER_AVAILABLE:
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info("using_crossencoder_reranker")
        else:
            logger.warning("no_reranker_available")
            self.use_cohere = False
            self.cross_encoder = None

    def rerank(
        self, query: str, documents: List[Dict], top_k: int = None
    ) -> List[Dict]:
        """Rerank documents."""
        top_k = top_k or settings.TOP_K_RERANK

        if not documents:
            return []

        try:
            if self.use_cohere:
                return self._rerank_cohere(query, documents, top_k)
            elif self.cross_encoder:
                return self._rerank_crossencoder(query, documents, top_k)
            else:
                # No reranker available, return original order with preserved scores
                # Ensure each document has a score field
                return [
                    {**doc, "score": doc.get("score") or doc.get("fusion_score") or 0.0}
                    for doc in documents[:top_k]
                ]
        except Exception as e:
            logger.error("reranking_failed", error=str(e))
            raise RerankingError(f"Reranking failed: {str(e)}")

    def _rerank_cohere(
        self, query: str, documents: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank using Cohere API."""
        try:
            doc_texts = [doc["content"] for doc in documents]

            response = self.client.rerank(
                model=settings.RERANK_MODEL,
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )

            return [
                {
                    **documents[result.index],
                    "rerank_score": result.relevance_score,
                }
                for result in response.results
            ]
        except Exception as e:
            logger.error("cohere_rerank_failed", error=str(e))
            raise

    def _rerank_crossencoder(
        self, query: str, documents: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank using local CrossEncoder."""
        try:
            pairs = [[query, doc["content"]] for doc in documents]
            scores = self.cross_encoder.predict(pairs)

            indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {**documents[idx], "rerank_score": score}
                for idx, score in indexed_scores[:top_k]
            ]
        except Exception as e:
            logger.error("crossencoder_rerank_failed", error=str(e))
            raise

