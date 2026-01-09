"""Redis Caching Support with Semantic Caching (from roadmap line 60)."""

import hashlib
import json
from typing import Any, Dict, List, Optional

import numpy as np
from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Try to import Redis
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis_not_available", message="Install redis package for caching")

# Try to import OpenAI embeddings for semantic caching
try:
    from langchain_openai import OpenAIEmbeddings

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("embeddings_not_available", message="Install langchain-openai for semantic caching")


class CacheManager:
    """Redis cache manager for query results with semantic caching support."""

    def __init__(self):
        self.enabled = settings.CACHE_ENABLED and REDIS_AVAILABLE
        self.client: Optional[Any] = None
        self.ttl = settings.CACHE_TTL
        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0
        self.embeddings: Optional[Any] = None

        if self.enabled:
            try:
                if settings.REDIS_URL:
                    self.client = redis.from_url(settings.REDIS_URL)
                else:
                    self.client = redis.Redis(host="localhost", port=6379, db=0)

                # Test connection
                self.client.ping()
                logger.info("redis_cache_initialized", ttl=self.ttl)
            except Exception as e:
                logger.warning("redis_connection_failed", error=str(e))
                self.enabled = False
                self.client = None

        # Initialize embeddings for semantic caching
        if (
            settings.SEMANTIC_CACHE_ENABLED
            and EMBEDDINGS_AVAILABLE
            and settings.OPENAI_API_KEY
        ):
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY,
                    dimensions=settings.EMBEDDING_DIMENSIONS,
                )
                logger.info("semantic_cache_embeddings_initialized")
            except Exception as e:
                logger.warning("semantic_cache_embeddings_failed", error=str(e))
                self.embeddings = None
        else:
            logger.info("semantic_caching_disabled")

        if not self.enabled:
            logger.info("caching_disabled")

    def _generate_key(self, query: str, **kwargs: Any) -> str:
        """Generate cache key from query and parameters."""
        key_data = {"query": query, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, query: str, **kwargs: Any) -> Optional[Any]:
        """Get cached result (exact match)."""
        if not self.enabled or not self.client:
            return None

        try:
            key = self._generate_key(query, **kwargs)
            cached = self.client.get(key)

            if cached:
                self.hits += 1
                logger.debug("cache_hit", query=query[:50])
                return json.loads(cached)

            self.misses += 1
            logger.debug("cache_miss", query=query[:50])
            return None

        except Exception as e:
            logger.error("cache_get_failed", error=str(e))
            return None

    def get_semantic(
        self, query: str, threshold: float = None, **kwargs: Any
    ) -> Optional[Any]:
        """Get cached result using semantic similarity."""
        if not self.enabled or not self.client:
            return None

        if not settings.SEMANTIC_CACHE_ENABLED or not self.embeddings:
            # Fallback to exact match
            return self.get(query, **kwargs)

        threshold = threshold or settings.SEMANTIC_CACHE_THRESHOLD

        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)

            # Get all cache keys (in production, use a separate index for this)
            # For now, we'll scan a limited set of keys
            # In production, maintain a separate Redis set with query embeddings
            pattern = f"cache:*"
            keys = self.client.keys(pattern)[:1000]  # Limit scan for performance

            best_match = None
            best_similarity = 0.0

            for key in keys:
                try:
                    # Try to get cached query from metadata
                    metadata_key = f"{key}:meta"
                    metadata = self.client.get(metadata_key)

                    if metadata:
                        meta_data = json.loads(metadata)
                        cached_query = meta_data.get("query", "")
                        cached_embedding = meta_data.get("embedding")

                        if cached_embedding:
                            # Ensure cached_embedding is a numpy array for calculation
                            if isinstance(cached_embedding, list):
                                cached_embedding = np.array(cached_embedding)
                            elif not isinstance(cached_embedding, np.ndarray):
                                cached_embedding = np.array(cached_embedding)
                            
                            # Ensure query_embedding is also a numpy array
                            if isinstance(query_embedding, list):
                                query_embedding = np.array(query_embedding)
                            elif not isinstance(query_embedding, np.ndarray):
                                query_embedding = np.array(query_embedding)
                            
                            # Calculate cosine similarity
                            similarity = np.dot(
                                query_embedding, cached_embedding
                            ) / (
                                np.linalg.norm(query_embedding)
                                * np.linalg.norm(cached_embedding)
                            )

                            if similarity >= threshold and similarity > best_similarity:
                                best_similarity = similarity
                                best_match = key
                except Exception as e:
                    logger.debug("semantic_cache_key_check_failed", key=key, error=str(e))
                    continue

            if best_match:
                self.semantic_hits += 1
                cached = self.client.get(best_match)
                if cached:
                    logger.info(
                        "semantic_cache_hit",
                        query=query[:50],
                        similarity=f"{best_similarity:.3f}",
                    )
                    return json.loads(cached)

            self.misses += 1
            logger.debug("semantic_cache_miss", query=query[:50])
            return None

        except Exception as e:
            logger.error("semantic_cache_get_failed", error=str(e))
            # Fallback to exact match
            return self.get(query, **kwargs)

    def set(
        self,
        query: str,
        value: Any,
        ttl: Optional[int] = None,
        enable_semantic: bool = True,
        **kwargs: Any,
    ):
        """Set cache value with optional semantic indexing."""
        if not self.enabled or not self.client:
            return

        try:
            key = self._generate_key(query, **kwargs)
            ttl = ttl or self.ttl
            self.client.setex(
                key, ttl, json.dumps(value, default=str)
            )  # default=str for non-serializable types
            logger.debug("cache_set", query=query[:50], ttl=ttl)

            # Store semantic metadata if enabled
            if (
                enable_semantic
                and settings.SEMANTIC_CACHE_ENABLED
                and self.embeddings
            ):
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    # Convert to list if it's a numpy array, otherwise use as-is
                    if hasattr(query_embedding, 'tolist'):
                        embedding_list = query_embedding.tolist()
                    elif isinstance(query_embedding, list):
                        embedding_list = query_embedding
                    else:
                        embedding_list = list(query_embedding)
                    
                    metadata = {
                        "query": query,
                        "embedding": embedding_list,
                    }
                    metadata_key = f"{key}:meta"
                    self.client.setex(
                        metadata_key, ttl, json.dumps(metadata, default=str)
                    )
                    logger.debug("semantic_cache_metadata_stored", query=query[:50])
                except Exception as e:
                    logger.warning("semantic_cache_metadata_failed", error=str(e))

        except Exception as e:
            logger.error("cache_set_failed", error=str(e))

    def delete(self, query: str, **kwargs: Any):
        """Delete cache entry."""
        if not self.enabled or not self.client:
            return

        try:
            key = self._generate_key(query, **kwargs)
            self.client.delete(key)
            logger.debug("cache_deleted", query=query[:50])

        except Exception as e:
            logger.error("cache_delete_failed", error=str(e))

    def clear(self):
        """Clear all cache entries."""
        if not self.enabled or not self.client:
            return

        try:
            self.client.flushdb()
            logger.info("cache_cleared")

        except Exception as e:
            logger.error("cache_clear_failed", error=str(e))

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses + self.semantic_hits
        exact_hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        semantic_hit_rate = (self.semantic_hits / total * 100) if total > 0 else 0.0
        overall_hit_rate = ((self.hits + self.semantic_hits) / total * 100) if total > 0 else 0.0

        return {
            "enabled": self.enabled,
            "semantic_enabled": settings.SEMANTIC_CACHE_ENABLED and self.embeddings is not None,
            "exact_hits": self.hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "exact_hit_rate": f"{exact_hit_rate:.2f}%",
            "semantic_hit_rate": f"{semantic_hit_rate:.2f}%",
            "overall_hit_rate": f"{overall_hit_rate:.2f}%",
            "total_requests": total,
        }


# Global cache manager instance
cache_manager = CacheManager()

