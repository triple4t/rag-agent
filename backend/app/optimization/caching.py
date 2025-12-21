"""Redis Caching Support (from roadmap line 60)."""

import hashlib
import json
from typing import Any, Optional

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


class CacheManager:
    """Redis cache manager for query results."""

    def __init__(self):
        self.enabled = settings.CACHE_ENABLED and REDIS_AVAILABLE
        self.client: Optional[Any] = None
        self.ttl = settings.CACHE_TTL
        self.hits = 0
        self.misses = 0

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
        else:
            logger.info("caching_disabled")

    def _generate_key(self, query: str, **kwargs: Any) -> str:
        """Generate cache key from query and parameters."""
        key_data = {"query": query, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, query: str, **kwargs: Any) -> Optional[Any]:
        """Get cached result."""
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

    def set(self, query: str, value: Any, ttl: Optional[int] = None, **kwargs: Any):
        """Set cache value."""
        if not self.enabled or not self.client:
            return

        try:
            key = self._generate_key(query, **kwargs)
            ttl = ttl or self.ttl
            self.client.setex(
                key, ttl, json.dumps(value, default=str)
            )  # default=str for non-serializable types
            logger.debug("cache_set", query=query[:50], ttl=ttl)

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
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0

        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total,
        }


# Global cache manager instance
cache_manager = CacheManager()

