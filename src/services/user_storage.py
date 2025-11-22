"""
User storage layer for persisting user beliefs and query history.

Handles:
- Storing/retrieving user belief models
- Query history tracking
- TTL management
- Graceful fallback if storage unavailable
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import redis

from src.models.phase1_models import CounterfactualQuery, UserBeliefModel

logger = logging.getLogger(__name__)


class UserStorage:
    """
    Manages user belief persistence using Redis.

    Key structure:
    - user:beliefs:{user_id} → UserBeliefModel (JSON)
    - user:queries:{user_id} → Sorted set of query IDs (by timestamp)
    - user:responses:{user_id} → List of responses

    TTL:
    - beliefs: 24 hours (extend on activity)
    - queries: 7 days
    - responses: 30 days
    """

    def __init__(self) -> None:
        """Initialize storage with Redis connection pool."""
        self.redis_enabled = True
        try:
            # Create connection pool for better performance
            pool = redis.ConnectionPool(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                max_connections=int(os.getenv("REDIS_POOL_SIZE", 20)),
                socket_connect_timeout=5,
                socket_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,  # Check connection health every 30s
                decode_responses=True,
            )

            self.redis_client = redis.Redis(connection_pool=pool)

            # Test connection
            self.redis_client.ping()
            logger.info(
                "redis_connected",
                extra={"pool_size": pool.max_connections}
            )
        except Exception as e:
            logger.warning(
                "redis_connection_failed",
                extra={"error": str(e)},
            )
            self.redis_enabled = False
            self.fallback_storage: Dict[str, any] = {}

    def store_beliefs(
        self,
        user_id: str,
        beliefs: UserBeliefModel,
        ttl_hours: int = 24,
    ) -> None:
        """
        Store user beliefs.

        Args:
            user_id: User identifier
            beliefs: Belief model to store
            ttl_hours: Time to live in hours
        """
        key = f"user:beliefs:{user_id}"

        try:
            if self.redis_enabled:
                # Serialize to JSON
                beliefs_json = beliefs.model_dump_json()

                # Store with TTL
                self.redis_client.setex(
                    key,
                    ttl_hours * 3600,
                    beliefs_json,
                )

                logger.info(
                    "beliefs_stored",
                    extra={
                        "user_id": self._hash_user_id(user_id),
                        "ttl_hours": ttl_hours,
                    },
                )
            else:
                # Fallback to in-memory storage
                self.fallback_storage[key] = beliefs
                logger.info("beliefs_stored_fallback", extra={"user_id": self._hash_user_id(user_id)})

        except Exception as e:
            logger.error(
                "beliefs_storage_error",
                extra={"error": str(e)},
                exc_info=True,
            )

    def get_beliefs(self, user_id: str) -> Optional[UserBeliefModel]:
        """
        Retrieve user beliefs.

        Args:
            user_id: User identifier

        Returns:
            UserBeliefModel if found, None otherwise
        """
        key = f"user:beliefs:{user_id}"

        try:
            if self.redis_enabled:
                beliefs_json = self.redis_client.get(key)

                if beliefs_json:
                    # Extend TTL on access
                    self.redis_client.expire(key, 24 * 3600)

                    # Deserialize
                    beliefs = UserBeliefModel.model_validate_json(beliefs_json)

                    logger.info(
                        "beliefs_retrieved",
                        extra={"user_id": self._hash_user_id(user_id)},
                    )

                    return beliefs
                else:
                    logger.info(
                        "beliefs_not_found",
                        extra={"user_id": self._hash_user_id(user_id)},
                    )
                    return None
            else:
                # Fallback storage
                beliefs = self.fallback_storage.get(key)
                if beliefs:
                    logger.info("beliefs_retrieved_fallback", extra={"user_id": self._hash_user_id(user_id)})
                return beliefs

        except Exception as e:
            logger.error(
                "beliefs_retrieval_error",
                extra={"error": str(e)},
                exc_info=True,
            )
            return None

    def add_query_to_history(
        self,
        user_id: str,
        query: CounterfactualQuery,
        response: Optional[str] = None,
    ) -> None:
        """
        Add query to user's query history.

        Args:
            user_id: User identifier
            query: Query that was presented
            response: User's response (if available)
        """
        queries_key = f"user:queries:{user_id}"
        responses_key = f"user:responses:{user_id}"

        try:
            if self.redis_enabled:
                # Add to sorted set with timestamp as score
                timestamp = datetime.utcnow().timestamp()
                self.redis_client.zadd(
                    queries_key,
                    {query.id: timestamp},
                )

                # Set TTL
                self.redis_client.expire(queries_key, 7 * 24 * 3600)  # 7 days

                # If response provided, store it
                if response:
                    response_data = {
                        "query_id": query.id,
                        "response": response,
                        "timestamp": timestamp,
                    }
                    self.redis_client.rpush(
                        responses_key,
                        json.dumps(response_data),
                    )
                    self.redis_client.expire(responses_key, 30 * 24 * 3600)  # 30 days

                logger.info(
                    "query_added_to_history",
                    extra={
                        "user_id": self._hash_user_id(user_id),
                        "query_id": query.id,
                    },
                )
            else:
                # Fallback - just log
                logger.info("query_added_to_history_fallback", extra={"user_id": self._hash_user_id(user_id)})

        except Exception as e:
            logger.error(
                "query_history_error",
                extra={"error": str(e)},
                exc_info=True,
            )

    def get_query_count(self, user_id: str) -> int:
        """
        Get number of queries user has answered.

        Args:
            user_id: User identifier

        Returns:
            Number of queries
        """
        key = f"user:queries:{user_id}"

        try:
            if self.redis_enabled:
                count = self.redis_client.zcard(key)
                return count if count else 0
            else:
                return 0

        except Exception as e:
            logger.error(
                "query_count_error",
                extra={"error": str(e)},
                exc_info=True,
            )
            return 0

    def get_query_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[str]:
        """
        Get user's recent query IDs.

        Args:
            user_id: User identifier
            limit: Maximum number to return

        Returns:
            List of query IDs (most recent first)
        """
        key = f"user:queries:{user_id}"

        try:
            if self.redis_enabled:
                # Get most recent queries (highest scores first)
                query_ids = self.redis_client.zrevrange(key, 0, limit - 1)
                return list(query_ids)
            else:
                return []

        except Exception as e:
            logger.error(
                "query_history_error",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []

    def delete_user_data(self, user_id: str) -> None:
        """
        Delete all data for a user.

        Args:
            user_id: User identifier
        """
        keys = [
            f"user:beliefs:{user_id}",
            f"user:queries:{user_id}",
            f"user:responses:{user_id}",
        ]

        try:
            if self.redis_enabled:
                self.redis_client.delete(*keys)
                logger.info(
                    "user_data_deleted",
                    extra={"user_id": self._hash_user_id(user_id)},
                )
            else:
                # Fallback
                for key in keys:
                    self.fallback_storage.pop(key, None)

        except Exception as e:
            logger.error(
                "user_data_deletion_error",
                extra={"error": str(e)},
                exc_info=True,
            )

    def _hash_user_id(self, user_id: str) -> str:
        """
        Hash user ID for privacy in logs.

        Args:
            user_id: User identifier

        Returns:
            Hashed user ID
        """
        import hashlib

        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
