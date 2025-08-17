"""
Redis service for chat conversation context management.
"""

import json
import redis.asyncio as redis
from typing import List, Dict, Any, Optional
from config import CONFIG
from core.logger.logger import LOG

class RedisService:
    """Manages chat conversation context in Redis."""
    
    def __init__(self):
        self.redis_client = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(CONFIG.redis_url)
            await self.redis_client.ping()
            LOG.info("✅ Connected to Redis")
        except Exception as e:
            LOG.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
    
    def _messages_key(self, user_id: str) -> str:
        """Get Redis key for user messages."""
        return f"chat:{user_id}:messages"
    
    def _history_loaded_key(self, user_id: str) -> str:
        """Get Redis key for history loaded flag."""
        return f"chat:{user_id}:history_loaded"
    
    async def is_history_loaded(self, user_id: str) -> bool:
        """Check if chat history is loaded for user."""
        key = self._history_loaded_key(user_id)
        return await self.redis_client.exists(key) > 0
    
    async def set_history_loaded(self, user_id: str):
        """Mark chat history as loaded for user."""
        key = self._history_loaded_key(user_id)
        await self.redis_client.set(key, "1", ex=CONFIG.redis_ttl)
    
    async def get_messages(self, user_id: str) -> List[Dict[str, str]]:
        """Get all messages for a user from Redis."""
        key = self._messages_key(user_id)
        messages_data = await self.redis_client.lrange(key, 0, -1)
        return [json.loads(msg) for msg in messages_data]
    
    async def add_message(self, user_id: str, role: str, content: str):
        """Add a message to Redis conversation context."""
        key = self._messages_key(user_id)
        message = {"role": role, "content": content}
        await self.redis_client.lpush(key, json.dumps(message))
        await self.redis_client.expire(key, CONFIG.redis_ttl)
    
    async def load_history_from_db(self, user_id: str, db_messages: List[Dict[str, Any]]):
        """Load conversation history from DB into Redis."""
        key = self._messages_key(user_id)
        
        # Clear existing messages
        await self.redis_client.delete(key)
        
        # Add messages in reverse order (Redis LPUSH adds to head)
        for message in reversed(db_messages):
            msg_data = {"role": message["role"], "content": message["content"]}
            await self.redis_client.lpush(key, json.dumps(msg_data))
        
        # Set TTL and mark history as loaded
        await self.redis_client.expire(key, CONFIG.redis_ttl)
        await self.set_history_loaded(user_id)
        
        LOG.info(f"🔄 Loaded {len(db_messages)} messages from DB to Redis for user {user_id}")
    
    async def clear_user_context(self, user_id: str):
        """Clear all Redis data for a user."""
        keys = [self._messages_key(user_id), self._history_loaded_key(user_id)]
        await self.redis_client.delete(*keys)

# Global Redis service instance
redis_service = RedisService()
