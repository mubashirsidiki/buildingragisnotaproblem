"""
RabbitMQ service for async chat message storage.
"""

import json
import aio_pika
from typing import Dict, Any
from config import CONFIG
from core.logger.logger import LOG

class RabbitMQService:
    """Manages async message publishing to RabbitMQ."""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue_name = "chat_messages"
    
    async def connect(self):
        """Connect to RabbitMQ and setup queue."""
        try:
            self.connection = await aio_pika.connect_robust(CONFIG.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # Declare queue with durability
            self.queue = await self.channel.declare_queue(
                self.queue_name,
                durable=True
            )
            
            LOG.info("✅ Connected to RabbitMQ")
        except Exception as e:
            LOG.error(f"❌ RabbitMQ connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from RabbitMQ."""
        if self.connection:
            await self.connection.close()
    
    async def publish_message(self, user_id: str, role: str, content: str):
        """Publish chat message to queue for DB storage."""
        try:
            message_data = {
                "user_id": user_id,
                "role": role,
                "content": content
            }
            
            message = aio_pika.Message(
                json.dumps(message_data).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )
            
            await self.channel.default_exchange.publish(
                message,
                routing_key=self.queue_name
            )
            
            LOG.debug(f"📤 Published message to queue: {role} message for user {user_id}")
        except Exception as e:
            LOG.error(f"❌ Failed to publish message: {e}")

# Global RabbitMQ service instance
rabbitmq_service = RabbitMQService()
