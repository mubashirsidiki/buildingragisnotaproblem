"""
RabbitMQ worker for async chat message storage to database.
"""

import asyncio
import json
import aio_pika
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import ChatMessage
from config import CONFIG
from core.logger.logger import LOG

# Create sync engine for worker (no async needed)
# Convert async URL to sync URL
database_url = CONFIG.database_url
if "+asyncpg" in database_url:
    database_url = database_url.replace("+asyncpg", "")
elif "+psycopg" in database_url:
    database_url = database_url.replace("+psycopg", "")

sync_engine = create_engine(database_url)
SyncSessionLocal = sessionmaker(bind=sync_engine)

class ChatWorker:
    """RabbitMQ consumer for storing chat messages in database."""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue_name = "chat_messages"
    
    async def connect(self):
        """Connect to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(CONFIG.rabbitmq_url)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)  # Process 10 messages at a time
            
            self.queue = await self.channel.declare_queue(
                self.queue_name,
                durable=True
            )
            
            LOG.info("✅ Chat worker connected to RabbitMQ")
        except Exception as e:
            LOG.error(f"❌ Chat worker connection failed: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming messages from the queue."""
        LOG.info("🔄 Chat worker started consuming messages...")
        
        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    await self._process_message(message)
    
    async def _process_message(self, message: aio_pika.IncomingMessage):
        """Process individual message and store in database."""
        try:
            # Parse message data
            data = json.loads(message.body.decode())
            user_id = data["user_id"]
            role = data["role"]
            content = data["content"]
            
            # Store in database using sync session
            with SyncSessionLocal() as session:
                chat_message = ChatMessage(
                    user_id=user_id,
                    role=role,
                    content=content,
                    timestamp=datetime.utcnow()
                )
                session.add(chat_message)
                session.commit()
            
            LOG.debug(f"💾 Stored {role} message for user {user_id} in database")
            
        except Exception as e:
            LOG.error(f"❌ Failed to process message: {e}")
            LOG.error(f"   Message data: {message.body.decode()}")
            # Message will be requeued automatically due to exception
    
    async def disconnect(self):
        """Disconnect from RabbitMQ."""
        if self.connection:
            await self.connection.close()

async def run_chat_worker():
    """Main function to run the chat worker."""
    worker = ChatWorker()
    
    try:
        await worker.connect()
        await worker.start_consuming()
    except KeyboardInterrupt:
        LOG.info("🛑 Chat worker stopped by user")
    except Exception as e:
        LOG.error(f"❌ Chat worker error: {e}")
    finally:
        await worker.disconnect()

if __name__ == "__main__":
    # Run the worker
    asyncio.run(run_chat_worker())
