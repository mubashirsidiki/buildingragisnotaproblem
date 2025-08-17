#!/usr/bin/env python3
"""
Standalone script to run the chat worker for processing RabbitMQ messages.
"""

import asyncio
import json
import aio_pika
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database setup
Base = declarative_base()

class ChatMessage(Base):
    """Chat message model for storing conversation history."""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("idx_user_timestamp", "user_id", "timestamp"),
        Index("idx_user_role", "user_id", "role"),
    )

# Create sync engine
database_url = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/pdf_rag")
if "+asyncpg" in database_url:
    database_url = database_url.replace("+asyncpg", "")
elif "+psycopg" in database_url:
    database_url = database_url.replace("+psycopg", "")

sync_engine = create_engine(database_url)
SyncSessionLocal = sessionmaker(bind=sync_engine)

# Create tables if they don't exist
Base.metadata.create_all(sync_engine)

class ChatWorker:
    """RabbitMQ consumer for storing chat messages in database."""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.queue_name = "chat_messages"
    
    async def connect(self):
        """Connect to RabbitMQ."""
        try:
            rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
            self.connection = await aio_pika.connect_robust(rabbitmq_url)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            self.queue = await self.channel.declare_queue(
                self.queue_name,
                durable=True
            )
            
            print("✅ Chat worker connected to RabbitMQ")
        except Exception as e:
            print(f"❌ Chat worker connection failed: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming messages from the queue."""
        print("🔄 Chat worker started consuming messages...")
        
        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    await self._process_message(message)
    
    async def _process_message(self, message: aio_pika.IncomingMessage):
        """Process individual message and store in database."""
        try:
            data = json.loads(message.body.decode())
            user_id = data["user_id"]
            role = data["role"]
            content = data["content"]
            
            with SyncSessionLocal() as session:
                chat_message = ChatMessage(
                    user_id=user_id,
                    role=role,
                    content=content,
                    timestamp=datetime.utcnow()
                )
                session.add(chat_message)
                session.commit()
            
            print(f"💾 Stored {role} message for user {user_id} in database")
            
        except Exception as e:
            print(f"❌ Failed to process message: {e}")
            print(f"   Message data: {message.body.decode()}")
    
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
        print("🛑 Chat worker stopped by user")
    except Exception as e:
        print(f"❌ Chat worker error: {e}")
    finally:
        await worker.disconnect()

if __name__ == "__main__":
    print("🚀 Starting Chat Worker...")
    try:
        asyncio.run(run_chat_worker())
    except KeyboardInterrupt:
        print("Chat Worker stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred in the Chat Worker: {e}")
