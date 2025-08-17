"""
Chat message database models for conversation storage.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ChatMessage(Base):
    """
    Chat message model for storing conversation history.
    
    Stores both user messages and assistant replies with timestamps.
    """
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Index for efficient queries
    __table_args__ = (
        Index("idx_user_timestamp", "user_id", "timestamp"),
        Index("idx_user_role", "user_id", "role"),
    )
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, user_id='{self.user_id}', role='{self.role}')>"
