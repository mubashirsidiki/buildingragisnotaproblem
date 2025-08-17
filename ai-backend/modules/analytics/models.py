from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class TokenUsage(Base):
    __tablename__ = "token_usage"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    operation_type = Column(String, nullable=False)  # 'simple_chat', 'rag_chat', 'search'
    model = Column(String, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    estimated_cost = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_data = Column(Text, nullable=True)  # JSON metadata
