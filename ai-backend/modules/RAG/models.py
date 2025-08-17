from sqlalchemy import Column, Integer, String, DateTime, Text, UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime

Base = declarative_base()

class PDFChunk(Base):
    """PDF chunk model for storing text chunks with embeddings and position tracking."""
    __tablename__ = "pdf_chunks"
    
    # User's preferred order:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String, nullable=False)  # Original PDF filename
    embedding = Column(Vector(1536))  # OpenAI embedding dimension
    chunk_text = Column(Text, nullable=False)  # The text chunk
    page_number = Column(Integer)  # Page number if available
    sentence_count = Column(Integer)  # Number of sentences in this chunk
    chunk_index = Column(String, nullable=False)  # Position of chunk in document (e.g., "1", "1a", "1b", "2")
    start_line = Column(Integer) # Start line number in original text
    end_line = Column(Integer)   # End line number in original text
    start_pos = Column(Integer)  # Start position in original text
    end_pos = Column(Integer)    # End position in original text 