from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    chunking_mode: str
    pdf_stats: Optional[Dict[str, Any]] = None
    message: str

class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class ChunkResponse(BaseModel):
    id: str
    filename: str
    chunk_text: str
    chunk_index: int
    page_number: Optional[int]
    similarity_score: float
    created_at: datetime
    
    # Position tracking information
    start_pos: Optional[int] = None  # Start position in original text
    end_pos: Optional[int] = None    # End position in original text
    start_line: Optional[int] = None # Start line number in original text
    end_line: Optional[int] = None   # End line number in original text
    sentence_count: Optional[int] = None  # Number of sentences in this chunk

class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkResponse] 