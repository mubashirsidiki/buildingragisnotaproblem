"""
Chat API schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., example="Hello! How can you help me today?")
    user_id: Optional[str] = Field(None, example="f47ac10b-58cc-4372-a567-0e02b2c3d479")

class ChatResponse(BaseModel):
    response: str = Field(..., example="Hello! I'm here to help you. What would you like to know?")
    user_id: str = Field(..., example="f47ac10b-58cc-4372-a567-0e02b2c3d479")

class ConversationMessage(BaseModel):
    role: str = Field(..., example="user", description="Role of the message sender: 'user' or 'assistant'")
    content: str = Field(..., example="Hi there! I have a question about machine learning.", description="The message content")

class SimpleChatRequest(BaseModel):
    message: str = Field(..., example="Can you explain what artificial intelligence is?", description="The user's current message")
    user_id: str = Field(..., example="a1b2c3d4-e5f6-7890-abcd-ef1234567890", description="Unique identifier for the user")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None, 
        example=[
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ],
        description="Previous conversation messages for context"
    )

class SimpleChatResponse(BaseModel):
    response: str = Field(..., example="Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence.", description="The AI assistant's response")
    user_id: str = Field(..., example="a1b2c3d4-e5f6-7890-abcd-ef1234567890", description="User identifier")

class RagChatRequest(BaseModel):
    message: str = Field(..., example="I need information from knowledge base about machine learning algorithms", description="The user's current message")
    user_id: str = Field(..., example="a1b2c3d4-e5f6-7890-abcd-ef1234567890", description="Unique identifier for the user")
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None, 
        example=[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! I can help you with information from the knowledge base."}
        ],
        description="Previous conversation messages for context"
    )

class RagChatResponse(BaseModel):
    response: str = Field(..., example="Based on the knowledge base, here are the machine learning algorithms...", description="The AI assistant's response with RAG information")
    user_id: str = Field(..., example="a1b2c3d4-e5f6-7890-abcd-ef1234567890", description="User identifier")
    used_rag: bool = Field(..., example=True, description="Whether RAG knowledge base was used")
    rag_sources: Optional[List[Dict[str, Any]]] = Field(None, example=[
        {
            "rank": 1,
            "cosine_similarity_score": 0.8523,
            "cross_encoder_score": 0.5,
            "id": "uuid-here",
            "filename": "machine_learning_guide.pdf",
            "chunk_text": "Machine learning algorithms are computational methods...",
            "chunk_index": "1",
            "page_number": 1,
            "created_at": "2025-01-27T10:30:00",
            "start_pos": 0,
            "end_pos": 1500,
            "start_line": 1,
            "end_line": 25,
            "sentence_count": 8
        }
    ], description="Full chunk details from knowledge base search (same as search endpoint)")
