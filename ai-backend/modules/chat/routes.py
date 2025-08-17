"""
Chat API routes for conversation endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from .schema import ChatRequest, ChatResponse, SimpleChatRequest, SimpleChatResponse, RagChatRequest, RagChatResponse, ConversationMessage
from .service import chat_service
from core.logger.logger import LOG
from config import CONFIG

API_ROUTER = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

@API_ROUTER.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with Redis caching and RabbitMQ queuing."""
    try:
        # Log incoming request details
        LOG.info("💬 MAIN CHAT REQUEST RECEIVED")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📝 Message: '{request.message}'")
        LOG.info(f"   │ 👤 User ID: {request.user_id or CONFIG.default_user_id}")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        response = await chat_service.chat(request.message, request.user_id)
        final_user_id = request.user_id or CONFIG.default_user_id
        return ChatResponse(response=response, user_id=final_user_id)
    except Exception as e:
        LOG.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@API_ROUTER.post("/simple", response_model=SimpleChatResponse)
async def simple_chat_endpoint(request: SimpleChatRequest):
    """Simple chat endpoint without Redis/RabbitMQ complexity."""
    try:
        # Log incoming request details
        LOG.info("💬 SIMPLE CHAT REQUEST RECEIVED")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📝 Message: '{request.message}'")
        LOG.info(f"   │ 👤 User ID: {request.user_id}")
        LOG.info(f"   │ 📊 Conversation history: {len(request.conversation_history) if request.conversation_history else 0} messages")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        # Convert schema objects to dict format for service
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content} 
                for msg in request.conversation_history
            ]
        
        response = await chat_service.simple_chat(
            request.message, 
            request.user_id, 
            conversation_history
        )
        
        return SimpleChatResponse(
            response=response,
            user_id=request.user_id
        )
        
    except Exception as e:
        LOG.error(f"❌ Simple chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@API_ROUTER.post("/rag", response_model=RagChatResponse)
async def rag_chat_endpoint(
    request: RagChatRequest,
    limit: int = Query(default=10, description="Maximum chunks to retrieve"),
    min_cosine_similarity: float = Query(default=0.5, description="Minimum cosine similarity threshold"),
    min_cross_score: float = Query(default=0.0, description="Minimum cross-encoder score threshold"),
    expand_query: bool = Query(default=True, description="Enable query expansion"),
    rerank: bool = Query(default=True, description="Enable cross-encoder reranking")
):
    """Chat endpoint with RAG function calling capabilities."""
    try:
        # Log incoming request details
        LOG.info("💬 RAG CHAT REQUEST RECEIVED")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📝 Message: '{request.message}'")
        LOG.info(f"   │ 👤 User ID: {request.user_id}")
        LOG.info(f"   │ 📊 Conversation history: {len(request.conversation_history) if request.conversation_history else 0} messages")
        LOG.info("   ├─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🎯 Limit: {limit}")
        LOG.info(f"   │ 📊 Min cosine similarity: {min_cosine_similarity}")
        LOG.info(f"   │ 🎯 Min cross-encoder score: {min_cross_score}")
        LOG.info(f"   │ 🔄 Query expansion: {'✅ enabled' if expand_query else '❌ disabled'}")
        LOG.info(f"   │ 🔄 Cross-encoder: {'✅ enabled' if rerank else '❌ disabled'}")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        # Convert schema objects to dict format for service
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content} 
                for msg in request.conversation_history
            ]
        
        response, used_rag, rag_sources = await chat_service.rag_chat(
            request.message, 
            request.user_id, 
            conversation_history,
            limit,
            min_cosine_similarity,
            min_cross_score,
            expand_query,
            rerank
        )
        
        return RagChatResponse(
            response=response,
            user_id=request.user_id,
            used_rag=used_rag,
            rag_sources=rag_sources if rag_sources else None
        )
        
    except Exception as e:
        LOG.error(f"❌ RAG chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
