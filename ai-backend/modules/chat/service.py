"""
Chat service for handling conversation logic.
"""

import openai
import json
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select
from .database import AsyncSessionLocal
from .models import ChatMessage
from .redis_service import redis_service
from .rabbitmq_service import rabbitmq_service
from config import CONFIG
from core.logger.logger import LOG
from modules.analytics.service import TokenTrackingService

class ChatService:
    """Handles chat conversation logic with Redis caching and DB persistence."""
    
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=CONFIG.openai_api_key)
    
    async def chat(self, user_message: str, user_id: str = None) -> str:
        """
        Process chat message and return GPT response.
        
        Args:
            user_message: User's message
            user_id: User identifier (uses default if not provided)
            
        Returns:
            GPT assistant response
        """
        # Use default user ID if not provided
        if not user_id:
            user_id = CONFIG.default_user_id
        
        LOG.info(f"💬 Processing chat for user: {user_id}")
        
        # Load conversation history if needed
        await self._ensure_history_loaded(user_id)
        
        # Add user message to Redis and queue for DB
        await redis_service.add_message(user_id, "user", user_message)
        await rabbitmq_service.publish_message(user_id, "user", user_message)
        
        # Get conversation context from Redis
        messages = await redis_service.get_messages(user_id)
        
        # Reverse messages for correct chronological order (Redis LPUSH reverses)
        messages.reverse()
        
        # Get GPT response
        gpt_response = await self._get_gpt_response(messages)
        
        # Add GPT response to Redis and queue for DB
        await redis_service.add_message(user_id, "assistant", gpt_response)
        await rabbitmq_service.publish_message(user_id, "assistant", gpt_response)
        
        LOG.info(f"✅ Chat response generated for user: {user_id}")
        return gpt_response
    
    async def _ensure_history_loaded(self, user_id: str):
        """Load chat history from DB to Redis if not already loaded."""
        if not await redis_service.is_history_loaded(user_id):
            LOG.info(f"📚 Loading chat history from DB for user: {user_id}")
            
            # Fetch history from database
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.user_id == user_id)
                    .order_by(ChatMessage.timestamp)
                )
                db_messages = result.scalars().all()
            
            # Convert to dict format
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in db_messages
            ]
            
            # Load into Redis
            await redis_service.load_history_from_db(user_id, history)
    
    async def _get_gpt_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from GPT using conversation context."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=CONFIG.gpt_model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            LOG.error(f"❌ GPT API error: {e}")
            return "I'm sorry, I'm having trouble responding right now. Please try again."

    async def simple_chat(self, user_message: str, user_id: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Simple chat without Redis/RabbitMQ complexity.
        
        Args:
            user_message: User's message
            user_id: User identifier
            conversation_history: Previous conversation messages
            
        Returns:
            GPT response string
        """
        LOG.info(f"💬 Simple chat: User {user_id}")
        
        # Prepare messages for OpenAI
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get GPT response
        response = await self.openai_client.chat.completions.create(
            model=CONFIG.gpt_model,
            messages=messages,
            max_tokens=1000
        )
        
        gpt_response = response.choices[0].message.content
        
        # Track token usage
        try:
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import create_engine
            engine = create_engine(CONFIG.database_url)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            
            tracker = TokenTrackingService(db)
            tracker.track_usage(
                user_id=user_id,
                operation_type="simple_chat",
                model=CONFIG.gpt_model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                metadata={"message_length": len(user_message)}
            )
            db.close()
        except Exception as e:
            LOG.error(f"Token tracking error: {e}")
        
        return gpt_response
    
    async def rag_chat(
        self, 
        user_message: str, 
        user_id: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        limit: int = 5,
        min_cosine_similarity: float = 0.5,
        min_cross_score: float = 0.0,
        expand_query: bool = True,
        rerank: bool = True
    ) -> Tuple[str, bool, Optional[List[Dict[str, Any]]]]:
        """
        Chat with RAG function calling capabilities.
        
        Args:
            user_message: User's message
            user_id: User identifier
            conversation_history: Previous conversation messages
            limit: Maximum chunks to retrieve
            min_cosine_similarity: Minimum cosine similarity threshold
            min_cross_score: Minimum cross-encoder score threshold
            expand_query: Enable query expansion
            rerank: Enable cross-encoder reranking and filtering
            
        Returns:
            Tuple of (response, used_rag, rag_sources)
        """
        LOG.info(f"💬 RAG chat: User {user_id}")
        
        # Define function for RAG search
        rag_function = {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for relevant information. Use the FULL user message as the search query, not just keywords. Preserve the complete context and meaning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complete search query. Use the full user message or question, preserving all context and details. Do not extract only keywords."
                    }
                },
                "required": ["query"]
            }
        }
        
        # Prepare messages for OpenAI
        messages = []
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Get GPT response with function calling
        response = await self.openai_client.chat.completions.create(
            model=CONFIG.gpt_model,
            messages=messages,
            functions=[rag_function],
            function_call="auto",
            max_tokens=1000
        )
        
        # Track initial token usage
        total_input_tokens = response.usage.prompt_tokens
        total_output_tokens = response.usage.completion_tokens
        
        used_rag = False
        rag_sources = []
        gpt_response = ""
        
        # Check if function was called
        if response.choices[0].message.function_call:
            # Extract search query from function call
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            extracted_query = function_args.get("query", "")
            
            # Validate if the extracted query is meaningful
            if len(extracted_query.strip()) < 10 or extracted_query.lower() in ["company", "information", "data", "details"]:
                # Use full user message if extracted query is too short or generic
                search_query = user_message
                LOG.info(f"🔍 RAG function called with generic query '{extracted_query}', using full message instead")
            else:
                search_query = extracted_query
                LOG.info(f"🔍 RAG function called with query: {search_query}")
            
            # Use RAG to search knowledge base
            from modules.RAG.service import pdf_rag_service
            rag_response = await pdf_rag_service.vector_search_only(
                query=search_query,
                limit=limit,
                min_cosine_similarity=min_cosine_similarity,
                min_cross_score=0.0,  # Filter out chunks with cross-encoder score > 0
                expand_query=expand_query,
                rerank=rerank
            )
            
            # Extract chunks from response
            rag_results = rag_response.get("chunks", [])
            
            used_rag = True
            # Return full chunk details like search endpoint
            rag_sources = rag_results
            
            # Create context from RAG results with full details
            context_parts = []
            for chunk in rag_results[:3]:
                context_parts.append(
                    f"Source: {chunk['filename']} (Page {chunk['page_number']}, Rank {chunk['rank']}, Score {chunk['cosine_similarity_score']})\n"
                    f"Text: {chunk['chunk_text']}"
                )
            context = "\n\n".join(context_parts)
            
            # Add function result to messages
            messages.append({
                "role": "assistant", 
                "content": None,
                "function_call": {
                    "name": "search_knowledge_base",
                    "arguments": json.dumps(function_args)
                }
            })
            messages.append({
                "role": "function",
                "name": "search_knowledge_base",
                "content": context
            })
            
            # Get final response with context
            final_response = await self.openai_client.chat.completions.create(
                model=CONFIG.gpt_model,
                messages=messages,
                max_tokens=1000
            )
            gpt_response = final_response.choices[0].message.content
            
            # Add second API call token usage
            total_input_tokens += final_response.usage.prompt_tokens
            total_output_tokens += final_response.usage.completion_tokens
            
        else:
            # No function call needed
            gpt_response = response.choices[0].message.content
        
        # Track total token usage for RAG chat (non-blocking)
        try:
            # Import token tracking service
            from modules.analytics.service import TokenTrackingService
            from modules.RAG.database import sync_engine
            from sqlalchemy.orm import sessionmaker
            
            # Create a new sync session for token tracking
            SessionLocal = sessionmaker(bind=sync_engine)
            db = SessionLocal()
            
            try:
                tracker = TokenTrackingService(db)
                tracker.track_usage(
                    user_id=user_id,
                    operation_type="rag_chat",
                    model=CONFIG.gpt_model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    metadata={
                        "message_length": len(user_message),
                        "used_rag": used_rag,
                        "sources_count": len(rag_sources) if rag_sources else 0
                    }
                )
                LOG.info(f"💰 RAG CHAT TOKEN USAGE TRACKED: {total_input_tokens} input, {total_output_tokens} output, ${tracker.calculate_cost(CONFIG.gpt_model, total_input_tokens, total_output_tokens):.6f}")
            finally:
                db.close()
        except Exception as e:
            LOG.error(f"Token tracking error: {e}")
        
        return gpt_response, used_rag, rag_sources

# Global chat service instance
chat_service = ChatService()
