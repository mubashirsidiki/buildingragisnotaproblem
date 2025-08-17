"""
PDF RAG Service

A comprehensive service for processing PDFs, creating semantic chunks,
generating embeddings, and performing intelligent document retrieval.
"""

import re
import io
import time
from typing import List, Tuple, Dict, Any

import PyPDF2
import numpy as np
from openai import AsyncOpenAI
from sqlalchemy import select, delete
from sentence_transformers import CrossEncoder

from config import CONFIG
from core.logger.logger import LOG
from .database import AsyncSessionLocal, setup_database
from .models import PDFChunk

from .prompts import get_query_expansion_prompt


class PDFRAGService:
    """
    PDF Retrieval-Augmented Generation Service
    
    Handles PDF processing, semantic chunking, vector storage,
    and intelligent document retrieval with advanced features.
    """

    def __init__(self):
        """Initialize the PDF RAG service with required clients."""
        self.openai_client = AsyncOpenAI(api_key=CONFIG.openai_api_key)
        self.cross_encoder = None  # Lazy-loaded when needed

    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================

    async def initialize_database(self):
        """Set up the database and create necessary tables."""
        await setup_database()

    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        chunking_mode: str = "sentence",
        max_chunk_size: int = 1500,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = None,
        chunk_overlap: int = 200,
        sentence_group_size: int = 3,
    ) -> Dict[str, Any]:
        """
        Process PDF file and create semantic chunks with embeddings.
    
    Args:
            file_content: PDF file bytes
            filename: Original PDF filename
            chunking_mode: 'sentence' or 'length'
            max_chunk_size: Maximum characters per chunk
            breakpoint_threshold_type: Statistical threshold type
            breakpoint_threshold_amount: Threshold value
            chunk_overlap: Character overlap between chunks
            sentence_group_size: Sentences to group for semantic analysis
        
    Returns:
            Processing results with statistics
        """
        # Set default threshold amounts
        if breakpoint_threshold_amount is None:
            threshold_defaults = {
                "percentile": 95.0,
                "standard_deviation": 3.0,
                "interquartile": 1.5
            }
            breakpoint_threshold_amount = threshold_defaults.get(breakpoint_threshold_type, 95.0)

        start_time = time.time()
        LOG.info(f"🚀 Processing PDF: {filename} with {chunking_mode} chunking")

        # Extract text and analyze structure
        text, page_numbers = self._extract_text_from_pdf(file_content)
        if not text.strip():
            return {"error": "No text could be extracted from the PDF"}

        pdf_stats = self._analyze_pdf_structure(file_content, text)
        self._log_pdf_statistics(pdf_stats)

        # Create chunks based on selected mode
        chunks = await self._create_chunks_with_mode(
            text=text,
            chunking_mode=chunking_mode,
            max_chunk_size=max_chunk_size,
            total_lines=pdf_stats["total_lines"],
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            chunk_overlap=chunk_overlap,
            sentence_group_size=sentence_group_size,
            page_numbers=page_numbers,
        )

        self._log_chunking_results(chunks, chunking_mode, max_chunk_size, chunk_overlap, sentence_group_size)

        # Generate embeddings and store in database
        user_id = "pdf_upload_user"  # Default user for PDF uploads
        chunks_created = await self._create_and_store_embeddings(chunks, filename, user_id)

        processing_time = time.time() - start_time
        self._log_processing_summary(filename, chunks_created, chunking_mode, processing_time)
        
        # Log total cost summary for this upload
        self._log_upload_cost_summary(user_id, chunks_created)

        return {
            "filename": filename,
            "chunks_created": chunks_created,
            "chunking_mode": chunking_mode,
            "pdf_stats": pdf_stats,
            "message": f"Successfully processed {filename} with {chunks_created} chunks using {chunking_mode} mode",
        }



    async def vector_search_only(
        self,
        query: str,
        limit: int = 10,
        min_cosine_similarity: float = 0.5,
        min_cross_score: float = 0.0,
        expand_query: bool = True,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Perform pure vector search without LLM generation.
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_cosine_similarity: Minimum cosine similarity threshold
            min_cross_score: Minimum cross-encoder score threshold (lower = more relevant)
            expand_query: Enable multi-query expansion
            rerank: Enable cross-encoder reranking and filtering
            
        Returns:
            Ranked search results with metadata
        """
        LOG.info(f"🔍 Starting vector search process...")
        self._log_search_parameters(query, limit, min_cosine_similarity, expand_query, rerank, min_cross_score)

        # Perform search with optional expansion
        user_id = "search_user"  # Default user for searches
        if expand_query:
            similar_chunks = await self._multi_query_search(query, limit, user_id)
        else:
            similar_chunks = await self._single_query_search(query, limit, user_id)

        # First filter: Remove chunks below similarity threshold
        LOG.info("🔽 PRE-FILTERING BY COSINE SIMILARITY")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🎯 Threshold: {min_cosine_similarity}")
        LOG.info(f"   │ 📊 Total chunks: {len(similar_chunks)}")
        
        pre_filtered_chunks = []
        filtered_by_similarity = 0
        
        for chunk in similar_chunks:
            similarity_score = 1 - chunk.distance
            if similarity_score >= min_cosine_similarity:
                pre_filtered_chunks.append(chunk)
            else:
                filtered_by_similarity += 1
        
        LOG.info(f"   │ 🔽 Filtered out: {filtered_by_similarity} chunks")
        LOG.info(f"   │ ✅ Remaining: {len(pre_filtered_chunks)} chunks")
        LOG.info("   └─────────────────────────────────────────────────────────────")

        # Apply reranking if requested (only on pre-filtered chunks)
        if rerank and pre_filtered_chunks:
            similar_chunks = await self._rerank_chunks(query, pre_filtered_chunks)
        else:
            similar_chunks = pre_filtered_chunks

        # Second filter: Apply cross-encoder threshold and format results
        chunks = self._filter_and_format_chunks(similar_chunks, min_cosine_similarity, rerank, min_cross_score)

        result = {
            "distance_metric": "cosine_distance",
            "total_results": len(chunks),
            "chunks": chunks
        }

        self._log_search_completion(chunks, expand_query)
        return result

    async def clear_database(self):
        """Clear all PDF chunks from the database."""
        async with AsyncSessionLocal() as session:
            await session.execute(delete(PDFChunk))
            await session.commit()
        LOG.info("🗑️ Cleared all PDF chunks from database")

    # ========================================================================
    # PDF PROCESSING METHODS
    # ========================================================================

    def _extract_text_from_pdf(self, file_content: bytes) -> Tuple[str, List[int]]:
        """
        Extract text from PDF with page number mapping.
        
        Args:
            file_content: PDF file bytes
            
        Returns:
            Tuple of (text, page_numbers_list)
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            page_numbers = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += page_text + "\n"
                page_numbers.extend([page_num] * len(page_text))
                page_numbers.append(page_num)

            return text, page_numbers

        except Exception as e:
            LOG.error(f"❌ Error extracting text from PDF: {e}")
            raise

    def _analyze_pdf_structure(self, file_content: bytes, text: str) -> Dict[str, Any]:
        """
        Analyze PDF structure and return comprehensive statistics.
        
        Args:
            file_content: PDF file bytes
            text: Extracted text
            
        Returns:
            Dictionary with PDF statistics
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            total_pages = len(pdf_reader.pages)
            total_characters = len(text)
            total_words = len(text.split())
            total_lines = text.count("\n") + 1

            sentences = self._split_into_sentences(text)
            total_sentences = len(sentences)

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            total_paragraphs = len(paragraphs)

            # Calculate averages
            avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
            avg_sentences_per_paragraph = total_sentences / total_paragraphs if total_paragraphs > 0 else 0
            avg_words_per_line = total_words / total_lines if total_lines > 0 else 0

            # Estimate lines if count seems too low
            if total_lines < 10 and total_characters > 1000:
                estimated_lines = max(49, total_characters // 65)
                total_lines = estimated_lines

            return {
                "total_pages": total_pages,
                "total_characters": total_characters,
                "total_words": total_words,
                "total_lines": total_lines,
                "total_sentences": total_sentences,
                "total_paragraphs": total_paragraphs,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
                "avg_words_per_line": avg_words_per_line,
            }

        except Exception as e:
            LOG.error(f"❌ Error analyzing PDF structure: {e}")
            return {
                "total_pages": 0,
                "total_characters": len(text),
                "total_words": len(text.split()),
                "total_lines": text.count("\n") + 1,
                "total_sentences": 0,
                "total_paragraphs": 0,
                "avg_words_per_sentence": 0,
                "avg_sentences_per_paragraph": 0,
                "avg_words_per_line": 0,
            }

    # ========================================================================
    # CHUNKING METHODS
    # ========================================================================

    async def _create_chunks_with_mode(
        self,
        text: str,
        chunking_mode: str = "sentence",
        max_chunk_size: int = 1500,
        total_lines: int = 0,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        chunk_overlap: int = 200,
        sentence_group_size: int = 3,
        page_numbers: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks based on specified mode with position tracking.
        
        Args:
            text: Text to chunk
            chunking_mode: 'sentence' or 'length'
            max_chunk_size: Maximum characters per chunk
            total_lines: Total lines in document
            breakpoint_threshold_type: Statistical threshold type
            breakpoint_threshold_amount: Threshold value
            chunk_overlap: Character overlap between chunks
            sentence_group_size: Sentences per group for semantic analysis
            page_numbers: Page number mapping
            
        Returns:
            List of chunk dictionaries with metadata
        """
        LOG.info(f"🎯 Starting chunking with {chunking_mode} mode and position tracking...")

        if chunking_mode.lower() == "length":
            return self._create_length_based_chunks(
                text, max_chunk_size, chunk_overlap, total_lines, page_numbers
            )
        elif chunking_mode.lower() == "sentence":
            return await self._create_sentence_based_chunks(
                text, max_chunk_size, total_lines, breakpoint_threshold_type,
                breakpoint_threshold_amount, chunk_overlap, sentence_group_size, page_numbers
            )
        else:
            raise ValueError(f"Invalid chunking_mode: {chunking_mode}. Must be 'sentence' or 'length'")

    async def _create_sentence_based_chunks(
        self,
        text: str,
        max_chunk_size: int,
        total_lines: int,
        breakpoint_threshold_type: str,
        breakpoint_threshold_amount: float,
        chunk_overlap: int,
        sentence_group_size: int,
        page_numbers: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create semantic chunks using sentence grouping and similarity analysis.
        
        Returns:
            List of semantic chunks with position metadata
        """
        # Split text into sentences with position tracking
        sentence_units = self._split_sentences_with_positions(text, total_lines, page_numbers)
        LOG.info(f"📝 Split text into {len(sentence_units)} sentences")

        if len(sentence_units) <= 1:
            return [self._create_single_chunk(text, total_lines)]

        # Group sentences for semantic analysis
        grouped_units = self._group_sentences(sentence_units, sentence_group_size)
        LOG.info(f"📦 Grouped {len(sentence_units)} sentences into {len(grouped_units)} groups of size {sentence_group_size}")

        # Generate embeddings for sentence groups
        unit_embeddings = await self._generate_unit_embeddings(grouped_units)

        # Perform semantic analysis and create chunks
        chunks = await self._perform_semantic_chunking(
            unit_embeddings, breakpoint_threshold_type, breakpoint_threshold_amount,
            max_chunk_size, chunk_overlap
        )

        LOG.info(f"✅ Created {len(chunks)} semantic chunks")
        return chunks

    def _create_length_based_chunks(
        self,
        text: str,
        max_chunk_size: int,
        chunk_overlap: int,
        total_lines: int,
        page_numbers: List[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create fixed-length chunks with sentence boundary awareness.
        
        Returns:
            List of length-based chunks with position metadata
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                search_start = max(start, end - 100)
                for i in range(end, search_start, -1):
                    if text[i - 1] in ".!?":
                        end = i
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = self._calculate_chunk_positions(
                    chunk_text, start, end, total_lines, page_numbers
                )
                chunks.append(chunk_metadata)

            start = end - chunk_overlap
            if start >= len(text):
                break

        return chunks

    # ========================================================================
    # EMBEDDING AND SIMILARITY METHODS
    # ========================================================================

    async def _generate_query_embedding(self, query: str, user_id: str = None) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=CONFIG.openai_embedding_model,
                input=query,
                encoding_format="float",
            )
            
            # Track token usage for search operations
            if user_id and user_id == "search_user" and hasattr(response, 'usage'):
                try:
                    # Use sync database connection for token tracking
                    from modules.analytics.service import TokenTrackingService
                    from modules.RAG.database import sync_engine
                    from sqlalchemy.orm import sessionmaker
                    
                    SessionLocal = sessionmaker(bind=sync_engine)
                    db = SessionLocal()
                    
                    tracker = TokenTrackingService(db)
                    
                    # Get token info before tracking
                    input_tokens = response.usage.prompt_tokens if response.usage else 0
                    output_tokens = 0  # Embeddings don't have output tokens
                    total_tokens = input_tokens + output_tokens
                    estimated_cost = tracker.calculate_cost(CONFIG.openai_embedding_model, input_tokens, output_tokens)
                    
                    usage_record = tracker.track_usage(
                        user_id="search_operations",
                        operation_type="search",
                        model=CONFIG.openai_embedding_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        metadata={"query": query}
                    )
                    
                    # Get record ID before closing session
                    record_id = usage_record.id
                    db.close()
                    
                    # Log the cost tracking with detailed info
                    LOG.info("💰 TOKEN USAGE TRACKED:")
                    LOG.info("   ┌─────────────────────────────────────────────────────────────")
                    LOG.info(f"   │ 🔧 Operation: Search Query Embedding")
                    LOG.info(f"   │ 🤖 Model: {CONFIG.openai_embedding_model}")
                    LOG.info(f"   │ 📝 Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                    LOG.info(f"   │ 📝 Query Length: {len(query)} characters")
                    LOG.info(f"   │ 📥 Input Tokens: {input_tokens:,}")
                    LOG.info(f"   │ 📤 Output Tokens: {output_tokens:,}")
                    LOG.info(f"   │ 🔢 Total Tokens: {total_tokens:,}")
                    LOG.info(f"   │ 💵 Estimated Cost: ${estimated_cost:.6f}")
                    LOG.info(f"   │ 🆔 Record ID: {record_id}")
                    LOG.info(f"   │ 👤 User ID: search_operations")
                    LOG.info("   └─────────────────────────────────────────────────────────────")
                    
                except Exception as e:
                    LOG.error(f"Token tracking error: {e}")
            
            return response.data[0].embedding
        except Exception as e:
            LOG.error(f"❌ Error getting query embedding: {e}")
            return []

    async def _generate_unit_embeddings(self, grouped_units: List[Tuple]) -> List[Tuple]:
        """
        Generate embeddings for grouped sentence units.
        
        Args:
            grouped_units: List of grouped sentence tuples
            
        Returns:
            List of unit embeddings with metadata
        """
        unit_embeddings = []
        
        for i, unit_data in enumerate(grouped_units):
            unit_text = unit_data[0]
            LOG.info(f"🧠 Getting embedding for sentence group {i + 1}/{len(grouped_units)}")
            
            embedding = await self._get_single_embedding(unit_text)
            if embedding:
                unit_embeddings.append(unit_data + (embedding,))

        return unit_embeddings

    async def _get_single_embedding(self, text: str, user_id: str = None) -> List[float]:
        """
        Get embedding for a single text string.
        
        Args:
            text: Text to embed
            user_id: User identifier for token tracking
            
        Returns:
            Embedding vector or empty list on error
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=CONFIG.openai_embedding_model,
                input=text,
                encoding_format="float",
            )
            
            # Track token usage for PDF upload operations
            if user_id and user_id == "pdf_upload_user" and hasattr(response, 'usage'):
                try:
                    # Use sync database connection for token tracking
                    from modules.analytics.service import TokenTrackingService
                    from modules.RAG.database import sync_engine
                    from sqlalchemy.orm import sessionmaker
                    
                    SessionLocal = sessionmaker(bind=sync_engine)
                    db = SessionLocal()
                    
                    tracker = TokenTrackingService(db)
                    
                    # Get token info before tracking
                    input_tokens = response.usage.prompt_tokens if response.usage else 0
                    output_tokens = 0  # Embeddings don't have output tokens
                    total_tokens = input_tokens + output_tokens
                    estimated_cost = tracker.calculate_cost(CONFIG.openai_embedding_model, input_tokens, output_tokens)
                    
                    usage_record = tracker.track_usage(
                        user_id="upload_operations",
                        operation_type="upload",
                        model=CONFIG.openai_embedding_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        metadata={"text_length": len(text)}
                    )
                    
                    # Get record ID before closing session
                    record_id = usage_record.id
                    db.close()
                    
                    # Log the cost tracking with detailed info
                    LOG.info("💰 TOKEN USAGE TRACKED:")
                    LOG.info("   ┌─────────────────────────────────────────────────────────────")
                    LOG.info(f"   │ 🔧 Operation: PDF Upload Embedding")
                    LOG.info(f"   │ 🤖 Model: {CONFIG.openai_embedding_model}")
                    LOG.info(f"   │ 📝 Text Length: {len(text):,} characters")
                    LOG.info(f"   │ 📥 Input Tokens: {input_tokens:,}")
                    LOG.info(f"   │ 📤 Output Tokens: {output_tokens:,}")
                    LOG.info(f"   │ 🔢 Total Tokens: {total_tokens:,}")
                    LOG.info(f"   │ 💵 Estimated Cost: ${estimated_cost:.6f}")
                    LOG.info(f"   │ 🆔 Record ID: {record_id}")
                    LOG.info(f"   │ 👤 User ID: upload_operations")
                    LOG.info("   └─────────────────────────────────────────────────────────────")
                    
                except Exception as e:
                    LOG.error(f"Token tracking error: {e}")
            
            return response.data[0].embedding
        except Exception as e:
            LOG.error(f"❌ Error getting embedding: {e}")
            return []

    async def _create_and_store_embeddings(self, chunks: List[Dict], filename: str, user_id: str = None) -> int:
        """
        Create embeddings for chunks and store in database.
        
        Args:
            chunks: List of chunk dictionaries
            filename: Original PDF filename
            
        Returns:
            Number of chunks created
        """
        async with AsyncSessionLocal() as session:
            chunks_created = 0
            chunk_counter = 0
            sub_chunk_counter = 0

            for chunk_data in chunks:
                # Generate embedding for chunk text
                embedding = await self._get_single_embedding(chunk_data["text"], user_id)

                # Determine chunk index (hierarchical naming)
                if chunk_data.get("is_sub_chunk", False):
                    chunk_index_str = f"{chunk_data['parent_chunk']}{chr(97 + sub_chunk_counter)}"
                    sub_chunk_counter += 1
                else:
                    chunk_counter += 1
                    sub_chunk_counter = 0
                    chunk_index_str = str(chunk_counter)

                # Create database record
                pdf_chunk = PDFChunk(
                    filename=filename,
                    chunk_text=chunk_data["text"],
                    chunk_index=chunk_index_str,
                    page_number=chunk_data.get("page_number", 1),
                    embedding=embedding,
                    start_pos=chunk_data["start_pos"],
                    end_pos=chunk_data["end_pos"],
                    start_line=chunk_data["start_line"],
                    end_line=chunk_data["end_line"],
                    sentence_count=chunk_data["sentences"],
                )
                
                session.add(pdf_chunk)
                chunks_created += 1

                self._log_database_record(chunk_data, chunk_index_str, embedding, chunks_created, filename)

            await session.commit()
            self._log_database_summary(chunks_created, filename)
            return chunks_created

    # ========================================================================
    # SEARCH AND QUERY METHODS
    # ========================================================================

    async def _single_query_search(self, query: str, limit: int, user_id: str = None) -> List:
        """
        Perform single query vector search.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of similar chunks
        """
        LOG.info(f"🧠 Generating query embedding...")
        embedding_start = time.time()
        
        query_embedding = await self._generate_query_embedding(query, user_id)
        
        embedding_time = time.time() - embedding_start
        LOG.info(f"✅ Query embedding generated in {embedding_time:.3f}s (dimension: {len(query_embedding)})")

        LOG.info(f"🔎 Searching database for similar chunks...")
        search_start = time.time()
        
        similar_chunks = await self._find_similar_chunks(query_embedding, limit)
        
        search_time = time.time() - search_start
        LOG.info(f"✅ Database search completed in {search_time:.3f}s")

        return similar_chunks

    async def _multi_query_search(self, query: str, limit: int, user_id: str = None) -> List:
        """
        Perform multi-query expansion search.
        
        Args:
            query: Original search query
            limit: Maximum results per query
            
        Returns:
            List of unique similar chunks
        """
        LOG.info(f"🔄 Expanding query with multiple perspectives...")
        expansion_start = time.time()
        
        expanded_queries = await self._expand_query(query, user_id)
        
        expansion_time = time.time() - expansion_start
        LOG.info(f"✅ Query expanded in {expansion_time:.3f}s: {len(expanded_queries)} queries generated")

        # Search with all queries and collect unique results
        all_chunks = []
        seen_ids = set()

        for i, expanded_query in enumerate(expanded_queries, 1):
            LOG.info(f"🔎 Searching with query {i}: '{expanded_query}'")
            
            query_embedding = await self._generate_query_embedding(expanded_query, user_id)
            chunks = await self._find_similar_chunks(query_embedding, limit)

            # Add unique chunks only
            for chunk in chunks:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        # Sort by best similarity and limit results
        all_chunks.sort(key=lambda x: x.distance)
        LOG.info(f"✅ Multi-query search found {len(seen_ids)} unique chunks")
        
        return all_chunks[:limit]

    async def _find_similar_chunks(self, query_embedding: List[float], limit: int) -> List:
        """
        Find similar chunks using vector similarity search.
        
        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            
        Returns:
            List of similar PDFChunk objects with distance
        """
        async with AsyncSessionLocal() as session:
            stmt = (
                select(
                    PDFChunk,
                    PDFChunk.embedding.cosine_distance(query_embedding).label("distance"),
                )
                .where(PDFChunk.embedding.is_not(None))
                .order_by("distance")
                .limit(limit)
            )

            # SELECT 
            #     pdf_chunk.*,
            #     pdf_chunk.embedding <=> '[0.123, 0.456, ...]' AS distance
            # FROM 
            #     pdf_chunk
            # WHERE 
            #     pdf_chunk.embedding IS NOT NULL
            # ORDER BY 
            #     distance ASC
            # LIMIT 
            #     5;

            
            result = await session.execute(stmt)
            rows = result.fetchall()
            
            # Attach distance to chunk objects
            for row in rows:
                row[0].distance = row[1]

            return [row[0] for row in rows]

    async def _rerank_chunks(self, query: str, chunks: List) -> List:
        """
        Rerank chunks using cross-encoder model for improved relevance.
        
        Args:
            query: Original query
            chunks: List of retrieved chunks
            
        Returns:
            Reranked list of chunks
        """
        try:
            LOG.info(f"🔄 Reranking {len(chunks)} chunks with cross-encoder...")
            rerank_start = time.time()

            # Lazy load cross-encoder model
            if self.cross_encoder is None:
                LOG.info("🔧 Loading cross-encoder model...")
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                LOG.info("✅ Cross-encoder model loaded")

            # Prepare query-chunk pairs for scoring
            pairs = [(query, chunk.chunk_text) for chunk in chunks]

            # Get cross-encoder relevance scores
            scores = self.cross_encoder.predict(pairs)

            # Attach scores and sort by relevance
            for chunk, score in zip(chunks, scores):
                chunk.cross_score = float(score)

            reranked_chunks = sorted(chunks, key=lambda x: x.cross_score, reverse=True)

            rerank_time = time.time() - rerank_start
            LOG.info(f"✅ Reranking completed in {rerank_time:.3f}s")
            LOG.info(f"📊 Top reranking scores: {[f'{c.cross_score:.3f}' for c in reranked_chunks[:5]]}")

            return reranked_chunks

        except Exception as e:
            LOG.error(f"❌ Reranking failed: {e}")
            return chunks

    # ========================================================================
    # QUERY EXPANSION METHODS
    # ========================================================================

    async def _expand_query(self, query: str, user_id: str = None) -> List[str]:
        """
        Generate multiple perspective queries using LLM.
        
        Args:
            query: Original query
            user_id: User identifier for token tracking
            
        Returns:
            List of expanded queries including original
        """
        try:
            prompt = get_query_expansion_prompt(query)

            response = await self.openai_client.chat.completions.create(
                model=CONFIG.gpt_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )

            # Track token usage for search operations (query expansion)
            if user_id and user_id == "search_user" and hasattr(response, 'usage'):
                try:
                    # Use sync database connection for token tracking
                    from modules.analytics.service import TokenTrackingService
                    from modules.RAG.database import sync_engine
                    from sqlalchemy.orm import sessionmaker
                    
                    SessionLocal = sessionmaker(bind=sync_engine)
                    db = SessionLocal()
                    
                    tracker = TokenTrackingService(db)
                    
                    # Get token info before tracking
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = input_tokens + output_tokens
                    estimated_cost = tracker.calculate_cost(CONFIG.gpt_model, input_tokens, output_tokens)
                    
                    usage_record = tracker.track_usage(
                        user_id="search_operations",
                        operation_type="search",
                        model=CONFIG.gpt_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        metadata={"original_query": query, "operation": "query_expansion"}
                    )
                    
                    # Get record ID before closing session
                    record_id = usage_record.id
                    db.close()
                    
                    # Log the cost tracking with detailed info
                    LOG.info("💰 TOKEN USAGE TRACKED:")
                    LOG.info("   ┌─────────────────────────────────────────────────────────────")
                    LOG.info(f"   │ 🔧 Operation: Query Expansion (LLM)")
                    LOG.info(f"   │ 🤖 Model: {CONFIG.gpt_model}")
                    LOG.info(f"   │ 📝 Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
                    LOG.info(f"   │ 📝 Query Length: {len(query)} characters")
                    LOG.info(f"   │ 📥 Input Tokens: {input_tokens:,}")
                    LOG.info(f"   │ 📤 Output Tokens: {output_tokens:,}")
                    LOG.info(f"   │ 🔢 Total Tokens: {total_tokens:,}")
                    LOG.info(f"   │ 💵 Estimated Cost: ${estimated_cost:.6f}")
                    LOG.info(f"   │ 🆔 Record ID: {record_id}")
                    LOG.info(f"   │ 👤 User ID: search_operations")
                    LOG.info("   └─────────────────────────────────────────────────────────────")
                    
                except Exception as e:
                    LOG.error(f"Token tracking error: {e}")

            expanded_text = response.choices[0].message.content.strip()

            # Parse numbered list of queries
            queries = [query]  # Include original query
            lines = expanded_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.')) or not line[0].isdigit()):
                    clean_query = line.lstrip('123.').strip()
                    if clean_query and clean_query != query:
                        queries.append(clean_query)

            return queries[:4]  # Maximum 4 queries (original + 3 expanded)

        except Exception as e:
            LOG.error(f"❌ Query expansion failed: {e}")
            return [query]



    # ========================================================================
    # TEXT PROCESSING UTILITY METHODS
    # ========================================================================

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence strings
        """
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_sentences_with_positions(
        self,
        text: str,
        total_lines: int,
        page_numbers: List[int] = None,
    ) -> List[Tuple[str, int, int, int, int, int]]:
        """
        Split text into sentences with position tracking.
        
        Args:
            text: Text to split
            total_lines: Total lines in document
            page_numbers: Page number mapping
            
        Returns:
            List of tuples: (sentence, start_pos, end_pos, start_line, end_line, page_num)
        """
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        sentences_with_pos = []
        sentence_matches = list(re.finditer(r"(?<=[.!?])\s+", text))

        if not sentence_matches:
            page_num = page_numbers[0] if page_numbers else 1
            return [(text.strip(), 0, len(text), 1, total_lines, page_num)]

        sentence_starts = [0] + [match.end() for match in sentence_matches]
        sentence_ends = [match.start() for match in sentence_matches] + [len(text)]

        for start, end in zip(sentence_starts, sentence_ends):
            sentence = text[start:end].strip()
            if sentence:
                start_line, end_line = self._calculate_line_numbers(start, end, text, total_lines)
                page_num = page_numbers[start] if page_numbers and start < len(page_numbers) else 1
                sentences_with_pos.append((sentence, start, end, start_line, end_line, page_num))

        return sentences_with_pos

    def _group_sentences(self, sentence_units: List[Tuple], group_size: int) -> List[Tuple]:
        """
        Group sentences for semantic analysis.
        
        Args:
            sentence_units: List of sentence tuples
            group_size: Number of sentences per group
            
        Returns:
            List of grouped sentence tuples
        """
        grouped_units = []
        
        for i in range(0, len(sentence_units), group_size):
            group = sentence_units[i:i + group_size]
            if group:
                # Combine sentences in the group
                combined_text = " ".join([unit[0] for unit in group])
                start_pos = group[0][1]
                end_pos = group[-1][2]
                start_line = group[0][3]
                end_line = group[-1][4]
                page_num = group[0][5]
                sentence_count = len(group)

                grouped_units.append((
                    combined_text, start_pos, end_pos, start_line,
                    end_line, page_num, sentence_count
                ))

        return grouped_units

    def _calculate_line_numbers(self, start: int, end: int, text: str, total_lines: int) -> Tuple[int, int]:
        """
        Calculate line numbers for character positions.
        
        Args:
            start: Start character position
            end: End character position
            text: Full text
            total_lines: Total lines in document
            
        Returns:
            Tuple of (start_line, end_line)
        """
        if total_lines > 1:
            chars_per_line = len(text) / total_lines
            start_line = max(1, int(start / chars_per_line) + 1)
            end_line = max(start_line, int(end / chars_per_line) + 1)
        else:
            start_line = text[:start].count("\n") + 1
            end_line = text[:end].count("\n") + 1

        return start_line, end_line

    def _calculate_chunk_positions(
        self,
        chunk_text: str,
        start: int,
        end: int,
        total_lines: int,
        page_numbers: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate position metadata for a chunk.
        
        Args:
            chunk_text: Text content of chunk
            start: Start character position
            end: End character position
            total_lines: Total lines in document
            page_numbers: Page number mapping
            
        Returns:
            Dictionary with chunk metadata
        """
        start_line, end_line = self._calculate_line_numbers(start, end, chunk_text, total_lines)
        page_num = page_numbers[start] if page_numbers and start < len(page_numbers) else 1
        sentences = self._split_into_sentences(chunk_text)
        sentence_count = len(sentences)

        return {
            "text": chunk_text,
            "start_pos": start,
            "end_pos": end,
            "start_line": start_line,
            "end_line": end_line,
            "sentences": sentence_count,
            "page_number": page_num,
            "is_sub_chunk": False,
        }

    # ========================================================================
    # SEMANTIC ANALYSIS METHODS
    # ========================================================================

    async def _perform_semantic_chunking(
        self,
        unit_embeddings: List[Tuple],
        breakpoint_threshold_type: str,
        breakpoint_threshold_amount: float,
        max_chunk_size: int,
        chunk_overlap: int,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic chunking with statistical threshold analysis.
        
        Args:
            unit_embeddings: List of sentence embeddings with metadata
            breakpoint_threshold_type: Type of statistical threshold
            breakpoint_threshold_amount: Threshold value
            max_chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of semantic chunks
        """
        if len(unit_embeddings) < 2:
            return self._create_single_chunk_from_embeddings(unit_embeddings)

        # Calculate cosine distances between consecutive embeddings
        distances = self._calculate_cosine_distances(unit_embeddings)

        # Find breakpoints using statistical thresholds
        threshold = self._calculate_threshold(distances, breakpoint_threshold_type, breakpoint_threshold_amount)
        breakpoints = [i + 1 for i, distance in enumerate(distances) if distance > threshold]

        self._log_semantic_analysis_results(distances, threshold, breakpoint_threshold_type, breakpoints)

        # Create chunks from breakpoints
        chunks = self._create_chunks_from_breakpoints(
            unit_embeddings, breakpoints, max_chunk_size, chunk_overlap
        )

        return chunks

    def _calculate_cosine_distances(self, unit_embeddings: List[Tuple]) -> List[float]:
        """
        Calculate cosine distances between consecutive embeddings.
        
        Args:
            unit_embeddings: List of embedding tuples
            
        Returns:
            List of cosine distances
        """
        distances = []
        
        for i in range(1, len(unit_embeddings)):
            current_embedding = unit_embeddings[i][-1]  # Last element is embedding
            previous_embedding = unit_embeddings[i - 1][-1]
            
            similarity = self._cosine_similarity(current_embedding, previous_embedding)
            distances.append(1 - similarity)  # Convert to distance

        return distances

    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)

    def _calculate_threshold(
        self,
        distances: List[float],
        threshold_type: str,
        threshold_amount: float
    ) -> float:
        """
        Calculate statistical threshold for breakpoint detection.
        
        Args:
            distances: List of cosine distances
            threshold_type: Type of threshold ('percentile', 'standard_deviation', etc.)
            threshold_amount: Threshold parameter value
            
        Returns:
            Calculated threshold value
        """
        if threshold_type == "percentile":
            threshold = np.percentile(distances, threshold_amount)
            LOG.info(f"📊 Using percentile threshold: {threshold:.4f} (amount: {threshold_amount})")
            
        elif threshold_type == "standard_deviation":
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = mean_dist + (threshold_amount * std_dist)
            LOG.info(f"📊 Using standard deviation threshold: {threshold:.4f} (amount: {threshold_amount})")
            
        elif threshold_type == "interquartile":
            q75, q25 = np.percentile(distances, [75, 25])
            iqr = q75 - q25
            threshold = q75 + (threshold_amount * iqr)
            LOG.info(f"📊 Using interquartile threshold: {threshold:.4f} (amount: {threshold_amount})")
            
        else:  # Default fallback
            threshold = threshold_amount
            LOG.info(f"📊 Using fixed threshold: {threshold:.4f}")

        return threshold

    # ========================================================================
    # CHUNK CREATION UTILITY METHODS
    # ========================================================================

    def _create_single_chunk(self, text: str, total_lines: int) -> List[Dict[str, Any]]:
        """Create a single chunk for very short texts."""
        return [{
            "text": text,
            "start_pos": 0,
            "end_pos": len(text),
            "start_line": 1,
            "end_line": text.count("\n") + 1,
            "sentences": 1,
            "page_number": 1,
            "is_sub_chunk": False,
        }]

    def _create_single_chunk_from_embeddings(self, unit_embeddings: List[Tuple]) -> List[Dict[str, Any]]:
        """Create a single chunk from embedding units."""
        chunk_text = " ".join([u[0] for u in unit_embeddings])
        total_sentences = sum(u[6] for u in unit_embeddings) if len(unit_embeddings[0]) > 6 else len(unit_embeddings)
        
        return [{
            "text": chunk_text,
            "start_pos": unit_embeddings[0][1],
            "end_pos": unit_embeddings[-1][2],
            "start_line": unit_embeddings[0][3],
            "end_line": unit_embeddings[-1][4],
            "sentences": total_sentences,
            "page_number": unit_embeddings[0][5],
            "is_sub_chunk": False,
        }]

    def _create_chunks_from_breakpoints(
        self,
        unit_embeddings: List[Tuple],
        breakpoints: List[int],
        max_chunk_size: int,
        chunk_overlap: int,
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from detected breakpoints with size limits.
        
        Args:
            unit_embeddings: List of embedding units
            breakpoints: List of breakpoint positions
            max_chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        start_idx = 0
        chunk_counter = 0

        # Process each breakpoint
        for breakpoint in breakpoints:
            chunk_counter += 1
            chunk_units = unit_embeddings[start_idx:breakpoint]
            
            if chunk_units:
                chunk_text = " ".join([u[0] for u in chunk_units])
                
                if len(chunk_text) > max_chunk_size:
                    # Split large chunk into sub-chunks
                    sub_chunks = self._split_large_chunk(
                        chunk_units, max_chunk_size, chunk_overlap, chunk_counter
                    )
                    chunks.extend(sub_chunks)
                else:
                    # Create regular chunk
                    chunk = self._create_chunk_from_units(chunk_units)
                    chunks.append(chunk)
                    
            start_idx = breakpoint

        # Handle remaining units after last breakpoint
        if start_idx < len(unit_embeddings):
            chunk_counter += 1
            chunk_units = unit_embeddings[start_idx:]
            
            if chunk_units:
                chunk_text = " ".join([u[0] for u in chunk_units])
                
                if len(chunk_text) > max_chunk_size:
                    sub_chunks = self._split_large_chunk(
                        chunk_units, max_chunk_size, chunk_overlap, chunk_counter
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunk = self._create_chunk_from_units(chunk_units)
                    chunks.append(chunk)

        return chunks

    def _create_chunk_from_units(self, chunk_units: List[Tuple]) -> Dict[str, Any]:
        """Create a chunk dictionary from embedding units."""
        chunk_text = " ".join([u[0] for u in chunk_units])
        total_sentences = sum(u[6] for u in chunk_units) if len(chunk_units[0]) > 6 else len(chunk_units)
        
        return {
            "text": chunk_text,
            "start_pos": chunk_units[0][1],
            "end_pos": chunk_units[-1][2],
            "start_line": chunk_units[0][3],
            "end_line": chunk_units[-1][4],
            "sentences": total_sentences,
            "page_number": chunk_units[0][5],
            "is_sub_chunk": False,
        }

    def _split_large_chunk(
        self,
        chunk_units: List[Tuple],
        max_chunk_size: int,
        chunk_overlap: int,
        parent_chunk_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Split a large chunk into smaller sub-chunks with overlap.
        
        Args:
            chunk_units: Units to split
            max_chunk_size: Maximum size per sub-chunk
            chunk_overlap: Overlap between sub-chunks
            parent_chunk_number: Parent chunk identifier
            
        Returns:
            List of sub-chunk dictionaries
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for unit in chunk_units:
            unit_size = len(unit[0])
            
            if current_size + unit_size > max_chunk_size and current_chunk:
                # Create sub-chunk
                chunk = self._create_sub_chunk(current_chunk, parent_chunk_number)
                chunks.append(chunk)

                # Calculate overlap units
                overlap_units = self._calculate_overlap_units(current_chunk, chunk_overlap)
                current_chunk = overlap_units + [unit]
                current_size = sum(len(u[0]) for u in current_chunk)
            else:
                current_chunk.append(unit)
                current_size += unit_size

        # Add final sub-chunk if units remain
        if current_chunk:
            chunk = self._create_sub_chunk(current_chunk, parent_chunk_number)
            chunks.append(chunk)

        return chunks

    def _create_sub_chunk(self, chunk_units: List[Tuple], parent_chunk: int) -> Dict[str, Any]:
        """Create a sub-chunk dictionary."""
        chunk_text = " ".join([u[0] for u in chunk_units])
        total_sentences = sum(u[6] for u in chunk_units) if len(chunk_units[0]) > 6 else len(chunk_units)
        
        return {
            "text": chunk_text,
            "start_pos": chunk_units[0][1],
            "end_pos": chunk_units[-1][2],
            "start_line": chunk_units[0][3],
            "end_line": chunk_units[-1][4],
            "sentences": total_sentences,
            "page_number": chunk_units[0][5],
            "is_sub_chunk": True,
            "parent_chunk": parent_chunk,
        }

    def _calculate_overlap_units(self, current_chunk: List[Tuple], chunk_overlap: int) -> List[Tuple]:
        """Calculate overlap units for chunk splitting."""
        overlap_units = []
        overlap_size = 0
        
        for unit in reversed(current_chunk):
            if overlap_size + len(unit[0]) <= chunk_overlap:
                overlap_units.insert(0, unit)
                overlap_size += len(unit[0])
                
        return overlap_units

    # ========================================================================
    # RESULT FORMATTING METHODS
    # ========================================================================

    def _filter_and_format_chunks(
        self,
        similar_chunks: List,
        min_cosine_similarity: float,
        rerank: bool,
        min_cross_score: float
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks by cross-encoder threshold and format for response.
        Note: Cosine similarity filtering is already done before this function.
        
        Args:
            similar_chunks: Pre-filtered chunks (already above cosine similarity threshold)
            min_cosine_similarity: Minimum cosine similarity threshold (for logging)
            rerank: Whether reranking was applied
            min_cross_score: Minimum cross-encoder score threshold
            
        Returns:
            List of formatted chunk dictionaries
        """
        LOG.info("📊 CROSS-ENCODER FILTERING & FORMATTING")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🎯 Threshold: {min_cross_score}")
        LOG.info(f"   │ 📊 Input chunks: {len(similar_chunks)}")
        
        chunks = []
        filtered_cross_encoder = 0

        for rank, chunk in enumerate(similar_chunks, 1):
            similarity_score = round(1 - chunk.distance, 4)

            # Apply cross-encoder threshold filter if reranking was used
            if rerank and hasattr(chunk, 'cross_score'):
                if chunk.cross_score >= min_cross_score:  # Higher score = more relevant
                    # Keep this chunk (score meets threshold)
                    pass
                else:
                    # Filter out this chunk (score below threshold)
                    filtered_cross_encoder += 1
                    continue

            # Format chunk data
            chunk_data = self._format_chunk_data(chunk, len(chunks) + 1, similarity_score, rerank)
            chunks.append(chunk_data)

            self._log_search_result(chunk, len(chunks), similarity_score)

        if rerank:
            LOG.info(f"   │ 🔽 Filtered out: {filtered_cross_encoder} chunks")
        LOG.info(f"   │ ✅ Final results: {len(chunks)} chunks")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        return chunks

    def _format_chunk_data(
        self,
        chunk,
        rank: int,
        similarity_score: float,
        rerank: bool
    ) -> Dict[str, Any]:
        """Format a single chunk for API response."""
        chunk_data = {
            "rank": rank,
            "cosine_similarity_score": similarity_score,
            "cross_encoder_score": round(chunk.cross_score, 4) if rerank and hasattr(chunk, 'cross_score') else None,
            "id": str(chunk.id),
            "filename": chunk.filename,
            "chunk_text": chunk.chunk_text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "created_at": chunk.created_at.isoformat(),
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "sentence_count": chunk.sentence_count,
        }

        return chunk_data



    # ========================================================================
    # LOGGING UTILITY METHODS
    # ========================================================================

    def _log_pdf_statistics(self, pdf_stats: Dict[str, Any]):
        """Log PDF structure analysis results with pretty formatting."""
        LOG.info("📊 PDF Structure Analysis:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📄 Total pages: {pdf_stats['total_pages']}")
        LOG.info(f"   │ 📝 Total characters: {pdf_stats['total_characters']}")
        LOG.info(f"   │ 📝 Total words: {pdf_stats['total_words']}")
        LOG.info(f"   │ 📏 Total lines: {pdf_stats['total_lines']}")
        LOG.info(f"   │ 🔤 Total sentences: {pdf_stats['total_sentences']}")
        LOG.info(f"   │ 📋 Total paragraphs: {pdf_stats['total_paragraphs']}")
        LOG.info(f"   │ 📏 Average words per sentence: {pdf_stats['avg_words_per_sentence']:.1f}")
        LOG.info(f"   │ 📏 Average sentences per paragraph: {pdf_stats['avg_sentences_per_paragraph']:.1f}")
        LOG.info(f"   │ 📏 Average words per line: {pdf_stats['avg_words_per_line']:.1f}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_chunking_results(
        self,
        chunks: List,
        chunking_mode: str,
        max_chunk_size: int,
        chunk_overlap: int,
        sentence_group_size: int
    ):
        """Log chunking process results with pretty formatting."""
        LOG.info("🎯 Chunking Results:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🔢 Total chunks created: {len(chunks)}")
        LOG.info(f"   │ 🎯 Chunking mode: {chunking_mode}")
        LOG.info(f"   │ 📏 Max chunk size: {max_chunk_size} characters")
        LOG.info(f"   │ 🔗 Chunk overlap: {chunk_overlap} characters")
        if chunking_mode == "sentence":
            LOG.info(f"   │ 📦 Sentence group size: {sentence_group_size}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_processing_summary(
        self,
        filename: str,
        chunks_created: int,
        chunking_mode: str,
        processing_time: float
    ):
        """Log final processing summary with pretty formatting."""
        avg_time_per_chunk = processing_time / chunks_created if chunks_created > 0 else 0
        
        LOG.info(f"⏱️ PDF processing completed in {processing_time:.2f} seconds")
        LOG.info("✅ PDF processing successful:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📄 Filename: {filename}")
        LOG.info(f"   │ 🔢 Chunks created: {chunks_created}")
        LOG.info(f"   │ 🎯 Chunking mode: {chunking_mode}")
        LOG.info(f"   │ 📊 Processing time: {processing_time:.2f}s")
        LOG.info(f"   │ 📈 Average time per chunk: {avg_time_per_chunk:.2f}s")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_search_parameters(
        self,
        query: str,
        limit: int,
        min_cosine_similarity: float,
        expand_query: bool,
        rerank: bool,
        min_cross_score: float
    ):
        """Log search parameters with pretty formatting."""
        LOG.info("🔍 SEARCH PARAMETERS")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📝 Query: '{query}'")
        LOG.info(f"   │ 📏 Length: {len(query)} characters")
        LOG.info("   ├─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🎯 Results limit: {limit}")
        LOG.info(f"   │ 📊 Min cosine similarity: {min_cosine_similarity}")
        LOG.info(f"   │ 🎯 Min cross-encoder score: {min_cross_score}")
        LOG.info("   ├─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🔄 Query expansion: {'✅ enabled' if expand_query else '❌ disabled'}")
        LOG.info(f"   │ 🔄 Cross-encoder: {'✅ enabled' if rerank else '❌ disabled'}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_search_completion(self, chunks: List, expand_query: bool):
        """Log search completion summary with pretty formatting."""
        LOG.info(f"🎉 Vector search completed successfully:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📊 Total results: {len(chunks)}")
        if expand_query:
            LOG.info(f"   │ ⏱️ Multi-query search completed")
        if chunks:
            LOG.info(f"   │ 🏆 Best cosine similarity: {chunks[0]['cosine_similarity_score']}")
            LOG.info(f"   │ 📉 Worst cosine similarity: {chunks[-1]['cosine_similarity_score']}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_semantic_analysis_results(
        self,
        distances: List[float],
        threshold: float,
        threshold_type: str,
        breakpoints: List[int]
    ):
        """Log semantic analysis results with pretty formatting."""
        distance_str = [f"{d:.3f}" for d in distances]
        LOG.info("🧮 Semantic Analysis Results:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📊 Calculated {len(distances)} sentence distances")
        LOG.info(f"   │ 📈 Distance values: {distance_str}")
        LOG.info(f"   │ 🎯 Using {threshold_type} threshold: {threshold:.4f}")
        LOG.info(f"   │ 📍 Found breakpoints at positions: {breakpoints}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_database_record(
        self,
        chunk_data: Dict,
        chunk_index_str: str,
        embedding: List[float],
        chunks_created: int,
        filename: str
    ):
        """Log database record insertion with pretty formatting."""
        chunk_preview = (
            chunk_data["text"][:100] + "..." if len(chunk_data["text"]) > 100 
            else chunk_data["text"]
        )
        embedding_preview = (
            f"[{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]"
            if embedding else "[]"
        )

        LOG.info(f"📝 RECORD #{chunks_created}:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📄 Filename: {filename}")
        LOG.info(f"   │ 🔢 Chunk Index: {chunk_index_str}")
        LOG.info(f"   │ 📄 Page Number: {chunk_data.get('page_number', 1)}")
        LOG.info(f"   │ 📏 Position: chars {chunk_data['start_pos']}-{chunk_data['end_pos']}")
        LOG.info(f"   │ 📏 Lines: {chunk_data['start_line']}-{chunk_data['end_line']}")
        LOG.info(f"   │ 📋 Sentences: {chunk_data['sentences']}")
        LOG.info(f"   │ 📝 Text Preview: {chunk_preview}")
        LOG.info(f"   │ 🧮 Embedding: {embedding_preview} (dim: {len(embedding) if embedding else 0})")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_database_summary(self, chunks_created: int, filename: str):
        """Log database insertion summary with pretty formatting."""
        LOG.info("💾 Database Insertion Summary:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ ✅ Successfully committed {chunks_created} chunks to database")
        LOG.info(f"   │ 📄 Filename: {filename}")
        LOG.info(f"   │ 🔢 Total records created: {chunks_created}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_search_result(self, chunk, rank: int, similarity_score: float):
        """Log individual search result with pretty formatting."""
        chunk_preview = (
            chunk.chunk_text[:100] + "..." if len(chunk.chunk_text) > 100 
            else chunk.chunk_text
        )

        LOG.info(f"📋 RESULT #{rank}:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 🏆 Rank: {rank}")
        LOG.info(f"   │ 📊 Cosine Similarity: {similarity_score}")
        LOG.info(f"   │ 📄 Filename: {chunk.filename}")
        LOG.info(f"   │ 🔢 Chunk Index: {chunk.chunk_index}")
        LOG.info(f"   │ 📄 Page: {chunk.page_number}")
        LOG.info(f"   │ 📏 Position: chars {chunk.start_pos}-{chunk.end_pos}, lines {chunk.start_line}-{chunk.end_line}")
        LOG.info(f"   │ 📋 Sentences: {chunk.sentence_count}")
        LOG.info(f"   │ 📝 Text Preview: {chunk_preview}")
        LOG.info("   └─────────────────────────────────────────────────────────────")

    def _log_upload_cost_summary(self, user_id: str, chunks_created: int):
        """Log total cost summary for PDF upload operation."""
        if user_id == "pdf_upload_user":
            try:
                from modules.analytics.service import TokenTrackingService
                from modules.RAG.database import sync_engine
                from sqlalchemy.orm import sessionmaker
                
                SessionLocal = sessionmaker(bind=sync_engine)
                db = SessionLocal()
                
                tracker = TokenTrackingService(db)
                analytics = tracker.get_user_analytics("upload_operations")
                db.close()
                
                total_operations = analytics.get("total_operations", 0)
                total_tokens = analytics.get("total_tokens", 0)
                total_cost = analytics.get("total_cost", 0.0)
                
                LOG.info("💰 UPLOAD COST SUMMARY:")
                LOG.info("   ┌─────────────────────────────────────────────────────────────")
                LOG.info(f"   │ 📊 Chunks Processed: {chunks_created:,}")
                LOG.info(f"   │ 🔢 Total API Calls: {total_operations:,}")
                LOG.info(f"   │ 📥 Total Tokens Used: {total_tokens:,}")
                LOG.info(f"   │ 💵 Total Estimated Cost: ${total_cost:.6f}")
                LOG.info(f"   │ 📈 Average Cost per Chunk: ${(total_cost/chunks_created):.6f}" if chunks_created > 0 else "   │ 📈 Average Cost per Chunk: $0.000000")
                LOG.info("   ├─────────────────────────────────────────────────────────────")
                LOG.info(f"   │ 💾 Stored in PostgreSQL: token_usage table")
                LOG.info(f"   │ 👤 User ID: upload_operations")
                LOG.info(f"   │ 🏷️ Operation Type: upload")
                LOG.info(f"   │ 🤖 Model Used: {CONFIG.openai_embedding_model}")
                LOG.info(f"   │ 💲 Input Cost: ${CONFIG.gpt4o_input_tokens_cost}/1M tokens")
                LOG.info(f"   │ 💲 Output Cost: ${CONFIG.gpt4o_output_tokens_cost}/1M tokens")
                LOG.info("   └─────────────────────────────────────────────────────────────")
                
            except Exception as e:
                LOG.error(f"Cost summary error: {e}")


# ========================================================================
# SERVICE INSTANCE
# ========================================================================

# Create singleton instance for application use
pdf_rag_service = PDFRAGService()