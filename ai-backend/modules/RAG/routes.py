"""
PDF RAG API Routes

FastAPI routes for PDF processing, semantic chunking, vector search,
and intelligent document retrieval with comprehensive error handling.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from .schema import UploadResponse
from .service import pdf_rag_service
from core.logger.logger import LOG
import time

# ========================================================================
# ROUTER CONFIGURATION
# ========================================================================

API_ROUTER = APIRouter(prefix="/api/v1/pdf-rag", tags=["PDF RAG"])

# ========================================================================
# PDF UPLOAD AND PROCESSING ENDPOINTS
# ========================================================================

@API_ROUTER.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and process PDF with semantic chunking",
    description="Upload a PDF file, extract text, create semantic chunks, and generate embeddings for intelligent retrieval."
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload and process"),
    chunking_mode: str = Query(
        default="sentence",
        description="Chunking mode: 'sentence' for semantic analysis or 'length' for fixed-size chunks"
    ),
    max_chunk_size: int = Query(
        default=1500,
        description="Maximum characters per chunk (applies to both modes)"
    ),
    breakpoint_threshold_type: str = Query(
        default="percentile",
        description="Statistical threshold type for semantic chunking: 'percentile', 'standard_deviation', or 'interquartile'"
    ),
    breakpoint_threshold_amount: float = Query(
        default=None,
        description="Threshold amount: 95.0 (percentile), 3.0 (std dev), 1.5 (interquartile)"
    ),
    chunk_overlap: int = Query(
        default=200,
        description="Number of characters to overlap between consecutive chunks"
    ),
    sentence_group_size: int = Query(
        default=3,
        description="Number of sentences to group before semantic comparison (sentence mode only)"
    )
):
    """
    Upload and process a PDF file with intelligent semantic chunking.
    
    This endpoint:
    1. Validates the uploaded file (PDF format)
    2. Extracts text with page number mapping
    3. Analyzes document structure (pages, sentences, paragraphs)
    4. Creates semantic chunks using statistical similarity analysis
    5. Generates embeddings for each chunk
    6. Stores chunks in vector database for retrieval
    
    Args:
        file: PDF file to process
        chunking_mode: Semantic or length-based chunking
        max_chunk_size: Maximum characters per chunk
        breakpoint_threshold_type: Statistical method for detecting chunk boundaries
        breakpoint_threshold_amount: Threshold value for the selected method
        chunk_overlap: Character overlap between chunks
        sentence_group_size: Sentences per group for semantic analysis
        
    Returns:
        UploadResponse with processing results and statistics
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    # Log request details
    LOG.info("📁 PDF UPLOAD REQUEST RECEIVED")
    LOG.info("   ┌─────────────────────────────────────────────────────────────")
    LOG.info(f"   │ 📄 Filename: '{file.filename}'")
    LOG.info(f"   │ 📏 File Size: {file.size:,} bytes")
    LOG.info("   ├─────────────────────────────────────────────────────────────")
    LOG.info(f"   │ 🎯 Chunking Mode: {chunking_mode}")
    LOG.info(f"   │ 📏 Max Chunk Size: {max_chunk_size:,} characters")
    LOG.info(f"   │ 📊 Threshold Type: {breakpoint_threshold_type}")
    LOG.info(f"   │ 📊 Threshold Amount: {breakpoint_threshold_amount or 'default'}")
    LOG.info(f"   │ 🔗 Chunk Overlap: {chunk_overlap:,} characters")
    if chunking_mode == "sentence":
        LOG.info(f"   │ 📦 Sentence Group Size: {sentence_group_size}")
    LOG.info("   └─────────────────────────────────────────────────────────────")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        LOG.error(f"❌ Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a valid PDF document."
        )
    
    # Validate chunking mode
    if chunking_mode.lower() not in ["sentence", "length"]:
        LOG.error(f"❌ Invalid chunking mode: {chunking_mode}")
        raise HTTPException(
            status_code=400,
            detail="chunking_mode must be either 'sentence' or 'length'"
        )
    
    # Validate threshold type for sentence mode
    if chunking_mode.lower() == "sentence":
        valid_thresholds = ["percentile", "standard_deviation", "interquartile"]
        if breakpoint_threshold_type not in valid_thresholds:
            LOG.error(f"❌ Invalid threshold type: {breakpoint_threshold_type}")
            raise HTTPException(
                status_code=400,
                detail=f"breakpoint_threshold_type must be one of: {', '.join(valid_thresholds)}"
            )
    
    try:
        # Read file content
        LOG.info(f"📖 Reading PDF file: {file.filename}")
        content = await file.read()
        LOG.info(f"✅ File read successfully - Size: {len(content)} bytes")
        
        # Process PDF with semantic chunking
        LOG.info(f"🔧 Starting PDF processing with {chunking_mode} chunking mode")
        start_time = time.time()
        
        result = await pdf_rag_service.process_pdf(
            content, 
            file.filename, 
            chunking_mode=chunking_mode,
            max_chunk_size=max_chunk_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            chunk_overlap=chunk_overlap,
            sentence_group_size=sentence_group_size
        )
        
        processing_time = time.time() - start_time
        
        # Check for processing errors
        if "error" in result:
            LOG.error(f"❌ PDF processing failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Log successful processing summary
        LOG.info(f"🎉 PDF processing completed successfully:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ 📄 Filename: {result['filename']}")
        LOG.info(f"   │ 🔢 Chunks created: {result['chunks_created']}")
        LOG.info(f"   │ 🎯 Chunking mode: {result['chunking_mode']}")
        LOG.info(f"   │ ⏱️ Total processing time: {processing_time:.2f}s")
        if result['chunks_created'] > 0:
            avg_time = processing_time / result['chunks_created']
            LOG.info(f"   │ 📈 Average time per chunk: {avg_time:.2f}s")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        return UploadResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors with full details
        LOG.error(f"❌ Unexpected error processing PDF {file.filename}:")
        LOG.error("   ┌─────────────────────────────────────────────────────────────")
        LOG.error(f"   │ 📄 Filename: {file.filename}")
        LOG.error(f"   │ ❌ Error: {str(e)}")
        LOG.error(f"   │ 🔍 Error type: {type(e).__name__}")
        LOG.error(f"   │ 📍 Error details: {e}")
        LOG.error("   └─────────────────────────────────────────────────────────────")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )

# ========================================================================
# SEARCH ENDPOINTS
# ========================================================================

@API_ROUTER.post(
    "/search",
    summary="Pure vector search without LLM",
    description="Perform vector similarity search to find relevant document chunks without AI answer generation."
)
async def search_chunks(
    query: str = Query(..., description="Search query to find relevant chunks"),
    limit: int = Query(
        default=10,
        description="Maximum number of chunks to return"
    ),
    min_cosine_similarity: float = Query(
        default=0.5,
        description="Minimum cosine similarity threshold (0.0-1.0) for initial filtering"
    ),
    min_cross_score: float = Query(
        default=0.0,
        description="Minimum cross-encoder score threshold (lower = more relevant, e.g., 0.0)"
    ),
    expand_query: bool = Query(
        default=True,
        description="Enable query expansion using LLM for better search coverage"
    ),
    rerank: bool = Query(
        default=True,
        description="Enable cross-encoder reranking and filtering for improved relevance"
    )
):
    """
    Perform pure vector search to find relevant document chunks.
    
    This endpoint:
    1. Generates embedding for the search query
    2. Optionally expands query with related terms using LLM
    3. Finds similar chunks using cosine similarity
    4. Pre-filters chunks by similarity threshold
    5. Optionally reranks results using cross-encoder model
    6. Filters results by cross-encoder score threshold
    7. Returns ranked chunks with metadata
    
    Args:
        query: Search query string
        limit: Maximum results to return
        min_cosine_similarity: Minimum cosine similarity score threshold
        min_cross_score: Minimum cross-encoder score threshold (lower = more relevant)
        expand_query: Enable multi-query expansion
        rerank: Enable cross-encoder reranking and filtering
        
    Returns:
        Search results with ranked chunks and metadata
        
    Raises:
        HTTPException: For search processing failures
    """
    LOG.info(f"🔍 Vector Search Request:")
    LOG.info("   ┌─────────────────────────────────────────────────────────────")
    LOG.info(f"   │ 📝 Query: {query}")
    LOG.info(f"   │ 🔢 Limit: {limit} chunks")
    LOG.info(f"   │ 📊 Min cosine similarity: {min_cosine_similarity}")
    LOG.info(f"   │ 🎯 Min cross-encoder score: {min_cross_score}")
    LOG.info(f"   │ 🔄 Query expansion: {'enabled' if expand_query else 'disabled'}")
    LOG.info(f"   │ 🔄 Cross-encoder reranking: {'enabled' if rerank else 'disabled'}")
    LOG.info("   └─────────────────────────────────────────────────────────────")
    
    try:
        start_time = time.time()
        result = await pdf_rag_service.vector_search_only(
            query, limit, min_cosine_similarity, min_cross_score, expand_query, rerank
        )
        search_time = time.time() - start_time
        
        # Log search completion summary
        LOG.info(f"✅ Vector Search Completed:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info(f"   │ ⏱️ Search time: {search_time:.3f} seconds")
        LOG.info(f"   │ 📊 Results found: {result['total_results']}")
        LOG.info(f"   │ 🎯 Distance metric: {result['distance_metric']}")
        if result['total_results'] > 0:
            best_score = result['chunks'][0]['cosine_similarity_score']
            worst_score = result['chunks'][-1]['cosine_similarity_score']
            LOG.info(f"   │ 🏆 Best cosine similarity score: {best_score}")
            LOG.info(f"   │ 📉 Worst cosine similarity score: {worst_score}")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        return result
        
    except Exception as e:
        LOG.error(f"❌ Vector search failed:")
        LOG.error("   ┌─────────────────────────────────────────────────────────────")
        LOG.error(f"   │ 📝 Query: {query}")
        LOG.error(f"   │ ❌ Error: {str(e)}")
        LOG.error(f"   │ 🔍 Error type: {type(e).__name__}")
        LOG.error("   └─────────────────────────────────────────────────────────────")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

# ========================================================================
# DATABASE MANAGEMENT ENDPOINTS
# ========================================================================

@API_ROUTER.delete(
    "/clear",
    summary="Clear all PDF data from database",
    description="Remove all uploaded PDFs, chunks, and embeddings from the system."
)
async def clear_database():
    """
    Clear all PDF data from the database.
    
    This endpoint:
    1. Removes all PDF chunks and embeddings
    2. Clears the vector database
    3. Resets the system to initial state
    
    Returns:
        Success message confirming data clearance
        
    Raises:
        HTTPException: For database operation failures
    """
    LOG.info("🗑️ Database Clear Request:")
    LOG.info("   ┌─────────────────────────────────────────────────────────────")
    LOG.info("   │ ⚠️  WARNING: This will delete ALL PDF data")
    LOG.info("   │ 🗑️  All chunks, embeddings, and metadata will be removed")
    LOG.info("   └─────────────────────────────────────────────────────────────")
    
    try:
        start_time = time.time()
        await pdf_rag_service.clear_database()
        clear_time = time.time() - start_time
        
        LOG.info("✅ Database cleared successfully:")
        LOG.info("   ┌─────────────────────────────────────────────────────────────")
        LOG.info("   │ 🗑️  All PDF data removed")
        LOG.info(f"   │ ⏱️  Clear operation time: {clear_time:.3f}s")
        LOG.info("   └─────────────────────────────────────────────────────────────")
        
        return {
            "message": "All PDF data cleared successfully",
            "operation_time": f"{clear_time:.3f}s"
        }
        
    except Exception as e:
        LOG.error(f"❌ Database clear failed:")
        LOG.error("   ┌─────────────────────────────────────────────────────────────")
        LOG.error(f"   │ ❌ Error: {str(e)}")
        LOG.error(f"   │ 🔍 Error type: {type(e).__name__}")
        LOG.error("   └─────────────────────────────────────────────────────────────")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear database: {str(e)}"
        )

# ========================================================================
# HEALTH CHECK ENDPOINT
# ========================================================================

@API_ROUTER.get(
    "/health",
    summary="Service health check",
    description="Check if the PDF RAG service is operational and ready to process requests."
)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        Service status information
    """
    return {
        "status": "healthy",
        "service": "PDF RAG API",
        "version": "1.0.0",
        "description": "PDF Retrieval-Augmented Generation Service"
    } 