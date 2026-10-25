"""API route handlers."""

import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query

from config import settings
from llm_factory import set_llm_provider
from models import (
    TranscriptRequest,
    PDFUploadResponse,
    AgentChatRequest,
    AgentChatResponse,
    TextIngestRequest,
)
from services.pdf_service import PDFService
from services.lancedb_service import LanceDBService
from services.clinical_service import run_clinical_supervisor_report
from services.agent_chat_service import CLINICAL_AGENT_CATALOG, run_single_agent_chat
from services.webhook_event_file import save_analyze_transcript_event

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()


@router.get("/clinical_agents")
async def list_clinical_agents() -> Dict[str, Any]:
    """Ids, titles, and short summaries for dashboard 'AI Assistants' chat picker."""
    return {"agents": CLINICAL_AGENT_CATALOG}


@router.post("/agent_chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    """
    Send one message to a single specialized agent (diagnosis_agent, rag_agent, etc.).

    For full multi-agent orchestration and JSON reports, use POST /analyze_transcript.
    """
    set_llm_provider(req.llm_provider)
    history_tuples = [(t.role, t.content) for t in req.history]
    try:
        reply = run_single_agent_chat(
            agent_id=req.agent_id,
            user_message=req.message,
            history=history_tuples,
            provider=req.llm_provider,
        )
        return AgentChatResponse(reply=reply, agent_id=req.agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent chat failed: {str(e)}",
        ) from e

# Initialize services
pdf_service = PDFService()
lancedb_service = LanceDBService()


@router.post("/analyze_transcript")
async def analyze_transcript(req: TranscriptRequest) -> Dict[str, Any]:
    """
    Analyze a patient transcript using the supervisor workflow.

    The supervisor will coordinate all clinical agents to generate a comprehensive report.
    Returns a structured JSON report matching the specified format.

    Use `llm_provider` in the request body to choose between 'openai' (default) and 'gemini'.
    """
    transcript = req.transcript
    userid = req.userid
    location = req.location or ""
    provider = req.llm_provider

    set_llm_provider(provider)

    try:
        parsed_response, _ = run_clinical_supervisor_report(
            transcript=transcript,
            userid=userid,
            location=location,
            provider=provider,
        )

        try:
            rel = save_analyze_transcript_event(
                userid=userid,
                provider=provider,
                location=location,
                transcript=transcript,
                report=dict(parsed_response),
            )
            parsed_response["saved_event_file"] = rel
        except OSError as e:
            logger.warning("Could not save analyze_transcript event to disk: %s", e)

        return parsed_response

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supervisor workflow failed: {str(e)}",
        )


@router.post("/upload_pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    source: str = Form(default="uploaded_pdf"),
    chunk_size: int = Form(default=None),
    overlap: int = Form(default=None)
) -> PDFUploadResponse:
    """
    Upload a PDF file for RAG (Retrieval Augmented Generation).
    
    The PDF will be processed, chunked, embedded, and stored in LanceDB.
    
    Parameters:
    - file: PDF file to upload
    - source: Source identifier for the document (default: "uploaded_pdf")
    - chunk_size: Number of words per chunk (default: from settings)
    - overlap: Number of overlapping words between chunks (default: from settings)
    
    Returns:
    - success: Whether the upload was successful
    - message: Status message
    - document_id: Unique identifier for the document
    - chunks_added: Number of chunks added to the database
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read PDF file
        pdf_bytes = await file.read()
        
        # Extract text from PDF
        text = pdf_service.extract_text_from_pdf(pdf_bytes)
        
        if not text or len(text.strip()) < settings.MIN_PDF_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="PDF appears to be empty or contains no extractable text"
            )
        
        # Chunk the text
        chunks = pdf_service.chunk_text(
            text, 
            chunk_size=chunk_size or settings.DEFAULT_CHUNK_SIZE,
            overlap=overlap or settings.DEFAULT_CHUNK_OVERLAP
        )
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to create text chunks from PDF"
            )
        
        # Create embeddings and prepare data
        data = pdf_service.create_embeddings_for_chunks(chunks, source=source)
        
        # Store in LanceDB
        chunks_added = lancedb_service.store_embeddings(data)
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        return PDFUploadResponse(
            success=True,
            message=f"Successfully processed PDF: {file.filename}. Added {chunks_added} chunks to RAG database.",
            document_id=document_id,
            chunks_added=chunks_added
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@router.post("/rag/ingest_text", response_model=PDFUploadResponse)
async def rag_ingest_text(req: TextIngestRequest) -> PDFUploadResponse:
    """
    Ingest plain text or markdown into LanceDB using the same chunk + embed pipeline as PDF upload.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty")
    if len(text) < settings.MIN_PDF_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too short (minimum {settings.MIN_PDF_TEXT_LENGTH} characters)",
        )
    try:
        chunks = pdf_service.chunk_text(
            text,
            chunk_size=req.chunk_size or settings.DEFAULT_CHUNK_SIZE,
            overlap=req.overlap or settings.DEFAULT_CHUNK_OVERLAP,
        )
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to chunk text")
        data = pdf_service.create_embeddings_for_chunks(chunks, source=req.source.strip() or "pasted_doc")
        chunks_added = lancedb_service.store_embeddings(data)
        document_id = str(uuid.uuid4())
        return PDFUploadResponse(
            success=True,
            message=f"Added {chunks_added} chunks to knowledge base '{req.source}'.",
            document_id=document_id,
            chunks_added=chunks_added,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rag_ingest_text failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to ingest text: {str(e)}") from e


@router.get("/rag/sources")
async def rag_list_sources() -> Dict[str, Any]:
    """Distinct ``source`` labels currently stored in LanceDB."""
    try:
        sources = lancedb_service.list_sources()
        return {"sources": sources}
    except Exception as e:
        logger.exception("rag_list_sources failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/rag/search")
async def rag_search_lance(
    query: str = Query(..., min_length=1, description="Natural-language or keyword query"),
    top_k: int = Query(5, ge=1, le=50, description="Max hits to return"),
    source: Optional[str] = Query(
        None,
        description="If set, only chunks with this knowledge-base label are kept",
    ),
) -> Dict[str, Any]:
    """Semantic search over the LanceDB RAG table (same store as /upload_pdf and /rag/ingest_text)."""
    try:
        hits = lancedb_service.search_by_text(
            query.strip(),
            top_k=top_k,
            source=source.strip() if source else None,
        )
        return {
            "query": query.strip(),
            "top_k": top_k,
            "source_filter": source.strip() if source else None,
            "results": hits,
            "count": len(hits),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("rag_search_lance failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

