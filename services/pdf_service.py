"""PDF processing service."""

import uuid
from typing import Dict, List, Any

import pdfplumber

from config import settings
from llm_factory import get_embeddings


class PDFService:
    """Service for PDF text extraction and chunking."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file: bytes) -> str:
        """Extract text from PDF file bytes."""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def chunk_text(
        text: str, 
        chunk_size: int = None, 
        overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap for better context preservation.
        
        Args:
            text: Text to chunk
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of chunks with metadata
        """
        chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        overlap = overlap or settings.DEFAULT_CHUNK_OVERLAP
        
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            return [{"text": text, "chunk_index": 0}]
        
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_index
            })
            
            start += chunk_size - overlap
            chunk_index += 1
        
        return chunks
    
    @staticmethod
    def create_embeddings_for_chunks(
        chunks: List[Dict[str, Any]], 
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Create embeddings for text chunks using Sentence Transformers.
        
        Args:
            chunks: List of text chunks with metadata
            source: Source identifier for the document
            
        Returns:
            List of data dictionaries ready for LanceDB storage
        """
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Create embeddings via the embeddings API (client batches internally)
        embeddings = get_embeddings().embed_documents(texts)

        # Prepare data for LanceDB
        data = []
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i]
            doc_id = str(uuid.uuid4())
            
            data.append({
                "id": doc_id,
                "vector": embedding,
                "text": chunk["text"],
                "source": source,
                "chunk_index": chunk["chunk_index"]
            })
        
        return data

