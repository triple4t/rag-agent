"""Document Management Endpoints."""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.exceptions import DocumentProcessingError, NotFoundError
from app.core.logging import get_logger
from app.optimization.caching import cache_manager
from app.utils.document_loader import load_pdf_documents

logger = get_logger(__name__)
router = APIRouter()

# In-memory document storage (in production, use database)
documents_store = {}


class DocumentResponse(BaseModel):
    """Document response model."""

    id: str
    filename: str
    total_chunks: int
    is_scanned: bool
    extraction_method: str
    file_size: int


@router.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents."""
    try:
        # Save uploaded files temporarily
        file_paths = []
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400, detail=f"{file.filename} is not a PDF file"
                )

            # Save to temporary location
            import tempfile
            import os

            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, file.filename)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            file_paths.append(file_path)

        # Load and process documents
        documents = load_pdf_documents(file_paths)

        if not documents:
            raise DocumentProcessingError("No documents were successfully processed")

        # Store documents
        responses = []
        for doc in documents:
            documents_store[doc.id] = doc
            responses.append(
                DocumentResponse(
                    id=doc.id,
                    filename=doc.metadata.get("filename", doc.id),
                    total_chunks=len(doc.chunks),
                    is_scanned=doc.is_scanned,
                    extraction_method=doc.extraction_method,
                    file_size=doc.metadata.get("file_size", 0),
                )
            )

        logger.info("documents_uploaded", count=len(documents))
        
        # Clear cache when new documents are uploaded to avoid stale results
        # The search engine will be reinitialized on next query (via document count check)
        cache_manager.clear()
        logger.info("cache_cleared_after_document_upload")
        
        return responses

    except Exception as e:
        logger.error("document_upload_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    """List all loaded documents."""
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.metadata.get("filename", doc.id),
            total_chunks=len(doc.chunks),
            is_scanned=doc.is_scanned,
            extraction_method=doc.extraction_method,
            file_size=doc.metadata.get("file_size", 0),
        )
        for doc in documents_store.values()
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get document details."""
    if document_id not in documents_store:
        raise NotFoundError(f"Document {document_id} not found")

    doc = documents_store[document_id]
    return DocumentResponse(
        id=doc.id,
        filename=doc.metadata.get("filename", doc.id),
        total_chunks=len(doc.chunks),
        is_scanned=doc.is_scanned,
        extraction_method=doc.extraction_method,
        file_size=doc.metadata.get("file_size", 0),
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Remove document."""
    if document_id not in documents_store:
        raise NotFoundError(f"Document {document_id} not found")

    del documents_store[document_id]
    logger.info("document_deleted", document_id=document_id)
    return {"message": f"Document {document_id} deleted"}


def get_all_documents():
    """Get all documents (for internal use)."""
    return list(documents_store.values())

