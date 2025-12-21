"""Document Loading with Azure Document Intelligence OCR Support."""

import logging
from pathlib import Path
from typing import List, Tuple

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.exceptions import DocumentProcessingError, OCRProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Document(BaseModel):
    """Document model with validation."""

    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    chunks: List[str] = Field(default_factory=list)
    is_scanned: bool = False
    extraction_method: str = "direct"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def extract_text_with_azure_ocr(pdf_path: str) -> Tuple[str, bool]:
    """
    Extract text using Azure Document Intelligence (Mistral Document AI).

    Returns:
        Tuple of (extracted_text, is_scanned)
    """
    # First attempt: Direct text extraction
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n\n".join([page.page_content for page in pages])

        # Validate text extraction quality
        if text.strip() and len(text.strip()) > 100:
            logger.info("direct_extraction_successful", file_path=pdf_path)
            return text, False
    except Exception as e:
        logger.debug("direct_extraction_failed", file_path=pdf_path, error=str(e))

    # Fallback: Azure Document Intelligence OCR
    logger.info("using_azure_ocr", file_path=pdf_path)

    try:
        # Initialize Azure client
        client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_MISTRAL_OCR_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_API_KEY),
        )

        # Read PDF file
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        # Analyze document with Mistral model
        poller = client.begin_analyze_document(
            model_id=settings.AZURE_MISTRAL_MODEL,
            analyze_request=pdf_data,
            content_type="application/pdf",
        )

        result = poller.result()

        # Extract text from result
        extracted_text = ""
        if hasattr(result, "content") and result.content:
            extracted_text = result.content
        elif hasattr(result, "pages"):
            # Extract from pages
            text_parts = []
            for page in result.pages:
                if hasattr(page, "lines"):
                    for line in page.lines:
                        if hasattr(line, "content"):
                            text_parts.append(line.content)
            extracted_text = "\n".join(text_parts)

        if not extracted_text.strip():
            raise ValueError("Azure OCR returned empty text")

        logger.info(
            "azure_ocr_successful",
            file_path=pdf_path,
            text_length=len(extracted_text),
        )
        return extracted_text, True

    except AzureError as e:
        logger.error("azure_ocr_error", file_path=pdf_path, error=str(e))
        raise OCRProcessingError(
            f"Azure Document Intelligence error: {str(e)}",
            details={"file_path": pdf_path, "error_type": type(e).__name__},
        )
    except Exception as e:
        logger.error("ocr_processing_failed", file_path=pdf_path, error=str(e))
        raise OCRProcessingError(
            f"OCR processing failed: {str(e)}",
            details={"file_path": pdf_path, "error_type": type(e).__name__},
        )


def load_pdf_documents(
    file_paths: List[str],
    use_ocr: bool = True,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Load and chunk PDF documents with production-grade error handling.

    Args:
        file_paths: List of PDF file paths
        use_ocr: Whether to use OCR for scanned documents
        chunk_size: Override default chunk size
        chunk_overlap: Override default chunk overlap

    Returns:
        List of Document objects
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []

    for file_path in file_paths:
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning("file_not_found", file_path=str(file_path))
            continue

        if not file_path.suffix.lower() == ".pdf":
            logger.warning("not_pdf_file", file_path=str(file_path))
            continue

        try:
            # Extract text
            if use_ocr:
                full_text, is_scanned = extract_text_with_azure_ocr(str(file_path))
            else:
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                full_text = "\n\n".join([page.page_content for page in pages])
                is_scanned = False

            if not full_text.strip():
                logger.warning("no_text_extracted", file_path=str(file_path))
                continue

            # Split into chunks
            chunks = text_splitter.split_text(full_text)

            if not chunks:
                logger.warning("no_chunks_created", file_path=str(file_path))
                continue

            # Create document
            doc = Document(
                id=file_path.stem,
                content=full_text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "total_chunks": len(chunks),
                    "is_scanned": is_scanned,
                    "extraction_method": (
                        "Azure Document Intelligence" if is_scanned else "direct"
                    ),
                    "file_size": file_path.stat().st_size,
                },
                chunks=chunks,
                is_scanned=is_scanned,
                extraction_method=(
                    "Azure Document Intelligence" if is_scanned else "direct"
                ),
            )

            documents.append(doc)
            logger.info(
                "document_loaded",
                file_path=str(file_path),
                chunks=len(chunks),
                method=doc.extraction_method,
            )

        except Exception as e:
            logger.error(
                "document_loading_failed",
                file_path=str(file_path),
                error=str(e),
                exc_info=True,
            )
            continue

    return documents

