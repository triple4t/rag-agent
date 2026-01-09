"""Document Loading with Azure Document Intelligence OCR Support and Image Processing."""

import base64
import logging
import re
from pathlib import Path
from typing import List, Tuple

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.exceptions import DocumentProcessingError, OCRProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def normalize_text(text: str) -> str:
    """
    Normalize text by removing excessive spaces between characters.
    
    Fixes issues like "7A d v a n c e d" -> "7Advanced"
    """
    if not text:
        return text
    
    # Remove spaces between single characters (but preserve word boundaries)
    # Pattern: single char + space(s) + single char -> merge
    # But be careful not to break normal word spacing
    
    # First, fix obvious character-level spacing issues
    # Match pattern like "7 A d v a n c e d" where each char is separated
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1\2\3\4', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1\2\3', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1\2', text)
    
    # Remove multiple spaces (normalize to single space)
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
    
    return text.strip()


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
            body=pdf_data,
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
            # Extract text with page tracking
            page_texts = []  # List of (page_num, text) tuples
            if use_ocr:
                full_text, is_scanned = extract_text_with_azure_ocr(str(file_path))
                # For OCR, we don't have easy page info, so estimate based on length
                # Approximate: assume ~2000 chars per page
                estimated_pages = max(1, len(full_text) // 2000)
                page_texts = [(i + 1, full_text) for i in range(estimated_pages)]
            else:
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                page_texts = [(i + 1, page.page_content) for i, page in enumerate(pages)]
                full_text = "\n\n".join([text for _, text in page_texts])
                is_scanned = False

            if not full_text.strip():
                logger.warning("no_text_extracted", file_path=str(file_path))
                continue

            # Normalize page texts first
            normalized_page_texts = [(page_num, normalize_text(text)) for page_num, text in page_texts]
            
            # Rebuild full_text from normalized pages
            full_text = "\n\n".join([text for _, text in normalized_page_texts])
            
            # Normalize the full text (in case concatenation introduced issues)
            full_text = normalize_text(full_text)

            # Split into chunks
            chunks = text_splitter.split_text(full_text)
            
            # Map chunks to page numbers by finding which page contains the chunk
            # This is approximate but works better than character position matching
            chunk_pages = []
            for chunk in chunks:
                # Normalize chunk for comparison
                normalized_chunk = normalize_text(chunk)
                # Take first 100 chars of chunk for matching (to avoid issues with splits)
                chunk_sample = normalized_chunk[:100] if len(normalized_chunk) > 100 else normalized_chunk
                
                page_num = 1  # default
                best_match_length = 0
                
                # Find the page that contains the most of this chunk
                for p_num, page_text in normalized_page_texts:
                    if chunk_sample in page_text:
                        # This page contains the chunk, use it
                        page_num = p_num
                        break
                    # Check for partial matches
                    match_length = 0
                    for i in range(len(chunk_sample)):
                        if i < len(page_text) and chunk_sample[:i+1] in page_text:
                            match_length = i + 1
                    if match_length > best_match_length:
                        best_match_length = match_length
                        page_num = p_num
                
                chunk_pages.append(page_num)

            if not chunks:
                logger.warning("no_chunks_created", file_path=str(file_path))
                continue

            # Create document with page mapping
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
                    "chunk_pages": chunk_pages,  # Store page mapping for each chunk
                    "total_pages": len(page_texts),
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text/description from image using OpenAI Vision API.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Text description of the image
    """
    try:
        # Read image and encode to base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image MIME type from extension
        image_ext = Path(image_path).suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
            '.heic': 'image/heic',
            '.heif': 'image/heif',
        }
        mime_type = mime_types.get(image_ext, 'image/png')
        
        # Create data URL
        image_url = f"data:{mime_type};base64,{base64_image}"
        
        # Initialize OpenAI LLM with vision support
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,  # gpt-4o supports vision
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.REQUEST_TIMEOUT,
            max_retries=settings.MAX_RETRIES,
        )
        
        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please provide a detailed description of this image. Include any text, objects, people, scenes, diagrams, charts, or other content visible in the image. Be thorough and descriptive."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        )
        
        # Get description from vision API
        response = llm.invoke([message])
        description = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(
            "image_processed_with_vision",
            file_path=image_path,
            description_length=len(description),
        )
        
        return description
        
    except Exception as e:
        logger.error("image_processing_failed", file_path=image_path, error=str(e), exc_info=True)
        raise DocumentProcessingError(
            f"Failed to process image: {str(e)}",
            details={"file_path": image_path, "error_type": type(e).__name__},
        )


def load_documents(
    file_paths: List[str],
    use_ocr: bool = True,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Load and process both PDF documents and images.
    
    Args:
        file_paths: List of file paths (PDFs or images)
        use_ocr: Whether to use OCR for scanned PDFs
        chunk_size: Override default chunk size
        chunk_overlap: Override default chunk overlap
    
    Returns:
        List of Document objects
    """
    documents = []
    pdf_paths = []
    image_paths = []
    
    # Separate PDFs and images
    for file_path in file_paths:
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()
        
        if suffix == ".pdf":
            pdf_paths.append(file_path)
        elif suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".heic", ".heif"]:
            image_paths.append(file_path)
        else:
            logger.warning("unsupported_file_type", file_path=file_path, suffix=suffix)
    
    # Process PDFs
    if pdf_paths:
        pdf_docs = load_pdf_documents(pdf_paths, use_ocr, chunk_size, chunk_overlap)
        documents.extend(pdf_docs)
    
    # Process images
    if image_paths:
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        for image_path in image_paths:
            image_path_obj = Path(image_path)
            
            if not image_path_obj.exists():
                logger.warning("image_not_found", file_path=str(image_path))
                continue
            
            try:
                # Extract description from image using vision API
                description = extract_text_from_image(str(image_path))
                
                if not description.strip():
                    logger.warning("no_description_extracted", file_path=str(image_path))
                    continue
                
                # Normalize description
                normalized_description = normalize_text(description)
                
                # Split into chunks (though images typically have one chunk)
                chunks = text_splitter.split_text(normalized_description)
                
                if not chunks:
                    chunks = [normalized_description]  # At least one chunk
                
                # Store image as base64 for future vision API queries
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # Determine image MIME type
                image_ext = image_path_obj.suffix.lower()
                mime_types = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp',
                    '.bmp': 'image/bmp',
                    '.svg': 'image/svg+xml',
                    '.heic': 'image/heic',
                    '.heif': 'image/heif',
                }
                mime_type = mime_types.get(image_ext, 'image/png')
                image_data_url = f"data:{mime_type};base64,{base64_image}"
                
                # Create document for image
                doc = Document(
                    id=image_path_obj.stem,
                    content=normalized_description,
                    metadata={
                        "source": str(image_path),
                        "filename": image_path_obj.name,
                        "total_chunks": len(chunks),
                        "is_scanned": False,
                        "extraction_method": "OpenAI Vision API",
                        "file_size": image_path_obj.stat().st_size,
                        "file_type": "image",
                        "image_data_url": image_data_url,  # Store for vision API queries
                        "chunk_pages": [1] * len(chunks),  # Images don't have pages
                        "total_pages": 1,
                    },
                    chunks=chunks,
                    is_scanned=False,
                    extraction_method="OpenAI Vision API",
                )
                
                documents.append(doc)
                logger.info(
                    "image_loaded",
                    file_path=str(image_path),
                    chunks=len(chunks),
                )
                
            except Exception as e:
                logger.error(
                    "image_loading_failed",
                    file_path=str(image_path),
                    error=str(e),
                    exc_info=True,
                )
                continue
    
    return documents

