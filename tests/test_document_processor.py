import pytest
from app.services.document_processor import document_processor
from app.core.exceptions import DocumentProcessingError


def test_chunking_with_markdown_sections():
    markdown = """
# System Design

## Architecture
The system consists of API gateway and worker services.

## Databases
Database queries are optimized using indexing and connection pools.
"""
    chunks = document_processor.process_document("doc_sys", "System Design", markdown)
    assert len(chunks) >= 2
    assert any(c["metadata"]["section_title"] == "Databases" for c in chunks)
    assert any("connection pools" in c["content"] for c in chunks)


def test_empty_content_raises_exception():
    with pytest.raises(DocumentProcessingError):
        document_processor.process_document("doc_empty", "Empty", "   ")
