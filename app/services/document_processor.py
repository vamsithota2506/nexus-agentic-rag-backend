import re
from typing import List, Dict, Any
from app.config import settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import logger


class DocumentProcessor:
    """
    Intelligent document chunking and metadata extractor.
    Splits documents recursively along markdown headings, paragraphs, and sentences,
    preserving context and attaching section metadata to each chunk.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def process_document(self, doc_id: str, title: str, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not content.strip():
            raise DocumentProcessingError(f"Document content for doc_id '{doc_id}' is empty.")

        sections = self._split_by_sections(content)
        chunks = []
        chunk_index = 0

        for section_title, section_text in sections:
            section_chunks = self._recursive_chunk_text(section_text, self.chunk_size, self.chunk_overlap)
            for chunk_str in section_chunks:
                chunk_index += 1
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
                chunk_meta = {
                    **(metadata or {}),
                    "doc_id": doc_id,
                    "document_title": title,
                    "section_title": section_title,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_str),
                    "estimated_tokens": len(chunk_str.split()),
                }
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": chunk_str.strip(),
                    "metadata": chunk_meta
                })

        logger.info(f"Processed doc '{doc_id}' into {len(chunks)} chunks.")
        return chunks

    def _split_by_sections(self, content: str) -> List[tuple]:
        """Splits markdown text into (header_title, body) tuples."""
        lines = content.split("\n")
        sections = []
        current_title = "Overview"
        current_buffer = []

        for line in lines:
            header_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if header_match:
                if current_buffer:
                    sections.append((current_title, "\n".join(current_buffer)))
                    current_buffer = []
                current_title = header_match.group(2).strip()
            current_buffer.append(line)

        if current_buffer:
            sections.append((current_title, "\n".join(current_buffer)))

        return sections if sections else [("Overview", content)]

    def _recursive_chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits text while respecting paragraph and sentence boundaries."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep overlap
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                    current_chunk = f"{overlap_text}\n\n{para}".strip() if overlap_text else para
                else:
                    # Paragraph itself exceeds chunk_size, split by sentences
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= chunk_size:
                            current_chunk = f"{current_chunk} {sent}".strip()
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


document_processor = DocumentProcessor()
