from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.logging import logger


class NexusBaseException(Exception):
    """Base exception for Nexus Agentic RAG system."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentProcessingError(NexusBaseException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, status_code=422, details=details)


class VectorStoreError(NexusBaseException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, status_code=500, details=details)


class RetrievalError(NexusBaseException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, status_code=500, details=details)


class LLMProviderError(NexusBaseException):
    def __init__(self, message: str, details: dict = None):
        super().__init__(message=message, status_code=502, details=details)


async def nexus_exception_handler(request: Request, exc: NexusBaseException):
    logger.error(
        f"Domain exception encountered: {exc.message}",
        extra={"request_id": getattr(request.state, "request_id", "N/A"), "details": exc.details}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path),
        }
    )
