from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIRECTORY,
    EMBEDDING_MODEL,
)


def load_pdf_documents(data_dir: str = DATA_DIRECTORY) -> List[Document]:
    """Load all PDF files from the configured data directory."""
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def normalize_document_metadata(docs: List[Document]) -> List[Document]:
    """Keep only source metadata to reduce vector store payload size."""
    normalized: List[Document] = []
    for doc in docs:
        normalized.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "unknown")},
            )
        )
    return normalized


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize HuggingFace embedding model for semantic search."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
