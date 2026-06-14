"""Build and upload medical document embeddings to Pinecone."""

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    DATA_DIRECTORY,
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from src.helper import (
    build_embedding_model,
    load_pdf_documents,
    normalize_document_metadata,
    split_documents,
)

load_dotenv()


def ensure_pinecone_index(client: Pinecone, index_name: str) -> None:
    if not client.has_index(index_name):
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")


def main() -> None:
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY is required.")

    print("Loading PDF documents...")
    raw_documents = load_pdf_documents(DATA_DIRECTORY)
    if not raw_documents:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIRECTORY}'. "
            "Add medical PDFs before running this script."
        )

    documents = normalize_document_metadata(raw_documents)
    chunks = split_documents(documents)
    print(f"Prepared {len(chunks)} text chunks for embedding.")

    embeddings = build_embedding_model()
    client = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(client, PINECONE_INDEX_NAME)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("Embeddings uploaded to Pinecone successfully.")


if __name__ == "__main__":
    main()
