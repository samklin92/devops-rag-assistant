from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.vector_store import save_index


def build_index() -> dict[str, int]:
    """Load, chunk, embed, and persist the knowledge base."""

    # 1. Load source documents.
    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    # 2. Split documents into chunks.
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    if not chunks:
        raise RuntimeError("No chunks were generated.")

    # 3. Generate embeddings.
    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    if not embedded_chunks:
        raise RuntimeError("No embeddings were generated.")

    # 4. Persist the searchable index.
    save_index(embedded_chunks)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embeddings": len(embedded_chunks),
    }


def main() -> None:
    """Run the indexing pipeline and display its results."""

    result = build_index()

    print("Indexing completed successfully!")
    print(f"Documents loaded: {result['documents']}")
    print(f"Chunks generated: {result['chunks']}")
    print(f"Embeddings generated: {result['embeddings']}")
    print("Vector index saved to .index/")


if __name__ == "__main__":
    main()