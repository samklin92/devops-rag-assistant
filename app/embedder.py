from sentence_transformers import SentenceTransformer

from app.chunker import chunk_document
from app.document_loader import load_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedder() -> SentenceTransformer:
    """Load and return the sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(
    chunks: list[dict[str, str]],
    model: SentenceTransformer,
) -> list[dict]:
    """Generate an embedding vector for every document chunk."""

    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "embedding": vector,
            }
        )

    return embedded_chunks


def main():
    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks embedded: {len(embedded_chunks)}")

    if embedded_chunks:
        first = embedded_chunks[0]

        print(f"First chunk source: {first['source']}")
        print(f"First chunk section: {first['section']}")
        print(f"Embedding dimensions: {first['embedding'].shape[0]}")
        print(f"First five values: {first['embedding'][:5]}")


if __name__ == "__main__":
    main()