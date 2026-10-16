from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.vector_search import search_chunks


def main():
    # Load the knowledge base.
    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    # Split documents into chunks.
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    # Generate document embeddings.
    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    # Ask a question about the knowledge base.
    query = "Why does my Kubernetes pod keep restarting?"

    results = search_chunks(
        query=query,
        embedded_chunks=embedded_chunks,
        model=model,
        top_k=3,
    )

    print(f"\nQuestion: {query}")
    print(f"Results returned: {len(results)}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {result['source']}")
        print(f"Section: {result['section']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Content preview: {result['content'][:250]}")


if __name__ == "__main__":
    main()