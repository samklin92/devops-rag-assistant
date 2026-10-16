import numpy as np

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks


def main():
    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    # Test 1: Every chunk should have an embedding.
    assert len(embedded_chunks) == len(chunks), (
        "The number of embedded chunks does not match the input."
    )

    # Test 2: Every embedding should contain 384 dimensions.
    for item in embedded_chunks:
        assert item["embedding"].shape == (384,), (
            "An embedding has an unexpected shape."
        )

    # Test 3: Source metadata should be preserved.
    for original, embedded in zip(chunks, embedded_chunks):
        assert embedded["source"] == original["source"]
        assert embedded["section"] == original["section"]
        assert embedded["content"] == original["content"]

    # Test 4: Normalized vectors should have a norm close to 1.
    for item in embedded_chunks:
        norm = np.linalg.norm(item["embedding"])
        assert np.isclose(norm, 1.0, atol=1e-5), (
            f"Embedding norm is not 1: {norm}"
        )

    # Test 5: Empty input should return an empty list.
    assert embed_chunks([], model) == []

    print("All embedding tests passed!")
    print(f"Chunks tested: {len(embedded_chunks)}")
    print("Dimensions per embedding: 384")
    print("Metadata preservation: passed")
    print("Embedding normalization: passed")
    print("Empty input handling: passed")


if __name__ == "__main__":
    main()