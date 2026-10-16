import tempfile
from pathlib import Path

import numpy as np

from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.vector_search import search_chunks
from app.vector_store import load_index, save_index


def main():
    # Load and prepare the knowledge base.
    documents = load_documents()
    assert documents, "No documents were loaded."

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    # Generate embeddings using the existing component.
    model = create_embedder()
    embedded_chunks = embed_chunks(chunks, model)

    assert embedded_chunks, "No embeddings were generated."

    # Use a temporary directory so this test doesn't modify .index/.
    with tempfile.TemporaryDirectory() as temporary_directory:
        index_dir = Path(temporary_directory)

        # Save the index.
        save_index(embedded_chunks, index_dir)

        assert (index_dir / "embeddings.npy").exists()
        assert (index_dir / "chunks.json").exists()

        # Reload the index.
        loaded_chunks = load_index(index_dir)

        # Test 1: The number of chunks must be preserved.
        assert len(loaded_chunks) == len(embedded_chunks)

        # Test 2: Document metadata and text must be preserved.
        for original, loaded in zip(embedded_chunks, loaded_chunks):
            assert loaded["source"] == original["source"]
            assert loaded["section"] == original["section"]
            assert loaded["content"] == original["content"]

        # Test 3: Embedding values must survive the round trip.
        for original, loaded in zip(embedded_chunks, loaded_chunks):
            np.testing.assert_allclose(
                loaded["embedding"],
                original["embedding"],
                rtol=0,
                atol=0,
            )

        # Test 4: The loaded vectors must work with vector search.
        results = search_chunks(
            query="Why does my Kubernetes pod keep restarting?",
            embedded_chunks=loaded_chunks,
            model=model,
            top_k=3,
        )

        assert results, "Search returned no results."
        assert results[0]["source"] == "crashloopbackoff.md"

        # Test 5: An incomplete index must be rejected.
        incomplete_dir = index_dir / "incomplete"
        incomplete_dir.mkdir()

        try:
            load_index(incomplete_dir)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(
                "Loading an incomplete index should raise FileNotFoundError."
            )

    print("All vector storage tests passed!")
    print(f"Chunks saved and loaded: {len(loaded_chunks)}")
    print("Metadata preservation: passed")
    print("Embedding round-trip integrity: passed")
    print("Search using loaded vectors: passed")
    print("Incomplete index handling: passed")


if __name__ == "__main__":
    main()