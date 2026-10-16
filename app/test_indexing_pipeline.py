import pytest

from app import indexing_pipeline


def test_build_index_success(monkeypatch):
    """Verify that the pipeline loads, embeds, and saves documents."""

    documents = [
        {
            "source": "crashloopbackoff.md",
            "content": "A Kubernetes pod keeps restarting.",
        }
    ]

    chunks = [
        {
            "source": "crashloopbackoff.md",
            "section": "Symptoms",
            "content": "The pod repeatedly restarts.",
        },
        {
            "source": "crashloopbackoff.md",
            "section": "Remediation",
            "content": "Inspect the application logs.",
        },
    ]

    embedded_chunks = [
        {**chunk, "embedding": [0.1, 0.2]}
        for chunk in chunks
    ]

    saved = {}

    monkeypatch.setattr(
        indexing_pipeline,
        "load_documents",
        lambda: documents,
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "chunk_document",
        lambda document: chunks,
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "create_embedder",
        lambda: object(),
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "embed_chunks",
        lambda input_chunks, model: embedded_chunks,
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "save_index",
        lambda input_chunks: saved.update(
            {"chunks": input_chunks}
        ),
    )

    result = indexing_pipeline.build_index()

    assert result == {
        "documents": 1,
        "chunks": 2,
        "embeddings": 2,
    }

    assert saved["chunks"] == embedded_chunks


def test_build_index_rejects_empty_documents(monkeypatch):
    """Verify that an empty knowledge base is rejected."""

    monkeypatch.setattr(
        indexing_pipeline,
        "load_documents",
        lambda: [],
    )

    with pytest.raises(
        RuntimeError,
        match="No documents were loaded",
    ):
        indexing_pipeline.build_index()


def test_build_index_rejects_empty_chunks(monkeypatch):
    """Verify that documents producing no chunks are rejected."""

    monkeypatch.setattr(
        indexing_pipeline,
        "load_documents",
        lambda: [
            {
                "source": "empty.md",
                "content": "Example content",
            }
        ],
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "chunk_document",
        lambda document: [],
    )

    with pytest.raises(
        RuntimeError,
        match="No chunks were generated",
    ):
        indexing_pipeline.build_index()


def test_build_index_rejects_empty_embeddings(monkeypatch):
    """Verify that an empty embedding result is rejected."""

    monkeypatch.setattr(
        indexing_pipeline,
        "load_documents",
        lambda: [
            {
                "source": "example.md",
                "content": "Example content",
            }
        ],
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "chunk_document",
        lambda document: [
            {
                "source": "example.md",
                "section": "Example",
                "content": "Example content",
            }
        ],
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "create_embedder",
        lambda: object(),
    )

    monkeypatch.setattr(
        indexing_pipeline,
        "embed_chunks",
        lambda chunks, model: [],
    )

    with pytest.raises(
        RuntimeError,
        match="No embeddings were generated",
    ):
        indexing_pipeline.build_index()