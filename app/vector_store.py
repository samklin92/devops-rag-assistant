import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / ".index"


def save_index(
    embedded_chunks: list[dict],
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> None:
    """Save document embeddings and chunk metadata to disk."""

    if not embedded_chunks:
        raise ValueError("Cannot save an empty vector index.")

    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.vstack(
        [chunk["embedding"] for chunk in embedded_chunks]
    )

    chunks = []

    for chunk in embedded_chunks:
        chunks.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
            }
        )

    np.save(index_dir / "embeddings.npy", embeddings)

    with (index_dir / "chunks.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)


def load_index(
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> list[dict]:
    """Load document chunks and embeddings from disk."""

    embeddings_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.json"

    if not embeddings_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Vector index is incomplete or missing: {index_dir}"
        )

    embeddings = np.load(embeddings_path, allow_pickle=False)

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError("Stored embeddings must be a two-dimensional array.")

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Stored embedding count does not match the chunk count."
        )

    required_fields = {"source", "section", "content"}

    for chunk in chunks:
        if not isinstance(chunk, dict) or not required_fields.issubset(chunk):
            raise ValueError("Stored chunk metadata is invalid.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Stored embeddings contain non-finite values.")

    loaded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        loaded_chunks.append(
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "embedding": embedding,
            }
        )

    return loaded_chunks