from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "knowledge_base" / "documents"


def load_documents() -> list[dict[str, str]]:
    """Load Markdown documents from the knowledge base."""
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            {
                "source": file_path.name,
                "content": content,
            }
        )

    return documents
