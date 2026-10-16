from app.chunker import chunk_document
from app.document_loader import load_documents


def main():
    documents = load_documents()

    if not documents:
        raise RuntimeError("No documents were loaded.")

    all_chunks = []

    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)

        print(f"\nSource: {document['source']}")
        print(f"Characters: {len(document['content'])}")
        print(f"Chunks created: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}")
            print(f"Section: {chunk['section']}")
            print(f"Characters: {len(chunk['content'])}")
            print(f"Preview: {chunk['content'][:150]!r}")

    print(f"\nTotal chunks created: {len(all_chunks)}")


if __name__ == "__main__":
    main()
