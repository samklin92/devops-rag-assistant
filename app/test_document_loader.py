from app.document_loader import load_documents


def main():
    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        print(f"\nSource: {document['source']}")
        print(f"Characters: {len(document['content'])}")
        print(f"Preview:\n{document['content'][:200]}")


if __name__ == "__main__":
    main()