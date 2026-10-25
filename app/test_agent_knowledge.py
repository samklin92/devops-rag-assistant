from app import agent_knowledge


def test_retrieve_agent_knowledge_returns_relevant_chunks(monkeypatch):
    expected_chunks = [
        {
            "source": "imagepullbackoff.md",
            "section": "Troubleshooting",
            "content": "Check the image name, tag, and registry credentials.",
            "score": 0.85,
        }
    ]

    def fake_retrieve(query, model, top_k):
        assert query == "How do I fix ImagePullBackOff?"
        assert model == "fake-model"
        assert top_k == 3
        return expected_chunks

    monkeypatch.setattr(agent_knowledge, "retrieve", fake_retrieve)

    results = agent_knowledge.retrieve_agent_knowledge(
        question="How do I fix ImagePullBackOff?",
        embedding_model="fake-model",
    )

    assert results == expected_chunks


def test_retrieve_agent_knowledge_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        agent_knowledge,
        "retrieve",
        lambda **kwargs: [],
    )

    results = agent_knowledge.retrieve_agent_knowledge(
        question="Unknown DevOps issue",
        embedding_model="fake-model",
    )

    assert results == []