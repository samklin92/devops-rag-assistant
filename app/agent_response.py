"""Generate evidence-aware responses for DevOps agent investigations."""

from typing import Any, Callable

from app.agent_state import AgentState
from app.generator import generate_answer


def generate_agent_response(
    state: AgentState,
    client: Any,
    answer_generator: Callable[..., str] = generate_answer,
) -> dict[str, Any]:
    """Generate a troubleshooting response from findings and RAG evidence."""
    if not state.knowledge:
        return {
            "answer": (
                "I cannot provide a documentation-grounded solution because "
                "no relevant knowledge-base evidence is available. Review "
                "the diagnostic findings, consult relevant DevOps "
                "documentation, and escalate if the issue remains unresolved."
            ),
            "sources": [],
            "grounded_in_retrieved_knowledge": False,
        }

    findings_text = "\n".join(
        f"- Tool: {finding.get('tool', 'unknown')}; "
        f"result: {finding.get('result', {})}"
        for finding in state.findings
    )

    question = (
        f"Troubleshooting task: {state.task}\n\n"
        "Diagnostic findings collected by the agent:\n"
        f"{findings_text or '- No diagnostic findings collected.'}\n\n"
        "Explain what the findings establish and what remains uncertain. "
        "Use the retrieved documentation to recommend practical next steps. "
        "Do not claim an unconfirmed cause is proven. If the evidence is "
        "insufficient, state what additional diagnostic information is needed."
    )

    answer = answer_generator(
        question=question,
        retrieved_chunks=state.knowledge,
        client=client,
    )

    sources = []
    seen_sources = set()

    for chunk in state.knowledge:
        source = chunk.get("source", "Unknown source")
        section = chunk.get("section", "Unknown section")
        source_key = (source, section)

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)
        sources.append(
            {
                "source": source,
                "section": section,
                "score": chunk.get("score"),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "grounded_in_retrieved_knowledge": True,
    }