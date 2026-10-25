from app.agent_state import AgentState
from app.agent_decision import decide_next_action


def test_no_findings_starts_with_pod_event_inspection():
    state = AgentState(task="Investigate ImagePullBackOff")

    action = decide_next_action(state)

    assert action == "inspect_pod_events"
    assert state.next_action == action


def test_authentication_failure_selects_secret_inspection():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "FailedToRetrieveImagePullSecret"},
    })

    action = decide_next_action(state)

    assert action == "inspect_image_pull_secret_configuration"


def test_missing_image_selects_image_inspection():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "ImageNotFound"},
    })

    action = decide_next_action(state)

    assert action == "inspect_image_name_and_tag"


def test_unknown_reason_requests_more_diagnostics():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })

    action = decide_next_action(state)

    assert action == "gather_additional_diagnostics"


def test_knowledge_is_preserved_when_diagnostics_are_inconclusive():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check registry credentials and image tags.",
        "score": 0.85,
    })

    action = decide_next_action(state)

    assert action == "gather_additional_diagnostics"
    assert state.knowledge[0]["source"] == "imagepullbackoff.md"


def test_empty_knowledge_does_not_break_decision_making():
    state = AgentState(task="Investigate ImagePullBackOff")

    action = decide_next_action(state)

    assert action == "inspect_pod_events"


def test_inconclusive_finding_keeps_knowledge_available_for_next_decision():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting",
        "content": "Check image names and registry credentials.",
        "score": 0.85,
    })

    action = decide_next_action(state)

    assert action == "gather_additional_diagnostics"
    assert state.knowledge


def test_registry_connectivity_guidance_keeps_action_supported():
    state = AgentState(task="Investigate ImagePullBackOff")
    state.findings.append({
        "tool": "inspect_pod_events",
        "result": {"reason": "UnknownError"},
    })
    state.knowledge.append({
        "source": "imagepullbackoff.md",
        "section": "Troubleshooting Steps",
        "content": (
            "Check registry connectivity. Investigate DNS resolution, "
            "firewall rules, and registry availability."
        ),
        "score": 0.85,
    })

    action = decide_next_action(state)

    assert action == "gather_additional_diagnostics"