"""API tests. No database, LLM or embedding model is used: the answering
function, the audit writer and (where needed) authentication are replaced
by fakes, so we test only the HTTP layer's own behaviour."""

import pytest
from fastapi.testclient import TestClient

from bankdoc.api import auth, main
from bankdoc.api.auth import User, generate_api_key, get_current_user, hash_api_key
from bankdoc.generation.answerer import AnswerResult
from bankdoc.retrieval.retriever import RetrievedChunk


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=7, title="MPR", source_uri="mpr-2026-10-28.pdf",
        page_number=23, content="Inflation remains above 3%. " * 30, score=0.7884,
    )


@pytest.fixture
def client(monkeypatch):
    """A client authenticated as alice (clearance 1), with fakes installed.

    Returns (client, state); state records what the endpoint did.
    """
    state = {"clearance": None, "audit": []}

    def fake_answer(question, clearance):
        state["clearance"] = clearance
        return AnswerResult(
            answer="Inflation eases [1].", refused=False, citations=[make_chunk()]
        )

    monkeypatch.setattr(main, "answer_question", fake_answer)
    monkeypatch.setattr(main, "write_audit_log", lambda **kwargs: state["audit"].append(kwargs))
    main.app.dependency_overrides[get_current_user] = lambda: User("alice", 1)
    yield TestClient(main.app), state
    main.app.dependency_overrides.clear()


@pytest.fixture
def anon_client():
    """A client with real authentication (no override)."""
    return TestClient(main.app)


def test_health_needs_no_auth(anon_client):
    assert anon_client.get("/health").json() == {"status": "ok"}


def test_missing_key_is_401(anon_client):
    response = anon_client.post("/ask", json={"question": "What is inflation?"})
    assert response.status_code == 401


def test_unknown_key_is_401(anon_client, monkeypatch):
    monkeypatch.setattr(auth, "find_user_by_key", lambda key: None)
    response = anon_client.post(
        "/ask", json={"question": "What is inflation?"}, headers={"X-API-Key": "bk_wrong"}
    )
    assert response.status_code == 401


def test_answer_contains_citations_with_snippet(client):
    test_client, _ = client
    body = test_client.post("/ask", json={"question": "What is inflation?"}).json()
    assert body["refused"] is False
    assert body["citations"][0]["source_uri"] == "mpr-2026-10-28.pdf"
    assert body["citations"][0]["page_number"] == 23
    assert len(body["citations"][0]["snippet"]) == 300  # truncated


def test_clearance_comes_from_the_user(client):
    test_client, state = client
    test_client.post("/ask", json={"question": "What is inflation?"})
    assert state["clearance"] == 1  # alice's clearance, set by the server


def test_request_cannot_set_its_own_clearance(client):
    test_client, state = client
    response = test_client.post("/ask", json={"question": "What is inflation?", "clearance": 2})
    assert response.status_code == 422
    assert state["clearance"] is None  # the QA function was never reached


def test_too_short_question_is_rejected(client):
    test_client, _ = client
    assert test_client.post("/ask", json={"question": "hi"}).status_code == 422


def test_answered_request_is_audited(client):
    test_client, state = client
    test_client.post("/ask", json={"question": "What is inflation?"})
    assert len(state["audit"]) == 1
    entry = state["audit"][0]
    assert entry["user_id"] == "alice"
    assert entry["cited_chunk_ids"] == [7]
    assert entry["blocked"] is False


def test_refusal_is_audited_as_blocked(client, monkeypatch):
    test_client, state = client
    monkeypatch.setattr(
        main, "answer_question",
        lambda question, clearance: AnswerResult(
            answer="I cannot find this information in the available documents.",
            refused=True, refusal_reason="no_relevant_sources",
        ),
    )
    body = test_client.post("/ask", json={"question": "How to bake cookies?"}).json()
    assert body["refused"] is True and body["citations"] == []
    assert state["audit"][0]["blocked"] is True
    assert state["audit"][0]["block_reason"] == "no_relevant_sources"


def test_hash_is_deterministic_and_not_plaintext():
    key = "bk_example"
    assert hash_api_key(key) == hash_api_key(key)
    assert hash_api_key(key) != key


def test_generated_keys_are_unique_and_prefixed():
    keys = {generate_api_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(k.startswith("bk_") for k in keys)
