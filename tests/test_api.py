"""Тесты API на заглушке-модераторе: проверяют контракт, не требуя обученных моделей."""
import pytest
from fastapi.testclient import TestClient

from moderation.api import create_app
from moderation.moderator import ModerationResult


class FakeModerator:
    cfg = {"service": {"batch_max_size": 8, "batch_max_wait_ms": 2}}
    components = ["rules", "fake"]
    calls = []

    def moderate_batch(self, texts):
        self.calls.append(len(texts))
        return [ModerationResult("BLOCK", ["profanity"], 0.97, ["хуй"], {"ensemble": 0.97}) if "хуй" in t
                else ModerationResult("ALLOW", [], 0.99, [], {"ensemble": 0.01}) for t in texts]


@pytest.fixture
def client():
    with TestClient(create_app(FakeModerator())) as c:
        yield c


def test_moderate_contract(client):
    r = client.post("/moderate", json={"text": "хуй"})
    assert r.status_code == 200
    assert r.json() == {"decision": "BLOCK", "categories": ["profanity"], "confidence": 0.97}


def test_explain_adds_details(client):
    body = client.post("/moderate?explain=true", json={"text": "хуй"}).json()
    assert body["evidence"] == ["хуй"] and body["scores"]["ensemble"] == 0.97


def test_allow(client):
    assert client.post("/moderate", json={"text": "привет"}).json()["decision"] == "ALLOW"


def test_batch(client):
    body = client.post("/moderate/batch", json={"texts": ["привет", "хуй"]}).json()
    assert [r["decision"] for r in body["results"]] == ["ALLOW", "BLOCK"]


def test_validation(client):
    assert client.post("/moderate", json={}).status_code == 422
    assert client.post("/moderate/batch", json={"texts": []}).status_code == 422


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
