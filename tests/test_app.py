"""Run with: pytest

Each test gets a fresh temporary SQLite database via the NOTES_DB environment variable.
Requires models/classifier.joblib (python train.py).
"""
import pytest
from fastapi.testclient import TestClient

LABELS = {"Study", "Money", "Travel", "Shopping"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DB", str(tmp_path / "test_notes.db"))
    from app.main import app
    with TestClient(app) as c:  # "with" runs the startup that loads the model
        yield c


def test_classify_normal_note(client):
    r = client.post("/classify", json={"text": "Pay my electricity bill tomorrow."})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in LABELS
    assert isinstance(body["uncertain"], bool)
    assert body["category"] == "Money"


def test_classify_out_of_scope_is_uncertain(client):
    r = client.post("/classify", json={"text": "Call grandma this weekend."})
    assert r.status_code == 200
    assert r.json()["uncertain"] is True
    assert r.json()["category"] in LABELS  # still returns a best guess


@pytest.mark.parametrize("body", [
    b'{"text": ""}', b'{"text": "   "}', b'{"note": "Pay rent"}', b'{"text": 42}',
    b'{"text": null}', b'{"text": ', b'not json at all',
])
def test_classify_invalid_input_is_4xx(client, body):
    r = client.post("/classify", content=body, headers={"Content-Type": "application/json"})
    assert 400 <= r.status_code < 500
    # Server keeps working afterwards.
    assert client.post("/classify", json={"text": "Book a train to Pune."}).status_code == 200


def test_add_empty_note_rejected(client):
    assert client.post("/notes", json={"text": "  "}).status_code == 422
    assert client.get("/notes").json() == []


def test_note_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DB", str(tmp_path / "restart.db"))
    from app.main import app
    with TestClient(app) as c:
        saved = c.post("/notes", json={"text": "Call grandma this weekend."}).json()
    with TestClient(app) as c:  # new startup = app restarted, same database file
        notes = c.get("/notes").json()
    assert notes == [saved]
    assert saved["id"] == "U0001" and saved["uncertain"] is True


def test_starter_notes_no_duplicates_and_ids_not_reused(client):
    assert client.post("/notes/load-starter").json() == {"added": 12, "skipped_existing": 0}
    assert client.post("/notes/load-starter").json() == {"added": 0, "skipped_existing": 12}
    new = client.post("/notes", json={"text": "Buy a phone charger."}).json()
    ids = [n["id"] for n in client.get("/notes").json()]
    assert len(ids) == len(set(ids)) == 13
    assert {f"N{i:03d}" for i in range(1, 13)} <= set(ids)
    assert new["id"] == "U0001"


def test_filter_by_category(client):
    client.post("/notes/load-starter")
    travel = client.get("/notes", params={"category": "Travel"}).json()
    assert travel and all(n["category"] == "Travel" for n in travel)


def test_search_finds_saved_note(client):
    client.post("/notes/load-starter")
    saved = client.post("/notes", json={"text": "My train to Pune leaves at 6 AM."}).json()
    results = client.get("/search", params={"q": "Pune"}).json()["results"]
    assert results[0]["id"] == saved["id"]
    results = client.get("/search", params={"q": "money transfer"}).json()["results"]
    assert "N006" in [r["id"] for r in results[:3]]
