"""FastAPI app: /classify (fixed contract), note routes, search, and the HTML UI at /.

    uvicorn app.main:app --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictStr, field_validator

from app.classifier import LABELS, NoteClassifier
from app.data import load_starter_notes
from app.search import default_searcher, visible_results
from app.store import NoteStore

STATIC = Path(__file__).resolve().parent / "static"
state = {}


@asynccontextmanager
async def lifespan(app):
    # Load the model, the search model and the database once at startup, not per request.
    state["classifier"] = NoteClassifier()
    state["store"] = NoteStore()
    state["searcher"] = default_searcher()
    yield
    state.clear()


app = FastAPI(title="Smart Notes Assistant", lifespan=lifespan)


class NoteIn(BaseModel):
    # StrictStr rejects numbers/null instead of coercing them; 422 is returned automatically.
    text: StrictStr = Field(max_length=10_000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return v


@app.post("/classify")
def classify(note: NoteIn):
    return state["classifier"].predict(note.text)


@app.post("/notes", status_code=201)
def add_note(note: NoteIn):
    text = note.text.strip()
    return state["store"].add(text, state["classifier"].predict(text))


@app.get("/notes")
def list_notes(category: Optional[str] = None, uncertain: Optional[bool] = None):
    if category and category not in LABELS:
        raise HTTPException(400, f"category must be one of {LABELS}")
    return state["store"].list(category=category, uncertain=uncertain)


@app.post("/notes/load-starter")
def load_starter():
    return load_starter_into(state["store"], state["classifier"])


def load_starter_into(store, classifier) -> dict:
    added = skipped = 0
    for n in load_starter_notes():
        if store.add_with_id(n["id"], n["text"], classifier.predict(n["text"])):
            added += 1
        else:
            skipped += 1
    return {"added": added, "skipped_existing": skipped}


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=50)):
    if not q.strip():
        raise HTTPException(400, "query must not be empty")
    searcher = state["searcher"]
    ranked = searcher.rank(q, state["store"].list())
    results = [{**note, "score": round(score, 3)} for note, score in visible_results(searcher, ranked, k)]
    return {"query": q, "method": searcher.name, "results": results,
            "message": None if results else "No notes match this query well."}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
