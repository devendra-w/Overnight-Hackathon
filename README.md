# Campus Helpdesk Triage Agent

An agent that answers routine campus questions (fees, exams, leave/attendance,
ID cards, hostel policy) from approved source documents, asks for missing
details before answering, and hands off unresolved or low-confidence
questions to the right human team as a structured ticket — instead of
guessing.

Built for the hackathon problem statement: *grounded responses, escalation
discipline, and useful handoff.*

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # optional — see below
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the demo chat UI. It shows the
conversation on the left and a live **decision trace** on the right —
which document matched, the confidence outcome, and (when escalated) the
ticket ID and routed department.

No API key is required to run the prototype: without `ANTHROPIC_API_KEY`
set, answers are generated extractively straight from the source markdown
files, so retrieval + escalation logic is fully demonstrable offline. If
you add a key, answers are phrased more naturally by Claude while staying
constrained to the same source text.

## Architecture

```
User message
     │
     ▼
Explicit "talk to a human"? ──yes──► Escalate: create ticket ──► routed to department
     │ no
     ▼
Mid slot-filling from a previous turn? ──yes──► fill slot ──► all slots filled? ──► answer
     │ no                                              │ no
     ▼                                                 ▼
Retrieve top matches from approved docs (TF-IDF)   ask next missing detail
     │
     ▼
score < LOW_CONFIDENCE ──yes──► Escalate: no confident source match
     │ no
     ▼
score < HIGH_CONFIDENCE ──yes──► ask ONE disambiguating question
     │ no                              │
     ▼                          still ambiguous? ──yes──► Escalate
Missing required info (e.g. semester, program)?
     │ yes                      │ no
     ▼                          ▼
ask for it, then answer     Answer, grounded + cited to source doc
```

### Components

| File | Responsibility |
|---|---|
| `app/rag.py` | Loads the approved source documents (`app/data/docs/*.md`), builds a TF-IDF index, retrieves top matches with a confidence score, and provides an extractive fallback answer generator. |
| `app/triage.py` | The agent's decision logic: confidence thresholds, slot-filling for missing details, disambiguation, and the escalation trigger. This is where "escalation discipline" lives. |
| `app/tickets.py` | Handoff layer. Creates structured tickets with the original question, gathered context, escalation reason, and department routing; persisted to `app/data/store/tickets.json`. |
| `app/main.py` | FastAPI app exposing `/chat`, `/tickets`, `/sources`, `/health`. |
| `static/index.html` | Demo UI: chat thread + live decision-trace panel for judges/staff to see *why* the agent did what it did. |

### Why TF-IDF instead of a vector DB / embeddings API

This is a hackathon prototype — the goal here was a **working, demoable
retrieval + escalation loop with zero external dependencies and zero
network calls required**. TF-IDF over the approved markdown docs gives
real similarity scores to gate the confidence thresholds, runs instantly,
and needs no API key. Swapping in a proper embedding model or vector store
(e.g. `sentence-transformers` + FAISS/Chroma) is a drop-in replacement in
`app/rag.py` — see "Next steps" below.

## Adding / editing approved sources

Drop a new markdown file into `app/data/docs/` with frontmatter like:

```markdown
---
title: Library Fine Policy
department: Library
required_info: []
tags: [library, fine, book, overdue]
---

# Library Fine Policy
...body text the agent is allowed to answer from...
```

`required_info` lists any details the agent must collect (e.g. `semester`,
`program`) before it's allowed to answer using that document. `department`
is where tickets get routed if that document is the best (but insufficient)
match. The index rebuilds automatically at each server start.

## API

- `POST /chat` — `{ "session_id": "...", "message": "..." }` → `{ reply, status, source?, department?, ticket? }`
  `status` is one of `answered`, `clarifying`, `escalated`.
- `GET /tickets` — list all handoff tickets (optional `?status=open`).
- `GET /tickets/{id}` — full ticket detail, including conversation context.
- `POST /tickets/{id}/resolve` — `{ "status": "resolved" }`.
- `GET /sources` — lists the approved documents the agent can draw from (transparency).
- `GET /health`

## Next steps (post-hackathon)

- Swap TF-IDF for real embeddings + a vector store for better semantic matching.
- Persist sessions in Redis/DB instead of in-memory (needed once multi-worker).
- Connect `/tickets` to a real ticketing system (Freshservice, Zendesk, email/Slack webhook) instead of local JSON.
- Add auth so `/tickets` is staff-only.
- Add an analytics view over escalated questions to find gaps in the source docs.
