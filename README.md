# Campus Helpdesk Triage Agent

An agent that answers routine campus questions (fees, exams, attendance,
hostel, mess, library, Wi-Fi/IT, health & clinic, academics) from approved
source documents, asks for missing details before answering, and hands off
unresolved or low-confidence questions to the right human team as a
structured ticket — instead of guessing.

Built for the hackathon problem statement: *grounded responses, escalation
discipline, and useful handoff.* Now populated with real VIT Bhopal campus
content and deployed live.

**Live demo:** https://overnight-hackathon.onrender.com
*(free-tier hosting — the first request after a period of inactivity may
take ~30-50s to wake the server up)*

## Quick start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # add your GEMINI_API_KEY (see below)
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the demo chat UI. It shows the
conversation on the left and a live **decision trace** panel on the right —
which document/section matched, the confidence outcome, and (when
escalated) the ticket ID and routed department. Trace entries are
collapsible and update in real time as you chat.

No API key is required to run the retrieval + escalation logic itself —
without `GEMINI_API_KEY` set, answers fall back to the raw matched source
text. With a key set, answers are phrased naturally by Gemini while staying
strictly grounded in the retrieved chunk (not the whole document — see
"Chunk-level retrieval" below).

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
Query contains no recognized campus vocabulary?   ask next missing detail
     │ yes                        │ no
     ▼                            ▼
Escalate: off-topic          Retrieve top matching chunks (TF-IDF, chunk-level)
                                   │
                                   ▼
                          score < LOW_CONFIDENCE ──yes──► Escalate: no confident match
                                   │ no
                                   ▼
                          High confidence?  ──no──►  Answer with best-guess chunk anyway
                                   │ yes                (no dead-end disambiguation loop)
                                   ▼
                          Missing required info (e.g. semester, program)?
                                   │ yes                      │ no
                                   ▼                          ▼
                          ask for it, then answer     Answer, grounded + cited to source doc
```

### Components

| File | Responsibility |
|---|---|
| `app/rag.py` | Loads approved source documents (`app/data/docs/*.md`), splits each into small, independently-searchable **chunks** (per section, per paragraph — see below), builds a TF-IDF index over chunks, and retrieves top matches with a confidence score. |
| `app/triage.py` | The agent's decision logic: confidence thresholds, off-topic gating, slot-filling for missing details, and the escalation trigger. This is where "escalation discipline" lives. |
| `app/tickets.py` | Handoff layer. Creates structured tickets with the original question, gathered context, escalation reason, and department routing; persisted to `app/data/store/tickets.json`. |
| `app/main.py` | FastAPI app exposing `/chat`, `/tickets`, `/sources`, `/health`. |
| `static/index.html` | Demo UI: chat thread + live decision-trace panel, markdown-bold and auto-linked URLs in replies, typing indicator, and quick-reply chips. |

### Chunk-level retrieval (not whole-document)

Earlier prototype versions scored whole markdown files against the query,
which caused two problems: short queries would sometimes match the wrong
document just because it happened to repeat a shared word, and — even when
the right document matched — the extractive fallback couldn't reliably
pick the right paragraph out of the whole file.

`rag.py` now splits each doc into chunks at each `## Section` (and
`### Subsection`, e.g. a day-of-week entry in the mess menu), one chunk per
paragraph/bullet within that section. Each chunk carries its own heading +
that section's `Aliases:` line in its **search** text (so paraphrased or
fresher-style queries still match), while only the actual content is shown
in the **answer**. Retrieval also uses a term-coverage tiebreaker on top of
raw cosine similarity, so a chunk matching *every* meaningful word in the
query wins over one that just scores marginally higher by coincidence.

### Why TF-IDF instead of a vector DB / embeddings API

This is a hackathon prototype — the goal was a **working, demoable
retrieval + escalation loop with minimal external dependencies**. TF-IDF
over chunked markdown docs gives real similarity scores to gate confidence
thresholds, runs instantly, and needs no API key for retrieval itself (only
final answer phrasing uses an LLM). Swapping in a proper embedding model or
vector store (e.g. `sentence-transformers` + FAISS/Chroma) is a drop-in
replacement in `app/rag.py` — see "Next steps" below.

## Adding / editing approved sources

Drop a new markdown file into `app/data/docs/` with frontmatter like:

```markdown
---
title: Library Fine Policy
department: Library
required_info: []
tags: [library, fine, book, overdue]
---

## Late Return Fine
Aliases: library fine, overdue book fine, late return charge

**For everyone:** ...body text the agent is allowed to answer from...
```

- `required_info` lists any details the agent must collect (e.g. `semester`,
  `program`) before it's allowed to answer using that document.
- `department` is where tickets get routed if that document is the best
  (but insufficient) match.
- Use `## Section` headings to split distinct topics within one doc (each
  becomes its own scoreable chunk), and an `Aliases:` line right under each
  heading to list alternate phrasings — especially fresher-style phrasing
  ("how do I get a room") alongside regular-student phrasing ("room
  rebooking process") for the same topic. This is the single biggest lever
  for retrieval accuracy with TF-IDF.
- The index rebuilds automatically at each server start (or via whatever
  reload endpoint/hook you've wired up).

## Current approved sources (VIT Bhopal)

Fees & payment deadlines · Hostel & accommodation (allotment, timings,
mess-fee waivers, guest policy, maintenance) · Mess timings & full weekly
menu · Academics (syllabus, results, revaluation, academic calendar) · IT
services (VTOP password reset, Wi-Fi setup & troubleshooting) · Library
(timings, facilities) · Health & wellness (clinic timings, emergency
support) · Student leave & attendance policy · Examination schedule &
procedures · Student ID card & campus access.

## API

- `POST /chat` — `{ "session_id": "...", "message": "..." }` → `{ reply, status, source?, department?, ticket? }`
  `status` is one of `answered`, `clarifying`, `escalated`.
- `GET /tickets` — list all handoff tickets (optional `?status=open`).
- `GET /tickets/{id}` — full ticket detail, including conversation context.
- `POST /tickets/{id}/resolve` — `{ "status": "resolved" }`.
- `GET /sources` — lists the approved documents the agent can draw from (transparency).
- `GET /health`

## Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Optional (recommended) | Enables natural-language answer generation via Gemini, grounded strictly in the retrieved source chunk. Without it, the raw matched chunk text is returned as-is. |

Set locally via `.env` (see `.env.example` — **never commit a real key
there**, only a placeholder) or as an environment variable in your
deployment platform (e.g. Render's Environment tab).

## Deployment

Currently deployed on **Render** (free tier):
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- `GEMINI_API_KEY` set via Render's Environment Variables panel

Free tier spins down after inactivity; the first request after idle time
takes ~30-50s to respond while the instance wakes up.

## Known gaps

- Monday's mess dinner menu is incomplete (source photo was unreadable) —
  placeholder left in `app/data/docs/mess.md`, pending confirmation.
- Hostel maintenance-reporting channel assumes the online form at
  `snm.vitbhopal.ac.in/hc/en` is current — worth periodic reconfirmation.
- Retrieval is TF-IDF based, so paraphrased queries rely on well-written
  `Aliases:` lines in each doc rather than true semantic understanding.

## Next steps (post-hackathon)

- Swap TF-IDF for real embeddings + a vector store for better semantic matching.
- Persist sessions in Redis/DB instead of in-memory (needed once multi-worker).
- Connect `/tickets` to a real ticketing system (Freshservice, Zendesk, email/Slack webhook) instead of local JSON.
- Add auth so `/tickets` is staff-only.
- Add an analytics view over escalated questions to find gaps in the source docs.
- Persist ticket storage to a real database (Supabase/Postgres) instead of a local JSON file, especially once deployed with multiple instances/workers.
