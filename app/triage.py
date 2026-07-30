"""
triage.py — The decision layer that ties retrieval + confidence +
slot-filling + escalation together. This is the "agent" logic.

Escalation discipline, in plain terms:
  HIGH confidence match + all required info present -> answer, grounded & cited.
  HIGH confidence match + missing info               -> ask for the missing info.
  MEDIUM confidence (ambiguous)                       -> ask a disambiguating
                                                          question, once.
  LOW confidence / still unclear after one attempt    -> escalate to a human
                                                          team with a ticket.
  Explicit user request ("talk to a human")           -> escalate immediately.
"""

import os
import re
from typing import Dict, List, Optional

from app.rag import DocumentStore, extract_best_snippet
from app import tickets as ticket_store

LOW_CONFIDENCE = 0.12       # below this: no approved source is a plausible match at all
CONFIDENT_FLOOR = 0.15      # minimum score to ever call a match "confident"
DOMINANCE_RATIO = 2.0       # top match must beat the runner-up by this multiple

MIN_MATCHED_TERMS = 2       # top doc must share at least this many real terms with the query
SOLE_MATCH_FLOOR = 0.30     # when there's no runner-up at all, require a much stronger score
KNOWN_TERM_RATIO_FLOOR = 0.3  # query must contain some real domain vocabulary at all

def _is_high_confidence(results: List[Dict]) -> bool:
    if not results:
        return False
    top = results[0]
    if top["score"] < CONFIDENT_FLOOR:
        return False
    if top.get("matched_terms", 0) < MIN_MATCHED_TERMS:
        return False

    runner_up = results[1] if len(results) > 1 else None
    if runner_up is None or runner_up["score"] == 0.0:
        return top["score"] >= SOLE_MATCH_FLOOR

    score_dominant = top["score"] >= runner_up["score"] * DOMINANCE_RATIO

    # Full term coverage: the top doc matched every meaningful word in the
    # query, and the runner-up did not. This catches short queries where a
    # runner-up doc scores similarly only because it shares one generic
    # word (e.g. "timings") — the doc matching ALL the query's real words
    # is the stronger, more reliable signal here, regardless of score ratio.
    total_terms = top.get("query_term_count", 0)
    full_coverage_dominant = (
        total_terms > 0
        and top.get("matched_terms", 0) >= total_terms
        and runner_up.get("matched_terms", 0) < total_terms
    )

    return score_dominant or full_coverage_dominant


HUMAN_REQUEST_PATTERNS = [
    r"\btalk to (a |someone from )?(a )?human\b",
    r"\breal person\b",
    r"\btalk to (the |a )?staff\b",
    r"\bcontact (the )?staff\b",
    r"\bspeak (to|with) (a |someone|the)\b.*\b(staff|human|person)\b",
    r"\bescalate this\b",
    r"\braise a ticket\b",
    r"\bcreate a ticket\b",
    r"\bopen a ticket\b",
    r"\bspeak to someone\b",
]

_store = DocumentStore()

# In-memory session state (fine for a hackathon prototype / single process).
# Each session: conversation history + any pending slot-filling state.
_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "pending_doc_id": None,
            "awaiting_info": [],
            "collected_info": {},
            "clarify_attempts": 0,
        }
    return _sessions[session_id]


def _wants_human(message: str) -> bool:
    m = message.lower()
    return any(re.search(p, m) for p in HUMAN_REQUEST_PATTERNS)


def _missing_required_info(doc_required: List[str], collected: Dict) -> List[str]:
    return [field for field in doc_required if field not in collected]


def _generate_grounded_answer(doc_title: str, body: str, query: str, department: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            system = (
                "You are a campus helpdesk assistant. Answer ONLY using the "
                "provided source document text. Do not add information that "
                "is not in the document. If the document does not fully "
                "answer the question, say so plainly. Keep the answer under "
                "120 words, plain and direct, no preamble."
            )
            prompt = (
                f"{system}\n\nSource document ({doc_title}):\n{body}\n\n"
                f"Question: {query}"
            )
            resp = model.generate_content(prompt)
            if resp and resp.text:
                return resp.text.strip()
        except Exception:
            pass  # fall through to extractive answer

    return extract_best_snippet(body, query)


def handle_message(session_id: str, message: str) -> Dict:
    session = _get_session(session_id)
    session["history"].append({"role": "user", "message": message})

    # 1. Explicit request for a human -> escalate immediately, no guessing.
    if _wants_human(message):
        ticket = ticket_store.create_ticket(
            session_id=session_id,
            question=message,
            conversation_context=session["history"],
            collected_info=session["collected_info"],
            reason="User explicitly requested a human / staff member.",
            best_guess_doc_id=session.get("pending_doc_id"),
        )
        reply = (
            f"Got it — I've routed this to {ticket['routed_to']} as ticket "
            f"{ticket['id']}. They'll follow up with you directly."
        )
        session["history"].append({"role": "agent", "message": reply})
        _reset_slotfilling(session)
        return _response(reply, "escalated", ticket=ticket)

    # 2. Continuing a slot-filling conversation (agent asked for missing info).
    if session["awaiting_info"]:
        field = session["awaiting_info"].pop(0)
        session["collected_info"][field] = message.strip()

        if session["awaiting_info"]:
            next_field = session["awaiting_info"][0]
            reply = f"Thanks. And what's your {next_field.replace('_', ' ')}?"
            session["history"].append({"role": "agent", "message": reply})
            return _response(reply, "clarifying")

        # All slots filled -> answer using the pending doc.
        doc = _store.get(session["pending_doc_id"])
        if doc:
            original_q = _last_original_question(session)
            answer = _generate_grounded_answer(doc.title, doc.body, original_q, doc.department)
            reply = f"{answer}\n\n(Source: {doc.title} — {doc.department})"
            session["history"].append({"role": "agent", "message": reply})
            _reset_slotfilling(session)
            return _response(reply, "answered", source=doc.title, department=doc.department)

     # 3. Fresh question -> retrieve.
    results = _store.search(message, top_k=3)
    top = results[0] if results else None
    session["clarify_attempts"] = session.get("clarify_attempts", 0)

    known_ratio = _store.known_term_ratio(message)
    if known_ratio < KNOWN_TERM_RATIO_FLOOR:
        ticket = ticket_store.create_ticket(
            session_id=session_id,
            question=message,
            conversation_context=session["history"],
            collected_info=session["collected_info"],
            reason="Query contains little to no recognized campus-domain vocabulary.",
            best_guess_doc_id=top["doc_id"] if top else None,
        )
        reply = (
            "That doesn't look like something I can answer from our approved "
            f"campus documents, so I've created ticket {ticket['id']} for "
            f"{ticket['routed_to']} to take a look."
        )
        session["history"].append({"role": "agent", "message": reply})
        return _response(reply, "escalated", ticket=ticket)

    if not _is_high_confidence(results):
        # Medium confidence: ask one disambiguating question before giving up.
        if session["clarify_attempts"] >= 1:
            ticket = ticket_store.create_ticket(
                session_id=session_id,
                question=message,
                conversation_context=session["history"],
                collected_info=session["collected_info"],
                reason="Question remained ambiguous after a clarification attempt.",
                best_guess_doc_id=top["doc_id"],
            )
            reply = (
                f"I still can't confidently match this to one topic, so I've "
                f"escalated it as ticket {ticket['id']} to {ticket['routed_to']}."
            )
            session["history"].append({"role": "agent", "message": reply})
            _reset_slotfilling(session)
            return _response(reply, "escalated", ticket=ticket)

        session["clarify_attempts"] += 1
        session["pending_doc_id"] = top["doc_id"]
        options = ", ".join(r["title"] for r in results[:2])
        reply = f"Just to make sure I point you the right way — is this about {options}?"
        session["history"].append({"role": "agent", "message": reply})
        return _response(reply, "clarifying")

    # High confidence match.
    doc = _store.get(top["doc_id"])
    session["clarify_attempts"] = 0
    missing = _missing_required_info(doc.required_info, session["collected_info"])
    if missing:
        session["pending_doc_id"] = doc.doc_id
        session["awaiting_info"] = missing
        session["_original_question"] = message
        first_field = missing[0]
        reply = f"I can help with that. What's your {first_field.replace('_', ' ')}?"
        session["history"].append({"role": "agent", "message": reply})
        return _response(reply, "clarifying")

    answer = _generate_grounded_answer(doc.title, doc.body, message, doc.department)
    reply = f"{answer}\n\n(Source: {doc.title} — {doc.department})"
    session["history"].append({"role": "agent", "message": reply})
    return _response(reply, "answered", source=doc.title, department=doc.department)


def _last_original_question(session: Dict) -> str:
    return session.get("_original_question") or (
        session["history"][0]["message"] if session["history"] else ""
    )


def _reset_slotfilling(session: Dict) -> None:
    session["pending_doc_id"] = None
    session["awaiting_info"] = []
    session["collected_info"] = {}
    session["clarify_attempts"] = 0
    session["_original_question"] = None


def _response(reply: str, status: str, **extra) -> Dict:
    out = {"reply": reply, "status": status}
    out.update(extra)
    return out


def reload_documents() -> int:
    _store.reload()
    return len(_store.documents)