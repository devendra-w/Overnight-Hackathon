"""
triage.py — The decision layer that ties retrieval + confidence +
slot-filling + escalation together. This is the "agent" logic.
"""

import os
import re
from typing import Dict, List, Optional

from app.rag import DocumentStore
from app import tickets as ticket_store

LOW_CONFIDENCE = 0.12
CONFIDENT_FLOOR = 0.15
DOMINANCE_RATIO = 2.0

MIN_MATCHED_TERMS_CAP = 2    # never require more than this many matched terms
SOLE_MATCH_FLOOR = 0.30
KNOWN_TERM_RATIO_FLOOR = 0.3


def _min_matched_terms(query_term_count: int) -> int:
    # Adaptive floor: a genuine 1-word query ("lunch") can only ever match
    # 1 term, so requiring 2 was silently rejecting every short query
    # regardless of score. Cap at MIN_MATCHED_TERMS_CAP for longer queries.
    return min(MIN_MATCHED_TERMS_CAP, max(query_term_count, 1))


def _is_high_confidence(results: List[Dict]) -> bool:
    if not results:
        return False
    top = results[0]
    if top["score"] < CONFIDENT_FLOOR:
        return False
    need = _min_matched_terms(top.get("query_term_count", 0))
    if top.get("matched_terms", 0) < need:
        return False

    runner_up = results[1] if len(results) > 1 else None
    if runner_up is None or runner_up["score"] == 0.0:
        return top["score"] >= SOLE_MATCH_FLOOR

    score_dominant = top["score"] >= runner_up["score"] * DOMINANCE_RATIO

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

_sessions: Dict[str, Dict] = {}


def _get_session(session_id: str) -> Dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "pending_doc_id": None,
            "pending_chunk_text": None,
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


def _generate_grounded_answer(doc_title: str, source_text: str, query: str, department: str) -> str:
    """Generate the answer text, grounded in the specific retrieved chunk
    (not the whole document) for tighter accuracy."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            system = (
                "You are a campus helpdesk assistant. Answer ONLY using the "
                "provided source text. Do not add information that is not "
                "in it. If it does not fully answer the question, say so "
                "plainly. Keep the answer under 120 words, plain and "
                "direct, no preamble."
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": f"Source ({doc_title}):\n{source_text}\n\n"
                        f"Question: {query}",
                    }
                ],
            )
            text_parts = [b.text for b in resp.content if b.type == "text"]
            if text_parts:
                return text_parts[0].strip()
        except Exception:
            pass

    return source_text


def handle_message(session_id: str, message: str) -> Dict:
    session = _get_session(session_id)
    session["history"].append({"role": "user", "message": message})

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

    if session["awaiting_info"]:
        field = session["awaiting_info"].pop(0)
        session["collected_info"][field] = message.strip()

        if session["awaiting_info"]:
            next_field = session["awaiting_info"][0]
            reply = f"Thanks. And what's your {next_field.replace('_', ' ')}?"
            session["history"].append({"role": "agent", "message": reply})
            return _response(reply, "clarifying")

        doc = _store.get(session["pending_doc_id"])
        if doc:
            original_q = _last_original_question(session)
            source_text = session.get("pending_chunk_text") or doc.body
            answer = _generate_grounded_answer(doc.title, source_text, original_q, doc.department)
            reply = f"{answer}\n\n(Source: {doc.title} — {doc.department})"
            session["history"].append({"role": "agent", "message": reply})
            _reset_slotfilling(session)
            return _response(reply, "answered", source=doc.title, department=doc.department)

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
        doc = _store.get(top["doc_id"])
        chunk_text = top.get("chunk_text") or ""
        answer = _generate_grounded_answer(doc.title, chunk_text or doc.body, message, doc.department)
        reply = f"{answer}\n\n(Source: {doc.title} — {doc.department})"
        session["history"].append({"role": "agent", "message": reply})
        session["clarify_attempts"] = 0
        return _response(reply, "answered", source=doc.title, department=doc.department)

    doc = _store.get(top["doc_id"])
    session["clarify_attempts"] = 0
    missing = _missing_required_info(doc.required_info, session["collected_info"])
    if missing:
        session["pending_doc_id"] = doc.doc_id
        session["pending_chunk_text"] = top.get("chunk_text")
        session["awaiting_info"] = missing
        session["_original_question"] = message
        first_field = missing[0]
        reply = f"I can help with that. What's your {first_field.replace('_', ' ')}?"
        session["history"].append({"role": "agent", "message": reply})
        return _response(reply, "clarifying")

    source_text = top.get("chunk_text") or doc.body
    answer = _generate_grounded_answer(doc.title, source_text, message, doc.department)
    reply = f"{answer}\n\n(Source: {doc.title} — {doc.department})"
    session["history"].append({"role": "agent", "message": reply})
    return _response(reply, "answered", source=doc.title, department=doc.department)


def _last_original_question(session: Dict) -> str:
    return session.get("_original_question") or (
        session["history"][0]["message"] if session["history"] else ""
    )


def _reset_slotfilling(session: Dict) -> None:
    session["pending_doc_id"] = None
    session["pending_chunk_text"] = None
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