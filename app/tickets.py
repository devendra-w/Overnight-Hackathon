"""
tickets.py — Handoff / escalation layer.

When the agent can't answer confidently from approved sources, it creates
a structured ticket instead of guessing. Tickets carry the conversation
context, gathered slot info, urgency, and the department the ticket is
routed to, so a human can pick it up without asking the user to repeat
themselves.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "store", "tickets.json")

DEPARTMENT_ROUTES = {
    "fee_deadlines": "Finance Office",
    "leave_policy": "Academic Office",
    "exam_schedule": "Examination Cell",
    "id_card": "Administration Office",
    "hostel": "Hostel Administration",
    None: "General Helpdesk",
}


def _load() -> List[Dict]:
    if not os.path.exists(STORE_PATH):
        return []
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(tickets: List[Dict]) -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)


def create_ticket(
    session_id: str,
    question: str,
    conversation_context: List[Dict],
    collected_info: Dict,
    reason: str,
    best_guess_doc_id: Optional[str] = None,
    urgency: str = "normal",
) -> Dict:
    tickets = _load()
    department = DEPARTMENT_ROUTES.get(best_guess_doc_id, "General Helpdesk")
    ticket = {
        "id": f"HD-{uuid.uuid4().hex[:8].upper()}",
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "urgency": urgency,
        "routed_to": department,
        "question": question,
        "reason_for_escalation": reason,
        "collected_info": collected_info,
        "conversation_context": conversation_context,
    }
    tickets.append(ticket)
    _save(tickets)
    return ticket


def list_tickets(status: Optional[str] = None) -> List[Dict]:
    tickets = _load()
    if status:
        return [t for t in tickets if t["status"] == status]
    return tickets


def get_ticket(ticket_id: str) -> Optional[Dict]:
    for t in _load():
        if t["id"] == ticket_id:
            return t
    return None


def update_ticket_status(ticket_id: str, status: str) -> Optional[Dict]:
    tickets = _load()
    for t in tickets:
        if t["id"] == ticket_id:
            t["status"] = status
            _save(tickets)
            return t
    return None
