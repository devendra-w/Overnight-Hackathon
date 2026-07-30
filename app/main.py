"""
main.py — FastAPI entrypoint for the Campus Helpdesk Triage Agent.

Endpoints:
  POST /chat            -> send a message, get a grounded answer / clarifying
                            question / escalation ticket
  GET  /tickets          -> list handoff tickets (the "staff view")
  GET  /tickets/{id}      -> ticket detail
  POST /tickets/{id}/resolve -> mark a ticket resolved
  GET  /sources           -> list approved source documents (transparency)
  GET  /health
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import triage
from app import tickets as ticket_store

app = FastAPI(title="Campus Helpdesk Triage Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    status: str
    ticket: Optional[dict] = None
    source: Optional[str] = None
    department: Optional[str] = None


class TicketStatusUpdate(BaseModel):
    status: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")
    result = triage.handle_message(req.session_id, req.message)
    return result


@app.get("/tickets")
def get_tickets(status: Optional[str] = None):
    return ticket_store.list_tickets(status=status)


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    t = ticket_store.get_ticket(ticket_id)
    if not t:
        raise HTTPException(status_code=404, detail="ticket not found")
    return t


@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: str, update: TicketStatusUpdate):
    t = ticket_store.update_ticket_status(ticket_id, update.status)
    if not t:
        raise HTTPException(status_code=404, detail="ticket not found")
    return t


@app.get("/sources")
def list_sources():
    return [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "department": d.department,
            "tags": d.tags,
        }
        for d in triage._store.documents
    ]


@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len(triage._store.documents)}


# Serve the demo chat UI at /
app.mount("/", StaticFiles(directory="static", html=True), name="static")
