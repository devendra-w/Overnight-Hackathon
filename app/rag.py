"""
rag.py — Retrieval layer for the Campus Helpdesk Triage Agent.

Design goals for the hackathon prototype:
  * Only ever answer from documents in app/data/docs (the "approved sources").
  * Zero heavy/compiled dependencies — TF-IDF + cosine similarity implemented
    in pure Python, so it installs instantly on any machine (no C/C++ build
    toolchain needed, unlike scikit-learn wheels on some Windows/Python
    combinations).
  * Returns a similarity score per document so the caller can apply
    escalation discipline (high confidence -> answer, medium -> clarify,
    low -> escalate to a human team).
"""

import os
import re
import glob
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional

DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at",
    "for", "with", "about", "as", "by", "from", "into", "over", "after",
    "before", "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her", "our", "their",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "have", "has", "had", "not", "no", "what", "when",
    "where", "who", "how", "which", "there", "here", "up", "out", "all",
}

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    department: str
    required_info: List[str]
    tags: List[str]
    body: str
    filepath: str


def _parse_frontmatter(text: str):
    """Very small YAML-ish frontmatter parser (avoids extra dependency)."""
    meta = {"title": "", "department": "", "required_info": [], "tags": []}
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm = text[3:end].strip().splitlines()
            body = text[end + 3:].strip()
            for line in fm:
                if ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                if val.startswith("[") and val.endswith("]"):
                    items = [v.strip() for v in val[1:-1].split(",") if v.strip()]
                    meta[key] = items
                else:
                    meta[key] = val
            return meta, body
    return meta, text


class _TfidfVector:
    __slots__ = ("weights", "norm")

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0


def _cosine(a: "_TfidfVector", b: "_TfidfVector") -> float:
    # iterate over the smaller dict for speed
    if len(a.weights) > len(b.weights):
        a, b = b, a
    dot = sum(w * b.weights.get(term, 0.0) for term, w in a.weights.items())
    return dot / (a.norm * b.norm)


class DocumentStore:
    """Loads all trusted markdown docs and builds a pure-Python TF-IDF index."""

    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
        self.documents: List[SourceDoc] = []
        self._doc_vectors: List[_TfidfVector] = []
        self._idf: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.documents = []
        paths = sorted(glob.glob(os.path.join(self.docs_dir, "*.md")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            meta, body = _parse_frontmatter(raw)
            doc_id = os.path.splitext(os.path.basename(path))[0]
            self.documents.append(
                SourceDoc(
                    doc_id=doc_id,
                    title=meta.get("title") or doc_id,
                    department=meta.get("department", "General Helpdesk"),
                    required_info=meta.get("required_info", []),
                    tags=meta.get("tags", []),
                    body=body,
                    filepath=path,
                )
            )
        self._build_index()

    def _build_index(self):
        n_docs = len(self.documents)
        self._doc_vectors = []
        self._idf = {}
        if n_docs == 0:
            return

        tokenized_docs = [_tokenize(self._searchable_text(d)) for d in self.documents]

        # document frequency
        df: Counter = Counter()
        for tokens in tokenized_docs:
            for term in set(tokens):
                df[term] += 1

        # smoothed idf, same formula sklearn's TfidfVectorizer uses by default
        self._idf = {
            term: math.log((1 + n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            weights = {term: count * self._idf[term] for term, count in tf.items()}
            self._doc_vectors.append(_TfidfVector(weights))

    @staticmethod
    def _searchable_text(doc: SourceDoc) -> str:
        return f"{doc.title} {' '.join(doc.tags)} {doc.body}"

    def reload(self):
        self._load()

    def _query_vector(self, query: str) -> _TfidfVector:
        tokens = _tokenize(query)
        tf = Counter(tokens)
        weights = {
            term: count * self._idf.get(term, 0.0)
            for term, count in tf.items()
            if term in self._idf
        }
        return _TfidfVector(weights)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.documents or not self._idf:
            return []
        q_vec = self._query_vector(query)
        if not q_vec.weights:
            return []

        query_term_count = len(q_vec.weights)

        scored = []
        for doc, vec in zip(self.documents, self._doc_vectors):
            score = _cosine(q_vec, vec)
            matched_terms = len(set(q_vec.weights.keys()) & set(vec.weights.keys()))
            scored.append((score, matched_terms, doc))

        # Sort by how many distinct query terms the doc actually contains FIRST,
        # cosine score second. Pure cosine ranking lets a short doc that repeats
        # one shared word (e.g. "timings") outscore a doc that genuinely covers
        # every word in the query — that's exactly backwards for short queries,
        # so term coverage has to win the sort, not just break a tie.
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

        results = []
        for score, matched_terms, doc in scored[:top_k]:
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "department": doc.department,
                    "required_info": doc.required_info,
                    "score": float(score),
                    "matched_terms": matched_terms,
                    "query_term_count": query_term_count,
                    "body": doc.body,
                }
            )
        return results

    def known_term_ratio(self, query: str) -> float:
        """Fraction of the query's meaningful tokens that exist anywhere in the
        approved-document vocabulary. Near-zero means the query is almost
        certainly off-topic, regardless of what cosine score it happens to hit."""
        tokens = _tokenize(query)
        if not tokens:
            return 0.0
        known = sum(1 for t in tokens if t in self._idf)
        return known / len(tokens)

        

    def get(self, doc_id: str) -> Optional[SourceDoc]:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        return None

def _split_paragraph_into_items(lines: List[str]) -> List[str]:
    items: List[str] = []
    for line in lines:
        starts_new = line.startswith("- ") or line.startswith("* ")
        cleaned = line.lstrip("-*").strip() if starts_new else line
        if starts_new or not items:
            items.append(cleaned)
        else:
            items[-1] = f"{items[-1]} {cleaned}"
    return items

def _reconstruct_bullets(body: str) -> List[str]:
    """Split a doc into standalone answerable chunks — one per blank-line
    paragraph — instead of one chunk per whole doc. Heading lines (#) and
    'Aliases:' metadata lines are dropped since they're for search matching
    only, never meant to be shown as part of an answer. Bullet lists within
    a paragraph still split into separate items, same as before."""
    paragraphs = re.split(r"\n\s*\n", body.strip())
    items: List[str] = []
    for para in paragraphs:
        lines = [l.strip() for l in para.splitlines() if l.strip()]
        lines = [
            l for l in lines
            if not l.startswith("#") and not l.lower().startswith("aliases:")
        ]
        if not lines:
            continue
        items.extend(_split_paragraph_into_items(lines))
    return items


def extract_best_snippet(body: str, query: str, max_sentences: int = 1) -> str:
    """Extractive fallback: pick the most query-relevant complete bullet(s)
    from a doc body when no LLM key is configured. Keeps the prototype fully
    answerable offline while remaining strictly grounded in the source text."""
    items = _reconstruct_bullets(body)
    if not items:
        return body[:400]

    q_terms = set(_tokenize(query))
    scored = []
    for item in items:
        terms = set(_tokenize(item))
        overlap = len(q_terms & terms)
        scored.append((overlap, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for score, item in scored[:max_sentences] if score > 0]
    if not top:
        top = items[:max_sentences]
    return " ".join(top)