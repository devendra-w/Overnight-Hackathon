import os
import re
import glob
import math
from collections import Counter
from dataclasses import dataclass, field
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
    # very light stemming: strip a trailing 's' on longer words so
    # "issue"/"issues", "ticket"/"tickets" match each other.
    out = []
    for t in tokens:
        if t in _STOPWORDS or len(t) <= 1:
            continue
        out.append(t)
    return out


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _tokenize_stemmed(text: str) -> List[str]:
    return [_stem(t) for t in _tokenize(text)]


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    department: str
    required_info: List[str]
    tags: List[str]
    body: str
    filepath: str


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    heading: str
    display_text: str   # what gets shown to the user as the answer
    search_text: str     # heading + aliases + display_text, used only for matching


def _parse_frontmatter(text: str):
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


def _cosine(a, b) -> float:
    if len(a.weights) > len(b.weights):
        a, b = b, a
    dot = sum(w * b.weights.get(term, 0.0) for term, w in a.weights.items())
    return dot / (a.norm * b.norm)


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


_LABEL_ONLY_RE = re.compile(r"^\*{0,2}[A-Za-z][A-Za-z '\-]*:\*{0,2}$")


def _is_label_only(item: str) -> bool:
    """True for lines that are just a role/section label with no actual
    content, e.g. '**For everyone:**' or '**For freshers:**' on its own
    line. These should never be returned as an answer by themselves —
    left in the index, their short, 'pure' vector (no content words to
    dilute it) lets them out-score the real answer lines that follow
    them, which is exactly backwards."""
    return bool(_LABEL_ONLY_RE.match(item.strip()))


def _chunk_document(doc_id: str, title: str, body: str) -> List[Chunk]:
    """Split a doc into small, independently-searchable chunks.

    Each '## Heading' section becomes its own context. Within a section,
    a '### Subheading' (e.g. a day name in a weekly menu) does NOT become
    its own chunk — it has no content by itself — instead it's folded into
    both the search text and the displayed text of every chunk that follows
    it, until the next '###' or '##' boundary. Each blank-line paragraph
    (and each bullet within it) becomes its own chunk, and the section
    heading + subheading + that section's 'Aliases:' line are folded into
    every chunk's SEARCH text (not necessarily its displayed text) so
    keywords like "curfew", "plumbing issue hostel", or "friday" survive
    and stay attached to the right chunk instead of being discarded or
    becoming a blank standalone answer.
    """
    chunks: List[Chunk] = []
    parts = re.split(r"\n(?=## )", body.strip())
    running_heading = title
    chunk_idx = 0

    for part in parts:
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = running_heading
        sub_heading = ""
        raw_paragraphs: List[tuple] = []
        current_para: List[str] = []
        alias_for_next_para = ""

        def _flush_para():
            nonlocal current_para, alias_for_next_para
            if current_para:
                raw_paragraphs.append(
                    ("\n".join(current_para), alias_for_next_para, sub_heading)
                )
                alias_for_next_para = ""
            current_para = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                heading = stripped[3:].strip()
                running_heading = heading
                sub_heading = ""
                alias_for_next_para = ""   # <-- add this line too
                continue
            if stripped.startswith("### "):
                _flush_para()
                alias_for_next_para = ""  # don't let a stale section-level
                                           # alias leak into whichever day
                                           # happens to be chunked first
                sub_heading = stripped[4:].strip()
                continue
            if stripped.startswith("# "):
                continue
            if not stripped:
                _flush_para()
                continue
            if stripped.lower().startswith("aliases:"):
                alias_for_next_para = stripped[len("aliases:"):].strip()
                continue
            current_para.append(stripped)
        _flush_para()

        for para_text, para_alias, para_sub in raw_paragraphs:
            plines = [l for l in para_text.splitlines() if l.strip()]
            if not plines:
                continue
            items = _split_paragraph_into_items(plines)
            for item in items:
                if not item.strip():
                    continue
                if _is_label_only(item):
                    continue
                chunk_idx += 1
                search_text = f"{heading} {para_sub} {para_alias} {item}"
                display_text = f"{para_sub} — {item.strip()}" if para_sub else item.strip()
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}#{chunk_idx}",
                        heading=f"{heading} — {para_sub}" if para_sub else heading,
                        display_text=display_text,
                        search_text=search_text,
                    )
                )
    return chunks


class DocumentStore:
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
        self.documents: List[SourceDoc] = []
        self.chunks: List[Chunk] = []
        self._chunk_vectors: List[_TfidfVector] = []
        self._idf: Dict[str, float] = {}
        self._load()

    def _load(self):
        self.documents = []
        self.chunks = []
        paths = sorted(glob.glob(os.path.join(self.docs_dir, "*.md")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            meta, body = _parse_frontmatter(raw)
            doc_id = os.path.splitext(os.path.basename(path))[0]
            doc = SourceDoc(
                doc_id=doc_id,
                title=meta.get("title") or doc_id,
                department=meta.get("department", "General Helpdesk"),
                required_info=meta.get("required_info", []),
                tags=meta.get("tags", []),
                body=body,
                filepath=path,
            )
            self.documents.append(doc)
            doc_chunks = _chunk_document(doc_id, doc.title, body)
            # NOTE: deliberately NOT injecting doc-level tags into every
            # chunk here. Doc-level tags mix vocabulary from all sections
            # (e.g. hostel.md's tags include both "mess" and "timings" for
            # two different sections), so blanket-applying them pollutes
            # every chunk in the doc and causes cross-section / cross-doc
            # collisions. Each chunk already carries its own heading +
            # that section's "Aliases:" line, which is the right-scoped
            # vocabulary for it.
            self.chunks.extend(doc_chunks)
        self._build_index()

    def _build_index(self):
        n_chunks = len(self.chunks)
        self._chunk_vectors = []
        self._idf = {}
        if n_chunks == 0:
            return

        tokenized = [_tokenize_stemmed(c.search_text) for c in self.chunks]

        df: Counter = Counter()
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1

        self._idf = {
            term: math.log((1 + n_chunks) / (1 + freq)) + 1.0
            for term, freq in df.items()
        }

        for tokens in tokenized:
            tf = Counter(tokens)
            weights = {term: count * self._idf[term] for term, count in tf.items()}
            self._chunk_vectors.append(_TfidfVector(weights))

    def reload(self):
        self._load()

    def _query_vector(self, query: str) -> _TfidfVector:
        tokens = _tokenize_stemmed(query)
        tf = Counter(tokens)
        weights = {
            term: count * self._idf.get(term, 0.0)
            for term, count in tf.items()
            if term in self._idf
        }
        return _TfidfVector(weights)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.chunks or not self._idf:
            return []
        q_vec = self._query_vector(query)
        if not q_vec.weights:
            return []
        query_term_count = len(q_vec.weights)

        scored = []
        for chunk, vec in zip(self.chunks, self._chunk_vectors):
            score = _cosine(q_vec, vec)
            matched_terms = len(set(q_vec.weights.keys()) & set(vec.weights.keys()))
            scored.append((score, matched_terms, chunk))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # Collapse to best chunk per doc for the top_k doc-level results,
        # but keep the winning chunk's text as the answer body.
        seen_docs = set()
        results = []
        for score, matched_terms, chunk in scored:
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            doc = self.get(chunk.doc_id)
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
                    "chunk_heading": chunk.heading,
                    "chunk_text": chunk.display_text,
                }
            )
            if len(results) >= top_k:
                break
        return results

    def known_term_ratio(self, query: str) -> float:
        tokens = _tokenize_stemmed(query)
        if not tokens:
            return 0.0
        known = sum(1 for t in tokens if t in self._idf)
        return known / len(tokens)

    def get(self, doc_id: str) -> Optional[SourceDoc]:
        for d in self.documents:
            if d.doc_id == doc_id:
                return d
        return None


def extract_best_snippet(body: str, query: str, max_sentences: int = 1) -> str:
    """Kept for backward compatibility / fallback only. With chunk-level
    search, callers should prefer results[i]['chunk_text'] directly."""
    return body[:400]