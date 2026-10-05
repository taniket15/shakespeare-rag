import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.vectorstore import FaissVectorStore

RRF_K = 60  # standard Reciprocal Rank Fusion constant; dampens the weight of top ranks
CANDIDATES = 50  # how deep each retriever's ranking goes into the fusion
KEYWORD_SLOTS = 2  # top keyword matches always kept, so exact wording can't be outvoted by the fusion

# Short names people use for works, beyond what can be derived from the titles
ALIASES = {
    "midsummer": "A Midsummer Night's Dream",
    "shrew": "The Taming of the Shrew",
    "lear": "King Lear",
    "much ado": "Much Ado About Nothing",
    "merry wives": "The Merry Wives of Windsor",
    "sonnets": "Shakespeare's Sonnets",
}


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes (’ and ') and turn other punctuation into spaces."""
    text = re.sub(r"[’']", "", text.lower())
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def quoted_phrases(query: str) -> list[str]:
    """Phrases of 2+ words in quotes. Single quotes count only at word edges, so "What's" isn't a quote."""
    phrases = re.findall(r'["“](.+?)["”]', query)
    phrases += re.findall(r"(?<![A-Za-z])['‘](.+?)['’](?![A-Za-z])", query)
    return [normalize(p) for p in phrases if len(p.split()) >= 2]


class HybridRetriever:
    """Combines FAISS vector search with BM25 keyword search using Reciprocal Rank Fusion.

    Vectors find passages with the same meaning; BM25 finds exact words such as quotes,
    character names and numbers ("Sonnet 18"), which embeddings tend to miss. If the
    question names a work (or a sonnet number), retrieval is limited to that work.
    """

    def __init__(self, store: FaissVectorStore):
        self.store = store
        self.normalized = [normalize(m["text"]) for m in store.metadata]
        self.bm25 = BM25Okapi([text.split() for text in self.normalized])

        titles = sorted({m["title"] for m in store.metadata if "title" in m})
        self.title_patterns: dict[str, set[str]] = {}

        def add(pattern: str, title: str):
            self.title_patterns.setdefault(pattern, set()).add(title)

        for title in titles:
            name = normalize(title)
            add(name, title)
            if name.startswith(" the "):
                add(name[4:], title)  # "The Tempest" -> "tempest"
            if " part " in name:
                add(name.split(" part ")[0] + " ", title)  # "Henry IV" -> both parts
        for alias, title in ALIASES.items():
            add(normalize(alias), title)

    def detect_filter(self, query: str) -> dict[str, Any]:
        """Work titles and sonnet number named in the question, if any."""
        q = normalize(query)
        if m := re.search(r" sonnet (\d+) ", q):
            return {"title": {"Shakespeare's Sonnets"}, "section": f"Sonnet {m.group(1)}"}
        titles = {t for pattern, ts in self.title_patterns.items() if pattern in q for t in ts}
        return {"title": titles} if titles else {}

    def synopses(self, titles: set[str]) -> list[int]:
        return [i for i, m in enumerate(self.store.metadata) if m.get("title") in titles and m.get("section") == "Synopsis"]

    def _allowed(self, idx: int, flt: dict[str, Any]) -> bool:
        meta = self.store.metadata[idx]
        if "title" in flt and meta.get("title") not in flt["title"]:
            return False
        return "section" not in flt or meta.get("section") == flt["section"]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        flt = self.detect_filter(query)

        # Vector ranking (search everything when filtering, then keep only allowed chunks)
        depth = self.store.index.ntotal if flt else CANDIDATES
        vector_hits = [r for r in self.store.query(query, top_k=depth) if self._allowed(r["index"], flt)]
        vector_rank = [r["index"] for r in vector_hits[:CANDIDATES]]

        # Keyword ranking
        scores = self.bm25.get_scores(tokenize(query))
        keyword_rank = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0 and self._allowed(int(i), flt)]
        keyword_rank = keyword_rank[:CANDIDATES]

        fused: dict[int, float] = {}
        for ranking in (vector_rank, keyword_rank):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1 / (RRF_K + rank + 1)

        # Rank fusion favors passages that do moderately well in both lists, which can bury a passage
        # that's top for keywords only. When the question quotes a line, put passages containing it first.
        for phrase in quoted_phrases(query):
            for idx, text in enumerate(self.normalized):
                if phrase in text and self._allowed(idx, flt):
                    fused[idx] = fused.get(idx, 0.0) + 1.0

        ranked = sorted(fused, key=fused.get, reverse=True)
        reserved = [idx for idx in keyword_rank[:KEYWORD_SLOTS] if idx not in ranked[:top_k]]
        best = ranked[:top_k - len(reserved)] + reserved

        # A question that names a play gets that play's synopsis too, so overview questions
        # ("What happens in As You Like It?") have the whole plot, not just a few scenes
        if flt.get("title") and "section" not in flt and len(flt["title"]) <= 2:
            best += [idx for idx in self.synopses(flt["title"]) if idx not in best]
        return [{"index": idx, "score": fused.get(idx, 0.0), "metadata": self.store.metadata[idx]} for idx in best]
