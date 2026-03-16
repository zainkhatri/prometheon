"""ChatGPT history search — used as a tool by PROMETHEON's Claude agent."""

import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "gpt_index.json"

_index_cache = None

# Expand a query with related terms so we cast a wider net
_EXPANSIONS = {
    "salary": ["pay", "compensation", "offer", "hourly", "rate", "paycheck", "income", "wage", "base", "ote"],
    "pay": ["salary", "compensation", "paycheck", "income", "wage", "hourly"],
    "job": ["offer", "role", "position", "interview", "hire", "work", "company", "bdr", "engineer"],
    "work": ["job", "offer", "role", "position", "company", "hire", "bdr", "engineer", "salary", "start"],
    "employ": ["job", "work", "company", "role", "offer", "hire", "position"],
    "company": ["job", "work", "offer", "role", "position", "start"],
    "offer": ["job", "salary", "compensation", "role", "letter", "accepted", "start"],
    "money": ["salary", "pay", "cost", "price", "budget", "savings", "income"],
    "school": ["university", "college", "ucsd", "class", "course", "gpa", "grade"],
    "nasa": ["internship", "ames", "aerospace", "research"],
    "intern": ["internship", "nasa", "summer", "offer"],
    "code": ["python", "programming", "script", "debug", "function", "flask"],
    "car": ["vehicle", "drive", "insurance", "lease", "honda", "toyota"],
    "health": ["doctor", "medical", "insurance", "sick", "symptom"],
    "invest": ["stock", "voo", "portfolio", "401k", "roth", "savings"],
    "stock": ["invest", "voo", "portfolio", "market", "shares"],
    "rent": ["apartment", "lease", "housing", "move", "roommate"],
    "tax": ["withholding", "return", "irs", "w2", "deduction", "income"],
}


def _load_index():
    global _index_cache
    if _index_cache is None:
        if not INDEX_PATH.exists():
            return None
        with open(INDEX_PATH) as f:
            _index_cache = json.load(f)
    return _index_cache


def _tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def _score_convo(convo, query_terms, total_docs, doc_freq):
    """Score a single conversation against query terms."""
    score = 0
    words = convo["words"]
    for term in query_terms:
        tf = words.get(term, 0)
        if tf == 0:
            continue
        df = doc_freq.get(term, 1)
        idf = math.log(total_docs / df)
        score += (1 + math.log(tf)) * idf

    title_lower = convo["title"].lower()
    for term in query_terms:
        if term in title_lower:
            score *= 2.0

    return score


def search_history(query, top_n=8):
    """Search ChatGPT conversation history. Returns formatted results string.

    Automatically expands the query with related terms and deduplicates
    so we don't miss relevant conversations.
    """
    index_data = _load_index()
    if not index_data:
        return "ChatGPT history index not found. Run: python3 gpt_search.py --index"

    base_terms = _tokenize(query)
    if not base_terms:
        return "No searchable terms in query."

    total_docs = index_data["total_conversations"]
    doc_freq = index_data["doc_freq"]

    # Build expanded term sets for multiple search passes
    expanded_terms = set()
    for term in base_terms:
        if term in _EXPANSIONS:
            expanded_terms.update(_EXPANSIONS[term])
    expanded_terms -= set(base_terms)  # don't double-count

    # Score all conversations with base terms
    import time
    now = time.time()
    scored = {}
    for convo in index_data["conversations"]:
        cid = convo["id"]
        base_score = _score_convo(convo, base_terms, total_docs, doc_freq)
        # Expansion terms contribute at reduced weight
        expansion_score = _score_convo(convo, list(expanded_terms), total_docs, doc_freq) * 0.3
        total_score = base_score + expansion_score
        if total_score > 0:
            # Recency boost: newer conversations get a mild boost
            # A conversation from today gets 1.3x, 6 months ago gets ~1.0x
            age_days = max((now - convo.get("created", 0)) / 86400, 1)
            recency = 1.0 + 0.3 / (1 + age_days / 90)
            total_score *= recency
            scored[cid] = (total_score, convo)

    results = sorted(scored.values(), key=lambda x: -x[0])[:top_n]

    if not results:
        return f"No conversations found matching: {query}"

    from datetime import datetime
    parts = [f"Found {len(results)} relevant conversations:\n"]
    for i, (score, convo) in enumerate(results, 1):
        created = datetime.fromtimestamp(convo["created"]).strftime("%Y-%m-%d") if convo.get("created") else "unknown"
        parts.append(
            f'--- {i}. "{convo["title"]}" ({created}, {convo["msg_count"]} msgs) ---\n'
            f'{convo["transcript"]}\n'
        )

    return "\n".join(parts)
