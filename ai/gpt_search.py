#!/usr/bin/env python3
"""
gpt-search: Ask questions about your ChatGPT history. Instant answers via Claude.

Usage:
    gpt-search what's my salary
    gpt-search --index          # rebuild index after new export
"""

import json
import glob
import os
import sys
import re
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
GPT_DIR = Path("/srv/mergerfs/PROMETHEUS/PERSONAL/GPT")
INDEX_PATH = PROJECT_ROOT / "gpt_index.json"
ENV_PATH = PROJECT_ROOT / ".env"


def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("ANTHROPIC_API_KEY")


def extract_messages(convo):
    mapping = convo.get("mapping", {})
    nodes = []
    for node_id, node in mapping.items():
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", {})
        parts = content.get("parts", [])
        text_parts = [p for p in parts if isinstance(p, str) and p.strip()]
        if text_parts:
            text = "\n".join(text_parts)
            create_time = msg.get("create_time") or 0
            nodes.append((create_time, role, text))
    nodes.sort(key=lambda x: x[0])
    return [{"role": r, "text": t} for _, r, t in nodes]


def build_index():
    print("Building index from ChatGPT export...")
    convo_files = sorted(glob.glob(str(GPT_DIR / "conversations-*.json")))
    if not convo_files:
        print(f"No conversation files found in {GPT_DIR}")
        sys.exit(1)

    index = []
    total = 0

    for fpath in convo_files:
        print(f"  {os.path.basename(fpath)}...", end="", flush=True)
        with open(fpath) as f:
            convos = json.load(f)
        for convo in convos:
            messages = extract_messages(convo)
            if not messages:
                continue

            # Build full transcript for context (capped at 5000 chars)
            transcript_lines = []
            for m in messages:
                prefix = "USER" if m["role"] == "user" else "ASSISTANT"
                transcript_lines.append(f"[{prefix}]: {m['text']}")
            full_transcript = "\n\n".join(transcript_lines)

            all_text = " ".join(m["text"] for m in messages)

            entry = {
                "id": convo.get("id", convo.get("conversation_id", "")),
                "title": convo.get("title", "Untitled"),
                "created": convo.get("create_time", 0),
                "model": convo.get("default_model_slug", ""),
                "msg_count": len(messages),
                "transcript": full_transcript[:5000],
                "words": dict(Counter(tokenize(all_text)).most_common(200)),
            }
            index.append(entry)
            total += 1
        print(f" {len(convos)}")

    doc_freq = defaultdict(int)
    for entry in index:
        for word in entry["words"]:
            doc_freq[word] += 1

    index_data = {
        "version": 2,
        "total_conversations": total,
        "built_at": time.time(),
        "doc_freq": dict(doc_freq),
        "conversations": index,
    }

    with open(INDEX_PATH, "w") as f:
        json.dump(index_data, f)

    size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)
    print(f"Done: {total} conversations, {size_mb:.1f} MB index")


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def search(query, index_data, top_n=5):
    query_terms = tokenize(query)
    if not query_terms:
        return []

    total_docs = index_data["total_conversations"]
    doc_freq = index_data["doc_freq"]
    results = []

    for convo in index_data["conversations"]:
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

        if score > 0:
            results.append((score, convo))

    results.sort(key=lambda x: -x[0])
    return results[:top_n]


def ask_claude(question, results):
    import anthropic

    api_key = load_env()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    context_parts = []
    for i, (score, convo) in enumerate(results[:3]):
        from datetime import datetime
        created = datetime.fromtimestamp(convo["created"]).strftime("%Y-%m-%d") if convo["created"] else "unknown"
        context_parts.append(
            f'=== "{convo["title"]}" ({created}, {convo["msg_count"]} msgs) ===\n{convo["transcript"]}'
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant. The user is searching their past ChatGPT conversations. "
        "Answer their question based on the conversations below. Be concise and direct. "
        "If the conversations don't contain the answer, say so."
    )

    # Stream for instant feel
    print()
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": f"{question}\n\n{context}"}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")


def main():
    args = sys.argv[1:]

    if not args:
        print("Usage: gpt-search <question>")
        print("       gpt-search --index")
        return

    if args[0] == "--index":
        build_index()
        return

    question = " ".join(args)

    if not INDEX_PATH.exists():
        print("No index found. Run: gpt-search --index")
        sys.exit(1)

    with open(INDEX_PATH) as f:
        index_data = json.load(f)

    results = search(question, index_data)

    if not results:
        print("Nothing found.")
        return

    ask_claude(question, results)


if __name__ == "__main__":
    main()
