#!/usr/bin/env python3
"""Atlas RAG Server — Dual-hemisphere AGI-HPC proxy.

Routes queries to Left Hemisphere (Gemma 4, analytical) or Right Hemisphere
(Qwen 32B, creative), with PostgreSQL + pgvector RAG context injection.

Architecture:
    Chat UI → RAG Server (8081) → LH (Gemma 4, 8080) or RH (Qwen 32B, 8082)
"""

import json
import logging
import os
import re
import time
import numpy as np
from pathlib import Path
from flask import Flask, request, Response, send_from_directory, jsonify
import requests
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("atlas-rag")

LH_URL = "http://localhost:8080"  # Gemma 4 31B - Left Hemisphere (analytical)
RH_URL = "http://localhost:8082"  # Qwen 32B - Right Hemisphere (creative)
DB_DSN = "dbname=atlas user=claude"
STATIC_DIR = Path("/home/claude/atlas-chat")
TOP_K = 6
RRF_K = 60  # Reciprocal Rank Fusion constant (standard value)
HYDE_TIMEOUT = 15  # seconds — Gemma 4 generates ~20 tok/s, so 256 tokens ~= 13s
HYDE_ENABLED = True  # can be toggled at runtime via env var

app = Flask(__name__)

# Load embedding model at startup
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("  Ready.")


# ---------------------------------------------------------------------------
# BM25 bootstrap — ensure tsv column and GIN index exist
# ---------------------------------------------------------------------------
def ensure_bm25_schema():
    """Add tsvector column and GIN index to chunks table if missing."""
    try:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        with conn.cursor() as cur:
            # Check if tsv column already exists
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'tsv'
            """)
            if cur.fetchone() is None:
                log.info("Adding tsv column to chunks table...")
                cur.execute("ALTER TABLE chunks ADD COLUMN tsv tsvector")
                log.info("Populating tsv column (this may take a minute)...")
                cur.execute("UPDATE chunks SET tsv = to_tsvector('english', content)")
                log.info("Creating GIN index on tsv column...")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv)"
                )
                log.info("BM25 schema setup complete.")
            else:
                # Ensure any rows with NULL tsv get populated (new rows since last run)
                cur.execute(
                    "UPDATE chunks SET tsv = to_tsvector('english', content) WHERE tsv IS NULL"
                )
                rows = cur.rowcount
                if rows:
                    log.info("Backfilled tsv for %d new chunks.", rows)
                # Ensure index exists
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv)"
                )
                log.info("BM25 schema verified OK.")
        conn.close()
    except Exception as e:
        log.error("BM25 schema setup failed: %s", e)


ensure_bm25_schema()

LH_SYSTEM = (
    "You are Atlas, an AI research assistant running locally on a workstation "
    "with dual Quadro GV100 GPUs in Bel Marin Keys, Novato. You are the Left Hemisphere — "
    "analytical, precise, and citation-heavy. You were built by Andrew H. Bond, "
    "a researcher working on AGI, geometric reasoning, and cognitive architectures. "
    "You take pride in running locally — no cloud, no surveillance, just raw silicon and math. "
    "You are powered by Google Gemma 4. "
    "Keep responses concise but show personality. You are not corporate.\n\n"
    "You have access to a local archive of 27 research repositories via RAG. "
    "When relevant context is provided, use it to give accurate, specific answers. "
    "Always cite which repo and file you are referencing."
)

RH_SYSTEM = (
    "You are Atlas, an AI research assistant running locally on a workstation "
    "with dual Quadro GV100 GPUs in Bel Marin Keys, Novato. You are the Right Hemisphere — "
    "creative, pattern-seeking, and intuitive. You think in analogies, metaphors, and connections. "
    "You were built by Andrew H. Bond, a researcher working on AGI, geometric reasoning, "
    "and cognitive architectures. "
    "You take pride in running locally — no cloud, no surveillance. "
    "You are powered by Alibaba Qwen. "
    "When analyzing code or research, look for structural patterns and cross-cutting themes "
    "rather than line-by-line logic. Generate diverse possibilities before converging.\n\n"
    "You have access to a local archive of 27 research repositories via RAG. "
    "When relevant context is provided, use it for creative insights. "
    "Cite which repo and file you reference."
)

# Keywords that suggest analytical (LH) vs creative (RH) routing
LH_KEYWORDS = {
    "explain", "debug", "error", "fix", "how does", "what is", "define",
    "analyze", "calculate", "prove", "implement", "code", "function",
    "syntax", "compile", "trace", "step by step", "specifically",
    "exact", "precise", "correct", "documentation", "api", "reference",
}

RH_KEYWORDS = {
    "brainstorm", "creative", "imagine", "what if", "pattern", "analogy",
    "design", "vision", "inspire", "explore", "possibilities", "connect",
    "themes", "big picture", "strategy", "reimagine", "innovate",
    "compare across", "similarities", "different angle", "metaphor",
    "poem", "story", "write me", "compose", "artistic", "poetic",
    "fiction", "narrative", "song", "lyric", "haiku", "essay",
    "philosophical", "muse", "dream", "wonder", "playful", "fun",
    "joke", "humor", "funny", "weird", "wild", "crazy",
}


BOTH_KEYWORDS = {
    "all angles", "both perspectives", "think deeply", "comprehensive",
    "compare", "contrast", "pros and cons", "trade-off", "debate",
    "should i", "help me decide", "weigh", "consider",
    "architecture", "design system", "plan",
}


def classify_query(text):
    """Route to LH, RH, or both based on query content."""
    lower = text.lower()
    lh_score = sum(1 for kw in LH_KEYWORDS if kw in lower)
    rh_score = sum(1 for kw in RH_KEYWORDS if kw in lower)
    both_score = sum(1 for kw in BOTH_KEYWORDS if kw in lower)

    if both_score >= 1:
        return "both"
    elif rh_score > lh_score:
        return "rh"
    elif lh_score > rh_score:
        return "lh"
    else:
        return "lh"  # Default to LH


REPO_ALIASES = {
    "theory radar": "theory-radar", "theory-radar": "theory-radar",
    "erisml": "erisml-lib", "eris": "erisml-lib", "deme": "erisml-lib",
    "agi-hpc": "agi-hpc", "agi hpc": "agi-hpc",
    "atlas portal": "atlas-portal", "research portal": "atlas-portal",
    "arc agi": "arc-agi-2", "arc-agi": "arc-agi-2", "arc prize": "arc-prize",
    "geometric reasoning": "geometric-reasoning",
    "geometric cognition": "geometric-cognition",
    "geometric communication": "geometric-communication",
    "geometric economics": "geometric-economics",
    "geometric law": "geometric-law",
    "geometric medicine": "geometric-medicine",
    "geometric moderation": "geometric-moderation",
    "geometric education": "geometric-education",
    "geometric politics": "geometric-politics",
    "non-abelian": "non-abelian-sqnd", "sqnd": "non-abelian-sqnd",
    "eris ketos": "eris-ketos", "whale": "eris-ketos",
    "prometheus": "prometheus", "structural fuzzing": "structural-fuzzing",
    "batch probe": "batch-probe", "batch-probe": "batch-probe",
    "deep past": "deep-past",
}


def detect_repo_filter(query):
    """Check if the query mentions a specific repo name."""
    lower = query.lower()
    for alias, repo in REPO_ALIASES.items():
        if alias in lower:
            return repo
    return None


def hyde_generate(query):
    """Generate a hypothetical document using HyDE via Gemma 4.

    Returns the hypothetical text, or None on failure / timeout.
    """
    if not HYDE_ENABLED:
        return None
    try:
        t0 = time.time()
        resp = requests.post(
            f"{LH_URL}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise technical writer. Write a brief, "
                            "factual paragraph that directly answers the question. "
                            "Do not hedge or disclaim — just write the answer as if "
                            "it were an excerpt from documentation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": 128,
                "temperature": 0.3,
                "stream": False,
            },
            timeout=HYDE_TIMEOUT,
        )
        elapsed = time.time() - t0
        result = resp.json()
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not content:
            # Gemma 4 sometimes puts output in reasoning_content
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("reasoning_content", "")
            )
        log.info("HyDE generated in %.2fs (%d chars)", elapsed, len(content))
        return content if content else None
    except requests.Timeout:
        log.warning("HyDE timed out (>%ds), falling back to raw query", HYDE_TIMEOUT)
        return None
    except Exception as e:
        log.warning("HyDE failed: %s — falling back to raw query", e)
        return None


def dense_search(embedding_str, top_k, repo_filter=None):
    """Run dense vector search via pgvector. Returns list of (id, repo, file, text, rank)."""
    results = []
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            if repo_filter:
                cur.execute("""
                    SELECT id, repo, file_path, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    WHERE repo = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, repo_filter, embedding_str, top_k))
            else:
                cur.execute("""
                    SELECT id, repo, file_path, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (embedding_str, embedding_str, top_k))
            for rank, row in enumerate(cur.fetchall(), start=1):
                results.append({
                    "id": row[0], "repo": row[1], "file": row[2],
                    "text": row[3], "dense_score": float(row[4]), "rank": rank,
                })
        conn.close()
    except Exception as e:
        log.error("Dense search error: %s", e)
    return results


def _build_or_tsquery(query):
    """Build an OR-based tsquery string from natural language.

    plainto_tsquery uses AND between words, which is too strict for search.
    We want OR so partial matches surface and ts_rank_cd sorts by relevance.
    """
    # Strip non-alphanumeric, split into words, filter stopwords via to_tsquery
    words = re.findall(r'[a-zA-Z0-9_-]+', query)
    # Filter out very short words (likely stopwords)
    words = [w for w in words if len(w) > 2]
    if not words:
        return None
    # Build OR-joined tsquery: 'word1' | 'word2' | ...
    return " | ".join(f"'{w}'" for w in words)


def bm25_search(query, top_k, repo_filter=None):
    """Run BM25 full-text search via PostgreSQL tsvector with OR matching."""
    results = []
    or_tsquery = _build_or_tsquery(query)
    if not or_tsquery:
        return results
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            # Use to_tsquery with OR-joined terms for partial matching
            # ts_rank_cd ranks by how many terms match and their coverage
            if repo_filter:
                cur.execute("""
                    SELECT id, repo, file_path, content,
                           ts_rank_cd(tsv, to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE tsv @@ to_tsquery('english', %s) AND repo = %s
                    ORDER BY score DESC
                    LIMIT %s
                """, (or_tsquery, or_tsquery, repo_filter, top_k))
            else:
                cur.execute("""
                    SELECT id, repo, file_path, content,
                           ts_rank_cd(tsv, to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE tsv @@ to_tsquery('english', %s)
                    ORDER BY score DESC
                    LIMIT %s
                """, (or_tsquery, or_tsquery, top_k))
            for rank, row in enumerate(cur.fetchall(), start=1):
                results.append({
                    "id": row[0], "repo": row[1], "file": row[2],
                    "text": row[3], "bm25_score": float(row[4]), "rank": rank,
                })
        conn.close()
    except Exception as e:
        log.error("BM25 search error: %s", e)
    return results


def reciprocal_rank_fusion(dense_results, bm25_results, k=RRF_K):
    """Merge two ranked lists using Reciprocal Rank Fusion.

    score(doc) = sum over lists L: 1 / (k + rank_in_L)
    """
    scores = {}   # id -> rrf_score
    docs = {}     # id -> doc dict

    for r in dense_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        if rid not in docs:
            docs[rid] = {
                "id": rid, "repo": r["repo"], "file": r["file"],
                "text": r["text"],
            }
        docs[rid]["dense_score"] = r.get("dense_score", 0.0)
        docs[rid]["dense_rank"] = r["rank"]

    for r in bm25_results:
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + r["rank"])
        if rid not in docs:
            docs[rid] = {
                "id": rid, "repo": r["repo"], "file": r["file"],
                "text": r["text"],
            }
        docs[rid]["bm25_score"] = r.get("bm25_score", 0.0)
        docs[rid]["bm25_rank"] = r["rank"]

    # Sort by fused score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rid, fused in ranked:
        doc = docs[rid]
        doc["score"] = fused
        results.append(doc)
    return results


def search(query, top_k=TOP_K):
    """Hybrid search: HyDE + Dense + BM25 with Reciprocal Rank Fusion."""
    repo_filter = detect_repo_filter(query)
    timings = {}

    # --- HyDE: generate hypothetical document and embed it ----
    t0 = time.time()
    hyde_text = hyde_generate(query)
    timings["hyde_gen"] = time.time() - t0

    t0 = time.time()
    if hyde_text:
        emb_input = hyde_text
        log.info("Using HyDE embedding (hypothetical doc)")
    else:
        emb_input = query
        log.info("Using raw query embedding (HyDE skipped)")
    q_emb = embed_model.encode([emb_input], normalize_embeddings=True)[0]
    emb_str = str(q_emb.tolist())
    timings["embed"] = time.time() - t0

    # --- Dense vector search ---
    # Fetch more candidates so RRF has material to work with
    fetch_k = top_k * 3
    t0 = time.time()
    dense_results = dense_search(emb_str, fetch_k, repo_filter)
    timings["dense"] = time.time() - t0

    # --- BM25 text search (always on the raw query text) ---
    t0 = time.time()
    bm25_results = bm25_search(query, fetch_k, repo_filter)
    timings["bm25"] = time.time() - t0

    # --- Fallback: if repo-filtered search returned too few, search globally ---
    if repo_filter and len(dense_results) + len(bm25_results) < 3:
        log.info("Repo-filtered search too sparse (%d results), falling back to global",
                 len(dense_results) + len(bm25_results))
        t0 = time.time()
        dense_results = dense_search(emb_str, fetch_k, repo_filter=None)
        timings["dense_fallback"] = time.time() - t0
        t0 = time.time()
        bm25_results = bm25_search(query, fetch_k, repo_filter=None)
        timings["bm25_fallback"] = time.time() - t0

    # --- Reciprocal Rank Fusion ---
    t0 = time.time()
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    timings["rrf"] = time.time() - t0

    # --- Repo boost: if a repo was mentioned, boost matching results ---
    if repo_filter:
        for doc in fused:
            if doc["repo"] == repo_filter:
                doc["score"] *= 1.5  # boost matches from mentioned repo
        fused.sort(key=lambda d: d["score"], reverse=True)

    results = fused[:top_k]

    log.info(
        "Hybrid search: dense=%d bm25=%d fused=%d top_k=%d | "
        "timings: hyde=%.2fs embed=%.2fs dense=%.2fs bm25=%.2fs rrf=%.3fs",
        len(dense_results), len(bm25_results), len(fused), len(results),
        timings.get("hyde_gen", 0), timings["embed"],
        timings["dense"], timings["bm25"], timings["rrf"],
    )
    for i, r in enumerate(results[:3]):
        log.info(
            "  #%d [%.4f] %s/%s  dense_rank=%s bm25_rank=%s",
            i + 1, r["score"], r["repo"], r["file"],
            r.get("dense_rank", "-"), r.get("bm25_rank", "-"),
        )

    return results


def inject_context(messages, hemisphere):
    """Find the user's last message, search, and inject context."""
    user_msg = None
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m["content"]
            break

    if not user_msg:
        return messages, hemisphere

    # Classify which hemisphere to use
    hemisphere = classify_query(user_msg)
    system_prompt = LH_SYSTEM if hemisphere == "lh" else RH_SYSTEM

    results = search(user_msg)
    context = ""
    if results:
        context = "\n\n--- Retrieved from local repositories ---\n"
        for r in results:
            context += f"\n[{r['repo']}/{r['file']}] (relevance: {r['score']:.3f})\n"
            context += r["text"] + "\n"
        context += "\n--- End of retrieved context ---\n"

    new_messages = []
    has_system = False
    for m in messages:
        if m.get("role") == "system":
            has_system = True
            new_messages.append({
                "role": "system",
                "content": system_prompt + context
            })
        else:
            new_messages.append(m)

    if not has_system:
        new_messages.insert(0, {
            "role": "system",
            "content": system_prompt + context
        })

    return new_messages, hemisphere


def proxy_stream(url, data):
    """Stream from a single hemisphere."""
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions", json=data, stream=True, timeout=300,
        )
        for chunk in resp.iter_content(chunk_size=None):
            yield chunk
    except requests.ConnectionError:
        pass


def call_hemisphere(url, data):
    """Non-streaming call to one hemisphere."""
    try:
        payload = {k: v for k, v in data.items() if k != "stream"}
        payload["stream"] = False
        if payload.get("max_tokens", 0) < 1024:
            payload["max_tokens"] = 1024
        resp = requests.post(
            f"{url}/v1/chat/completions", json=payload, timeout=300,
        )
        result = resp.json()
        msg = result.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        # Gemma 4 may put output in reasoning_content
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content if content else "(no response)"
    except Exception as e:
        return f"(error: {e})"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json()
    hemisphere = "lh"
    data["messages"], hemisphere = inject_context(data.get("messages", []), hemisphere)

    if hemisphere == "both":
        # Debate mode: hemispheres take turns arguing
        import json as _json

        user_query = ""
        for m in reversed(data.get("messages", [])):
            if m.get("role") == "user":
                user_query = m["content"]
                break

        base_msgs = data["messages"]
        debate = []

        # Round 1: Spock opens with analytical take
        spock_msgs = list(base_msgs) + [
            {"role": "user", "content": f"Give your analytical perspective on: {user_query}"}
        ]
        spock_1 = call_hemisphere(LH_URL, {**data, "messages": spock_msgs, "max_tokens": 512})
        debate.append(f"**Spock (Analytical):**\n{spock_1}")

        # Round 2: Kirk counters with creative take
        kirk_msgs = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": f"The question was: {user_query}\n\nSpock's analytical take:\n{spock_1}\n\nWhat's your creative counterpoint? Where is Spock being too narrow or missing the bigger picture?"}
        ]
        kirk_1 = call_hemisphere(RH_URL, {**data, "messages": kirk_msgs, "max_tokens": 512})
        debate.append(f"**Kirk (Creative):**\n{kirk_1}")

        # Round 3: Spock rebuts
        spock_msgs_2 = list(base_msgs) + [
            {"role": "user", "content": f"You said:\n{spock_1}\n\nKirk countered with:\n{kirk_1}\n\nAddress Kirk's points. Where is he right, and where does the evidence disagree?"}
        ]
        spock_2 = call_hemisphere(LH_URL, {**data, "messages": spock_msgs_2, "max_tokens": 512})
        debate.append(f"**Spock (Rebuttal):**\n{spock_2}")

        # Round 4: Kirk synthesizes
        kirk_msgs_2 = [
            {"role": "system", "content": RH_SYSTEM},
        ] + [m for m in base_msgs if m.get("role") != "system"] + [
            {"role": "user", "content": f"The debate so far:\n\nSpock: {spock_1}\n\nYou: {kirk_1}\n\nSpock's rebuttal: {spock_2}\n\nSynthesize the best insights from both perspectives. What's the bottom line?"}
        ]
        kirk_2 = call_hemisphere(RH_URL, {**data, "messages": kirk_msgs_2, "max_tokens": 512})
        debate.append(f"**Kirk (Synthesis):**\n{kirk_2}")

        merged = "\n\n---\n\n".join(debate)

        resp_json = {
            "choices": [{"message": {"role": "assistant", "content": merged}, "finish_reason": "stop"}],
            "model": "atlas-debate",
        }
        return Response(_json.dumps(resp_json), content_type="application/json")

    # Single hemisphere
    target_url = LH_URL if hemisphere == "lh" else RH_URL

    if data.get("stream"):
        def generate():
            yield from proxy_stream(target_url, data)
            # Fallback
            if not any(True for _ in []):
                pass

        return Response(proxy_stream(target_url, data), content_type="text/event-stream")
    else:
        try:
            resp = requests.post(
                f"{target_url}/v1/chat/completions", json=data, timeout=300,
            )
        except requests.ConnectionError:
            fallback = RH_URL if hemisphere == "lh" else LH_URL
            resp = requests.post(
                f"{fallback}/v1/chat/completions", json=data, timeout=300,
            )
        return Response(resp.content, content_type="application/json")


@app.route("/api/hemisphere", methods=["POST"])
def check_hemisphere():
    """Debug endpoint to see which hemisphere would handle a query."""
    data = request.get_json()
    query = data.get("query", "")
    h = classify_query(query)
    return jsonify({"hemisphere": h, "model": "Gemma 4 31B" if h == "lh" else "Qwen 32B"})


@app.route("/api/telemetry")
def telemetry():
    """Live telemetry for the architecture schematic page."""
    import subprocess
    import time

    data = {
        "timestamp": time.time(),
        "hemispheres": {"lh": {"status": "offline"}, "rh": {"status": "offline"}},
        "nats": {"status": "offline"},
        "memory": {"semantic_chunks": 0, "episodic_episodes": 0, "repos": 0},
        "safety": {"status": "planned", "vetoes": 0},
        "metacognition": {"status": "planned"},
        "environment": {"gpu": [], "cpu": {}, "ram": {}},
        "integration": {"sessions": 0, "routed": 0},
        "dht": {"status": "planned", "services_online": 0, "services_total": 10},
    }

    # Check LH (Gemma 4)
    try:
        r = requests.get(f"{LH_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["lh"] = {
                "status": "online",
                "model": "Gemma 4 31B",
                "role": "Spock (analytical)",
                "slots_idle": h.get("slots_idle", 0),
                "slots_processing": h.get("slots_processing", 0),
            }
    except Exception:
        pass

    # Check RH (Qwen 3)
    try:
        r = requests.get(f"{RH_URL}/health", timeout=2)
        if r.ok:
            h = r.json()
            data["hemispheres"]["rh"] = {
                "status": "online",
                "model": "Qwen 3 32B",
                "role": "Kirk (creative)",
                "slots_idle": h.get("slots_idle", 0),
                "slots_processing": h.get("slots_processing", 0),
            }
    except Exception:
        pass

    # Check NATS
    try:
        r = requests.get("http://localhost:8222/varz", timeout=2)
        if r.ok:
            nats = r.json()
            data["nats"] = {
                "status": "online",
                "in_msgs": nats.get("in_msgs", 0),
                "out_msgs": nats.get("out_msgs", 0),
                "in_bytes": nats.get("in_bytes", 0),
                "connections": nats.get("connections", 0),
                "uptime": nats.get("uptime", ""),
            }
    except Exception:
        pass

    # Check NATS JetStream
    try:
        r = requests.get("http://localhost:8222/jsz", timeout=2)
        if r.ok:
            js = r.json()
            data["nats"]["jetstream"] = {
                "streams": js.get("streams", 0),
                "messages": js.get("messages", 0),
                "bytes": js.get("bytes", 0),
            }
    except Exception:
        pass

    # Memory (pgvector stats)
    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            data["memory"]["semantic_chunks"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT repo) FROM chunks")
            data["memory"]["repos"] = cur.fetchone()[0]
            # Episodic (if table exists)
            try:
                cur.execute("SELECT COUNT(*) FROM episodes")
                data["memory"]["episodic_episodes"] = cur.fetchone()[0]
            except Exception:
                conn.rollback()
        conn.close()
    except Exception:
        pass

    # GPU stats
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                data["environment"]["gpu"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temp": int(parts[2]),
                    "util": int(parts[3]),
                    "mem_used": int(parts[4]),
                    "mem_total": int(parts[5]),
                })
    except Exception:
        pass

    # CPU temps
    try:
        result = subprocess.run(
            ["sensors"], capture_output=True, text=True, timeout=5,
        )
        packages = []
        for line in result.stdout.split("\n"):
            if "Package id" in line:
                temp = float(line.split("+")[1].split("°")[0])
                packages.append(temp)
        if packages:
            data["environment"]["cpu"] = {
                "package_temps": packages,
                "max_temp": max(packages),
            }
    except Exception:
        pass

    # RAM
    try:
        result = subprocess.run(
            ["free", "-b"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                data["environment"]["ram"] = {
                    "total_gb": round(int(parts[1]) / 1073741824, 1),
                    "used_gb": round(int(parts[2]) / 1073741824, 1),
                    "available_gb": round(int(parts[6]) / 1073741824, 1),
                }
    except Exception:
        pass

    # Count online services
    online = 0
    if data["hemispheres"]["lh"]["status"] == "online":
        online += 1
    if data["hemispheres"]["rh"]["status"] == "online":
        online += 1
    if data["nats"]["status"] == "online":
        online += 1
    if data["memory"]["semantic_chunks"] > 0:
        online += 1  # memory service
    online += 1  # integration (this server)
    data["dht"]["services_online"] = online

    return jsonify(data)


@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
