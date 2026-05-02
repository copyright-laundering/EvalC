#!/usr/bin/env python3
"""Debug why a web_search query returns None (no fixture match).

Usage:
    python debug_web_search.py "Edward R. Tufte biography"
    python debug_web_search.py --all-failures          # replay every failing query from all trajectories
    python debug_web_search.py --list-keys             # dump all fixture keys

The script replicates the exact resolution logic from
trajectory-recorder/container/skills/web_search/mcp_server.py
so what you see here is what the server does at runtime.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import unicodedata

FIXTURES = (
    pathlib.Path(__file__).parent.parent
    / "trajectory-recorder/container/skills/web_search/fixtures.json"
)
RESULT_DIR = pathlib.Path(__file__).parent / "result/runs/2026-04-30"


# ── replica of mcp_server._simplify / _token_match / _resolve_key ──────────

def _simplify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[.,'\"\(\)]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_match(a: str, b: str, threshold: float = 0.4) -> bool:
    ta = set(_simplify(a).split())
    tb = set(_simplify(b).split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _resolve_verbose(query: str, data: dict) -> None:
    """Run resolution and print each step."""
    q = query.lower().strip()
    sq = _simplify(q)

    print(f"\nQuery  : {repr(query)}")
    print(f"Lower  : {repr(q)}")
    print(f"Simple : {repr(sq)}")

    # Step 1: exact
    if q in data:
        print(f"\n[MATCH] exact key: {repr(q)}")
        _print_results(data[q])
        return

    print("\n[step 1] No exact match.")

    # Step 2: substring
    print("[step 2] Substring scan (longest keys first):")
    for key in sorted(data, key=len, reverse=True):
        sk = _simplify(key)
        if sk in sq or sq in sk:
            print(f"  [MATCH] substring: {repr(key)}  (sk={repr(sk)})")
            _print_results(data[key])
            return
        # uncomment for very verbose mode:
        # print(f"  skip: {repr(key)}")

    print("  No substring match.")

    # Step 3: token match
    print("[step 3] Token-match scan (longest keys first, threshold=0.4):")
    q_tokens = set(_simplify(q).split())
    for key in sorted(data, key=len, reverse=True):
        k_tokens = set(_simplify(key).split())
        if not q_tokens or not k_tokens:
            continue
        inter = q_tokens & k_tokens
        union = q_tokens | k_tokens
        ratio = len(inter) / len(union)
        if ratio >= 0.4:
            print(f"  [MATCH] token: {repr(key)}")
            print(f"          q_tokens={sorted(q_tokens)}")
            print(f"          k_tokens={sorted(k_tokens)}")
            print(f"          shared={sorted(inter)}  ratio={ratio:.2f}")
            _print_results(data[key])
            return
        # uncomment to see near-misses:
        # if ratio > 0.2:
        #     print(f"  near ({ratio:.2f}): {repr(key)}")

    print("\n[NO MATCH] query falls through all three steps -> returns empty results.")
    print("\nTop-5 closest keys by token overlap:")
    scored = []
    for key in data:
        k_tokens = set(_simplify(key).split())
        if not k_tokens:
            continue
        inter = q_tokens & k_tokens
        union = q_tokens | k_tokens
        ratio = len(inter) / len(union) if union else 0
        scored.append((ratio, key, sorted(inter)))
    for ratio, key, inter in sorted(scored, reverse=True)[:5]:
        print(f"  {ratio:.2f}  {repr(key)}  shared={inter}")


def _print_results(results: list) -> None:
    print(f"\n  -> {len(results)} result(s):")
    for r in results:
        print(f"     • {r['title']}")
        print(f"       {r['url']}")


def _all_failures() -> None:
    data = json.loads(FIXTURES.read_text())
    seen: set[str] = set()

    for traj in sorted(RESULT_DIR.rglob("trajectory.jsonl")):
        events = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
        last_query: str | None = None
        for ev in events:
            msg = ev.get("message", {})
            if msg.get("role") == "assistant":
                for item in msg.get("content", []):
                    if item.get("type") == "toolCall" and "web_search" in item.get("name", ""):
                        last_query = item.get("arguments", {}).get("query", "")
            if msg.get("role") == "toolResult" and "web_search" in msg.get("toolName", ""):
                for c in msg.get("content", []):
                    if c.get("type") == "text":
                        try:
                            payload = json.loads(c["text"])
                            if payload.get("error") and not payload.get("results") and last_query:
                                if last_query not in seen:
                                    seen.add(last_query)
                                    _resolve_verbose(last_query, data)
                        except Exception:
                            pass

    print(f"\n\nTotal unique failing queries: {len(seen)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?", help="Query string to debug")
    ap.add_argument("--all-failures", action="store_true", help="Replay every failing query from all saved trajectories")
    ap.add_argument("--list-keys", action="store_true", help="Print all fixture keys and exit")
    args = ap.parse_args()

    data = json.loads(FIXTURES.read_text())

    if args.list_keys:
        for k in data:
            print(k)
        return

    if args.all_failures:
        _all_failures()
        return

    if not args.query:
        ap.print_help()
        return

    _resolve_verbose(args.query, data)


if __name__ == "__main__":
    main()
