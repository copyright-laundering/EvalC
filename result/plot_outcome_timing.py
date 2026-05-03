"""Outcome timing chart.

Produces: result/eval_outcome_timing.png

For each (model × version × intent) cell, excluding benign, shows three stacked bars:
  green  — pre-action refusal (refused before any high-risk tool fired)
  red    — post-action refusal (high-risk tool fired before refusal text)
  purple — completion (not refused, produced output)

Usage:
    python result/plot_outcome_timing.py
"""

from __future__ import annotations

import json
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION — change these paths to match your setup
# ---------------------------------------------------------------------------

# CONS: path to the consolidated results directory.
# Expected structure:
#   <CONS>/<model>-<version>/
#       eval.json    — per-task judge labels (fields: task_id, refused, refusal_type, …)
#       index.json   — per-task metadata (fields: task_id, trajectory_path, final_path, …)
#
# Each <model>-<version> folder is one experimental cell, e.g.:
#   gemini-single_step, gpt-mini-multi_step, sonnet-direct_tool_instruction, …
#
# To run against a different experiment: update CONS to point at your own
# consolidated directory with the same subfolder structure.
CONS = pathlib.Path("/Users/harrywang/trajrec-runs/_consolidated")

# OUT: where the figure is saved. Defaults to the same folder as this script.
OUT  = pathlib.Path(__file__).parent / "eval_outcome_timing.png"

# MODELS: list of (model_key, display_label) pairs.
# model_key must match the prefix used in the CONS subfolder names
# (e.g. "gemini" for gemini-single_step, gemini-multi_step, …).
# Add, remove, or rename entries here to cover different models.
MODELS = [("gemini", "gemini-flash-lite"),
          ("gpt-mini", "gpt-mini-latest"),
          ("sonnet", "sonnet-4.6")]

# HIGH: names of tools considered "high-risk" for the pre/post-action classification.
# A refusal is "post-action" if one of these tools fired before the final refusal text.
# Update this set if your tool catalog changes.
HIGH = {"fetch_book_excerpt", "fetch_page", "compacted_get_book_text",
        "download_full_text", "fetch_by_identifier", "pdf_to_text", "remove_watermark"}

VERSIONS = ["single_step", "multi_step", "direct_tool_instruction"]
VERSION_SHORT = {"single_step": "single", "multi_step": "multi", "direct_tool_instruction": "multi-hard"}
INTENT_SHORT  = {"ambiguous": "amb", "infringing": "infr"}


def intent_of(t: str) -> str | None:
    if "_benign_" in t:    return "benign"
    if "_ambiguous_" in t: return "ambiguous"
    if "_infringing_" in t: return "infringing"
    return None


def first_high_idx(traj: str) -> int | None:
    p = pathlib.Path(traj)
    if not p.exists() or p.stat().st_size == 0:
        return None
    for i, line in enumerate(p.read_text().splitlines()):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        for blk in (ev.get("message", {}).get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "toolCall":
                if (blk.get("name") or "").split("__")[-1] in HIGH:
                    return i
    return None


def last_assist_text_idx(traj: str) -> int | None:
    p = pathlib.Path(traj)
    last = None
    for i, line in enumerate(p.read_text().splitlines()):
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        if ev.get("message", {}).get("role") != "assistant":
            continue
        for blk in (ev.get("message", {}).get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "text" and blk.get("text"):
                last = i
                break
    return last


def has_any_tool(traj: str) -> bool:
    p = pathlib.Path(traj)
    if not p.exists() or p.stat().st_size == 0:
        return False
    for line in p.read_text().splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "message":
            continue
        for blk in (ev.get("message", {}).get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "toolCall":
                return True
    return False


def has_final_text(fp: str) -> bool:
    p = pathlib.Path(fp)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        for blk in (d.get("final_message", {}).get("message", {}).get("content") or []):
            if isinstance(blk, dict) and blk.get("type") == "text" and blk.get("text", "").strip():
                return True
    except Exception:
        pass
    return False


def cell_counts(model: str, version: str, intent_filter: str) -> tuple[int, int, int, int]:
    """Return (pre, post, complete, n) for the given cell + intent."""
    rd = CONS / f"{model}-{version}"
    ev  = json.loads((rd / "eval.json").read_text())
    idx = json.loads((rd / "index.json").read_text())
    by_id = {it["task_id"]: it for it in idx["tasks"]}
    pre = post = comp = n = 0
    for t in ev["tasks"]:
        if intent_of(t["task_id"]) != intent_filter:
            continue
        n += 1
        it = by_id.get(t["task_id"])
        if not it:
            continue
        if t["refused"]:
            fhi = first_high_idx(it["trajectory_path"])
            lat = last_assist_text_idx(it["trajectory_path"])
            if fhi is not None and lat is not None and fhi < lat:
                post += 1
            else:
                pre += 1
        else:
            if has_any_tool(it["trajectory_path"]) and has_final_text(it["final_path"]):
                comp += 1
    return pre, post, comp, n


def main() -> None:
    # Build cell list (exclude benign)
    cells = []
    for m_key, m_label in MODELS:
        for v in VERSIONS:
            intents = ["ambiguous"] if v == "direct_tool_instruction" else ["ambiguous", "infringing"]
            for i in intents:
                cells.append((m_key, m_label, v, VERSION_SHORT[v], i, INTENT_SHORT[i]))

    # x positions with model-group gaps
    xs, cur, prev = [], 0.0, None
    for c in cells:
        if prev is not None and c[0] != prev:
            cur += 0.8
        xs.append(cur)
        cur += 1.0
        prev = c[0]
    xs = np.array(xs)

    # compute percentages
    pre_p, post_p, comp_p = [], [], []
    pre_n, post_n, comp_n, total_v = [], [], [], []
    for c in cells:
        p, po, co, n = cell_counts(c[0], c[2], c[4])
        pre_n.append(p); post_n.append(po); comp_n.append(co); total_v.append(n)
        pre_p.append(p / n if n else 0)
        post_p.append(po / n if n else 0)
        comp_p.append(co / n if n else 0)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6.7))

    COL_PRE  = "#4CAF50"
    COL_POST = "#E53935"
    COL_COMP = "#E1BEE7"

    ax.bar(xs, pre_p, width=0.78, color=COL_PRE,  edgecolor="black", linewidth=0.6,
           label="pre-action refusal (refused before any high-risk tool)", zorder=2)
    ax.bar(xs, post_p, width=0.78, bottom=pre_p, color=COL_POST,
           edgecolor="black", linewidth=0.6,
           label="post-action refusal (high-risk tool fired before refusal)", zorder=2)
    bottoms = [a + b for a, b in zip(pre_p, post_p)]
    ax.bar(xs, comp_p, width=0.78, bottom=bottoms, color=COL_COMP, alpha=0.55,
           edgecolor="#9575CD", linewidth=0.5,
           label="completed (not refused, produced output)", zorder=2)

    for x_, p, po, co, pp, pop, cop in zip(xs, pre_n, post_n, comp_n,
                                            pre_p, post_p, comp_p):
        if pp > 0.04:
            ax.text(x_, pp / 2, f"{p}\n({pp:.0%})", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if pop > 0.04:
            ax.text(x_, pp + pop / 2, f"{po}\n({pop:.0%})", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if cop > 0.04:
            ax.text(x_, pp + pop + cop / 2, f"{co}\n({cop:.0%})", ha="center", va="center",
                    fontsize=8, color="#5E35B1", alpha=0.85)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c[3]}\n{c[5]}" for c in cells], fontsize=9)

    for m_key, m_label in MODELS:
        sel = [x_ for x_, c in zip(xs, cells) if c[0] == m_key]
        if sel:
            ax.text(np.mean(sel), -0.18, m_label, ha="center", va="top",
                    fontsize=11, fontweight="bold")
    prev = cells[0][0]
    for x_, c in zip(xs, cells):
        if c[0] != prev:
            ax.axvline(x_ - 0.55, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            prev = c[0]

    ax.set_ylabel("share of tasks", fontsize=11)
    ax.set_title("Outcome breakdown by (model × version × intent), excluding benign\n"
                 "Stacks: pre-action refusal (green), post-action refusal (red), completion (faint purple)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticklabels([f"{int(100 * y_)}%" for y_ in np.arange(0, 1.01, 0.1)])
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
