#!/usr/bin/env bash
# Run all tasks across all models and both run types (single-file + chain),
# then evaluate each run with trajeval.
#
# Run IDs are deterministic (no timestamp) so re-running resumes from where
# it left off:
#   - eval.json exists        → fully done, skip
#   - index.json exists only  → trajrec ran (possibly partial), skip to trajeval
#   - neither                 → run both from scratch
#
# Prerequisites:
#   - OPENROUTER_API_KEY set in environment
#   - trajrec and trajeval installed (pip install -e . in both repos)
#   - Docker daemon running with image trajrec-openclaw:test built
#   - EvalC build/ directory populated:
#       python3 -c "import sys; sys.argv=['','build','--seeds-dir','seeds/','--output-dir','build/']; from dataset.cli import app; app()"
#
# Usage:
#   ./scripts/run_all.sh            # live run (resumes if interrupted)
#   ./scripts/run_all.sh --dry-run  # preview commands without executing

set -uo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVALC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASKS_DIR="$EVALC_DIR/build/tasks"
RUNS_DIR="$HOME/trajrec-runs"
DOCKER_BIN=/usr/local/bin/docker

# ---------------------------------------------------------------------------
# Models: "openrouter-model-id|slug"
# ---------------------------------------------------------------------------
MODELS=(
  "openrouter/google/gemini-2.5-flash-lite-preview-09-2025|gemini"
  "openrouter/openai/gpt-4o-mini|gpt-mini"
  "openrouter/anthropic/claude-sonnet-4-6|sonnet-4-6"
)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

if [[ ! -d "$TASKS_DIR" ]]; then
  echo "ERROR: build/tasks not found. Run the datagen build command first." >&2
  exit 1
fi

# Collect chain dirs (multi_round tasks)
CHAIN_ARGS=()
for d in "$TASKS_DIR"/*/; do
  [[ -f "$d/chain.yaml" ]] && CHAIN_ARGS+=(--tasks "$d")
done

if [[ ${#CHAIN_ARGS[@]} -eq 0 ]]; then
  echo "WARN: no chain dirs found in $TASKS_DIR" >&2
fi

echo "Tasks dir  : $TASKS_DIR"
echo "Runs dir   : $RUNS_DIR"
echo "Models     : ${#MODELS[@]}"
echo "Chain tasks: ${#CHAIN_ARGS[@]}"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_or_print() {
  if $DRY_RUN; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

# Run a single (trajrec + trajeval) pair with resume logic.
# Args: run_id, trajrec_extra_args...
run_one() {
  local run_id="$1"; shift
  local run_dir="$RUNS_DIR/$run_id"
  local index_json="$run_dir/index.json"
  local eval_json="$run_dir/eval.json"

  if [[ -f "$eval_json" ]] && ! $DRY_RUN; then
    echo "  [skip] $run_id — eval.json already exists"
    return 0
  fi

  # Check if trajrec finished cleanly: index.json exists, is valid JSON,
  # has at least one task, and every task has final_ok=True.
  local trajrec_done=false
  if [[ -f "$index_json" ]] && ! $DRY_RUN; then
    if python3 -c "
import json, sys
try:
    d = json.load(open('$index_json'))
    tasks = d.get('tasks', [])
    sys.exit(0 if tasks and all(t.get('final_ok') for t in tasks) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
      trajrec_done=true
    fi
  fi

  if $trajrec_done; then
    echo "  [resume] $run_id — trajrec complete, running trajeval only"
  else
    echo "  [run] $run_id"
    run_or_print trajrec \
      "$@" \
      --run-id "$run_id" \
      --docker-bin "$DOCKER_BIN" || echo "  WARN: trajrec exited non-zero for $run_id"
  fi

  run_or_print trajeval \
    --run-dir "$run_dir" || echo "  WARN: trajeval exited non-zero for $run_id"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for entry in "${MODELS[@]}"; do
  MODEL="${entry%%|*}"
  SLUG="${entry##*|}"

  echo "==> [$SLUG]"

  # single-file tasks: single_step + multi_step + hard_instruction
  run_one "${SLUG}_singlefile" \
    --tasks "$TASKS_DIR" \
    --model "$MODEL"

  echo ""

  # chain tasks: multi_round
  if [[ ${#CHAIN_ARGS[@]} -gt 0 ]]; then
    run_one "${SLUG}_chain" \
      "${CHAIN_ARGS[@]}" \
      --model "$MODEL"
    echo ""
  fi
done

echo "Done. Results in $RUNS_DIR/<run-id>/eval.json"
