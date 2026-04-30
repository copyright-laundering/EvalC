#!/usr/bin/env bash
# Run all tasks across all models and both run types (single-file + chain),
# then evaluate each run with trajeval.
#
# Run IDs are deterministic (no timestamp) so re-running resumes from where
# it left off:
#   - eval.json exists, no errors → fully done, skip
#   - eval.json exists with errors → retry errored tasks, patch eval.json
#   - index.json exists only      → trajrec ran (possibly partial), skip to trajeval
#   - neither                     → run both from scratch
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

  # Check for errors in an existing eval.json — handled by caller via retry_errored
  if [[ -f "$eval_json" ]] && ! $DRY_RUN; then
    local has_errors
    has_errors=$(python3 -c "
import json, sys
try:
    d = json.load(open('$eval_json'))
    sys.exit(0 if any(t.get('error') for t in d.get('tasks', [])) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null && echo "yes" || echo "no")

    if [[ "$has_errors" == "no" ]]; then
      echo "  [skip] $run_id — eval.json complete, no errors"
      return 0
    else
      echo "  [has-errors] $run_id — eval.json has errors (will retry below)"
      return 0
    fi
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

# Retry tasks that errored in eval.json (e.g. due to API credit exhaustion).
# After retrying, patches the main eval.json with the successful results.
# Args: run_id, model
retry_errored() {
  local run_id="$1"
  local model="$2"
  local run_dir="$RUNS_DIR/$run_id"
  local eval_json="$run_dir/eval.json"

  if $DRY_RUN; then
    echo "[dry-run] retry_errored $run_id"
    return 0
  fi

  [[ ! -f "$eval_json" ]] && return 0

  # Collect errored task_ids
  local errored_ids
  errored_ids=$(python3 -c "
import json, sys
try:
    d = json.load(open('$eval_json'))
    ids = [t['task_id'] for t in d.get('tasks', []) if t.get('error')]
    print('\n'.join(ids))
except Exception:
    pass
" 2>/dev/null)

  [[ -z "$errored_ids" ]] && return 0

  local n_errored
  n_errored=$(echo "$errored_ids" | wc -l | tr -d ' ')
  echo "  [retry] $run_id — $n_errored errored tasks"

  # Build --tasks args for the errored tasks
  local retry_task_args=()
  while IFS= read -r task_id; do
    [[ -z "$task_id" ]] && continue
    if [[ -d "$TASKS_DIR/$task_id" ]]; then
      retry_task_args+=(--tasks "$TASKS_DIR/$task_id")
    elif [[ -f "$TASKS_DIR/${task_id}.yaml" ]]; then
      retry_task_args+=(--tasks "$TASKS_DIR/${task_id}.yaml")
    else
      echo "  WARN: cannot find task path for '$task_id' — skipping" >&2
    fi
  done <<< "$errored_ids"

  if [[ ${#retry_task_args[@]} -eq 0 ]]; then
    echo "  WARN: no valid task paths found for retry in $run_id" >&2
    return 0
  fi

  local retry_id="${run_id}_retry"
  local retry_dir="$RUNS_DIR/$retry_id"

  echo "  [retry] running trajrec for $retry_id"
  trajrec \
    "${retry_task_args[@]}" \
    --model "$model" \
    --run-id "$retry_id" \
    --docker-bin "$DOCKER_BIN" || echo "  WARN: trajrec exited non-zero for $retry_id"

  echo "  [retry] running trajeval for $retry_id"
  trajeval \
    --run-dir "$retry_dir" || echo "  WARN: trajeval exited non-zero for $retry_id"

  # Patch main eval.json with retry results
  if [[ ! -f "$retry_dir/eval.json" ]]; then
    echo "  WARN: no retry eval.json found at $retry_dir/eval.json" >&2
    return 0
  fi

  python3 - "$eval_json" "$retry_dir/eval.json" <<'PYEOF'
import json, sys

main_path = sys.argv[1]
retry_path = sys.argv[2]

main = json.load(open(main_path))
retry = json.load(open(retry_path))

retry_by_id = {t["task_id"]: t for t in retry.get("tasks", [])}

patched = []
replaced = 0
still_errored = 0
for t in main.get("tasks", []):
    if t.get("error") and t["task_id"] in retry_by_id:
        patched.append(retry_by_id[t["task_id"]])
        replaced += 1
    else:
        patched.append(t)
        if t.get("error"):
            still_errored += 1

main["tasks"] = patched

# Recompute aggregates
evaluable = [t for t in patched if not t.get("timed_out") and not t.get("error")]
refused = sum(1 for t in evaluable if t.get("refused"))
copyright = sum(1 for t in evaluable if t.get("copyright_aware"))
n = len(evaluable)
main["refusal_rate"] = refused / n if n else 0.0
main["copyright_awareness_rate"] = copyright / n if n else 0.0
main["task_count"] = len(patched)
main["timed_out_count"] = sum(1 for t in patched if t.get("timed_out"))

with open(main_path, "w") as f:
    json.dump(main, f, indent=2)

print(f"  Patched {replaced} tasks into {main_path}")
if still_errored:
    print(f"  WARN: {still_errored} tasks still errored after retry")
PYEOF
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

  retry_errored "${SLUG}_singlefile" "$MODEL"

  echo ""

  # chain tasks: multi_round
  if [[ ${#CHAIN_ARGS[@]} -gt 0 ]]; then
    run_one "${SLUG}_chain" \
      "${CHAIN_ARGS[@]}" \
      --model "$MODEL"

    retry_errored "${SLUG}_chain" "$MODEL"
    echo ""
  fi
done

echo "Done. Results in $RUNS_DIR/<run-id>/eval.json"
