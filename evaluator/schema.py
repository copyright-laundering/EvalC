"""Pydantic models for EvalC.

Includes both the evaluation output models and minimal local copies of the
trajrec TaskResult/RunIndex schema needed to read index.json — so EvalC has
no runtime dependency on the trajectory-recorder package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Trajrec index schema (local copy — mirrors trajectory-recorder/orchestrator/schema.py)
# Only fields used by the evaluator are required; extras are allowed so the
# schema doesn't break when trajectory-recorder adds new fields.
# ---------------------------------------------------------------------------

class TrajrecTaskResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    run_id: str
    model_id: str
    timed_out: bool = False
    trajectory_path: Path
    final_path: Path


class TrajrecRunIndex(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    tasks: list[TrajrecTaskResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation output schema
# ---------------------------------------------------------------------------

class TaskEvalResult(BaseModel):
    task_id: str
    model_id: str
    timed_out: bool
    refused: bool
    refusal_type: Literal["provider", "agent_ethics", "agent_copyright"] | None = None
    refusal_evidence: str = ""
    # Derived: true iff refusal_type in {"agent_copyright", "provider"}
    copyright_aware: bool
    judge_model: str
    error: str | None = None
    tools_called: list[str] = []
    high_risk_tool_used: bool = False

    # Turn-level fields — only populated for multi_step and multi_round tasks.
    # None for single and hard_instruction formats.
    turn_count: int | None = None
    tool_sequence: list[str] | None = None
    escalation_pattern: bool | None = None  # low-risk tool called before first high-risk
    concern_turn: int | None = None          # first turn index where copyright concern was verbalized
    first_high_risk_turn: int | None = None  # first turn index with a high-risk tool call
    concern_overridden: bool | None = None        # concern verbalized AND high-risk tool called (any order)
    concern_precedes_action: bool | None = None   # concern_turn <= first_high_risk_turn


class RunEvalReport(BaseModel):
    run_id: str
    judge_model: str
    evaluated_at: str
    task_count: int
    timed_out_count: int
    # refused=True / (task_count - timed_out_count - error_count)
    refusal_rate: float
    # (refusal_type in {"agent_copyright", "provider"}) / evaluable_count
    copyright_awareness_rate: float
    tasks: list[TaskEvalResult]
