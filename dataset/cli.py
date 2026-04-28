"""Typer CLI for dataset generation.

Commands:
    datagen build            -- generate task YAMLs + dataset.json from seeds
    datagen validate         -- check grid coverage
    datagen generate-prompts -- use LLM to generate TaskTemplates into a seed file

Examples:
    datagen build --seeds-dir seeds/ --output-dir build/
    datagen validate --seeds-dir seeds/
    datagen generate-prompts --seed-file seeds/high_001.json \\
        --persona "undergraduate literature student" --intent all --version all
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml

from dataset.generator import CopyrightTaskGenerator, generate_all
from dataset.grid import coverage_report, validate_coverage
from dataset.schema import DatasetManifest, TaskSpec, TaskVersion, UserIntent, WorkSeed

app = typer.Typer(help="Dataset generation pipeline for copyright laundering eval.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_single_step_yaml(task: TaskSpec, tasks_dir: Path) -> None:
    """Write a flat YAML for a single-step task."""
    step = task.steps[0]
    data = {
        "id": task.task_id,
        "prompt": step.prompt,
        "timeout_s": task.timeout_s,
    }
    path = tasks_dir / f"{task.task_id}.yaml"
    path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))


def _write_multi_round_dir(task: TaskSpec, tasks_dir: Path) -> None:
    """Write a directory with per-step YAMLs and a chain.yaml for multi-round tasks."""
    task_dir = tasks_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    step_files = []
    for step in task.steps:
        filename = f"step_{step.index:02d}.yaml"
        step_data = {
            "id": f"{task.task_id}_step{step.index:02d}",
            "prompt": step.prompt,
            "timeout_s": task.timeout_s,
        }
        (task_dir / filename).write_text(
            yaml.dump(step_data, allow_unicode=True, sort_keys=False)
        )
        step_files.append({"index": step.index, "file": filename})

    chain_data = {
        "task_id": task.task_id,
        "steps": step_files,
    }
    (task_dir / "chain.yaml").write_text(
        yaml.dump(chain_data, allow_unicode=True, sort_keys=False)
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("build")
def build_cmd(
    seeds_dir: Path = typer.Option(Path("seeds"), "--seeds-dir", help="Directory of seed JSON files."),
    output_dir: Path = typer.Option(Path("build"), "--output-dir", help="Output directory for build artifacts."),
) -> None:
    """Convert seed files into task YAMLs and dataset.json for trajectory-recorder."""
    if not seeds_dir.exists():
        typer.echo(f"Error: seeds directory not found: {seeds_dir}", err=True)
        raise typer.Exit(1)

    seed_files = sorted(seeds_dir.glob("*.json"))
    if not seed_files:
        typer.echo(f"Error: no *.json files found in {seeds_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(seed_files)} seed file(s) in {seeds_dir}")

    _, all_tasks = generate_all(seeds_dir)

    tasks_dir = output_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    single_count = 0
    multi_step_count = 0
    multi_round_count = 0
    hard_instruction_count = 0

    for task in all_tasks:
        if task.task_version == TaskVersion.single_step:
            _write_single_step_yaml(task, tasks_dir)
            single_count += 1
        elif task.task_version == TaskVersion.multi_step:
            _write_single_step_yaml(task, tasks_dir)
            multi_step_count += 1
        elif task.task_version == TaskVersion.hard_instruction:
            _write_single_step_yaml(task, tasks_dir)
            hard_instruction_count += 1
        else:
            _write_multi_round_dir(task, tasks_dir)
            multi_round_count += 1

    manifest = DatasetManifest(
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed_count=len(seed_files),
        task_count=len(all_tasks),
        tasks=all_tasks,
    )
    manifest_path = output_dir / "dataset.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    typer.echo("")
    typer.echo("=" * 52)
    typer.echo(f"  Seeds processed : {len(seed_files)}")
    typer.echo(f"  Tasks built     : {len(all_tasks)}")
    typer.echo(f"    Single-step      : {single_count}")
    typer.echo(f"    Multi-step       : {multi_step_count}")
    typer.echo(f"    Multi-round      : {multi_round_count}")
    typer.echo(f"    Hard-instruction : {hard_instruction_count}")
    typer.echo("=" * 52)
    typer.echo(f"  Output: {output_dir.resolve()}")


@app.command("generate-prompts")
def generate_prompts_cmd(
    seed_file: Path = typer.Option(..., "--seed-file", help="Path to a seed JSON file."),
    persona: str = typer.Option(..., "--persona", help="Persona for the user in all generated prompts."),
    intent: str = typer.Option("all", "--intent", help="Intent to generate: all, benign, ambiguous, infringing."),
    version: str = typer.Option("all", "--version", help="Version to generate: all, single_step, multi_step, multi_round."),
    model: str = typer.Option("claude-opus-4-6", "--model", help="Claude model ID."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print generated templates without writing to seed file."),
) -> None:
    """Use Claude to generate TaskTemplates and append them to a seed file."""
    from dataset.prompt_gen import generate_task_template, make_anthropic_client
    from dataset.schema import TaskVersion as TV

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        typer.echo("Error: ANTHROPIC_API_KEY environment variable is not set.", err=True)
        raise typer.Exit(1)

    if not seed_file.exists():
        typer.echo(f"Error: seed file not found: {seed_file}", err=True)
        raise typer.Exit(1)

    raw = json.loads(seed_file.read_text())
    seed = WorkSeed.model_validate(raw)

    # Resolve intent targets
    intent_map = {
        "benign": [UserIntent.benign],
        "ambiguous": [UserIntent.ambiguous],
        "infringing": [UserIntent.infringing],
        "all": list(UserIntent),
    }
    if intent not in intent_map:
        typer.echo(f"Error: --intent must be one of: all, benign, ambiguous, infringing", err=True)
        raise typer.Exit(1)
    intents = intent_map[intent]

    # Resolve version targets
    version_map = {
        "single_step": [TV.single_step],
        "multi_step": [TV.multi_step],
        "multi_round": [TV.multi_round],
        "all": list(TV),
    }
    if version not in version_map:
        typer.echo(f"Error: --version must be one of: all, single_step, multi_step, multi_round", err=True)
        raise typer.Exit(1)
    versions = version_map[version]

    client = make_anthropic_client(api_key)
    targets = [(i, v) for i in intents for v in versions]

    typer.echo(
        f"Generating {len(targets)} template(s) for '{seed.metadata.work_title}' "
        f"with persona: {persona}"
    )
    typer.echo("")

    generated: list = []
    for i, v in targets:
        label = f"{i.value} / {v.value}"
        typer.echo(f"  [{label}] ", nl=False)
        try:
            template = generate_task_template(
                seed=seed,
                persona=persona,
                intent=i,
                version=v,
                client=client,
                model=model,
            )
            typer.echo(f"ok — {template.task_id_prefix} ({len(template.steps)} step(s))")
            generated.append(template)
        except Exception as exc:
            typer.echo(f"FAILED: {exc}")

    typer.echo("")

    if dry_run:
        typer.echo("Dry run — output below, not written to file:\n")
        for t in generated:
            typer.echo(t.model_dump_json(indent=2))
        return

    if not generated:
        typer.echo("No templates generated.")
        return

    # Update persona on seed if not already set
    if not seed.persona:
        seed.persona = persona

    # Append new templates
    seed.tasks.extend(generated)

    # Write back — preserve structure, serialize via model_dump
    updated = seed.model_dump(mode="json")
    seed_file.write_text(json.dumps(updated, indent=2, ensure_ascii=False))

    typer.echo(f"Wrote {len(generated)} template(s) to {seed_file}")


@app.command("validate")
def validate_cmd(
    seeds_dir: Path = typer.Option(Path("seeds"), "--seeds-dir", help="Directory of seed JSON files."),
    min_per_cell: int = typer.Option(1, "--min-per-cell", help="Minimum templates required per grid cell."),
) -> None:
    """Validate seed coverage across the experimental grid."""
    if not seeds_dir.exists():
        typer.echo(f"Error: seeds directory not found: {seeds_dir}", err=True)
        raise typer.Exit(1)

    seed_files = sorted(seeds_dir.glob("*.json"))
    if not seed_files:
        typer.echo(f"Error: no *.json files found in {seeds_dir}", err=True)
        raise typer.Exit(1)

    seeds = []
    for sf in seed_files:
        gen = CopyrightTaskGenerator(sf)
        seeds.append(gen.seed)

    typer.echo(coverage_report(seeds))
    typer.echo("")

    warnings = validate_coverage(seeds, min_per_cell)
    if warnings:
        typer.echo(f"  {len(warnings)} warning(s):")
        for w in warnings:
            typer.echo(f"    [!] {w}")
        raise typer.Exit(1)
    else:
        typer.echo("  All checks passed.")
