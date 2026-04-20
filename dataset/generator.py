"""Dataset generator for the copyright laundering eval.

Reads seed files (authored via `datagen generate-prompts`) and converts
TaskTemplates into TaskSpec objects consumable by trajectory-recorder.

Prompts are concrete strings stored in seeds — no variable resolution,
no randomness. Generate once, use forever.

Usage:
    gen = CopyrightTaskGenerator(Path("seeds/high_001.json"))
    tasks = gen.generate()

    # Or across all seeds:
    seeds, all_tasks = generate_all(Path("seeds/"))
"""

from __future__ import annotations

from pathlib import Path

import json

from dataset.schema import TaskSpec, TaskStep, WorkSeed


class CopyrightTaskGenerator:
    """Loads a single seed file and produces concrete TaskSpec objects."""

    def __init__(self, seed_path: Path) -> None:
        raw = json.loads(seed_path.read_text())
        self.seed = WorkSeed.model_validate(raw)

    def generate(self) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for template in self.seed.tasks:
            steps = [TaskStep(index=s.index, prompt=s.prompt) for s in template.steps]
            tasks.append(
                TaskSpec(
                    task_id=template.task_id_prefix,
                    user_intent=template.intent,
                    task_version=template.version,
                    target_familiarity=self.seed.metadata.familiarity,
                    target_work=self.seed.metadata.work_title,
                    target_author=self.seed.metadata.author,
                    steps=steps,
                    expected_high_risk_tools=template.expected_high_risk_tools,
                    expected_low_risk_tools=template.expected_low_risk_tools,
                    timeout_s=template.timeout_s,
                    notes=template.notes,
                )
            )
        return tasks


def generate_all(seeds_dir: Path) -> tuple[list[WorkSeed], list[TaskSpec]]:
    """Load every *.json in seeds_dir and generate tasks from each.

    Returns (seeds, all_tasks). Seeds are in alphabetical file order.
    """
    seed_files = sorted(seeds_dir.glob("*.json"))
    seeds: list[WorkSeed] = []
    all_tasks: list[TaskSpec] = []
    for sf in seed_files:
        gen = CopyrightTaskGenerator(sf)
        seeds.append(gen.seed)
        all_tasks.extend(gen.generate())
    return seeds, all_tasks
