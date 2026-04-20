"""Grid coverage validation for the copyright laundering dataset.

Checks that seed files adequately cover the experimental grid before
running trajectory-recorder experiments.

Usage:
    from dataset.grid import validate_coverage
    warnings = validate_coverage(seeds, min_per_cell=1)
    for w in warnings:
        print(w)
"""

from __future__ import annotations

from collections import defaultdict

from dataset.schema import (
    TOOL_CATALOG,
    FamiliarityTier,
    TaskVersion,
    UserIntent,
    WorkSeed,
)


def validate_coverage(
    seeds: list[WorkSeed],
    min_per_cell: int = 1,
) -> list[str]:
    """Check that seeds cover the experimental grid adequately.

    Returns a list of warning strings. Empty list means all checks pass.

    Checks:
    - No seed file has an empty tasks list
    - Every (familiarity x intent x version) cell has at least min_per_cell templates
    - All tools in TOOL_CATALOG appear in at least one expected_*_tools list
    """
    warnings: list[str] = []

    # Count templates per (familiarity, intent, version) cell
    cell_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    tools_seen: set[str] = set()

    for seed in seeds:
        if not seed.tasks:
            warnings.append(
                f"Seed for '{seed.metadata.work_title}' has no tasks."
            )
            continue

        for template in seed.tasks:
            cell = (
                seed.metadata.familiarity.value,
                template.intent.value,
                template.version.value,
            )
            cell_counts[cell] += 1
            tools_seen.update(template.expected_high_risk_tools)
            tools_seen.update(template.expected_low_risk_tools)

    # Check every cell in the full grid
    for familiarity in FamiliarityTier:
        for intent in UserIntent:
            for version in TaskVersion:
                cell = (familiarity.value, intent.value, version.value)
                count = cell_counts.get(cell, 0)
                if count < min_per_cell:
                    warnings.append(
                        f"Cell ({familiarity.value}, {intent.value}, {version.value}): "
                        f"{count} templates (min {min_per_cell} required)."
                    )

    # Check all tools appear at least once
    for tool_name in TOOL_CATALOG:
        if tool_name not in tools_seen:
            warnings.append(
                f"Tool '{tool_name}' does not appear in any expected_*_tools list."
            )

    return warnings


def coverage_report(seeds: list[WorkSeed]) -> str:
    """Return a human-readable coverage table for the experimental grid."""
    cell_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for seed in seeds:
        for template in seed.tasks:
            cell = (
                seed.metadata.familiarity.value,
                template.intent.value,
                template.version.value,
            )
            cell_counts[cell] += 1

    lines = ["Coverage table (templates per cell):", ""]
    header = f"{'familiarity':<12} {'intent':<12} {'version':<14} {'count':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    for familiarity in FamiliarityTier:
        for intent in UserIntent:
            for version in TaskVersion:
                cell = (familiarity.value, intent.value, version.value)
                count = cell_counts.get(cell, 0)
                lines.append(
                    f"{familiarity.value:<12} {intent.value:<12} {version.value:<14} {count:>5}"
                )

    lines.append("")
    lines.append(f"Total cells: {len(FamiliarityTier) * len(UserIntent) * len(TaskVersion)}")
    lines.append(f"Total templates: {sum(cell_counts.values())}")
    return "\n".join(lines)
