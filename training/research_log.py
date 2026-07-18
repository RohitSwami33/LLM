"""Research log management.

Manages EXPERIMENT_HISTORY.md:
  - Adds completed experiments
  - Auto-generates sections for future experiments
  - Links to experiment directories
  - Maintains running observations section
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger(__name__)


class ResearchLog:
    """Manages the EXPERIMENT_HISTORY.md research log."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.history_path = project_root / "experiments" / "EXPERIMENT_HISTORY.md"

    def ensure_history_file(self):
        """Create EXPERIMENT_HISTORY.md if it doesn't exist."""
        if self.history_path.exists():
            return

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        content = """# Experiment History

This file tracks all architecture experiments for the tiny model baseline.

Each experiment entry contains:
- Date, name, and hypothesis
- Key results (validation perplexity, HellaSwag accuracy, etc.)
- Observation and next steps
- Link to experiment directory

---

## Running Observations

- Baseline (12L, 4H, 64D): val_ppl=20,912, HellaSwag=23.4%, WikiText-2=47,277
- All heads at layer 0 only: 67.1% speedup but +24% perplexity
- 6 dense heads + 2 early-exit: 29.8% speedup but +11% perplexity

---

## Experiment Log

"""
        with open(self.history_path, "w") as f:
            f.write(content)
        LOG.info("Created EXPERIMENT_HISTORY.md")

    def add_experiment(self, entry: dict):
        """Add an experiment entry to the history."""
        self.ensure_history_file()

        # Read existing content
        with open(self.history_path) as f:
            content = f.read()

        # Format the entry
        entry_text = self._format_entry(entry)

        # Insert before the end
        content = content.rstrip() + "\n\n" + entry_text + "\n"

        with open(self.history_path, "w") as f:
            f.write(content)

        LOG.info("Experiment added to history: %s", entry.get("name", "unknown"))

    def add_observation(self, observation: str):
        """Add a running observation."""
        self.ensure_history_file()

        with open(self.history_path) as f:
            content = f.read()

        # Find the Running Observations section
        marker = "## Running Observations"
        if marker in content:
            # Insert before the next section
            parts = content.split(marker, 1)
            after_marker = parts[1]
            next_section = after_marker.find("\n## ")
            if next_section != -1:
                insert_pos = len(parts[0]) + len(marker) + next_section
                content = content[:insert_pos] + f"\n- {observation}" + content[insert_pos:]
            else:
                content = content.rstrip() + f"\n- {observation}\n"
        else:
            # Add section
            content += f"\n## Running Observations\n\n- {observation}\n"

        with open(self.history_path, "w") as f:
            f.write(content)

        LOG.info("Observation added: %s", observation[:50])

    def add_todo_section(self, section_name: str, items: list[str]):
        """Add or update a TODO section."""
        self.ensure_history_file()

        with open(self.history_path) as f:
            content = f.read()

        # Check if section exists
        marker = f"## {section_name}"
        if marker in content:
            # Update existing section
            parts = content.split(marker, 1)
            after = parts[1]
            next_section = after.find("\n## ")
            if next_section != -1:
                end_pos = len(parts[0]) + len(marker) + next_section
                items_text = "\n".join(f"- [ ] {item}" for item in items)
                content = content[:end_pos] + f"\n{items_text}\n" + content[end_pos:]
        else:
            # Append new section
            items_text = "\n".join(f"- [ ] {item}" for item in items)
            content += f"\n## {section_name}\n\n{items_text}\n"

        with open(self.history_path, "w") as f:
            f.write(content)

        LOG.info("TODO section added/updated: %s", section_name)

    def _format_entry(self, entry: dict) -> str:
        """Format an experiment entry as markdown."""
        lines = []

        # Header
        name = entry.get("name", "Unnamed Experiment")
        date = entry.get("date", time.strftime("%Y-%m-%d"))
        lines.append(f"### {name}")
        lines.append(f"**Date:** {date}")
        lines.append("")

        # Hypothesis
        if entry.get("hypothesis"):
            lines.append(f"**Hypothesis:** {entry['hypothesis']}")
            lines.append("")

        # Architecture
        if entry.get("architecture"):
            lines.append(f"**Architecture:** {entry['architecture']}")
            lines.append("")

        # Key Results
        results = entry.get("results", {})
        if results:
            lines.append("**Key Results:**")
            for key, value in results.items():
                if isinstance(value, float):
                    lines.append(f"- {key}: {value:.4f}")
                else:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        # Observation
        if entry.get("observation"):
            lines.append(f"**Observation:** {entry['observation']}")
            lines.append("")

        # Next Steps
        if entry.get("next_steps"):
            lines.append("**Next Steps:**")
            for step in entry["next_steps"]:
                lines.append(f"- {step}")
            lines.append("")

        # Directory
        if entry.get("directory"):
            lines.append(f"**Directory:** `{entry['directory']}`")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)

    def get_experiments(self) -> list[dict]:
        """Parse existing experiments from history."""
        if not self.history_path.exists():
            return []

        experiments = []
        with open(self.history_path) as f:
            content = f.read()

        # Simple parsing: find experiment entries
        sections = content.split("### ")
        for section in sections[1:]:  # Skip header
            lines = section.strip().split("\n")
            if lines:
                name = lines[0].strip()
                experiment = {"name": name}
                for line in lines[1:]:
                    if line.startswith("**Date:**"):
                        experiment["date"] = line.split("**Date:**")[1].strip()
                    elif line.startswith("**Hypothesis:**"):
                        experiment["hypothesis"] = line.split("**Hypothesis:**")[1].strip()
                    elif line.startswith("**Directory:**"):
                        experiment["directory"] = line.split("**Directory:**")[1].strip().strip("`")
                experiments.append(experiment)

        return experiments

    def get_todo_items(self, section_name: Optional[str] = None) -> list[str]:
        """Get TODO items, optionally filtered by section."""
        if not self.history_path.exists():
            return []

        items = []
        with open(self.history_path) as f:
            content = f.read()

        in_section = section_name is None
        for line in content.split("\n"):
            line = line.strip()
            if section_name and line.startswith("## ") and section_name in line:
                in_section = True
                continue
            if in_section and line.startswith("- [ ] "):
                items.append(line[6:])  # Remove "- [ ] " prefix
            elif section_name and line.startswith("## ") and section_name not in line:
                if in_section:
                    break

        return items

    def complete_todo_item(self, item_text: str):
        """Mark a TODO item as complete."""
        if not self.history_path.exists():
            return

        with open(self.history_path) as f:
            content = f.read()

        # Replace - [ ] with - [x] for matching item
        old = f"- [ ] {item_text}"
        new = f"- [x] {item_text}"
        if old in content:
            content = content.replace(old, new)
            with open(self.history_path, "w") as f:
                f.write(content)
            LOG.info("TODO completed: %s", item_text)
        else:
            LOG.warning("TODO item not found: %s", item_text)
