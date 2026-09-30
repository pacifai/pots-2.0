#!/usr/bin/env python3
"""SessionStart hook: offers the open design tasks and routes the user's choice.

Builds the task menu from the open-only task files in docs/verification/ (layout in
docs/verification/STATUS.md) and injects it as context. On startup, resume, or /clear,
Claude asks which task to continue; after a compaction it resumes the task in progress.
"""

import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "verification"
TASK_FILES = [
    ("SETUP_TASKS.md", "Setup (stage 3, in progress)"),
    ("EVALUATION_TASKS.md", "Evaluation (parked until setup closes)"),
    ("FULL_SCALE_TASKS.md", "Full scale (parked until the test-scale run is done)"),
]
# A bold item title can wrap onto following indented lines, so match across newlines.
ITEM = re.compile(r"^- \*\*([A-Z]\d+[a-z]?) — (.+?)\*\*", re.MULTILINE | re.DOTALL)
MAX_TITLE = 90


def open_items(path):
    items = []
    for m in ITEM.finditer(path.read_text(encoding="utf-8")):
        title = " ".join(m.group(2).split()).rstrip(".")
        if len(title) > MAX_TITLE:
            title = title[: MAX_TITLE - 1].rstrip() + "…"
        items.append(f"{m.group(1)}: {title}")
    return items


def next_setup_task():
    status = DOCS / "STATUS.md"
    if not status.exists():
        return None
    m = re.search(r"the next item is \*\*(\w+)\*\*", status.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def menu():
    lines = []
    for name, label in TASK_FILES:
        path = DOCS / name
        if path.exists():
            items = open_items(path)
            lines.append(f"{label} — `docs/verification/{name}`:")
            lines += [f"  - {i}" for i in items] or ["  - (no open items)"]
    return "\n".join(lines)


def main():
    if not DOCS.is_dir():
        return
    try:
        source = json.load(sys.stdin).get("source", "startup")
    except (json.JSONDecodeError, ValueError):
        source = "startup"

    nxt = next_setup_task()
    rec = f" `STATUS.md` names **{nxt}** as the next setup task." if nxt else ""

    if source == "compact":
        instruction = (
            "The context was just compacted. Continue the task that was in progress. "
            "If the summary doesn't make clear which task that was, ask the user, "
            "offering the open tasks below."
        )
    else:
        instruction = (
            "Before doing anything else in this session, ask the user which task to "
            f"continue, offering the open tasks below.{rec} Then route by the answer:\n"
            "1. If the user names an open task, a task file, or a subject that matches an "
            "open item, read `docs/verification/STATUS.md` and that task file, then open "
            "only the decision or background sections the item cites, and start the task.\n"
            "2. If the user names none of the existing files, task IDs, or subjects, treat "
            "the request as a new task and start it directly. If it belongs to the "
            "verification design, add it to the matching task file first, under that "
            "file's lock rule.\n"
            "If the user's first message already makes the choice, skip the question and "
            "route it the same way."
        )

    context = (
        "Verification-protocol design session (pots-2.0).\n"
        f"{instruction}\n\n"
        "Open tasks, read from the task files at session start:\n"
        f"{menu()}\n\n"
        "Parked lists (evaluation, full scale) are taken up only when the user explicitly "
        "chooses them. Follow the stage gates and rules in `STATUS.md` and `CLAUDE.md`."
    )
    json.dump(
        {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}},
        sys.stdout,
    )


if __name__ == "__main__":
    main()
