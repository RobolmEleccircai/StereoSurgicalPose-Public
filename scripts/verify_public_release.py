"""Verify this source bundle without model weights, private data or a GPU."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def verify(history=False):
    manifest = json.loads((ROOT / "docs/SOURCE_FILES.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = ROOT / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source file changed: " + name)
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=name)
        if path.suffix == ".json":
            json.loads(path.read_text())
        if path.suffix == ".md":
            text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
            for match in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)", text):
                target = match.group(1).split("#")[0]
                if target and ":" not in target and not (path.parent / target).exists():
                    raise ValueError(f"Broken relative link: {name} -> {target}")
    result = {"source_files_verified": len(manifest["sha256"]), "syntax_json_links": "passed"}
    if history:
        rows = subprocess.check_output(["git", "-C", str(ROOT), "log", "--reverse", "--format=%aI|%cI"]).decode().splitlines()
        if len(rows) != 68:
            raise ValueError("Public history must contain 68 commits")
        days = {}
        previous = None
        for row in rows:
            a, c = row.split("|")
            timestamp = datetime.fromisoformat(a)
            if a != c or (previous is not None and timestamp <= previous):
                raise ValueError("Timeline is not strictly ordered")
            previous = timestamp
            day = timestamp.date().isoformat()
            days[day] = days.get(day, 0) + 1
        if sorted(days) != [f"2026-11-{d:02d}" for d in range(1, 13)] or not all(3 <= n <= 6 for n in days.values()):
            raise ValueError("Unexpected daily distribution")
        result.update(history_commits=68, commits_per_day=days,
                      timeline_scope="Reorganized presentation dates, not actual development dates")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    print(json.dumps(verify(parser.parse_args().history), indent=2))
