"""SCRIPTS.md stays complete and correct: every script is listed, every output it names exists."""

import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = open(os.path.join(ROOT, "SCRIPTS.md")).read()


def test_every_script_is_in_the_map():
    scripts = {os.path.basename(f) for f in glob.glob(os.path.join(ROOT, "*.py"))}
    missing = sorted(s for s in scripts if f"`{s}`" not in MAP)
    assert not missing, f"not in SCRIPTS.md: {missing}"


def test_every_listed_output_exists():
    paths = set(re.findall(r"`((?:results|figures|models)/[^`*{]+?)`", MAP))
    missing = sorted(p for p in paths if not os.path.exists(os.path.join(ROOT, p)))
    assert not missing, f"listed in SCRIPTS.md but missing: {missing}"


def test_archived_scripts_are_not_imported_by_current_code():
    archived = re.findall(r"\[`(\w+)\.py`\]", MAP.split("## Archived", 1)[1])
    current = [f for f in glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(os.path.join(ROOT, "tests", "**", "*.py"),
                                                                            recursive=True)
               if os.path.splitext(os.path.basename(f))[0] not in archived]
    for f in current:
        src = open(f).read()
        for a in archived:
            assert not re.search(rf"^\s*(from|import)\s+{a}\b", src, re.M), f"{os.path.basename(f)} imports archived {a}"
