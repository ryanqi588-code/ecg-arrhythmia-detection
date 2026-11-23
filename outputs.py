"""Where scripts write their outputs, so a re-run can't overwrite the committed ones.

Everything under results/, figures/, models/, tests/golden/ and
registration_evidence/ is committed: the registered network, the saved
test results the paper quotes, the golden test data. A script that
re-creates one of these files writes it to rerun/<same path> instead
(rerun/ is gitignored), so the new output can be compared with the
committed one, e.g. `diff results/x.csv rerun/results/x.csv`. To replace
the committed file, run the script with --force:

    python afib_cpsc_test.py            # writes rerun/results/afib_cpsc_test.csv
    python afib_cpsc_test.py --force    # overwrites results/afib_cpsc_test.csv

Files that don't exist yet are written in place. Scripts still read their
inputs from the committed paths, so a chain of re-runs (e.g.
afib_compare_dnn.py, then a test that reads its probabilities) uses the
committed inputs unless the earlier step was run with --force.
Standard library only, so the TensorFlow environment can use it too.
"""

import os
import sys

PROTECTED = ("results", "figures", "models", "registration_evidence", os.path.join("tests", "golden"))
RERUN_DIR = "rerun"
ROOT = os.path.dirname(os.path.abspath(__file__))

FORCE = "--force" in sys.argv
if FORCE:
    sys.argv = [a for a in sys.argv if a != "--force"]      # scripts that read sys.argv don't see it


def _protected(rel):
    return any(rel == p or rel.startswith(p + os.sep) for p in PROTECTED)


def write_path(path):
    """The path to write `path` to: itself, or rerun/<path> if it would overwrite a committed file."""
    full = os.path.abspath(path)
    rel = os.path.relpath(full, ROOT)
    if FORCE or rel.startswith("..") or not _protected(rel) or not os.path.exists(full):
        return path
    new = os.path.join(ROOT, RERUN_DIR, rel)
    os.makedirs(os.path.dirname(new), exist_ok=True)
    print(f"[outputs] {rel} is committed; writing {os.path.join(RERUN_DIR, rel)} instead (--force to overwrite)",
          file=sys.stderr)
    return new
