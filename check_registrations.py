"""Check the registration claims against GitHub's own record of when each commit was pushed.

Commit times in git are written by the author's computer and can be set to
anything, so on their own they prove nothing. GitHub keeps a separate,
server-side log of every push to the repository (the repository activity
API, the same data as the repository's "Activity" page), which the author
can't edit. This script, for each registered test:

  1. checks the registration commit comes before the results commit in the
     published history;
  2. finds the first push that carried each commit (GitHub's time, UTC),
     and whether the registration was pushed in an earlier, separate push;
  3. lists force pushes (which could replace history) after the first
     registration;
  4. lists the Python files changed between the registration and the
     results commit, with lines added and removed, so you can read
     exactly what changed (git diff <registration> <results> -- '*.py').

Run it in a full clone (not shallow), with network access:
    git clone https://github.com/ryanqi588-code/ecg-arrhythmia-detection
    cd ecg-arrhythmia-detection && python check_registrations.py
Set GITHUB_TOKEN to avoid the unauthenticated rate limit (60 requests an
hour; the script needs one request per 100 pushes). Standard library only.

What this can't show: when the test data were downloaded or opened. Data
sources (PhysioNet, figshare, Zenodo) keep no public download log. What it
does show is that each registered plan was public, in a form that can't
later be changed without a force push on record, before the results were.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import csv
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.request

REPO = "ryanqi588-code/ecg-arrhythmia-detection"

# (test, registration commit, results commit, registered test script)
# The paper numbers its tests 1-5; the repository keeps the project's own numbers (in script names,
# commit messages and the supplement's full log), so the history stays checkable.
PAPER_TEST = {"9": "1", "11": "2", "12": "3", "17": "4", "18": "5"}

REGISTRATIONS = [
    ("2", "efc7dfa", "57c7e27", "afib_svdb_test.py"),
    ("3", "d52e2f5", "2df82b5", "afib_cinc_test.py"),
    ("4", "5f92136", "7b2b983", "afib_cpsc_test.py"),
    ("5", "23da508", "f923132", "afib_cpscx_test.py"),
    ("6", "e1b24b9", "f9d8fe7", "afib_georgia_test.py"),
    ("7", "dba1e05", "290a3c5", "afib_ptbxl_test.py"),
    ("8", "fa9efa8", "3f023d1", "afib_chapman_test.py"),
    ("9", "759d00c", "a55d260", "afib_sph_test.py"),
    ("10", "85c65d2", "b11926a", "afib_flutter_test.py"),
    ("11", "d8c5ce2", "cc3a4a8", "afib_comparison_test.py"),
    ("12", "6da6e6b", "1ba7844", "afib_unmeasured_test.py"),
    ("13", "007f6ab", "258bf77", "afib_hybrid_test.py"),
    ("14", "9d82cbf", "85017f0", "afib_mimic_test.py"),
    ("15", "041a560", "47339b8", "afib_mimic15_test.py"),
    ("16", "ba3839e", "2f9bf16", "afib_slowaf_test.py"),
    ("label check", "e14302f", "9cf49b2", "afib_mimic_labels.py"),
    ("17", "8884656", "f09d3d7", "afib_abstain_test.py"),
    ("18", "9d6edc8", "36e30aa", "afib_afpac_test.py"),
]


def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def is_ancestor(a, b):
    return git("merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def known(sha):
    return git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def activity():
    url = f"https://api.github.com/repos/{REPO}/activity?per_page=100&direction=asc"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "registration-check"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:                                   # python.org builds on macOS ship without CA certificates
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    events = []
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), context=ctx) as resp:
            events += json.load(resp)
            nxt = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
            url = nxt.group(1) if nxt else None
    return sorted(events, key=lambda e: e["timestamp"])


def first_push(sha, pushes):
    for e in pushes:
        if known(e["after"]) and is_ancestor(sha, e["after"]):
            return e
    return None


def diff_summary(a, b):
    rows = git("diff", "--numstat", a, b, "--", "*.py").stdout.split("\n")
    return [f"{p} (+{add} -{rem})" for add, rem, p in (r.split("\t") for r in rows if r)]


def main():
    if git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        sys.exit("This is a shallow clone; run `git fetch --unshallow` first.")
    pushes = [e for e in activity() if e["activity_type"] in ("push", "force_push", "branch_creation")
              and e["ref"] == "refs/heads/main"]
    head = git("rev-parse", "origin/main", check=False).stdout.strip() or "HEAD"

    rows, ok_all = [], True
    for test, reg, res, script in REGISTRATIONS:
        in_history = is_ancestor(reg, res) and is_ancestor(res, head)
        script_registered = git("cat-file", "-e", f"{reg}:{script}", check=False).returncode == 0
        p_reg, p_res = first_push(reg, pushes), first_push(res, pushes)
        separate = bool(p_reg and p_res and p_reg["timestamp"] < p_res["timestamp"])
        ok_all &= in_history and script_registered and p_reg is not None
        rows.append({"test": test, "paper test": PAPER_TEST.get(test, ""), "registration": reg, "results": res, "script": script,
                     "order in history ok": in_history, "script in registration commit": script_registered,
                     "registration pushed (GitHub, UTC)": p_reg["timestamp"] if p_reg else "not found",
                     "results pushed (GitHub, UTC)": p_res["timestamp"] if p_res else "not found",
                     "registration pushed separately, before results": separate,
                     "Python changed between": "; ".join(diff_summary(reg, res)) or "none"})

    first_reg = min(r["registration pushed (GitHub, UTC)"] for r in rows)
    forced = [e for e in pushes if e["activity_type"] == "force_push"]
    late_forced = [e for e in forced if e["timestamp"] >= first_reg]

    w = max(len(r["test"]) for r in rows)
    print("Paper test numbers: " + ", ".join(f"project test {k} = paper test {v}" for k, v in PAPER_TEST.items()))
    print(f"{'test':<{w}}  paper  registration  pushed (UTC)          results  pushed (UTC)          separate  "
          f"Python changed between registration and results")
    for r in rows:
        print(f"{r['test']:<{w}}  {r['paper test']:<5}  {r['registration']:<12}  {r['registration pushed (GitHub, UTC)']:<20}  "
              f"{r['results']}  {r['results pushed (GitHub, UTC)']:<20}  "
              f"{'yes' if r['registration pushed separately, before results'] else 'no':<8}  "
              f"{r['Python changed between']}")
    print(f"\nForce pushes on main: {len(forced)} "
          f"({', '.join(e['timestamp'] for e in forced) or 'none'}); "
          f"after the first registration push ({first_reg}): {len(late_forced)}")
    print("Every registration commit is in the published history before its results, "
          "contains its test script, and was pushed:", "yes" if ok_all else "NO")

    with open(write_path("results/registration_check.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return 0 if ok_all and not late_forced else 1


if __name__ == "__main__":
    sys.exit(main())
