"""outputs.write_path: a re-run writes to rerun/ instead of over a committed file, unless --force."""

import os

import pytest

import outputs


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(outputs, "ROOT", str(tmp_path))
    monkeypatch.setattr(outputs, "FORCE", False)
    for p in ["results/a.csv", "models/cnn_model.keras", "figures/f.png", "tests/golden/expected.json", "data/x.npy"]:
        os.makedirs(tmp_path / os.path.dirname(p), exist_ok=True)
        (tmp_path / p).write_text("committed")
    return tmp_path


@pytest.mark.parametrize("p", ["results/a.csv", "models/cnn_model.keras", "figures/f.png", "tests/golden/expected.json"])
def test_committed_file_is_redirected_to_rerun(repo, monkeypatch, p):
    monkeypatch.chdir(repo)
    assert write(p) == os.path.join(str(repo), "rerun", p)
    assert (repo / p).read_text() == "committed"


def write(p):
    path = outputs.write_path(p)
    open(path, "w").write("new")
    return path


def test_absolute_paths_are_redirected_too(repo, monkeypatch):
    monkeypatch.chdir(repo)
    assert outputs.write_path(str(repo / "results" / "a.csv")) == os.path.join(str(repo), "rerun", "results", "a.csv")


def test_force_overwrites(repo, monkeypatch):
    monkeypatch.chdir(repo)
    monkeypatch.setattr(outputs, "FORCE", True)
    assert write("results/a.csv") == "results/a.csv"
    assert (repo / "results/a.csv").read_text() == "new"


def test_new_outputs_and_caches_are_written_in_place(repo, monkeypatch):
    monkeypatch.chdir(repo)
    assert outputs.write_path("results/new.csv") == "results/new.csv"      # doesn't exist yet
    assert outputs.write_path("data/x.npy") == "data/x.npy"                # not a committed folder


def test_no_script_writes_a_committed_path_directly():
    # Every write of a results/, figures/ or models/ literal goes through write_path.
    import glob
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for f in glob.glob(os.path.join(root, "*.py")) + glob.glob(os.path.join(root, "tests", "golden", "*.py")):
        s = open(f).read()
        for m in re.finditer(r'(\.(?:to_csv|savefig|to_pickle|save)|np\.save\w*|open)\(\s*f?"(results|figures|models|registration_evidence)/[^"]*"'
                             r'(?:,\s*"[wa])?', s):
            if m.group(1) != "open" or re.search(r',\s*"[wa]', m.group(0)):
                bad.append(f"{os.path.basename(f)}: {m.group(0)}")
    assert not bad, "\n".join(bad)
