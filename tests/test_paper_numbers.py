"""The paper's numbers come from results/: every number paper_numbers.py computes must appear in README.md."""

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def numbers():
    cwd = os.getcwd()
    os.chdir(ROOT)                      # paper_numbers reads results/ and the scripts by relative path
    try:
        import paper_numbers
        return paper_numbers, paper_numbers.numbers(), open("README.md").read()
    finally:
        os.chdir(cwd)


def test_every_number_in_the_paper_matches_the_results(numbers):
    pn, n, readme = numbers
    out = pn.check(readme, n)
    missing = out[~out["in README"]]
    assert missing.empty, "README.md doesn't say:\n" + "\n".join(f"  {r.where} / {r.what}: {r.text!r}"
                                                                  for r in missing.itertuples())


def test_a_changed_number_is_caught(numbers):
    pn, n, readme = numbers
    assert "sensitivity **98.4%**" in readme
    out = pn.check(readme.replace("sensitivity **98.4%**", "sensitivity **98.5%**"), n)
    assert list(out.loc[~out["in README"], "what"]) == ["sensitivity"]
