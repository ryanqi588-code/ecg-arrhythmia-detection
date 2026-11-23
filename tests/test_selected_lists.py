"""The committed recording lists (results/*_selected.txt) cover every recording in the saved results.

Needs no downloaded data: each list is checked against the per-recording
results the tests and networks saved. Listed recordings missing from the
results are the unscorable ones, so only results -> list is checked.
"""

import os

import pandas as pd
import pytest

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def listed(name):
    rows = [l.split() for l in open(os.path.join(R, f"{name}_selected.txt")).read().splitlines() if l.strip()]
    return {r[0].split("/")[-1]: (r[1] if len(r) > 1 else None) for r in rows}


@pytest.mark.parametrize("name, n", [("cpsc2018", 931), ("cpsc2018_extra", 1236), ("ptbxl", 2875),
                                     ("ptbxl_flutter_dev", 374), ("chapman", 3815), ("chapman_flutter_dev", 1753),
                                     ("ningbo", 2541), ("georgia", 2942), ("incart", 447)])
def test_list_is_committed_with_its_size(name, n):
    lines = open(os.path.join(R, f"{name}_selected.txt")).read().splitlines()
    assert len([l for l in lines if l.strip()]) == n


@pytest.mark.parametrize("name, results", [("cpsc2018", "afib_cpsc_test_recordings"),
                                           ("cpsc2018_extra", "afib_cpscx_test_recordings"),
                                           ("chapman", "afib_chapman_test_recordings"),
                                           ("georgia", "afib_georgia_test_recordings"),
                                           ("ningbo", "compare_probs_ningbo")])
def test_every_scored_recording_is_listed_with_its_label(name, results):
    lst = listed(name)
    t = pd.read_csv(os.path.join(R, f"{results}.csv"))
    recs = t.record.astype(str).str.split("/").str[-1]
    assert recs.isin(lst.keys()).all()
    if all(v is not None for v in lst.values()):
        assert (recs.map(lst) == t.label.astype(str)).all()


def test_ningbo_networks_ran_on_exactly_the_list():
    t = pd.read_csv(os.path.join(R, "compare_probs_ningbo.csv"))
    assert set(t.record.astype(str)) == set(listed("ningbo"))
