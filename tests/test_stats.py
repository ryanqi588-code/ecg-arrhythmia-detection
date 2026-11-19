"""The statistics behind every confidence interval: Wilson intervals and the patient-level bootstrap."""

import ast
import os

import numpy as np
import pytest

from stats import B, boot_ci, rate_str, wilson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ wilson

@pytest.mark.parametrize("k, n, lo, hi", [
    # Newcombe RG, Two-sided confidence intervals for the single proportion: comparison of seven
    # methods. Stat Med 1998;17:857-872 (Wilson score interval, the paper's worked examples)
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0, 0.1611),
    (1, 29, 0.0061, 0.1718),
    # the README's doctor comparison: 12 of 13 AF caught
    (12, 13, 0.6669, 0.9863),
])
def test_wilson_matches_published_values(k, n, lo, hi):
    a, b = wilson(k, n)
    assert a == pytest.approx(lo, abs=5e-5) and b == pytest.approx(hi, abs=5e-5)


def test_wilson_is_symmetric_and_stays_in_range():
    for n in (1, 7, 50, 1000):
        for k in range(0, n + 1, max(1, n // 7)):
            lo, hi = wilson(k, n)
            assert -1e-12 <= lo <= k / n <= hi <= 1 + 1e-12
            lo2, hi2 = wilson(n - k, n)
            assert lo == pytest.approx(1 - hi2) and hi == pytest.approx(1 - lo2)


def test_wilson_narrows_with_more_data():
    widths = [np.subtract(*wilson(n // 10, n)[::-1]) for n in (10, 100, 1000, 10000)]
    assert all(a > b for a, b in zip(widths, widths[1:]))


# ------------------------------------------------------------------ boot_ci

def _data(seed=1, patients=200):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(patients), rng.integers(1, 4, patients))     # 1-3 recordings each
    x = rng.random(len(groups)) < 0.3
    return x, groups


def test_boot_ci_is_identical_with_a_fixed_seed():
    x, g = _data()
    a = boot_ci(lambda i: x[i].mean(), g, seed=0)
    b = boot_ci(lambda i: x[i].mean(), g, seed=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, boot_ci(lambda i: x[i].mean(), g, seed=1))


def test_boot_ci_default_seed_and_resamples():
    x, g = _data()
    assert B == 2000
    assert np.array_equal(boot_ci(lambda i: x[i].mean(), g), boot_ci(lambda i: x[i].mean(), g, n=2000, seed=0))


def test_boot_ci_resamples_whole_patients():
    # every patient has exactly 3 recordings, so every resample has a multiple of 3 and keeps them together
    g = np.repeat(np.arange(40), 3)
    seen = []
    boot_ci(lambda i: seen.append(i) or 0.0, g, n=50)
    for i in seen:
        assert len(i) == len(g)
        assert all(set(i[k:k + 3]) == {i[k], i[k] + 1, i[k] + 2} for k in range(0, len(i), 3))


def test_boot_ci_brackets_the_estimate_and_agrees_with_wilson():
    # one recording per patient: the bootstrap interval of a proportion is close to Wilson's
    rng = np.random.default_rng(3)
    x = rng.random(2000) < 0.2
    lo, hi = boot_ci(lambda i: x[i].mean(), np.arange(len(x)))
    assert lo < x.mean() < hi
    wlo, whi = wilson(int(x.sum()), len(x))
    assert lo == pytest.approx(wlo, abs=0.004) and hi == pytest.approx(whi, abs=0.004)


def test_boot_ci_paired_difference_of_identical_calls_is_zero():
    x, g = _data()
    assert np.array_equal(boot_ci(lambda i: x[i].mean() - x[i].mean(), g), [0.0, 0.0])


# ------------------------------------------------------------------ rate_str

def test_rate_str_format():
    assert rate_str([True] * 12 + [False]) == "92.3% (66.7%–98.6%) [12/13]"
    assert rate_str([]) == "–"


# ------------------------------------------------------- the registered copies

def _function_source(path, name):
    tree = ast.parse(open(path).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node)


@pytest.mark.parametrize("script, name", [
    ("afib_sph_test.py", "wilson"), ("afib_chapman_test.py", "wilson"), ("afib_cpscx_test.py", "wilson"),
    ("afib_georgia_test.py", "wilson"), ("afib_ptbxl_test.py", "wilson"),
    ("afib_comparison_test.py", "boot_ci"), ("afib_comparison_test.py", "rate_str"),
])
def test_registered_copies_are_identical_to_stats(script, name):
    # read, not imported: several of these scripts run their test on import
    assert _function_source(os.path.join(ROOT, script), name) == _function_source(os.path.join(ROOT, "stats.py"), name)


def _constant(path, name):
    tree = ast.parse(open(path).read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def test_registered_bootstrap_resample_count_matches_stats():
    # boot_ci's default n=B comes from each script's own B, which the code comparison above doesn't cover
    assert _constant(os.path.join(ROOT, "afib_comparison_test.py"), "B") == B == 2000
