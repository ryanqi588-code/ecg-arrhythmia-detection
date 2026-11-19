"""Statistics helpers behind every confidence interval in the paper: one copy, for new code.

wilson, boot_ci and rate_str are copied unchanged from the scripts that
produced the published results: wilson from afib_sph_test.py (the same
code is in four other test scripts), boot_ci and rate_str from
afib_comparison_test.py. Those registered copies stay as they are, so the
history stays checkable; tests/test_stats.py checks that every copy is
still identical to this one.
"""

import numpy as np

B = 2000        # bootstrap resamples


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def boot_ci(stat, groups, n=B, seed=0):
    """Paired bootstrap 95% interval of stat(index array), resampling groups (patients)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(stat(np.concatenate([members[g] for g in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])


def rate_str(hits):
    hits = np.asarray(hits, bool)
    if not len(hits):
        return "–"
    lo, hi = wilson(int(hits.sum()), len(hits))
    return f"{hits.mean():.1%} ({lo:.1%}–{hi:.1%}) [{int(hits.sum())}/{len(hits)}]"
