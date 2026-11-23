"""Look for duplicate recordings within and across the 12-lead datasets the tests and development used.

A recording used for development and again in a test (say CPSC 2018 and
SPH, or Chapman and Ningbo, released together in Challenge 2021) would
leak; a duplicate within a set counts one ECG twice. Two passes:

  exact  - each signal (mV, padding stripped, rounded to 1 µV) is hashed;
           equal hashes are identical copies. This only finds copies
           stored the same way: the same ECG resampled, re-filtered or
           saved in another format (SPH's float16, CPSC's integers) hashes
           differently.
  timing - format-independent. Each recording's median beat over the 12
           leads (after a common 0.5-40 Hz band-pass) proposes its 20
           nearest neighbours. A pair's beat times are lined up at the
           shift that matches the most beats; if at least 5 beats, and 80%
           of those in the overlap, line up within 20 ms, the two signals
           are aligned at that shift and correlated over all 12 leads.
           A duplicate has a median per-lead correlation >= 0.95 and the
           same amplitude in each lead (within 7%), or a correlation >= 0.99
           at any amplitude (the same tracing stored at another gain, as in
           CODE-15%); different ECGs with very regular rhythms can line up
           their beats, but they differ in amplitude or correlate less. (Per lead, so that one lead with a
           large artifact can't decide it.) Neighbours are searched in the pooled data, within each
           dataset, and between datasets that could share recordings (the
           Chinese ones; CODE-15% and the CODE test set).

Positive controls check the timing pass: 60 recordings (CPSC, Chapman,
SPH) are resampled to 400 Hz, cropped 1.37 s later, band-pass filtered and
stored as float16, and each must be matched to its original.

Scanned: CPSC 2018 and CPSC-extra, PTB-XL, Chapman-Shaoxing, Ningbo,
Georgia (the lists in results/), all of SPH, CODE-15% parts 0-3, the CODE
test set, and MIMIC-IV-ECG's selected recordings. Not scanned: INCART and
the Holter databases (long recordings, not 10 s ECGs) and CinC 2017
(single lead). Writes results/duplicate_check.csv (each pair found) and
results/duplicate_check_sources.csv (per source: recordings scanned, and
positive controls made from it and found).

    python check_duplicates.py      # ~10 min with all cores; fingerprints cached in data/
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import glob
import hashlib
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

CACHE = "data/duplicate_fingerprints.pkl"
FS_MORPH = 100                       # median beat resampled to this rate, -200 to +400 ms
TOL_S, MIN_BEATS, MIN_SHARE = 0.02, 5, 0.8    # beats line up within 20 ms: at least 5, and 80% of the overlap
K = 20                                         # beat-shape neighbours checked per recording
# A duplicate's aligned signals: median over the 12 leads of the per-lead correlation, and of the
# |log amplitude ratio| (SD over the overlap). The 60 positive controls (resampled, re-filtered,
# float16) gave correlation >= 0.963 and amplitude difference <= 0.065; of 2,769 beat-timing
# matches between different recordings the median amplitude difference was 0.19 (5th centile 0.08).
MIN_CORR, MAX_AMP = 0.95, 0.07
# A pure rescaling: CODE-15% holds the same tracings stored at two gains (factor 1.282), which
# correlate 0.996-1.000; no two different recordings anywhere reached 0.95 on this measure.
RESCALED_CORR = 0.99
REVIEW_CORR = 0.9                              # timing matches from here up that aren't duplicates are listed
MATCHES = "data/duplicate_matches.pkl"         # beat-timing matches (the slow step), cached
# Datasets that could share recordings: searched against each other, not only in the pooled search,
# so a copy isn't crowded out of the neighbours by a large unrelated set (CODE-15%'s 80,000).
GROUPS = [["cpsc", "cpsc_extra", "chapman", "ningbo", "sph", "control"], ["code15", "codetest"]]


# ------------------------------------------------------------ recordings

def items():
    """(source, id, loader args) for every recording to scan."""
    out = []
    for r in open("results/cpsc2018_selected.txt").read().split():
        out.append(("cpsc", r, ("wfdb", f"data/cpsc2018/{r}")))
    for r in open("results/cpsc2018_extra_selected.txt").read().split():
        out.append(("cpsc_extra", r, ("wfdb", f"data/cpsc2018_extra/{r}")))
    ptb = {l.split()[0] for f in ["ptbxl_selected", "ptbxl_flutter_dev_selected"] for l in open(f"results/{f}.txt")}
    out += [("ptbxl", p, ("wfdb", f"data/ptbxl/{p}")) for p in sorted(ptb)]
    ch = {os.path.basename(l.split()[0]) for f in ["chapman_selected", "chapman_flutter_dev_selected"]
          for l in open(f"results/{f}.txt")}
    out += [("chapman", r, ("wfdb", f"data/chapman/{r}")) for r in sorted(ch)]
    for l in open("results/ningbo_selected.txt"):
        r = os.path.basename(l.split()[0])
        out.append(("ningbo", r, ("wfdb", f"data/ningbo/rec/{r}")))
    for l in open("results/georgia_selected.txt"):
        out.append(("georgia", l.split()[0], ("wfdb", f"data/georgia/{l.split()[0]}")))
    for p in sorted(glob.glob("data/sph/records/*.h5")):
        out.append(("sph", os.path.basename(p)[:-3], ("sph", p)))
    exams = pd.read_csv("data/code15/exams.csv")
    import h5py
    for part in range(4):
        f = f"exams_part{part}.hdf5"
        known = set(exams.exam_id[exams.trace_file == f])
        with h5py.File(f"data/code15/{f}", "r") as h:
            ids = h["exam_id"][()]
        out += [("code15", int(e), ("h5", f"data/code15/{f}", i, 400)) for i, e in enumerate(ids) if e in known]
    out += [("codetest", i, ("h5", "data/ribeiro/data/ecg_tracings.hdf5", i, 400)) for i in range(827)]
    from afib_mimic_select import MIMIC_DIR, read_selected
    seen = set()
    for name in ["mimic_dev_selected", "mimic_selected", "mimic15_selected", "mimic16_selected", "mimic18_selected"]:
        for path, _, _ in read_selected(name):
            if path not in seen:
                seen.add(path)
                out.append(("mimic", path.split("/")[-1], ("mimic", f"{MIMIC_DIR}/{path}")))
    return out


def load(args, h5cache={}):
    kind = args[0]
    if kind == "wfdb":
        from af import load_12lead_all
        return load_12lead_all(args[1])
    if kind == "sph":
        import h5py
        with h5py.File(args[1], "r") as f:
            return f["ecg"][()].T.astype(float), 500
    if kind == "h5":
        import h5py
        if args[1] not in h5cache:
            h5cache[args[1]] = h5py.File(args[1], "r")
        return h5cache[args[1]]["tracings"][args[2]].astype(float), args[3]
    if kind == "array":                              # a positive control's altered signal
        return args[1], args[2]
    if kind == "mimic":
        from afib_mimic_select import load_mimic
        return load_mimic(args[1])
    raise ValueError(kind)


# ------------------------------------------------------------ fingerprints

def fingerprint(sig, fs):
    from ecg import detect_peaks_multilead
    from predict import strip_flat_edges
    sig = strip_flat_edges(np.nan_to_num(np.asarray(sig, float)))
    if len(sig) < fs:
        return None
    sha = hashlib.sha1(np.round(sig * 1000).astype(np.int64).tobytes() + str(sig.shape).encode()).hexdigest()
    peaks = np.asarray(detect_peaks_multilead(sig[:, [0, 1, 6]], fs))
    morph = None
    from scipy.signal import butter, sosfiltfilt
    bp = sosfiltfilt(butter(2, [0.5, 40], "bandpass", fs=fs, output="sos"), sig, axis=0)
    pre, post = int(0.2 * fs), int(0.4 * fs)
    beats = [bp[p - pre:p + post] for p in peaks if p - pre >= 0 and p + post <= len(sig)]
    if len(beats) >= 2:
        med = np.median(beats, axis=0)
        med = med - np.median(med[: int(0.08 * fs)], axis=0)
        x = np.linspace(0, len(med) - 1, int(0.6 * FS_MORPH))
        morph = np.stack([np.interp(x, np.arange(len(med)), med[:, j]) for j in range(12)], 1).ravel().astype(np.float32)
    return {"fs": fs, "n": len(sig), "sha": sha, "t": (peaks / fs).astype(np.float64), "morph": morph}


def _work(chunk):
    out = []
    for source, rid, args in chunk:
        try:
            fp = fingerprint(*load(args))
        except Exception as e:                      # unreadable file: counted, not fatal
            fp = {"error": f"{type(e).__name__}: {e}"}
        out.append((source, rid, args, fp))
    return out


def controls(n_each=20, seed=0):
    """Altered copies of real recordings that the timing pass must match to their originals."""
    from scipy.signal import butter, resample_poly, sosfiltfilt
    from af import load_12lead_all
    rng = np.random.default_rng(seed)
    pick = []
    cps = open("results/cpsc2018_selected.txt").read().split()
    pick += [("cpsc", r, lambda r=r: load_12lead_all(f"data/cpsc2018/{r}")) for r in rng.choice(cps, n_each, replace=False)]
    chs = [os.path.basename(l.split()[0]) for l in open("results/chapman_selected.txt")]
    pick += [("chapman", r, lambda r=r: load_12lead_all(f"data/chapman/{r}")) for r in rng.choice(chs, n_each, replace=False)]
    sph = sorted(glob.glob("data/sph/records/*.h5"))
    pick += [("sph", os.path.basename(p)[:-3], lambda p=p: load(("sph", p))) for p in rng.choice(sph, n_each, replace=False)]
    out = []
    for source, rid, get in pick:
        sig, fs = get()
        sig = resample_poly(np.asarray(sig, float), 4, 5, axis=0) if fs == 500 else np.asarray(sig, float)
        sig = sig[int(1.37 * 400): int(1.37 * 400) + 4000]
        sig = sosfiltfilt(butter(2, [0.5, 40], "bandpass", fs=400, output="sos"), sig, axis=0)
        sig = sig.astype(np.float16).astype(float)
        out.append(("control", f"{source}:{rid}", ("array", sig, 400), fingerprint(sig, 400)))
    return out


def fingerprints(todo=None, cache=CACHE):
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    todo = items() if todo is None else todo
    chunks = [todo[i:i + 400] for i in range(0, len(todo), 400)]
    rows = []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        for k, res in enumerate(pool.map(_work, chunks)):
            rows += res
            if k % 25 == 0:
                print(f"{len(rows)} of {len(todo)} ...", flush=True)
    rows += controls()
    d = pd.DataFrame(rows, columns=["source", "id", "args", "fp"])
    d.to_pickle(cache)
    return d


# ------------------------------------------------------------ matching

def beat_match(ta, tb):
    """Shift of b's beat times that lines up the most beats with a's, how many line up, and the overlap's beats."""
    if len(ta) < 2 or len(tb) < 2:
        return 0.0, 0, 0
    diff = (ta[:, None] - tb[None, :]).ravel()
    best, off = 0, 0.0
    for start in (0.0, TOL_S / 2):                       # two binnings, so a cluster isn't split by a bin edge
        bins = np.arange(diff.min() - TOL_S + start, diff.max() + 2 * TOL_S, TOL_S)
        h, e = np.histogram(diff, bins)
        k = int(np.argmax(h))
        if h[k] > best:
            best, off = h[k], float(np.median(diff[(diff >= e[k]) & (diff < e[k + 1])]))
    tbs = tb + off
    near = np.abs(ta[:, None] - tbs[None, :]) <= TOL_S
    matched = int(near.any(axis=0).sum())
    lo, hi = max(ta[0], tbs[0]) - TOL_S, min(ta[-1], tbs[-1]) + TOL_S
    overlap = min(int(((ta >= lo) & (ta <= hi)).sum()), int(((tbs >= lo) & (tbs <= hi)).sum()))
    return off, matched, overlap


def aligned_corr(ra, rb, off, fa, fb):
    """Over the overlap, b shifted by `off` s: median per-lead correlation, and median |log amplitude ratio|.

    Each lead detrended (0.6 s moving average) and resampled to 250 Hz.
    """
    from scipy.ndimage import uniform_filter1d
    from predict import strip_flat_edges
    sa, fsa = load(ra)
    sb, fsb = load(rb)
    sa, sb = strip_flat_edges(np.nan_to_num(sa)), strip_flat_edges(np.nan_to_num(sb))
    lo, hi = max(fa["t"][0], fb["t"][0] + off), min(fa["t"][-1], fb["t"][-1] + off)
    grid = np.arange(lo, hi, 1 / 250)
    if len(grid) < 250:
        return np.nan, np.nan
    def take(s, fs, shift):
        x = np.stack([np.interp(grid - shift, np.arange(len(s)) / fs, s[:, j]) for j in range(12)], 1)
        return x - uniform_filter1d(x, int(0.6 * 250), axis=0)
    xa, xb = take(sa, fsa, 0.0), take(sb, fsb, off)
    sda, sdb = xa.std(0), xb.std(0)
    live = (sda > 1e-6) & (sdb > 1e-6)
    if not live.any():
        return np.nan, np.nan
    corr = [np.corrcoef(xa[:, j], xb[:, j])[0, 1] for j in np.flatnonzero(live)]
    return float(np.median(corr)), float(np.median(np.abs(np.log(sda[live] / sdb[live]))))


def _corr(job):
    try:
        return aligned_corr(*job)
    except Exception:
        return np.nan, np.nan


def _match(chunk):
    out = []
    for a, b, ta, tb in chunk:
        off, matched, overlap = beat_match(ta, tb)
        if matched >= MIN_BEATS and matched >= MIN_SHARE * overlap:
            out.append((a, b, matched, off))
    return out


def main(todo=None, cache=CACHE, write=True):
    d = fingerprints(todo, cache)
    ok = d.fp.map(lambda f: f is not None and "error" not in f)
    d["has_timing"] = d.fp.map(lambda f: f is not None and "error" not in f and f["morph"] is not None
                                and len(f["t"]) >= MIN_BEATS)
    src = d[d.source != "control"].groupby("source").agg(scanned=("id", "size")).assign(
        unreadable=d[d.source != "control"].assign(bad=~ok).groupby("source").bad.sum(),
        too_few_beats_for_timing=d[d.source != "control"].assign(x=ok & ~d.has_timing).groupby("source").x.sum())
    print(src.to_string())

    pairs = {}
    # exact pass
    real = d[ok & (d.source != "control")]
    for sha, g in real.groupby(real.fp.map(lambda f: f["sha"])):
        for a, b in itertools.combinations(g.index, 2):
            pairs[(a, b)] = {"method": "exact", "beats": np.nan, "corr": 1.0, "amp": 0.0}
    # timing pass: candidate pairs from beat-shape neighbours
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    t = d[d.has_timing]
    Z = PCA(40, random_state=0).fit_transform(np.stack(t.fp.map(lambda f: f["morph"]).values))
    idx, srcs = t.index.values, t.source.values
    cand = set()
    def search(query, pool):
        if len(pool) < 2 or not len(query):
            return
        k = min(K + 1, len(pool))
        _, nn = NearestNeighbors(n_neighbors=k).fit(Z[pool]).kneighbors(Z[query])
        for q, row in zip(query, nn):
            for j in pool[row]:
                if j != q:
                    cand.add(tuple(sorted((idx[q], idx[j]))))
    allpos = np.arange(len(t))
    search(allpos, allpos)                                            # pooled
    for s_ in np.unique(srcs):                                        # within each dataset
        p = allpos[srcs == s_]
        search(p, p)
    for g in GROUPS:                                                  # between related datasets
        for s1, s2 in itertools.permutations([x for x in g if x in set(srcs)], 2):
            search(allpos[srcs == s1], allpos[srcs == s2])
    cand = sorted(c for c in cand if c not in pairs and not (d.source[c[0]] == d.source[c[1]] == "control"))
    mcache = cache.replace(".pkl", "_matches.pkl") if cache != CACHE else MATCHES
    if os.path.exists(mcache):
        matches = pd.read_pickle(mcache)
    else:
        jobs = [(a, b, d.fp[a]["t"], d.fp[b]["t"]) for a, b in cand]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
            matches = [m for part in pool.map(_match, [jobs[i:i + 20000] for i in range(0, len(jobs), 20000)])
                       for m in part]
        pd.to_pickle(matches, mcache)
    print(f"timing pass: {len(t)} recordings, {len(cand)} neighbour pairs checked, {len(matches)} with matching beats")
    scache = mcache.replace("matches", "scores")
    if os.path.exists(scache):
        scores = pd.read_pickle(scache)
    else:
        jobs = [(d.args[a], d.args[b], off, d.fp[a], d.fp[b]) for a, b, _, off in matches]
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
            scores = list(pool.map(_corr, jobs, chunksize=50))
        pd.to_pickle(scores, scache)
    for (a, b, matched, off), (c, amp) in zip(matches, scores):
        pairs[(a, b)] = {"method": "timing", "beats": matched, "corr": c, "amp": amp}

    rows = []
    for (a, b), p in pairs.items():
        rows.append({"source_a": d.source[a], "id_a": d.id[a], "source_b": d.source[b], "id_b": d.id[b],
                     "method": p["method"], "beats_matched": p["beats"], "corr": p["corr"], "amp_logdiff": p["amp"],
                     "duplicate": p["method"] == "exact" or p["corr"] >= RESCALED_CORR
                                  or (p["corr"] >= MIN_CORR and p["amp"] <= MAX_AMP)})
    res = pd.DataFrame(rows, columns=["source_a", "id_a", "source_b", "id_b", "method", "beats_matched", "corr",
                                      "amp_logdiff", "duplicate"])

    ctrl = res[(res.source_a == "control") | (res.source_b == "control")]
    exact = {frozenset((f"{r.source_a}:{r.id_a}", f"{r.source_b}:{r.id_b}")) for r in res[res.method == "exact"].itertuples()}
    right, wrong, explained = set(), 0, 0
    for r in ctrl.itertuples():
        c, other = (r.id_a, f"{r.source_b}:{r.id_b}") if r.source_a == "control" else (r.id_b, f"{r.source_a}:{r.id_a}")
        if c == other and r.duplicate:
            right.add(c)
        elif r.duplicate:                           # a control called a duplicate of some other recording:
            wrong += 1                              # fine if that recording is an exact copy of its original
            explained += frozenset((c, other)) in exact
    n_ctrl = int((d.source == "control").sum())
    present = sum(d.source.eq(c.split(":")[0]).any() and (d.source + ":" + d.id.astype(str)).eq(c).any()
                  for c in d.id[d.source == "control"])
    own = ctrl[[(r.id_a if r.source_a == "control" else r.id_b) ==
                (f"{r.source_b}:{r.id_b}" if r.source_a == "control" else f"{r.source_a}:{r.id_a}") for r in ctrl.itertuples()]]
    print(f"positive controls found as duplicates of their original: {len(right)} of {present} whose original was "
          f"scanned (corr {own['corr'].min():.3f}-{own['corr'].max():.3f}, amplitude difference up to "
          f"{own.amp_logdiff.max():.3f}); called a duplicate of something else: {wrong}, of which "
          f"{explained} are exact copies of the control's original")
    found = right

    res = res[(res.source_a != "control") & (res.source_b != "control")]
    near = res[~res.duplicate & (res["corr"] >= REVIEW_CORR)].sort_values("corr", ascending=False)
    print(f"\nbeat-timing matches below the duplicate cutoff: {int((~res.duplicate).sum())} "
          f"(corr up to {res.loc[~res.duplicate, 'corr'].max():.3f}); for review, corr >= {REVIEW_CORR}:")
    print(near.to_string(index=False))
    res = res[res.duplicate | (res["corr"] >= REVIEW_CORR)].sort_values(["source_a", "source_b", "id_a"])
    dup = res[res.duplicate]
    print(f"\nduplicates: {len(dup)} pairs")
    print(dup.groupby(["source_a", "source_b", "method"]).size().to_string())
    # Positive controls per source they were made from (20 each from CPSC, Chapman and SPH); none elsewhere.
    ctrl_src = d.id[d.source == "control"].str.split(":").str[0]
    src["controls_planted"] = ctrl_src.value_counts().reindex(src.index).fillna(0).astype(int)
    src["controls_found"] = (pd.Series([c.split(":")[0] for c in found]).value_counts()
                             .reindex(src.index).fillna(0).astype("Int64").mask(src.controls_planted == 0))
    if write:
        res.to_csv(write_path("results/duplicate_check.csv"), index=False)
        src.to_csv(write_path("results/duplicate_check_sources.csv"))
    return res, src


if __name__ == "__main__":
    main()
