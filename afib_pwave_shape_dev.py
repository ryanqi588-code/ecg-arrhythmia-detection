"""Development: the P-wave window's shape as model inputs, with smooth weights (after test 18).

Idea. The CNN clears many PAC recordings the 12-lead model calls AF, and
P-wave evidence seems to be what it uses. The model only sees how well
beats' P waves match each other (p_match), not what the P wave looks like.
Here the median P-wave window itself goes in, and the weights a logistic
regression gives it can be plotted over time: if they trace a P-wave-like
bump, the explanation is "a waveform shaped like this counts against AF",
which a cardiologist can check.

Fixed before running:
  - Waveform: for each development unit and each of leads II, III, aVF and
    V1, the median over regular, non-ventricular beats (as for p_match) of
    ecg.pr_segments (300 to 60 ms before the R peak, band-passed),
    averaged into 24 bins of 10 ms; minus its first 20 ms (the baseline
    before the P wave), divided by the lead's median QRS height
    (peak-to-peak within 60 ms of R). 96 numbers. With fewer than 3 such
    beats (P waves unmeasurable): the training set's mean waveform, as
    the adopted model fills p_match then, with p_unmeasured an input.
  - Model: balanced logistic regression on the adopted model's 11 inputs
    (standardised, L2 penalty as sklearn's default, C = 1) plus the 96
    points (in units of QRS height, not standardised), whose weights get
    the same L2 plus lambda * sum (w[k+1] - w[k])^2 within each lead, so
    neighbouring weights stay similar. lambda from {1, 10, 100, 1000,
    10000} by 5-fold inner cross-validation (balanced log-loss).
  - Cutoff and comparison as afib_fa_dev.fit and the adopted model: the
    strictest per-source 95%-sensitivity cutoff from 10-fold out-of-fold
    probabilities; 5-fold outer cross-validation grouped by patient, the
    same folds for both, on the adopted model's development units only
    (CPSC 2018 and CPSC-extra crops, PTB-XL, Chapman); no test set.
  - Decision rule: goes on to a registered test on new data if, against
    the adopted model, PAC false alarms fall with the 95% interval of the
    difference (paired bootstrap over patients) wholly below 0, AF caught
    falls by no more than 1 point, and the AF-vs-PAC AUC doesn't fall.
    Nothing is adopted here.
  - The weight plot (fitted on all the development data) is made whatever
    the result: figures/afib_pwave_shape_weights.png.

Not a first look at these data for PAC false alarms: several earlier
feature ideas were developed on them (supplement S8), so a pass here is
more likely to be chance than a first attempt's would be.

    python afib_pwave_shape_dev.py    # computes the waveforms once (~25 min, cached in data/), then minutes
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import afib_pac_dev
from af import CLINICAL_FEATURES_12, LEADS_12, auc, model_inputs, regular_beats
from afib_fa_dev import CUTOFF_SOURCES, fit, strictest_cutoff, with_unmeasured_fix
from afib_pac_dev import dev_units, prepare
from stats import boot_ci, rate_str

FEATURES = CLINICAL_FEATURES_12 + ["p_unmeasured"]
P_LEADS = ["II", "III", "AVF", "V1"]
N_BINS, BIN_S, BASE_S, QRS_S = 24, 0.010, 0.020, 0.060
WAVE = [f"pw_{lead}_{k}" for lead in P_LEADS for k in range(N_BINS)]
LAMBDAS = [1, 10, 100, 1000, 10000]
CACHE = "data/pac_dev_multilead12_pwave.pkl"
BEAT_FEATURES = ["cv", "nrmssd", "entropy", "entropy_norm", "tpr", "pnn50", "ectopic_frac"]


# ------------------------------------------------------------ the waveform

def pwave_shape(sig12, fs):
    """{pw_<lead>_<k>: ...} - the normalised median P-wave window, 24 x 10 ms per lead (NaN if unmeasurable)."""
    from ecg import bandpass, detect_peaks_multilead, flag_ectopic, pr_segments
    peaks = np.asarray(detect_peaks_multilead(sig12[:, [0, 1, 6]], fs))
    out = {c: np.nan for c in WAVE}
    if len(peaks) < 4:
        return out
    use = regular_beats(peaks, fs) & ~flag_ectopic(sig12[:, 1], fs, peaks)
    use &= peaks - int(0.30 * fs) >= 0
    if use.sum() < 3:
        return out
    q = int(QRS_S * fs)
    for lead in P_LEADS:
        filt = bandpass(sig12[:, LEADS_12.index(lead)], fs)
        med = np.median(pr_segments(filt, peaks, fs)[use], axis=0)
        edges = np.linspace(0, len(med), N_BINS + 1).round().astype(int)
        w = np.array([med[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
        w -= w[: int(round(BASE_S / BIN_S))].mean()
        beats = [filt[p - q:p + q] for p in peaks[use] if p + q <= len(filt)]
        height = np.ptp(np.median(beats, axis=0)) if beats else 0.0
        for k in range(N_BINS):
            out[f"pw_{lead}_{k}"] = w[k] / height if height > 0 else np.nan
    return out


def units():
    """The saved development units (as the adopted model was fitted on), with the P-wave shape added."""
    saved = dev_units("multilead12")
    original = afib_pac_dev.unit_features

    def with_shape(sig12, fs, is_af, detector):
        row = original(sig12, fs, is_af, detector)
        return {**row, **pwave_shape(sig12, fs)} if row else row

    afib_pac_dev.unit_features = with_shape
    try:
        new = dev_units("multilead12", cache=CACHE)
    finally:
        afib_pac_dev.unit_features = original
    assert len(new) == len(saved)
    for c in ["record", "label", "source", "af"]:
        assert (new[c].values == saved[c].values).all(), c
    for c in BEAT_FEATURES:                                  # the same beats
        assert np.array_equal(new[c].values, saved[c].values, equal_nan=True), c
    d = prepare(saved.assign(**{c: new[c].values for c in WAVE}))
    d["patient"] = d.record
    d["p_unmeasured"] = (d.p_match_mean == 0).astype(float)
    return d


# ------------------------------------------------------------ the smooth logistic regression

def _difference_matrix():
    """Rows (w[k+1] - w[k]) within each lead, over the 96 waveform weights."""
    rows = []
    for j in range(len(P_LEADS)):
        for k in range(N_BINS - 1):
            r = np.zeros(len(WAVE)); r[j * N_BINS + k], r[j * N_BINS + k + 1] = -1, 1
            rows.append(r)
    return np.array(rows)


DIFF = _difference_matrix()


class SmoothLogit:
    """Balanced logistic regression: standardised features (L2) + waveform points (L2 + smoothness)."""

    def __init__(self, lam):
        self.lam = lam

    def _design(self, d, fill=None):
        x = model_inputs(d, FEATURES)
        w = d[WAVE].values.astype(float)
        if fill is None:
            fill = np.nanmean(w, axis=0)
        w = np.where(np.isnan(w), fill, w)
        return x, w, fill

    def fit(self, d):
        x, w, self.fill = self._design(d)
        self.mu, self.sd = x.mean(0), x.std(0)
        self.sd[self.sd == 0] = 1
        X = np.hstack([(x - self.mu) / self.sd, w])
        y = d.af.values.astype(float)
        cw = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
        nf = x.shape[1]
        P = self.lam * DIFF.T @ DIFF

        def loss(theta):
            b, beta = theta[0], theta[1:]
            z = b + X @ beta
            ll = np.sum(cw * (np.logaddexp(0, z) - y * z))
            pen = 0.5 * beta @ beta + 0.5 * beta[nf:] @ P @ beta[nf:]
            r = cw * (1 / (1 + np.exp(-z)) - y)
            g = np.concatenate([[r.sum()], X.T @ r + beta + np.concatenate([np.zeros(nf), P @ beta[nf:]])])
            return ll + pen, g

        theta = minimize(loss, np.zeros(1 + X.shape[1]), jac=True, method="L-BFGS-B",
                         options={"maxiter": 5000}).x
        self.b, self.beta, self.nf = theta[0], theta[1:], nf
        return self

    def predict_proba(self, d):
        x, w, _ = self._design(d, self.fill)
        z = self.b + np.hstack([(x - self.mu) / self.sd, w]) @ self.beta
        return 1 / (1 + np.exp(-z))

    @property
    def wave_weights(self):
        return self.beta[self.nf:].reshape(len(P_LEADS), N_BINS)


def choose_lambda(train):
    folds = np.array_split(np.random.default_rng(3).permutation(train.patient.unique()), 5)
    y = train.af.values
    scores = {}
    for lam in LAMBDAS:
        p = np.zeros(len(train))
        for fold in folds:
            held = train.patient.isin(fold).values
            tr, te = with_unmeasured_fix(train[~held], train[held])
            p[held] = SmoothLogit(lam).fit(tr).predict_proba(te)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        w = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
        scores[lam] = -np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    return min(scores, key=scores.get)


def fit_smooth(train):
    """As afib_fa_dev.fit: the model on all of `train`, and its cutoff from 10-fold out-of-fold probabilities."""
    lam = choose_lambda(train)
    oof = np.zeros(len(train))
    for fold in np.array_split(np.random.default_rng(0).permutation(train.patient.unique()), 10):
        held = train.patient.isin(fold).values
        tr, te = with_unmeasured_fix(train[~held], train[held])
        oof[held] = SmoothLogit(lam).fit(tr).predict_proba(te)
    tr = with_unmeasured_fix(train, train)[0]
    return SmoothLogit(lam).fit(tr), strictest_cutoff(oof, train), lam


# ------------------------------------------------------------ comparison

def main():
    d = units()
    af, pac = d.af.values, (d.label == "PAC").values
    other = ~af & ~pac
    print(f"{len(d)} units ({af.sum()} AF, {pac.sum()} PAC, {other.sum()} other); "
          f"P-wave shape unmeasurable in {d[WAVE[0]].isna().mean():.1%}")
    folds = np.array_split(np.random.default_rng(2).permutation(d.patient.unique()), 5)
    res = {k: (np.zeros(len(d)), np.zeros(len(d), bool)) for k in ["adopted", "shape"]}
    lams = []
    for fold in folds:
        held = d.patient.isin(fold).values
        model, cutoff = fit(d[~held], FEATURES, True)
        test = with_unmeasured_fix(d[~held], d[held])[1]
        res["adopted"][0][held] = model.predict_proba(model_inputs(test, FEATURES))[:, 1]
        res["adopted"][1][held] = res["adopted"][0][held] >= cutoff
        sm, cut, lam = fit_smooth(d[~held])
        lams.append(lam)
        res["shape"][0][held] = sm.predict_proba(test)
        res["shape"][1][held] = res["shape"][0][held] >= cut
        print(f"fold done: lambda {lam}", flush=True)
    g = d.patient.values
    both = af | pac
    rows = []
    for name, (prob, pred) in res.items():
        r = {"model": name, "AF caught": rate_str(pred[af]), "PAC called AF": rate_str(pred[pac]),
             "other non-AF called AF": rate_str(pred[other]), "AUC AF vs PAC": auc(prob[both], af[both]),
             "AUC all": auc(prob, af)}
        for s in CUTOFF_SOURCES:
            m = (d.source == s).values
            r[f"AF caught, {s}"] = rate_str(pred[m & af])
            r[f"PAC called AF, {s}"] = rate_str(pred[m & pac])
        rows.append(r)
    (pa, ca), (ps, cs) = res["adopted"], res["shape"]
    diffs = {
        "PAC called AF": (cs[pac].mean() - ca[pac].mean(), *boot_ci(lambda i: cs[i][pac[i]].mean() - ca[i][pac[i]].mean(), g)),
        "AF caught": (cs[af].mean() - ca[af].mean(), *boot_ci(lambda i: cs[i][af[i]].mean() - ca[i][af[i]].mean(), g)),
        "other non-AF called AF": (cs[other].mean() - ca[other].mean(),
                                   *boot_ci(lambda i: cs[i][other[i]].mean() - ca[i][other[i]].mean(), g)),
        "AUC AF vs PAC": (auc(ps[both], af[both]) - auc(pa[both], af[both]),
                          *boot_ci(lambda i: auc(ps[i][both[i]], af[i][both[i]]) - auc(pa[i][both[i]], af[i][both[i]]), g)),
    }
    out = pd.DataFrame(rows).set_index("model").T
    pd.set_option("display.width", 200)
    print(out.to_string())
    print(f"lambda chosen per outer fold: {lams}")
    for k, (v, lo, hi) in diffs.items():
        print(f"shape minus adopted, {k}: {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    ok = diffs["PAC called AF"][2] < 0 and diffs["AF caught"][0] >= -0.01 and diffs["AUC AF vs PAC"][0] >= 0
    verdict = "goes on to a registered test" if ok else "does not meet the rule"
    print(f"DECISION RULE: {verdict}")

    # The explanation: weights over time, fitted on all the development data.
    final, cut, lam = fit_smooth(d)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = -300 + BIN_S * 1000 * (np.arange(N_BINS) + 0.5)
    fig, ax = plt.subplots(2, len(P_LEADS), figsize=(15, 6), sharex=True)
    ww = final.wave_weights
    for j, lead in enumerate(P_LEADS):
        ax[0, j].bar(t, ww[j], width=8, color=np.where(ww[j] < 0, "C0", "C3"))
        ax[0, j].axhline(0, color="k", lw=.5)
        ax[0, j].set_title(f"lead {lead.replace('AV', 'aV')}")
        cols = [f"pw_{lead}_{k}" for k in range(N_BINS)]
        for lab, m, col in [("AF", af, "C3"), ("PAC", pac, "C2"), ("other non-AF", other, "C0")]:
            ax[1, j].plot(t, np.nanmean(d.loc[m, cols].values, axis=0), col, label=lab)
        ax[1, j].axhline(0, color="k", lw=.5)
        ax[1, j].set_xlabel("ms before R peak")
    ax[0, 0].set_ylabel("weight (log-odds of AF per\nunit of QRS height); < 0 = against AF")
    ax[1, 0].set_ylabel("mean waveform\n(units of QRS height)")
    ax[1, 0].legend(fontsize=8)
    fig.suptitle(f"P-wave window weights, fitted on all development data (lambda {lam}); {verdict}")
    plt.tight_layout()
    plt.savefig(write_path("figures/afib_pwave_shape_weights.png"), dpi=110)
    out.to_csv(write_path("results/afib_pwave_shape_dev.csv"))
    pd.DataFrame([{"comparison": f"shape minus adopted, {k}", "diff": v, "lo": lo, "hi": hi} for k, (v, lo, hi) in diffs.items()]
                 ).assign(verdict=verdict, lambdas=str(lams)).to_csv(write_path("results/afib_pwave_shape_dev_diffs.csv"), index=False)
    pd.DataFrame(ww, index=P_LEADS, columns=[f"{x:.0f} ms" for x in t]).to_csv(write_path("results/afib_pwave_shape_weights.csv"))
    return d, res, diffs, final


if __name__ == "__main__":
    main()
