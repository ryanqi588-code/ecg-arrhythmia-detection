"""Score one 12-lead ECG with the adopted AF model: AF, not AF, or unscored, with each feature's contribution.

    python predict.py path/to/record            # WFDB record (.hea/.dat; leads matched by name)
    python predict.py ecg.csv --fs 500          # CSV, one column per lead (header with lead names,
                                                #   or no header and the standard order I..V6)
    python predict.py path/to/record --json

The model is read from results/final_af_model.csv (written by
export_final_model.py), so no training data is needed. Features come from
af.clinical_features, as in the tests: beats found on leads I, II and V1,
rhythm features after removing ventricular ectopics, and P-wave and
fibrillatory-wave features from all 12 leads. Only the first 10 s are
used; the model was developed and tested on 10 s recordings (400-500 Hz).

Each feature's contribution is its coefficient times its standardized
input, coef_per_sd * (input - mean) / sd, in log-odds. They add up, with
the intercept (the log-odds of a recording at the training average), to
the score z, and the probability of AF is 1 / (1 + e^-z). Positive
contributions push towards AF. The probability is not a calibrated risk.

"Unscored" means the recording has too few beats to measure the rhythm
(fewer than 7 detected) or missing samples. The model is a research tool,
not a medical device.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from af import LEADS_12, clinical_features

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_af_model.csv")
I_II_V1 = [0, 1, 6]
SECONDS = 10


def load_model(path=MODEL):
    t = pd.read_csv(path).set_index("feature")
    return t.drop(["intercept", "cutoff"]), float(t.loc["intercept", "coef_per_sd"]), float(t.loc["cutoff", "value"])


def read_ecg(path, fs=None):
    """(n x 12 signal in the standard lead order, sampling rate)."""
    if path.lower().endswith((".csv", ".txt")):
        if fs is None:
            sys.exit("--fs is needed for a CSV file")
        d = pd.read_csv(path)
        names = [str(c).strip().upper() for c in d.columns]
        if set(LEADS_12) <= set(names):
            d.columns = names
            return d[LEADS_12].to_numpy(float), fs
        d = pd.read_csv(path, header=None)
        if d.shape[1] != 12:
            sys.exit(f"expected 12 columns (I, II, III, aVR, aVL, aVF, V1-V6), found {d.shape[1]}")
        return d.to_numpy(float), fs
    import wfdb
    rec = wfdb.rdrecord(os.path.splitext(path)[0] if path.endswith((".hea", ".dat")) else path)
    names = [n.strip().upper() for n in rec.sig_name]
    missing = [n for n in LEADS_12 if n not in names]
    if missing:
        sys.exit(f"all 12 leads are needed; missing {missing} (found {rec.sig_name})")
    return rec.p_signal[:, [names.index(n) for n in LEADS_12]], rec.fs


def score_features(row, model=None):
    """Probability, call and contributions from a clinical_features row (dict)."""
    feats, intercept, cutoff = model or load_model()
    p_unmeasured = float(row["p_match_mean"] == 0)
    contributions = []
    for name, f in feats.iterrows():
        value = p_unmeasured if name == "p_unmeasured" else row.get(name, np.nan)
        filled = False
        fill_when = str(f.fill_when) if pd.notna(f.fill_when) else ""
        if pd.notna(f.fill) and (("p_unmeasured" in fill_when and p_unmeasured) or not np.isfinite(value)):
            value, filled = float(f.fill), True
        x = np.log(max(value, 1e-3)) if f.input.startswith("log(") else value
        z = (x - f["mean"]) / f.sd
        contributions.append({"feature": name, "value": float(value), "filled": filled,
                              "standardized": float(z), "contribution": float(f.coef_per_sd * z)})
    z = intercept + sum(c["contribution"] for c in contributions)
    prob = 1 / (1 + np.exp(-z))
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    return {"result": "AF" if prob >= cutoff else "not AF", "probability": float(prob), "cutoff": cutoff,
            "score": float(z), "intercept": intercept, "contributions": contributions}


def predict(sig, fs, model=None):
    sig = np.asarray(sig, float)[: int(round(SECONDS * fs))]
    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError("expected an n x 12 signal")
    if not np.isfinite(sig).all():
        return {"result": "unscored", "reason": "missing samples"}
    row = clinical_features(sig[:, I_II_V1], fs, False, "multilead", sig)
    if row is None:
        return {"result": "unscored", "reason": "too few beats to measure the rhythm"}
    return score_features(row, model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("record", help="WFDB record path (without extension) or a CSV file")
    ap.add_argument("--fs", type=float, help="sampling rate in Hz (CSV only)")
    ap.add_argument("--json", action="store_true", help="print JSON")
    a = ap.parse_args()
    sig, fs = read_ecg(a.record, a.fs)
    out = predict(sig, fs)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    if out["result"] == "unscored":
        print(f"unscored: {out['reason']}")
        return
    print(f"{out['result']}  (probability {out['probability']:.3f}; AF if >= {out['cutoff']:.4f})")
    print(f"\n{'feature':<16}{'value':>10}{'standardized':>14}{'contribution':>14}")
    for c in out["contributions"]:
        print(f"{c['feature']:<16}{c['value']:>10.4g}{c['standardized']:>14.2f}{c['contribution']:>+14.3f}"
              f"{'  (filled)' if c['filled'] else ''}")
    print(f"{'intercept':<40}{out['intercept']:>+14.3f}")
    print(f"{'score (log-odds of AF)':<40}{out['score']:>+14.3f}")


if __name__ == "__main__":
    main()
