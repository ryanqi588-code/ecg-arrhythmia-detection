"""How much does resampling (predict.py --resample) change the model's output?

The model was developed and tested at 400-500 Hz. predict.py can resample
other rates to 500 Hz. To see what that costs, 1500 SPH recordings from
test 9 (500 Hz; 500 each of AF, PAC and sinus rhythm, seed 0) are taken to
each rate as a device recording there would be (anti-aliased polyphase
resampling), then scored by predict.py --resample (the input checks at
that rate, then resampled to 500 Hz), and compared with the call on the
original; "unscored" counts recordings the checks wrongly refused.
Rates below RESAMPLE_FROM are included to show where it breaks down.

Writes results/resampling_check.csv: per rate, how often the call is the
same, AF caught and PAC/sinus false alarms against the original's, and the
change in probability.

    python check_resampling.py
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

RATES = [150, 200, 250, 300, 360, 1000, 2000]
N_EACH, SEED = 500, 0


def _score(item):
    from afib_sph_test import FS, load_sph
    from predict import RESAMPLE_FROM, load_model, predict, resample_to
    rec, label = item
    model = load_model()
    sig = load_sph(rec)
    out = {"record": rec, "label": label}
    o = predict(sig, FS, model)
    out["result_500"], out["prob_500"] = o["result"], o.get("probability", np.nan)
    for r in RATES:
        x, _ = resample_to(sig, FS, target=r)                    # the recording as a device at r Hz would give it
        if RESAMPLE_FROM[0] <= r <= RESAMPLE_FROM[1]:
            o = predict(x, r, model, resample=True)               # exactly predict.py --resample: checks at r, then 500 Hz
        else:                                                     # below the supported range: resampled directly
            y, fs = resample_to(x, r)
            o = predict(y, fs, model)
        out[f"result_{r}"], out[f"prob_{r}"] = o["result"], o.get("probability", np.nan)
    return out


def main():
    from afib_sph_test import sph_selected
    sel = pd.DataFrame(sph_selected(), columns=["record", "x", "label"])
    rng = np.random.default_rng(SEED)
    pick = pd.concat([g.iloc[rng.choice(len(g), min(N_EACH, len(g)), replace=False)]
                      for _, g in sel[sel.label.isin(["A", "PAC", "N"])].groupby("label")])
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        t = pd.DataFrame(list(pool.map(_score, list(zip(pick.record, pick.label)), chunksize=20)))
    t = t[t.result_500 != "unscored"]
    af, pac, n = (t.label == "A").values, (t.label == "PAC").values, (t.label == "N").values
    rows = []
    for r in [500] + RATES:
        res, base = t[f"result_{r}"], t.result_500
        d = (t[f"prob_{r}"] - t.prob_500).abs()
        rows.append({"rate_hz": r, "recordings": len(t), "same_call": (res == base).mean(),
                     "calls_changed": int((res != base).sum()), "unscored": int((res == "unscored").sum()),
                     "af_caught": (res[af] == "AF").mean(), "pac_called_af": (res[pac] == "AF").mean(),
                     "sinus_called_af": (res[n] == "AF").mean(), "needs_review": (res == "needs review").mean(),
                     "prob_change_median": d.median(), "prob_change_p95": d.quantile(0.95), "prob_change_max": d.max()})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{len(t)} recordings scored at 500 Hz ({af.sum()} AF, {pac.sum()} PAC, {n.sum()} sinus)")
    print(out.round(4).to_string(index=False))
    out.to_csv(write_path("results/resampling_check.csv"), index=False)
    return out, t


if __name__ == "__main__":
    main()
