"""Error analysis of the Long-Term AF Database test (run after afib_test.py).

For each LTAFDB record: how many false AF alarms the 5-feature model raised,
how well our beat detector did against the database's beat labels, what
fraction of beats are ventricular / supraventricular ectopics, and which
rhythm labels appear. This doesn't change any model; it explains where the
test's false alarms came from.
"""

import collections

import numpy as np
import pandas as pd
import wfdb

from af import LTAF_RECORDS, load_ltaf_record
from ecg import BEAT_SYMS, detect_peaks_adaptive, score

SV_SYMS = set("AaSJ")

windows = pd.read_csv("results/afib_ltaf_test_windows.csv", dtype={"record": str})
false_alarms = windows[~windows.af & windows.pred_logreg5].record.value_counts()

rows = []
for rec_id in LTAF_RECORDS:
    print(f"record {rec_id} ...", flush=True)
    sig, fs, _ = load_ltaf_record(rec_id)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="ltafdb", sampto=len(sig))
    beats = [s for s in ann.symbol if s in BEAT_SYMS]
    ref = np.array([s for s, y in zip(ann.sample, ann.symbol) if y in BEAT_SYMS])
    tp, fp, fn, sens, ppv = score(ref, detect_peaks_adaptive(sig, fs), int(0.15 * fs))
    counts = collections.Counter(beats)
    rhythms = collections.Counter(x.strip() for x, y in zip(ann.aux_note, ann.symbol) if y == "+")
    rows.append({
        "record": rec_id,
        "false_alarms": int(false_alarms.get(rec_id, 0)),
        "TP": tp, "FP": fp, "FN": fn,
        "detection_sensitivity": sens, "detection_precision": ppv,
        "v_frac": counts["V"] / len(beats),
        "sv_frac": sum(counts[s] for s in SV_SYMS) / len(beats),
        "rhythm_labels": " ".join(sorted(rhythms)) or "(none)",
    })

df = pd.DataFrame(rows).sort_values("false_alarms", ascending=False)
df.to_csv("results/ltaf_errors.csv", index=False)

tot = df[["TP", "FP", "FN"]].sum()
print(f"\nBeat detection on all 84 records: sensitivity {tot.TP / (tot.TP + tot.FN):.4f}  "
      f"precision {tot.TP / (tot.TP + tot.FP):.4f}")
print(f"Top 10 records hold {df.false_alarms.head(10).sum() / df.false_alarms.sum():.0%} "
      f"of {df.false_alarms.sum()} false alarms")
print(df.head(12).round(3).to_string(index=False))
