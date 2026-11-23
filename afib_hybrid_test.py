"""Test 13: the hybrid (interpretable model screens, CNN confirms) against the interpretable model alone.

Hybrid (afib_hybrid_dev.py): AF is called only if the adopted interpretable
model's probability is at least 0.0567 and the same-data CNN's (afib_cnn.py,
models/cnn_model.keras) is at least 0.06308. Both thresholds were chosen on
the development data only. There the hybrid cut PAC false alarms from 11.8%
to 5.5% with at least 95% sensitivity per source; on sets already used
(preview, not a test) SPH 8.9% -> 3.4% and Ningbo 25.2% -> 8.4%. With these
thresholds the interpretable model is a lenient screen and the CNN makes
most decisions; every AF call still has the interpretable explanation.
Comparator: the adopted interpretable model alone at its cutoff (0.4309).

Data, never opened, chosen from annotations or labels only:
  incart   - St Petersburg INCART 12-lead Holter database (257 Hz): every
             non-overlapping 10 s crop with at least 2 annotated atrial
             premature beats (A or S), excluding the 2 patients whose
             recordings are described with atrial fibrillation (447 crops, 14
             patients; results/incart_selected.txt). Two patients supply most crops,
             so intervals are by patient and results are also shown with
             every patient weighted equally.
  code15p2 - CODE-15% part 2 (Brazil), every exam (no PAC label; AF labels
             required an irregular pulse, so sensitivity is optimistic).
Both models are scored on the recordings the interpretable model can score.

Primary, fixed before the data are opened (paired bootstrap by patient,
2000 resamples):
  1. INCART PAC crops called AF, hybrid - alone: 95% interval entirely below 0
  2. CODE-15% part 2 AF sensitivity, hybrid - alone: lower bound above -0.02
The hybrid is offered as an alternative to the interpretable model only if
both hold; the interpretable model stays the main model either way.
Secondary: CODE-15% part 2 non-AF false alarms; per-patient INCART results.
Run once.
"""

import h5py
import numpy as np
import pandas as pd
import wfdb

from af import CLINICAL_FEATURES_12, LEADS_12, model_inputs, auc
from afib_compare_dnn import strip_padding
from afib_comparison_test import boot_ci, rate_str
from afib_fa_dev import fit, with_unmeasured_fix
from afib_hybrid_dev import ADOPTED, aligned_dev
from afib_pac_dev import unit_features

ADOPTED_CUTOFF = 0.4309408380135387


def incart_features():
    rows, skipped = [], {}
    for line in open("results/incart_selected.txt").read().splitlines():
        rec, k, patient, label = line.split()
        r = wfdb.rdrecord(f"data/incart/{rec}", sampfrom=int(k) * 2570, sampto=(int(k) + 1) * 2570)
        assert [x.upper() for x in r.sig_name] == LEADS_12
        row = unit_features(r.p_signal, r.fs, False, "multilead12")
        if row is None:
            skipped[label] = skipped.get(label, 0) + 1
        else:
            rows.append({**row, "record": f"{rec}-{k}", "patient": f"incart-{patient}", "label": label})
    return pd.DataFrame(rows), skipped


def code15_features(part="exams_part2.hdf5"):
    exams = pd.read_csv("data/code15/exams.csv")
    exams = exams[exams.trace_file == part].set_index("exam_id")
    rows, skipped = [], {}
    with h5py.File(f"data/code15/{part}", "r") as f:
        ids = f["exam_id"][()]
        for i, eid in enumerate(ids):
            if eid not in exams.index:
                continue
            if i % 2000 == 0:
                print(f"code15 {i} ...", flush=True)
            label = "A" if exams.AF[eid] else "notAF"
            row = unit_features(strip_padding(f["tracings"][i]), 400, label == "A", "multilead12")
            if row is None:
                skipped[label] = skipped.get(label, 0) + 1
            else:
                rows.append({**row, "record": str(int(eid)), "patient": f"code15-{int(exams.patient_id[eid])}",
                             "label": label})
    return pd.DataFrame(rows), skipped


if __name__ == "__main__":
    d = aligned_dev()
    model, c_alone = fit(d, ADOPTED, True)
    assert abs(c_alone - ADOPTED_CUTOFF) < 1e-9, c_alone
    th = pd.read_csv("results/afib_hybrid_thresholds.csv").iloc[0]
    t_lr, t_cnn = float(th.t_lr), float(th.t_cnn)
    fills = {c: d[c].median() for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]}
    print(f"alone cutoff {c_alone:.4f}; hybrid: interpretable >= {t_lr:.4f} and CNN >= {t_cnn:.4g}")
    out, diffs = [], {}
    for name, (t, skipped) in [("incart", incart_features()), ("code15p2", code15_features())]:
        for c, v in fills.items():
            t[c] = t[c].fillna(v)
        t["p_unmeasured"] = (t.p_match_mean == 0).astype(float)
        t["prob_lr"] = model.predict_proba(model_inputs(with_unmeasured_fix(d, t)[1], ADOPTED))[:, 1]
        cnn = pd.read_csv(f"results/compare_probs_{name}.csv")
        cnn["record"] = cnn.record.astype(str)
        t = t.merge(cnn[["record", "prob_cnn"]], on="record", how="left")
        assert t.prob_cnn.notna().all()
        alone = t.prob_lr.values >= c_alone
        hybrid = (t.prob_lr.values >= t_lr) & (t.prob_cnn.values >= t_cnn)
        y, g = (t.label == "A").values, t.patient.values
        print(f"\n{name}: {len(t)} scored {t.label.value_counts().to_dict()}; unscorable: {skipped}")
        for mname, called in [("alone", alone), ("hybrid", hybrid)]:
            r = {"set": name, "model": mname, "n": len(t)}
            if y.any():
                r["AF caught"] = rate_str(called[y])
                if mname == "alone":
                    r["auc"] = auc(t.prob_lr.values, y)
            for lab in sorted(set(t.label) - {"A"}):
                k = (t.label == lab).values
                r[f"{lab} called AF"] = rate_str(called[k])
                if name == "incart":
                    per = pd.Series(called[k]).groupby(g[k]).mean()
                    r["PAC called AF, patients weighted equally"] = per.mean()
            out.append(r)
            print(r)
        if name == "incart":
            k = (t.label == "PAC").values
            d1 = hybrid[k].mean() - alone[k].mean()
            ci1 = boot_ci(lambda i: hybrid[i][k[i]].mean() - alone[i][k[i]].mean(), g)
            diffs["incart_pac_fa"] = (d1, *ci1)
            per = pd.DataFrame({"patient": g[k], "alone": alone[k], "hybrid": hybrid[k]}).groupby("patient").agg(
                crops=("alone", "size"), alone=("alone", "mean"), hybrid=("hybrid", "mean"))
            per.to_csv("results/afib_hybrid_test_incart_patients.csv")
            print(per.round(3).to_string())
        else:
            d2 = hybrid[y].mean() - alone[y].mean()
            ci2 = boot_ci(lambda i: hybrid[i][y[i]].mean() - alone[i][y[i]].mean(), g)
            diffs["code15_sens"] = (d2, *ci2)
            d3 = hybrid[~y].mean() - alone[~y].mean()
            ci3 = boot_ci(lambda i: hybrid[i][~y[i]].mean() - alone[i][~y[i]].mean(), g)
            diffs["code15_nonaf_fa"] = (d3, *ci3)
        t.assign(called_alone=alone, called_hybrid=hybrid)[["record", "patient", "label", "prob_lr", "prob_cnn",
                                                            "called_alone", "called_hybrid"]].to_csv(
            f"results/afib_hybrid_test_{name}_recordings.csv", index=False)
    pd.DataFrame(out).to_csv("results/afib_hybrid_test.csv", index=False)
    ok1 = diffs["incart_pac_fa"][2] < 0
    ok2 = diffs["code15_sens"][1] > -0.02
    for k, (v, lo, hi) in diffs.items():
        print(f"{k}: hybrid - alone {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"PRIMARY 1 (INCART PAC false alarms lower): {'yes' if ok1 else 'NO'}; "
          f"PRIMARY 2 (CODE-15% sensitivity non-inferior): {'yes' if ok2 else 'NO'}; "
          f"VERDICT: {'offer the hybrid as an alternative' if ok1 and ok2 else 'interpretable model only'}")
    pd.DataFrame([{"comparison": k, "diff": v, "lo": lo, "hi": hi} for k, (v, lo, hi) in diffs.items()]
                 ).assign(primary1=ok1, primary2=ok2).to_csv("results/afib_hybrid_test_diffs.csv", index=False)
