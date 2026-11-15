"""Better MIMIC labels from open data: is the machine's AF call confirmed by the patient's other ECGs?

Why: on MIMIC-IV-ECG the AF labels are the ECG machine's statements, not a
cardiologist's (the cardiologist reports are not openly released). The
error analysis found that 38% of the missed AF strips are almost perfectly
regular, which suggests some machine AF labels are wrong. Real AF usually
shows on the same patient's other ECGs taken within hours. A single machine
AF call between sinus-rhythm ECGs is less certain (it may be paroxysmal AF
or a machine error).

Relabelling (machine reports only; no signal or model output is used):
for each AF recording of tests 14, 15 and 16 (2947 scored, one per patient,
patients disjoint across tests), take the same patient's other ECGs within
24 hours (ecg_time in machine_measurements.csv) and group the recording as
  confirmed      - at least one other ECG within 24 h states "atrial
                   fibrillation" without possible / probable / consider
  sinus nearby   - not confirmed, and at least one other ECG within 24 h
                   states sinus rhythm without fibrillation or flutter
  no comparison  - neither (no ECG within 24 h, or only other rhythms)

Primary (adopted model, cutoff 0.4309, calls as saved by tests 14-16):
  AF sensitivity, confirmed - sinus nearby, bootstrap by patient (2000
  resamples). If the 95% interval is entirely above 0, label errors explain
  part of the missed AF, and the sensitivity on confirmed AF is reported in
  the README as the best open-data estimate of the model's sensitivity on
  real AF at this hospital, alongside (not replacing) the tests' numbers.
Secondary: sensitivity in each group with Wilson intervals; the same on
AF under 80 a minute (machine heart rate); the CNN (tests 14-15, its dev
cutoff 0.06515) by group; test 16's new model against the adopted model on
confirmed AF; share of near-regular strips (RR CV < 0.05, tests 14-15) by
group; PAC recordings called AF, split by whether the patient has an AF ECG
within 24 h; the primary repeated with a 6 h window. Run once.
"""

import numpy as np
import pandas as pd

from afib_comparison_test import boot_ci, rate_str
from afib_mimic_select import MIMIC_DIR, report_text

WINDOW_H = 24
CNN_CUTOFF = 0.06515
AF_STATED = r"atrial fibrillation"
UNSURE = r"possible|probable|consider"
SINUS = r"sinus"
NOT_SINUS = r"fibrillation|flutter"


def load_reports():
    m = pd.read_csv(f"{MIMIC_DIR}/machine_measurements.csv", low_memory=False)
    text = report_text(m)
    return pd.DataFrame({"subject": m.subject_id.values, "study": m.study_id.values,
                         "time": pd.to_datetime(m.ecg_time).values,
                         "af": (text.str.contains(AF_STATED) & ~text.str.contains(UNSURE)).values,
                         "sinus": (text.str.contains(SINUS) & ~text.str.contains(NOT_SINUS)).values,
                         "hr": (60000 / m.rr_interval.where(m.rr_interval > 0)).values})


def neighbour_groups(records, reports, window_h=WINDOW_H):
    """records: DataFrame with 'record' (study id). Returns group, af_near, sinus_near per record."""
    r = reports.set_index("study")
    tgt = r.loc[records.record.astype(int).values, ["subject", "time"]].reset_index()
    pairs = tgt.merge(reports[reports.subject.isin(tgt.subject)], on="subject", suffixes=("", "_n"))
    pairs = pairs[(pairs.study_n != pairs.study)
                  & ((pairs.time_n - pairs.time).abs() <= pd.Timedelta(hours=window_h))]
    near = pairs.groupby("study")[["af", "sinus"]].any()
    out = pd.DataFrame({"record": tgt.study.values})
    out["af_near"] = out.record.map(near.af).fillna(False).astype(bool).values
    out["sinus_near"] = out.record.map(near.sinus).fillna(False).astype(bool).values
    out["group"] = np.where(out.af_near, "confirmed", np.where(out.sinus_near, "sinus nearby", "no comparison"))
    out["hr"] = r.loc[out.record.values, "hr"].values
    return out


def load_tests():
    t14 = pd.read_csv("results/afib_mimic_test_recordings.csv").assign(test=14)
    t15 = pd.read_csv("results/afib_mimic15_test_recordings.csv").assign(test=15)
    t16 = pd.read_csv("results/afib_slowaf_test_recordings.csv").assign(test=16)
    t16 = t16.rename(columns={"called_adopted": "called_alone"})
    cols = ["record", "patient", "label", "test", "called_alone"]
    d = pd.concat([t14[cols + ["prob_cnn"]], t15[cols + ["prob_cnn"]],
                   t16[cols + ["called_new"]]], ignore_index=True)
    assert not d.patient.duplicated().any()
    return d


def diff_ci(called, a, b, g):
    v = called[a].mean() - called[b].mean()
    return v, *boot_ci(lambda i: called[i][a[i]].mean() - called[i][b[i]].mean(), g)


if __name__ == "__main__":
    reports = load_reports()
    d = load_tests()
    d = d.merge(neighbour_groups(d, reports), on="record")
    called, g = d.called_alone.values.astype(bool), d.patient.values
    af = (d.label == "A").values
    conf, sinus = af & (d.group == "confirmed").values, af & (d.group == "sinus nearby").values
    print(f"AF recordings by group: {d[af].group.value_counts().to_dict()}")

    v, lo, hi = diff_ci(called, conf, sinus, g)
    ok = lo > 0
    print(f"PRIMARY: sensitivity confirmed - sinus nearby {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}): "
          f"{'label errors explain part of the misses' if ok else 'not shown'}")

    rows, slow = [], d.hr.values < 80
    for grp in ["confirmed", "sinus nearby", "no comparison", "all"]:
        m = af & ((d.group == grp).values if grp != "all" else True)
        cnn = m & d.prob_cnn.notna().values
        new = m & d.called_new.notna().values
        r = {"group": grp, "n": int(m.sum()), "adopted": rate_str(called[m]),
             "adopted, under 80/min": rate_str(called[m & slow]),
             "CNN (tests 14-15)": rate_str(d.prob_cnn.values[cnn] >= CNN_CUTOFF),
             "adopted (test 16)": rate_str(called[new]),
             "new model (test 16)": rate_str(d.called_new.values[new].astype(bool))}
        rows.append(r)
        print(r)
    new_conf = conf & d.called_new.notna().values
    nv = d.called_new.fillna(False).values.astype(bool)
    dv = (nv[new_conf].mean() - called[new_conf].mean(),
          *boot_ci(lambda i: nv[i][new_conf[i]].mean() - called[i][new_conf[i]].mean(), g))
    print(f"test 16 on confirmed AF, new - adopted: {dv[0]:+.4f} (95% CI {dv[1]:+.4f} to {dv[2]:+.4f})")

    err = pd.read_csv("results/afib_mimic_errors_af.csv")[["record", "cv"]]
    e = d[af].merge(err, on="record")
    reg = e.assign(regular=e.cv < 0.05).groupby("group").regular.agg(["mean", "sum", "count"])
    print("near-regular share (tests 14-15):\n", reg)

    pac = (d.label == "PAC").values
    pac_rows = []
    for name, m in [("PAC, AF ECG within 24 h", pac & d.af_near.values),
                    ("PAC, no AF ECG within 24 h", pac & ~d.af_near.values)]:
        pac_rows.append({"group": name, "n": int(m.sum()), "called AF": rate_str(called[m])})
        print(pac_rows[-1])

    d6 = d.drop(columns=["group", "af_near", "sinus_near", "hr"]).merge(
        neighbour_groups(d, reports, window_h=6), on="record")
    c6 = af & (d6.group == "confirmed").values
    s6 = af & (d6.group == "sinus nearby").values
    v6 = diff_ci(d6.called_alone.values.astype(bool), c6, s6, d6.patient.values)
    print(f"6 h window: confirmed {int(c6.sum())}, sinus nearby {int(s6.sum())}; "
          f"difference {v6[0]:+.4f} (95% CI {v6[1]:+.4f} to {v6[2]:+.4f})")

    pd.DataFrame(rows).to_csv("results/afib_mimic_labels.csv", index=False)
    pd.DataFrame(pac_rows).to_csv("results/afib_mimic_labels_pac.csv", index=False)
    reg.to_csv("results/afib_mimic_labels_regular.csv")
    pd.DataFrame([{"comparison": "confirmed - sinus nearby (24 h)", "diff": v, "lo": lo, "hi": hi, "primary": ok},
                  {"comparison": "confirmed - sinus nearby (6 h)", "diff": v6[0], "lo": v6[1], "hi": v6[2]},
                  {"comparison": "test 16 confirmed AF, new - adopted", "diff": dv[0], "lo": dv[1], "hi": dv[2]}]
                 ).to_csv("results/afib_mimic_labels_diffs.csv", index=False)
    d[["record", "patient", "label", "test", "group", "af_near", "sinus_near", "hr", "called_alone"]].to_csv(
        "results/afib_mimic_labels_recordings.csv", index=False)
