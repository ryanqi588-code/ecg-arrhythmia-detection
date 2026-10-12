"""Test of the 12-lead AF model, built to cut PAC false alarms, on the Shandong (SPH) data.

The model tested on Chapman-Shaoxing caught 98.9% of AF but called 24% of
PAC recordings AF. Development on CPSC + PTB-XL + Chapman
(afib_pac_dev.py, 5-fold CV, strictest per-source 95% cutoff) found:
  - RR-pattern features (af.rr_pattern_features) didn't help;
  - beats from all three leads (ecg.detect_peaks_multilead) halved
    false alarms on sinus and other rhythms but not on PAC;
  - P-wave consistency averaged over all 12 leads plus the share of
    beat-free activity at 4-9 Hz in II and V1 (af.atrial_features) cut
    PAC false alarms 20.1% -> 11.6% (sinus 0.9% -> 0.4%), sensitivity
    98.2% / 94.9% / 98.5% on CPSC / PTB-XL / Chapman.

Models:
  new       - multi-lead beats, CLINICAL_FEATURES_12, trained on CPSC +
              PTB-XL + Chapman, strictest per-source 95% cutoff (primary)
  previous  - the model tested on Chapman: lead II beats,
              CLINICAL_FEATURES, trained on CPSC + PTB-XL, strictest cutoff

Data: the Shandong Provincial Hospital database (Liu et al. 2022), 12-lead
ECGs at 500 Hz labelled with AHA statements per ECG; the first 10 s of
each. Chosen from metadata.csv alone, using every recording that qualifies:
  A   - atrial fibrillation (50), no atrial flutter (51)          (675)
  PAC - atrial premature complexes (30), no AF or flutter         (539)
  N   - "Normal ECG" (1) as the only statement                    (13905)
None of it was used before. This plan is committed before the signal
archive is opened; the test is run once. Some patients have several ECGs.
"""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (CLINICAL_FEATURES, CLINICAL_FEATURES_12, clinical_features, fit_with_source_cutoffs,
                model_inputs, auc, summarise)
from afib_pac_dev import I_II_V1, dev_units, prepare, unit_features
from afib_pwave_dev import dev_units as dev_units_cpsc_ptbxl, prepare as prepare_cpsc_ptbxl

SPH_DIR = "data/sph"
FS = 500


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sph_selected():
    """(ECG_ID, patient, label) for every recording that qualifies, from metadata.csv alone."""
    meta = pd.read_csv(os.path.join(SPH_DIR, "metadata.csv"))
    out = []
    for ecg_id, codes, patient in zip(meta.ECG_ID, meta.AHA_Code.astype(str), meta.Patient_ID):
        c = {part for statement in codes.split(";") for part in statement.split("+")}
        if "50" in c and "51" not in c:
            out.append((ecg_id, patient, "A"))
        elif "30" in c and not c & {"50", "51"}:
            out.append((ecg_id, patient, "PAC"))
        elif c == {"1"}:
            out.append((ecg_id, patient, "N"))
    return out


def load_sph(ecg_id, seconds=10):
    """First `seconds` of an SPH recording, n x 12 (I, II, III, aVR, aVL, aVF, V1-V6)."""
    with h5py.File(os.path.join(SPH_DIR, "records", f"{ecg_id}.h5"), "r") as f:
        sig = f["ecg"][()]
    assert sig.shape[0] == 12, sig.shape
    return sig[:, :seconds * FS].T.astype(float)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # -------------------------------------------------------- the models
    units = prepare(dev_units("multilead12"))
    m_new, cut_new = fit_with_source_cutoffs(units, CLINICAL_FEATURES_12)
    c_new = min(cut_new.values())
    old_units = prepare_cpsc_ptbxl(dev_units_cpsc_ptbxl())
    m_old, cut_old = fit_with_source_cutoffs(old_units, CLINICAL_FEATURES)
    c_old = min(cut_old.values())
    print(f"cutoffs: new {cut_new} -> {c_new:.3f}; previous {cut_old} -> {c_old:.3f}")
    fills = {col: units[col].median() for col in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]}
    fill_old = old_units.log_f_wave.median()

    # -------------------------------------------------------------- test
    selected = sph_selected()
    print(f"selected: {pd.Series([s[2] for s in selected]).value_counts().to_dict()}")
    new_rows, old_rows, skipped = [], [], {}
    for i, (ecg_id, patient, label) in enumerate(selected):
        if i % 1000 == 0:
            print(f"test {i} ...", flush=True)
        sig12 = load_sph(ecg_id)
        new = unit_features(sig12, FS, label == "A", "multilead12")
        old = clinical_features(sig12[:, I_II_V1], FS, label == "A")
        if new is None or old is None:            # score both models on the same recordings
            skipped[label] = skipped.get(label, 0) + 1
            continue
        meta = {"record": ecg_id, "patient": patient, "label": label}
        new_rows.append({**new, **meta})
        old_rows.append({**old, **meta})
    new_df, old_df = pd.DataFrame(new_rows), pd.DataFrame(old_rows)
    for col, value in fills.items():
        new_df[col] = new_df[col].fillna(value)
    old_df["log_f_wave"] = old_df.log_f_wave.fillna(fill_old)
    is_af = (new_df.label == "A").values

    p_new = m_new.predict_proba(model_inputs(new_df, CLINICAL_FEATURES_12))[:, 1]
    p_old = m_old.predict_proba(model_inputs(old_df, CLINICAL_FEATURES))[:, 1]
    preds = {"new 12-lead model (primary)": (p_new >= c_new, p_new),
             "previous model (tested on Chapman)": (p_old >= c_old, p_old)}

    print(f"\nSPH: {len(new_df)} recordings scored ({is_af.sum()} AF, {(new_df.label == 'PAC').sum()} PAC, "
          f"{(new_df.label == 'N').sum()} normal; {new_df.patient.nunique()} patients); "
          f"skipped for too few beats: {skipped}")
    out = []
    for name, (pred, prob) in preds.items():
        s = summarise(name, is_af, pred, prob)
        lo, hi = wilson(int((pred & is_af).sum()), int(is_af.sum()))
        pac = (new_df.label == "PAC").values
        plo, phi = wilson(int(pred[pac].sum()), int(pac.sum()))
        s.update({"sens_ci_low": lo, "sens_ci_high": hi,
                  "false_alarm_rate_PAC": pred[pac].mean(), "fa_PAC_ci_low": plo, "fa_PAC_ci_high": phi,
                  "false_alarm_rate_N": pred[(new_df.label == "N").values].mean()})
        out.append(s)
    results = pd.DataFrame(out)
    results.to_csv("results/afib_sph_test.csv", index=False)
    new_df.assign(prob_new=p_new, prob_previous=p_old).to_csv("results/afib_sph_test_recordings.csv", index=False)
    print(results.set_index("classifier")[["sensitivity", "sens_ci_low", "sens_ci_high", "false_alarm_rate_PAC",
                                           "fa_PAC_ci_low", "fa_PAC_ci_high", "false_alarm_rate_N",
                                           "missed_af", "auc"]].round(3).to_string())

    plt.figure(figsize=(5.5, 5))
    for (name, (pred, prob)), color in zip(preds.items(), ["tab:blue", "tab:gray"]):
        fpr, tpr_, _ = roc_curve(is_af, prob)
        plt.plot(fpr, tpr_, color=color, label=f"{name.split(' (')[0]} (AUC {auc(prob, is_af):.3f})")
        plt.plot(pred[~is_af].mean(), pred[is_af].mean(), "o", color=color)
    plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
    plt.xlabel("False alarm rate (PAC + normal)")
    plt.ylabel("Sensitivity")
    plt.title("Shandong (SPH), first 10 s, 12-lead (circles: cutoffs)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/afib_sph_test_roc.png", dpi=120)
    plt.show()
