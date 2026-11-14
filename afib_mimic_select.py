"""Choose test 14's MIMIC-IV-ECG recordings from the machine reports only (no signals opened).

MIMIC-IV-ECG v1.0 (PhysioNet, open access): about 800,000 10 s 12-lead ECGs
(500 Hz) from about 160,000 patients at Beth Israel Deaconess Medical Center,
Boston, each with the ECG machine's report statements (machine_measurements.csv).
The labels are the machine's, not a cardiologist's.

  PAC - a PAC statement ("pac", "pacs", "pac(s)", "atrial premature",
        "premature atrial", "supraventricular extrasystole", "apc"), a sinus
        rhythm statement, and none of: fibrillation, flutter, pacing or
        pacemaker, ectopic atrial or wandering or junctional or multifocal
        rhythm, supraventricular tachycardia, ventricular premature beats
        (PVC, ventricular premature, premature ventricular, ventricular
        extrasystole, ventricular bigeminy/trigeminy), irregular or
        undetermined rhythm, artifact or noise.
  A   - an "atrial fibrillation" statement and none of: possible, probable,
        consider, flutter, pacing or pacemaker, sinus, artifact or noise,
        undetermined.
One recording per patient, drawn at random (seed 0): 2000 PAC patients, then
1000 AF patients from the remaining patients (the groups share no patient).
Writes results/mimic_selected.txt: path, subject, label.
"""

import re

import numpy as np
import pandas as pd
import wfdb

from af import LEADS_12

MIMIC_DIR = "data/mimic"
URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
N_PAC, N_AF = 2000, 1000

PAC = re.compile(r"\bpacs?\b|\bpac\(s\)|atrial premature|premature atrial|supraventricular extrasystole|\bapcs?\b")
PAC_EXCLUDE = re.compile(r"fibrillation|flutter|pacing|pacemaker|paced|pacer|ectopic atrial|wandering|junctional|"
                         r"multifocal|supraventricular tachycardia|\bsvt\b|\bpvcs?\b|pvc\(s\)|ventricular premature|"
                         r"premature ventricular|ventricular extrasystole|ventricular bigeminy|ventricular trigeminy|"
                         r"irregular|undetermined|artifact|noise")
AF = re.compile(r"atrial fibrillation")
AF_EXCLUDE = re.compile(r"possible|probable|consider|flutter|pacing|pacemaker|paced|pacer|sinus|artifact|noise|"
                        r"undetermined")


def report_text(m):
    cols = [f"report_{i}" for i in range(18)]
    return m[cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()


def load_mimic(path):
    """A MIMIC-IV-ECG record as n x 12 in the standard lead order (MIMIC stores aVF before aVL)."""
    rec = wfdb.rdrecord(path)
    names = [n.upper() for n in rec.sig_name]
    return rec.p_signal[:, [names.index(n) for n in LEADS_12]], rec.fs


def read_selected():
    return [l.split() for l in open("results/mimic_selected.txt").read().splitlines()]


if __name__ == "__main__":
    m = pd.read_csv(f"{MIMIC_DIR}/machine_measurements.csv", low_memory=False)
    paths = pd.read_csv(f"{MIMIC_DIR}/record_list.csv").set_index("study_id").path
    t = report_text(m)
    is_pac = t.str.contains(PAC) & t.str.contains("sinus") & ~t.str.contains(PAC_EXCLUDE)
    is_af = t.str.contains(AF) & ~t.str.contains(AF_EXCLUDE)
    rng = np.random.default_rng(0)

    def one_per_patient(rows, n, exclude=()):
        rows = rows[~rows.subject_id.isin(exclude)]
        pick = rows.groupby("subject_id").sample(1, random_state=0)
        print(f"{len(rows)} recordings from {len(pick)} patients")
        return pick.iloc[rng.permutation(len(pick))[:n]]

    pac = one_per_patient(m[is_pac], N_PAC)
    af = one_per_patient(m[is_af], N_AF, set(pac.subject_id))
    out = [(paths[s], subj, lab) for df, lab in [(pac, "PAC"), (af, "A")] for s, subj in zip(df.study_id, df.subject_id)]
    with open("results/mimic_selected.txt", "w") as f:
        f.writelines(f"{p} {s} {lab}\n" for p, s, lab in out)
    print(f"{len(pac)} PAC, {len(af)} AF written to results/mimic_selected.txt")
