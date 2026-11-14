"""Part A of the comparison with other methods: the neural networks' probabilities.

Runs in the TensorFlow environment. For each comparison set it writes the
AF probabilities of
  ribeiro - the published deep neural network of Ribeiro et al., Nature
            Communications 11:1760 (2020), trained on 2.3 million Brazilian
            ECGs (CODE); pretrained weights from Zenodo (model.hdf5); input
            12 x 4096 samples at 400 Hz in mV; its AF output (index 4)
  cnn     - the same-data black box of afib_cnn.py
to results/compare_probs_<set>.csv. The Ribeiro network's cutoff is set
like every other model's: the strictest per-source 95%-sensitivity cutoff
on the development recordings (CPSC, PTB-XL, Chapman), which come from
outside its training data.

Sets (see afib_comparison_test.py for how each was chosen):
  dev      - the development recordings (for the Ribeiro cutoff only)
  sph      - the 15,113 recordings scored in test 9
  ningbo   - Ningbo PAC and sinus recordings (data/ningbo/SELECTED)
  code15   - CODE-15% part 0 (Ribeiro's network is skipped: CODE-15% is a
             sample of its training data)
  codetest - the 827 CODE test ECGs annotated by cardiologists
  incart, code15p2 - test 13's sets (afib_hybrid_test.py); CNN only
"""

import os
import sys

import h5py
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly

from afib_cnn import MODEL_PATH, LEADS_12, CPSC_CODES, AF_CODE, load_wfdb_12, preprocess

RIBEIRO_MODEL = "data/ribeiro/model/model.hdf5"
FS_R = 400
N_R = 4096


def to_ribeiro(sig, fs):
    """n x 12 signal in mV -> 4096 x 12 at 400 Hz, first 10 s, zero-padded on both sides."""
    sig = np.nan_to_num(np.asarray(sig, dtype=float))
    if fs != FS_R:
        g = np.gcd(FS_R, int(fs))
        sig = resample_poly(sig, FS_R // g, int(fs) // g, axis=0)
    sig = sig[:10 * FS_R]
    out = np.zeros((N_R, 12), dtype=np.float32)
    start = (N_R - len(sig)) // 2
    out[start:start + len(sig)] = sig
    return out


def strip_padding(sig):
    """CODE tracings are zero-padded on both sides; keep the recorded part."""
    nz = np.flatnonzero(np.abs(sig).sum(axis=1) > 0)
    return sig[nz[0]:nz[-1] + 1] if len(nz) else sig


def iter_set(name):
    """Yield (record, label, source, signal n x 12 in mV, fs) for a comparison set."""
    if name == "dev":
        for r in open("data/cpsc2018/SELECTED").read().split():
            dx = next(l for l in open(f"data/cpsc2018/{r}.hea") if l.startswith("# Dx")).split(":")[1].strip()
            sig, fs = load_wfdb_12(f"data/cpsc2018/{r}")
            n10 = int(10 * fs)
            for k in range(min(3, len(sig) // n10)):
                yield f"data/cpsc2018/{r}", CPSC_CODES[dx], "cpsc", sig[k * n10:(k + 1) * n10], fs
        for r in open("data/cpsc2018_extra/SELECTED").read().split():
            dx = next(l for l in open(f"data/cpsc2018_extra/{r}.hea") if l.startswith("# Dx")).split(":")[1]
            sig, fs = load_wfdb_12(f"data/cpsc2018_extra/{r}")
            n10 = int(10 * fs)
            for k in range(min(3, len(sig) // n10)):
                yield (f"data/cpsc2018_extra/{r}", "A" if AF_CODE in dx.strip().split(",") else "other", "cpsc",
                       sig[k * n10:(k + 1) * n10], fs)
        db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
        for line in open("data/ptbxl/SELECTED").read().splitlines():
            path, label = line.split()
            yield (f"ptbxl-{int(db.patient_id[path])}", label, "ptbxl",
                   *load_wfdb_12(os.path.join("data/ptbxl", path)))
        for line in open("data/chapman/SELECTED").read().splitlines():
            path, label = line.split()
            rec_id = os.path.basename(path)
            yield f"chapman-{rec_id}", label, "chapman", *load_wfdb_12(os.path.join("data/chapman", rec_id))
    elif name == "sph":
        t = pd.read_csv("results/afib_sph_test_recordings.csv", usecols=["record", "label"])
        for rec, label in zip(t.record, t.label):
            with h5py.File(f"data/sph/records/{rec}.h5", "r") as f:
                sig = f["ecg"][()]
            yield rec, label, "sph", sig[:, :5000].T, 500
    elif name == "ningbo":
        for line in open("data/ningbo/SELECTED").read().splitlines():
            path, label = line.split()
            rec = os.path.basename(path)
            yield rec, label, "ningbo", *load_wfdb_12(f"data/ningbo/rec/{rec}")
    elif name == "code15":
        exams = pd.read_csv("data/code15/exams.csv")
        exams = exams[exams.trace_file == "exams_part0.hdf5"].set_index("exam_id")
        with h5py.File("data/code15/exams_part0.hdf5", "r") as f:
            ids = f["exam_id"][()]
            for i, eid in enumerate(ids):
                if eid not in exams.index:              # the file's last row is padding
                    continue
                yield int(eid), "A" if exams.AF[eid] else "notAF", "code15", strip_padding(f["tracings"][i]), FS_R
    elif name == "incart":
        for line in open("data/incart/SELECTED").read().splitlines():
            rec, k, patient, label = line.split()
            n = 2570                                        # 10 s at 257 Hz
            r = wfdb.rdrecord(f"data/incart/{rec}", sampfrom=int(k) * n, sampto=(int(k) + 1) * n)
            assert [x.upper() for x in r.sig_name] == LEADS_12, r.sig_name
            yield f"{rec}-{k}", label, "incart", r.p_signal, r.fs
    elif name == "code15p2":
        exams = pd.read_csv("data/code15/exams.csv")
        exams = exams[exams.trace_file == "exams_part2.hdf5"].set_index("exam_id")
        with h5py.File("data/code15/exams_part2.hdf5", "r") as f:
            ids = f["exam_id"][()]
            for i, eid in enumerate(ids):
                if eid not in exams.index:
                    continue
                yield int(eid), "A" if exams.AF[eid] else "notAF", "code15", strip_padding(f["tracings"][i]), FS_R
    elif name == "codetest":
        gold = pd.read_csv("data/ribeiro/data/annotations/gold_standard.csv")
        with h5py.File("data/ribeiro/data/ecg_tracings.hdf5", "r") as f:
            tr = f["tracings"][()]
        for i in range(len(tr)):
            yield i, "A" if gold.AF[i] == 1 else "notAF", "codetest", strip_padding(tr[i]), FS_R


if __name__ == "__main__":
    import tensorflow as tf
    import tf_keras
    sets = sys.argv[1:] or ["dev", "sph", "ningbo", "code15", "codetest"]
    needs_ribeiro = any(n in ("dev", "sph", "ningbo", "codetest") for n in sets)
    ribeiro = tf_keras.models.load_model(RIBEIRO_MODEL, compile=False) if needs_ribeiro else None
    cnn = tf.keras.models.load_model(MODEL_PATH)
    for name in sets:
        rows, xr, xc = [], [], []
        use_ribeiro = name in ("dev", "sph", "ningbo", "codetest")      # test 11's sets only
        use_cnn = name != "dev"
        for i, (rec, label, source, sig, fs) in enumerate(iter_set(name)):
            if i % 1000 == 0:
                print(f"{name} {i} ...", flush=True)
            rows.append({"record": rec, "label": label, "source": source})
            if use_ribeiro:
                xr.append(to_ribeiro(sig, fs))
            if use_cnn:
                xc.append(preprocess(sig, fs))
        out = pd.DataFrame(rows)
        if use_ribeiro:
            out["prob_ribeiro"] = ribeiro.predict(np.stack(xr), batch_size=64, verbose=0)[:, 4]
        if use_cnn:
            out["prob_cnn"] = cnn.predict(np.stack(xc), batch_size=256, verbose=0)[:, 0]
        out.to_csv(f"results/compare_probs_{name}.csv", index=False)
        print(f"{name}: {len(out)} recordings written", flush=True)
