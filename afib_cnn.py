"""A black-box comparator: a 1D convolutional network trained on the same data as the AF model.

The interpretable 12-lead AF model is logistic regression on 10 named
features. To show what a black box gains, this network is trained on the
same development recordings (10 s crops of CPSC 2018 and CPSC-extra,
PTB-XL, Chapman-Shaoxing; all leads), with the same balanced class
weights and the same cutoff rule: the strictest per-source 95%-sensitivity
cutoff, set on 5-fold out-of-fold probabilities grouped by recording or
patient (af.fit_with_source_cutoffs does the same for the logistic model).

Architecture and training were fixed before any training run and were not
tuned: 12 leads, band-passed 0.5-40 Hz, resampled to 250 Hz (2500 samples),
each lead scaled by its own standard deviation; a stem convolution and four
residual blocks (32, 64, 96, 128 filters, kernel 9, each downsampling by
4), global average pooling, dropout 0.3, one sigmoid output; Adam (learning
rate 1e-3), batch 64, 25 epochs, no early stopping.

Runs in a TensorFlow environment (see README): it imports neither af.py nor
scikit-learn, so its loaders mirror af.load_12lead_all.
    python afib_cnn.py   # builds data/cnn_dev.npz, cross-validates, trains the final network
"""

import os

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly

FS_OUT = 250
N_OUT = 10 * FS_OUT
LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEV_CACHE = "data/cnn_dev.npz"
MODEL_PATH = "models/cnn_model.keras"
CUTOFF_PATH = "results/afib_cnn_dev.csv"
CPSC_CODES = {"164889003": "A", "284470004": "PAC", "426783006": "N"}
AF_CODE = "164889003"


def preprocess(sig, fs):
    """n x 12 signal (any length >= ~7 s) -> 2500 x 12 at 250 Hz, each lead scaled to unit SD."""
    sig = np.nan_to_num(np.asarray(sig, dtype=float))
    b, a = butter(3, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
    sig = filtfilt(b, a, sig, axis=0)
    up, down = FS_OUT, int(fs)
    g = np.gcd(up, down)
    sig = resample_poly(sig, up // g, down // g, axis=0)
    out = np.zeros((N_OUT, 12), dtype=np.float32)
    n = min(N_OUT, len(sig))
    start = (N_OUT - n) // 2                      # centre shorter recordings, zero-padded
    out[start:start + n] = sig[:n]
    sd = out.std(axis=0)
    return np.clip(out / np.where(sd > 0, sd, 1), -10, 10).astype(np.float32)


def load_wfdb_12(path):
    rec = wfdb.rdrecord(path)
    assert [n.upper() for n in rec.sig_name] == LEADS_12, rec.sig_name
    return rec.p_signal, rec.fs


def dev_data():
    """Signals, labels, sources and groups of the development recordings (cached)."""
    if os.path.exists(DEV_CACHE):
        z = np.load(DEV_CACHE, allow_pickle=True)
        return z["x"], pd.DataFrame(z["meta"].tolist())
    x, meta = [], []
    cpsc = [r for r in open("data/cpsc2018/SELECTED").read().split()]
    paths = []
    for r in cpsc:
        dx = next(l for l in open(f"data/cpsc2018/{r}.hea") if l.startswith("# Dx")).split(":")[1].strip()
        paths.append((f"data/cpsc2018/{r}", CPSC_CODES[dx]))
    for r in open("data/cpsc2018_extra/SELECTED").read().split():
        dx = next(l for l in open(f"data/cpsc2018_extra/{r}.hea") if l.startswith("# Dx")).split(":")[1]
        paths.append((f"data/cpsc2018_extra/{r}", "A" if AF_CODE in dx.strip().split(",") else "other"))
    for i, (path, label) in enumerate(paths):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_wfdb_12(path)
        n10 = int(10 * fs)
        for k in range(min(3, len(sig) // n10)):
            x.append(preprocess(sig[k * n10:(k + 1) * n10], fs))
            meta.append({"source": "cpsc", "record": path, "label": label})
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    for line in open("data/ptbxl/SELECTED").read().splitlines():
        path, label = line.split()
        x.append(preprocess(*load_wfdb_12(os.path.join("data/ptbxl", path))))
        meta.append({"source": "ptbxl", "record": f"ptbxl-{int(db.patient_id[path])}", "label": label})
    for line in open("data/chapman/SELECTED").read().splitlines():
        path, label = line.split()
        rec_id = os.path.basename(path)
        x.append(preprocess(*load_wfdb_12(os.path.join("data/chapman", rec_id))))
        meta.append({"source": "chapman", "record": f"chapman-{rec_id}", "label": label})
    x = np.stack(x)
    np.savez(DEV_CACHE, x=x, meta=np.array(meta, dtype=object))
    return x, pd.DataFrame(meta)


def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(0)
    inp = layers.Input((N_OUT, 12))
    h = layers.Conv1D(32, 15, padding="same", use_bias=False)(inp)
    h = layers.ReLU()(layers.BatchNormalization()(h))
    for filters in [32, 64, 96, 128]:
        skip = layers.Conv1D(filters, 1, strides=4, padding="same", use_bias=False)(h)
        y = layers.Conv1D(filters, 9, padding="same", use_bias=False)(h)
        y = layers.ReLU()(layers.BatchNormalization()(y))
        y = layers.Conv1D(filters, 9, strides=4, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        h = layers.ReLU()(layers.Add()([y, skip]))
    h = layers.Dropout(0.3)(layers.GlobalAveragePooling1D()(h))
    out = layers.Dense(1, activation="sigmoid")(h)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


def fit(x, y):
    w = {0: 0.5 * len(y) / max(1, (y == 0).sum()), 1: 0.5 * len(y) / max(1, (y == 1).sum())}
    model = build_model()
    model.fit(x, y, batch_size=64, epochs=25, class_weight=w, verbose=0, shuffle=True)
    return model


def cutoff_for_sensitivity(prob, af, records, target=0.95):
    """As af.cutoff_for_sensitivity: records' AF units weighted equally."""
    p, r = prob[af], np.asarray(records)[af]
    counts = pd.Series(r).map(pd.Series(r).value_counts()).values
    w = 1 / counts
    order = np.argsort(p)
    cum = np.cumsum(w[order]) / w.sum()
    return p[order][np.searchsorted(cum, 1 - target)]


if __name__ == "__main__":
    x, meta = dev_data()
    y = (meta.label == "A").values.astype(int)
    print(f"{len(x)} development units, {y.sum()} AF", flush=True)
    folds = np.array_split(np.random.default_rng(2).permutation(meta.record.unique()), 5)
    oof = np.zeros(len(x))
    for i, fold in enumerate(folds):
        held = meta.record.isin(fold).values
        m = fit(x[~held], y[~held])
        oof[held] = m.predict(x[held], batch_size=256, verbose=0)[:, 0]
        print(f"fold {i}: held-out AUC-ready, {held.sum()} units", flush=True)
    cutoffs = {s: cutoff_for_sensitivity(oof[(meta.source == s).values], y[(meta.source == s).values] == 1,
                                         meta.record.values[(meta.source == s).values])
               for s in ["cpsc", "ptbxl", "chapman"]}
    cutoff = min(cutoffs.values())
    pred = oof >= cutoff
    is_af = y == 1
    from scipy.stats import mannwhitneyu
    auc = mannwhitneyu(oof[is_af], oof[~is_af]).statistic / (is_af.sum() * (~is_af).sum())
    rows = {"cutoff": cutoff, **{f"cutoff_{k}": v for k, v in cutoffs.items()}, "auc": auc}
    for s in ["cpsc", "ptbxl", "chapman"]:
        rows[f"sens_{s}"] = pred[(meta.source == s).values & is_af].mean()
    for lab in ["PAC", "N", "other"]:
        rows[f"fa_{lab}"] = pred[(meta.label == lab).values].mean()
    pd.DataFrame([rows]).to_csv(CUTOFF_PATH, index=False)
    print(pd.DataFrame([rows]).round(3).to_string(index=False), flush=True)
    meta.assign(prob_cnn=oof).to_csv("results/afib_cnn_dev_oof.csv", index=False)
    final = fit(x, y)
    final.save(MODEL_PATH)
    print("saved", MODEL_PATH)
