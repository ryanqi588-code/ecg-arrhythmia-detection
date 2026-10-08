# ECG arrhythmia detection

R-peak (heartbeat) detection and atrial fibrillation (AF) screening on
PhysioNet's MIT-BIH databases. Beat detection is written from scratch
with NumPy and SciPy and benchmarked against an established detector;
AF screening uses RR-interval features and logistic regression.

| | Result |
|---|---|
| Beat detection, 36 held-out MIT-BIH records | sensitivity **99.57%**, precision **99.57%** |
| `wfdb`'s XQRS on the same records | sensitivity 98.82%, precision 99.96% |
| AF detection, 5-feature logistic regression, 23 AFDB records | ROC AUC **0.951**; sensitivity 86.6%, specificity 87.8% |
| AF detection, original two-threshold rule | sensitivity 95.6%, specificity 76.3% |

## 1. Beat detection

[`ecg.py`](ecg.py) contains the detector:

1. **Band-pass filter** (0.5–40 Hz Butterworth, zero-phase) removes
   baseline wander and high-frequency noise.
2. **Adaptive threshold.** A peak must exceed 40% of the largest
   |signal| value within a sliding 2 s window, with a floor so the
   threshold can't collapse onto noise during pauses. Using |signal|
   catches beats whose QRS complex points downwards, which is common for
   premature ventricular beats.
3. **T-wave rejection** (from Pan & Tompkins, 1985). A candidate within
   360 ms of the previous beat whose steepest slope is under half that
   beat's is a T wave, not a new beat.

A detection counts as correct if it is within 150 ms of an annotated
beat. Ventricular flutter episodes contain no separate beats, so they
are left out of scoring, as the ANSI/AAMI EC57 standard specifies.

### Why a fixed threshold isn't enough

The first version used a single threshold at half the record's maximum.
One tall beat or artefact then sets the bar too high for everything
else. On record 215 it detected only 5% of beats:

![Record 215: fixed vs adaptive threshold](figures/record215_thresholds.png)

### Results

The detector was developed while looking at 10 records (100, 101, 103,
105, 108, 200, 203, 207, 210, 215), so it is also scored on the other
36 records, which played no part in tuning. Records 102 and 104 have no
MLII lead and are skipped. Each record is scored on its first 5 minutes.

| Detector | Dev sens. | Dev prec. | Held-out sens. | Held-out prec. | Held-out missed / false |
|---|---|---|---|---|---|
| Fixed threshold | 61.13% | 99.10% | 91.90% | 97.72% | 1092 / 289 |
| **Adaptive (ours)** | 98.19% | 99.26% | **99.57%** | 99.57% | **58** / 58 |
| XQRS (`wfdb`) | 97.06% | 96.72% | 98.82% | **99.96%** | 159 / **6** |

On held-out data the two detectors make different trade-offs. Ours
misses about a third as many beats, and XQRS produces about a tenth as
many false detections. Ours is also about 25× faster (0.3 s against
7.8 s for all 36 records), though speed wasn't a design goal.

Record 203, a very noisy record with ventricular beats of many shapes,
is the weakest for our detector (91.6% sensitivity).

![Record 100: detected vs annotated beats](figures/record100_beats.png)

## 2. Atrial fibrillation check

In AF the heartbeat becomes irregularly irregular.
[`afib.py`](afib.py) runs the beat detector on the
[MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
and measures how irregular the RR intervals (times between beats) are in
each 60 s window:

| Feature | Definition | AUC alone |
|---|---|---|
| **CV** | standard deviation of RR / mean RR | 0.895 |
| **nRMSSD** | root-mean-square of successive RR differences / mean RR | 0.896 |
| **Shannon entropy** | entropy of a 16-bin histogram of the window's RR intervals, scaled to 0–1 | 0.721 |
| **TPR** (turning points ratio) | fraction of RR intervals that are a local peak or trough (≈ 2/3 for a random series) | 0.843 |
| **pNN50** | fraction of successive RR differences over 50 ms | 0.957 |

A window is labelled AF if at least half of it is annotated `(AFIB`;
atrial flutter and junctional rhythm count as non-AF. That gives 2759
windows (610 AF, 2149 non-AF) from the first 2 hours of 23 records.

Every classifier is evaluated by **leave-one-record-out
cross-validation**: each record is classified by a model fitted on the
other 22 records only, so no record is ever scored by a model that saw it.

### Classifiers

1. **Threshold rule** (the first version): AF when both CV and nRMSSD
   exceed thresholds, grid-searched for the best balanced accuracy.
2. **Logistic regression** on CV + nRMSSD, and on all 5 features. CV and
   nRMSSD are log-transformed, since they span two orders of magnitude.
   All inputs are standardized, and classes are weighted equally, matching
   the threshold rule's objective. Windows with probability ≥ 0.5 are
   called AF.
3. **Logistic regression on pNN50 alone.** I added this *after* seeing
   that pNN50 alone scores as well as the 5-feature model, so treat it as
   a follow-up observation, not a planned comparison.

### Results

| Classifier | Sensitivity | Specificity | Precision | False alarms | Missed AF | ROC AUC |
|---|---|---|---|---|---|---|
| Threshold rule (CV, nRMSSD) | 95.6% | 76.3% | 53.4% | 509 | 27 | – |
| Logistic (CV, nRMSSD) | 88.5% | 78.0% | 53.3% | 473 | 70 | 0.889 |
| **Logistic (5 features)** | 86.6% | **87.8%** | 66.8% | **262** | 82 | **0.951** |
| Logistic (pNN50 only) | 97.2% | 87.8% | 69.4% | 262 | 17 | 0.953 |

Sensitivity and specificity depend on where the 0.5 cutoff happens to
fall, so the ROC curves are the fairer comparison:

![ROC curves](figures/afib_roc.png)

What this shows:

- **The new features help.** AUC rises from 0.889 to 0.951, and at the
  0.5 cutoff false alarms nearly halve (509 → 262). The cost at that
  particular cutoff is more missed AF (27 → 82).
- **pNN50 carries almost all of it.** On its own it matches the
  5-feature model at every operating point. A plausible reason: one
  ectopic beat creates two large RR jumps. Those inflate CV and nRMSSD,
  which measure how *big* the changes are, but move pNN50 only slightly,
  because it *counts* how many changes exceed 50 ms.
- **The threshold rule beats logistic regression on the same two
  features.** Requiring *both* features to be high draws a rectangular
  boundary, and that suits this data better than logistic regression's
  straight line.

### Reading the coefficients

Coefficients of the 5-feature model fitted on all windows, in log-odds
of AF per standard deviation of each feature, with the range across the
23 cross-validation folds:

| Feature | Coefficient | Fold range |
|---|---|---|
| log CV | +1.89 | +1.28 to +2.68 |
| log nRMSSD | −1.28 | −1.70 to −0.58 |
| entropy | +1.07 | +0.81 to +2.65 |
| TPR | −0.30 | −0.65 to −0.07 |
| pNN50 | +2.86 | +2.14 to +3.25 |

Every sign is stable across folds, but two of them shouldn't be read at
face value. **log CV and log nRMSSD are 97% correlated**, so the model
gives one a positive weight and the other a negative one; together they
add up to "overall variability". The negative nRMSSD weight doesn't mean
nRMSSD argues against AF; on its own it rises with AF (AUC 0.896). TPR's
small negative weight is similar: it is 66% correlated with pNN50 and
only adjusts what pNN50 already says. pNN50 has the largest, most
stable weight, which agrees with its single-feature AUC.

### Where it fails

Most false alarms come from a few records (08219, 05261, 04015) with
**frequent ectopic (extra) beats**. Ectopic beats make the RR series
irregular without the rhythm being AF. In the first 48 minutes of record
08378 below, the RR intervals split into two bands, short and long; that
is ectopy, not AF, but it scores as irregular:

![Record 08378 RR intervals](figures/afib_tachogram.png)

For the threshold rule this is a limit of the method, not of the beat
detector. Using PhysioNet's reference beat positions in place of ours
gives similar results: 79.4% specificity and 91.1% sensitivity. The
5-feature model handles ectopy better. Accuracy on record 08378 rises
from 66% to 91%, and on 05261 from 48% to 73%. Record 08219 is still
poor, at 40%. Separating AF from heavy ectopy reliably would probably
need information beyond RR intervals, such as checking for missing
P waves.

![CV vs nRMSSD for every window](figures/afib_scatter.png)

### A wrong reference annotation

Measured against PhysioNet's `qrs` beat annotations for this database,
our detector scores 96.6%. Those annotations were made by a computer
program and never checked by hand. On record 07162 most of them fall
between heartbeats, while our detections land on each QRS complex:

![Record 07162](figures/afib_07162_reference.png)

Excluding 07162, beat detection scores 99.10% sensitivity and 98.93%
precision.

## Running it

```bash
pip install -r requirements.txt
python qrs_benchmark.py    # ~5 min; MIT-BIH data is streamed from PhysioNet
python afib.py             # first run downloads ~2 h of each AFDB record (slow); cached in data/
```

Per-record numbers are written to [`results/`](results/) and plots to
[`figures/`](figures/).

## Limitations

- Beat detection is scored on the first 5 minutes of each MIT-BIH record,
  not the full 30 minutes.
- The AF check uses only the first 2 hours of each 10-hour AFDB record,
  to keep downloads manageable.
- Only one ECG lead is used (MLII for MIT-BIH, ECG1 for AFDB).
- This is a learning project, not a medical device.

## Data

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.*
  IEEE Eng Med Biol 20(3):45-50 (2001).
- Moody GB, Mark RG. *A new method for detecting atrial fibrillation
  using R-R intervals.* Computers in Cardiology 10:227-230 (1983).
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.*
  Circulation 101(23):e215-e220 (2000).
- Pan J, Tompkins WJ. *A real-time QRS detection algorithm.*
  IEEE Trans Biomed Eng 32(3):230-236 (1985).
