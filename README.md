# ECG arrhythmia detection

R-peak (heartbeat) detection and atrial fibrillation (AF) screening on
PhysioNet's MIT-BIH databases, written from scratch with NumPy and SciPy
and benchmarked against an established detector.

| | Result |
|---|---|
| Beat detection, 36 held-out MIT-BIH records | sensitivity **99.57%**, precision **99.57%** |
| `wfdb`'s XQRS on the same records | sensitivity 98.82%, precision 99.96% |
| AF detection, 60 s windows, 23 AFDB records | sensitivity **95.6%**, specificity **76.3%** |

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

- **CV**: standard deviation of RR / mean RR
- **nRMSSD**: root-mean-square of successive RR differences / mean RR

A window is called AF when both exceed a threshold. The thresholds are
fitted by **leave-one-record-out cross-validation**: each record is
classified with thresholds chosen on the other 22 records only. A window
is labelled AF if at least half of it is annotated `(AFIB`; atrial
flutter and junctional rhythm count as non-AF.

### Results

2759 windows (610 AF, 2149 non-AF) from the first 2 hours of 23 records:

| Metric | Value |
|---|---|
| Sensitivity (AF windows caught) | 95.6% (583 / 610) |
| Specificity (non-AF windows passed) | 76.3% (1640 / 2149) |
| Precision | 53.4% |
| ROC AUC, CV alone / nRMSSD alone | 0.895 / 0.896 |
| Median thresholds | CV > 0.120 and nRMSSD > 0.170 |

![CV vs nRMSSD for every window](figures/afib_scatter.png)

### Where it fails

Most false alarms come from a few records (08219, 05261, 04015) with
**frequent ectopic (extra) beats**. Ectopic beats make the RR series
irregular without the rhythm being AF. In the first 48 minutes of record
08378 below, the RR intervals split into two bands, short and long; that
is ectopy, not AF, but it scores as irregular:

![Record 08378 RR intervals](figures/afib_tachogram.png)

This is a limit of the method, not of the beat detector. Re-running the
analysis on PhysioNet's reference beat positions instead of ours gives
similar results: 79.4% specificity and 91.1% sensitivity. Telling AF apart
from ectopy would need features beyond RR irregularity, such as removing
ectopic beats first or checking for missing P waves.

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
