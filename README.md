# ECG arrhythmia detection

[![tests](https://github.com/ryanqi588-code/ecg-arrhythmia-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/ryanqi588-code/ecg-arrhythmia-detection/actions/workflows/tests.yml)

R-peak (heartbeat) detection and atrial fibrillation (AF) screening on
PhysioNet's MIT-BIH databases. Beat detection is written from scratch
with NumPy and SciPy and benchmarked against an established detector;
AF screening uses RR-interval features and logistic regression.

| | Result |
|---|---|
| Beat detection, 36 held-out MIT-BIH records | sensitivity **99.57%**, precision **99.57%** |
| wfdb's XQRS on the same records | sensitivity 98.82%, precision 99.96% |
| AF detection, 84 new patients (Long-Term AF Database), 5-feature logistic regression | sensitivity **96.4%**, specificity 81.8%, ROC AUC 0.938 |
| Same, with ectopic beats removed first | sensitivity 95.8%, specificity **84.3%** |
| Original two-threshold rule, same patients | sensitivity 94.3%, specificity 59.1% |

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

In AF the heartbeat becomes irregularly irregular. The AF scripts run
the beat detector on the
[MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
(AFDB: 23 records of about 10 hours each) and measure how irregular the RR
intervals (times between beats) are in each 60 s window:

| Feature | Definition | AUC alone (dev) |
|---|---|---|
| **CV** | standard deviation of RR / mean RR | 0.895 |
| **nRMSSD** | root-mean-square of successive RR differences / mean RR | 0.896 |
| **Shannon entropy** | entropy of a 16-bin histogram of RR intervals between the window's 5th and 95th percentiles, scaled to 0–1 | 0.831 |
| **TPR** (turning points ratio) | fraction of RR intervals that are a local peak or trough (≈ 2/3 for a random series) | 0.843 |
| **pNN50** | fraction of successive RR differences over 50 ms | 0.957 |

A window is labelled AF if at least half of it is annotated `(AFIB`;
atrial flutter and junctional rhythm count as non-AF.

### How the data was split

| Set | Data | Used for |
|---|---|---|
| **Development** ([`afib.py`](afib.py)) | AFDB, first 2 h of all 23 records (2759 windows, 610 AF) | every design decision, by **leave-one-record-out cross-validation** (each record scored by a model fitted on the other 22 only) |
| **AFDB later hours** ([`afib_test.py`](afib_test.py)) | AFDB, hours 2 to end (11,180 windows, 4955 AF) | first held-out test; same 23 patients, later in their recordings |
| **New patients** ([`afib_test.py`](afib_test.py)) | [Long-Term AF Database](https://physionet.org/content/ltafdb/1.0.0/), first 2 h of all 84 records (9895 windows, 4790 AF) | final test: different people, a different recorder, 128 Hz instead of 250 Hz |

Every model is trained on the development set only. The new-patient test
plan (data, models, cutoff rule) was written down before any of that data
was looked at, and the test was run once.

### Classifiers

1. **Threshold rule** (the first version): AF when both CV and nRMSSD
   exceed grid-searched thresholds.
2. **Logistic regression** on CV + nRMSSD, and on all 5 features. CV and
   nRMSSD are log-transformed, since they span two orders of magnitude.
   Inputs are standardized and classes weighted equally.
3. **Logistic regression on pNN50 alone.** I added this *after* seeing
   that pNN50 alone scored as well as an earlier version of the 5-feature
   model, so it is a follow-up comparison, not a planned one.
4. **Logistic regression on 5 features, with ectopic beats removed** (see
   below).

**Choosing a cutoff.** A missed AF episode matters more than a false
alarm, which a clinician can dismiss by looking at the strip. So each
classifier's cutoff is set on its training data to catch **95% of AF
windows**, then applied unchanged to the data being scored.

### How the model got here

The development history matters, because two of the fixes below were
found by studying earlier results.

**1. Entropy outliers (found on development data).** At first, entropy's
histogram bins spanned each window's full RR range. On record 07859, which
is AF throughout, a few outlier intervals per window stretched that range
about 2.6× and pulled AF windows' entropy from about 0.93 down to 0.69, so
the model missed 80 of its AF windows. Dropping intervals outside the
5th–95th percentiles first, as Dash et al. (2009) do, fixed it. The
5-feature model's development AUC rose from 0.951 to 0.979.

**2. A cutoff that didn't transfer (found on the first held-out test).**
The first run on AFDB's later hours reached only **92.5%** sensitivity
against the 95% target, with 98.0% specificity and AUC 0.990. The model
ranked windows well, but the cutoff carried over from development landed
too high. Investigating on development data only, I found two causes:

- The cutoff was set on the same windows the model was trained on, where
  its probabilities are overconfident. Setting it on **out-of-fold**
  probabilities instead (each training record scored by a model that
  didn't see it) moved development sensitivity from 93.8% to 95.2%.
- Four records supply about 73% of development AF windows, so a cutoff
  that catches 95% of *windows* mostly reflects them. **Weighting records
  equally** raised development sensitivity to 97.7%, and the worst record's
  from 63% to 83%.

Both are now used ([`af.fit_with_cutoff`](af.py)). Re-running the AFDB
later hours with them is a **second look** at that data, so it's reported
as such below.

**3. Ectopic beats.** Frequent extra beats make a normal rhythm look
irregular. [`ecg.flag_ectopic`](ecg.py) flags a beat if its shape
correlates under 0.9 with the median of its neighbours (ventricular
ectopics have a wide, different QRS; AF beats keep their normal shape), or
if it comes early (< 0.8× the local RR) and is followed by a pause
(> 1.15×). Checked against MIT-BIH's hand-labelled beats (first 5 minutes
of 46 records, the same data the thresholds were picked on), it catches
**97.6% of ventricular ectopics** and wrongly flags 3.5% of normal beats.
But it catches only **5% of atrial ectopics**, which look normal and are
often only slightly early. RR intervals touching a flagged beat are
dropped before computing the features. If fewer than 20 remain, the
window uses its unfiltered intervals, so every window is still scored.

### Development results (leave-one-record-out, 95% sensitivity target)

| Classifier | Sensitivity | Specificity | False alarms | Missed AF | ROC AUC |
|---|---|---|---|---|---|
| Threshold rule (CV, nRMSSD) | 92.1% | 79.1% | 449 | 48 | – |
| Logistic (CV, nRMSSD) | 93.1% | 76.5% | 505 | 42 | 0.889 |
| Logistic (5 features) | **97.7%** | 92.6% | 160 | **14** | 0.979 |
| Logistic (pNN50 only) | 98.5% | 86.6% | 289 | 9 | 0.953 |
| **Logistic (5 features, ectopic beats removed)** | 96.6% | **95.3%** | **100** | 21 | **0.982** |

![ROC curves, development](figures/afib_roc.png)

Removing ectopic beats cut development false alarms by 38%; record 08219,
the worst ectopy case, went from 69% to 95% accuracy. The filter also
flags about 13% of beats inside AF windows (the timing rule fires by
chance on irregular rhythm), which costs 7 more missed AF windows.

### AFDB later hours (second look)

| Classifier | Sensitivity | Specificity | False alarms | Missed AF | ROC AUC |
|---|---|---|---|---|---|
| Threshold rule (CV, nRMSSD) | 95.9% | 83.0% | 1060 | 204 | – |
| Logistic (5 features), old cutoff, first look | 92.5% | 98.0% | 126 | 372 | 0.990 |
| Logistic (5 features) | 95.9% | 96.0% | 248 | 205 | 0.990 |
| Logistic (pNN50 only) | 97.4% | 89.9% | 628 | 131 | 0.970 |
| **Logistic (5 features, ectopic beats removed)** | **97.0%** | **97.3%** | **167** | 151 | **0.993** |

The new cutoff method meets the 95% target here. Because the method was
changed after this set was first used, treat these numbers as somewhat
optimistic.

### New patients: Long-Term AF Database (clean test, run once)

| Classifier | Sensitivity | Specificity | Precision | False alarms | Missed AF | ROC AUC |
|---|---|---|---|---|---|---|
| Threshold rule (CV, nRMSSD) | 94.3% | 59.1% | 68.4% | 2089 | 273 | – |
| Logistic (pNN50 only) | 94.9% | 68.5% | 73.9% | 1607 | 245 | 0.879 |
| **Logistic (5 features)** | **96.4%** | 81.8% | 83.3% | 927 | **171** | **0.938** |
| **Logistic (5 features, ectopic beats removed)** | 95.8% | **84.3%** | **85.1%** | **803** | 200 | 0.935 |

![ROC curves, new patients](figures/afib_ltaf_test_roc.png)

- **Sensitivity transfers.** Both 5-feature models catch about 96% of AF
  windows in people they've never seen, so the cutoff fix works across
  patients, not just across time.
- **Specificity doesn't.** It falls from about 96% to 82–84%, and AUC
  from 0.99 to 0.94. New patients' non-AF rhythms are much more varied
  than anything in the 23 development records.
- **Ectopic removal helps a little.** It cuts false alarms by 13%
  (927 → 803), with 29 more missed AF windows and essentially the same
  AUC. That's a much smaller gain than in development.
- **pNN50 alone doesn't generalize.** It nearly matched the 5-feature
  model on AFDB, but trails it badly on new patients (AUC 0.879 vs 0.938).
- **The 5-feature models beat the original rule by a wide margin**, at
  higher sensitivity and with less than half its false alarms.

### Where the new-patient false alarms come from

[`ltaf_errors.py`](ltaf_errors.py) breaks the 927 false alarms down by
record, using the database's beat and rhythm labels. **73% of them come
from 10 of the 84 records**, and those fall into three groups:

1. **Atrial ectopy and other irregular non-AF rhythms** (records 23, 01,
   58): 27–30% of beats are atrial premature beats, with atrial bigeminy
   and supraventricular tachyarrhythmia labelled. These rhythms are
   genuinely irregular, and atrial ectopics are exactly what the ectopic
   filter can't detect.
2. **Beat-detector errors on noisy recordings** (records 45, 56, 24, 03):
   our detector's precision falls to 47% on record 45 and 74–84% on the
   others. False detections chop real RR intervals into irregular pieces.
   Across all 84 records the detector finds 98.8% of beats, but only
   94.4% of its detections are real, against 99.6% on MIT-BIH.
3. **A probable labelling gap** (record 64): its only rhythm labels are at
   hour 18.9 (`(VT`, then `(AFIB` to the end), so its first 18.9 hours
   have no rhythm label and count as non-AF here. Its first 2 hours look
   like AF (CV 0.15, pNN50 0.58, entropy 0.87). It accounts for 76 false
   alarms. Excluding it, a choice made after seeing the results,
   specificity is 82.9% (5 features) and 85.6% (ectopic removed).

So the next improvements would be atrial-ectopy handling (probably needing
P-wave information), a noise check that skips unreadable stretches, and
training on more varied patients.

### Reading the coefficients

Coefficients of the 5-feature model fitted on all development windows, in
log-odds of AF per standard deviation of each feature, with the range
across the 23 cross-validation folds:

| Feature | Coefficient | Fold range |
|---|---|---|
| log CV | +2.43 | +2.12 to +3.50 |
| log nRMSSD | −0.62 | −0.99 to −0.29 |
| entropy | +2.97 | +2.59 to +4.01 |
| TPR | −0.15 | −0.38 to +0.03 |
| pNN50 | +2.43 | +2.27 to +2.66 |

- **Entropy, pNN50 and CV carry the model.** Once outliers are trimmed,
  entropy is almost uncorrelated with CV (r = 0.03), so it adds
  information the others don't have, which is why its weight is large.
- **The negative nRMSSD weight shouldn't be read at face value.** log CV
  and log nRMSSD are 97% correlated, so the model splits "overall
  variability" between them, one positive and one negative. On its own
  nRMSSD rises with AF (AUC 0.896).
- **TPR contributes almost nothing.** Its weight is small and changes
  sign across folds; what it measures is largely covered by pNN50
  (r = 0.66).

### Ectopy in the development data

In the first 48 minutes of AFDB record 08378 below, the RR intervals split
into two bands, short and long. That's ectopy, not AF, but it's
irregular:

![Record 08378 RR intervals](figures/afib_tachogram.png)

For the threshold rule this is a limit of the method, not of the beat
detector. Using PhysioNet's reference beat positions in place of ours
gives similar results (79.4% specificity, 91.1% sensitivity).

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
python afib.py             # development; first run downloads 2 h of each AFDB record (slow); cached in data/
python afib_test.py        # held-out tests; first run downloads AFDB's other ~8 h per record
                           # and 2 h of 84 LTAFDB records (~2-3 h in total); ~15 min after that
python ltaf_errors.py      # breakdown of the new-patient false alarms (run after afib_test.py)
```

Per-record numbers are written to [`results/`](results/) and plots to
[`figures/`](figures/).

### Tests

```bash
pip install pytest
pytest
```

The tests in [`tests/`](tests/) use synthetic ECG and RR series, so they
run offline in about a second. They check that the detector finds
upside-down beats, rejects T waves and copes with one tall spike, and
that `score()` counts matches correctly. They also check that each AF
feature gives the expected value on series where the answer is known,
for example TPR = 1 for alternating intervals and ≈ 2/3 for random
ones, and that entropy ignores a single long pause. The ectopic-beat
tests check that a synthetic premature ventricular beat is flagged,
irregular normal-shaped beats mostly aren't, and the RR intervals around
flagged beats are removed. GitHub Actions runs the tests on Python 3.11
and 3.12 on every push.

## Limitations

- Beat detection is scored on the first 5 minutes of each MIT-BIH record,
  not the full 30 minutes.
- AF models are trained on only 23 patients. On 84 new patients,
  specificity falls to 82–84%, mostly from atrial ectopy, noisy
  recordings and one probable labelling gap.
- The ectopic-beat filter catches ventricular ectopics but almost no
  atrial ones, and its thresholds were chosen on the same MIT-BIH data
  used to measure it.
- The AFDB later-hours results are a second look at that data, after
  the cutoff method was changed in response to the first.
- One very tall beat or artefact (several times the height of its
  neighbours) still hides the beats within about 1 s of it from the
  adaptive detector, because it raises that window's threshold. The
  fixed threshold loses the whole recording in the same situation.
- Only one ECG lead is used (MLII for MIT-BIH, the first channel for
  AFDB and LTAFDB).
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
- Petrutiu S, Sahakian AV, Swiryn S. *Abrupt changes in fibrillatory
  wave characteristics at the termination of paroxysmal atrial
  fibrillation in humans.* Europace 9(7):466-470 (2007). (Long-Term AF
  Database)
- Dash S, Chon KH, Lu S, Raeder EA. *Automatic real time detection of
  atrial fibrillation.* Ann Biomed Eng 37(9):1701-1709 (2009).
