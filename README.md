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
training on more varied patients. The first of these is tried below.

### Trying P waves against atrial ectopy

AF has no P waves, only chaotic fibrillatory activity. Atrial ectopy is
mostly normal beats *with* P waves. I tried using that in two ways, both
looking at the 300–60 ms before each R peak:

**1. Detecting individual atrial ectopic beats** ([`ecg.flag_atrial_ectopic`](ecg.py)):
flag a beat that comes early (< 0.85× the local RR) and whose pre-QRS
stretch correlates < 0.8 with its neighbours'. On a single lead this works
poorly. On the full 30 minutes of 46 MIT-BIH records (excluding record 232,
whose long pauses distort timing), it catches 55% of supraventricular
ectopics but also flags 2.3% of normal beats. Normal beats vastly
outnumber ectopics, so only about 10% of its flags are real atrial
ectopics, and the P-wave check adds little to timing alone. Removing the
flagged beats also cost AF sensitivity in development, because the rule
flags 15% of beats inside AF windows. **No model uses it.**

**2. P-wave presence per window** ([`af.p_wave_features`](af.py)):
`p_match` is the fraction of a window's beats whose pre-QRS stretch
correlates > 0.5 with the window's median one. Averaging over a minute of
beats beats the noise that defeats the beat-level check: on its own,
`p_match` separates AF from non-AF development windows with **AUC 0.937**
(median 0.94 in non-AF windows, 0.38 in AF). A second measure, `p_snr`,
was 85% correlated with it, added nothing and got a wrong-signed weight,
so it was dropped. In development, adding `p_match` to the ectopy-filtered
5-feature model gave AUC 0.985 (vs 0.982), 97.4% sensitivity and 94.5%
specificity: essentially no change. But AFDB's development records
contain little atrial ectopy, so they couldn't show whether it helps.

**The test: 78 new patients with frequent atrial ectopy.** The
[MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/)
(SVDB) was chosen for exactly this problem: 30 minutes each, 128 Hz,
12,188 labelled supraventricular ectopic beats. The test
([`afib_svdb_test.py`](afib_svdb_test.py)) was committed before it was run,
and run once.

| Classifier (78 patients, 2340 windows) | Specificity | False alarms |
|---|---|---|
| Threshold rule (CV, nRMSSD) | 59.4% | 951 |
| Logistic (5 features) | 78.4% | 506 |
| **Logistic (5 features, ectopic beats removed)** | **83.3%** | **390** |
| Logistic (5 features + `p_match`, ectopic beats removed) | 82.8% | 402 |

There are no labelled AF windows here, so only specificity can be
measured (see the caveat below).

- **`p_match` didn't reduce false alarms** (402 vs 390). Removing
  ventricular ectopics did, by 23% (506 → 390).
- **Atrial ectopy drives the false alarms.** The 14 records where over
  10% of beats are supraventricular ectopics have a 40% false-alarm rate;
  the other 64 have 12%. `p_match` changes neither.
- **Why `p_match` fails here:** P waves do come through (median `p_match`
  0.91, close to development's 0.94), but atrial ectopic beats bring their
  own, different P waves. When 20–30% of beats are ectopic, `p_match` falls
  to about 0.7 (median 0.69 in the windows the model wrongly flags), which
  overlaps with real AF windows in development (0.4–0.85). So atrial
  ectopy pushes the P-wave feature towards AF, just as it does the RR
  features.
- **On the same data, the beat-level atrial detector** catches 54% of the
  supraventricular ectopics and flags 1.8% of normal beats; 41% of its
  flags are real atrial ectopics (more than on MIT-BIH, because ectopics
  are far more common here). It also flags 66% of ventricular ectopics,
  which are early and lack a normal P wave too. The ventricular detector
  catches 84% of ventricular ectopics here, against 97.6% on the MIT-BIH
  data its thresholds were chosen on.

**A caveat I should have caught beforehand:** SVDB has **no rhythm labels**
(only one `(N` annotation in all 78 records). The registered rule counted
unlabelled windows as non-AF, so if any of these patients were actually in
AF, some "false alarms" are correct and true specificity is higher than
shown. Record 804, flagged in all 30 windows despite having no ectopic
beats, is a candidate. The comparison *between* models is unaffected,
since they're all scored on the same windows.

### P waves over regular beats only: an override rule

**The fix.** `p_match` failed because the ectopic beats themselves have
odd P waves. The *regular* beats between them (not early, not ventricular
ectopic) still share one sinus P wave, while in AF no beats do. So
`p_match_regular` measures P-wave consistency over regular beats only
([`af.regular_beats`](af.py)). On AFDB development windows its AUC alone
is **0.962** (vs 0.937 for `p_match`). In SVDB's high-ectopy records the
median is 0.97, against 0.81 for `p_match`, so it sees through the ectopy.

**As a model feature it barely helped** (SVDB false alarms 409 vs 390),
because the AFDB training data has almost no "irregular, but with P
waves" windows for the model to learn from. The definition of AF settles
it instead: AF means no P waves. So it's applied as a **rule**
([`af.p_wave_override`](af.py)): a window isn't called AF if at least 95%
of its regular beats share its typical P wave.

| Override threshold | AFDB sensitivity | AFDB false alarms | SVDB false alarms |
|---|---|---|---|
| none | 96.6% | 100 | 390 |
| **0.95** | **96.1%** | **71** | **277** |
| 0.90 | 93.3% | 60 | 251 |
| 0.85 | 89.2% | 47 | 217 |

0.95 is the lowest threshold that keeps AFDB sensitivity above 95%. It was
chosen while looking at SVDB, so SVDB's 29% drop isn't independent
evidence. That's what the next test is for.

**The test: PhysioNet/CinC Challenge 2017.** 8528 single-lead recordings
from a handheld device (300 Hz, 9–61 s), each labelled by experts as
Normal, AF, Other rhythm or Noisy. "Other" covers ectopy and other
non-AF arrhythmias, exactly the hard case. Each recording is scored as
one window. The test ([`afib_cinc_test.py`](afib_cinc_test.py)) was
committed before it was run, and run once. 7903 recordings were scored
(279 noisy excluded, 346 with fewer than 20 usable beats skipped).

| Classifier | Catches AF | False alarms, Normal | False alarms, Other | False alarms | Missed AF | ROC AUC |
|---|---|---|---|---|---|---|
| Threshold rule | 87.9% | 40.5% | 59.5% | 3361 | 84 | – |
| Logistic (5 features) | **79.2%** | 12.3% | 19.1% | 1045 | **145** | **0.881** |
| Logistic (5 features, ectopic removed) | 77.8% | 12.4% | 20.3% | 1077 | 155 | 0.868 |
| **+ P-wave override** | 76.0% | **10.3%** | **16.3%** | **882** | 167 | – |

- **The override does what it was built for.** False alarms on "Other"
  rhythms fall by 20% (20.3% → 16.3%) and 18% overall, for 12 more missed
  AF recordings. That matches what development predicted.
- **But the pipeline transfers poorly to these recordings.** Sensitivity
  is about 78%, far below the 95% target, and AUC 0.87–0.88 against
  0.94–0.99 on Holter data. Removing ventricular ectopics doesn't help
  here either.

**Why sensitivity falls (error analysis only; nothing was changed in
response):**

- **Not mainly recording length.** Sensitivity is 81% on 50–61 s
  recordings and 77% on 20–35 s ones.
- **The missed AF is less irregular.** Median CV 0.126 and pNN50 0.47,
  against 0.19 and 0.72 for typical AF. RR features struggle with AF that
  has a fairly regular pulse.
- **Entropy depends on the number of beats.** About 30 beats can't fill a
  16-bin histogram the way a minute of Holter beats does, so AF entropy
  falls from 0.92 to 0.86 for purely technical reasons. Normalizing it for
  beat count would fix that.
- **The handheld device shows more P-wave-like activity during AF**
  (median `p_match_regular` 0.48 vs 0.33 on Holter), which is probably
  why the override cost some sensitivity here.

Each of these would need yet another fresh dataset to test fairly, since
the Challenge data has now been used.

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
python afib_svdb_test.py   # atrial-ectopy test on 78 SVDB patients; first run downloads ~55 MB
python afib_cinc_test.py   # P-wave override test; needs the Challenge 2017 data (below)
```

Per-record numbers are written to [`results/`](results/) and plots to
[`figures/`](figures/).

The Challenge 2017 recordings aren't fetched automatically. Download
[`training2017.zip`](https://physionet.org/files/challenge-2017/1.0.0/training2017.zip)
(99 MB) and
[`REFERENCE-v3.csv`](https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv)
into `data/cinc2017/`, then unzip the zip there.

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
flagged beats are removed. The P-wave tests check that `p_match` is high
when every beat shares a P wave and low for fibrillation-like activity,
and that the atrial detector flags early beats without a P wave and
nothing else. Others check that `regular_beats` excludes exactly the early
beats, and that the P-wave override only clears windows whose regular
beats share a P wave. GitHub Actions runs the tests on Python 3.11 and 3.12 on
every push.

## Limitations

- Beat detection is scored on the first 5 minutes of each MIT-BIH record,
  not the full 30 minutes.
- AF models are trained on only 23 patients. On 84 new patients,
  specificity falls to 82–84%, mostly from atrial ectopy, noisy
  recordings and one probable labelling gap.
- The ectopic-beat filter catches ventricular ectopics but almost no
  atrial ones, and its thresholds were chosen on the same MIT-BIH data
  used to measure it.
- Atrial ectopy remains the main source of false alarms. Beat-level
  atrial detection and the plain `p_match` feature didn't reduce them;
  the P-wave override cut false alarms on "Other" rhythms by 20% on the
  Challenge data, but most remain.
- On short handheld recordings (Challenge 2017), AF sensitivity falls
  to about 78%. The models are trained only on Holter data, and entropy
  isn't normalized for beat count.
- SVDB has no rhythm labels, so its specificity assumes none of its
  patients were in AF.
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
- Clifford GD, Liu C, Moody B, et al. *AF classification from a short
  single lead ECG recording: the PhysioNet/Computing in Cardiology
  Challenge 2017.* Computing in Cardiology 44 (2017).
- Greenwald SD, Patil RS, Mark RG. *Improved detection and
  classification of arrhythmias in noise-corrupted electrocardiograms
  using contextual information.* Computers in Cardiology 461-464 (1990).
  (Supraventricular Arrhythmia Database)
- Dash S, Chon KH, Lu S, Raeder EA. *Automatic real time detection of
  atrial fibrillation.* Ann Biomed Eng 37(9):1701-1709 (2009).
