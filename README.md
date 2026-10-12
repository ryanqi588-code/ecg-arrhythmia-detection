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
| AF vs atrial premature beats, 777 clinical recordings (CPSC 2018), 5-feature model + P-wave override | sensitivity 92.3%; false alarms 10.4% on PAC (18.4% without the override), 3.2% on normal |
| 917 further clinical recordings (CPSC 2018 extra), clinical cutoff + P-wave override | sensitivity 88.9% (95% CI 77–95%, 45 AF), specificity 95.8% |
| 2938 10 s clinical recordings (Georgia), 12-lead clinical model | sensitivity 68.5%; false alarms 7.3% on PAC, 0.9% on normal (see the caveat on its AF labels) |
| 2867 10 s recordings with per-recording labels (PTB-XL), same model | sensitivity 87.4% (95% CI 85–89%); false alarms 10.3% on PAC, 1.0% on sinus; ROC AUC 0.976 |
| 3813 new 10 s recordings (Chapman-Shaoxing), model retrained on CPSC + PTB-XL | sensitivity **98.9%** (95% CI 98.3–99.3%); false alarms 24.0% on PAC, 1.5% on sinus; ROC AUC 0.990 |

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
the Challenge data has now been used. The next section tries the first
two.

### Short recordings: normalized entropy and handheld training

**Two changes, developed on the now-used data** (Challenge 2017 by 5-fold
cross-validation, AFDB leave-one-record-out):

1. **`entropy_norm`**: entropy divided by that of a random series of the
   same length. A random 20-interval series scores only 0.82 (60 intervals:
   0.95), so short recordings looked less AF-like for purely technical
   reasons. Challenge AF medians by length go from 0.77 / 0.85 / 0.90 to
   0.94 / 0.95 / 0.95. On its own it mostly moved the operating point
   (handheld sensitivity 77.8% → 83.5%, AUC unchanged).
2. **Training on AFDB plus all Challenge 2017 recordings**, with a
   separate cutoff for short recordings
   ([`af.fit_with_source_cutoffs`](af.py)). One shared cutoff let short
   recordings drag it down and cut AFDB specificity from 95% to 83–86%;
   per-source cutoffs kept it at about 91%. In development on handheld
   data, with the override: **92.4%** sensitivity, against 76.0% before,
   but more false alarms (normal 10.3% → 18.4%, other 16.3% → 30.4%).

**The test: CPSC 2018** ([`afib_cpsc_test.py`](afib_cpsc_test.py)), from
the [PhysioNet/CinC Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/)
data. Short clinical 12-lead recordings (500 Hz), lead I only, one window
per recording. 931 recordings were chosen from their headers alone (single
diagnosis of AF, PAC or normal sinus rhythm, at least 15 s), and the plan
was committed before running. 777 were scored; 154 had fewer than 20
usable RR intervals, mostly normal recordings with slow heart rates.

| Model (777 recordings: 286 AF, 212 PAC, 279 normal) | Catches AF | False alarms, PAC | False alarms, normal | ROC AUC |
|---|---|---|---|---|
| Previous (AFDB only, 5 features) | 92.3% | 18.4% | 4.7% | **0.952** |
| **Previous + P-wave override** | 92.3% | **10.4%** | **3.2%** | – |
| New (AFDB + Challenge, `entropy_norm`, short-recording cutoff) | **98.3%** | 50.5% | 22.2% | 0.917 |
| New + P-wave override | 97.9% | 24.1% | 5.0% | – |

![CPSC 2018 ROC](figures/afib_cpsc_test_roc.png)

- **The P-wave override is confirmed.** On recordings explicitly
  diagnosed with atrial premature contractions, it halves false alarms
  (18.4% → 10.4%, and 50.5% → 24.1%) at essentially no sensitivity cost.
  In the plot, both override points (squares) sit above and to the left
  of their model's entire ROC curve. The override does better than any
  cutoff on the model alone, because P waves are information the RR
  features don't have.
- **The handheld training didn't transfer.** It raised sensitivity to 98%
  but tripled false alarms and *lowered* AUC (0.952 → 0.917). The cutoff
  learned for short handheld recordings (0.23) is far too low here.
  Grouping recordings by *length* was the mistake: these clinical
  recordings are short like the handheld ones, but their signal quality
  is much closer to Holter, and the AFDB-only model already catches 92%
  of their AF. Handheld recordings seem to need their own model or
  cutoff by *device*, not length.
- **Best on this test: the AFDB-only model + P-wave override**: 92.3%
  sensitivity, 10.4% false alarms on PAC, 3.2% on normal.

### Routing by device instead of length

| Device | Model | Cutoff |
|---|---|---|
| Holter | AFDB-trained | from AFDB |
| Clinical 12-lead | AFDB-trained (it ranked CPSC best) | for 95% sensitivity on the 777 used CPSC 2018 recordings: 0.398, against 0.552 from AFDB |
| Handheld | trained on Challenge 2017 only | from Challenge 2017 |

All three apply the P-wave override.

**Handheld: cross-validation only.** No unused public handheld AF data is
left, so the only evidence is 5-fold cross-validation on Challenge 2017
([`afib_handheld_cv.py`](afib_handheld_cv.py)). There, the handheld-only
model with the override catches 92.7% of AF with false alarms of 14.8% on
normal and 28.5% on other rhythms, against 93.0% / 19.2% / 31.8% for the
AFDB + Challenge model. `entropy_norm` makes no difference when all the
recordings come from one device.

**Clinical: tested on CPSC 2018 "extra" recordings**
([`afib_cpscx_test.py`](afib_cpscx_test.py)): a second, unused batch from
the same challenge, with no records shared with the CPSC set. AF only
occurs alongside other diagnoses there, so AF = "AF among the diagnoses"
and non-AF = "neither AF nor atrial flutter". Lead I, recordings of at
least 15 s chosen from headers; the plan was committed before the run.
*Disclosure:* a dry run of the script accidentally used the first 25
downloaded test files instead of stand-ins (19 scored: 1 AF, 18 non-AF,
all correct for every model). Nothing was changed after it.

| CPSC-extra (917 recordings: 45 AF, 32 PAC, 840 other) | Catches AF (95% CI) | Specificity | False alarms, PAC | False alarms, other | Missed AF |
|---|---|---|---|---|---|
| Holter cutoff | 86.7% (74–94%) | 95.0% | 6.2% | 5.0% | 6 |
| Holter cutoff + override | 86.7% (74–94%) | **96.0%** | 6.2% | 3.9% | 6 |
| Clinical cutoff | 88.9% (77–95%) | 94.0% | 15.6% | 5.6% | 5 |
| **Clinical cutoff + override** | **88.9% (77–95%)** | 95.8% | 9.4% | 4.0% | **5** |

- **The clinical cutoff made only a small difference:** one more AF
  recording caught (5 missed instead of 6) for 2 more false alarms with
  the override. With 45 AF recordings that's well within the noise.
  Neither cutoff reaches the 95% target here.
- **The model ranks well on this new clinical data** (AUC 0.96).
- **The P-wave override helped for the third time on unseen data:**
  20–30% fewer false alarms at no sensitivity cost.
- **The missed AF has mixed causes:** two recordings with frequent PVCs
  (one has 30% of beats flagged ectopic and very clumped RR intervals),
  one with a fairly regular ventricular response (CV 0.087), one labelled
  "atrial fibrillation and flutter", and two just under the cutoff. AF
  with a regular pulse or heavy ventricular ectopy remains the hard case
  for RR-interval features.

### A clinical 12-lead model, tested on 10 s recordings (Georgia)

**The idea.** Clinical recordings have 12 leads, so atrial activity can be
read where it's clearest: P-wave consistency over regular beats in
**lead II**, and fibrillatory-wave activity in **V1**
([`ecg.f_wave_activity`](ecg.py): subtract the median beat, then measure
the 4–9 Hz activity left, relative to beat size). These go into a model
trained on clinical recordings ([`af.clinical_features`](af.py)).

**Development** ([`afib_clinical_dev.py`](afib_clinical_dev.py)): the
used CPSC 2018 and CPSC-extra recordings, cut into 10 s crops (3942 crops,
654 AF), compared by 5-fold cross-validation grouped by recording. Each
cutoff aims at 95% sensitivity.

| Model (development) | AUC | Sensitivity | False alarms |
|---|---|---|---|
| AFDB model, clinical cutoff | 0.959 | 95.6% | 13.0% |
| Clinical model, RR only | 0.950 | 95.4% | 11.5% |
| Clinical model, RR + P waves (II) | 0.976 | 95.7% | 5.1% |
| **Clinical model, RR + P waves (II) + f-waves (V1)** | **0.978** | **95.9%** | **4.7%** |

Lead II P waves work as a model feature here, unlike with AFDB training,
because clinical training data includes PAC and other rhythms. For 10 s
recordings the RR minimum is 6 intervals and `p_match_regular` needs 3
regular beats; with 5, many normal recordings had too few and were scored
as P-wave-free.

**The test: Georgia** ([`afib_georgia_test.py`](afib_georgia_test.py)),
from the PhysioNet/CinC Challenge 2020: 2942 10 s clinical recordings
chosen from headers only. AF = AF among the diagnoses (570); PAC = PAC
without AF or flutter (620); normal = sinus rhythm as the only diagnosis
(1752). The plan was committed before the run. (The first attempt crashed
on three truncated downloads before producing any result; they were
re-downloaded, and every file was checked against its header's size.)

| Georgia (2938 scored: 569 AF, 618 PAC, 1751 normal) | Catches AF (95% CI) | False alarms, PAC | False alarms, normal | ROC AUC |
|---|---|---|---|---|
| Current (AFDB model, clinical cutoff) | **75.2%** (72–79%) | 35.0% | 3.6% | 0.844 |
| Current + P-wave override | 71.0% (67–75%) | 12.9% | 1.1% | – |
| Clinical model (registered primary) | 68.5% (65–72%) | **7.3%** | **0.9%** | **0.881** |
| Clinical model + P-wave override | 67.7% (64–71%) | 6.3% | 0.8% | – |

![Georgia ROC](figures/afib_georgia_test_roc.png)

- **The fix failed its main goal.** The clinical model ranks better (AUC
  0.881 vs 0.844) and cuts false alarms on PAC recordings from 35% to
  7%, but it catches *less* AF (68.5% vs 75.2%), far below the 95% target.
- **Both models do much worse than in development** (AUC 0.98 there,
  0.84–0.88 here), so Georgia's AF recordings differ from CPSC's.

**Why (error analysis after the run; nothing was changed in response).**
Many of the missed "AF" recordings don't look like AF. Their median CV
is 0.067 and pNN50 0.23, close to normal rhythm, and their median
`p_match_regular` is 1.000: every regular beat has the same P wave, which
AF by definition doesn't. 67 of the clinical model's 179 missed AF
recordings are perfectly regular (CV < 0.05) with consistent P waves.
Four of them, at random:

![Missed Georgia AF](figures/georgia_missed_af.png)

Georgia lists every diagnosis attached to the patient's ECG, so "AF" may
mean paroxysmal AF elsewhere in the patient's history rather than AF in
this 10 s. One of the four (E03935) is coded with both AF and sinus
bradycardia, which can't both describe one 10 s strip. Others look like
sinus rhythm with visible P waves, and one has a wavy baseline that could
be flutter or noise. I'm not a cardiologist and haven't excluded these
recordings: dropping them after the fact would remove exactly the cases
the model gets wrong. The registered numbers above stand. A fair verdict
on this model needs AF labels that describe the rhythm in each
recording.

### Re-test on PTB-XL, whose labels describe each recording

**The test** ([`afib_ptbxl_test.py`](afib_ptbxl_test.py)):
[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), 10 s 12-lead
recordings at 500 Hz, chosen from the label table alone and keeping only
reports validated by a cardiologist. Its rhythm statements describe the
ECG itself, not the patient's history. AF = AFIB without atrial flutter
(all 984); PAC = PAC without AF or flutter (all 391); sinus = SR with no
AF, flutter, ectopy or pacing (1500 drawn at random, seed 0). The models
are exactly the Georgia ones; nothing was retrained or re-tuned. The plan
was committed before the run, and it was run once. (17 downloads were
incomplete and were re-fetched before the run; every file was then
checked for its expected size.)

| PTB-XL (2867 scored: 978 AF, 389 PAC, 1500 sinus) | Catches AF (95% CI) | False alarms, PAC | False alarms, sinus | ROC AUC |
|---|---|---|---|---|
| Current (AFDB model, clinical cutoff) | **92.4%** (91–94%) | 46.8% | 6.6% | 0.942 |
| Current + P-wave override | 87.2% (85–89%) | 14.9% | 0.9% | – |
| Clinical model (registered primary) | 87.4% (85–89%) | **10.3%** | **1.0%** | **0.976** |
| Clinical model + P-wave override | 86.2% (84–88%) | 8.7% | 0.8% | – |

![PTB-XL ROC](figures/afib_ptbxl_test_roc.png)

- **The labels explain most of the Georgia gap.** With labels that describe
  each recording, the clinical model's AUC is 0.976, the same as in
  development (0.978) and far above Georgia's 0.881. Its sensitivity rose
  from 68.5% to 87.4%, and only 4 of its 123 missed AF recordings are
  perfectly regular with consistent P waves (Georgia: 67 of 179).
- **The model still misses its 95% target.** 87.4% (CI 85–89%) is in
  line with the CPSC-extra test (88.9%). It's the same shortfall as
  before, now measured cleanly.
- **The ranking transfers but the cutoff doesn't.** On the same
  recordings, catching 95% of AF would need a cutoff of 0.27 instead of
  0.61, which would raise false alarms on PAC to 25% (sinus 2.1%). This
  was worked out after the run and isn't a test result.

**Why it misses (after the run; nothing was changed in response).** The
missed AF recordings are clearly irregular (median CV 0.13, pNN50 0.56;
sinus: 0.02 and 0.00), but their median `p_match_regular` is 0.90, close
to sinus (1.00) and far from the AF it catches (0.43). With only a few
regular beats in 10 s of AF, the lead II window before each QRS can look
alike by chance, and the model, which leans on that feature to reject
PAC, then talks itself out of AF. The fix to try next is a P-wave measure
that doesn't depend on picking out a few regular beats; it would need
fresh data to test.

### A sturdier P-wave measure (didn't help) and more training data (did)

**Development** ([`afib_pwave_dev.py`](afib_pwave_dev.py)), on data
already used as tests: the CPSC crops and the 2867 PTB-XL recordings
(6809 units, 1632 AF), 5-fold cross-validation grouped by recording or
patient. [`ecg.p_wave_consistency`](ecg.py) tackles the two suspected
causes above: it compares each beat with the median of the *other* beats
(leave-one-out), and it can first subtract the median QRS-T complex so
the previous beat's T wave can't pass for a P wave.

| Model: RR + P-wave measure + f-waves (CV) | AUC | Sens., CPSC | Sens., PTB-XL | False alarms, PAC | False alarms, sinus |
|---|---|---|---|---|---|
| `p_match_regular` (as before), shared cutoff | 0.975 | 98.2% | 93.1% | 12.3% | 1.3% |
| leave-one-out, all beats | 0.975 | 98.0% | 93.7% | 12.8% | 1.4% |
| leave-one-out + QRS-T removed | 0.973 | 97.9% | 93.9% | 17.9% | 1.6% |
| `p_match_regular`, **strictest per-source cutoff** | 0.975 | 98.5% | **94.6%** | 16.9% | 1.8% |

- **The P-wave idea was wrong.** Leave-one-out matching changes almost
  nothing, and removing the QRS-T complex adds PAC false alarms. Neither
  self-matching nor T-wave overlap was what fooled the feature, so the
  existing feature was kept.
- **Training data mattered more.** With PTB-XL among the training data,
  cross-validated PTB-XL sensitivity is 93.1%, against 87.4% when the
  CPSC-only model was tested on it. One shared cutoff still lets the easier
  source (CPSC) set it, so the cutoff became the *strictest* of the
  per-source 95% cutoffs: every training source reaches 95%, at the cost
  of more PAC false alarms.

**The test: Chapman-Shaoxing** ([`afib_chapman_test.py`](afib_chapman_test.py)),
from the PhysioNet/CinC Challenge 2021: 10 s 12-lead recordings at 500 Hz
with rhythm labelled per ECG by physicians, none used before. Every
recording that qualified from its header: AF without flutter (1780), PAC
without AF or flutter (258), and sinus rhythm with no AF, flutter, ectopy
or pacing (1777). The plan was committed and pushed before the signals
were downloaded; it was run once. Every file was checked for its expected
size first.

| Chapman-Shaoxing (3813 scored: 1778 AF, 258 PAC, 1777 sinus) | Catches AF (95% CI) | False alarms, PAC | False alarms, sinus | ROC AUC |
|---|---|---|---|---|
| **Retrained, strictest-source cutoff (registered primary)** | **98.9%** (98.3–99.3%) | 24.0% | 1.5% | 0.990 |
| Retrained, shared cutoff | 98.3% (97.5–98.8%) | 19.4% | 1.3% | – |
| Previous clinical model | 94.1% (92.9–95.1%) | **9.7%** | **0.8%** | **0.991** |

![Chapman-Shaoxing ROC](figures/afib_chapman_test_roc.png)

- **The 95% target is met on new data**: 98.9%, with the whole
  confidence interval above 95%. It misses 20 of 1778 AF recordings,
  against 105 for the previous model.
- **But it's the same trade, not a better model.** The two models rank
  recordings equally well (AUC 0.990 vs 0.991); the gain comes from a
  lower cutoff (0.42 vs 0.61). Its price shows on PAC: 24% are called
  AF, against 9.7% before. Sinus false alarms stay under 2%.
- **Chapman is easier than PTB-XL.** Even the previous model reaches
  94.1% here (87.4% on PTB-XL), so 98.9% here doesn't mean 98.9% on
  every hospital's recordings; it does show the cutoff now errs on the
  side of catching AF.

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
python afib_cpsc_test.py   # handheld-training test; needs the Challenge 2017 data and CPSC 2018 (below)
python afib_handheld_cv.py # handheld-only model, 5-fold CV on Challenge 2017
python afib_cpscx_test.py  # clinical-cutoff test; needs CPSC 2018 and CPSC 2018 extra (below)
python afib_clinical_dev.py   # clinical 12-lead model development on CPSC crops
python afib_georgia_test.py   # clinical model test on Georgia 10 s recordings (below)
python afib_ptbxl_test.py     # re-test on PTB-XL 10 s recordings (below)
python afib_pwave_dev.py      # P-wave measure and cutoff development on CPSC crops + PTB-XL
python afib_chapman_test.py   # retrained clinical model test on Chapman-Shaoxing (below)
```

Per-record numbers are written to [`results/`](results/) and plots to
[`figures/`](figures/).

The Challenge 2017 recordings aren't fetched automatically. Download
[`training2017.zip`](https://physionet.org/files/challenge-2017/1.0.0/training2017.zip)
(99 MB) and
[`REFERENCE-v3.csv`](https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv)
into `data/cinc2017/`, then unzip the zip there.

For CPSC 2018, download the `.hea` files from
[`training/cpsc_2018/g1`–`g7`](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/)
into `data/cpsc2018/`. List the single-diagnosis AF (164889003), PAC
(284470004) and normal sinus rhythm (426783006) recordings of at least
15 s in `data/cpsc2018/SELECTED`, one per line, and download their `.mat`
files too (about 280 MB). [`results/afib_cpsc_test_recordings.csv`](results/afib_cpsc_test_recordings.csv)
lists the 777 that were scored.

CPSC 2018 extra works the same way, from
[`training/cpsc_2018_extra/g1`–`g4`](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018_extra/)
into `data/cpsc2018_extra/`. `SELECTED` lists every recording of at least
15 s that either includes AF or includes neither AF nor atrial flutter
(1236 recordings, about 370 MB);
[`results/afib_cpscx_test_recordings.csv`](results/afib_cpscx_test_recordings.csv)
lists the 917 scored.

Georgia works the same way, from
[`training/georgia/g1`–`g11`](https://physionet.org/content/challenge-2020/1.0.2/training/georgia/)
into `data/georgia/`. `SELECTED` lists one `record label` pair per line
(A, PAC or N, as defined above; 2942 recordings, about 350 MB).

For PTB-XL, choose recordings from
[`ptbxl_database.csv`](https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv)
as defined above and download their `records500/` `.hea` and `.dat`
files into `data/ptbxl/` (2875 recordings, about 350 MB). `SELECTED`
lists one `records500/xxxxx/yyyyy_hr label` pair per line;
[`results/afib_ptbxl_test_recordings.csv`](results/afib_ptbxl_test_recordings.csv)
lists the 2867 scored.

For Chapman-Shaoxing, download the `.hea` files from
[`training/chapman_shaoxing/g1`–`g11`](https://physionet.org/content/challenge-2021/1.0.3/training/chapman_shaoxing/)
of the Challenge 2021 data into `data/chapman/` (each folder's `RECORDS`
also names the first recording of the next folder, which lives there),
choose recordings as defined above, and download their `.mat` files into
the same folder (3815 recordings, about 460 MB). `SELECTED` lists one
`gN/JSxxxxx label` pair per line.

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
beats share a P wave. The last two check that `entropy_norm` is about 1
for random intervals at any length (while raw entropy is biased low), and
that each recording source gets its own cutoff. GitHub Actions runs the tests on Python 3.11 and 3.12 on
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
  Challenge data and halved them on CPSC 2018's PAC recordings, but about
  10% of PAC recordings are still called AF.
- On short handheld recordings (Challenge 2017), AF sensitivity falls
  to about 78% with Holter-only training. A handheld-only model fixes that
  in cross-validation (92.7% with the override), but there's no unused
  handheld AF data left to confirm it.
- On clinical recordings, earlier models reached 87–92% sensitivity on
  CPSC and PTB-XL and 68–75% on Georgia (mostly a labelling problem). The
  model retrained on CPSC + PTB-XL reached 98.9% on Chapman-Shaoxing, but
  by lowering its cutoff rather than ranking better, so about a quarter of
  PAC recordings are now called AF. Chapman also proved easier than
  PTB-XL, so sensitivity elsewhere may be lower.
- Recordings under about 20 beats can't be scored (154 of 931 in the
  CPSC test).
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
- Liu F, Liu C, Zhao L, et al. *An open access database for evaluating
  the algorithms of electrocardiogram rhythm and morphology abnormality
  detection.* J Med Imaging Health Inform 8(7):1368-1373 (2018). (CPSC
  2018)
- Perez Alday EA, Gu A, Shah AJ, et al. *Classification of 12-lead ECGs:
  the PhysioNet/Computing in Cardiology Challenge 2020.* Physiol Meas
  41(12):124003 (2020).
- Clifford GD, Liu C, Moody B, et al. *AF classification from a short
  single lead ECG recording: the PhysioNet/Computing in Cardiology
  Challenge 2017.* Computing in Cardiology 44 (2017).
- Wagner P, Strodthoff N, Bousseljot RD, et al. *PTB-XL, a large
  publicly available electrocardiography dataset.* Sci Data 7:154 (2020).
- Zheng J, Zhang J, Danioko S, et al. *A 12-lead electrocardiogram
  database for arrhythmia research covering more than 10,000 patients.*
  Sci Data 7:48 (2020). (Chapman-Shaoxing)
- Reyna MA, Sadr N, Perez Alday EA, et al. *Will two do? Varying
  dimensions in electrocardiography: the PhysioNet/Computing in
  Cardiology Challenge 2021.* Computing in Cardiology 48 (2021).
- Greenwald SD, Patil RS, Mark RG. *Improved detection and
  classification of arrhythmias in noise-corrupted electrocardiograms
  using contextual information.* Computers in Cardiology 461-464 (1990).
  (Supraventricular Arrhythmia Database)
- Dash S, Chon KH, Lu S, Raeder EA. *Automatic real time detection of
  atrial fibrillation.* Ann Biomed Eng 37(9):1701-1709 (2009).
