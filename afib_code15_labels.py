"""Where does regular AF go in CODE-15%? (worked out after tests 11-12; not registered)

CODE's AF label (Ribeiro et al. 2020, Methods): accepted when the
cardiologist's report and the automatic analysis (Uni-G statement or
Minnesota code) agreed; when only the cardiologist reported AF, it was
accepted only if the standard deviation of the NN intervals was above 646
(units not given); exams still unresolved were reviewed by medical
students. A rejected diagnosis was "considered absent", and no exams are
described as removed. So cardiologist-reported AF on a regular rhythm can
end up labelled not-AF.

Uses test 11's saved recordings (CODE-15% part 0 and SPH) only:
  1. share of AF recordings that are almost perfectly regular (RR CV < 0.05)
     in CODE-15% against SPH, whose labels are physicians' own;
  2. each model's AF catch rate on regular and irregular SPH AF (at its
     test 11 cutoff) - how much does dropping regular AF favour a model?
  3. CODE-15% "not-AF" recordings called AF, regular and irregular.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd

from afib_comparison_test import rate_str

C_OURS, C_CNN = 0.473559896031711, 0.06515      # test 11 cutoffs (afib_comparison_test.py)
REG = 0.05

if __name__ == "__main__":
    code = pd.read_csv("results/afib_comparison_code15_recordings.csv")
    code = code[code.prob_ours.notna() & code.cv.notna()]
    sph_f = pd.read_csv("results/afib_sph_test_recordings.csv")[["record", "cv"]]
    sph = pd.read_csv("results/afib_comparison_sph_recordings.csv").merge(sph_f, on="record")
    rows = []
    for name, d in [("SPH (physician labels)", sph), ("CODE-15% part 0", code)]:
        af = d[d.label == "A"]
        reg = af.cv < REG
        rows.append({"set": name, "AF recordings": len(af), "regular AF (CV<0.05)": rate_str(reg.values)})
        print(rows[-1])
    sd_ms = (code.cv * 60000 / code.hr)[code.label == "A"]
    print("CODE-15% AF, SD of RR (ms): percentiles 0/1/5/50:", np.percentile(sd_ms.dropna(), [0, 1, 5, 50]).round(1))

    print("\nSPH AF caught, by regularity (test 11 cutoffs; Ribeiro's cutoff as in afib_comparison_test.py)")
    import afib_comparison_test as t11
    from af import cutoff_for_sensitivity
    dev = pd.read_csv("results/compare_probs_dev.csv")
    c_rib = min(cutoff_for_sensitivity(dev.prob_ribeiro.values[(dev.source == s).values],
                                       (dev.label == "A").values[(dev.source == s).values],
                                       dev.record.values[(dev.source == s).values]) for s in ["cpsc", "ptbxl", "chapman"])
    by = []
    for name, d in [("SPH", sph), ("CODE-15% part 0", code)]:
        af = d[d.label == "A"]
        for grp, m in [("regular (CV < 0.05)", af.cv < REG), ("irregular", af.cv >= REG)]:
            r = {"AF": f"{name}, {grp}", "n": int(m.sum()),
                 "this model": rate_str(af.prob_ours[m] >= C_OURS), "CNN": rate_str(af.prob_cnn[m] >= C_CNN),
                 "Ribeiro": rate_str(af.prob_ribeiro[m] >= c_rib) if "prob_ribeiro" in af else "–"}
            by.append(r)
            print(r)

    print("\nCODE-15% part 0 'not-AF' called AF, by regularity")
    na = code[code.label != "A"]
    fa = []
    for grp, m in [("regular (CV < 0.05)", na.cv < REG), ("irregular", na.cv >= REG)]:
        r = {"CODE-15% not-AF": grp, "n": int(m.sum()), "this model": rate_str(na.prob_ours[m] >= C_OURS),
             "CNN": rate_str(na.prob_cnn[m] >= C_CNN)}
        fa.append(r)
        print(r)
    pd.concat([pd.DataFrame(rows), pd.DataFrame(by), pd.DataFrame(fa)]).to_csv(
        write_path("results/afib_code15_labels.csv"), index=False)
