"""The paper's headline numbers (README.md), computed from results/*.csv.

    python paper_numbers.py            # print each number, its source, and the text it appears as
    python paper_numbers.py --check    # also confirm each text appears in README.md (exit 1 if not)

The paper and the code drift apart when a number is typed by hand (as
"10 vs 11 features" once did). Here each number is computed from the saved
results and formatted the way the README writes it; --check (and
tests/test_paper_numbers.py, run by CI) fails if the README says anything
else. When a number changes, update the README from this output.
Covers the summary, the results table, the conclusion and discussion, and
the headline numbers in each test's section. The detailed tables in each
test's section and in the supplement are typed by hand and not checked here.
"""

import ast
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pandas as pd

R = "results/"
NUM_WORDS = {17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def p0(x):
    """A percentage without a trailing .0 (100% not 100.0%)."""
    s = f"{100 * x:.1f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


def neg(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def half_up(x, d=1):
    """Round an exact Fraction half up (51/20 = 2.55 -> 2.6), as written by hand; floats would give 2.5."""
    q = Decimal(x.numerator) / Decimal(x.denominator)
    return str(q.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def counts(s):
    """'9.5% (7.3%–12.3%) [51/537]' -> (51, 537)."""
    k, n = re.search(r"\[(\d+)/(\d+)\]", s).groups()
    return int(k), int(n)


def frac(s):
    k, n = counts(s)
    return k / n


def ci(s):
    """'9.5% (7.3%–12.3%) [51/537]' -> '7.3–12.3' style bounds as floats."""
    lo, hi = re.search(r"\(([\d.]+)%–([\d.]+)%\)", s).groups()
    return float(lo), float(hi)


def code_constant(path, name):
    tree = ast.parse(open(path).read())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def numbers():
    out = []

    def add(where, what, text, source):
        out.append({"where": where, "what": what, "text": text, "source": source})

    # ------------------------------------------------------------ the model
    m = pd.read_csv(R + "final_af_model.csv")
    feats = m[~m.feature.isin(["intercept", "cutoff"])]
    add("Approach", "model inputs", f"logistic regression on {len(feats)} named features", "final_af_model.csv")
    cut = float(m.loc[m.feature == "cutoff", "value"].iloc[0])
    add("Footnote 2", "adopted cutoff", f"cutoff {cut:.4f}", "final_af_model.csv")
    add("Footnote 1", "tests 9-11 cutoff", f"cutoff {code_constant('afib_comparison_test.py', 'OUR_CUTOFF_TEST9'):.4f}",
        "afib_comparison_test.py OUR_CUTOFF_TEST9")
    reg = pd.read_csv(R + "registration_check.csv")
    n_tests = max(int(t) for t in reg.test if str(t).isdigit())
    add("Summary", "number of tests", f"The project ran {NUM_WORDS[n_tests]} tests", "registration_check.csv")

    # ------------------------------------------------------------ test 9
    s = pd.read_csv(R + "afib_sph_test.csv").iloc[0]
    c = pd.read_csv(R + "afib_comparison_test.csv").set_index(["set", "model"])
    add("Results table, test 9", "SPH recordings", f"{c.loc[('sph', 'ours'), 'n']:,} ECGs, Shandong", "afib_comparison_test.csv")
    add("Results table, test 9", "sensitivity",
        f"sensitivity **{pct(s.sensitivity)}** (95% CI {100 * s.sens_ci_low:.1f}–{pct(s.sens_ci_high)})", "afib_sph_test.csv")
    add("Results table, test 9", "PAC false alarms", f"**{pct(s.false_alarm_rate_PAC)}** on atrial premature beats",
        "afib_sph_test.csv")
    add("Results table, test 9", "normal false alarms", f"**{pct(s.false_alarm_rate_N)}** on normal ECGs", "afib_sph_test.csv")
    add("Results table, test 9", "AUC", f"normal ECGs; AUC {s.auc:.3f}", "afib_sph_test.csv")

    # ------------------------------------------------------------ test 11
    a = pd.read_csv(R + "afib_comparison_afpac.csv").set_index("model")
    ours, rib, cnn = (a.loc[k, "AF vs PAC AUC"] for k in ["this model", "Ribeiro's network", "same-data CNN"])
    assert round(rib, 3) == round(cnn, 3)
    add("Results table, test 11", "AF-vs-PAC AUC (after the test)", f"AF-vs-PAC AUC {ours:.3f} against {rib:.3f} for both",
        "afib_comparison_afpac.csv")
    wrong = a["pairs ranked wrong"]
    add("Results table, test 11", "pairs ranked wrong",
        f"({pct(wrong['this model'])} vs {pct(min(wrong[['Ribeiro' + chr(39) + 's network', 'same-data CNN']]))}",
        "afib_comparison_afpac.csv")
    ratios = a["error ratio, this model / network"].dropna()
    assert all(round(r) == 3 for r in ratios)
    add("Conclusion", "SPH ranking-error ratio", "about three times the ranking errors at SPH", "afib_comparison_afpac.csv")
    pac = {(st, mo): Fraction(*counts(c.loc[(st, mo), "PAC called AF"]))
           for st in ["sph", "ningbo"] for mo in ["ours", "cnn", "ribeiro"]}
    fa_ratio = [pac[st, "ours"] / pac[st, net] for st in ["sph", "ningbo"] for net in ["cnn", "ribeiro"]]
    add("Results table, test 11", "PAC false-alarm ratio",
        f"{half_up(min(fa_ratio))}–{half_up(max(fa_ratio))} times as many false alarms", "afib_comparison_test.csv")
    share = [float(1 / r) for r in fa_ratio]
    add("Discussion", "networks' PAC false alarms as a share",
        f"networks make {round(min(share) * 10) * 10}–{round(max(share) * 10) * 10}% as many PAC", "afib_comparison_test.csv")
    au = {mo: c.loc[("sph", mo), "auc"] for mo in ["ours", "ribeiro", "cnn"]}
    add("Results table, test 11", "overall AUC (registered primary)",
        f"({au['ours']:.3f} vs {au['ribeiro']:.3f} and {au['cnn']:.3f})", "afib_comparison_test.csv")
    d = pd.read_csv(R + "afib_comparison_doctors.csv")
    d = d[d.on == "scored by ours"].set_index("reader")
    k, n = counts(d.loc["model: ours", "sensitivity"])
    add("Results table, doctors", "AF caught", f"best readers ({k} of {n})", "afib_comparison_doctors.csv")
    humans = [frac(d.loc[r, "specificity"]) for r in ["cardiologist1", "cardiologist2", "cardiology_residents",
                                                        "emergency_residents", "medical_students"]]
    add("Results table, doctors", "specificity",
        f"specificity ({pct(frac(d.loc['model: ours', 'specificity']))} vs {pct(min(humans))[:-1]}–{p0(max(humans))})",
        "afib_comparison_doctors.csv")

    # ------------------------------------------------------------ after test 11: ectopy split
    e = pd.read_csv(R + "afib_comparison_ectopy.csv").set_index(["group", "model"])
    only = [pct(frac(e.loc[("PAC only", mo), "PAC called AF"])) for mo in ["this model", "Ribeiro's network", "same-data CNN"]]
    add("Conclusion", "PAC-only false alarms", f"({only[0]} called AF against {only[1]} and {only[2]})",
        "afib_comparison_ectopy.csv")
    vk, vn = counts(e.loc[("PAC+VPC", "this model"), "PAC called AF"])
    nets = {counts(e.loc[("PAC+VPC", mo), "PAC called AF"])[0] for mo in ["Ribeiro's network", "same-data CNN"]}
    assert len(nets) == 1
    add("Conclusion", "mixed ectopy", f"it called {vk} of {vn} such recordings AF, against {nets.pop()} for each network",
        "afib_comparison_ectopy.csv")

    # ------------------------------------------------------------ test 12
    u = pd.read_csv(R + "afib_unmeasured_test.csv").set_index("model")
    n12 = counts(u.loc["final", "AF caught"])[1] + counts(u.loc["final", "non-AF called AF"])[1]
    add("Results table, test 12", "recordings", f"{n12:,} ECGs, CODE-15% part 1", "afib_unmeasured_test.csv")
    add("Results table, test 12", "false alarms",
        f"false alarms {pct(frac(u.loc['final', 'non-AF called AF']))} → **{pct(frac(u.loc['fixed', 'non-AF called AF']))}**",
        "afib_unmeasured_test.csv")
    assert u.loc["final", "AF caught"] == u.loc["fixed", "AF caught"]
    add("Results table, test 12", "sensitivity", f"sensitivity unchanged ({pct(frac(u.loc['fixed', 'AF caught']))})",
        "afib_unmeasured_test.csv")
    add("Results table, test 12", "AUC", f"AUC {u.loc['final', 'auc']:.3f} → {u.loc['fixed', 'auc']:.3f}",
        "afib_unmeasured_test.csv")

    # ------------------------------------------------------------ test 17
    t = pd.read_csv(R + "afib_abstain_test_recordings.csv")
    y, rv = t.label == "A", t.p_unmeasured.astype(bool)
    add("Results table, test 17", "recordings", f"{len(t):,} ECGs, CODE-15% part 3", "afib_abstain_test_recordings.csv")
    add("Results table, test 17", "false calls relabelled", f"{int((rv & ~y & t.adopted).sum())} false AF calls became",
        "afib_abstain_test_recordings.csv")
    add("Results table, test 17", "new flags", f"the cost of {int((rv & ~y & ~t.adopted).sum())} new flags and "
        f"{int((rv & y & t.adopted).sum())} AF calls", "afib_abstain_test_recordings.csv")
    add("Results table, test 17", "recordings needing a look",
        f"recordings needing a human look {int(t.adopted.sum())} → {int((t.abstain | rv).sum())}",
        "afib_abstain_test_recordings.csv")

    # ------------------------------------------------------------ test 18
    f = pd.read_csv(R + "afib_afpac_test.csv").set_index("model")
    df = pd.read_csv(R + "afib_afpac_test_diffs.csv").set_index("comparison")
    naf, npac = counts(f.loc["interpretable", "AF caught"])[1], counts(f.loc["interpretable", "PAC called AF"])[1]
    add("Results table, test 18", "patients", f"{naf + npac} new US patients", "afib_afpac_test.csv")
    add("Results table, test 18", "AF and PAC", f"({naf} AF, {npac} PAC", "afib_afpac_test.csv")
    auc18 = f["AF vs PAC AUC"]
    add("Results table, test 18", "AUCs", f"AUC {auc18['interpretable']:.3f} against {auc18['CNN']:.3f} (CNN) and "
        f"{auc18['Ribeiro']:.3f} (Ribeiro)", "afib_afpac_test.csv")
    dc, dr = df.loc["interpretable - CNN"], df.loc["interpretable - Ribeiro"]
    add("Results table, test 18", "AUC differences (primary)",
        f"{neg(dc['diff'])} (95% CI {neg(dc.lo)} to {neg(dc.hi)}) and {neg(dr['diff'])} ({neg(dr.lo)} to {neg(dr.hi)})",
        "afib_afpac_test_diffs.csv")
    r18 = f["error ratio, interpretable / network"].dropna()
    assert len({round(x, 1) for x in r18}) == 1
    add("Results table, test 18", "ranking-error ratio", f"{r18.iloc[0]:.1f} times the ranking errors here",
        "afib_afpac_test.csv")
    add("Conclusion", "ranking-error ratio", f"({r18.iloc[0]:.1f} times the ranking errors)", "afib_afpac_test.csv")
    add("Discussion", "ranking-error ratio range", f"{r18.iloc[0]:.1f}–3 times as often", "afib_afpac_test.csv, afib_comparison_afpac.csv")
    # The model's actual output: "needs review" is no call (the registered rates counted it by probability).
    fa18 = [pct(frac(f.loc[mo, "output: PAC called AF"])) for mo in ["interpretable", "CNN", "Ribeiro"]]
    rv18 = pct(frac(f.loc["interpretable", "output: PAC needs review"]))
    add("Results table, test 18", "PAC called AF (model output)",
        f"PAC called AF {fa18[0]} against {fa18[1]} and {fa18[2]} (the adopted model's output; {rv18} more",
        "afib_afpac_test.csv")
    out18 = lambda col: [pct(frac(f.loc[mo, col])) for mo in ["interpretable", "CNN", "Ribeiro"]]
    add("Test 18", "PAC called AF row", "| PAC called AF (model's output) | {} | {} | {} |".format(*out18("output: PAC called AF")),
        "afib_afpac_test.csv")
    add("Test 18", "AF called AF row", "| AF called AF (model's output) | {} | {} | {} |".format(*out18("output: AF called AF")),
        "afib_afpac_test.csv")
    add("Test 18", "needs review row", "| Needs review, no call (AF / PAC) | {} / {} |".format(
        pct(frac(f.loc["interpretable", "output: AF needs review"])), rv18), "afib_afpac_test.csv")
    reg = [f"{pct(frac(f.loc[mo, 'PAC called AF']))} / {pct(frac(f.loc[mo, 'AF caught']))}" for mo in ["interpretable", "CNN", "Ribeiro"]]
    add("Test 18", "registered row", "| As registered, probability ≥ cutoff (PAC called AF / AF caught) | {} | {} | {} |".format(*reg),
        "afib_afpac_test.csv")
    nconf = int(re.search(r"confirmed AF \((\d+)\)", " ".join(df.index)).group(1))
    add("Test 18", "confirmed AF", f"the {nconf} AF recordings confirmed", "afib_afpac_test_diffs.csv")
    # The README says how many numbers this checks; check that too, so it can't go stale.
    add("Running it", "count of checked numbers", f"its {len(out)} headline numbers", "paper_numbers.py")
    return pd.DataFrame(out)


def check(readme_text, n=None):
    """numbers() with an 'in README' column: is each text in readme_text (line breaks ignored)?"""
    n = numbers() if n is None else n.copy()
    readme = re.sub(r"\s+", " ", readme_text)
    n["in README"] = [re.sub(r"\s+", " ", x) in readme for x in n.text]
    return n


if __name__ == "__main__":
    n = check(open("README.md").read())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(n[["where", "what", "text", "source", "in README"]].to_string(index=False))
    missing = n[~n["in README"]]
    print(f"\n{len(n)} numbers; {len(n) - len(missing)} found in README.md as written")
    if "--check" in sys.argv and len(missing):
        print("NOT FOUND:\n" + "\n".join(f"  {r.where} / {r.what}: {r.text!r}" for r in missing.itertuples()))
        sys.exit(1)
