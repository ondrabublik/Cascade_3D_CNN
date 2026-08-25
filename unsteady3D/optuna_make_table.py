"""Generate article-ready LaTeX tables from the Optuna studies.

Reads the SQLite storage written by optuna_search.py and writes into
optuna_results/ (suffix = study kind, e.g. _arch / _train):

    table_search_space_<kind>.tex   searched parameters, ranges, best values
    table_top_trials_<kind>.tex     the best N complete trials
    table_importance_<kind>.tex     fANOVA hyperparameter importances

All tables use booktabs (\\usepackage{booktabs} in the article preamble).
A Markdown preview of each table is printed to the console for a quick check.

Usage:
    python optuna_make_table.py --study arch
    python optuna_make_table.py --study train --top 10
"""

import argparse
from pathlib import Path

import optuna

STUDY_NAMES = {"arch": "unet3d_arch", "train": "unet3d_train"}
STORAGE = f"sqlite:///{(Path(__file__).resolve().parent / 'optuna_unet3d.db').as_posix()}"
OUT_DIR = Path(__file__).resolve().parent / "optuna_results"

# How each searched parameter is presented in the article.
# 'symbol' is the short column header for the top-trials table.
PARAM_INFO = {
    "nChannel": {
        "label": r"Base channels $n_{\mathrm{ch}}$",
        "symbol": r"$n_{\mathrm{ch}}$",
        "range": r"$\{8, 10, \dots, 32\}$",
        "kind": "int",
    },
    "frameWidth": {
        "label": r"Kernel half-width $w$ (kernel $2w{+}1$)",
        "symbol": r"$w$",
        "range": r"$\{1, 2, 3\}$",
        "kind": "int",
    },
    "growFactor": {
        "label": r"Channel growth exponent $g$",
        "symbol": r"$g$",
        "range": r"$\{0, 1\}$",
        "kind": "int",
    },
    "act": {
        "label": r"Hidden activation",
        "symbol": r"act.",
        "range": r"relu, elu, swish",
        "kind": "str",
    },
    "learningRate": {
        "label": r"Learning rate $\eta$",
        "symbol": r"$\eta$",
        "range": r"$[10^{-5},\,3\times10^{-3}]$ (log-uniform)",
        "kind": "float",
    },
    "clipNorm": {
        "label": r"Gradient clipping norm $c$",
        "symbol": r"$c$",
        "range": r"$[0.1,\,10]$ (log-uniform)",
        "kind": "float",
    },
    "pressureLossWeight": {
        "label": r"Pressure loss weight $\lambda_p$",
        "symbol": r"$\lambda_p$",
        "range": r"$[0.02,\,1]$ (log-uniform)",
        "kind": "float",
    },
}


def latex_sci(v):
    """3.162e-04 -> $3.16\\times10^{-4}$"""
    mantissa, exp = f"{v:.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exp)}}}$"


def fmt_value(name, value, latex=True):
    kind = PARAM_INFO.get(name, {}).get("kind", "str")
    if kind == "int":
        return str(int(value))
    if kind == "float":
        return latex_sci(value) if latex else f"{value:.3e}"
    return str(value)


def write_table(path, lines, caption_hint):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {path}  ({caption_hint})")


def markdown_preview(header, rows):
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["arch", "train"], default="arch")
    parser.add_argument("--top", type=int, default=10, help="rows in the top-trials table")
    parser.add_argument("--study-name", default=None, help="override the study name")
    parser.add_argument("--storage", default=STORAGE)
    args = parser.parse_args()

    study_name = args.study_name or STUDY_NAMES[args.study]
    study = optuna.load_study(study_name=study_name, storage=args.storage)
    trials = study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    if not complete:
        raise SystemExit("No complete trials in the study yet.")

    best = study.best_trial
    param_names = [p for p in PARAM_INFO if p in best.params]  # stable order
    OUT_DIR.mkdir(exist_ok=True)
    suffix = args.study

    print(f"study '{study_name}': {len(trials)} trials "
          f"({len(complete)} complete, {len(pruned)} pruned)")
    print(f"best trial: #{best.number}, val_ref_loss = {best.value:.6e}")

    # ------------------------------------------------------------------ 1
    # Search space + best value
    lines = [
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Hyperparameter & Search range & Best value \\",
        r"\midrule",
    ]
    md_rows = []
    for name in param_names:
        info = PARAM_INFO[name]
        best_val = fmt_value(name, best.params[name])
        lines.append(f"{info['label']} & {info['range']} & {best_val} " + r"\\")
        md_rows.append([name, info["range"], fmt_value(name, best.params[name], latex=False)])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(OUT_DIR / f"table_search_space_{suffix}.tex", lines, "search space + best values")
    markdown_preview(["parameter", "range", "best"], md_rows)

    # ------------------------------------------------------------------ 2
    # Top-N complete trials (with network size and the objective value)
    top = sorted(complete, key=lambda t: t.value)[: args.top]
    col_spec = "rc" + "c" * len(param_names) + "cc"
    header_syms = [PARAM_INFO[p]["symbol"] for p in param_names]
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Rank & Trial & " + " & ".join(header_syms) + r" & Parameters & Val. loss \\",
        r"\midrule",
    ]
    md_rows = []
    for rank, t in enumerate(top, start=1):
        n_params = t.user_attrs.get("n_params")
        n_params_tex = f"{n_params:,}".replace(",", r"\,") if n_params else "--"
        row = [str(rank), str(t.number)]
        row += [fmt_value(p, t.params[p]) for p in param_names]
        row += [n_params_tex, latex_sci(t.value)]
        lines.append(" & ".join(row) + r" \\")
        md_rows.append([rank, t.number]
                       + [fmt_value(p, t.params[p], latex=False) for p in param_names]
                       + [n_params if n_params else "--", f"{t.value:.4e}"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(OUT_DIR / f"table_top_trials_{suffix}.tex", lines, f"top {len(top)} trials")
    markdown_preview(["rank", "trial"] + param_names + ["n_params", "val_ref_loss"], md_rows)

    # ------------------------------------------------------------------ 3
    # fANOVA importances (needs >= 2 complete trials and some variation)
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as exc:
        print(f"importance evaluation skipped: {exc}")
        importances = None

    if importances:
        lines = [
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Hyperparameter & Importance \\",
            r"\midrule",
        ]
        md_rows = []
        for name, value in importances.items():
            label = PARAM_INFO.get(name, {}).get("label", name)
            lines.append(f"{label} & {value:.3f} " + r"\\")
            md_rows.append([name, f"{value:.3f}"])
        lines += [r"\bottomrule", r"\end{tabular}"]
        write_table(OUT_DIR / f"table_importance_{suffix}.tex", lines, "fANOVA importances")
        markdown_preview(["parameter", "importance"], md_rows)


if __name__ == "__main__":
    main()
