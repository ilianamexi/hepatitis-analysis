"""Hepatitis C stage classification from routine blood tests (UCI "HCV data", 615 patients).

What this script does
1. Downloads the UCI HCV dataset (if data/hcvdat0.csv is missing).
2. Describes the data: class counts, missing values, biomarker distributions by stage.
3. Measures how each biomarker relates to disease stage with Spearman's rank correlation,
   using only the ordered stages (0 donor < 1 hepatitis < 2 fibrosis < 3 cirrhosis). Suspect
   donors are not a stage, so they are left out of this step.
4. Compares classifiers with repeated stratified cross-validation. Imputation, scaling and
   SMOTE are inside the pipeline, so they are fitted on the training folds only.
5. Reports accuracy, balanced accuracy, macro-F1 and per-class recall, a confusion matrix
   and permutation feature importance computed on held-out folds.

Usage:  python hepatitis_analysis.py          (writes results/ and figures/)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "hcvdat0.csv"
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv",
    "https://archive.ics.uci.edu/static/public/571/hcv+data.zip",
]
BIOMARKERS = ["ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]
# Readable class names, in clinical order
CLASS_NAMES = {
    "0=Blood Donor": "Blood donor",
    "0s=suspect Blood Donor": "Suspect donor",
    "1=Hepatitis": "Hepatitis",
    "2=Fibrosis": "Fibrosis",
    "3=Cirrhosis": "Cirrhosis",
}
CLASS_ORDER = list(CLASS_NAMES.values())
STAGE = {"Blood donor": 0, "Hepatitis": 1, "Fibrosis": 2, "Cirrhosis": 3}

# Colours: one hue per model, fixed order; text stays neutral.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8d5bd6"


# ----------------------------------------------------------------------------- data
def download() -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    name = next(n for n in z.namelist() if n.endswith(".csv"))
                    raw = z.read(name)
            DATA.write_bytes(raw)
            print(f"downloaded {url}")
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = e
    raise RuntimeError(f"Could not download the HCV dataset ({last}). "
                       f"Download hcvdat0.csv from the UCI repository and save it as {DATA}.")


def load() -> pd.DataFrame:
    if not DATA.exists():
        download()
    df = pd.read_csv(DATA, index_col=0)
    df["Class"] = df["Category"].map(CLASS_NAMES)
    if df["Class"].isna().any():
        raise ValueError(f"Unexpected categories: {df.loc[df['Class'].isna(), 'Category'].unique()}")
    df["Sex_male"] = (df["Sex"] == "m").astype(int)
    return df


# ----------------------------------------------------------------------------- style
def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


# ----------------------------------------------------------------------------- analysis
def describe(df: pd.DataFrame) -> dict:
    counts = df["Class"].value_counts().reindex(CLASS_ORDER)
    missing = df[BIOMARKERS].isna().sum()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(CLASS_ORDER[::-1], counts[::-1], color=BLUE, height=0.6)
    for i, v in enumerate(counts[::-1]):
        ax.text(v + 5, i, str(v), va="center", color=INK)
    ax.set(xlabel="Patients", title=f"Class balance: {counts.iloc[0]} of {len(df)} are healthy donors")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "class_distribution.png", dpi=160)
    plt.close(fig)
    return {"n": int(len(df)), "class_counts": counts.to_dict(), "missing": missing[missing > 0].to_dict()}


def stage_correlation(df: pd.DataFrame) -> pd.DataFrame:
    staged = df[df["Class"].isin(STAGE)].copy()
    staged["stage"] = staged["Class"].map(STAGE)
    rows = []
    for m in BIOMARKERS:
        d = staged[[m, "stage"]].dropna()
        rho, p = spearmanr(d[m], d["stage"])
        rows.append({"biomarker": m, "spearman_rho": rho, "p_value": p, "n": len(d)})
    out = pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)

    # Distributions of the six markers most related to stage (log scale: enzymes are skewed)
    top = out["biomarker"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    order = ["Blood donor", "Hepatitis", "Fibrosis", "Cirrhosis"]
    for ax, m in zip(axes.ravel(), top):
        data = [staged.loc[staged["Class"] == c, m].dropna() for c in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=BLUE, alpha=0.18, edgecolor=BLUE)
        for line in bp["medians"]:
            line.set(color=BLUE, linewidth=2)
        for line in bp["whiskers"] + bp["caps"]:
            line.set(color=MUTED, linewidth=1)
        ax.set_xticks(range(1, 5), ["Donor", "Hepatitis", "Fibrosis", "Cirrhosis"], fontsize=8.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        # label the 2x and 5x minor ticks too, so narrow ranges (e.g. CHOL) still get numbers
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
        rho = out.set_index("biomarker").loc[m, "spearman_rho"]
        ax.set_title(f"{m}  (Spearman ρ = {rho:+.2f})")
    fig.suptitle("Biomarkers most related to disease stage (outliers hidden, log scale)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "biomarkers_by_stage.png", dpi=160)
    plt.close(fig)
    return out


def models() -> dict:
    imp = ("impute", SimpleImputer(strategy="median"))
    return {
        "Logistic regression": Pipeline([imp, ("scale", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        "LDA": Pipeline([imp, ("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
        "Random forest": Pipeline([imp, ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED))]),
        "Random forest + SMOTE": Pipeline([imp, ("smote", SMOTE(k_neighbors=2, random_state=SEED)),
                                           ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED))]),
    }


def evaluate(X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    labels = np.arange(len(CLASS_ORDER))
    rows, pooled = [], {}
    for name, pipe in models().items():
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for split, (tr, te) in enumerate(cv.split(X, y)):
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict(X.iloc[te])
            rec = recall_score(y[te], p, labels=labels, average=None, zero_division=0)
            rows.append({"model": name, "split": split, "accuracy": accuracy_score(y[te], p),
                         "balanced_accuracy": balanced_accuracy_score(y[te], p),
                         "macro_f1": f1_score(y[te], p, average="macro", labels=labels, zero_division=0),
                         **{f"recall_{c}": r for c, r in zip(CLASS_ORDER, rec)}})
            cm += confusion_matrix(y[te], p, labels=labels)
        pooled[name] = cm
        print(f"  {name}: done")
    return pd.DataFrame(rows), pooled


def plot_confusion(cm: np.ndarray, name: str, n_repeats: int):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(norm)):
        for j in range(len(norm)):
            ax.text(j, i, f"{100 * norm[i, j]:.0f}%", ha="center", va="center",
                    color="white" if norm[i, j] > 0.6 else INK, fontsize=9)
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"{name}: confusion matrix\n(row %, pooled over {n_repeats}×5 held-out folds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def held_out_importance(X: pd.DataFrame, y: np.ndarray, pipe_name: str) -> pd.DataFrame:
    """Permutation importance (drop in macro-F1) measured on each held-out fold, averaged."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in cv.split(X, y):
        pipe = models()[pipe_name].fit(X.iloc[tr], y[tr])
        r = permutation_importance(pipe, X.iloc[te], y[te], scoring="f1_macro", n_repeats=20,
                                   random_state=SEED)
        scores.append(r.importances_mean)
    imp = pd.DataFrame({"feature": X.columns, "importance": np.mean(scores, axis=0),
                        "std_across_folds": np.std(scores, axis=0)}).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    d = imp.iloc[::-1]
    ax.barh(d["feature"], d["importance"], xerr=d["std_across_folds"], color=BLUE, height=0.6,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.set(xlabel="Drop in macro-F1 when the feature is shuffled",
           title=f"Which tests matter most ({pipe_name}, held-out folds)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=160)
    plt.close(fig)
    return imp


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    style()
    df = load()
    info = describe(df)
    print(f"{info['n']} patients; classes {info['class_counts']}; missing {info['missing']}")

    corr = stage_correlation(df)
    corr.to_csv(RESULTS / "stage_correlation.csv", index=False)

    X = df[["Age", "Sex_male"] + BIOMARKERS]
    y = df["Class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    n_repeats = 10
    folds, pooled = evaluate(X, y, n_repeats)
    folds.to_csv(RESULTS / "cv_folds.csv", index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASS_ORDER]
    summary = folds.groupby("model", sort=False)[metric_cols].agg(["mean", "std"])
    summary.to_csv(RESULTS / "cv_summary.csv")
    best = summary[("macro_f1", "mean")].idxmax()
    plot_confusion(pooled[best], best, n_repeats)
    imp = held_out_importance(X, y, best)
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)

    # Markdown summary
    lines = [f"Dataset: {info['n']} patients. Classes: " + ", ".join(f"{k} {v}" for k, v in info["class_counts"].items()),
             f"Missing values (imputed with the training-fold median): {info['missing']}", "",
             "Spearman correlation with stage (donor 0 → cirrhosis 3, suspect donors excluded):", "",
             "| Biomarker | ρ | p-value |", "|---|---|---|"]
    lines += [f"| {r.biomarker} | {r.spearman_rho:+.2f} | {r.p_value:.1e} |" for r in corr.itertuples()]
    lines += ["", f"Repeated stratified 5-fold CV × {n_repeats} (mean ± std, %):", "",
              "| Model | Accuracy | Balanced acc. | Macro-F1 | " + " | ".join(f"Recall {c}" for c in CLASS_ORDER) + " |",
              "|---" * (4 + len(CLASS_ORDER)) + "|"]
    for model in summary.index:
        s = summary.loc[model]
        cells = [f"{100 * s[(c, 'mean')]:.1f} ± {100 * s[(c, 'std')]:.1f}" for c in metric_cols[:3]]
        cells += [f"{100 * s[(c, 'mean')]:.0f}" for c in metric_cols[3:]]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", f"Best by macro-F1: {best}. Top held-out permutation importances: "
              + ", ".join(f"{r.feature} ({r.importance:.3f})" for r in imp.head(5).itertuples())]
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
