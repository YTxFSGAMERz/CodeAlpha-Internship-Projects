"""
Disease prediction from structured medical data
===============================================
Trains and compares four classifiers - Logistic Regression, SVM, Random Forest and
XGBoost - on three UCI datasets (heart disease, diabetes, breast cancer).

    python train.py --dataset heart
    python train.py --dataset all              # heart + diabetes + breast_cancer
    python train.py --dataset diabetes --quick # smaller hyper-parameter grids

For every dataset the script
  1. splits the data 80 / 20 (stratified, so both parts keep the disease ratio),
  2. builds a Pipeline  (impute -> scale / one-hot -> model)  so no information from the
     test set leaks into preprocessing,
  3. tunes every model with 5-fold stratified GridSearchCV,
  4. evaluates on the untouched test set (accuracy, precision, recall, F1, ROC-AUC),
  5. saves a comparison table, ROC curves, confusion matrix, feature importance and the
     best model (models/<dataset>_model.joblib).
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from data_loader import DATASETS, load_dataset

try:
    from xgboost import XGBClassifier
except ImportError:  # the project still runs (with 3 models) if XGBoost is missing
    XGBClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


# --------------------------------------------------------------------------- preprocessing
def build_preprocessor(numeric, categorical):
    """Median-impute + standardise numbers; mode-impute + one-hot encode categories."""
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                sparse_output=False))]), categorical))
    return ColumnTransformer(parts)


# --------------------------------------------------------------------------- models
def shrink(grid):
    """Keep only the first two values of every parameter (used by --quick)."""
    if isinstance(grid, list):
        return [shrink(g) for g in grid]
    return {k: v[:2] for k, v in grid.items()}


def build_model_zoo(pos_weight: float, quick: bool):
    """name -> (estimator, hyper-parameter grid).  'model__' targets the pipeline step."""
    zoo = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=SEED),
            [{"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
             {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.1]}],
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 4, 8],
             "model__min_samples_leaf": [1, 3]},
        ),
    }
    if XGBClassifier is not None:
        zoo["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                          random_state=SEED, n_jobs=1, verbosity=0),
            {"model__n_estimators": [100, 300], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.1], "model__subsample": [0.8, 1.0]},
        )
    if quick:
        zoo = {k: (est, shrink(grid)) for k, (est, grid) in zoo.items()}
    return zoo


# --------------------------------------------------------------------------- evaluation
def evaluate(pipe, X_test, y_test, threshold=0.5):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),      # sensitivity
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in metrics.items()}, proba


def plot_roc(y_test, probas, title, path):
    plt.figure(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False positive rate (1 - specificity)")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title(f"ROC curves - {title}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_test, proba, title, path):
    pred = (proba >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No disease", "Disease"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix - {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(pipe, X_test, y_test, title, path, top=10):
    """Permutation importance: how much worse does the model get when one feature is shuffled?

    Log-loss is used instead of ROC-AUC because AUC is already ~1.0 on some of these small
    test sets, which makes AUC-based importances very noisy."""
    result = permutation_importance(pipe, X_test, y_test, scoring="neg_log_loss",
                                    n_repeats=30, random_state=SEED)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values().tail(top)
    err = pd.Series(result.importances_std, index=X_test.columns)[imp.index]
    plt.figure(figsize=(6, 4.5))
    plt.barh(imp.index, imp.values, xerr=err.values, color="#3b7dd8")
    plt.xlabel("Increase in log-loss when the feature is shuffled")
    plt.title(f"Feature importance - {title}", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return imp.sort_values(ascending=False)


# --------------------------------------------------------------------------- one dataset
def run_dataset(name: str, args) -> dict:
    X, y, info = load_dataset(name)
    title = info["title"]
    print(f"\n{'=' * 78}\n{title}: {X.shape[0]} patients, {X.shape[1]} features, "
          f"{y.mean():.1%} have the disease\n{'=' * 78}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    rows, fitted, probas = [], {}, {}
    for model_name, (estimator, grid) in build_model_zoo(pos_weight, args.quick).items():
        pipe = Pipeline([("prep", build_preprocessor(info["numeric"], info["categorical"])),
                         ("model", estimator)])
        search = GridSearchCV(pipe, grid, scoring=args.scoring, cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        metrics, proba = evaluate(search.best_estimator_, X_test, y_test)
        cv_std = search.cv_results_["std_test_score"][search.best_index_]
        rows.append({"model": model_name,
                     f"cv_{args.scoring}": round(search.best_score_, 4),
                     f"cv_{args.scoring}_std": round(cv_std, 4),
                     **metrics,
                     "best_params": json.dumps({k.replace("model__", ""): v
                                                for k, v in search.best_params_.items()})})
        fitted[model_name], probas[model_name] = search.best_estimator_, proba
        print(f"  {model_name:<20} CV {args.scoring} = {search.best_score_:.3f}   |  test: "
              f"acc {metrics['accuracy']:.3f}  prec {metrics['precision']:.3f}  "
              f"recall {metrics['recall']:.3f}  F1 {metrics['f1']:.3f}  AUC {metrics['roc_auc']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(out / "model_comparison.csv", index=False)

    # the best model is chosen on cross-validation score (never on the test set)
    best_row = table.loc[table[f"cv_{args.scoring}"].idxmax()]
    best_name = best_row["model"]
    best_pipe = fitted[best_name]
    print(f"\n  Best model (by CV {args.scoring}): {best_name}")

    plot_roc(y_test, probas, title, out / "roc_curves.png")
    plot_confusion(y_test, probas[best_name], f"{best_name}\n{title}", out / "confusion_matrix.png")
    importance = plot_importance(best_pipe, X_test, y_test, f"{best_name} - {title}",
                                 out / "feature_importance.png")
    print("  Most important features:", ", ".join(importance.index[:5]))

    best_metrics = {k: best_row[k] for k in ["accuracy", "precision", "recall", "specificity",
                                              "f1", "roc_auc", "tn", "fp", "fn", "tp"]}
    best_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                    for k, v in best_metrics.items()}
    bundle = {"pipeline": best_pipe, "dataset": name, "title": title, "model_name": best_name,
              "features": list(X.columns), "threshold": 0.5, "metrics": best_metrics}
    joblib.dump(bundle, MODEL_DIR / f"{name}_model.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump({"dataset": title, "best_model": best_name, "test_metrics": best_metrics,
                   "top_features": list(importance.index[:5]),
                   "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)

    return {"dataset": title, "best_model": best_name, **{k: best_metrics[k] for k in
            ["accuracy", "precision", "recall", "f1", "roc_auc"]}, "table": table}


def write_summary(results):
    """Markdown summary of every dataset (paste it into the README / video)."""
    lines = []
    for r in results:
        lines += [f"### {r['dataset']}", "",
                  "| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | Test ROC-AUC |",
                  "|---|---|---|---|---|---|---|"]
        cv_col = [c for c in r["table"].columns if c.startswith("cv_") and not c.endswith("_std")][0]
        for _, row in r["table"].iterrows():
            lines.append(f"| {row['model']} | {row[cv_col]:.3f} | {row['accuracy']:.3f} | "
                         f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
                         f"{row['roc_auc']:.3f} |")
        lines += ["", f"Best model (by cross-validation): **{r['best_model']}**", ""]
    (OUT_DIR / "results_summary.md").write_text("\n".join(lines))
    pd.DataFrame([{k: v for k, v in r.items() if k != "table"} for r in results]).to_csv(
        OUT_DIR / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Train disease-prediction models.")
    parser.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    parser.add_argument("--scoring", default="roc_auc",
                        choices=["roc_auc", "recall", "f1", "accuracy"],
                        help="metric used to tune and to pick the best model (default: roc_auc)")
    parser.add_argument("--quick", action="store_true", help="use much smaller hyper-parameter grids")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    OUT_DIR.mkdir(exist_ok=True)
    results = [run_dataset(n, args) for n in names]
    write_summary(results)
    print(f"\nDone. Results are in '{OUT_DIR.name}/', trained models in '{MODEL_DIR.name}/'.")


if __name__ == "__main__":
    main()
