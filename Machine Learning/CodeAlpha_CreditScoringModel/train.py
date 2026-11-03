"""
Credit scoring model
====================
Predicts whether a customer will default, using their financial history.

    python train.py                  # UCI credit-card default data (bundled in ./data)
    python train.py --quick          # smaller grids + 3-fold CV, finishes in a few minutes
    python train.py --source csv --data cs-training.csv --target SeriousDlqin2yrs   # your own data

Pipeline (one scikit-learn Pipeline, so it is tuned, evaluated and saved as ONE object):
    raw applicant data -> feature engineering -> impute / scale / one-hot -> classifier

Four models are compared - Logistic Regression, Decision Tree, Random Forest and Gradient
Boosting - using 5-fold stratified cross-validation, then scored on a held-out test set with
Precision, Recall, F1 and ROC-AUC.
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
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
                             accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_dataset
from features import build_pipeline
from scorecard import probability_to_score

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


# --------------------------------------------------------------------------- models
def candidate_models(quick: bool):
    """name -> (estimator, hyper-parameter grid). 'model__' targets the classifier step."""
    zoo = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [20, 50, 100]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100 if quick else 200, random_state=SEED, n_jobs=-1),
            {"model__max_depth": [8, 12], "model__min_samples_leaf": [10, 30]}),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
             "model__l2_regularization": [0.0, 1.0]}),
    }
    if quick:  # keep only the first two values of each parameter
        zoo = {k: (est, {p: v[:2] for p, v in grid.items()}) for k, (est, grid) in zoo.items()}
    return zoo


# --------------------------------------------------------------------------- evaluation
def best_f1_threshold(y_true, proba) -> float:
    """Probability cut-off that maximises F1 of the 'default' class (found on training data only)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics_at(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def r4(x):
    return round(float(x), 4)


# --------------------------------------------------------------------------- plots
def plot_roc(y_test, probas, path):
    plt.figure(figsize=(6, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the test set")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(y_test, proba, threshold, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= threshold).astype(int), display_labels=["Repaid", "Defaulted"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{title}\nthreshold = {threshold:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(pipe, X_test, y_test, path, top=15):
    """Permutation importance on the engineered features (names include the new ones)."""
    Xt = pipe[:-1].transform(X_test)
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    result = permutation_importance(pipe.named_steps["model"], Xt, y_test, scoring="roc_auc",
                                    n_repeats=5, random_state=SEED)
    imp = pd.Series(result.importances_mean, index=names).sort_values().tail(top)
    plt.figure(figsize=(7, 5))
    plt.barh(imp.index, imp.values, color="#3b7dd8")
    plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
    plt.title("Most important features (best model)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return imp.sort_values(ascending=False)


def plot_scores(y_test, proba, path):
    scores = probability_to_score(proba)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(300, 850, 45)
    plt.hist(scores[y_test.values == 0], bins=bins, alpha=0.6, label="Repaid", color="#2a9d8f")
    plt.hist(scores[y_test.values == 1], bins=bins, alpha=0.6, label="Defaulted", color="#e76f51")
    plt.xlabel("Credit score (300-850)")
    plt.ylabel("Customers")
    plt.title("Score distribution - good vs bad customers (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train and evaluate credit-scoring models.")
    ap.add_argument("--source", default="credit_default", choices=["credit_default", "csv"])
    ap.add_argument("--data", help="path to a CSV file (with --source csv)")
    ap.add_argument("--target", help="name of the 0/1 default column (with --source csv)")
    ap.add_argument("--positive-label", help="value meaning 'default' if the target is text, e.g. bad")
    ap.add_argument("--keep-protected", action="store_true",
                    help="keep sex / marital status as model inputs (dropped by default for fairness)")
    ap.add_argument("--quick", action="store_true", help="smaller grids and 3-fold CV")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_dataset(args.source, args.data, args.target, args.positive_label)
    print(f"Loaded {len(X):,} customers, {X.shape[1]} raw features, "
          f"default rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=3 if args.quick else 5, shuffle=True, random_state=SEED)

    rows, fitted, probas = [], {}, {}
    for name, (estimator, grid) in candidate_models(args.quick).items():
        pipe = build_pipeline(estimator, drop_protected=not args.keep_protected)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # choose the decision threshold from out-of-fold predictions on the TRAINING data
        oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold = best_f1_threshold(y_train, oof)

        proba = best.predict_proba(X_test)[:, 1]
        tuned, default = metrics_at(y_test, proba, threshold), metrics_at(y_test, proba, 0.5)
        rows.append({
            "model": name,
            "cv_roc_auc": r4(search.best_score_),
            "test_roc_auc": r4(roc_auc_score(y_test, proba)),
            "test_pr_auc": r4(average_precision_score(y_test, proba)),
            "threshold": round(threshold, 2),
            "accuracy": r4(tuned["accuracy"]), "precision": r4(tuned["precision"]),
            "recall": r4(tuned["recall"]), "f1": r4(tuned["f1"]),
            "precision@0.5": r4(default["precision"]), "recall@0.5": r4(default["recall"]),
            "f1@0.5": r4(default["f1"]),
            "best_params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()}),
        })
        fitted[name], probas[name] = best, proba
        print(f"  {name:<20} CV AUC {search.best_score_:.3f} | test AUC {rows[-1]['test_roc_auc']:.3f} | "
              f"thr {threshold:.2f}: precision {tuned['precision']:.3f} recall {tuned['recall']:.3f} "
              f"F1 {tuned['f1']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # the winner is picked by cross-validation on the training data, never by the test set
    best_row = table.loc[table["cv_roc_auc"].idxmax()]
    best_name, best_pipe = best_row["model"], fitted[best_row["model"]]
    threshold = float(best_row["threshold"])
    print(f"\nBest model (by CV ROC-AUC): {best_name}  |  decision threshold {threshold:.2f}")

    plot_roc(y_test, probas, OUT_DIR / "roc_curves.png")
    plot_confusion(y_test, probas[best_name], threshold, best_name, OUT_DIR / "confusion_matrix.png")
    plot_scores(y_test, probas[best_name], OUT_DIR / "score_distribution.png")
    importance = plot_importance(best_pipe, X_test, y_test, OUT_DIR / "feature_importance.png")
    print("Top features:", ", ".join(importance.index[:6]))

    final = metrics_at(y_test, probas[best_name], threshold)
    summary = {"best_model": best_name, "threshold": threshold,
               "test_roc_auc": float(best_row["test_roc_auc"]), "test_pr_auc": float(best_row["test_pr_auc"]),
               **{k: r4(v) if isinstance(v, float) else v for k, v in final.items()},
               "top_features": list(importance.index[:8]),
               "n_train": len(X_train), "n_test": len(X_test), "default_rate": r4(y.mean())}
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Model | CV ROC-AUC | Test ROC-AUC | Precision | Recall | F1 | Threshold |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['model']} | {r['cv_roc_auc']:.3f} | {r['test_roc_auc']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['threshold']:.2f} |")
    (OUT_DIR / "results_summary.md").write_text("\n".join(lines) + "\n")

    joblib.dump({"pipeline": best_pipe, "model_name": best_name, "threshold": threshold,
                 "input_columns": list(X.columns), "source": args.source, "data": args.data,
                 "target": args.target, "positive_label": args.positive_label,
                 "metrics": summary}, MODEL_DIR / "credit_model.joblib")
    print(f"\nSaved: outputs/ (tables + plots) and models/credit_model.joblib")


if __name__ == "__main__":
    main()
