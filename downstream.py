import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)


def load_arrays(run_dir: str):
    imp_dir = os.path.join(run_dir, "imputed_data")
    if not os.path.isdir(imp_dir):
        raise FileNotFoundError(f"Cannot find imputed_data/ under: {run_dir}")

    X_orig = np.load(os.path.join(imp_dir, "original.npy"))
    y = np.load(os.path.join(imp_dir, "labels.npy"))

    X_imp_path = os.path.join(imp_dir, "imputed.npy")
    X_imp = np.load(X_imp_path) if os.path.exists(X_imp_path) else None

    return X_orig, X_imp, y


def build_model(model_name: str, seed: int):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000))
        ])
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError("model_name must be one of: logreg, rf")


def eval_model_with_indices(X, y, idx_tr, idx_te, model_name, seed):
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    clf = build_model(model_name, seed)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te)[:, 1]
        auroc = roc_auc_score(y_te, y_prob)
    else:
        auroc = float("nan")

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=4)

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "auroc": float(auroc),
    }
    return metrics, cm, report


def save_confusion_matrix_png(cm, save_path, title="Confusion Matrix"):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_outputs(save_dir, metrics, cm, report):
    os.makedirs(save_dir, exist_ok=True)

    # metrics.csv
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    # confusion_matrix.png
    save_confusion_matrix_png(
        cm,
        os.path.join(save_dir, "confusion_matrix.png"),
        title="Confusion Matrix"
    )

    # classification_report.txt
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--also_eval_imputed", action="store_true")
    args = parser.parse_args()

    X_orig, X_imp, y = load_arrays(args.run_dir)

    print(f"[Loaded]")
    print(f"  X_orig: {X_orig.shape}")
    print(f"  y     : {y.shape} (pos rate={y.mean():.3f})")
    if X_imp is not None:
        print(f"  X_imp : {X_imp.shape}")

    # --- single stratified split ---
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # --- ORIGINAL eval + save ---
    m_orig, cm_orig, rep_orig = eval_model_with_indices(
        X_orig, y, idx_tr, idx_te, args.model, args.seed
    )
    print("\n=== ORIGINAL ===")
    print(m_orig)
    print(rep_orig)

    save_dir_orig = os.path.join(args.run_dir, "downstream_original")
    save_outputs(save_dir_orig, m_orig, cm_orig, rep_orig)
    print(f"Saved ORIGINAL results -> {save_dir_orig}")

    # --- IMPUTED eval + save (optional) ---
    if args.also_eval_imputed:
        if X_imp is None:
            raise FileNotFoundError("imputed.npy not found but --also_eval_imputed was set.")

        m_imp, cm_imp, rep_imp = eval_model_with_indices(
            X_imp, y, idx_tr, idx_te, args.model, args.seed
        )
        print("\n=== IMPUTED ===")
        print(m_imp)
        print(rep_imp)

        save_dir_imp = os.path.join(args.run_dir, "downstream_imputed")
        save_outputs(save_dir_imp, m_imp, cm_imp, rep_imp)
        print(f"Saved IMPUTED results -> {save_dir_imp}")


if __name__ == "__main__":
    main()