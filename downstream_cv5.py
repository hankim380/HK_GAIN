import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def load_arrays(run_dir: str):
    imp_dir = os.path.join(run_dir, "imputed_data")
    if not os.path.isdir(imp_dir):
        raise FileNotFoundError(f"Cannot find imputed_data/ under: {run_dir}")

    X_orig_path = os.path.join(imp_dir, "original.npy")
    y_path = os.path.join(imp_dir, "labels.npy")

    if not os.path.exists(X_orig_path):
        raise FileNotFoundError(f"Missing: {X_orig_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing: {y_path}")

    X_orig = np.load(X_orig_path)
    y = np.load(y_path)

    X_imp_path = os.path.join(imp_dir, "imputed.npy")
    X_imp = np.load(X_imp_path) if os.path.exists(X_imp_path) else None

    # basic sanity checks
    if X_orig.ndim != 2:
        raise ValueError(f"original.npy must be 2D, got shape={X_orig.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if len(X_orig) != len(y):
        raise ValueError(f"Row mismatch: X_orig={len(X_orig)} vs y={len(y)}")
    if X_imp is not None and len(X_imp) != len(y):
        raise ValueError(f"Row mismatch: X_imp={len(X_imp)} vs y={len(y)}")

    return X_orig, X_imp, y


def build_model(model_name: str, seed: int):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=100, random_state=seed))
        ])
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1
        )
    else:
        raise ValueError("model must be one of: logreg, rf")


def save_confusion_matrix_png(cm, save_path, title="Confusion Matrix (sum over folds)"):
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


def run_cv5(X, y, model_name: str, seed: int, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    per_fold = []
    cm_sum = None

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = build_model(model_name, seed)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)

        # AUROC needs probabilities; if not available, NaN
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_te, y_prob)
        else:
            auroc = float("nan")

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)

        cm = confusion_matrix(y_te, y_pred)
        cm_sum = cm if cm_sum is None else (cm_sum + cm)

        per_fold.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "accuracy": float(acc),
            "f1": float(f1),
            "auroc": float(auroc),
        })

    df = pd.DataFrame(per_fold)

    summary = pd.DataFrame([{
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=1)),
        "f1_mean": float(df["f1"].mean()),
        "f1_std": float(df["f1"].std(ddof=1)),
        "auroc_mean": float(df["auroc"].mean()),
        "auroc_std": float(df["auroc"].std(ddof=1)),
    }])

    return df, summary, cm_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--also_eval_imputed", action="store_true",
                        help="If set, evaluate imputed.npy with the SAME CV splits (same seed/shuffle).")
    args = parser.parse_args()

    X_orig, X_imp, y = load_arrays(args.run_dir)

    print(f"[Loaded]")
    print(f"  X_orig: {X_orig.shape}")
    print(f"  y     : {y.shape} (pos rate={y.mean():.3f})")
    if X_imp is not None:
        print(f"  X_imp : {X_imp.shape}")

    # --------------------
    # ORIGINAL CV5
    # --------------------
    out_orig = os.path.join(args.run_dir, "downstream_original_cv5")
    os.makedirs(out_orig, exist_ok=True)

    df_o, summ_o, cm_o = run_cv5(
        X_orig, y,
        model_name=args.model,
        seed=args.seed,
        n_splits=args.folds
    )

    df_o.to_csv(os.path.join(out_orig, "metrics_per_fold.csv"), index=False)
    summ_o.to_csv(os.path.join(out_orig, "metrics_summary.csv"), index=False)
    save_confusion_matrix_png(cm_o, os.path.join(out_orig, "confusion_matrix_sum.png"),
                              title=f"ORIGINAL Confusion Matrix (sum over {args.folds} folds)")

    print("\n=== ORIGINAL CV Summary ===")
    print(summ_o.to_string(index=False))
    print(f"Saved -> {out_orig}")

    # --------------------
    # IMPUTED CV5 (optional)
    # --------------------
    if args.also_eval_imputed:
        if X_imp is None:
            raise FileNotFoundError("imputed.npy not found, but --also_eval_imputed was set.")

        out_imp = os.path.join(args.run_dir, "downstream_imputed_cv5")
        os.makedirs(out_imp, exist_ok=True)

        # IMPORTANT: same seed + same folds + same shuffle => same splits as original
        df_i, summ_i, cm_i = run_cv5(
            X_imp, y,
            model_name=args.model,
            seed=args.seed,
            n_splits=args.folds
        )

        df_i.to_csv(os.path.join(out_imp, "metrics_per_fold.csv"), index=False)
        summ_i.to_csv(os.path.join(out_imp, "metrics_summary.csv"), index=False)
        save_confusion_matrix_png(cm_i, os.path.join(out_imp, "confusion_matrix_sum.png"),
                                  title=f"IMPUTED Confusion Matrix (sum over {args.folds} folds)")

        print("\n=== IMPUTED CV Summary ===")
        print(summ_i.to_string(index=False))
        print(f"Saved -> {out_imp}")


if __name__ == "__main__":
    main()