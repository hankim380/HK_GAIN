import os
import json
import argparse
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_inputs(run_dir):
    imp_dir = os.path.join(run_dir, "imputed_data")
    if not os.path.isdir(imp_dir):
        raise FileNotFoundError(f"Cannot find imputed_data dir: {imp_dir}")

    paths = {
        "gain_imputed": os.path.join(imp_dir, "imputed.npy"),
        "filled_only": os.path.join(imp_dir, "missing_filled.npy"),
        "filled_plus_indicator": os.path.join(imp_dir, "missing_indicator_concat.npy"),
        "indicator_only": os.path.join(imp_dir, "missing_indicator.npy"),
        "y": os.path.join(imp_dir, "labels.npy"),
        "ids": os.path.join(imp_dir, "ids.npy"),
    }

    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            "Missing required files in run_dir/imputed_data: " + ", ".join(missing)
        )

    X = {k: np.load(v) for k, v in paths.items() if k not in ["y", "ids"]}
    y = np.load(paths["y"])
    ids = np.load(paths["ids"])

    # y shape 정리: (n,)로 강제
    y = np.array(y).reshape(-1)
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"labels.npy must be binary (0/1). got unique={np.unique(y)}")

    # 길이 검증
    n = len(y)
    if len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != y length {n}")

    for name, Xi in X.items():
        if Xi.shape[0] != n:
            raise ValueError(f"{name} n mismatch: {Xi.shape[0]} != {n}")

    return X, y, ids


def make_model(C=1.0, max_iter=2000, random_state=42):
    """
    공정 비교를 위해 모든 condition에 동일 모델 사용.
    - SimpleImputer: 혹시 NaN이 남아있어도 안전 (대부분은 이미 fill됨)
    - StandardScaler: 로지스틱 회귀 성능 안정화
    - LogisticRegression: 확률 출력 가능(ROC-AUC, PR-AUC)
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state
        ))
    ])


def eval_one_condition(name, X, y, skf, model_kwargs, out_dir):
    fold_rows = []
    y_all = []
    p_all = []
    pred_all = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = make_model(**model_kwargs)
        model.fit(X[tr], y[tr])

        # proba: positive class prob
        p = model.predict_proba(X[te])[:, 1]
        pred = (p >= 0.5).astype(int)

        # fold metrics
        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], p)
        ap = average_precision_score(y[te], p)
        f1 = f1_score(y[te], pred)
        prec = precision_score(y[te], pred, zero_division=0)
        rec = recall_score(y[te], pred, zero_division=0)

        fold_rows.append({
            "condition": name,
            "fold": fold,
            "n_test": int(len(te)),
            "acc": float(acc),
            "roc_auc": float(auc),
            "pr_auc": float(ap),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
        })

        y_all.append(y[te])
        p_all.append(p)
        pred_all.append(pred)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    pred_all = np.concatenate(pred_all)

    # aggregate metrics
    agg = {
        "condition": name,
        "acc_mean": float(np.mean([r["acc"] for r in fold_rows])),
        "acc_std": float(np.std([r["acc"] for r in fold_rows])),
        "roc_auc_mean": float(np.mean([r["roc_auc"] for r in fold_rows])),
        "roc_auc_std": float(np.std([r["roc_auc"] for r in fold_rows])),
        "pr_auc_mean": float(np.mean([r["pr_auc"] for r in fold_rows])),
        "pr_auc_std": float(np.std([r["pr_auc"] for r in fold_rows])),
        "f1_mean": float(np.mean([r["f1"] for r in fold_rows])),
        "f1_std": float(np.std([r["f1"] for r in fold_rows])),
        "precision_mean": float(np.mean([r["precision"] for r in fold_rows])),
        "recall_mean": float(np.mean([r["recall"] for r in fold_rows])),
    }

    # confusion matrix over all folds (out-of-fold)
    cm = confusion_matrix(y_all, pred_all)
    agg["confusion_matrix"] = cm.tolist()

    # save cm plot
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix (OOF) - {name}")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    fig_path = os.path.join(out_dir, f"cm_{name}.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # save OOF predictions (for later analysis)
    np.save(os.path.join(out_dir, f"oof_y_{name}.npy"), y_all)
    np.save(os.path.join(out_dir, f"oof_p_{name}.npy"), p_all)
    np.save(os.path.join(out_dir, f"oof_pred_{name}.npy"), pred_all)

    return fold_rows, agg


def save_csv(rows, path):
    # no pandas dependency
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g. _outputs/xxx_both_YYYYMMDD_HHMMSS")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=2000)
    args = ap.parse_args()

    X_dict, y, ids = load_inputs(args.run_dir)

    out_dir = os.path.join(args.run_dir, "downstream_cv5")
    ensure_dir(out_dir)

    # 같은 fold split을 4조건 모두에 재사용하기 위해 skf를 한 번만 정의
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    model_kwargs = dict(C=args.C, max_iter=args.max_iter, random_state=args.seed)

    conditions = [
        ("gain_imputed", X_dict["gain_imputed"]),
        ("filled_only", X_dict["filled_only"]),
        ("filled_plus_indicator", X_dict["filled_plus_indicator"]),
        ("indicator_only", X_dict["indicator_only"]),
    ]

    all_fold_rows = []
    all_aggs = []

    for name, X in conditions:
        fold_rows, agg = eval_one_condition(
            name=name,
            X=X,
            y=y,
            skf=skf,
            model_kwargs=model_kwargs,
            out_dir=out_dir
        )
        all_fold_rows.extend(fold_rows)
        all_aggs.append(agg)

    # 저장 1) fold별 결과
    save_csv(all_fold_rows, os.path.join(out_dir, "fold_metrics.csv"))

    # 저장 2) condition별 요약
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": args.run_dir,
                "kfold": args.kfold,
                "seed": args.seed,
                "model": "LogisticRegression(StandardScaler)",
                "model_params": model_kwargs,
                "summary": all_aggs,
            },
            f,
            indent=2
        )

    # 보기 좋은 summary table (csv)
    summary_rows = []
    for a in all_aggs:
        summary_rows.append({
            "condition": a["condition"],
            "roc_auc_mean": a["roc_auc_mean"],
            "roc_auc_std": a["roc_auc_std"],
            "pr_auc_mean": a["pr_auc_mean"],
            "pr_auc_std": a["pr_auc_std"],
            "acc_mean": a["acc_mean"],
            "acc_std": a["acc_std"],
            "f1_mean": a["f1_mean"],
            "f1_std": a["f1_std"],
        })
    save_csv(summary_rows, os.path.join(out_dir, "summary_table.csv"))

    # 콘솔 출력
    print("\n=== Downstream 4-condition summary (mean ± std over folds) ===")
    for r in summary_rows:
        print(
            f"{r['condition']:>20} | "
            f"ROC-AUC {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f} | "
            f"PR-AUC {r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f} | "
            f"ACC {r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
            f"F1 {r['f1_mean']:.4f}±{r['f1_std']:.4f}"
        )

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()