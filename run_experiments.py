# =============================================================================
# CDS535 DataBridge 实验代码「导读版」
# -----------------------------------------------------------------------------
# 目的：把队长原来的 run_experiments.py 拆清楚，方便你吃透逻辑。
# 逻辑与原版一致；多了中文注释和更直白的函数名。
# 用法：放在 CDS535_Project/ 目录下，与原版同样依赖，可运行：
#       python run_experiments_导读版.py
# 注意：运行会覆盖 results/ 与 figures/；想保留原结果请先复制整个文件夹。
# =============================================================================

from pathlib import Path
import json
import time
import hashlib
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve,
    silhouette_score, adjusted_rand_score,
)
from sklearn.base import clone
from threadpoolctl import threadpool_limits

# ---------------------------------------------------------------------------
# 0. 路径与全局设置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Windows 路径含中文/【】时，matplotlib/PIL 直接 savefig 可能报 Errno 22
# 用「先写入内存 → 再 Path.write_bytes」绕过
# ---------------------------------------------------------------------------
from io import BytesIO

def safe_savefig(fig, path):
    """把图存到 path；兼容含中文的 Windows 路径。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    path.write_bytes(buf.getvalue())
    buf.close()


SEED = 535  # 全项目固定种子，保证可复现

# Mushroom 原始文件没有表头，这里按 UCI .names 顺序起名
MUSHROOM_COLUMNS = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat",
]

# 四个分类器（超参固定，不用测试集调参）
MODELS = {
    "Decision tree": DecisionTreeClassifier(min_samples_leaf=2, random_state=SEED),
    "Random forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=1, random_state=SEED, n_jobs=1
    ),
    "Logistic regression": LogisticRegression(C=1, max_iter=3000, solver="lbfgs"),
    "RBF SVM": SVC(C=1, gamma="scale"),
}


# ---------------------------------------------------------------------------
# 1. 读数据 load()
# ---------------------------------------------------------------------------
def load_datasets():
    """
    读两份 UCI 原始文件，返回：
      {
        "Mushroom": (特征表 X, 标签 y),
        "Ionosphere": (特征表 X, 标签 y),
      }
    y 已变成 0/1：
      Mushroom   1 = poisonous(p)  有毒
      Ionosphere 1 = bad(b)        坏雷达回波
    """
    # --- Mushroom ---
    mushroom = pd.read_csv(
        ROOT / "data/mushroom/agaricus-lepiota.data",
        header=None,
        names=MUSHROOM_COLUMNS,
        na_values="?",  # 文件里的 ? 当成缺失
    )
    X_m = mushroom.drop(columns="class")
    y_m = (mushroom["class"] == "p").astype(int)

    # --- Ionosphere ---
    iono_names = [f"x{j:02d}" for j in range(1, 35)] + ["class"]
    ionosphere = pd.read_csv(
        ROOT / "data/ionosphere/ionosphere.data",
        header=None,
        names=iono_names,
    )
    # 删完全重复行（在任何随机划分之前）；不改原始文件
    ionosphere = ionosphere.drop_duplicates().reset_index(drop=True)
    X_i = ionosphere.drop(columns="class")
    y_i = (ionosphere["class"] == "b").astype(int)

    return {
        "Mushroom": (X_m, y_m),
        "Ionosphere": (X_i, y_i),
    }


# ---------------------------------------------------------------------------
# 2. 预处理 prep()  —— 你找不到 Ionosphere 的原因在这里！
# ---------------------------------------------------------------------------
def make_preprocessor(dataset_name, variant="base"):
    """
    返回一条 sklearn Pipeline（只含预处理，不含分类器）。

    ★ 关键：不是两个 if 并列写「Mushroom / Ionosphere」名字，
      而是：
        if 名字 == "Mushroom":
            return 类别特征那条管道
        else:
            return 数值特征那条管道   ← Ionosphere 走这里！

    variant 用来做「消融对照」（RQ2）：
      Mushroom:   "base" = 众数填补； "unknown" = 缺失填成字符串 unknown
      Ionosphere: "base" = 做标准化； "unscaled" = 跳过 StandardScaler
    """

    # ========== 分支 A：Mushroom（全是类别特征）==========
    if dataset_name == "Mushroom":
        if variant == "unknown":
            # 消融：把缺失显式当成一个叫 "unknown" 的类别
            imputer = SimpleImputer(strategy="constant", fill_value="unknown")
        else:
            # 默认：用训练集里出现最多的类别填缺失
            imputer = SimpleImputer(strategy="most_frequent")

        return Pipeline([
            ("impute", imputer),
            # One-Hot：类别 → 0/1 列；没见过的类别在 transform 时忽略
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            # 方差为 0 的列删掉（如 veil-type 常量）
            ("drop_constant", VarianceThreshold()),
        ])

    # ========== 分支 B：Ionosphere（以及一切非 Mushroom）==========
    # ★ 你截图里「只看到 Mushroom」就是因为 Ionosphere 写在这个 return 里，
    #   没有再写 if name == "Ionosphere"。
    if variant == "unscaled":
        scaler = "passthrough"  # 消融：不做标准化，原样通过
    else:
        scaler = StandardScaler()  # 默认：按训练集均值方差标准化

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # 本集无缺失，但管道保留一致性
        ("drop_constant", VarianceThreshold()),          # 去掉 x02 这类常量列
        ("scale", scaler),
    ])


# ---------------------------------------------------------------------------
# 3. 评价指标
# ---------------------------------------------------------------------------
def compute_scores(y_true, y_pred, y_score):
    """正类相关的一组指标；y_score 用于 AUC（概率或 decision_function）。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def decision_scores(fitted_pipeline, X):
    """SVM 用 decision_function；树/RF/LR 用正类概率。"""
    model = fitted_pipeline  # 整个 Pipeline 也能调这些方法
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 4. 画 EDA 图（图 01–03）
# ---------------------------------------------------------------------------
def plot_eda(data):
    """data = load_datasets() 的返回值。"""
    # --- 图 01：类别平衡 ---
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (name, (X, y)) in zip(axs, data.items()):
        counts = y.value_counts().sort_index()
        ax.bar(["negative", "positive"], counts, color=["#1c728e", "#dc8541"])
        for j, v in enumerate(counts):
            ax.text(j, v + max(counts) * 0.02, str(v), ha="center")
        ax.set_title(name)
        ax.set_ylim(0, max(counts) * 1.17)
        ax.set_ylabel("Records")
    fig.tight_layout()
    safe_savefig(fig, ROOT / "figures/01_class_balance.png")
    plt.close(fig)

    # --- 图 02：分布 ---
    X_m, y_m = data["Mushroom"]
    X_i, y_i = data["Ionosphere"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    pd.crosstab(X_m["odor"], y_m).plot.bar(
        stacked=True, ax=axs[0], color=["#1c728e", "#dc8541"]
    )
    axs[0].set_title("Mushroom: odor by class")
    axs[0].set_ylabel("Records")
    axs[0].legend(["edible", "poisonous"], fontsize=8)
    for label in [0, 1]:
        axs[1].hist(
            X_i.loc[y_i == label, "x03"],
            bins=15, alpha=0.6, label=["good", "bad"][label],
        )
    axs[1].set_title("Ionosphere: x03")
    axs[1].set_xlabel("Signal feature value")
    axs[1].legend()
    fig.tight_layout()
    safe_savefig(fig, ROOT / "figures/02_distributions.png")
    plt.close(fig)

    # --- 图 03：Ionosphere 相关（去掉常量 x02）---
    fig, ax = plt.subplots(figsize=(7, 4))
    corr = X_i.drop(columns="x02").corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Ionosphere Pearson correlation (constant x02 excluded)")
    ticks = list(range(0, 33, 4))
    cols = X_i.drop(columns="x02").columns
    ax.set_xticks(ticks, cols[ticks], rotation=45)
    ax.set_yticks(ticks, cols[ticks])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    safe_savefig(fig, ROOT / "figures/03_correlations.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 单个数据集：划分 → 10 折 CV → 测试 → 消融 → 聚类
# ---------------------------------------------------------------------------
def run_one_dataset(name, X, y, rows, cvrows, ablations, clusters, metadata):
    """
    对「一个」数据集跑完整流水线。

    顺序（防泄漏核心）：
      ① 先分层留出 20% 测试集
      ② 只在剩下 80% 开发集上做分层 10 折
      ③ 每一折：预处理+模型都只在该折训练集上 fit
      ④ 10 折结束后：用全部开发集 refit，再评一次独立测试集
      ⑤ 消融、聚类都只用开发集，测试集不参与
    """
    print("DATASET", name, flush=True)

    # 导出带 target 的 CSV（方便人工查看）
    clean = X.copy()
    clean["target"] = y
    clean.to_csv(ROOT / f"data/{name.lower()}.csv", index=False)

    # ----- ① 先留 20% 测试 -----
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
    )
    X_dev, y_dev = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # ----- ② 开发集上建 10 折 -----
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fold_splits = list(kfold.split(X_dev, y_dev))

    # 记录每行属于哪一折；测试行为 -1
    fold_id = np.full(len(y), -1)
    for fold, (_, val_local) in enumerate(fold_splits, start=1):
        # val_local 是「开发集内部」下标，要映射回全局行号
        fold_id[train_idx[val_local]] = fold
    pd.DataFrame({
        "row": np.arange(len(y)),
        "development_fold": fold_id,
        "target": y,
    }).to_csv(ROOT / f"results/{name}_splits.csv", index=False)

    # 元数据
    metadata[name] = {
        "raw_n": 8124 if name == "Mushroom" else 351,
        "n": len(y),
        "features": X.shape[1],
        "class_counts": y.value_counts().to_dict(),
        "missing_cells": int(X.isna().sum().sum()),
        "missing_rows": int(X.isna().any(axis=1).sum()),
        "constant_columns": list(X.columns[X.nunique(dropna=False) == 1]),
        "duplicate_predictors": int(X.duplicated().sum()),
        "development_n": len(train_idx),
        "test_n": len(test_idx),
    }
    pd.DataFrame({
        "attribute": X.columns,
        "type": [str(t) for t in X.dtypes],
        "missing": X.isna().sum().values,
        "unique_observed": X.nunique().values,
    }).to_csv(ROOT / f"data/{name}_schema.csv", index=False)

    # ----- ③④ 四个模型：10 折 + 独立测试 + ROC -----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for model_name, estimator in MODELS.items():
        t0 = time.perf_counter()

        # 每一折：新的预处理 + 新的模型（clone），只在折内训练集 fit
        for fold, (tr_local, va_local) in enumerate(fold_splits, start=1):
            pipe = Pipeline([
                ("prep", make_preprocessor(name)),          # 默认 base 预处理
                ("model", clone(estimator)),
            ])
            pipe.fit(X_dev.iloc[tr_local], y_dev.iloc[tr_local])
            pred = pipe.predict(X_dev.iloc[va_local])
            score = decision_scores(pipe, X_dev.iloc[va_local])
            cvrows.append({
                "dataset": name,
                "model": model_name,
                "fold": fold,
                **compute_scores(y_dev.iloc[va_local], pred, score),
            })

        elapsed = time.perf_counter() - t0

        # 全部开发集 refit → 评独立测试集
        pipe = Pipeline([
            ("prep", make_preprocessor(name)),
            ("model", clone(estimator)),
        ]).fit(X_dev, y_dev)
        pred_te = pipe.predict(X_test)
        score_te = decision_scores(pipe, X_test)
        test_metrics = compute_scores(y_test, pred_te, score_te)

        fold_df = pd.DataFrame([
            r for r in cvrows
            if r["dataset"] == name and r["model"] == model_name
        ])
        row = {
            "dataset": name,
            "model": model_name,
            "cv_seconds": elapsed,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        for k in test_metrics:
            row[f"cv_{k}"] = fold_df[k].mean()
            row[f"sd_{k}"] = fold_df[k].std()
        rows.append(row)

        # 存预测与混淆矩阵
        safe = model_name.replace(" ", "_")
        pd.DataFrame({
            "row": test_idx,
            "actual": y_test.to_numpy(),
            "predicted": pred_te,
            "score": score_te,
        }).to_csv(ROOT / f"results/{name}_{safe}_test_predictions.csv", index=False)
        np.savetxt(
            ROOT / f"results/{name}_{safe}_confusion.csv",
            confusion_matrix(y_test, pred_te, labels=[0, 1]),
            fmt="%d", delimiter=",",
        )

        fpr, tpr, _ = roc_curve(y_test, score_te)
        ax.plot(fpr, tpr, label=f"{model_name}: {test_metrics['auc']:.3f}")
        print(
            model_name,
            "CV F1", round(row["cv_f1"], 4),
            "test", round(test_metrics["f1"], 4),
            flush=True,
        )

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{name}: untouched 20% test ROC",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    safe_savefig(fig, ROOT / f"figures/04_ROC_{name}.png")
    plt.close(fig)

    # ----- ⑤ 消融（RQ2）：只改预处理 variant，折与模型不变 -----
    # Mushroom → 试 "unknown"；Ionosphere → 试 "unscaled"
    ablation_variant = "unknown" if name == "Mushroom" else "unscaled"
    for model_name in ["Logistic regression", "RBF SVM"]:
        for fold, (tr_local, va_local) in enumerate(fold_splits, start=1):
            pipe = Pipeline([
                ("prep", make_preprocessor(name, ablation_variant)),
                ("model", clone(MODELS[model_name])),
            ]).fit(X_dev.iloc[tr_local], y_dev.iloc[tr_local])
            pred = pipe.predict(X_dev.iloc[va_local])
            score = decision_scores(pipe, X_dev.iloc[va_local])
            ablations.append({
                "dataset": name,
                "model": model_name,
                "variant": ablation_variant,
                "fold": fold,
                **compute_scores(y_dev.iloc[va_local], pred, score),
            })

    # ----- ⑥ KMeans（RQ3）：只用开发集特征，fit 不用 y -----
    Z = make_preprocessor(name).fit_transform(X_dev)
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        sil = silhouette_score(
            Z, km.labels_, sample_size=min(1500, len(Z)), random_state=SEED
        )
        # 下面 crosstab / ARI 才用到标签，仅事后对照
        ct = pd.crosstab(km.labels_, y_dev.to_numpy())
        clusters.append({
            "dataset": name,
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "purity": ct.max(axis=1).sum() / len(y_dev),
            "ari": adjusted_rand_score(y_dev, km.labels_),
        })
        ct.to_csv(ROOT / f"results/{name}_clusters_k{k}.csv")


# ---------------------------------------------------------------------------
# 6. 主函数：串起来
# ---------------------------------------------------------------------------
def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
    })

    data = load_datasets()
    rows, cvrows, ablations, clusters, metadata = [], [], [], [], {}

    # 先画 EDA
    plot_eda(data)

    # 再对每个数据集跑完整实验
    for name, (X, y) in data.items():
        run_one_dataset(name, X, y, rows, cvrows, ablations, clusters, metadata)

    # 存表
    pd.DataFrame(rows).to_csv(ROOT / "results/model_comparison.csv", index=False)
    pd.DataFrame(cvrows).to_csv(ROOT / "results/fold_metrics.csv", index=False)
    pd.DataFrame(ablations).to_csv(ROOT / "results/ablation_folds.csv", index=False)
    pd.DataFrame(clusters).to_csv(ROOT / "results/clustering.csv", index=False)
    (ROOT / "results/metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf8"
    )
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": SEED,
    }
    (ROOT / "results/environment.json").write_text(
        json.dumps(versions, indent=2), encoding="utf8"
    )
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "data").rglob("*") if p.is_file()
    }
    (ROOT / "results/data_sha256.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf8"
    )

    # 图 05：聚类曲线
    c = pd.DataFrame(clusters)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for name in data:
        sub = c[c.dataset == name]
        axs[0].plot(sub.k, sub.inertia / sub.inertia.iloc[0], marker="o", label=name)
        axs[1].plot(sub.k, sub.silhouette, marker="o", label=name)
    axs[0].set(xlabel="k", ylabel="SSE / SSE at k=2", title="Relative within-cluster dispersion")
    axs[1].set(xlabel="k", ylabel="Silhouette", title="Separation in transformed feature space")
    for ax in axs:
        ax.legend()
    fig.tight_layout()
    safe_savefig(fig, ROOT / "figures/05_clustering.png")
    plt.close(fig)

    print("COMPLETE", flush=True)


if __name__ == "__main__":
    # 限制底层线程数，避免结果抖动
    with threadpool_limits(limits=1):
        main()
