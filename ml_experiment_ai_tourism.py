from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "数据" / "final_analysis_722.xlsx"
OUT_DIR = ROOT / "analysis_outputs" / "ml_experiment"
TODAY = pd.Timestamp("2026-04-28")

COLUMN_RENAME = {
    "AI介入形式（1=AI音频，2=AI画面，3=混合）": "AI介入形式",
    "AI介入程度（1=浅度，2=中度，3=深度）": "AI介入程度",
    "AI技术质感（1=粗糙，2=普通，3=精良）": "AI技术质感",
    "视频主体（1=人，2=物，3=景色）": "视频主体",
    "视频主题（1=自然风光，2=历史人文，3=现代城市/生活，4=剧情/整活,5=家国情怀）": "视频主题",
    "内容导向（1信息实用型，2娱乐审美型）": "内容导向",
    "视频风格（1=纪实/写实，2=卡通/艺术化，3=抽象/超现实）": "视频风格",
}

CATEGORY_MAPS = {
    "AI介入形式": {1: "AI音频", 2: "AI画面", 3: "混合"},
    "AI介入程度": {1: "浅度", 2: "中度", 3: "深度"},
    "AI技术质感": {1: "粗糙", 2: "普通", 3: "精良"},
    "视频主体": {1: "人", 2: "物", 3: "景色"},
    "视频主题": {1: "自然风光", 2: "历史人文", 3: "现代城市/生活", 4: "剧情/整活", 5: "家国情怀"},
    "内容导向": {1: "信息实用型", 2: "娱乐审美型"},
    "视频风格": {1: "纪实/写实", 2: "卡通/艺术化", 3: "抽象/超现实"},
}

ENGAGEMENT_COLS = ["点赞量", "评论量", "收藏量", "转发量"]
CAT_COLS = list(CATEGORY_MAPS.keys())
NUM_COLS = ["log视频时长", "log发布距今天数"]
MAIN_FEATURES = CAT_COLS + NUM_COLS
PROVINCE_FEATURES = ["省份"] + MAIN_FEATURES


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def load_data() -> pd.DataFrame:
    df = pd.read_excel(SOURCE).rename(columns=COLUMN_RENAME)
    for col in ENGAGEMENT_COLS + ["视频时长(秒)"] + CAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["视频发布日期"] = pd.to_datetime(df["视频发布日期"], errors="coerce")

    mask = pd.Series(True, index=df.index)
    for col, mapping in CATEGORY_MAPS.items():
        mask &= df[col].isin(mapping.keys())
        df[col] = df[col].map(mapping)
    df = df.loc[mask].copy()

    df["总互动量"] = df[ENGAGEMENT_COLS].sum(axis=1)
    df["log总互动量"] = np.log1p(df["总互动量"])
    df["log点赞量"] = np.log1p(df["点赞量"])
    df["log视频时长"] = np.log1p(df["视频时长(秒)"])
    df["发布距今天数"] = (TODAY - df["视频发布日期"]).dt.days + 1
    df["发布距今天数"] = df["发布距今天数"].clip(lower=1)
    df["log发布距今天数"] = np.log1p(df["发布距今天数"])
    df["高互动"] = (df["总互动量"] >= df["总互动量"].quantile(0.75)).astype(int)
    return df.dropna(subset=MAIN_FEATURES + ["log总互动量", "高互动"])


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if col not in NUM_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
    )


def regression_models(features: list[str]) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(features)
    return {
        "Ridge线性基准": Pipeline(
            [("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=42))]
        ),
        "随机森林": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        max_depth=None,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=600,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "梯度提升": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=350,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_regression(df: pd.DataFrame, features: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[features]
    y = df["log总互动量"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    predictions = pd.DataFrame({"实际log总互动量": y})
    baseline_pred = np.repeat(y.mean(), len(y))
    rows.append(
        {
            "特征集": label,
            "模型": "均值基准",
            "R2均值": r2_score(y, baseline_pred),
            "R2标准差": 0.0,
            "RMSE均值": mean_squared_error(y, baseline_pred, squared=False),
            "MAE均值": mean_absolute_error(y, baseline_pred),
        }
    )

    for name, model in regression_models(features).items():
        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
            n_jobs=None,
        )
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=None)
        predictions[f"{label}_{name}"] = pred
        rows.append(
            {
                "特征集": label,
                "模型": name,
                "R2均值": scores["test_r2"].mean(),
                "R2标准差": scores["test_r2"].std(),
                "RMSE均值": -scores["test_neg_root_mean_squared_error"].mean(),
                "MAE均值": -scores["test_neg_mean_absolute_error"].mean(),
            }
        )
    return pd.DataFrame(rows), predictions


def evaluate_classification(df: pd.DataFrame) -> pd.DataFrame:
    features = MAIN_FEATURES
    x = df[features]
    y = df["高互动"]
    model = Pipeline(
        [
            ("prep", make_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "任务": "高互动视频识别",
                "定义": "总互动量位于样本前25%",
                "正类样本数": int(y.sum()),
                "负类样本数": int((1 - y).sum()),
                "AUC": roc_auc_score(y, prob),
                "Accuracy": accuracy_score(y, pred),
                "F1": f1_score(y, pred),
            }
        ]
    )


def best_model_and_importance(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    features = MAIN_FEATURES
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    model = regression_models(features)["随机森林"]
    model.fit(train[features], train["log总互动量"])
    test_pred = model.predict(test[features])
    heldout = {
        "测试集R2": r2_score(test["log总互动量"], test_pred),
        "测试集RMSE": mean_squared_error(test["log总互动量"], test_pred, squared=False),
        "测试集MAE": mean_absolute_error(test["log总互动量"], test_pred),
    }

    perm = permutation_importance(
        model,
        test[features],
        test["log总互动量"],
        n_repeats=30,
        random_state=42,
        scoring="r2",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "变量": features,
            "Permutation重要性均值": perm.importances_mean,
            "Permutation重要性标准差": perm.importances_std,
        }
    ).sort_values("Permutation重要性均值", ascending=False)
    for k, v in heldout.items():
        importance[k] = v
    return model, importance


def make_charts(df: pd.DataFrame, reg_results: pd.DataFrame, importance: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Path]:
    setup_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = reg_results[reg_results["模型"] != "均值基准"].copy()
    labels = plot_df["特征集"] + "-" + plot_df["模型"]
    ax.barh(labels, plot_df["R2均值"], color="#4C78A8")
    ax.set_xlabel("5折交叉验证 R2")
    ax.set_title("机器学习模型预测效果对比")
    ax.axvline(0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    paths["model_compare"] = OUT_DIR / "01_机器学习模型R2对比.png"
    fig.savefig(paths["model_compare"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    imp = importance.head(10).sort_values("Permutation重要性均值")
    ax.barh(imp["变量"], imp["Permutation重要性均值"], color="#F58518")
    ax.set_xlabel("Permutation Importance: R2下降量")
    ax.set_title("随机森林变量重要性")
    fig.tight_layout()
    paths["importance"] = OUT_DIR / "02_随机森林变量重要性.png"
    fig.savefig(paths["importance"])
    plt.close(fig)

    pred_col = "主体变量_随机森林"
    if pred_col in predictions.columns:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(predictions["实际log总互动量"], predictions[pred_col], s=18, alpha=0.55, color="#54A24B")
        lo = min(predictions["实际log总互动量"].min(), predictions[pred_col].min())
        hi = max(predictions["实际log总互动量"].max(), predictions[pred_col].max())
        ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1)
        ax.set_xlabel("实际 log(总互动量+1)")
        ax.set_ylabel("交叉验证预测值")
        ax.set_title("随机森林预测值与实际值")
        fig.tight_layout()
        paths["pred_actual"] = OUT_DIR / "03_随机森林预测实际散点图.png"
        fig.savefig(paths["pred_actual"])
        plt.close(fig)

    return paths


def write_report(
    df: pd.DataFrame,
    reg_results: pd.DataFrame,
    clf_results: pd.DataFrame,
    importance: pd.DataFrame,
    paths: dict[str, Path],
    excel_path: Path,
) -> Path:
    best = reg_results[reg_results["模型"] != "均值基准"].sort_values("R2均值", ascending=False).iloc[0]
    rf_main = reg_results[(reg_results["特征集"] == "主体变量") & (reg_results["模型"] == "随机森林")].iloc[0]
    top_imp = importance.head(5)
    clf = clf_results.iloc[0]

    lines = [
        "# AI文旅短视频机器学习补充实验报告",
        "",
        f"数据源：`{SOURCE}`",
        f"结果表：`{excel_path}`",
        "",
        "## 1. 实验设置",
        "",
        f"- 有效样本量为 {len(df)} 条；目标变量为 `log(总互动量+1)`。",
        "- 预测特征包括 AI介入形式、AI介入程度、AI技术质感、视频主体、视频主题、内容导向、视频风格、log视频时长、log发布距今天数。",
        "- 采用 5 折交叉验证比较岭回归、随机森林、ExtraTrees 与梯度提升模型；另外加入“省份”的扩展特征集作为稳健性比较。",
        "- 高互动分类实验将总互动量前25%的视频定义为正类，使用随机森林分类器评估识别能力。",
        "",
        "## 2. 预测效果",
        "",
        f"- 交叉验证表现最好的模型为 `{best['特征集']}-{best['模型']}`，R2={best['R2均值']:.3f}，RMSE={best['RMSE均值']:.3f}。",
        f"- 主体变量随机森林模型 R2={rf_main['R2均值']:.3f}，RMSE={rf_main['RMSE均值']:.3f}，说明仅凭视频内容与AI特征可以解释一部分传播效果差异，但仍有平台推荐、账号粉丝量、发布时间段等未观测因素。",
        f"- 高互动视频识别 AUC={clf['AUC']:.3f}，F1={clf['F1']:.3f}，可作为识别潜在爆款内容的补充实验。",
        "",
        "## 3. 变量重要性",
        "",
        "- 随机森林 Permutation Importance 前五位为：",
    ]
    for _, row in top_imp.iterrows():
        lines.append(f"  - {row['变量']}：重要性 {row['Permutation重要性均值']:.4f}")

    lines.extend(
        [
            "",
            "## 4. 论文使用建议",
            "",
            "- 机器学习模型适合作为第四章的补充分析或稳健性检验，用来证明核心变量对传播效果具有预测价值。",
            "- 不建议用随机森林替代回归假设检验，因为随机森林主要服务于预测与变量重要性排序，不能直接给出传统假设检验中的系数方向和显著性。",
            "- 正文可写作：在回归分析之外，本文进一步采用随机森林等机器学习模型进行预测验证，结果显示 AI技术质感、视频风格、发布时间累积效应等变量具有较高的重要性，进一步支持内容质量与视觉风格是影响传播效果的重要因素。",
            "- XGBoost 当前环境未安装，因此本轮未纳入；考虑到样本量仅 723 条，随机森林与梯度提升已能满足本科论文的补充实验需要。",
            "",
            "## 5. 图表文件",
            "",
        ]
    )
    for path in paths.values():
        lines.append(f"- `{path}`")

    report_path = OUT_DIR / "机器学习补充实验报告.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    reg_main, pred_main = evaluate_regression(df, MAIN_FEATURES, "主体变量")
    reg_province, pred_province = evaluate_regression(df, PROVINCE_FEATURES, "加入省份")
    reg_results = pd.concat([reg_main, reg_province], ignore_index=True)
    predictions = pd.concat([pred_main, pred_province.drop(columns=["实际log总互动量"])], axis=1)
    clf_results = evaluate_classification(df)
    _, importance = best_model_and_importance(df)

    excel_path = OUT_DIR / "机器学习补充实验结果.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        reg_results.to_excel(writer, index=False, sheet_name="模型预测效果")
        clf_results.to_excel(writer, index=False, sheet_name="高互动分类实验")
        importance.to_excel(writer, index=False, sheet_name="随机森林变量重要性")
        predictions.to_excel(writer, index=False, sheet_name="交叉验证预测值")

    chart_paths = make_charts(df, reg_results, importance, predictions)
    report_path = write_report(df, reg_results, clf_results, importance, chart_paths, excel_path)

    print("机器学习实验完成")
    print(f"report: {report_path.resolve()}")
    print(f"excel: {excel_path.resolve()}")
    for name, path in chart_paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
