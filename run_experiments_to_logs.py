from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "数据" / "final_analysis_722.xlsx"
RAW_SOURCE_DIR = ROOT / "数据" / "原始数据"
LOG_DIR = Path("logs")
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
NUM_COLS = ["log视频时长", "log发布距采集日天数"]
FEATURES = CAT_COLS + NUM_COLS
OUTCOMES = {
    "log总互动量": "总互动量",
    "log日均互动量": "日均互动量",
    "log点赞量": "点赞量",
    "log评论量": "评论量",
    "log收藏量": "收藏量",
    "log转发量": "转发量",
}

def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "experiment_run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140


def p_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "+"
    return ""


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("读取数据：%s", SOURCE)
    raw = pd.read_excel(SOURCE).rename(columns=COLUMN_RENAME)
    if "Unnamed: 3" in raw.columns:
        raw = raw.rename(columns={"Unnamed: 3": "播放入口"})
    for col in ENGAGEMENT_COLS + ["视频时长(秒)"] + CAT_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["视频发布日期"] = pd.to_datetime(raw["视频发布日期"], errors="coerce")

    invalid_pieces = []
    valid_mask = pd.Series(True, index=raw.index)
    for col, mapping in CATEGORY_MAPS.items():
        bad = ~raw[col].isin(mapping.keys())
        valid_mask &= ~bad
        if bad.any():
            invalid = raw.loc[bad, ["省份", "标题", "链接", col] + ENGAGEMENT_COLS].copy()
            invalid.insert(0, "异常字段", col)
            invalid = invalid.rename(columns={col: "异常取值"})
            invalid_pieces.append(invalid)

    df = raw.loc[valid_mask].copy()
    for col, mapping in CATEGORY_MAPS.items():
        df[col] = df[col].map(mapping)

    collection_dates: dict[str, pd.Timestamp] = {}
    if RAW_SOURCE_DIR.exists():
        for file in RAW_SOURCE_DIR.glob("*.xlsx"):
            collection_dates[file.stem] = pd.Timestamp(file.stat().st_mtime, unit="s").normalize()

    df["总互动量"] = df[ENGAGEMENT_COLS].sum(axis=1)
    df["采集日期"] = df["省份"].map(collection_dates)
    df["采集日期"] = df["采集日期"].fillna(TODAY)
    df["发布距采集日天数"] = (df["采集日期"] - df["视频发布日期"]).dt.days + 1
    df["发布距采集日天数"] = df["发布距采集日天数"].clip(lower=1)
    df["日均互动量"] = df["总互动量"] / df["发布距采集日天数"]
    for outcome, base_col in OUTCOMES.items():
        df[outcome] = np.log1p(df[base_col])
    df["log视频时长"] = np.log1p(df["视频时长(秒)"])
    df["log发布距采集日天数"] = np.log1p(df["发布距采集日天数"])
    df["高互动"] = (df["总互动量"] >= df["总互动量"].quantile(0.75)).astype(int)

    invalid_rows = (
        pd.concat(invalid_pieces, ignore_index=True)
        if invalid_pieces
        else pd.DataFrame(columns=["异常字段", "省份", "标题", "链接", "异常取值"] + ENGAGEMENT_COLS)
    )
    logging.info("原始样本=%s，有效样本=%s，编码复核记录=%s", len(raw), len(df), len(invalid_rows))
    return df, invalid_rows


def data_quality(df: pd.DataFrame, invalid_rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"项目": "有效样本量", "结果": len(df), "说明": "进入建模分析"},
            {"项目": "省份/地区数", "结果": df["省份"].nunique(), "说明": ""},
            {"项目": "编码复核记录数", "结果": len(invalid_rows), "说明": "仅用于数据整理过程记录"},
            {"项目": "采集日期最早", "结果": str(df["采集日期"].min().date()), "说明": "按省份原始文件时间近似还原"},
            {"项目": "采集日期最晚", "结果": str(df["采集日期"].max().date()), "说明": "按省份原始文件时间近似还原"},
            {"项目": "发布日期最早", "结果": str(df["视频发布日期"].min().date()), "说明": ""},
            {"项目": "发布日期最晚", "结果": str(df["视频发布日期"].max().date()), "说明": ""},
            {"项目": "总互动量P75", "结果": float(df["总互动量"].quantile(0.75)), "说明": "高互动分类阈值"},
        ]
    )


def numeric_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    cols = ENGAGEMENT_COLS + ["总互动量", "日均互动量", "视频时长(秒)", "发布距采集日天数"] + list(OUTCOMES.keys())
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    desc["偏度"] = df[cols].skew()
    return desc.reset_index(names="变量")


def category_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CAT_COLS:
        tmp = df.groupby(col, dropna=False).size().reset_index(name="频数")
        tmp["占比"] = tmp["频数"] / len(df)
        tmp.insert(0, "变量", col)
        tmp = tmp.rename(columns={col: "类别"})
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CAT_COLS:
        tmp = (
            df.groupby(col)
            .agg(
                样本量=("总互动量", "size"),
                总互动量均值=("总互动量", "mean"),
                总互动量中位数=("总互动量", "median"),
                log总互动量均值=("log总互动量", "mean"),
                点赞量中位数=("点赞量", "median"),
                评论量中位数=("评论量", "median"),
                收藏量中位数=("收藏量", "median"),
                转发量中位数=("转发量", "median"),
            )
            .reset_index()
            .rename(columns={col: "类别"})
        )
        tmp.insert(0, "变量", col)
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def single_factor_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CAT_COLS:
        groups = [g["总互动量"].to_numpy() for _, g in df.groupby(col)]
        k = len(groups)
        if k == 2:
            u, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            n1, n2 = len(groups[0]), len(groups[1])
            effect = 1 - (2 * u) / (n1 * n2)
            test = "Mann-Whitney U"
            statistic = u
            effect_name = "rank-biserial r"
        else:
            h, p = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            effect = (h - k + 1) / (n - k)
            test = "Kruskal-Wallis H"
            statistic = h
            effect_name = "epsilon-squared"
        rows.append(
            {
                "变量": col,
                "检验方法": test,
                "统计量": statistic,
                "p值": p,
                "显著性": p_stars(p),
                "效应量名称": effect_name,
                "效应量": effect,
                "结论": "存在显著差异" if p < 0.05 else "未发现显著差异",
            }
        )
    return pd.DataFrame(rows).sort_values("p值")


def build_design_matrix(df: pd.DataFrame, include_province: bool = False) -> tuple[pd.DataFrame, dict[str, str]]:
    parts = [pd.Series(1.0, index=df.index, name="截距")]
    refs = {}
    cats = CAT_COLS + (["省份"] if include_province else [])
    for col in cats:
        labels = df[col].astype("category")
        if col in CATEGORY_MAPS:
            ordered = list(CATEGORY_MAPS[col].values())
            labels = labels.cat.set_categories(ordered, ordered=True)
            refs[col] = ordered[0]
        else:
            ordered = sorted(df[col].dropna().unique().tolist())
            labels = labels.cat.set_categories(ordered, ordered=True)
            refs[col] = ordered[0] if ordered else ""
        dummies = pd.get_dummies(labels, prefix=col, drop_first=True, dtype=float)
        parts.append(dummies)
    parts.append(df[NUM_COLS].astype(float))
    x = pd.concat(parts, axis=1)
    return x, refs


def ols(df: pd.DataFrame, y_col: str, include_province: bool = False) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    x, refs = build_design_matrix(df, include_province=include_province)
    y = df[y_col].to_numpy(dtype=float)
    x_mat = x.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    fitted = x_mat @ beta
    resid = y - fitted
    n, p = x_mat.shape
    df_resid = n - p
    sigma2 = float((resid @ resid) / df_resid)
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_values = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_values), df=df_resid)
    ci = stats.t.ppf(0.975, df=df_resid) * se
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid

    result = pd.DataFrame(
        {
            "因变量": y_col,
            "模型": "含省份固定效应" if include_province else "主模型",
            "变量": x.columns,
            "系数": beta,
            "标准误": se,
            "t值": t_values,
            "p值": p_values,
            "显著性": [p_stars(v) for v in p_values],
            "95%CI下限": beta - ci,
            "95%CI上限": beta + ci,
            "影响比例_exp系数减1": np.exp(beta) - 1,
        }
    )
    metrics = {"样本量": n, "自变量数": p, "R2": r2, "调整R2": adj_r2, "残差自由度": df_resid}
    return result, metrics, refs


def run_regressions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main, main_metrics, refs = ols(df, "log总互动量", include_province=False)
    province, province_metrics, _ = ols(df, "log总互动量", include_province=True)

    robust_parts = []
    metric_rows = []
    for outcome in OUTCOMES:
        res, metrics, _ = ols(df, outcome, include_province=False)
        robust_parts.append(res)
        metric_rows.append({"因变量": outcome, "模型": "主模型", **metrics})

    metric_rows.extend(
        [
            {"因变量": "log总互动量", "模型": "主模型", **main_metrics},
            {"因变量": "log总互动量", "模型": "含省份固定效应", **province_metrics},
        ]
    )
    regression_all = pd.concat([main, province], ignore_index=True)
    robust = pd.concat(robust_parts, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows).drop_duplicates(subset=["因变量", "模型"], keep="last")

    ref_rows = [{"变量": k, "参照组": v} for k, v in refs.items()]
    ref_rows.extend([{"变量": "log视频时长", "参照组": "连续变量"}, {"变量": "log发布距采集日天数", "参照组": "连续变量"}])
    refs_df = pd.DataFrame(ref_rows)
    return regression_all, robust, metrics_df, refs_df


def key_effects(main_regression: pd.DataFrame) -> pd.DataFrame:
    skip = {"截距"}
    out = main_regression[
        (main_regression["模型"] == "主模型")
        & (main_regression["因变量"] == "log总互动量")
        & (~main_regression["变量"].isin(skip))
    ].copy()
    out["影响百分比"] = out["影响比例_exp系数减1"] * 100
    out["方向"] = np.where(out["系数"] > 0, "正向", "负向")
    out["是否显著_0.05"] = np.where(out["p值"] < 0.05, "显著", "不显著")
    out["显著排序"] = np.where(out["p值"] < 0.05, 0, 1)
    return out.sort_values(["显著排序", "p值"]).drop(columns=["显著排序"])


def variable_level_conclusion(key: pd.DataFrame) -> pd.DataFrame:
    groups = CAT_COLS + NUM_COLS
    rows = []
    for group in groups:
        if group in NUM_COLS:
            part = key[key["变量"] == group].copy()
        else:
            part = key[key["变量"].str.startswith(f"{group}_")].copy()
        sig = part[part["p值"] < 0.05].copy()
        if sig.empty:
            rows.append(
                {
                    "指标": group,
                    "控制其他变量后是否有影响": "未发现显著影响",
                    "显著类别或变量": "",
                    "具体影响": "",
                    "最小p值": part["p值"].min() if not part.empty else np.nan,
                }
            )
        else:
            details = []
            for _, row in sig.sort_values("p值").iterrows():
                details.append(f"{row['变量']}：{row['方向']}，约{row['影响百分比']:.1f}%")
            rows.append(
                {
                    "指标": group,
                    "控制其他变量后是否有影响": "有显著影响",
                    "显著类别或变量": "；".join(sig.sort_values("p值")["变量"].tolist()),
                    "具体影响": "；".join(details),
                    "最小p值": sig["p值"].min(),
                }
            )
    return pd.DataFrame(rows)


def robustness_summary(robust: pd.DataFrame) -> pd.DataFrame:
    out = robust[(robust["变量"] != "截距") & (robust["p值"] < 0.05)].copy()
    out["影响百分比"] = (np.exp(out["系数"]) - 1) * 100
    out["方向"] = np.where(out["系数"] > 0, "正向", "负向")
    return out[
        ["因变量", "变量", "方向", "系数", "影响百分比", "p值", "显著性", "95%CI下限", "95%CI上限"]
    ].sort_values(["因变量", "p值"])


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [col for col in features if col not in NUM_COLS]
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical), ("num", StandardScaler(), NUM_COLS)]
    )


def ml_experiments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[FEATURES]
    y = df["log总互动量"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pre = make_preprocessor(FEATURES)
    models = {
        "Ridge线性基准": Pipeline([("prep", pre), ("model", Ridge(alpha=1.0))]),
        "随机森林": Pipeline(
            [
                ("prep", pre),
                ("model", RandomForestRegressor(n_estimators=600, min_samples_leaf=4, random_state=42, n_jobs=-1)),
            ]
        ),
        "梯度提升": Pipeline(
            [
                ("prep", pre),
                ("model", GradientBoostingRegressor(n_estimators=350, learning_rate=0.04, max_depth=3, min_samples_leaf=8, random_state=42)),
            ]
        ),
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"),
        )
        rows.append(
            {
                "模型": name,
                "R2均值": scores["test_r2"].mean(),
                "R2标准差": scores["test_r2"].std(),
                "RMSE均值": -scores["test_neg_root_mean_squared_error"].mean(),
                "MAE均值": -scores["test_neg_mean_absolute_error"].mean(),
            }
        )
    model_results = pd.DataFrame(rows).sort_values("R2均值", ascending=False)

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    rf = models["随机森林"]
    rf.fit(train[FEATURES], train["log总互动量"])
    perm = permutation_importance(rf, test[FEATURES], test["log总互动量"], n_repeats=30, random_state=42, scoring="r2", n_jobs=-1)
    importance = pd.DataFrame(
        {
            "变量": FEATURES,
            "Permutation重要性均值": perm.importances_mean,
            "Permutation重要性标准差": perm.importances_std,
        }
    ).sort_values("Permutation重要性均值", ascending=False)

    clf = Pipeline(
        [
            ("prep", make_preprocessor(FEATURES)),
            ("model", RandomForestClassifier(n_estimators=600, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
    )
    y_bin = df["高互动"]
    cv_bin = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, x, y_bin, cv=cv_bin, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    clf_results = pd.DataFrame(
        [
            {
                "任务": "高互动视频识别",
                "定义": "总互动量位于样本前25%",
                "AUC": roc_auc_score(y_bin, prob),
                "Accuracy": accuracy_score(y_bin, pred),
                "F1": f1_score(y_bin, pred),
                "正类样本数": int(y_bin.sum()),
                "负类样本数": int((1 - y_bin).sum()),
            }
        ]
    )
    return model_results, importance, clf_results


def make_charts(df: pd.DataFrame, single_tests: pd.DataFrame, key: pd.DataFrame, ml_importance: pd.DataFrame) -> dict[str, Path]:
    setup_plot_style()
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["log总互动量"], bins=35, color="#4C78A8", edgecolor="white")
    ax.set_title("log总互动量分布")
    ax.set_xlabel("log(总互动量+1)")
    ax.set_ylabel("视频数量")
    fig.tight_layout()
    paths["log_total_distribution"] = LOG_DIR / "01_log总互动量分布.png"
    fig.savefig(paths["log_total_distribution"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    tmp = single_tests.sort_values("效应量")
    ax.barh(tmp["变量"], tmp["效应量"], color="#F58518")
    ax.set_title("单因素检验效应量")
    ax.set_xlabel("效应量")
    fig.tight_layout()
    paths["single_factor_effect"] = LOG_DIR / "02_单因素检验效应量.png"
    fig.savefig(paths["single_factor_effect"])
    plt.close(fig)

    sig = key[key["p值"] < 0.05].copy()
    if not sig.empty:
        sig = sig.reindex(sig["影响百分比"].abs().sort_values().index)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sig))))
        colors = np.where(sig["影响百分比"] > 0, "#54A24B", "#E45756")
        ax.barh(sig["变量"], sig["影响百分比"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title("主回归显著变量的影响百分比")
        ax.set_xlabel("exp(系数)-1，百分比")
        fig.tight_layout()
        paths["regression_effects"] = LOG_DIR / "03_主回归显著影响百分比.png"
        fig.savefig(paths["regression_effects"])
        plt.close(fig)

    if not ml_importance.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        imp = ml_importance.head(10).sort_values("Permutation重要性均值")
        ax.barh(imp["变量"], imp["Permutation重要性均值"], color="#4C78A8")
        ax.set_title("随机森林变量重要性")
        ax.set_xlabel("Permutation Importance")
        fig.tight_layout()
        paths["ml_importance"] = LOG_DIR / "04_随机森林变量重要性.png"
        fig.savefig(paths["ml_importance"])
        plt.close(fig)

    return paths


def write_report(
    df: pd.DataFrame,
    dq: pd.DataFrame,
    single_tests: pd.DataFrame,
    regression_all: pd.DataFrame,
    robust: pd.DataFrame,
    metrics: pd.DataFrame,
    refs: pd.DataFrame,
    key: pd.DataFrame,
    variable_conclusion: pd.DataFrame,
    robust_sig: pd.DataFrame,
    ml_results: pd.DataFrame,
    ml_importance: pd.DataFrame,
    clf_results: pd.DataFrame,
    charts: dict[str, Path],
    excel_path: Path,
) -> Path:
    main_metrics = metrics[(metrics["因变量"] == "log总互动量") & (metrics["模型"] == "主模型")].iloc[0]
    sig_key = key[key["p值"] < 0.05].copy()

    lines = [
        "# 数据分析实验记录",
        "",
        f"数据源：`{SOURCE}`",
        f"结果表：`{excel_path}`",
        "",
        "## 1. 实验概况",
        "",
        f"- 有效样本量：{len(df)}。",
        f"- 传播效果主指标：`总互动量 = 点赞量 + 评论量 + 收藏量 + 转发量`。",
        "- 主模型因变量：`log(总互动量+1)`。",
        "- 主要实验：描述统计、单因素差异检验、多元OLS回归、分互动指标稳健性检验、极端值处理稳健性检验。",
        "",
        "## 2. 单因素检验",
        "",
    ]
    for _, row in single_tests.iterrows():
        lines.append(
            f"- {row['变量']}：{row['检验方法']} p={row['p值']:.4g}，效应量={row['效应量']:.4f}，{row['结论']}。"
        )

    lines.extend(
        [
            "",
            "## 3. 主回归结果",
            "",
            f"- 主模型调整R2={main_metrics['调整R2']:.3f}，R2={main_metrics['R2']:.3f}。",
            "- 参照组设置如下：",
        ]
    )
    for _, row in refs.iterrows():
        lines.append(f"  - {row['变量']}：{row['参照组']}")

    lines.append("")
    lines.append("- p<0.05 的显著影响如下，影响百分比按 `exp(系数)-1` 计算：")
    if sig_key.empty:
        lines.append("  - 主模型未发现 p<0.05 的显著变量。")
    else:
        for _, row in sig_key.sort_values("p值").iterrows():
            lines.append(
                f"  - {row['变量']}：{row['方向']}，系数={row['系数']:.3f}，影响约 {row['影响百分比']:.1f}%，p={row['p值']:.4g}。"
            )

    lines.extend(
        [
            "",
            "## 4. 指标层面结论",
            "",
        ]
    )
    for _, row in variable_conclusion.iterrows():
        if row["控制其他变量后是否有影响"] == "有显著影响":
            lines.append(f"- {row['指标']}：有显著影响。{row['具体影响']}。")
        else:
            lines.append(f"- {row['指标']}：控制其他变量后未发现显著影响。")

    lines.extend(
        [
            "",
            "## 5. 稳健性检验",
            "",
            "- 已分别以 log点赞量、log评论量、log收藏量、log转发量作为因变量重复估计，完整结果见 Excel 的 `稳健性_分因变量回归` 工作表。",
            f"- 稳健性模型中 p<0.05 的结果共 {len(robust_sig)} 条，摘要见 Excel 的 `稳健性_显著结果摘要` 工作表。",
            "- 该部分用于判断某一指标主要影响点赞、评论、收藏还是转发，不建议只看总互动量下结论。",
            "",
            "## 6. 补充说明",
            "",
        ]
    )
    if not ml_results.empty and not clf_results.empty and not ml_importance.empty:
        best_ml = ml_results.iloc[0]
        lines.append(f"- 交叉验证表现最好的模型为 `{best_ml['模型']}`，R2={best_ml['R2均值']:.3f}，RMSE={best_ml['RMSE均值']:.3f}。")
        lines.append(f"- 高互动视频识别 AUC={clf_results.iloc[0]['AUC']:.3f}，F1={clf_results.iloc[0]['F1']:.3f}。")
        lines.append("- 随机森林变量重要性前五位：")
        for _, row in ml_importance.head(5).iterrows():
            lines.append(f"  - {row['变量']}：{row['Permutation重要性均值']:.4f}")
    else:
        lines.append("- 本轮正式结果以描述统计、单因素检验、多元回归及稳健性检验为主，机器学习部分未纳入正式输出。")

    lines.extend(["", "## 7. 输出图表", ""])
    for path in charts.values():
        lines.append(f"- `{path}`")

    report_path = LOG_DIR / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    setup_logging()
    logging.info("开始数据分析实验")
    LOG_DIR.mkdir(exist_ok=True)

    df, invalid_rows = load_data()
    dq = data_quality(df, invalid_rows)
    desc = numeric_descriptive(df)
    freq = category_frequency(df)
    gstats = group_stats(df)
    single_tests = single_factor_tests(df)
    regression_all, robust, metrics, refs = run_regressions(df)
    key = key_effects(regression_all)
    variable_conclusion = variable_level_conclusion(key)
    robust_sig = robustness_summary(robust)
    ml_results = pd.DataFrame()
    ml_importance = pd.DataFrame(columns=["变量", "Permutation重要性均值", "Permutation重要性标准差"])
    clf_results = pd.DataFrame()
    charts = make_charts(df, single_tests, key, ml_importance if not ml_importance.empty else pd.DataFrame({"变量": [], "Permutation重要性均值": []}))

    excel_path = LOG_DIR / "experiment_results.xlsx"
    logging.info("写入结果表：%s", excel_path)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        dq.to_excel(writer, index=False, sheet_name="数据质量")
        invalid_rows.to_excel(writer, index=False, sheet_name="编码复核明细")
        desc.to_excel(writer, index=False, sheet_name="描述统计_数值变量")
        freq.to_excel(writer, index=False, sheet_name="描述统计_分类频数")
        gstats.to_excel(writer, index=False, sheet_name="分组互动表现")
        single_tests.to_excel(writer, index=False, sheet_name="单因素差异检验")
        refs.to_excel(writer, index=False, sheet_name="回归参照组")
        regression_all.to_excel(writer, index=False, sheet_name="主回归与省份稳健")
        key.to_excel(writer, index=False, sheet_name="主回归_具体影响")
        variable_conclusion.to_excel(writer, index=False, sheet_name="指标层面结论")
        robust.to_excel(writer, index=False, sheet_name="稳健性_分因变量回归")
        robust_sig.to_excel(writer, index=False, sheet_name="稳健性_显著结果摘要")
        metrics.to_excel(writer, index=False, sheet_name="回归模型指标")
        if not ml_results.empty:
            ml_results.to_excel(writer, index=False, sheet_name="机器学习模型效果")
        if not ml_importance.empty:
            ml_importance.to_excel(writer, index=False, sheet_name="随机森林变量重要性")
        if not clf_results.empty:
            clf_results.to_excel(writer, index=False, sheet_name="高互动分类实验")

    report_path = write_report(
        df,
        dq,
        single_tests,
        regression_all,
        robust,
        metrics,
        refs,
        key,
        variable_conclusion,
        robust_sig,
        ml_results,
        ml_importance,
        clf_results,
        charts,
        excel_path,
    )
    logging.info("实验完成：%s", report_path)
    print(f"report: {report_path.resolve()}")
    print(f"excel: {excel_path.resolve()}")
    print(f"log: {(LOG_DIR / 'experiment_run.log').resolve()}")


if __name__ == "__main__":
    main()
