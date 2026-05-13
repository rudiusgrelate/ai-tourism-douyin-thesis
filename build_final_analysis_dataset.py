from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "数据" / "最终版数据.xlsx"
OUT_XLSX = ROOT / "数据" / "final_analysis_722.xlsx"
OUT_CSV = ROOT / "数据" / "final_analysis_722.csv"

# Keep this exclusion explicit so the analysis sample used in all scripts is identical.
EXCLUDED_LINKS = {
    "https://www.douyin.com/video/7174963771973143845",
}


def main() -> None:
    df = pd.read_excel(SOURCE)
    df["链接"] = df["链接"].astype(str).str.strip()
    final_df = df[~df["链接"].isin(EXCLUDED_LINKS)].copy()
    final_df.to_excel(OUT_XLSX, index=False)
    final_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print({"source_rows": len(df), "final_rows": len(final_df), "out_xlsx": str(OUT_XLSX), "out_csv": str(OUT_CSV)})


if __name__ == "__main__":
    main()
