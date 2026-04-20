"""
01_data_cleaning.py
-------------------
Loads all raw data, cleans and merges into a single analysis panel,
and saves to data/processed/panel.parquet.

Output columns: see vardict.COLUMN_LABELS for full reference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make nbs/ importable when run from project root or directly
sys.path.insert(0, str(Path(__file__).parent))
from vardict import ESG_THEME_EN, PILLAR_THEMES, ENV_TOPICS, SUB_SCORE_SUFFIXES

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw data ──────────────────────────────────────────────────────────

print("Loading raw data...")

match_raw = pd.read_csv(RAW / "match_scores.csv")
tend_raw  = pd.read_csv(RAW / "tendency_scores.csv")
sent_raw  = pd.read_csv(RAW / "sentiment_scores.csv")
msci_raw  = pd.read_excel(RAW / "MSCI_category_Japan_listed_companies.xlsx",
                          sheet_name="GICS_Classification",
                          dtype={"Stock Code": str})
mcap_raw  = pd.read_excel(RAW / "Market_cap_annual.xlsx",
                          sheet_name="Sheet1",
                          dtype={"Stock Code": str})
rate_raw  = pd.read_excel(RAW / "Japan_interest_rate_annual.xlsx",
                          sheet_name="Sheet1")

print(f"  match_scores  : {match_raw.shape}")
print(f"  tendency_scores: {tend_raw.shape}")
print(f"  sentiment_scores: {sent_raw.shape}")
print(f"  MSCI/GICS     : {msci_raw.shape}")
print(f"  Market cap    : {mcap_raw.shape}")
print(f"  Interest rate : {rate_raw.shape}")

# ── 2. Clean ESG scores ───────────────────────────────────────────────────────

def clean_esg(df: pd.DataFrame, score_prefix: str) -> pd.DataFrame:
    """
    Rename theme columns from Japanese to English and add benchmark
    composite plus descriptive pillar-level indices.

    For tendency data (score_prefix='tend'), also renames the 6 sub-score
    columns for the 3 Environmental topics produced by compute_tendency.py:
    {jp_topic}_{suf} → tend_{en_topic}_{suf}
    where suf ∈ {related_ratio, pos_ratio, neg_ratio, related_mean, pos_mean, neg_mean}

    Parameters
    ----------
    df           : raw ESG dataframe (match or tendency)
    score_prefix : 'match' or 'tend'
    """
    df = df.copy()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)
    df = df.drop(columns=["company_name"])

    # Rename Japanese theme base columns to English
    theme_rename = {jp: f"{score_prefix}_{en}"
                    for jp, en in ESG_THEME_EN.items()
                    if jp in df.columns}

    # For tendency: also rename governance topic sub-score columns
    if score_prefix == "tend":
        env_jp_to_en = {jp: en for jp, en in ESG_THEME_EN.items()
                        if en in ENV_TOPICS}
        for jp, en in env_jp_to_en.items():
            for suf in SUB_SCORE_SUFFIXES:
                col = f"{jp}_{suf}"
                if col in df.columns:
                    theme_rename[col] = f"tend_{en}_{suf}"

    df = df.rename(columns=theme_rename)

    # Composite score: mean across all 23 themes
    theme_cols = [f"{score_prefix}_{en}" for en in ESG_THEME_EN.values()
                  if f"{score_prefix}_{en}" in df.columns]
    df[f"esg_{score_prefix}_composite"] = df[theme_cols].mean(axis=1)

    # Pillar sub-indices
    for pillar, themes in PILLAR_THEMES.items():
        pillar_cols = [f"{score_prefix}_{t}" for t in themes
                       if f"{score_prefix}_{t}" in df.columns]
        df[f"esg_{score_prefix}_{pillar.lower()}"] = df[pillar_cols].mean(axis=1)

    return df


match = clean_esg(match_raw, "match")
tend  = clean_esg(tend_raw,  "tend")

# ── 3. Clean sentiment ────────────────────────────────────────────────────────

sent = sent_raw.copy()
sent["stock_code"] = sent["stock_code"].astype(str).str.strip()
sent["year"] = sent["year"].astype(int)
sent = sent[["stock_code", "year", "sentiment_mean"]]

# ── 4. Reshape market cap wide → long ─────────────────────────────────────────

year_cols = [c for c in mcap_raw.columns if "(JPY)" in str(c)]

mcap = (
    mcap_raw[["Stock Code"] + year_cols]
    .rename(columns={"Stock Code": "stock_code"})
    .melt(id_vars="stock_code", var_name="year_str", value_name="market_cap")
)
mcap["year"] = mcap["year_str"].str.extract(r"(\d{4})").astype(int)
mcap["stock_code"] = mcap["stock_code"].astype(str).str.strip()
mcap["market_cap"] = pd.to_numeric(mcap["market_cap"], errors="coerce")
mcap = mcap.drop(columns="year_str").dropna(subset=["market_cap"])

# ── 5. Clean GICS ─────────────────────────────────────────────────────────────

gics = msci_raw.rename(columns={
    "Stock Code":   "stock_code",
    "Company Name": "company_name",
    "Sector":       "sector",
    "Industry Group": "industry_group",
    "Industry":     "industry",
    "Sub-Industry": "sub_industry",
})
gics["stock_code"] = gics["stock_code"].astype(str).str.strip()
gics = gics[["stock_code", "company_name", "sector",
             "industry_group", "industry", "sub_industry"]]

# ── 6. Clean interest rate ────────────────────────────────────────────────────

rate = rate_raw[["Year", "Interest_Rate_Pct"]].rename(
    columns={"Year": "year", "Interest_Rate_Pct": "interest_rate"}
)
rate["year"] = rate["year"].astype(int)

# ── 7. Merge into panel ───────────────────────────────────────────────────────

print("\nMerging datasets...")

# Start from match scores (defines the universe of ESG observations)
panel = match.merge(tend,  on=["stock_code", "year"], how="inner")
panel = panel.merge(sent,  on=["stock_code", "year"], how="left")
panel = panel.merge(mcap,  on=["stock_code", "year"], how="left")
panel = panel.merge(gics,  on="stock_code",            how="left")
panel = panel.merge(rate,  on="year",                  how="left")

# ── 8. Construct analysis variables ───────────────────────────────────────────

print("Constructing analysis variables...")

# Log market cap
panel["log_mc"] = np.log(panel["market_cap"])

# Excess log market cap: log_mc demeaned by year cross-sectional mean.
# Strips out the common macro valuation environment (including interest rate effects)
# without relying on direct rate deflation (which is numerically unstable for Japan's
# near-zero / negative call rates in 2016–2023).
year_mean_log_mc = panel.groupby("year")["log_mc"].transform("mean")
panel["log_mc_excess"] = panel["log_mc"] - year_mean_log_mc

# Year-over-year log growth (ΔlogMC)
# Use pre-panel market cap (t-1) when available to recover dlog_mc for the
# first panel year (e.g. 2016).  The mcap table may contain years before the
# ESG-score panel starts; bring those in as "prior-year" log MC for diffing.
panel = panel.sort_values(["stock_code", "year"])

# Build a lookup: stock_code × year → log market cap (covers all mcap years)
mcap_log = mcap.copy()
mcap_log["log_mc_ext"] = np.log(mcap_log["market_cap"])
mcap_log = mcap_log[["stock_code", "year", "log_mc_ext"]].dropna()

# For each panel row, look up log_mc from the *prior* calendar year in mcap
panel = panel.merge(
    mcap_log.rename(columns={"year": "year_lag", "log_mc_ext": "log_mc_prior"}),
    left_on=["stock_code", panel["year"] - 1],
    right_on=["stock_code", "year_lag"],
    how="left",
).drop(columns=["year_lag"], errors="ignore")

# dlog_mc: prefer within-panel diff; fall back to external prior-year log MC
panel["dlog_mc_internal"] = panel.groupby("stock_code")["log_mc"].diff()
panel["dlog_mc"] = panel["dlog_mc_internal"].where(
    panel["dlog_mc_internal"].notna(),
    panel["log_mc"] - panel["log_mc_prior"],
)
panel = panel.drop(columns=["dlog_mc_internal", "log_mc_prior"])

# Lagged benchmark ESG variables (t-1), used in secondary summary models
lag_vars = [
    "esg_tend_composite",
    "esg_tend_e", "esg_tend_s", "esg_tend_g", "esg_tend_b",
    "esg_match_composite",
    "sentiment_mean",
]
for v in lag_vars:
    if v in panel.columns:
        panel[f"{v}_lag1"] = panel.groupby("stock_code")[v].shift(1)

# Individual tendency theme lags (t-1)
for v in sorted(c for c in panel.columns if c.startswith("tend_")):
    panel[f"{v}_lag1"] = panel.groupby("stock_code")[v].shift(1)

# Governance topic sub-score lags (t-1) for the decomposition analysis
env_sub_vars = [
    f"tend_{en}_{suf}"
    for en in ENV_TOPICS
    for suf in SUB_SCORE_SUFFIXES
    if f"tend_{en}_{suf}" in panel.columns
]
for v in env_sub_vars:
    lag_col = f"{v}_lag1"
    if lag_col not in panel.columns:   # avoid duplicate if already created above
        panel[lag_col] = panel.groupby("stock_code")[v].shift(1)

# Individual match theme lags (t-1), for topic-level robustness checks
for v in sorted(c for c in panel.columns if c.startswith("match_")):
    panel[f"{v}_lag1"] = panel.groupby("stock_code")[v].shift(1)

# ── 9. Column ordering ────────────────────────────────────────────────────────

id_cols      = ["stock_code", "year", "company_name"]
gics_cols    = ["sector", "industry_group", "industry", "sub_industry"]
outcome_cols = ["market_cap", "log_mc", "log_mc_excess", "dlog_mc"]
macro_cols   = ["interest_rate"]
_esg_seen = set()
esg_idx_cols = []
for _c in (["sentiment_mean"]
           + [c for c in panel.columns if c.startswith("esg_")]
           + [c for c in panel.columns if c.endswith("_lag1")]):
    if _c not in _esg_seen:
        _esg_seen.add(_c)
        esg_idx_cols.append(_c)
theme_cols   = sorted([c for c in panel.columns
                        if (c.startswith("match_") or c.startswith("tend_"))
                        and not c.endswith("_lag1")
                        and not any(f"_{s}" in c for s in SUB_SCORE_SUFFIXES)])

ordered_cols = (id_cols + gics_cols + outcome_cols + macro_cols
                + esg_idx_cols + theme_cols)
# keep any remaining columns at the end
remaining = [c for c in panel.columns if c not in ordered_cols]
panel = panel[ordered_cols + remaining]

# ── 10. Summary ───────────────────────────────────────────────────────────────

print(f"\nPanel shape          : {panel.shape}")
print(f"Companies (total)    : {panel.stock_code.nunique()}")
print(f"Years                : {sorted(panel.year.unique())}")
print(f"Companies with MC    : {panel.market_cap.notna().sum()} obs")
print(f"Companies w/ dlog_mc : {panel.dlog_mc.notna().sum()} obs")
print(f"Sectors              : {panel.sector.nunique()} "
      f"({panel.sector.notna().sum()} obs with sector)")

missing_mc = panel.market_cap.isna().sum()
missing_gics = panel.sector.isna().sum()
print(f"\nMissing market cap   : {missing_mc} obs "
      f"({missing_mc / len(panel):.1%})")
print(f"Missing GICS sector  : {missing_gics} obs "
      f"({missing_gics / len(panel):.1%})")

# ── 11. Save ──────────────────────────────────────────────────────────────────

out_path = PROCESSED / "panel.parquet"
panel.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
print("Done.")
