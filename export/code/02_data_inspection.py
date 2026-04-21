"""
nbs/02_data_inspection.py
Generates notebooks/02_data_inspection.ipynb.

Run:
    python nbs/02_data_inspection.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "src" / "02_data_inspection.ipynb"


def code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

cells += [md("# 02 Data Inspection")]

cells += [code("""\
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(".")))
from vardict import COLUMN_LABELS, ESG_THEME_PILLAR

PROCESSED = Path("../data/processed")

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

PILLAR_COLORS = {"E": "#27ae60", "S": "#2980b9", "G": "#8e44ad", "B": "#e67e22"}
""")]

cells += [code("""\
panel = pd.read_parquet(PROCESSED / "panel.parquet")
years = sorted(panel.year.unique())
THEME_LAG_VARS = sorted([c for c in panel.columns if c.startswith("tend_") and c.endswith("_lag1")])
MATCH_LAG_VARS = [v.replace("tend_", "match_") for v in THEME_LAG_VARS if v.replace("tend_", "match_") in panel.columns]

THEME_ORDER = []
THEME_PILLAR_LIST = []
for pillar in ("E", "S", "G", "B"):
    for theme, p in ESG_THEME_PILLAR.items():
        if p == pillar:
            THEME_ORDER.append(theme)
            THEME_PILLAR_LIST.append(pillar)

main_topics = panel.dropna(subset=["log_mc"] + THEME_LAG_VARS).copy()
main_bench = panel.dropna(subset=["log_mc", "esg_tend_composite_lag1"]).copy()

print(f"Shape            : {panel.shape}")
print(f"Companies        : {panel.stock_code.nunique():,}")
print(f"Years            : {years}")
print(f"Topic sample     : {len(main_topics):,} obs | {main_topics.stock_code.nunique():,} companies")
print(f"Benchmark sample : {len(main_bench):,} obs | {main_bench.stock_code.nunique():,} companies")
""")]

cells += [
    md("## Figure Reference"),
    code("""\
FIGURE_META = {
    "fig01": {
        "name": "Sample Coverage by Year",
        "description": (
            "Number of companies with ESG data vs. with matched market cap data per year. "
            "The growing N reflects new listings and expanding coverage."
        ),
    },
    "fig02": {
        "name": "Missing Data Rates",
        "description": (
            "Proportion of missing values (%) for key outcome, benchmark, and topic-sample variables."
        ),
    },
    "fig03": {
        "name": "Market Cap Variable Distributions by Year",
        "panels": {
            "a": "Log market cap",
            "b": "Excess log market cap (market-demeaned)",
            "c": "Annual log market cap growth",
        },
        "description": (
            "Box plots per year. log_mc_excess is centered at 0 each year by construction."
        ),
    },
    "fig04": {
        "name": "Benchmark ESG Score Trends",
        "panels": {
            "a": "ESG match score (composite)",
            "b": "ESG tendency score (composite)",
            "c": "ESG sentiment",
        },
        "description": (
            "Cross-sectional mean (line) +/- 1 SD (shaded band) per year."
        ),
    },
    "fig05": {
        "name": "Topic-Level ESG Trends",
        "panels": {
            "a": "Match score z-scores by topic",
            "b": "Tendency score z-scores by topic",
            "c": "Raw mean topic scores",
        },
        "description": (
            "All 23 topics grouped by E / S / G / B. Heatmaps remove between-topic scale differences."
        ),
    },
    "fig06": {
        "name": "Company Count by GICS Sector",
        "description": (
            "Unique companies per GICS sector. Small sectors remain visible before heterogeneity analysis."
        ),
    },
    "fig07": {
        "name": "Sector-Level Distributions",
        "panels": {
            "a": "Benchmark ESG tendency by sector",
            "b": "Annual market cap growth by sector",
        },
        "description": (
            "Horizontal box plots by GICS sector. Sectors ordered by median benchmark ESG tendency score."
        ),
    },
    "fig08": {
        "name": "Topic Correlations with Market Cap Outcomes",
        "panels": {
            "a": "Correlation with log market cap",
            "b": "Correlation with annual log market cap growth",
        },
        "description": (
            "Pearson correlations between each current-year ESG tendency topic and the two outcome variables."
        ),
    },
}
print("FIGURE_META keys:", list(FIGURE_META.keys()))
"""),
]

cells += [
    md("## Helpers"),
    code("""\
def label_panel(ax, letter, name, fontsize=8.5):
    ax.text(
        0.02, 0.98, f"({letter})  {name}",
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8),
    )


def boxplot_by_year(ax, col, year_list, color="#2c3e50"):
    data = [panel.loc[panel.year == y, col].dropna().values for y in year_list]
    bp = ax.boxplot(
        data,
        positions=range(len(year_list)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whiskerprops=dict(linewidth=0.8, color="#666"),
        capprops=dict(linewidth=0.8, color="#666"),
        medianprops=dict(color="white", linewidth=1.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_xticks(range(len(year_list)))
    ax.set_xticklabels(year_list, rotation=45, ha="right", fontsize=7.5)
    ax.axhline(0, color="#ccc", linewidth=0.7, linestyle="--")


def zscore_rows(mat):
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (mat - mu) / sd
"""),
]

cells += [
    md("## 1. Basic Dimensions"),
    code("""\
panel[[
    "log_mc", "log_mc_excess", "dlog_mc",
    "esg_tend_composite", "esg_match_composite", "sentiment_mean",
    "roa_pct", "ltdebt_assets_pct", "rnd_share_pct",
    "ibes_eps_mean", "ibes_rec_mean",
]].describe().round(4)
"""),
]

cells += [
    md("## 2. Sample Coverage"),
    code("""\
n_esg  = panel.groupby("year")["stock_code"].count()
n_mc   = panel.dropna(subset=["market_cap"]).groupby("year")["stock_code"].count()
n_full = panel.dropna(subset=["market_cap"] + THEME_LAG_VARS).groupby("year")["stock_code"].count()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(years, n_esg.reindex(years).values,  "o-",  lw=1.8, ms=5, label="ESG data")
ax.plot(years, n_mc.reindex(years).values,   "s-",  lw=1.8, ms=5, label="ESG + market cap")
ax.plot(years, n_full.reindex(years).values, "^--", lw=1.6, ms=5, label="Topic-level sample", alpha=0.8)
ax.set_ylabel("Number of companies")
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)
ax.legend(fontsize=8, framealpha=0.7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 3. Missing Data"),
    code("""\
check_vars = [
    "market_cap", "log_mc", "log_mc_excess", "dlog_mc",
    "sector", "interest_rate",
    "esg_tend_composite", "esg_match_composite", "sentiment_mean",
    "esg_tend_composite_lag1", "esg_match_composite_lag1",
    THEME_LAG_VARS[0], MATCH_LAG_VARS[0],
    "roa_pct", "ltdebt_assets_pct", "debt_equity_pct",
    "rnd_share_pct", "total_revenue",
    "ibes_eps_mean", "ibes_rev_mean", "ibes_rec_mean", "ibes_pt_mean",
]
missing  = (panel[check_vars].isna().mean() * 100).sort_values(ascending=True)
readable = [COLUMN_LABELS.get(v, v) for v in missing.index]

fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.barh(readable, missing.values, color="#c0392b", alpha=0.72, height=0.62)
ax.set_xlabel("Missing (%)")
ax.axvline(5, color="#aaa", linestyle="--", linewidth=0.8, label="5% threshold")
ax.legend(fontsize=8)
for bar, val in zip(bars, missing.values):
    if val > 0.3:
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7.5)
plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 4. Market Cap Variables"),
    code("""\
years_dlog = sorted(panel.dropna(subset=["dlog_mc"]).year.unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
specs = [
    ("log_mc",        years,      "a", "Log Market Cap",               "#2980b9"),
    ("log_mc_excess", years,      "b", "Excess Log Market Cap",        "#27ae60"),
    ("dlog_mc",       years_dlog, "c", "Annual Log Market Cap Growth", "#e67e22"),
]
for ax, (col, yr, lbl, name, color) in zip(axes, specs):
    boxplot_by_year(ax, col, yr, color=color)
    label_panel(ax, lbl, name)
    ax.set_xlabel("Year")

plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 5. Benchmark ESG Score Trends"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
specs = [
    ("esg_match_composite", "a", "ESG Match Score (composite)",    "#2980b9"),
    ("esg_tend_composite",  "b", "ESG Tendency Score (composite)", "#e67e22"),
    ("sentiment_mean",      "c", "ESG Sentiment",                  "#8e44ad"),
]
for ax, (col, lbl, name, color) in zip(axes, specs):
    grp = panel.groupby("year")[col]
    mu  = grp.mean().reindex(years)
    sd  = grp.std().reindex(years)
    ax.plot(years, mu.values, color=color, lw=1.8, marker="o", ms=4)
    ax.fill_between(years, (mu - sd).values, (mu + sd).values, color=color, alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    label_panel(ax, lbl, name)

plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 6. Topic-Level ESG Trends"),
    code("""\
mat_match = np.array([[panel.loc[panel.year == y, f"match_{t}"].mean() for y in years]
                       for t in THEME_ORDER])
mat_tend  = np.array([[panel.loc[panel.year == y, f"tend_{t}"].mean() for y in years]
                       for t in THEME_ORDER])
mat_z_match = zscore_rows(mat_match)
mat_z_tend  = zscore_rows(mat_tend)
vmax_z = max(np.abs(mat_z_match).max(), np.abs(mat_z_tend).max())
norm_z = TwoSlopeNorm(vmin=-vmax_z, vcenter=0, vmax=vmax_z)
means_match = mat_match.mean(axis=1)
means_tend  = mat_tend.mean(axis=1)
row_labels  = [t.replace("_", " ") for t in THEME_ORDER]
bar_colors  = [PILLAR_COLORS[p] for p in THEME_PILLAR_LIST]
pillar_handles = [Patch(color=PILLAR_COLORS[p], label=p) for p in ("E", "S", "G", "B")]

fig = plt.figure(figsize=(30, 9))
gs  = fig.add_gridspec(1, 3, width_ratios=[3.5, 3.5, 2.2], wspace=0.06)
ax_m = fig.add_subplot(gs[0])
ax_t = fig.add_subplot(gs[1])
ax_r = fig.add_subplot(gs[2])

for ax, mat_z, lbl, name in [
    (ax_m, mat_z_match, "a", "Match Score"),
    (ax_t, mat_z_tend,  "b", "Tendency Score"),
]:
    im = ax.imshow(mat_z, aspect="auto", cmap="RdBu_r", norm=norm_z)
    cb = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Std. within Topic", fontsize=7.5)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(THEME_ORDER)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    for ytl, pillar in zip(ax.get_yticklabels(), THEME_PILLAR_LIST):
        ytl.set_color(PILLAR_COLORS[pillar])
    ax.legend(handles=pillar_handles, title="Pillar", fontsize=7, title_fontsize=7, loc="lower right")
    label_panel(ax, lbl, name)

y_pos = np.arange(len(THEME_ORDER))
ax_r.barh(y_pos + 0.2, means_match, height=0.35, color=bar_colors, alpha=0.45, label="Match")
ax_r.barh(y_pos - 0.2, means_tend,  height=0.35, color=bar_colors, alpha=0.85, label="Tendency")
ax_r.axvline(0, color="#888", linewidth=0.7)
ax_r.set_yticks(y_pos)
ax_r.set_yticklabels([])
ax_r.set_xlabel("Mean score", fontsize=7.5)
ax_r.tick_params(axis="x", labelsize=7)
ax_r.legend(fontsize=7, loc="lower right")
label_panel(ax_r, "c", "Mean (reference)")

plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 7. Sector Analysis"),
    code("""\
sector_n = (
    panel.drop_duplicates("stock_code")
         .dropna(subset=["sector"])
         .sector.value_counts()
         .sort_values()
)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(sector_n.index, sector_n.values, color="#34495e", alpha=0.75, height=0.6)
ax.set_xlabel("Number of companies")
for bar, val in zip(bars, sector_n.values):
    ax.text(val + 8, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=8)
plt.tight_layout()
plt.show()
"""),
    code("""\
sector_order = (
    panel.dropna(subset=["sector", "esg_tend_composite"])
         .groupby("sector")["esg_tend_composite"]
         .median()
         .sort_values()
         .index.tolist()
)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
specs = [
    ("esg_tend_composite", "a", "Benchmark ESG Tendency by Sector", "#e67e22"),
    ("dlog_mc",            "b", "Annual Market Cap Growth by Sector", "#2980b9"),
]
for ax, (col, lbl, name, color) in zip(axes, specs):
    data = [panel.loc[panel.sector == s, col].dropna().values for s in sector_order]
    bp = ax.boxplot(
        data, vert=False, patch_artist=True, showfliers=False,
        whiskerprops=dict(linewidth=0.8, color="#666"),
        capprops=dict(linewidth=0.8, color="#666"),
        medianprops=dict(color="white", linewidth=1.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_yticks(range(1, len(sector_order) + 1))
    ax.set_yticklabels(sector_order, fontsize=8)
    ax.axvline(0, color="#ccc", linewidth=0.7, linestyle="--")
    label_panel(ax, lbl, name)

plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("## 8. Topic Correlations with Outcomes"),
    code("""\
outcome_pairs = [("log_mc", "a", "Correlation with log MC"), ("dlog_mc", "b", "Correlation with Δlog MC")]
tend_cols_cur = [f"tend_{t}" for t in THEME_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (out_col, lbl, name) in zip(axes, outcome_pairs):
    rs = [panel[[tc, out_col]].dropna().corr().iloc[0, 1] for tc in tend_cols_cur]
    bar_colors = [PILLAR_COLORS[p] for p in THEME_PILLAR_LIST]
    x = range(len(THEME_ORDER))
    ax.bar(x, rs, color=bar_colors, width=0.7, alpha=0.82, edgecolor="white")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", "\\n") for t in THEME_ORDER], rotation=90, fontsize=6.5)
    for xtl, pillar in zip(ax.get_xticklabels(), THEME_PILLAR_LIST):
        xtl.set_color(PILLAR_COLORS[pillar])
    ax.set_ylabel("Pearson r")
    ax.legend(handles=[Patch(color=PILLAR_COLORS[p], label=p) for p in ("E", "S", "G", "B")],
              title="Pillar", fontsize=7.5, title_fontsize=7.5)
    label_panel(ax, lbl, name)

plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("""\
## 9. Fundamentals Trends

Cross-sectional mean ± 1 SD per year for ROA, long-term leverage, and R&D intensity.
R&D share is shown only for companies that disclose it (~35% of the sample).
"""),
    code("""\
fund_specs = [
    ("roa_pct",            "a", "ROA (%)",                "#2980b9"),
    ("ltdebt_assets_pct",  "b", "LT Debt / Assets (%)",   "#e67e22"),
    ("rnd_share_pct",      "c", "R&D / Revenue (%, disclosers only)", "#27ae60"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, lbl, name, color) in zip(axes, fund_specs):
    grp = panel.dropna(subset=[col]).groupby("year")[col]
    mu  = grp.mean().reindex(years)
    sd  = grp.std().reindex(years)
    n   = grp.count().reindex(years)
    ax.plot(years, mu.values, color=color, lw=1.8, marker="o", ms=4)
    ax.fill_between(years, (mu - sd).values, (mu + sd).values, color=color, alpha=0.15)
    ax2 = ax.twinx()
    ax2.bar(years, n.values, alpha=0.12, color=color, width=0.6, label="N")
    ax2.set_ylabel("N obs", fontsize=7.5, color="#999")
    ax2.tick_params(axis="y", labelsize=7, colors="#999")
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    label_panel(ax, lbl, name)

plt.suptitle("Fundamentals Trends  [mean ± 1 SD, bars = N obs]", fontsize=10)
plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("""\
## 10. IBES Analyst Consensus Trends

December month-end FY1 consensus, restricted to companies covered by IBES (~33% of panel).
Recommendation scale: 1 = Strong Buy → 5 = Sell (lower is more bullish).
"""),
    code("""\
ibes_panel = panel.dropna(subset=["ibes_rec_mean"])
ibes_years = sorted(ibes_panel.year.unique())

ibes_specs = [
    ("ibes_rec_mean",  "a", "Analyst Recommendation (1=Strong Buy, 5=Sell)", "#8e44ad"),
    ("ibes_eps_mean",  "b", "IBES EPS Consensus (FY1 mean)",                  "#2980b9"),
    ("ibes_pt_mean",   "c", "IBES Price Target (mean)",                        "#e67e22"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, lbl, name, color) in zip(axes, ibes_specs):
    grp = ibes_panel.groupby("year")[col]
    mu  = grp.mean().reindex(ibes_years)
    sd  = grp.std().reindex(ibes_years)
    ax.plot(ibes_years, mu.values, color=color, lw=1.8, marker="o", ms=4)
    ax.fill_between(ibes_years, (mu - sd).values, (mu + sd).values, color=color, alpha=0.15)
    ax.set_xlabel("Year")
    ax.set_xticks(ibes_years)
    ax.set_xticklabels(ibes_years, rotation=45, fontsize=7.5)
    label_panel(ax, lbl, name)

n_ibes_co = ibes_panel.stock_code.nunique()
n_ibes_obs = len(ibes_panel)
plt.suptitle(f"IBES Analyst Consensus Trends  [mean ± 1 SD, N={n_ibes_co:,} companies, {n_ibes_obs:,} obs]", fontsize=10)
plt.tight_layout()
plt.show()
"""),
    code("""\
# IBES recommendation distribution by sector
sector_order_ibes = (
    ibes_panel.dropna(subset=["sector"])
              .groupby("sector")["ibes_rec_mean"]
              .median()
              .sort_values()
              .index.tolist()
)

fig, ax = plt.subplots(figsize=(8, 4.5))
data = [ibes_panel.loc[ibes_panel.sector == s, "ibes_rec_mean"].dropna().values
        for s in sector_order_ibes]
bp = ax.boxplot(
    data, vert=False, patch_artist=True, showfliers=False,
    whiskerprops=dict(linewidth=0.8, color="#666"),
    capprops=dict(linewidth=0.8, color="#666"),
    medianprops=dict(color="white", linewidth=1.8),
)
for patch in bp["boxes"]:
    patch.set_facecolor("#8e44ad")
    patch.set_alpha(0.72)
ax.set_yticks(range(1, len(sector_order_ibes) + 1))
ax.set_yticklabels(sector_order_ibes, fontsize=8)
ax.axvline(3, color="#e74c3c", lw=0.9, ls="--", label="Neutral (3.0)")
ax.set_xlabel("Mean Analyst Recommendation (1=Strong Buy → 5=Sell)")
ax.legend(fontsize=8)
ax.set_title("Analyst Recommendation by GICS Sector (IBES-covered companies)", fontsize=9)
plt.tight_layout()
plt.show()
"""),
]

cells += [
    md("""\
## 11. Fundamentals & IBES Correlations with Market Cap Outcomes

Pearson correlations between each control variable and the two outcome variables (log_mc, dlog_mc).
Helps identify which controls are most important for model specification.
"""),
    code("""\
control_vars = [
    "roa_pct", "ltdebt_assets_pct", "debt_equity_pct",
    "rnd_share_pct", "ibes_rec_mean", "ibes_eps_mean", "ibes_pt_mean",
]
control_labels = [
    "ROA (%)", "LT Debt/Assets (%)", "Debt/Equity (%)",
    "R&D/Revenue (%)", "IBES Rec.", "IBES EPS", "IBES Price Target",
]
outcome_pairs = [
    ("log_mc",  "a", "Correlation with log Market Cap", "#2980b9"),
    ("dlog_mc", "b", "Correlation with Δlog Market Cap", "#e67e22"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (out_col, lbl, title, color) in zip(axes, outcome_pairs):
    rs = [
        panel[[v, out_col]].dropna().corr().iloc[0, 1]
        for v in control_vars
    ]
    bar_c = [color if r > 0 else "#e74c3c" for r in rs]
    x = range(len(control_vars))
    bars = ax.bar(x, rs, color=bar_c, width=0.6, alpha=0.82, edgecolor="white")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(control_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r")
    for bar, r in zip(bars, rs):
        va = "bottom" if r >= 0 else "top"
        offset = 0.003 if r >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2, r + offset,
                f"{r:.3f}", ha="center", va=va, fontsize=7.5)
    label_panel(ax, lbl, title)

plt.suptitle("Pairwise Correlations: Controls vs Market Cap Outcomes", fontsize=10)
plt.tight_layout()
plt.show()
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (ZDP02n)",
            "language": "python",
            "name": "zdp02n",
        },
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook written → {OUT}")
