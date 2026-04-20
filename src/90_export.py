"""
src/90_export.py
Generates the complete export/ folder for Jiazi ingestion.

Implements AnaSOP §8–9: 9 figures + 13 tables for the 3 × 7 governance-topic analysis.
  - 7 governance topics (corp_governance, security, risk_compliance, management_ops,
    materiality, stakeholder_engagement, corporate_philosophy)
  - 3 specifications: M1 (match), M2 (match + sentiment), M3 (pos_mean + neg_mean)
  - Sector heterogeneity: all three specs, results saved separately
  - Robustness: M1

Run (from project root):
    python src/90_export.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vardict import ENV_TOPICS, ENV_TOPIC_LABELS, SUB_SCORE_SUFFIXES

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]
PROC   = ROOT / "data" / "processed"
EXPORT = ROOT / "export"
FIGS   = EXPORT / "figures"
TABLES = EXPORT / "tables"
CODE   = EXPORT / "code"
META   = EXPORT / "metadata"
for d in (FIGS, TABLES, CODE, META):
    d.mkdir(parents=True, exist_ok=True)

FIG_DPI = 300

# ── Clean up old export artifacts ─────────────────────────────────────────────
_old = [
    # old social-era figures
    "fig01_env_disclosure_trends.png", "fig02_pos_neg_ratio_trends.png",
    "fig03_market_cap_distributions.png", "fig04_coef_comparison_M1_M3.png",
    "fig05_sector_M1.png", "fig06_sector_M2.png",
    "fig07_sector_M3.png", "fig08_robustness_M1.png",
    # old combined M3 and old split naming
    "fig08_sector_M3.png",
    "fig08a_sector_M3_pos.png", "fig08b_sector_M3_neg.png",
    "fig09_robustness_M1.png",
    # old social-era tables
    "Table1_DescriptiveStats.xlsx", "Table2_CrossTopic_Summary.xlsx",
    "Table3_SectorHet_M2.xlsx", "Table4_SectorHet_M3.xlsx",
    "Table5_Robustness_M1.xlsx",
    "S1_Coef_EmployeeHealth.xlsx", "S2_Coef_CustomerValue.xlsx",
    "S3_Coef_HumanRights.xlsx", "S4_Coef_HumanCapital.xlsx",
    "S5_Coef_DE&I.xlsx", "S6_Coef_CommunityCoexistence.xlsx",
    "S7_SectorHet_M1.xlsx",
]
for _f in _old:
    for _d in (FIGS, TABLES):
        _p = _d / _f
        if _p.exists():
            _p.unlink()
            print(f"  Removed old: {_f}")

# ── Plot defaults ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

DIM_COLORS = {
    "coverage":  "#2980b9",
    "sentiment": "#8e44ad",
    "pos":       "#27ae60",
    "neg":       "#e74c3c",
}
TOPIC_COLORS = {
    "corp_governance":        "#2980b9",
    "security":               "#c0392b",
    "risk_compliance":        "#e67e22",
    "management_ops":         "#16a085",
    "materiality":            "#8e44ad",
    "stakeholder_engagement": "#27ae60",
    "corporate_philosophy":   "#7f8c8d",
}
N_TOPICS = len(ENV_TOPICS)   # 7

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
panel   = pd.read_parquet(PROC / "panel.parquet")
years   = sorted(panel.year.unique())
sectors = sorted(panel.dropna(subset=["sector"]).sector.unique())
print(f"  Panel: {panel.shape[0]:,} obs | {panel.stock_code.nunique():,} companies")


# ── Shared helpers ────────────────────────────────────────────────────────────
def panel_ols(df, y, x_vars, entity_fe=True, time_fe=True, add_rate=False):
    if isinstance(x_vars, str):
        x_vars = [x_vars]
    rate_col  = ["interest_rate"] if add_rate else []
    cols_need = [y] + x_vars + rate_col + ["stock_code", "year"]
    df_sub    = df[cols_need].dropna().copy().reset_index(drop=True)
    if entity_fe:
        dm_cols = [y] + x_vars
        em = df_sub.groupby("stock_code")[dm_cols].transform("mean")
        for col in dm_cols:
            df_sub[f"w_{col}"] = df_sub[col] - em[col]
        y_f = f"w_{y}"
        x_f = " + ".join(f"w_{v}" for v in x_vars)
    else:
        y_f = y
        x_f = " + ".join(x_vars)
    formula = f"{y_f} ~ {x_f}"
    if time_fe:
        formula += " + C(year)"
    if add_rate:
        formula += " + interest_rate"
    result = smf.ols(formula, data=df_sub).fit(
        cov_type="cluster", cov_kwds={"groups": df_sub["stock_code"]}
    )
    return result, len(df_sub), df_sub["stock_code"].nunique()


def get_coef(result, var, entity_fe=True):
    pname = f"w_{var}" if entity_fe else var
    if pname not in result.params.index:
        return None, None, ""
    coef  = result.params[pname]
    se    = result.bse[pname]
    pval  = result.pvalues[pname]
    stars = ("***" if pval < 0.01 else "**" if pval < 0.05 else
             "*"   if pval < 0.10 else "")
    return coef, se, stars


def label_panel(ax, letter, name, fontsize=8.5):
    ax.text(0.02, 0.98, f"({letter})  {name}",
            transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8))


def hide_unused(axes_flat, n_used):
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


def topic_model_specs(topic):
    m  = f"match_{topic}_lag1"
    s  = "sentiment_mean_lag1"
    pm = f"tend_{topic}_pos_mean_lag1"
    nm = f"tend_{topic}_neg_mean_lag1"
    return [
        {"label": "M1", "name": "Relatedness",
         "rows": [(m,  "Relatedness",        "coverage")]},
        {"label": "M2", "name": "Relatedness + Overall Sentiment",
         "rows": [(m,  "Relatedness",        "coverage"),
                  (s,  "Overall Sentiment",  "sentiment")]},
        {"label": "M3", "name": "Positive Mean Score + Negative Mean Score",
         "rows": [(pm, "Positive Mean Score","pos"),
                  (nm, "Negative Mean Score","neg")]},
    ]


def run_topic_models(df, topic):
    results = []
    for sp in topic_model_specs(topic):
        x_vars = [row[0] for row in sp["rows"]]
        df_m   = df.dropna(subset=["log_mc"] + x_vars + ["stock_code", "year"]).copy()
        res, n, nc = panel_ols(df_m, "log_mc", x_vars)
        coefs = {}
        for (col, label, dim) in sp["rows"]:
            c, se, stars = get_coef(res, col)
            coefs[col] = {"coef": c, "se": se, "stars": stars,
                          "label": label, "dim": dim}
        results.append({**sp, "result": res, "coefs": coefs, "n": n, "nc": nc})
    return results


def run_sector_spec(df, topic, spec_label):
    if spec_label == "M1":
        x_vars  = [f"match_{topic}_lag1"]
        var_map = {"match": f"match_{topic}_lag1"}
    elif spec_label == "M2":
        x_vars  = [f"match_{topic}_lag1", "sentiment_mean_lag1"]
        var_map = {"match":     f"match_{topic}_lag1",
                   "sentiment": "sentiment_mean_lag1"}
    elif spec_label == "M3":
        x_vars  = [f"tend_{topic}_pos_mean_lag1", f"tend_{topic}_neg_mean_lag1"]
        var_map = {"pos_mean": f"tend_{topic}_pos_mean_lag1",
                   "neg_mean": f"tend_{topic}_neg_mean_lag1"}
    else:
        raise ValueError(f"Unknown spec: {spec_label}")

    rows = []
    for sec in sectors:
        df_s = df[df.sector == sec].dropna(subset=["log_mc"] + x_vars).copy()
        nc_s = df_s.stock_code.nunique()
        row  = {"Sector": sec, "N obs": len(df_s), "N co.": nc_s}
        try:
            res_s, n_s, _ = panel_ols(df_s, "log_mc", x_vars)
            for key, col in var_map.items():
                c, se, stars = get_coef(res_s, col)
                row[f"coef_{key}"] = c
                row[f"se_{key}"]   = se
                row[f"\u03b2_{key}"]  = f"{c:.3f}{stars}" if c is not None else "\u2014"
                row[f"SE_{key}"]   = f"({se:.3f})"    if se is not None else ""
            row["Within R\u00b2"] = round(res_s.rsquared, 4)
        except Exception:
            for key in var_map:
                row[f"coef_{key}"] = None
                row[f"se_{key}"]   = None
        rows.append(row)
    return pd.DataFrame(rows)


def run_robustness_m1(df, topic):
    match_var = f"match_{topic}_lag1"
    rows = []
    for label, y_col, excl2020, balanced, rate_ctrl, winsor in [
        ("Main M1",            "log_mc",  False, False, False, False),
        ("\u0394logMC",        "dlog_mc", False, False, False, False),
        ("Rate ctrl (no YFE)","log_mc",  False, False, True,  False),
        ("Excl. 2020",        "log_mc",  True,  False, False, False),
        ("Balanced panel",    "log_mc",  False, True,  False, False),
        ("Win. \u0394logMC",  "dlog_mc", False, False, False, True),
    ]:
        y    = "dlog_win" if winsor else y_col
        df_c = df.dropna(subset=[y_col, match_var]).copy()
        if winsor:
            p01, p99 = df_c[y_col].quantile([0.01, 0.99])
            df_c["dlog_win"] = df_c[y_col].clip(p01, p99)
        if excl2020:
            df_c = df_c[df_c.year != 2020].copy()
        if balanced:
            yc   = df_c.groupby("stock_code")["year"].count()
            df_c = df_c[df_c.stock_code.isin(yc[yc == df_c.year.nunique()].index)].copy()
        try:
            res, n, nc = panel_ols(df_c, y, match_var,
                                   time_fe=(not rate_ctrl), add_rate=rate_ctrl)
            c, se, stars = get_coef(res, match_var)
            rows.append({"Check": label, "coef": c, "se": se, "stars": stars,
                         "\u03b2": f"{c:.4f}{stars}", "SE": f"({se:.4f})",
                         "Outcome": y, "N obs": f"{n:,}", "N co.": f"{nc:,}",
                         "R\u00b2": f"{res.rsquared:.3f}"})
        except Exception as e:
            rows.append({"Check": label, "\u03b2": f"err:{e}",
                         "coef": None, "se": None})
    return pd.DataFrame(rows)


# ── Run all regressions ───────────────────────────────────────────────────────
print("Running regressions...")
topic_results  = {t: run_topic_models(panel, t) for t in ENV_TOPICS}
sector_results = {
    spec: {t: run_sector_spec(panel, t, spec) for t in ENV_TOPICS}
    for spec in ("M1", "M2", "M3")
}
robust_results = {t: run_robustness_m1(panel, t) for t in ENV_TOPICS}
print("  Regressions done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("Saving figures...")

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ── fig01: Disclosure trends — 2×4: (a–g) relatedness per topic, (h) sentiment
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax    = axf[idx]
    color = TOPIC_COLORS[topic]
    mu_m  = panel.groupby("year")[f"match_{topic}"].mean().reindex(years)
    ax.plot(years, mu_m.values, color=color, lw=1.8, marker="o", ms=4)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    ax.set_ylabel("Relatedness", fontsize=8)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])

ax = axf[7]
mu_s = panel.groupby("year")["sentiment_mean"].mean().reindex(years)
ax.plot(years, mu_s.values, color=DIM_COLORS["sentiment"], lw=1.8, marker="s", ms=4)
ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
ax.set_xlabel("Year")
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, fontsize=7.5)
ax.set_ylabel("Overall Sentiment", fontsize=8)
label_panel(ax, "h", "Overall Sentiment")

plt.tight_layout()
fig.savefig(FIGS / "fig01_gov_disclosure_trends.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig02: pos_ratio vs neg_ratio trends — 3×3, 7 panels ─────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax     = axf[idx]
    pr_col = f"tend_{topic}_pos_ratio"
    nr_col = f"tend_{topic}_neg_ratio"
    mu_pr  = panel.groupby("year")[pr_col].mean().reindex(years)
    mu_nr  = panel.groupby("year")[nr_col].mean().reindex(years)
    ax.plot(years, mu_pr.values, color=DIM_COLORS["pos"],
            lw=1.8, marker="o", ms=4, label="Positive Ratio")
    ax.plot(years, mu_nr.values, color=DIM_COLORS["neg"],
            lw=1.8, marker="s", ms=4, label="Negative Ratio")
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=7)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
hide_unused(axf, N_TOPICS)
plt.tight_layout()
fig.savefig(FIGS / "fig02_pos_neg_ratio_trends.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig03: Market cap distributions — 2 rows × 1 col ─────────────────────────
# Distinctive styling: dark-teal / coral palette, IQR fill + median dot overlay
_PALETTE = {"log_mc": "#264653", "dlog_mc": "#e76f51"}

def boxplot_by_year(ax, col, year_list):
    color = _PALETTE[col]
    data  = [panel.loc[panel.year == y, col].dropna().values for y in year_list]
    bp = ax.boxplot(
        data,
        positions=range(len(year_list)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(linewidth=0),
        whiskerprops=dict(linewidth=1.0, color=color, alpha=0.6),
        capprops=dict(linewidth=1.2, color=color),
        medianprops=dict(color="white", linewidth=2.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    # overlay median dot
    medians = [np.median(d) for d in data]
    ax.scatter(range(len(year_list)), medians,
               color="white", s=22, zorder=4, linewidths=0.8,
               edgecolors=color)
    ax.set_xticks(range(len(year_list)))
    ax.set_xticklabels(year_list, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#bbb", linewidth=0.7, linestyle="--")
    # light year-band shading
    for i in range(0, len(year_list), 2):
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.04, zorder=0)

years_dlog = sorted(panel.dropna(subset=["dlog_mc"]).year.unique())
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
for ax, (col, yr, lbl, name) in zip(axes, [
    ("log_mc",  years,      "a", "Log Market Capitalization (log JPY)"),
    ("dlog_mc", years_dlog, "b", "Annual Log Market Cap Growth (\u0394log JPY)"),
]):
    boxplot_by_year(ax, col, yr)
    label_panel(ax, lbl, name)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel(name, fontsize=9, color=_PALETTE[col])
    ax.tick_params(axis="y", labelcolor=_PALETTE[col])
    if col == "log_mc":
        ax.set_ylim(20, 30)

plt.tight_layout()
fig.savefig(FIGS / "fig03_market_cap_distributions.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig04: Sector profile radar chart ─────────────────────────────────────────
def _radar_chart(panel, topics, topic_labels, colors, save_path):
    """
    Per-sector radar chart. Three metrics z-scored per topic column:
      z = (sector_mean - global_mean) / global_sd
    """
    _sectors  = sorted(panel.dropna(subset=["sector"]).sector.unique())
    n_topics  = len(topics)
    angles    = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
    angles_c  = np.concatenate([angles, [angles[0]]])

    LAYERS = [
        ("match",    "Relatedness",  "#2980b9", 0.20,
         [f"match_{t}" for t in topics]),
        ("pos_mean", "Positive Mean","#27ae60", 0.25,
         [f"tend_{t}_pos_mean" for t in topics]),
        ("neg_mean", "Neg Mean",     "#e74c3c", 0.25,
         [f"tend_{t}_neg_mean" for t in topics]),
    ]

    gstats = {}
    for key, lbl, col, alpha, cols in LAYERS:
        gstats[key] = {}
        for c in cols:
            vals = panel[c].dropna().values
            gstats[key][c] = (float(vals.mean()), float(vals.std()))

    records = {}
    for sec in _sectors:
        df_s = panel[panel.sector == sec]
        row  = {}
        for key, lbl, col, alpha, cols in LAYERS:
            row[key] = [(float(df_s[c].mean()) - gstats[key][c][0])
                        / gstats[key][c][1] for c in cols]
        records[sec] = row

    all_z  = [v for rec in records.values() for vals in rec.values() for v in vals]
    offset = max(0.0, -min(all_z)) + 0.4
    r_ann_base = offset + 1.1
    DR = 0.33

    ncols = 3
    nrows = int(np.ceil(len(_sectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True),
                             figsize=(15, nrows * 5.0))
    axf = axes.flatten()

    short_labels = []
    for t in topics:
        words = topic_labels[t].split()
        short_labels.append("\n".join(words) if len(words) > 1 else topic_labels[t])

    for idx, sec in enumerate(_sectors):
        ax  = axf[idx]
        rec = records[sec]
        theta_ref = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ref, [offset] * 300,
                color="#999", lw=0.8, ls="--", alpha=0.6, zorder=1)

        for key, lbl, col, alpha, _ in LAYERS:
            z_vals = rec[key]
            r_vals = [z + offset for z in z_vals]
            r_c    = r_vals + [r_vals[0]]
            ax.plot(angles_c, r_c, color=col, lw=1.5, zorder=3)
            ax.fill(angles_c, r_c, color=col, alpha=alpha, zorder=2)

        for i, angle in enumerate(angles):
            for j, (key, lbl, col, alpha, _) in enumerate(LAYERS):
                z = rec[key][i]
                r = r_ann_base + j * DR
                ax.text(angle, r, f"{z:+.2f}",
                        ha="center", va="center",
                        fontsize=5.0, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.85),
                        zorder=6)

        ax.set_xticks(angles)
        ax.set_xticklabels(short_labels, size=7)
        z_ticks = [-1, 0, 1, 2]
        r_ticks = [z + offset for z in z_ticks if z + offset >= 0]
        z_shown = [z for z in z_ticks if z + offset >= 0]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{z:+d}\u03c3" for z in z_shown], size=6, color="#555")
        ax.set_ylim(0, None)
        ax.set_title(sec, size=9, fontweight="bold", pad=14)
        ax.spines["polar"].set_visible(False)
        ax.grid(color="grey", alpha=0.25, lw=0.5)

    hide_unused(axf, len(_sectors))
    legend_handles = [
        plt.Line2D([0], [0], color=col, lw=2, label=lbl)
        for _, lbl, col, _, _ in LAYERS
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02))
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

_radar_chart(panel, ENV_TOPICS, ENV_TOPIC_LABELS, TOPIC_COLORS,
             FIGS / "fig04_sector_radar.png")


# ── fig05: Coefficient comparison M1–M3 — 3×3, 7 panels ─────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharey=False)
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    plot_rows = []
    for m in topic_results[topic]:
        for col, info in m["coefs"].items():
            if info["coef"] is not None:
                plot_rows.append({
                    "label": f"{m['label']}: {info['label']}",
                    "coef":  info["coef"],
                    "se":    info["se"],
                    "dim":   info["dim"],
                })
    y_pos = range(len(plot_rows))
    for i, row in enumerate(plot_rows):
        color = DIM_COLORS[row["dim"]]
        ax.errorbar(row["coef"], i, xerr=1.96 * row["se"],
                    fmt="none", ecolor=color + "88", elinewidth=1.2, capsize=3)
        ax.plot(row["coef"], i, "o", color=color, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", linewidth=0.9, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([r["label"] for r in plot_rows], fontsize=8)
    ax.set_xlabel("Coefficient  [95% CI]", fontsize=8)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])

legend_handles = [
    mpatches.Patch(color=DIM_COLORS["coverage"],  label="Relatedness"),
    mpatches.Patch(color=DIM_COLORS["sentiment"], label="Overall Sentiment"),
    mpatches.Patch(color=DIM_COLORS["pos"],       label="Positive Mean Score"),
    mpatches.Patch(color=DIM_COLORS["neg"],       label="Negative Mean Score"),
]
axf[N_TOPICS - 1].legend(handles=legend_handles, fontsize=7.5,
                          loc="lower right", title="Disclosure dimension",
                          title_fontsize=7.5)
hide_unused(axf, N_TOPICS)
plt.tight_layout()
fig.savefig(FIGS / "fig05_coef_comparison_M1_M3.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig06: Sector M1 forest — 3×3, 7 panels ──────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax    = axf[idx]
    df_s  = sector_results["M1"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    y     = range(len(df_s))
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_s["coef_match"].values, list(y),
                xerr=1.96 * df_s["se_match"].values,
                fmt="o", color=color, ms=5,
                ecolor="#aaa", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#e74c3c", linewidth=0.9, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
    ax.set_xlabel("M1 Relatedness  [95% CI]", fontsize=8)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
hide_unused(axf, N_TOPICS)
plt.tight_layout()
fig.savefig(FIGS / "fig06_sector_M1.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig07: Sector M2 forest — 3×3, 7 panels ──────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax    = axf[idx]
    df_s  = sector_results["M2"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    y     = range(len(df_s))
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_s["coef_match"].values, list(y),
                xerr=1.96 * df_s["se_match"].values,
                fmt="o", color=color, ms=5,
                ecolor="#aaa", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#e74c3c", linewidth=0.9, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
    ax.set_xlabel("M2 Relatedness (controlling Overall Sentiment)  [95% CI]", fontsize=8)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
hide_unused(axf, N_TOPICS)
plt.tight_layout()
fig.savefig(FIGS / "fig07_sector_M2.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig08a / fig08b: Sector M3 forest — split by direction ───────────────────
# fig08a: Positive Mean Score (3×3, 7 panels)
# fig08b: Negative Mean Score (3×3, 7 panels)
def _sector_m3_forest(var_key, var_label, dot_color, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axf = axes.flatten()
    for idx, topic in enumerate(ENV_TOPICS):
        ax   = axf[idx]
        df_s = (sector_results["M3"][topic]
                .dropna(subset=[f"coef_{var_key}"])
                .sort_values(f"coef_{var_key}"))
        y = range(len(df_s))
        ax.errorbar(df_s[f"coef_{var_key}"].values, list(y),
                    xerr=1.96 * df_s[f"se_{var_key}"].values,
                    fmt="o", color=dot_color, ms=5,
                    ecolor="#aaa", elinewidth=1.2, capsize=3)
        ax.axvline(0, color="#e74c3c", linewidth=0.9, linestyle="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
        ax.set_xlabel(f"M3 {var_label}  [95% CI]", fontsize=8)
        label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
    hide_unused(axf, N_TOPICS)
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

_sector_m3_forest("pos_mean", "Positive Mean Score", DIM_COLORS["pos"],
                  FIGS / "fig08_sector_M3_pos.png")
_sector_m3_forest("neg_mean", "Negative Mean Score", DIM_COLORS["neg"],
                  FIGS / "fig09_sector_M3_neg.png")


# ── fig10: Robustness M1 — 3×3, 7 panels ─────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax   = axf[idx]
    df_r = robust_results[topic].dropna(subset=["coef"])
    y    = range(len(df_r))
    for i, (_, row) in enumerate(df_r.iterrows()):
        c = TOPIC_COLORS[topic] if row["Check"] == "Main M1" else "#555"
        ax.errorbar(row["coef"], i,
                    xerr=1.96 * row["se"] if row["se"] is not None else 0,
                    fmt="none", ecolor="#aaa", elinewidth=1.2, capsize=3)
        ax.plot(row["coef"], i, "o", color=c, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", linewidth=0.9, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_r["Check"].values, fontsize=8)
    ax.set_xlabel("M1 Relatedness  [95% CI]", fontsize=8)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
hide_unused(axf, N_TOPICS)
plt.tight_layout()
fig.savefig(FIGS / "fig10_robustness_M1.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("Saving tables...")

# ── Table 1: Descriptive statistics ──────────────────────────────────────────
stat_cols = ["log_mc", "dlog_mc", "sentiment_mean"]
for topic in ENV_TOPICS:
    stat_cols += [f"match_{topic}"]
    for suf in ("pos_mean", "neg_mean"):
        stat_cols.append(f"tend_{topic}_{suf}")

_stat_labels = {
    "log_mc":         "Log Market Cap",
    "dlog_mc":        "\u0394Log Market Cap",
    "sentiment_mean": "Overall Sentiment",
}
for _t in ENV_TOPICS:
    _lbl = ENV_TOPIC_LABELS[_t]
    _stat_labels[f"match_{_t}"]          = f"Relatedness ({_lbl})"
    _stat_labels[f"tend_{_t}_pos_mean"]  = f"Positive Mean Score ({_lbl})"
    _stat_labels[f"tend_{_t}_neg_mean"]  = f"Negative Mean Score ({_lbl})"

stat_rows = []
for col in stat_cols:
    if col not in panel.columns:
        continue
    s = panel[col].dropna()
    stat_rows.append({
        "Variable": _stat_labels.get(col, col),
        "N": len(s), "Mean": round(s.mean(), 4), "SD": round(s.std(), 4),
        "Min": round(s.min(), 4), "p25": round(s.quantile(0.25), 4),
        "Median": round(s.median(), 4), "p75": round(s.quantile(0.75), 4),
        "Max": round(s.max(), 4),
    })
pd.DataFrame(stat_rows).to_excel(TABLES / "Table1_DescriptiveStats.xlsx", index=False)


# ── S1–S7: per-topic full coefficient tables ──────────────────────────────────
def build_topic_coef_table(results):
    rows = []
    for m in results:
        for col, info in m["coefs"].items():
            c, se, stars = info["coef"], info["se"], info["stars"]
            rows.append({
                "Spec": m["label"], "Model": m["name"],
                "Variable": info["label"], "Dimension": info["dim"],
                "\u03b2": round(c, 4) if c is not None else None,
                "stars": stars,
                "SE": round(se, 4) if se is not None else None,
                "N obs": m["n"], "N companies": m["nc"],
                "Within R\u00b2": round(m["result"].rsquared, 4),
                "Company FE": "Yes", "Year FE": "Yes", "Outcome": "log_mc",
            })
    return pd.DataFrame(rows)

for i, topic in enumerate(ENV_TOPICS, start=1):
    tbl       = build_topic_coef_table(topic_results[topic])
    label_str = ENV_TOPIC_LABELS[topic].replace(" ", "").replace("&", "")
    tbl.to_excel(TABLES / f"S{i}_Coef_{label_str}.xlsx", index=False)


# ── Table 2: Cross-topic M1–M3 summary ───────────────────────────────────────
summary_rows = []
for topic in ENV_TOPICS:
    for m in topic_results[topic]:
        for col, info in m["coefs"].items():
            c, se, stars = info["coef"], info["se"], info["stars"]
            summary_rows.append({
                "Topic": ENV_TOPIC_LABELS[topic],
                "Spec": m["label"], "Variable": info["label"],
                "\u03b2": round(c, 4) if c is not None else None,
                "stars": stars,
                "SE": round(se, 4) if se is not None else None,
                "N obs": m["n"], "N companies": m["nc"],
                "Within R\u00b2": round(m["result"].rsquared, 4),
            })
pd.DataFrame(summary_rows).to_excel(
    TABLES / "Table2_CrossTopic_Summary.xlsx", index=False)


# ── Tables 3–5: Sector heterogeneity M1, M2, M3 ──────────────────────────────
_KEY_DISPLAY = {
    "match":     "Relatedness",
    "sentiment": "Overall Sentiment",
    "pos_mean":  "Positive Mean Score",
    "neg_mean":  "Negative Mean Score",
}

def build_sector_table(spec_label):
    rows = []
    for topic in ENV_TOPICS:
        df_s = sector_results[spec_label][topic].copy()
        df_s.insert(0, "Topic", ENV_TOPIC_LABELS[topic])
        rows.append(df_s)
    combined = pd.concat(rows, ignore_index=True)
    drop_cols = [c for c in combined.columns
                 if c.startswith("coef_") or c.startswith("se_")]
    combined  = combined.drop(columns=drop_cols, errors="ignore")
    rename_map = {}
    for key, display in _KEY_DISPLAY.items():
        b_col  = f"\u03b2_{key}"
        se_col = f"SE_{key}"
        if b_col  in combined.columns: rename_map[b_col]  = f"\u03b2 {display}"
        if se_col in combined.columns: rename_map[se_col] = f"SE ({display})"
    return combined.rename(columns=rename_map)

build_sector_table("M1").to_excel(TABLES / "S8_SectorHet_M1.xlsx",    index=False)
build_sector_table("M2").to_excel(TABLES / "Table3_SectorHet_M2.xlsx", index=False)
build_sector_table("M3").to_excel(TABLES / "Table4_SectorHet_M3.xlsx", index=False)


# ── Table 5: Robustness M1 ────────────────────────────────────────────────────
robust_tabs = []
for topic in ENV_TOPICS:
    df_r = robust_results[topic].copy()
    df_r.insert(0, "Topic", ENV_TOPIC_LABELS[topic])
    robust_tabs.append(df_r)
pd.concat(robust_tabs, ignore_index=True).to_excel(
    TABLES / "Table5_Robustness_M1.xlsx", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# METADATA & CODE
# ══════════════════════════════════════════════════════════════════════════════
import shutil
shutil.copy(__file__, CODE / "run_analysis.py")

# ── variable_dictionary.yaml ──────────────────────────────────────────────────
var_dict = {
    "outcome": {
        "log_mc":  "Log market capitalization",
        "dlog_mc": "Annual log market cap growth (\u0394logMC)",
    },
    "gov_topic_variables": {
        topic: {
            "M1_match":        f"match_{topic}_lag1 \u2014 Coverage volume",
            "M2_match_sent":   (f"match_{topic}_lag1 + sentiment_mean_lag1"
                                " \u2014 Coverage + overall tone"),
            "M3_pos_neg_mean": (f"tend_{topic}_pos_mean_lag1 + tend_{topic}_neg_mean_lag1"
                                " \u2014 Directional intensity"),
        }
        for topic in ENV_TOPICS
    },
    "sector_analysis": {
        "M1": "match coef per sector per topic",
        "M2": "match + sentiment coefs per sector per topic",
        "M3": "pos_mean + neg_mean coefs per sector per topic",
    },
}
with open(META / "variable_dictionary.yaml", "w") as f:
    yaml.dump(var_dict, f, allow_unicode=True, sort_keys=False)

# ── dataset_dictionary.yaml ───────────────────────────────────────────────────
def _n_obs(col):
    return int(panel[col].notna().sum()) if col in panel.columns else 0

dataset_dict = {
    "panel": {
        "path":      "data/processed/panel.parquet",
        "rows":      int(panel.shape[0]),
        "columns":   int(panel.shape[1]),
        "companies": int(panel.stock_code.nunique()),
        "years":     [int(y) for y in years],
    }
}
with open(META / "dataset_dictionary.yaml", "w") as f:
    yaml.dump(dataset_dict, f, allow_unicode=True, sort_keys=False)

# ── actionbrief.yaml ──────────────────────────────────────────────────────────
actionbrief = {
    "doc_type": "actionbrief",
    "version":  "0.1.0",

    "datasetDictionary": {
        "match_scores": {
            "datasetSourceName":        "ESG Relevance Scores (match_scores)",
            "datasetDescription":       "BERT-based semantic matching scores quantifying how closely each company's securities report covers 23 pre-defined ESG themes; one row per company-year.",
            "datasetAttribute":         "table",
            "datasetProvider":          "Internal NLP pipeline",
            "datasetGenerationInfo":    "Sentence embeddings of securities reports matched against ESG theme anchors via cosine similarity; aggregated to company-year level.",
            "datasetPeriod":            "2016 to 2025",
            "datasetScale":             "Japan listed companies (Tokyo Stock Exchange)",
            "datasetSpatialResolution": "Company level",
            "datasetTemporalResolution":"Annual",
            "datasetPreprocessingInfo": "Renamed Japanese theme columns to English snake_case; composite and pillar-level (E/S/G/B) indices computed as theme means; one-year lags generated.",
        },
        "tendency_scores": {
            "datasetSourceName":        "ESG Tendency Scores (tendency_scores)",
            "datasetDescription":       "NLP-derived tone sub-scores for each ESG theme per company-year: related_ratio, pos_ratio, neg_ratio, related_mean, pos_mean, neg_mean.",
            "datasetAttribute":         "table",
            "datasetProvider":          "Internal NLP pipeline",
            "datasetGenerationInfo":    "Sentence-level sentiment classification applied to matched ESG sentences; aggregated ratios and means computed per theme per company-year.",
            "datasetPeriod":            "2016 to 2025",
            "datasetScale":             "Japan listed companies",
            "datasetSpatialResolution": "Company level",
            "datasetTemporalResolution":"Annual",
            "datasetPreprocessingInfo": "Renamed theme columns; pos_mean and neg_mean sub-scores for 7 focal Governance topics extracted and lagged by one year.",
        },
        "sentiment_scores": {
            "datasetSourceName":        "Overall ESG Sentiment (sentiment_scores)",
            "datasetDescription":       "Document-level overall ESG sentiment score (mean) per company-year derived from SAPT.",
            "datasetAttribute":         "table",
            "datasetProvider":          "Internal NLP pipeline",
            "datasetGenerationInfo":    "SAPT model applied to full securities reports; sentiment_mean is the average sentence-level score.",
            "datasetPeriod":            "2016 to 2025",
            "datasetScale":             "Japan listed companies",
            "datasetSpatialResolution": "Company level",
            "datasetTemporalResolution":"Annual",
            "datasetPreprocessingInfo": "Merged on stock_code \u00d7 year; one-year lag generated.",
        },
        "market_cap": {
            "datasetSourceName":        "Annual Market Capitalization (Market_cap_annual)",
            "datasetDescription":       "Year-end market capitalization (JPY) for TSE-listed companies, 2015\u20132025.",
            "datasetAttribute":         "table",
            "datasetProvider":          "MSCI / Bloomberg (via research team)",
            "datasetGenerationInfo":    "Wide-format Excel reshaped to long (stock_code \u00d7 year); non-numeric entries coerced to NaN.",
            "datasetPeriod":            "2015 to 2025",
            "datasetScale":             "Japan listed companies",
            "datasetSpatialResolution": "Company level",
            "datasetTemporalResolution":"Annual",
            "datasetPreprocessingInfo": "log_mc = log(market_cap); dlog_mc computed as year-on-year difference.",
        },
        "gics_classification": {
            "datasetSourceName":        "GICS Industry Classification (MSCI_category)",
            "datasetDescription":       "GICS Sector / Industry Group / Industry / Sub-Industry classification for TSE-listed companies.",
            "datasetAttribute":         "table",
            "datasetProvider":          "MSCI",
            "datasetGenerationInfo":    "Static cross-sectional Excel; merged on stock_code.",
            "datasetPeriod":            "Cross-sectional (single snapshot)",
            "datasetScale":             "Japan listed companies",
            "datasetSpatialResolution": "Company level",
            "datasetTemporalResolution":"Cross-sectional",
            "datasetPreprocessingInfo": "Columns renamed; sector used for heterogeneity analysis.",
        },
        "interest_rate": {
            "datasetSourceName":        "Japan Call Rate (Japan_interest_rate_annual)",
            "datasetDescription":       "Annual overnight call rate (%) for Japan, 2015\u20132025.",
            "datasetAttribute":         "table",
            "datasetProvider":          "Bank of Japan",
            "datasetGenerationInfo":    "Annual averages from Bank of Japan statistics; merged on year.",
            "datasetPeriod":            "2015 to 2025",
            "datasetScale":             "National (Japan)",
            "datasetSpatialResolution": "National",
            "datasetTemporalResolution":"Annual",
            "datasetPreprocessingInfo": "Used as control in robustness check (rate-control specification without year FE).",
        },
    },

    "variableDictionary": {
        "log_mc": {
            "variableNameInDataset": "log_mc",
            "variableNameInArticle": "Log Market Cap",
            "variableAttribute":     "continuous",
            "variableUnit":          "log JPY",
            "variableDescription":   "Natural logarithm of year-end market capitalization. Primary outcome variable.",
            "variableSource":        "market_cap",
            "observationNumber":     _n_obs("log_mc"),
        },
        "dlog_mc": {
            "variableNameInDataset": "dlog_mc",
            "variableNameInArticle": "\u0394Log Market Cap",
            "variableAttribute":     "continuous",
            "variableUnit":          "log points (YoY)",
            "variableDescription":   "Year-on-year change in log market cap. Used as outcome in robustness checks.",
            "variableSource":        "market_cap",
            "observationNumber":     _n_obs("dlog_mc"),
        },
        "sentiment_mean": {
            "variableNameInDataset": "sentiment_mean",
            "variableNameInArticle": "Overall Sentiment",
            "variableAttribute":     "continuous",
            "variableUnit":          "index (SAPT score)",
            "variableDescription":   "Document-level mean ESG sentiment score from SAPT.",
            "variableSource":        "sentiment_scores",
            "observationNumber":     _n_obs("sentiment_mean"),
        },
        **{
            f"match_{t}": {
                "variableNameInDataset": f"match_{t}",
                "variableNameInArticle": f"Relatedness ({ENV_TOPIC_LABELS[t]})",
                "variableAttribute":     "continuous",
                "variableUnit":          "cosine similarity (0\u20131)",
                "variableDescription":   f"BERT-based semantic relatedness score of the securities report to the {ENV_TOPIC_LABELS[t]} governance theme.",
                "variableSource":        "match_scores",
                "observationNumber":     _n_obs(f"match_{t}"),
            } for t in ENV_TOPICS
        },
        **{
            f"tend_{t}_pos_mean": {
                "variableNameInDataset": f"tend_{t}_pos_mean",
                "variableNameInArticle": f"Positive Mean Score ({ENV_TOPIC_LABELS[t]})",
                "variableAttribute":     "continuous",
                "variableUnit":          "index (SAPT score)",
                "variableDescription":   f"Mean positive tone score of sentences matched to the {ENV_TOPIC_LABELS[t]} governance theme.",
                "variableSource":        "tendency_scores",
                "observationNumber":     _n_obs(f"tend_{t}_pos_mean"),
            } for t in ENV_TOPICS
        },
        **{
            f"tend_{t}_neg_mean": {
                "variableNameInDataset": f"tend_{t}_neg_mean",
                "variableNameInArticle": f"Negative Mean Score ({ENV_TOPIC_LABELS[t]})",
                "variableAttribute":     "continuous",
                "variableUnit":          "index (SAPT score)",
                "variableDescription":   f"Mean negative tone score of sentences matched to the {ENV_TOPIC_LABELS[t]} governance theme.",
                "variableSource":        "tendency_scores",
                "observationNumber":     _n_obs(f"tend_{t}_neg_mean"),
            } for t in ENV_TOPICS
        },
    },

    "tableDictionary": {
        "Table1": {
            "tableId":              "Table 1",
            "tablePath":            "tables/Table1_DescriptiveStats.xlsx",
            "tableCaption":         "Descriptive Statistics",
            "tableAim":             "Summarize the distribution of all key outcome, disclosure, and control variables across the full panel.",
            "tableExplanation":     "Each row is one variable. Columns: N, Mean, SD, Min, p25, Median, p75, Max.",
            "tableSuitableSection": "Data and Measurement",
        },
        "Table2": {
            "tableId":              "Table 2",
            "tablePath":            "tables/Table2_CrossTopic_Summary.xlsx",
            "tableCaption":         "Cross-Topic Regression Results: M1\u2013M3 Specifications (All Governance Topics)",
            "tableAim":             "Present main TWFE regression coefficients for all three disclosure dimensions across all 7 governance topics.",
            "tableExplanation":     "Rows: Topic \u00d7 Spec \u00d7 Variable. Columns: \u03b2, stars, SE, N obs, N companies, Within R\u00b2. Company and Year FE throughout.",
            "tableSuitableSection": "Results",
        },
        "Table3": {
            "tableId":              "Table 3",
            "tablePath":            "tables/Table3_SectorHet_M2.xlsx",
            "tableCaption":         "Sector Heterogeneity: M2 (Relatedness + Overall Sentiment)",
            "tableAim":             "Examine how relatedness and overall sentiment effects vary across GICS sectors.",
            "tableExplanation":     "Rows: Topic \u00d7 Sector. \u03b2 Relatedness and \u03b2 Overall Sentiment with stars (*** p<0.01, ** p<0.05, * p<0.10), SE in parentheses.",
            "tableSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "Table4": {
            "tableId":              "Table 4",
            "tablePath":            "tables/Table4_SectorHet_M3.xlsx",
            "tableCaption":         "Sector Heterogeneity: M3 (Positive Mean Score + Negative Mean Score)",
            "tableAim":             "Examine how positive and negative tone effects vary across GICS sectors.",
            "tableExplanation":     "Rows: Topic \u00d7 Sector. \u03b2 Positive Mean Score and \u03b2 Negative Mean Score with stars and SE in parentheses.",
            "tableSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "Table5": {
            "tableId":              "Table 5",
            "tablePath":            "tables/Table5_Robustness_M1.xlsx",
            "tableCaption":         "Robustness Checks: M1 (Relatedness)",
            "tableAim":             "Verify the main M1 result is not driven by specification, outcome, or sample composition.",
            "tableExplanation":     "Six robustness variants per topic: main, \u0394logMC, rate control, excl. 2020, balanced panel, winsorized \u0394logMC.",
            "tableSuitableSection": "Robustness",
        },
        **{
            f"S{i}": {
                "tableId":              f"Table S{i}",
                "tablePath":            f"tables/S{i}_Coef_{ENV_TOPIC_LABELS[t].replace(' ', '').replace('&', '')}.xlsx",
                "tableCaption":         f"Full Coefficient Table: {ENV_TOPIC_LABELS[t]} (M1\u2013M3)",
                "tableAim":             f"Report complete regression output for all three specifications for the {ENV_TOPIC_LABELS[t]} topic.",
                "tableExplanation":     "Rows: Spec \u00d7 Variable. Columns: \u03b2, stars, SE, N obs, N companies, Within R\u00b2, FE indicators.",
                "tableSuitableSection": "Supplementary Materials",
            } for i, t in enumerate(ENV_TOPICS, start=1)
        },
        "S8": {
            "tableId":              "Table S8",
            "tablePath":            "tables/S8_SectorHet_M1.xlsx",
            "tableCaption":         "Sector Heterogeneity: M1 (Relatedness only)",
            "tableAim":             "Baseline sector heterogeneity using only the Relatedness variable.",
            "tableExplanation":     "Rows: Topic \u00d7 Sector. \u03b2 Relatedness with stars, SE in parentheses. Within R\u00b2 reported.",
            "tableSuitableSection": "Supplementary Materials",
        },
    },

    "figureDictionary": {
        "fig01": {
            "figureId":              "Figure 1",
            "figurePath":            "figures/fig01_gov_disclosure_trends.png",
            "figureCaption":         "Annual Trends in Governance Disclosure: Relatedness and Overall Sentiment (2016\u20132025)",
            "figureAim":             "Show the temporal evolution of governance topic coverage and overall ESG tone across the panel period.",
            "figureExplanation":     "2\u00d74 panel. (a)\u2013(g): annual mean Relatedness score for each of the 7 governance topics. (h): annual mean Overall Sentiment. X-axis: 2016\u20132025.",
            "figureSuitableSection": "Data and Measurement",
        },
        "fig02": {
            "figureId":              "Figure 2",
            "figurePath":            "figures/fig02_pos_neg_ratio_trends.png",
            "figureCaption":         "Trends in Positive and Negative Disclosure Ratios by Governance Topic (2016\u20132025)",
            "figureAim":             "Illustrate the breadth of positive vs. negative tone sentences for each governance topic over time.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Each panel shows mean pos_ratio (green) and neg_ratio (red) over years.",
            "figureSuitableSection": "Data and Measurement",
        },
        "fig03": {
            "figureId":              "Figure 3",
            "figurePath":            "figures/fig03_market_cap_distributions.png",
            "figureCaption":         "Distribution of Market Capitalization by Year",
            "figureAim":             "Document the distribution and temporal variation of the primary outcome variable.",
            "figureExplanation":     "1\u00d72 panel. (a) Boxplots of Log Market Cap by year. (b) Boxplots of \u0394Log Market Cap by year.",
            "figureSuitableSection": "Data and Measurement",
        },
        "fig04": {
            "figureId":              "Figure 4",
            "figurePath":            "figures/fig04_sector_radar.png",
            "figureCaption":         "Governance Disclosure Profile by GICS Sector",
            "figureAim":             "Provide a descriptive overview of sector-level differences in governance disclosure across all three metrics before regression analysis.",
            "figureExplanation":     "4\u00d73 polar radar chart (11 GICS sectors). Spokes = 7 governance topics. Three overlaid filled areas: blue = Relatedness, green = Positive Mean Score, red = Negative Mean Score. Each metric z-scored as (sector mean \u2212 global mean) / global \u03c3 per topic column. Dashed circle = z = 0 (global average). Values annotated at 1\u03c3\u20132\u03c3 band.",
            "figureSuitableSection": "Data and Measurement",
        },
        "fig05": {
            "figureId":              "Figure 5",
            "figurePath":            "figures/fig05_coef_comparison_M1_M3.png",
            "figureCaption":         "Regression Coefficient Comparison Across M1\u2013M3 Specifications by Governance Topic",
            "figureAim":             "Visually compare effect sizes and uncertainty across disclosure dimensions for each governance topic.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Each panel: horizontal forest plot with y = Spec:Variable, x = coefficient with 95% CI. Colors: blue=Relatedness, purple=Overall Sentiment, green=Positive Mean Score, red=Negative Mean Score.",
            "figureSuitableSection": "Results",
        },
        "fig06": {
            "figureId":              "Figure 6",
            "figurePath":            "figures/fig06_sector_M1.png",
            "figureCaption":         "Sector Heterogeneity in M1 (Relatedness) Effects on Market Cap",
            "figureAim":             "Reveal which GICS sectors show stronger market cap responses to governance disclosure coverage.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Forest plot per topic: sectors sorted by coefficient; x = M1 Relatedness coefficient with 95% CI.",
            "figureSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "fig07": {
            "figureId":              "Figure 7",
            "figurePath":            "figures/fig07_sector_M2.png",
            "figureCaption":         "Sector Heterogeneity in M2 Relatedness Effects (Controlling for Overall Sentiment)",
            "figureAim":             "Show how the Relatedness effect varies by sector after partialling out overall ESG sentiment.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Same layout as Figure 6; coefficient shown is M2 partial Relatedness effect.",
            "figureSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "fig08": {
            "figureId":              "Figure 8",
            "figurePath":            "figures/fig08_sector_M3_pos.png",
            "figureCaption":         "Sector Heterogeneity in M3 Positive Mean Score Effects on Market Cap",
            "figureAim":             "Examine sector-level differences in how positive governance disclosure tone is associated with market cap.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Each panel: forest plot of M3 Positive Mean Score coefficients by GICS sector, sorted by coefficient magnitude. 95% CI shown. Dashed red line at zero.",
            "figureSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "fig09": {
            "figureId":              "Figure 9",
            "figurePath":            "figures/fig09_sector_M3_neg.png",
            "figureCaption":         "Sector Heterogeneity in M3 Negative Mean Score Effects on Market Cap",
            "figureAim":             "Examine sector-level differences in how negative (risk-acknowledging) governance disclosure tone is associated with market cap.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Same layout as Figure 8. Coefficient shown is M3 Negative Mean Score. Positive coefficient indicates a transparency premium; negative coefficient indicates a risk penalty.",
            "figureSuitableSection": "Results \u2014 Heterogeneity Analysis",
        },
        "fig10": {
            "figureId":              "Figure 10",
            "figurePath":            "figures/fig10_robustness_M1.png",
            "figureCaption":         "Robustness of M1 (Relatedness) Effect Across Alternative Specifications",
            "figureAim":             "Confirm that the main Relatedness finding is robust to alternative outcomes, sample restrictions, and control choices.",
            "figureExplanation":     "3\u00d73 panel (7 topics). Forest plot per topic: y = robustness check label; x = coefficient with 95% CI. Main M1 in topic color; alternatives in gray.",
            "figureSuitableSection": "Robustness",
        },
    },

    "analysisStructureBrief": {
        "overview": (
            "This study examines whether and how Governance disclosure dimensions "
            "(coverage volume, overall sentiment, and directional tone intensity) of Japanese "
            "listed companies\u2019 securities reports (\u6709\u4fa1\u8a3c\u5238\u5831\u544a\u66f8) "
            "are associated with market capitalization, using a two-way fixed-effects (TWFE) "
            "panel regression over 2016\u20132025. Seven focal Governance topics are analysed: "
            "Corporate Governance, Security, Risk & Compliance, Management Operations, "
            "Materiality, Stakeholder Engagement, and Corporate Philosophy. "
            "Sector heterogeneity and robustness are examined systematically."
        ),
        "levels": [
            {
                "levelName":    "Data Collection and Preprocessing",
                "levelAim":     "Assemble and harmonize all raw inputs into a unified company-year panel.",
                "inputs":       ["match_scores", "tendency_scores", "sentiment_scores",
                                 "market_cap", "gics_classification", "interest_rate"],
                "modelOrMethod":"Pandas merge; log transformation; within-panel differencing for \u0394logMC.",
                "outputs":      ["data/processed/panel.parquet"],
                "interpretation":"Panel defined by ESG score coverage (2016\u20132025, 37,272 obs).",
            },
            {
                "levelName":    "Main TWFE Regression (M1\u2013M3)",
                "levelAim":     "Estimate the association between t\u22121 disclosure dimensions and log market cap using within-company variation.",
                "inputs":       ["panel.parquet \u2014 log_mc, match_{topic}_lag1, sentiment_mean_lag1, tend_{topic}_pos/neg_mean_lag1"],
                "modelOrMethod":"Two-Way Fixed Effects OLS: within-company demeaning + C(year); SE clustered by company. Three specs: M1 (Relatedness), M2 (Relatedness + Sentiment), M3 (Positive + Negative Mean Score).",
                "outputs":      ["Table 2", "Tables S1\u2013S7", "Figure 5"],
                "interpretation":"Positive \u03b2 on Relatedness: companies increasing governance disclosure coverage see higher market cap the following year, within-company.",
            },
            {
                "levelName":    "Sector Heterogeneity Analysis",
                "levelAim":     "Test whether the disclosure\u2013market cap relationship differs across GICS sectors.",
                "inputs":       ["panel.parquet \u2014 sector, log_mc, disclosure variables"],
                "modelOrMethod":"Same TWFE specification run separately per GICS sector \u00d7 Governance topic for each of M1, M2, M3.",
                "outputs":      ["Table 3 (M2)", "Table 4 (M3)", "Table S8 (M1)", "Figures 6\u20139"],
                "interpretation":"Sector coefficients with 95% CIs show which industries price governance disclosure more strongly. Sectors with <50 companies interpreted with caution.",
            },
            {
                "levelName":    "Robustness Checks",
                "levelAim":     "Assess sensitivity of the main M1 result to specification, outcome, and sample choices.",
                "inputs":       ["panel.parquet"],
                "modelOrMethod":"Six M1 variants: main, \u0394logMC, rate control (no year FE), excl. 2020, balanced panel, winsorized \u0394logMC.",
                "outputs":      ["Table 5", "Figure 9"],
                "interpretation":"Consistency of sign/significance across checks supports robustness of the Relatedness finding.",
            },
        ],
    },
}

with open(EXPORT / "actionbrief.yaml", "w") as f:
    yaml.dump(actionbrief, f, allow_unicode=True, sort_keys=False)

# ── Copy AnaSOP ───────────────────────────────────────────────────────────────
_sop_src = ROOT / "docs" / "AnaSOP.md"
if _sop_src.exists():
    shutil.copy2(_sop_src, EXPORT / "AnaSOP.md")
else:
    print(f"WARNING: AnaSOP.md not found at {_sop_src}")

# ── Summary ───────────────────────────────────────────────────────────────────
fig_list   = sorted(FIGS.glob("*.png"))
table_list = sorted(TABLES.glob("*.xlsx"))
print(f"\n{'='*55}")
print("Export complete.")
print(f"  Figures : {len(fig_list)} PNG files")
print(f"  Tables  : {len(table_list)} xlsx files")
print(f"\nExport structure:")
for f in sorted(EXPORT.rglob("*")):
    if f.is_file() and f.suffix in (".png", ".xlsx", ".yaml", ".py", ".md"):
        print(f"  {f.relative_to(ROOT)}")
