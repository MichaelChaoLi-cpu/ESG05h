"""
src/90_export.py
Generates the complete export/ folder for Jiazi ingestion.

Implements AnaSOP §8–9 for the Business-disclosure mechanism study:
  11 figures + 13 tables + metadata + actionbrief.yaml + nbs/ + AnaSOP.md

Run (from project root):
    python src/90_export.py
"""

import shutil
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
NBS    = EXPORT / "nbs"
for d in (FIGS, TABLES, CODE, META, NBS):
    d.mkdir(parents=True, exist_ok=True)

FIG_DPI = 300
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# ── Clean up stale artifacts ──────────────────────────────────────────────────
_stale = [
    "fig01_gov_disclosure_trends.png", "fig02_pos_neg_ratio_trends.png",
    "fig03_market_cap_distributions.png", "fig04_sector_radar.png",
    "fig05_coef_comparison_M1_M3.png",
    "fig06_sector_M1.png", "fig07_sector_M2.png",
    "fig08_sector_M3_pos.png", "fig09_sector_M3_neg.png",
    "fig10_robustness_M1.png",
]
for _f in _stale:
    _p = FIGS / _f
    if _p.exists():
        _p.unlink()
        print(f"  Removed stale: {_f}")

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
    "competitiveness":        "#2980b9",
    "stakeholder_cocreation": "#c0392b",
    "business_portfolio":     "#e67e22",
    "intellectual_capital":   "#16a085",
    "digital_transformation": "#8e44ad",
    "financial_metrics":      "#27ae60",
    "corporate_value":        "#7f8c8d",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
panel   = pd.read_parquet(PROC / "panel.parquet")
years   = sorted(panel.year.unique())
sectors = sorted(panel.dropna(subset=["sector"]).sector.unique())
print(f"  Panel: {panel.shape[0]:,} obs | {panel.stock_code.nunique():,} companies")

# Compute t-2 lags (for identification robustness)
panel = panel.sort_values(["stock_code", "year"])
for topic in ENV_TOPICS:
    lag1 = f"match_{topic}_lag1"
    if lag1 in panel.columns:
        panel[f"match_{topic}_lag2"] = panel.groupby("stock_code")[lag1].shift(1)

# ── Regression helpers ────────────────────────────────────────────────────────
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
    return smf.ols(formula, data=df_sub).fit(
        cov_type="cluster", cov_kwds={"groups": df_sub["stock_code"]}
    ), len(df_sub), df_sub["stock_code"].nunique()


def get_coef(result, var, entity_fe=True):
    pname = f"w_{var}" if entity_fe else var
    if pname not in result.params.index:
        return None, None, ""
    coef = result.params[pname]
    se   = result.bse[pname]
    pval = result.pvalues[pname]
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
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
         "rows": [(m,  "Relatedness",         "coverage")]},
        {"label": "M2", "name": "Relatedness + Overall Sentiment",
         "rows": [(m,  "Relatedness",         "coverage"),
                  (s,  "Overall Sentiment",   "sentiment")]},
        {"label": "M3", "name": "Positive Mean + Negative Mean",
         "rows": [(pm, "Positive Mean Score", "pos"),
                  (nm, "Negative Mean Score", "neg")]},
    ]


def run_topic_models(df, topic):
    results = []
    for sp in topic_model_specs(topic):
        x_vars = [row[0] for row in sp["rows"]]
        df_m   = df.dropna(subset=["log_mc"] + x_vars + ["stock_code", "year"]).copy()
        res, n, nc = panel_ols(df_m, "log_mc", x_vars)
        coefs = {}
        for col, label, dim in sp["rows"]:
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
        var_map = {"match": f"match_{topic}_lag1", "sentiment": "sentiment_mean_lag1"}
    elif spec_label == "M3":
        x_vars  = [f"tend_{topic}_pos_mean_lag1", f"tend_{topic}_neg_mean_lag1"]
        var_map = {"pos_mean": f"tend_{topic}_pos_mean_lag1",
                   "neg_mean": f"tend_{topic}_neg_mean_lag1"}
    rows = []
    for sec in sectors:
        df_s = df[df.sector == sec].dropna(subset=["log_mc"] + x_vars).copy()
        row  = {"Sector": sec, "N obs": len(df_s), "N co.": df_s.stock_code.nunique()}
        try:
            res_s, _, _ = panel_ols(df_s, "log_mc", x_vars)
            for key, col in var_map.items():
                c, se, stars = get_coef(res_s, col)
                row[f"coef_{key}"] = c
                row[f"se_{key}"]   = se
                row[f"β_{key}"] = f"{c:.4f}{stars}" if c is not None else "—"
                row[f"SE_{key}"]   = f"({se:.4f})"      if se is not None else ""
            row["Within R²"] = round(res_s.rsquared, 4)
        except Exception:
            for key in var_map:
                row[f"coef_{key}"] = row[f"se_{key}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def run_identification_checks(df, topic):
    match_l1 = f"match_{topic}_lag1"
    match_l2 = f"match_{topic}_lag2"
    rows = []
    checks = [
        ("R0 Baseline M1 (t-1)", match_l1, "log_mc",  False, False, False, False),
        ("R1 t-2 lag",           match_l2, "log_mc",  False, False, False, False),
        ("R2 ΔlogMC",       match_l1, "dlog_mc", False, False, False, False),
        ("R3 Excl. 2020",        match_l1, "log_mc",  True,  False, False, False),
        ("R4 Balanced panel",    match_l1, "log_mc",  False, True,  False, False),
        ("R5 Win. ΔlogMC",  match_l1, "dlog_mc", False, False, False, True),
    ]
    for label, mvar, y_col, excl2020, balanced, rate_ctrl, winsor in checks:
        if mvar not in df.columns:
            rows.append({"Check": label, "coef": None, "se": None,
                         "β": "n/a", "SE": "", "Outcome": y_col,
                         "N obs": "", "N co.": "", "R²": ""})
            continue
        y = "dlog_win" if winsor else y_col
        df_c = df.dropna(subset=[y_col, mvar]).copy()
        if winsor:
            p01, p99 = df_c[y_col].quantile([0.01, 0.99])
            df_c["dlog_win"] = df_c[y_col].clip(p01, p99)
        if excl2020:
            df_c = df_c[df_c.year != 2020].copy()
        if balanced:
            yc   = df_c.groupby("stock_code")["year"].count()
            df_c = df_c[df_c.stock_code.isin(yc[yc == df_c.year.nunique()].index)].copy()
        try:
            res, n, nc = panel_ols(df_c, y, mvar,
                                   time_fe=(not rate_ctrl), add_rate=rate_ctrl)
            c, se, stars = get_coef(res, mvar)
            rows.append({"Check": label, "coef": c, "se": se,
                         "β": f"{c:.4f}{stars}", "SE": f"({se:.4f})",
                         "Outcome": y, "N obs": f"{n:,}", "N co.": f"{nc:,}",
                         "R²": f"{res.rsquared:.3f}"})
        except Exception as e:
            rows.append({"Check": label, "coef": None, "se": None,
                         "β": f"err: {e}", "SE": "",
                         "Outcome": y, "N obs": "", "N co.": "", "R²": ""})
    return pd.DataFrame(rows)


# ── Run all regressions ───────────────────────────────────────────────────────
print("Running regressions...")
topic_results  = {t: run_topic_models(panel, t) for t in ENV_TOPICS}
sector_results = {
    spec: {t: run_sector_spec(panel, t, spec) for t in ENV_TOPICS}
    for spec in ("M1", "M2", "M3")
}
robust_results = {t: run_identification_checks(panel, t) for t in ENV_TOPICS}

# Mechanism Test 1: fundamentals-controlled
fund_ctrl = ["roa_pct_lag1", "ltdebt_assets_pct_lag1"]
mech1_rows = []
for topic in ENV_TOPICS:
    lbl, match_var = ENV_TOPIC_LABELS[topic], f"match_{topic}_lag1"
    ctrl = [v for v in fund_ctrl if v in panel.columns]
    df_b = panel.dropna(subset=["log_mc", match_var] + ctrl).copy()
    res_b, n_b, nc_b = panel_ols(df_b, "log_mc", match_var)
    c_b, se_b, st_b = get_coef(res_b, match_var)
    res_c, _, _ = panel_ols(df_b, "log_mc", [match_var] + ctrl)
    c_c, se_c, st_c = get_coef(res_c, match_var)
    att = (1 - abs(c_c) / abs(c_b)) * 100 if c_b and c_c else None
    mech1_rows.append({
        "Topic": lbl,
        "β Baseline": f"{c_b:.4f}{st_b}" if c_b is not None else "—",
        "SE Base":   f"({se_b:.4f})" if se_b is not None else "",
        "β + Controls": f"{c_c:.4f}{st_c}" if c_c is not None else "—",
        "SE Ctrl":   f"({se_c:.4f})" if se_c is not None else "",
        "Attenuation %": f"{att:.1f}%" if att is not None else "",
        "N obs":     f"{n_b:,}",
        "coef_b": c_b, "se_b": se_b, "coef_c": c_c, "se_c": se_c,
    })
df_mech1 = pd.DataFrame(mech1_rows)

# Mechanism Test 2: IBES mediation
mech2_rows = []
for topic in ENV_TOPICS:
    lbl, match_var = ENV_TOPIC_LABELS[topic], f"match_{topic}_lag1"
    df_s2 = panel.dropna(subset=["log_mc", match_var, "ibes_rec_mean_lag1"]).copy()
    try:
        r_s1, n_s1, _ = panel_ols(
            panel.dropna(subset=["ibes_rec_mean", match_var]).copy(),
            "ibes_rec_mean", match_var)
        c1, se1, st1 = get_coef(r_s1, match_var)
    except Exception:
        c1 = se1 = st1 = None
    try:
        r_2a, n_2a, nc_2a = panel_ols(df_s2, "log_mc", match_var)
        c2a, se2a, st2a = get_coef(r_2a, match_var)
    except Exception:
        c2a = se2a = st2a = None; n_2a = nc_2a = 0
    try:
        r_2b, _, _ = panel_ols(df_s2, "log_mc", [match_var, "ibes_rec_mean_lag1"])
        c2b, se2b, st2b = get_coef(r_2b, match_var)
        c_rec, se_rec, st_rec = get_coef(r_2b, "ibes_rec_mean_lag1")
    except Exception:
        c2b = se2b = st2b = c_rec = se_rec = st_rec = None
    att2 = (1 - abs(c2b) / abs(c2a)) * 100 if c2a and c2b else None
    mech2_rows.append({
        "Topic": lbl,
        "β S1 (Match→Rec)": f"{c1:.4f}{st1}"   if c1   is not None else "—",
        "β S2a (Match→MC)": f"{c2a:.4f}{st2a}" if c2a  is not None else "—",
        "β S2b (Match+Rec→MC)": f"{c2b:.4f}{st2b}" if c2b is not None else "—",
        "β S2b (Rec→MC)":   f"{c_rec:.4f}{st_rec}" if c_rec is not None else "—",
        "Attenuation %": f"{att2:.1f}%" if att2 is not None else "",
        "N (IBES)": f"{n_2a:,}",
        "c2a": c2a, "se2a": se2a, "c2b": c2b, "se2b": se2b,
    })
df_mech2 = pd.DataFrame(mech2_rows)

# Mechanism Test 3: M3 signal direction
mech3_rows = []
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    pm, nm = f"tend_{topic}_pos_mean_lag1", f"tend_{topic}_neg_mean_lag1"
    df_m3 = panel.dropna(subset=["log_mc", pm, nm]).copy()
    try:
        res_m3, n_m3, _ = panel_ols(df_m3, "log_mc", [pm, nm])
        c_p, se_p, st_p = get_coef(res_m3, pm)
        c_n, se_n, st_n = get_coef(res_m3, nm)
    except Exception:
        c_p = se_p = st_p = c_n = se_n = st_n = None; n_m3 = 0
    if c_p is not None and c_n is not None:
        pattern = ("Bilateral signalling" if c_p > 0 and c_n > 0 else
                   "Growth signal / Risk penalty" if c_p > 0 and c_n < 0 else
                   "Risk transparency dominates" if c_p < 0 and c_n > 0 else
                   "Ambiguous")
    else:
        pattern = "n/a"
    mech3_rows.append({
        "Topic": lbl,
        "β Pos Mean": f"{c_p:.4f}{st_p}" if c_p is not None else "—",
        "SE Pos": f"({se_p:.4f})" if se_p is not None else "",
        "β Neg Mean": f"{c_n:.4f}{st_n}" if c_n is not None else "—",
        "SE Neg": f"({se_n:.4f})" if se_n is not None else "",
        "Pattern": pattern, "N obs": f"{n_m3:,}",
        "c_p": c_p, "se_p": se_p, "c_n": c_n, "se_n": se_n,
    })
df_mech3 = pd.DataFrame(mech3_rows)

# Mechanism Test 4: ROA quartile
panel["roa_q"] = panel.groupby("year")["roa_pct"].transform(
    lambda x: pd.qcut(x, q=4,
                      labels=["Q1 (Low ROA)", "Q2", "Q3", "Q4 (High ROA)"],
                      duplicates="drop"))
ROA_QUARTILES = ["Q1 (Low ROA)", "Q2", "Q3", "Q4 (High ROA)"]
Q_COLORS = {"Q1 (Low ROA)": "#e74c3c", "Q2": "#e67e22",
            "Q3": "#27ae60",  "Q4 (High ROA)": "#2980b9"}
mech4 = {}
for topic in ENV_TOPICS:
    mvar = f"match_{topic}_lag1"
    rows = []
    for q in ROA_QUARTILES:
        df_q = panel[panel.roa_q == q].dropna(subset=["log_mc", mvar]).copy()
        try:
            res_q, n_q, nc_q = panel_ols(df_q, "log_mc", mvar)
            c, se, st = get_coef(res_q, mvar)
        except Exception:
            c = se = st = None; n_q = nc_q = 0
        rows.append({"ROA Quartile": q, "coef": c, "se": se,
                     "β": f"{c:.4f}{st}" if c is not None else "—",
                     "SE": f"({se:.4f})" if se is not None else "",
                     "N obs": f"{n_q:,}", "N co.": f"{nc_q:,}"})
    mech4[topic] = pd.DataFrame(rows)

print("  Regressions done.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("Saving figures...")


# ── fig01: Business disclosure trends ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    color = TOPIC_COLORS[topic]
    mu = panel.groupby("year")[f"match_{topic}"].mean().reindex(years)
    ax.plot(years, mu.values, color=color, lw=1.8, marker="o", ms=4)
    ax.set_xlabel("Year")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    ax.set_ylabel("Relatedness")
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
ax = axf[7]
mu_s = panel.groupby("year")["sentiment_mean"].mean().reindex(years)
ax.plot(years, mu_s.values, color=DIM_COLORS["sentiment"], lw=1.8, marker="s", ms=4)
ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
ax.set_xlabel("Year"); ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, fontsize=7.5)
ax.set_ylabel("Overall Sentiment")
label_panel(ax, "h", "Overall Sentiment")
plt.tight_layout()
fig.savefig(FIGS / "fig01_business_disclosure_trends.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig02: pos_ratio vs neg_ratio trends ─────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    mu_pr = panel.groupby("year")[f"tend_{topic}_pos_ratio"].mean().reindex(years)
    mu_nr = panel.groupby("year")[f"tend_{topic}_neg_ratio"].mean().reindex(years)
    ax.plot(years, mu_pr.values, color=DIM_COLORS["pos"], lw=1.8, marker="o", ms=4, label="Positive")
    ax.plot(years, mu_nr.values, color=DIM_COLORS["neg"], lw=1.8, marker="s", ms=4, label="Negative")
    ax.set_xlabel("Year"); ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, fontsize=7.5)
    ax.set_ylabel("Ratio"); ax.legend(fontsize=7)
    label_panel(ax, LETTERS[idx], ENV_TOPIC_LABELS[topic])
hide_unused(axf, len(ENV_TOPICS))
plt.tight_layout()
fig.savefig(FIGS / "fig02_pos_neg_ratio_trends.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig03: Market cap distributions ──────────────────────────────────────────
_PAL = {"log_mc": "#264653", "dlog_mc": "#e76f51"}

def boxplot_by_year(ax, col, year_list):
    color = _PAL[col]
    data  = [panel.loc[panel.year == y, col].dropna().values for y in year_list]
    bp = ax.boxplot(data, positions=range(len(year_list)), widths=0.6,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(linewidth=0),
                    whiskerprops=dict(linewidth=1.0, color=color, alpha=0.6),
                    capprops=dict(linewidth=1.2, color=color),
                    medianprops=dict(color="white", linewidth=2.2))
    for patch in bp["boxes"]:
        patch.set_facecolor(color); patch.set_alpha(0.75)
    medians = [np.median(d) for d in data]
    ax.scatter(range(len(year_list)), medians, color="white", s=22, zorder=4,
               linewidths=0.8, edgecolors=color)
    ax.set_xticks(range(len(year_list)))
    ax.set_xticklabels(year_list, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#bbb", linewidth=0.7, linestyle="--")
    for i in range(0, len(year_list), 2):
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.04, zorder=0)

years_dlog = sorted(panel.dropna(subset=["dlog_mc"]).year.unique())
fig, axes = plt.subplots(2, 1, figsize=(10, 9))
for ax, (col, yr, lbl, name) in zip(axes, [
    ("log_mc",  years,      "a", "Log Market Capitalisation (log JPY)"),
    ("dlog_mc", years_dlog, "b", "Annual Log Market Cap Growth (Δlog JPY)"),
]):
    boxplot_by_year(ax, col, yr)
    label_panel(ax, lbl, name)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel(name, fontsize=9, color=_PAL[col])
    ax.tick_params(axis="y", labelcolor=_PAL[col])
    if col == "log_mc":
        ax.set_ylim(20, 30)
plt.tight_layout()
fig.savefig(FIGS / "fig03_market_cap_distributions.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig04: Sector profile radar ───────────────────────────────────────────────
_sec_list = sectors
n_topics  = len(ENV_TOPICS)
angles    = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
angles_c  = np.concatenate([angles, [angles[0]]])
LAYERS = [
    ("match",    "Relatedness",   "#2980b9", 0.20, [f"match_{t}" for t in ENV_TOPICS]),
    ("pos_mean", "Positive Mean", "#27ae60", 0.25, [f"tend_{t}_pos_mean" for t in ENV_TOPICS]),
    ("neg_mean", "Neg Mean",      "#e74c3c", 0.25, [f"tend_{t}_neg_mean" for t in ENV_TOPICS]),
]
gstats = {}
for key, lbl, col, alpha, cols in LAYERS:
    gstats[key] = {}
    for c in cols:
        vals = panel[c].dropna().values
        gstats[key][c] = (float(vals.mean()), float(vals.std()))
records = {}
for sec in _sec_list:
    df_s = panel[panel.sector == sec]
    row  = {}
    for key, lbl, col, alpha, cols in LAYERS:
        row[key] = [(float(df_s[c].mean()) - gstats[key][c][0])
                    / gstats[key][c][1] for c in cols]
    records[sec] = row
all_z  = [v for rec in records.values() for vals in rec.values() for v in vals]
offset = max(0.0, -min(all_z)) + 0.4
r_ann_base = offset + 1.1; DR = 0.33
ncols = 3; nrows = int(np.ceil(len(_sec_list) / ncols))
fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True),
                         figsize=(15, nrows * 5.0))
axf = axes.flatten()
short_labels = []
for t in ENV_TOPICS:
    words = ENV_TOPIC_LABELS[t].split()
    short_labels.append("\n".join(words) if len(words) > 1 else ENV_TOPIC_LABELS[t])
for idx, sec in enumerate(_sec_list):
    ax  = axf[idx]
    rec = records[sec]
    theta_ref = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_ref, [offset] * 300, color="#999", lw=0.8, ls="--", alpha=0.6, zorder=1)
    for key, lbl, col, alpha, _ in LAYERS:
        r_vals = [z + offset for z in rec[key]]
        r_c    = r_vals + [r_vals[0]]
        ax.plot(angles_c, r_c, color=col, lw=1.5, zorder=3)
        ax.fill(angles_c, r_c, color=col, alpha=alpha, zorder=2)
    for i, angle in enumerate(angles):
        for j, (key, lbl, col, alpha, _) in enumerate(LAYERS):
            z = rec[key][i]
            ax.text(angle, r_ann_base + j * DR, f"{z:+.2f}",
                    ha="center", va="center", fontsize=5.0, color=col,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                    zorder=6)
    ax.set_xticks(angles); ax.set_xticklabels(short_labels, size=7)
    z_ticks = [-1, 0, 1, 2]
    r_ticks = [z + offset for z in z_ticks if z + offset >= 0]
    z_shown = [z for z in z_ticks if z + offset >= 0]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{z:+d}σ" for z in z_shown], size=6, color="#555")
    ax.set_ylim(0, None)
    ax.set_title(f"({LETTERS[idx]}) {sec}", size=9, fontweight="bold", pad=14)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", alpha=0.25, lw=0.5)
hide_unused(axf, len(_sec_list))
fig.legend(handles=[plt.Line2D([0], [0], color=col, lw=2, label=lbl)
                    for _, lbl, col, _, _ in LAYERS],
           loc="lower right", fontsize=9, bbox_to_anchor=(0.98, 0.02))
plt.tight_layout()
fig.savefig(FIGS / "fig04_sector_radar.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig05: Baseline coefficient comparison M1-M3 (7 topic panels) ────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    plot_rows = []
    for m in topic_results[topic]:
        for col, info in m["coefs"].items():
            if info["coef"] is not None:
                plot_rows.append({"label": f"{m['label']}: {info['label']}",
                                  "coef": info["coef"], "se": info["se"],
                                  "dim": info["dim"]})
    for i, row in enumerate(plot_rows):
        color = DIM_COLORS[row["dim"]]
        ax.errorbar(row["coef"], i, xerr=1.96 * row["se"],
                    fmt="none", ecolor=color + "88", elinewidth=1.2, capsize=3)
        ax.plot(row["coef"], i, "o", color=color, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels([r["label"] for r in plot_rows], fontsize=7.5)
    ax.set_xlabel("Coefficient  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
hide_unused(axf, len(ENV_TOPICS))
fig.legend(handles=[mpatches.Patch(color=DIM_COLORS[k], label=v)
                    for k, v in [("coverage", "Coverage (M1/M2)"),
                                 ("sentiment", "Overall Sentiment (M2)"),
                                 ("pos", "Positive Mean (M3)"),
                                 ("neg", "Negative Mean (M3)")]],
           fontsize=8, loc="lower right", bbox_to_anchor=(0.99, 0.01))
plt.tight_layout()
fig.savefig(FIGS / "fig05_baseline_coef_M1_M3.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig06: Mechanism Test 1 — information channel ────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    row = df_mech1.iloc[idx]
    specs = [("Baseline M1", row["coef_b"], row["se_b"], "#2980b9"),
             ("M1 + ROA & Leverage", row["coef_c"], row["se_c"], "#e67e22")]
    for i, (name, c, se, color) in enumerate(specs):
        if c is not None and se is not None:
            ax.errorbar(c, i, xerr=1.96 * se,
                        fmt="none", ecolor=color + "99", elinewidth=1.5, capsize=4)
            ax.plot(c, i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Baseline", "+ Controls"], fontsize=8)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
    att = df_mech1.iloc[idx]["Attenuation %"]
    ax.text(0.98, 0.05, f"Atten.: {att}", transform=ax.transAxes,
            ha="right", fontsize=7.5, color="#555")
hide_unused(axf, len(ENV_TOPICS))
plt.tight_layout()
fig.savefig(FIGS / "fig06_mechanism_t1_info_channel.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig07: Mechanism Test 2 — IBES mediation ─────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax  = axf[idx]
    row = df_mech2.iloc[idx]
    specs = [("Baseline (IBES sample)", row["c2a"], row["se2a"], "#2980b9"),
             ("+ IBES Rec ctrl",        row["c2b"], row["se2b"], "#8e44ad")]
    for i, (name, c, se, color) in enumerate(specs):
        if c is not None and se is not None:
            ax.errorbar(c, i, xerr=1.96 * se,
                        fmt="none", ecolor=color + "88", elinewidth=1.4, capsize=4)
            ax.plot(c, i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Baseline", "+ IBES Rec"], fontsize=8)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
    att = row["Attenuation %"]
    ax.text(0.98, 0.05, f"Atten.: {att}", transform=ax.transAxes,
            ha="right", fontsize=7.5, color="#555")
hide_unused(axf, len(ENV_TOPICS))
plt.tight_layout()
fig.savefig(FIGS / "fig07_mechanism_t2_ibes_mediation.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig08: Mechanism Test 3 — signal direction bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(ENV_TOPICS))
w = 0.35
for i, row in df_mech3.iterrows():
    c_p, se_p = row["c_p"], row["se_p"]
    c_n, se_n = row["c_n"], row["se_n"]
    if c_p is not None and se_p is not None:
        ax.bar(x[i] - w/2, c_p, width=w, color="#27ae60", alpha=0.75,
               label="Positive Mean" if i == 0 else "")
        ax.errorbar(x[i] - w/2, c_p, yerr=1.96 * se_p,
                    fmt="none", color="#27ae60", capsize=3, lw=1.2)
    if c_n is not None and se_n is not None:
        ax.bar(x[i] + w/2, c_n, width=w, color="#e74c3c", alpha=0.75,
               label="Negative Mean" if i == 0 else "")
        ax.errorbar(x[i] + w/2, c_n, yerr=1.96 * se_n,
                    fmt="none", color="#e74c3c", capsize=3, lw=1.2)
ax.axhline(0, color="#555", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([l.replace(" ", "\n") for l in df_mech3["Topic"]], fontsize=8)
ax.set_ylabel("Coefficient  [TWFE, clustered SE]")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIGS / "fig08_mechanism_t3_signal_direction.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig09: Mechanism Test 4 — ROA quartile ────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax = axf[idx]
    plot_rows = [(row["ROA Quartile"], row["coef"], row["se"])
                 for _, row in mech4[topic].iterrows()
                 if row["coef"] is not None]
    for i, (q, c, se) in enumerate(plot_rows):
        color = Q_COLORS[q]
        ax.errorbar(c, i, xerr=1.96 * se,
                    fmt="none", ecolor=color + "99", elinewidth=1.4, capsize=4)
        ax.plot(c, i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels([r[0] for r in plot_rows], fontsize=8)
    ax.set_xlabel("M1 Match coef  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
hide_unused(axf, len(ENV_TOPICS))
fig.legend(handles=[mpatches.Patch(color=Q_COLORS[q], label=q) for q in ROA_QUARTILES],
           loc="lower right", fontsize=8, bbox_to_anchor=(0.99, 0.02))
plt.tight_layout()
fig.savefig(FIGS / "fig09_mechanism_t4_roa_quartile.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig10: Identification robustness ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax   = axf[idx]
    df_r = robust_results[topic].dropna(subset=["coef"])
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_r["coef"].values, range(len(df_r)),
                xerr=1.96 * df_r["se"].values,
                fmt="none", ecolor="#aaa", elinewidth=1.2, capsize=3)
    for i, (_, row) in enumerate(df_r.iterrows()):
        dot_color = (color if "Baseline" in row["Check"] else
                     "#c0392b" if "t-2" in row["Check"] else "#555")
        ax.plot(row["coef"], i, "o", color=dot_color, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks(range(len(df_r)))
    ax.set_yticklabels(df_r["Check"].values, fontsize=7.5)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
hide_unused(axf, len(ENV_TOPICS))
fig.legend(handles=[
    mpatches.Patch(color=TOPIC_COLORS[ENV_TOPICS[0]], label="Baseline (t−1)"),
    mpatches.Patch(color="#c0392b", label="t-2 lag (sharpest RC test)"),
    mpatches.Patch(color="#555", label="Other checks"),
], loc="lower right", fontsize=8, bbox_to_anchor=(0.99, 0.02))
plt.tight_layout()
fig.savefig(FIGS / "fig10_identification_robustness.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)


# ── fig11: Sector heterogeneity M1 ───────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axf = axes.flatten()
for idx, topic in enumerate(ENV_TOPICS):
    ax   = axf[idx]
    df_s = sector_results["M1"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_s["coef_match"].values, range(len(df_s)),
                xerr=1.96 * df_s["se_match"].values,
                fmt="o", color=color, ms=5, ecolor="#aaa", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks(range(len(df_s))); ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
    ax.set_xlabel("M1 Relatedness  [95% CI]", fontsize=8)
    ax.set_title(f"({LETTERS[idx]}) {ENV_TOPIC_LABELS[topic]}", fontsize=9, fontweight="bold")
hide_unused(axf, len(ENV_TOPICS))
plt.tight_layout()
fig.savefig(FIGS / "fig11_sector_heterogeneity_M1.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)

print("  Figures done.")


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("Saving tables...")


def save_xlsx(df, path, sheet="Sheet1"):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=True)


# Table 1: Descriptive Statistics
desc_cols = (
    [f"match_{t}" for t in ENV_TOPICS] +
    [f"tend_{t}_pos_mean" for t in ENV_TOPICS] +
    [f"tend_{t}_neg_mean" for t in ENV_TOPICS] +
    ["sentiment_mean", "log_mc", "dlog_mc",
     "roa_pct", "ltdebt_assets_pct", "rnd_share_pct", "ibes_rec_mean"]
)
desc_cols = [c for c in desc_cols if c in panel.columns]
desc_data = panel[desc_cols].copy()
desc_data.replace([np.inf, -np.inf], np.nan, inplace=True)  # drop inf (e.g. R&D share)

_t1_labels = {}
for t in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[t]
    _t1_labels[f"match_{t}"]          = f"{lbl} — Relatedness (M1)"
    _t1_labels[f"tend_{t}_pos_mean"]   = f"{lbl} — Positive Mean (M3)"
    _t1_labels[f"tend_{t}_neg_mean"]   = f"{lbl} — Negative Mean (M3)"
_t1_labels.update({
    "sentiment_mean":     "Overall ESG Sentiment",
    "log_mc":             "Log Market Cap",
    "dlog_mc":            "ΔLog Market Cap (YoY)",
    "roa_pct":            "ROA (%)",
    "ltdebt_assets_pct":  "LT Debt / Assets (%)",
    "rnd_share_pct":      "R&D / Revenue (%)",
    "ibes_rec_mean":      "IBES Recommendation (Mean)",
})

desc = desc_data.describe().T.round(4)
desc.index = [_t1_labels.get(c, c) for c in desc.index]
save_xlsx(desc, TABLES / "Table1_DescriptiveStats.xlsx", "Descriptive Statistics")

# Table 2: Cross-topic summary M1-M3
rows2 = []
for spec_label, var_key, var_label in [
    ("M1", "match",    "Relatedness"),
    ("M2", "match",    "Relatedness"),
    ("M2", "sentiment","Overall Sentiment"),
    ("M3", "pos_mean", "Positive Mean Score"),
    ("M3", "neg_mean", "Negative Mean Score"),
]:
    row = {"Spec": spec_label, "Variable": var_label}
    for topic in ENV_TOPICS:
        for m in topic_results[topic]:
            if m["label"] != spec_label:
                continue
            for col, info in m["coefs"].items():
                col_key = (col.replace(f"match_{topic}_lag1", "match")
                              .replace(f"tend_{topic}_pos_mean_lag1", "pos_mean")
                              .replace(f"tend_{topic}_neg_mean_lag1", "neg_mean")
                              .replace("sentiment_mean_lag1", "sentiment"))
                if col_key != var_key:
                    continue
                c, se, stars = info["coef"], info["se"], info["stars"]
                row[ENV_TOPIC_LABELS[topic]] = (
                    f"{c:.4f}{stars}" if c is not None else "—")
                row[f"{ENV_TOPIC_LABELS[topic]} SE"] = (
                    f"({se:.4f})" if se is not None else "")
    rows2.append(row)
    rows2.append({"Spec": "", "Variable": "",
                  **{ENV_TOPIC_LABELS[t]: rows2[-1].get(f"{ENV_TOPIC_LABELS[t]} SE", "")
                     for t in ENV_TOPICS}})
df2 = pd.DataFrame(rows2).set_index(["Spec", "Variable"])
save_xlsx(df2, TABLES / "Table2_CrossTopic_Summary.xlsx", "Cross-Topic Summary")

# Tables S1-S7: Per-topic coefficient tables
for i, topic in enumerate(ENV_TOPICS, 1):
    lbl = ENV_TOPIC_LABELS[topic]
    rows = []
    for m in topic_results[topic]:
        for col, info in m["coefs"].items():
            c, se, stars = info["coef"], info["se"], info["stars"]
            rows.append({
                "Spec": m["label"], "Model": m["name"],
                "Variable": info["label"],
                "β": f"{c:.4f}{stars}" if c is not None else "—",
                "SE": f"({se:.4f})" if se is not None else "",
                "N obs": f"{m['n']:,}", "N co.": f"{m['nc']:,}",
                "Within R²": f"{m['result'].rsquared:.3f}",
            })
    df_t = pd.DataFrame(rows).set_index(["Spec", "Variable"])
    fname = f"S{i}_Coef_{lbl.replace(' ', '').replace('&', '')[:20]}.xlsx"
    save_xlsx(df_t, TABLES / fname, lbl[:31])

# Table 3: Mechanism Test 1 — information channel
t3 = df_mech1.drop(columns=["coef_b", "se_b", "coef_c", "se_c"]).set_index("Topic")
save_xlsx(t3, TABLES / "Table3_Mechanism_InfoChannel.xlsx", "Test1 Info Channel")

# Table 4: Mechanism Test 2 — IBES mediation
t4 = df_mech2.drop(columns=["c2a", "se2a", "c2b", "se2b"]).set_index("Topic")
save_xlsx(t4, TABLES / "Table4_Mechanism_IBESMediation.xlsx", "Test2 IBES Mediation")

# Table 5: Mechanism Test 3 — signal direction
t5 = df_mech3.drop(columns=["c_p", "se_p", "c_n", "se_n"]).set_index("Topic")
save_xlsx(t5, TABLES / "Table5_Mechanism_SignalDirection.xlsx", "Test3 Signal Direction")

# Table 6: Mechanism Test 4 — ROA quartile (all topics combined)
rows6 = []
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    for _, row in mech4[topic].iterrows():
        rows6.append({"Topic": lbl, **row.to_dict()})
df6 = pd.DataFrame(rows6).drop(columns=["coef", "se"]).set_index(["Topic", "ROA Quartile"])
save_xlsx(df6, TABLES / "Table6_Mechanism_ROAQuartile.xlsx", "Test4 ROA Quartile")

# Table 7: Identification robustness
rows7 = []
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    for _, row in robust_results[topic].iterrows():
        rows7.append({"Topic": lbl, "Check": row["Check"],
                      "β": row["β"], "SE": row["SE"],
                      "Outcome": row["Outcome"],
                      "N obs": row.get("N obs",""), "N co.": row.get("N co.",""),
                      "R²": row.get("R²","")})
df7 = pd.DataFrame(rows7).set_index(["Topic", "Check"])
save_xlsx(df7, TABLES / "Table7_IdentificationRobustness.xlsx", "Identification Robustness")

# Table 8: Sector heterogeneity M1
rows8 = []
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    for _, row in sector_results["M1"][topic].iterrows():
        rows8.append({"Topic": lbl, "Sector": row["Sector"],
                      "β match": row.get("β_match", "—"),
                      "SE match": row.get("SE_match", ""),
                      "Within R²": row.get("Within R²", ""),
                      "N obs": row["N obs"], "N co.": row["N co."]})
df8 = pd.DataFrame(rows8).set_index(["Topic", "Sector"])
save_xlsx(df8, TABLES / "Table8_SectorHet_M1.xlsx", "Sector Heterogeneity M1")

# Table 9: Sector heterogeneity M3
rows9 = []
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    for _, row in sector_results["M3"][topic].iterrows():
        rows9.append({"Topic": lbl, "Sector": row["Sector"],
                      "β pos_mean": row.get("β_pos_mean", "—"),
                      "SE pos": row.get("SE_pos_mean", ""),
                      "β neg_mean": row.get("β_neg_mean", "—"),
                      "SE neg": row.get("SE_neg_mean", ""),
                      "Within R²": row.get("Within R²", ""),
                      "N obs": row["N obs"], "N co.": row["N co."]})
df9 = pd.DataFrame(rows9).set_index(["Topic", "Sector"])
save_xlsx(df9, TABLES / "Table9_SectorHet_M3.xlsx", "Sector Heterogeneity M3")

print("  Tables done.")


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("Saving metadata...")

variable_dict = {
    "log_mc": {
        "variableNameInDataset": "log_mc",
        "variableNameInArticle": "Log Market Capitalisation",
        "variableAttribute": "continuous",
        "variableUnit": "log JPY",
        "variableDescription": "Natural logarithm of annual market capitalisation. Primary outcome variable.",
        "variableSource": "market_cap",
        "observationNumber": int(panel["log_mc"].notna().sum()),
    },
    "dlog_mc": {
        "variableNameInDataset": "dlog_mc",
        "variableNameInArticle": "Annual Log MC Growth",
        "variableAttribute": "continuous",
        "variableUnit": "log JPY difference",
        "variableDescription": "Year-over-year change in log market cap. Secondary outcome.",
        "variableSource": "market_cap",
        "observationNumber": int(panel["dlog_mc"].notna().sum()),
    },
}
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    variable_dict[f"match_{topic}_lag1"] = {
        "variableNameInDataset": f"match_{topic}_lag1",
        "variableNameInArticle": f"{lbl} — Relatedness (t-1)",
        "variableAttribute": "continuous",
        "variableUnit": "index [0,1]",
        "variableDescription": f"M1 variable. Coverage volume for {lbl} topic, lagged 1 year.",
        "variableSource": "nlp_scores",
        "observationNumber": int(panel[f"match_{topic}_lag1"].notna().sum()),
    }
    variable_dict[f"tend_{topic}_pos_mean_lag1"] = {
        "variableNameInDataset": f"tend_{topic}_pos_mean_lag1",
        "variableNameInArticle": f"{lbl} — Positive Mean Score (t-1)",
        "variableAttribute": "continuous",
        "variableUnit": "index (positive)",
        "variableDescription": f"M3 variable. Mean positive-sentiment intensity among {lbl}-related fragments, lagged 1 year.",
        "variableSource": "nlp_scores",
        "observationNumber": int(panel[f"tend_{topic}_pos_mean_lag1"].notna().sum()),
    }
    variable_dict[f"tend_{topic}_neg_mean_lag1"] = {
        "variableNameInDataset": f"tend_{topic}_neg_mean_lag1",
        "variableNameInArticle": f"{lbl} — Negative Mean Score (t-1)",
        "variableAttribute": "continuous",
        "variableUnit": "index (negative)",
        "variableDescription": f"M3 variable. Mean negative-sentiment intensity among {lbl}-related fragments, lagged 1 year.",
        "variableSource": "nlp_scores",
        "observationNumber": int(panel[f"tend_{topic}_neg_mean_lag1"].notna().sum()),
    }
for v, lbl_, unit, desc in [
    ("roa_pct_lag1",          "ROA (t-1)",            "%",       "Return on assets, lagged 1 year. Profitability control."),
    ("ltdebt_assets_pct_lag1","LT Debt/Assets (t-1)", "%",       "Long-term debt as % of assets, lagged 1 year. Leverage control."),
    ("rnd_share_pct_lag1",    "R&D/Revenue (t-1)",    "%",       "R&D expenditure as % of revenue, lagged 1 year. Innovation intensity."),
    ("ibes_rec_mean_lag1",    "IBES Recommendation (t-1)", "1-5","Analyst consensus recommendation, lagged 1 year. 1=Strong Buy, 5=Sell."),
]:
    if v in panel.columns:
        variable_dict[v] = {
            "variableNameInDataset": v,
            "variableNameInArticle": lbl_,
            "variableAttribute": "continuous",
            "variableUnit": unit,
            "variableDescription": desc,
            "variableSource": "lseg",
            "observationNumber": int(panel[v].notna().sum()),
        }

with open(META / "variable_dictionary.yaml", "w", encoding="utf-8") as f:
    yaml.dump(variable_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

dataset_dict = {
    "nlp_scores": {
        "datasetSourceName": "ESG NLP Scores (match, sentiment, tendency)",
        "datasetDescription": "Fragment-level NLP scores for 23 ESG themes derived from Japanese securities reports (有価証券報告書). Aggregated to company-year level.",
        "datasetAttribute": "table",
        "datasetProvider": "Japan Financial Services Agency (金融庁) / EDINET — NLP pipeline applied by research team",
        "datasetGenerationInfo": "Securities reports downloaded from EDINET. Fragment-level processing via SAPT model. match_scores, tendency_scores, sentiment_scores CSV files.",
        "datasetPeriod": "2016 to 2025",
        "datasetScale": "Japan — all TSE-listed companies",
        "datasetSpatialResolution": "company",
        "datasetTemporalResolution": "annual",
        "datasetPreprocessingInfo": "Japanese theme names translated to English snake_case. Sub-scores extracted for 7 Business topics. All ESG variables lagged 1 year (t-1) and 2 years (t-2) for identification.",
    },
    "market_cap": {
        "datasetSourceName": "Annual Market Capitalisation",
        "datasetDescription": "Annual market capitalisation in JPY for Japanese listed companies.",
        "datasetAttribute": "table",
        "datasetProvider": "LSEG (London Stock Exchange Group, formerly Refinitiv)",
        "datasetGenerationInfo": "Downloaded from LSEG Workspace. Wide format (companies × years), reshaped to long.",
        "datasetPeriod": "2015 to 2025",
        "datasetScale": "Japan — TSE-listed companies",
        "datasetSpatialResolution": "company",
        "datasetTemporalResolution": "annual",
        "datasetPreprocessingInfo": "Reshaped wide-to-long. log(MC) and year-over-year log growth computed.",
    },
    "lseg": {
        "datasetSourceName": "LSEG Fundamentals and IBES Analyst Consensus",
        "datasetDescription": "Annual financial statement data (ROA, leverage, R&D, revenue) and IBES analyst consensus forecasts (EPS, revenue, recommendation, price target).",
        "datasetAttribute": "table",
        "datasetProvider": "LSEG (London Stock Exchange Group)",
        "datasetGenerationInfo": "Downloaded from LSEG Workspace using RIC identifiers (.T suffix for TSE). Fundamentals: fiscal-year end data. IBES: monthly FY1 consensus aggregated to annual via December month-end snapshot.",
        "datasetPeriod": "2015 to 2025",
        "datasetScale": "Japan — TSE-listed companies",
        "datasetSpatialResolution": "company",
        "datasetTemporalResolution": "annual (fundamentals); monthly aggregated to annual (IBES)",
        "datasetPreprocessingInfo": "RIC to stock_code: strip .T suffix. Fiscal-year duplicates (from year changes): keep row with most complete data. IBES: December month-end snapshot per RIC-year. All variables lagged 1 year.",
    },
    "gics": {
        "datasetSourceName": "GICS Sector Classification",
        "datasetDescription": "GICS sector/industry classification for Japanese listed companies.",
        "datasetAttribute": "table",
        "datasetProvider": "MSCI / S&P (via LSEG Workspace)",
        "datasetGenerationInfo": "MSCI_category_Japan_listed_companies.xlsx, static cross-section.",
        "datasetPeriod": "2025 (static)",
        "datasetScale": "Japan — TSE-listed companies",
        "datasetSpatialResolution": "company",
        "datasetTemporalResolution": "static",
        "datasetPreprocessingInfo": "11 GICS sectors. Merged to panel on stock_code.",
    },
}

with open(META / "dataset_dictionary.yaml", "w", encoding="utf-8") as f:
    yaml.dump(dataset_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

print("  Metadata done.")


# ══════════════════════════════════════════════════════════════════════════════
# ACTIONBRIEF
# ══════════════════════════════════════════════════════════════════════════════
print("Writing actionbrief.yaml...")

figure_dict = {
    "fig01": {
        "figureId": "Figure 1",
        "figurePath": "figures/fig01_business_disclosure_trends.png",
        "figureCaption": "Business ESG Disclosure Trends, 2016–2025. Cross-sectional mean relatedness score per Business topic and overall ESG sentiment by year.",
        "figureAim": "Establish temporal trends in Business disclosure coverage to motivate the panel regression design.",
        "figureExplanation": "8 panels (a–g: 7 Business topics, h: overall sentiment). Each panel shows the annual mean relatedness (match) score. Upward trends reflect growing Business disclosure intensity over the study period.",
        "figureSuitableSection": "Data and Measurement",
    },
    "fig02": {
        "figureId": "Figure 2",
        "figurePath": "figures/fig02_pos_neg_ratio_trends.png",
        "figureCaption": "Positive vs. Negative Disclosure Ratio Trends by Business Topic, 2016–2025.",
        "figureAim": "Show how the balance between positive and risk-acknowledging Business content evolves over time.",
        "figureExplanation": "3×3 grid, one panel per Business topic. Green line: positive fragment ratio; red line: negative fragment ratio. A widening gap suggests growing asymmetry in disclosure tone.",
        "figureSuitableSection": "Data and Measurement",
    },
    "fig03": {
        "figureId": "Figure 3",
        "figurePath": "figures/fig03_market_cap_distributions.png",
        "figureCaption": "Market Capitalisation Distributions by Year.",
        "figureAim": "Characterise the outcome variable distribution and confirm no structural breaks in the panel.",
        "figureExplanation": "Two box plots: (a) log MC level, (b) annual log MC growth. Boxes show IQR; whiskers at 10th/90th percentile; white dot = median. Dark teal = level; coral = growth.",
        "figureSuitableSection": "Data and Measurement",
    },
    "fig04": {
        "figureId": "Figure 4",
        "figurePath": "figures/fig04_sector_radar.png",
        "figureCaption": "Business Disclosure Profile by GICS Sector. z-scored sector mean relative to global average.",
        "figureAim": "Descriptive overview of which sectors lead or lag the market on Business disclosure dimensions.",
        "figureExplanation": "11 polar radar panels (one per GICS sector). Each spoke = one Business topic. Three overlaid areas: blue = Relatedness, green = Positive Mean, red = Negative Mean. All metrics z-scored as (sector mean − global mean) / global σ. Dashed circle = global average (z=0).",
        "figureSuitableSection": "Data and Measurement",
    },
    "fig05": {
        "figureId": "Figure 5",
        "figurePath": "figures/fig05_baseline_coef_M1_M3.png",
        "figureCaption": "Baseline Association: M1/M2/M3 Coefficients per Business Topic.",
        "figureAim": "Establish the baseline disclosure-market cap association across all three specifications for each Business topic.",
        "figureExplanation": "3×3 grid, one panel per topic (7 panels, 2 hidden). Horizontal CI plots: blue dots = M1/M2 coverage; purple = sentiment; green = positive mean; red = negative mean. Dashed red vertical = zero. Error bars = 95% CI.",
        "figureSuitableSection": "Results — Baseline Association",
    },
    "fig06": {
        "figureId": "Figure 6",
        "figurePath": "figures/fig06_mechanism_t1_info_channel.png",
        "figureCaption": "Test 1: Information Channel. Baseline M1 vs. M1 with Fundamentals Controls.",
        "figureAim": "Test whether Business disclosure predicts market cap beyond fundamental quality (ROA + leverage).",
        "figureExplanation": "7 panels. Each shows two dots: blue (baseline M1) and orange (M1 + ROA + LT leverage controls) on the same sample. Attenuation % shown in panel corner. If orange dot remains significant, information channel is confirmed.",
        "figureSuitableSection": "Results — Mechanism Tests",
    },
    "fig07": {
        "figureId": "Figure 7",
        "figurePath": "figures/fig07_mechanism_t2_ibes_mediation.png",
        "figureCaption": "Test 2: Analyst Attention Channel (IBES Mediation).",
        "figureAim": "Test whether the disclosure-MC association is partly transmitted through analyst recommendation (analyst attention channel).",
        "figureExplanation": "7 panels. Blue dot = match coefficient in IBES sample (S2a); purple dot = match coefficient after adding analyst recommendation control (S2b). Attenuation % = coefficient shrinkage. Larger attenuation signals stronger analyst mediation.",
        "figureSuitableSection": "Results — Mechanism Tests",
    },
    "fig08": {
        "figureId": "Figure 8",
        "figurePath": "figures/fig08_mechanism_t3_signal_direction.png",
        "figureCaption": "Test 3: M3 Signal Direction — Positive vs. Negative Business Disclosure.",
        "figureAim": "Determine whether positive and risk-acknowledging Business content carry the same or opposite valuation effects (bilateral signalling vs. risk penalty).",
        "figureExplanation": "Grouped bar chart. Green bars = pos_mean coefficient; red bars = neg_mean coefficient. Both above zero → bilateral signalling (transparency premium). Neg bar below zero → risk penalty. One group per Business topic.",
        "figureSuitableSection": "Results — Mechanism Tests",
    },
    "fig09": {
        "figureId": "Figure 9",
        "figurePath": "figures/fig09_mechanism_t4_roa_quartile.png",
        "figureCaption": "Test 4: ROA Quartile Heterogeneity.",
        "figureAim": "Distinguish signalling theory (stronger for low-ROA firms) from complementarity (stronger for high-ROA firms).",
        "figureExplanation": "7 panels. Four dots per panel, one per within-year ROA quartile (Q1 low → Q4 high). If the coefficient decreases monotonically from Q1 to Q4, signalling dominates. If it increases, complementarity dominates.",
        "figureSuitableSection": "Results — Mechanism Tests",
    },
    "fig10": {
        "figureId": "Figure 10",
        "figurePath": "figures/fig10_identification_robustness.png",
        "figureCaption": "Identification Robustness: M1 Match Coefficient across Six Checks.",
        "figureAim": "Verify that the baseline association is not driven by reverse causality, COVID shock, sample attrition, or outliers.",
        "figureExplanation": "7 panels. Topic-coloured dot = baseline (t-1). Red dot = t-2 lag (hardest reverse-causality test). Grey dots = R2–R5. Error bars = 95% CI. Consistent sign and approximate magnitude across checks supports robustness.",
        "figureSuitableSection": "Results — Identification",
    },
    "fig11": {
        "figureId": "Figure 11",
        "figurePath": "figures/fig11_sector_heterogeneity_M1.png",
        "figureCaption": "Sector Heterogeneity: M1 Coverage Volume Coefficient by GICS Sector.",
        "figureAim": "Identify which GICS sectors show the strongest Business disclosure premium.",
        "figureExplanation": "7 panels. Sectors sorted by coefficient magnitude. Error bars = 95% CI. Sectors with small sample (Utilities N=33, Energy N=26) should be interpreted cautiously.",
        "figureSuitableSection": "Results — Sector Heterogeneity",
    },
}

table_dict = {
    "Table1": {
        "tableId": "Table 1",
        "tablePath": "tables/Table1_DescriptiveStats.xlsx",
        "tableCaption": "Descriptive Statistics for Key Variables.",
        "tableAim": "Characterise the distribution of outcome, disclosure, and control variables.",
        "tableExplanation": "Rows = variables (7 match scores, 7 pos_mean, 7 neg_mean, sentiment, logMC, ΔlogMC, ROA, leverage, R&D share, IBES rec). Columns = N, mean, SD, min, p25, p50, p75, max.",
        "tableSuitableSection": "Data and Measurement",
    },
    "Table2": {
        "tableId": "Table 2",
        "tablePath": "tables/Table2_CrossTopic_Summary.xlsx",
        "tableCaption": "Cross-Topic Coefficient Summary: M1/M2/M3 for all 7 Business Topics.",
        "tableAim": "Provide a compact overview of all 21 baseline regression coefficients.",
        "tableExplanation": "Rows = 5 coefficient types (M1 match; M2 match, sentiment; M3 pos_mean, neg_mean). Columns = 7 Business topics. β and SE shown; stars: * p<0.10, ** p<0.05, *** p<0.01.",
        "tableSuitableSection": "Results — Baseline Association",
    },
    "Table3": {
        "tableId": "Table 3",
        "tablePath": "tables/Table3_Mechanism_InfoChannel.xlsx",
        "tableCaption": "Mechanism Test 1: Information Channel — Baseline vs. Fundamentals-Controlled M1.",
        "tableAim": "Show whether the disclosure coefficient survives controlling for ROA and leverage.",
        "tableExplanation": "Rows = 7 Business topics. Columns = β baseline, SE, β + controls, SE, attenuation %, N obs. Controls: ROA (t-1) + LT Debt/Assets (t-1), same sample.",
        "tableSuitableSection": "Results — Mechanism Tests",
    },
    "Table4": {
        "tableId": "Table 4",
        "tablePath": "tables/Table4_Mechanism_IBESMediation.xlsx",
        "tableCaption": "Mechanism Test 2: Analyst Attention Channel (Baron-Kenny IBES Mediation).",
        "tableAim": "Quantify the analyst-channel proportion of the Business disclosure-MC association.",
        "tableExplanation": "Rows = 7 Business topics. Columns = β S1 (match→IBES rec), β S2a (match→logMC, IBES sample), β S2b (match+rec→logMC), β S2b (rec→logMC), attenuation %, N (IBES).",
        "tableSuitableSection": "Results — Mechanism Tests",
    },
    "Table5": {
        "tableId": "Table 5",
        "tablePath": "tables/Table5_Mechanism_SignalDirection.xlsx",
        "tableCaption": "Mechanism Test 3: M3 Signal Direction Summary.",
        "tableAim": "Classify the sign pattern of positive and negative Business disclosure effects per topic.",
        "tableExplanation": "Rows = 7 Business topics. Columns = β pos_mean, SE, β neg_mean, SE, pattern (bilateral signalling / risk penalty / transparency dominates / ambiguous), N obs.",
        "tableSuitableSection": "Results — Mechanism Tests",
    },
    "Table6": {
        "tableId": "Table 6",
        "tablePath": "tables/Table6_Mechanism_ROAQuartile.xlsx",
        "tableCaption": "Mechanism Test 4: ROA Quartile Heterogeneity.",
        "tableAim": "Test whether Business disclosure premium is larger for lower- or higher-quality firms.",
        "tableExplanation": "Rows = 7 topics × 4 ROA quartiles. Columns = β, SE, N obs, N companies. Q1 = lowest within-year ROA; Q4 = highest.",
        "tableSuitableSection": "Results — Mechanism Tests",
    },
    "Table7": {
        "tableId": "Table 7",
        "tablePath": "tables/Table7_IdentificationRobustness.xlsx",
        "tableCaption": "Identification Robustness: M1 Match Coefficient across Six Checks.",
        "tableAim": "Verify that results are robust to alternative outcomes, sample restrictions, and the t-2 lag identification check.",
        "tableExplanation": "Rows = 7 topics × 6 checks (R0 baseline, R1 t-2 lag, R2 ΔlogMC, R3 excl. 2020, R4 balanced panel, R5 win. ΔlogMC). Columns = β, SE, outcome, N obs, N co., R².",
        "tableSuitableSection": "Results — Identification",
    },
    "Table8": {
        "tableId": "Table 8",
        "tablePath": "tables/Table8_SectorHet_M1.xlsx",
        "tableCaption": "Sector Heterogeneity: M1 Coverage Volume by GICS Sector.",
        "tableAim": "Identify sector-level variation in the Business disclosure premium.",
        "tableExplanation": "Rows = 7 topics × 11 sectors. Columns = β match, SE, Within R², N obs, N companies.",
        "tableSuitableSection": "Results — Sector Heterogeneity",
    },
    "Table9": {
        "tableId": "Table 9",
        "tablePath": "tables/Table9_SectorHet_M3.xlsx",
        "tableCaption": "Sector Heterogeneity: M3 Positive and Negative Mean by GICS Sector.",
        "tableAim": "Identify which sectors show bilateral signalling vs. risk-penalty patterns in Business disclosure.",
        "tableExplanation": "Rows = 7 topics × 11 sectors. Columns = β pos_mean, SE, β neg_mean, SE, Within R², N obs, N companies.",
        "tableSuitableSection": "Results — Sector Heterogeneity",
    },
}

analysis_levels = [
    {
        "levelName": "Data Preparation",
        "levelAim": "Merge NLP scores, market cap, GICS, fundamentals, and IBES into a single analysis panel.",
        "inputs": ["match_scores.csv", "tendency_scores.csv", "sentiment_scores.csv",
                   "Market_cap_annual.xlsx", "annual_financial_raw.xlsx",
                   "IBES_monthly_raw.xlsx", "MSCI_category_Japan_listed_companies.xlsx",
                   "Japan_interest_rate_annual.xlsx"],
        "modelOrMethod": "Merge on (stock_code, year); lag ESG and control variables by 1 year and 2 years; compute log MC and ΔlogMC.",
        "outputs": ["data/processed/panel.parquet (37,272 obs × 328 cols)"],
        "interpretation": "Panel ready for TWFE estimation. t-2 lag columns computed in notebook for identification robustness.",
    },
    {
        "levelName": "Part I — Baseline Association (M1/M2/M3)",
        "levelAim": "Establish that Business ESG disclosure associates with market cap and characterise the operative channel among coverage volume, overall tone, and directional intensity.",
        "inputs": ["panel.parquet", "ENV_TOPICS (7 Business topics)"],
        "modelOrMethod": "Two-way FE panel OLS (Frisch-Waugh within-company demeaning + C(year) dummies). SE clustered by company. 21 models: 7 topics × 3 specs (M1, M2, M3). ESG var lagged t-1.",
        "outputs": ["Table 2 (cross-topic summary)", "Tables S1–S7 (per-topic)", "Figure 5 (coefficient plots)"],
        "interpretation": "Significant M1 β: coverage volume associates with MC. M3 β_pos > 0, β_neg sign reveals signalling vs. penalty mechanism.",
    },
    {
        "levelName": "Part II — Mechanism Test 1: Information Channel",
        "levelAim": "Determine whether Business disclosure predicts MC beyond fundamental quality (ROA and leverage).",
        "inputs": ["panel.parquet", "roa_pct_lag1", "ltdebt_assets_pct_lag1"],
        "modelOrMethod": "TWFE M1 on identical sample with and without ROA + LT leverage controls. Attenuation % = (1 − |β_ctrl/β_base|) × 100.",
        "outputs": ["Table 3 (attenuation summary)", "Figure 6 (before/after forest plot)"],
        "interpretation": "Attenuation < 30%: information channel dominates firm quality. Attenuation > 70%: association primarily reflects firm quality.",
    },
    {
        "levelName": "Part II — Mechanism Test 2: Analyst Attention Channel",
        "levelAim": "Test whether Business disclosure transmits to market cap through improved analyst consensus (mediation).",
        "inputs": ["panel.parquet", "ibes_rec_mean", "ibes_rec_mean_lag1"],
        "modelOrMethod": "Baron-Kenny three-stage TWFE mediation. Stage 1: match(t-1) → IBES rec(t). Stage 2a: match(t-1) → logMC, IBES sample. Stage 2b: match(t-1) + IBES rec(t-1) → logMC.",
        "outputs": ["Table 4 (mediation summary)", "Figure 7 (S2a vs S2b forest plot)"],
        "interpretation": "Coefficient attenuation from S2a to S2b quantifies the analyst-channel proportion. Significant Stage 1 + attenuation confirms analyst transmission.",
    },
    {
        "levelName": "Part II — Mechanism Test 3: Signal Direction (M3 Asymmetry)",
        "levelAim": "Classify whether positive and risk-acknowledging Business disclosure carry the same or opposite valuation effects.",
        "inputs": ["panel.parquet", "tend_{topic}_pos_mean_lag1", "tend_{topic}_neg_mean_lag1"],
        "modelOrMethod": "TWFE M3 model. Compare sign of β_pos_mean and β_neg_mean per topic. Classify: bilateral signalling (both > 0), risk penalty (pos > 0, neg < 0), transparency dominates (neg > 0 > pos), ambiguous.",
        "outputs": ["Table 5 (pattern classification)", "Figure 8 (grouped bar chart)"],
        "interpretation": "Bilateral signalling: market rewards both growth signals and risk transparency. Risk penalty: market treats negative Business content as bad news.",
    },
    {
        "levelName": "Part II — Mechanism Test 4: ROA Quartile Heterogeneity",
        "levelAim": "Distinguish signalling theory from complementarity by testing whether the disclosure premium varies monotonically with ROA.",
        "inputs": ["panel.parquet", "roa_pct (within-year quartile split)"],
        "modelOrMethod": "TWFE M1 estimated separately per within-year ROA quartile per topic.",
        "outputs": ["Table 6 (quartile results)", "Figure 9 (quartile forest plot)"],
        "interpretation": "Q1 (low ROA) > Q4 (high ROA): signalling theory. Q4 > Q1: complementarity.",
    },
    {
        "levelName": "Part III — Identification Robustness",
        "levelAim": "Verify that the baseline association is not driven by reverse causality, COVID shock, sample selection, or outliers.",
        "inputs": ["panel.parquet", "match_{topic}_lag1", "match_{topic}_lag2"],
        "modelOrMethod": "Six checks: R0 baseline (t-1), R1 t-2 lag, R2 ΔlogMC outcome, R3 exclude 2020, R4 balanced panel, R5 winsorised ΔlogMC.",
        "outputs": ["Table 7 (robustness summary)", "Figure 10 (robustness forest plot)"],
        "interpretation": "Consistent sign and approximate magnitude across R0–R5 (especially R1 t-2 lag) strengthens identification. R1 survival is the sharpest test against reverse causality.",
    },
    {
        "levelName": "Part IV — Sector Heterogeneity",
        "levelAim": "Identify in which GICS sectors Business disclosure commands the largest valuation premium.",
        "inputs": ["panel.parquet", "sector (11 GICS sectors)"],
        "modelOrMethod": "TWFE M1 and M3 estimated separately within each sector for each of 7 Business topics (11 × 7 × 2 = 154 models).",
        "outputs": ["Table 8 (M1 sector)", "Table 9 (M3 sector)", "Figure 11 (M1 sector forest plot)"],
        "interpretation": "Sectors with high Business disclosure materiality (IT, Consumer Discretionary, Industrials) should show larger β. Financials: financial_metrics topic likely prominent.",
    },
]

actionbrief = {
    "doc_type": "actionbrief",
    "version": "2.0.2",
    "datasetDictionary": dataset_dict,
    "variableDictionary": variable_dict,
    "tableDictionary": table_dict,
    "figureDictionary": figure_dict,
    "analysisStructureBrief": {
        "overview": (
            "Mechanism study examining how Business-related ESG disclosure affects market "
            "capitalisation among Japanese listed companies (37,272 obs, 2016–2025). "
            "Indicator construction follows ESG05e/f/g: M1 (relatedness/coverage volume), "
            "M2 (relatedness + overall document sentiment), M3 (positive mean + negative mean "
            "topic-specific intensity). Four mechanism tests: (1) information channel beyond "
            "fundamental quality, (2) analyst attention mediation via IBES, (3) M3 signal "
            "direction asymmetry, (4) ROA quartile heterogeneity. Identification robustness "
            "includes t-2 lag as the primary reverse-causality check. NLP source: securities "
            "reports (有価証券報告書) from 金融庁 EDINET. Market cap and fundamentals from LSEG."
        ),
        "levels": analysis_levels,
    },
}

with open(EXPORT / "actionbrief.yaml", "w", encoding="utf-8") as f:
    yaml.dump(actionbrief, f, allow_unicode=True, sort_keys=False, default_flow_style=False,
              width=120)
print("  actionbrief.yaml done.")


# ══════════════════════════════════════════════════════════════════════════════
# CODE / NBS / AnaSOP
# ══════════════════════════════════════════════════════════════════════════════
print("Exporting code and nbs...")

# run_analysis.py — minimal reproducible script
run_analysis = """\
\"\"\"
run_analysis.py
Minimal reproducible script — regenerates all export/ artifacts.

Run from project root:
    python export/code/run_analysis.py
\"\"\"
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for script in ["src/01_data_cleaning.py", "src/90_export.py"]:
    print(f"Running {script}...")
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=str(ROOT))
print("Done. All export/ artifacts regenerated.")
"""
(CODE / "run_analysis.py").write_text(run_analysis, encoding="utf-8")

# Copy all src/*.py to export/code/  (law_export_folder: source_sync)
for src in sorted((ROOT / "src").glob("*.py")):
    shutil.copy2(src, CODE / src.name)

# Also mirror to nbs/ for backward compatibility
for src in sorted((ROOT / "src").glob("*.py")):
    shutil.copy2(src, NBS / src.name)

# Copy AnaSOP.md to export root
sop_src = ROOT / "docs" / "AnaSOP.md"
if sop_src.exists():
    shutil.copy2(sop_src, EXPORT / "AnaSOP.md")
    print("  AnaSOP.md copied.")

print("  Code/nbs done.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
figs_found   = sorted(FIGS.glob("fig*.png"))
tables_found = sorted(TABLES.glob("Table*.xlsx")) + sorted(TABLES.glob("S*.xlsx"))
print(f"\n{'='*55}")
print(f"Export complete → {EXPORT}")
code_found = sorted(CODE.glob("*.py"))
print(f"  Figures  : {len(figs_found):2d}  {[f.name for f in figs_found]}")
print(f"  Tables   : {len(tables_found):2d}  {[t.name for t in tables_found]}")
print(f"  Code     : {[f.name for f in code_found]}")
print(f"  Metadata : {[f.name for f in sorted(META.iterdir())]}")
print(f"  AnaSOP   : {(EXPORT / 'AnaSOP.md').exists()}")
print(f"  actionbrief: {(EXPORT / 'actionbrief.yaml').exists()}")

# law validation: verify_src_py_files_copied_to_code
src_py = {f.name for f in (ROOT / "src").glob("*.py")}
code_py = {f.name for f in CODE.glob("*.py")} - {"run_analysis.py"}
missing = src_py - code_py
if missing:
    print(f"  [WARN] verify_src_py_files_copied_to_code: missing {missing}")
else:
    print(f"  [OK]   verify_src_py_files_copied_to_code")
print(f"{'='*55}")
