"""
src/03_analysis.py
Generates src/03_analysis.ipynb.

Mechanism study: How does Business-related ESG disclosure affect market cap?
Structure:
  Part I  — Baseline Association (M1 / M2 / M3, ESG05e/f/g style)
  Part II — Mechanism Tests
              Test 1: Information channel (fundamentals-controlled)
              Test 2: Analyst attention channel (IBES mediation)
              Test 3: Signal direction — M3 asymmetry (signalling vs. risk transparency)
              Test 4: Information environment heterogeneity (ROA quartile)
  Part III— Identification Robustness (t-2 lag + standard checks)
  Part IV — Sector Heterogeneity

Run:
    python src/03_analysis.py
"""

import json, uuid, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vardict import ENV_TOPICS, ENV_TOPIC_LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "src" / "03_analysis.ipynb"


def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src}


cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells += [md("""\
# 03 Analysis — Business Disclosure and Market Capitalization: A Mechanism Study

**Research question:** Through what channel does Business-related ESG disclosure
associate with market capitalisation among Japanese listed companies?

**Indicator construction** follows the ESG05e/f/g framework:
- M1 (Relatedness): coverage volume — does the company discuss this Business topic?
- M2 (Relatedness + Overall Sentiment): does overall tone add information beyond volume?
- M3 (Positive Mean + Negative Mean): do growth signals and risk transparency carry
  opposite valuation effects?

**Mechanism structure:**
1. Baseline association (M1/M2/M3)
2. Information channel: does the effect survive fundamental-quality controls?
3. Analyst attention channel: IBES mediation
4. Signal direction: M3 asymmetry test
5. Information environment heterogeneity: ROA quartile split
6. Identification robustness: t-2 lag, standard checks
7. Sector heterogeneity
""")]

# ── Setup ─────────────────────────────────────────────────────────────────────
cells += [code("""\
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(".")))
from vardict import ENV_TOPICS, ENV_TOPIC_LABELS, SUB_SCORE_SUFFIXES

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
""")]

# ── Load panel + compute t-2 lags ─────────────────────────────────────────────
cells += [code("""\
panel = pd.read_parquet(PROCESSED / "panel.parquet")

# Verify B-pillar sub-score lag columns are present
needed = [f"tend_{t}_{s}_lag1" for t in ENV_TOPICS for s in SUB_SCORE_SUFFIXES]
missing = [c for c in needed if c not in panel.columns]
if missing:
    print(f"WARNING: {len(missing)} sub-score lag columns missing. Re-run 01_data_cleaning.py")
else:
    print(f"All {len(needed)} Business sub-score lag columns present \\u2713")

# Compute t-2 lags (shift lag1 by one more period) — used in identification
# robustness to make reverse-causality argument harder to sustain.
panel = panel.sort_values(["stock_code", "year"])
for topic in ENV_TOPICS:
    for col in [f"match_{topic}_lag1",
                f"tend_{topic}_pos_mean_lag1",
                f"tend_{topic}_neg_mean_lag1"]:
        if col in panel.columns:
            lag2_col = col.replace("_lag1", "_lag2")
            panel[lag2_col] = panel.groupby("stock_code")[col].shift(1)

print(f"Panel: {panel.shape[0]:,} obs | {panel.stock_code.nunique():,} companies")
print(f"Years : {sorted(panel.year.unique())}")
print(f"Sectors: {sorted(panel.dropna(subset=['sector']).sector.unique())}")
""")]

# ── Regression engine ─────────────────────────────────────────────────────────
cells += [md("## Regression Engine"), code("""\
def panel_ols(df, y, x_vars, entity_fe=True, time_fe=True, add_rate=False):
    \"\"\"TWFE via within-company demeaning + C(year) dummies. SE clustered by stock_code.\"\"\"
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


def topic_model_specs(topic):
    \"\"\"Return M1, M2, M3 spec dicts for one Business topic.\"\"\"
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
        {"label": "M3", "name": "Positive Mean Score + Negative Mean Score",
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
        for (col, label, dim) in sp["rows"]:
            c, se, stars = get_coef(res, col)
            coefs[col] = {"coef": c, "se": se, "stars": stars, "label": label, "dim": dim}
        results.append({**sp, "result": res, "coefs": coefs, "n": n, "nc": nc})
    return results


def build_topic_table(results):
    rows = []
    for m in results:
        for col, info in m["coefs"].items():
            c, se, stars = info["coef"], info["se"], info["stars"]
            rows.append({
                "Spec": m["label"],
                "Model name": m["name"],
                "Variable": info["label"],
                "\\u03b2": f"{c:.4f}{stars}" if c is not None else "\\u2014",
                "SE": f"({se:.4f})" if se is not None else "",
                "N obs": f"{m['n']:,}",
                "N co.": f"{m['nc']:,}",
                "Within R\\u00b2": f"{m['result'].rsquared:.3f}",
            })
    return pd.DataFrame(rows).set_index(["Spec", "Variable"])


def plot_topic_coefficients(ax, results, title=""):
    plot_rows = []
    for m in results:
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
    ax.set_xlabel("Coefficient  [95% CI, TWFE, clustered SE]")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=DIM_COLORS["coverage"],  label="Coverage (match)"),
        mpatches.Patch(color=DIM_COLORS["sentiment"], label="Overall Sentiment"),
        mpatches.Patch(color=DIM_COLORS["pos"],       label="Positive Mean Score"),
        mpatches.Patch(color=DIM_COLORS["neg"],       label="Negative Mean Score"),
    ], fontsize=7, loc="lower right")

print("Regression engine ready.")
""")]

# ══════════════════════════════════════════════════════════════════════════════
# PART I: BASELINE ASSOCIATION
# ══════════════════════════════════════════════════════════════════════════════
cells += [md("""\
---
## Part I — Baseline Association (M1 / M2 / M3)

Establishes that Business disclosure associates with market cap.
Indicator construction follows ESG05e/f/g:
- **M1** Relatedness (t−1) → log MC
- **M2** Relatedness + Overall Sentiment (t−1) → log MC
- **M3** Positive Mean Score + Negative Mean Score (t−1) → log MC

All models: company FE + year FE, SE clustered by company.
""")]

# Sector radar
cells += [md("### Sector Profile — Business Disclosure Landscape"), code("""\
def plot_sector_radar(panel, topics, topic_labels, colors):
    sectors  = sorted(panel.dropna(subset=["sector"]).sector.unique())
    n_topics = len(topics)
    angles   = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])
    LAYERS = [
        ("match",    "Relatedness",   "#2980b9", 0.20, [f"match_{t}" for t in topics]),
        ("pos_mean", "Positive Mean", "#27ae60", 0.25, [f"tend_{t}_pos_mean" for t in topics]),
        ("neg_mean", "Neg Mean",      "#e74c3c", 0.25, [f"tend_{t}_neg_mean" for t in topics]),
    ]
    gstats = {}
    for key, lbl, col, alpha, cols in LAYERS:
        gstats[key] = {}
        for c in cols:
            vals = panel[c].dropna().values
            gstats[key][c] = (float(vals.mean()), float(vals.std()))
    records = {}
    for sec in sectors:
        df_s = panel[panel.sector == sec]
        row  = {}
        for key, lbl, col, alpha, cols in LAYERS:
            zvals = []
            for c in cols:
                mu, sd = gstats[key][c]
                zvals.append((float(df_s[c].mean()) - mu) / sd)
            row[key] = zvals
        records[sec] = row
    all_z  = [v for rec in records.values() for vals in rec.values() for v in vals]
    offset = max(0.0, -min(all_z)) + 0.4
    ncols = 3
    nrows = int(np.ceil(len(sectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True),
                             figsize=(15, nrows * 5.0))
    axes_flat = axes.flatten()
    short_labels = []
    for t in topics:
        words = topic_labels[t].split()
        short_labels.append("\\n".join(words) if len(words) > 1 else topic_labels[t])
    r_ann_base = offset + 1.1
    DR = 0.33
    for idx, sec in enumerate(sectors):
        ax  = axes_flat[idx]
        rec = records[sec]
        theta_ref = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ref, [offset] * 300, color="#999", lw=0.8, ls="--", alpha=0.6, zorder=1)
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
                ax.text(angle, r, f"{z:+.2f}", ha="center", va="center",
                        fontsize=5.0, color=col, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                        zorder=6)
        ax.set_xticks(angles)
        ax.set_xticklabels(short_labels, size=7)
        z_ticks = [-1, 0, 1, 2]
        r_ticks = [z + offset for z in z_ticks if z + offset >= 0]
        z_shown = [z for z in z_ticks if z + offset >= 0]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{z:+d}\\u03c3" for z in z_shown], size=6, color="#555")
        ax.set_ylim(0, None)
        ax.set_title(sec, size=9, fontweight="bold", pad=14)
        ax.spines["polar"].set_visible(False)
        ax.grid(color="grey", alpha=0.25, lw=0.5)
    for idx in range(len(sectors), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    legend_handles = [plt.Line2D([0], [0], color=col, lw=2, label=lbl)
                      for _, lbl, col, _, _ in LAYERS]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9, bbox_to_anchor=(0.98, 0.02))
    plt.suptitle("Business Disclosure Profile by GICS Sector\\n"
                 "z-score: (sector mean − global mean) / global \\u03c3  |  dashed = global average",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()

plot_sector_radar(panel, ENV_TOPICS, ENV_TOPIC_LABELS, TOPIC_COLORS)
""")]

# Per-topic M1-M3
for topic in ENV_TOPICS:
    topic_label = ENV_TOPIC_LABELS[topic]
    cells += [md(f"### {topic_label} — M1 / M2 / M3")]
    cells += [code(f"""\
results_{topic} = run_topic_models(panel, "{topic}")
table_{topic}   = build_topic_table(results_{topic})
print("* p<0.10  ** p<0.05  *** p<0.01   Company FE: Yes | Year FE: Yes | SE clustered by company")
display(table_{topic})
""")]
    cells += [code(f"""\
fig, ax = plt.subplots(figsize=(9, 4))
plot_topic_coefficients(ax, results_{topic}, title="{topic_label}")
plt.tight_layout()
plt.show()
""")]

# ══════════════════════════════════════════════════════════════════════════════
# PART II: MECHANISM TESTS
# ══════════════════════════════════════════════════════════════════════════════
cells += [md("""\
---
## Part II — Mechanism Tests

Four complementary tests of the channel through which Business disclosure
associates with market capitalisation.
""")]

# ── Test 1: Information channel (fundamentals-controlled) ─────────────────────
cells += [md("""\
### Test 1 — Information Channel Beyond Fundamental Quality

**Logic:** If the association is purely because better-quality firms disclose more
*and* command higher valuations, adding ROA and leverage controls should eliminate
the disclosure coefficient. If the coefficient survives, disclosure carries
information *beyond* fundamental quality — consistent with an information
asymmetry / signalling channel.

Controls added: ROA (t−1) + LT Debt/Assets (t−1). Both have ~93% coverage so
sample loss is minimal.
""")]

cells += [code("""\
# For each topic: compare M1 baseline vs. M1 + fundamentals controls
fund_ctrl = ["roa_pct_lag1", "ltdebt_assets_pct_lag1"]

info_rows = []
for topic in ENV_TOPICS:
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"
    ctrl_avail = [v for v in fund_ctrl if v in panel.columns]

    # Baseline M1
    df_b = panel.dropna(subset=["log_mc", match_var] + ctrl_avail).copy()
    res_b, n_b, nc_b = panel_ols(df_b, "log_mc", match_var)
    c_b, se_b, st_b = get_coef(res_b, match_var)

    # M1 + fundamentals controls (same sample for comparability)
    res_c, n_c, nc_c = panel_ols(df_b, "log_mc", [match_var] + ctrl_avail)
    c_c, se_c, st_c = get_coef(res_c, match_var)

    attenuation = (1 - abs(c_c) / abs(c_b)) * 100 if c_b and c_c else None
    info_rows.append({
        "Topic":         lbl,
        "\\u03b2 Baseline": f"{c_b:.4f}{st_b}" if c_b is not None else "\\u2014",
        "SE Baseline":   f"({se_b:.4f})"       if se_b is not None else "",
        "\\u03b2 + Controls": f"{c_c:.4f}{st_c}" if c_c is not None else "\\u2014",
        "SE Controls":   f"({se_c:.4f})"       if se_c is not None else "",
        "Attenuation %": f"{attenuation:.1f}%" if attenuation is not None else "",
        "N obs":         f"{n_b:,}",
    })

df_info = pd.DataFrame(info_rows).set_index("Topic")
print("Baseline M1 vs. M1 + ROA & Leverage controls  (same sample)")
print("* p<0.10  ** p<0.05  *** p<0.01")
display(df_info)
""")]

cells += [code("""\
# Forest plot: baseline vs. controlled coefficients side by side
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
ctrl_avail = [v for v in ["roa_pct_lag1", "ltdebt_assets_pct_lag1"] if v in panel.columns]

for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"
    df_b = panel.dropna(subset=["log_mc", match_var] + ctrl_avail).copy()

    res_b, _, _ = panel_ols(df_b, "log_mc", match_var)
    c_b, se_b, _ = get_coef(res_b, match_var)
    res_c, _, _ = panel_ols(df_b, "log_mc", [match_var] + ctrl_avail)
    c_c, se_c, _ = get_coef(res_c, match_var)

    for i, (label, c, se, color) in enumerate([
        ("Baseline M1",         c_b, se_b, "#2980b9"),
        ("M1 + ROA & Leverage", c_c, se_c, "#e67e22"),
    ]):
        if c is not None and se is not None:
            ax.errorbar(c, i, xerr=1.96 * se,
                        fmt="none", ecolor=color + "99", elinewidth=1.5, capsize=4)
            ax.plot(c, i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Baseline", "+ Controls"], fontsize=8)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(lbl, fontsize=9, fontweight="bold")

plt.suptitle(
    "Test 1: Does disclosure predict MC beyond fundamental quality?\\n"
    "If coefficient survives: information channel, not just firm quality.",
    fontsize=10
)
plt.tight_layout()
plt.show()
""")]

# ── Test 2: Analyst attention channel (IBES mediation) ───────────────────────
cells += [md("""\
### Test 2 — Analyst Attention Channel (IBES Mediation)

**Logic:** If Business disclosure reduces analyst uncertainty, it should
(a) raise analyst recommendations (Stage 1), and (b) the market cap premium
should partly disappear when analyst recommendation is controlled (Stage 2b
vs. 2a). A shrinking coefficient in Stage 2b indicates the disclosure effect
is partly *mediated* through analyst attention.

Baron–Kenny procedure in TWFE. Sample restricted to IBES-covered firms.
""")]

cells += [code("""\
for topic in ENV_TOPICS:
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"

    df_s1 = panel.dropna(subset=["ibes_rec_mean", match_var]).copy()
    try:
        r_s1, n_s1, nc_s1 = panel_ols(df_s1, "ibes_rec_mean", match_var)
        c1, se1, st1 = get_coef(r_s1, match_var)
    except Exception:
        c1, se1, st1 = None, None, ""

    df_s2 = panel.dropna(subset=["log_mc", match_var, "ibes_rec_mean_lag1"]).copy()
    try:
        r_s2a, n_s2a, nc_s2a = panel_ols(df_s2, "log_mc", match_var)
        c2a, se2a, st2a = get_coef(r_s2a, match_var)
    except Exception:
        c2a, se2a, st2a = None, None, ""
    try:
        r_s2b, n_s2b, nc_s2b = panel_ols(df_s2, "log_mc",
                                          [match_var, "ibes_rec_mean_lag1"])
        c2b,  se2b,  st2b  = get_coef(r_s2b, match_var)
        c_rec, se_rec, st_rec = get_coef(r_s2b, "ibes_rec_mean_lag1")
    except Exception:
        c2b = se2b = st2b = c_rec = se_rec = st_rec = None

    attn = (1 - abs(c2b) / abs(c2a)) * 100 if c2a and c2b else None
    rows = [
        ("S1: Match \\u2192 IBES Rec",         c1,   se1,   st1,   "ibes_rec_mean"),
        ("S2a: Match \\u2192 log MC (IBES)",   c2a,  se2a,  st2a,  "log_mc"),
        ("S2b: Match \\u2192 log MC + Rec",    c2b,  se2b,  st2b,  "log_mc"),
        ("S2b: IBES Rec (t-1) \\u2192 log MC", c_rec, se_rec, st_rec, "log_mc"),
    ]
    med_rows = []
    for label, c, se, st, outcome in rows:
        med_rows.append({
            "Stage": label,
            "\\u03b2": f"{c:.4f}{st}" if c is not None else "\\u2014",
            "SE": f"({se:.4f})" if se is not None else "",
            "Outcome": outcome,
        })
    df_med = pd.DataFrame(med_rows).set_index("Stage")
    attn_str = f"  |  S2 attenuation: {attn:.1f}%" if attn is not None else ""
    print(f"\\n=== {lbl} \\u2014 IBES Mediation{attn_str} ===")
    display(df_med)
""")]

cells += [code("""\
# Summary forest plot: S2a vs S2b match coefs
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"
    df_s2 = panel.dropna(subset=["log_mc", match_var, "ibes_rec_mean_lag1"]).copy()
    specs = []
    try:
        r_a, _, _ = panel_ols(df_s2, "log_mc", match_var)
        c_a, se_a, st_a = get_coef(r_a, match_var)
        specs.append(("Baseline (IBES sample)", c_a, se_a, "#2980b9"))
    except Exception:
        pass
    try:
        r_b, _, _ = panel_ols(df_s2, "log_mc", [match_var, "ibes_rec_mean_lag1"])
        c_b, se_b, st_b = get_coef(r_b, match_var)
        specs.append(("+ IBES Rec ctrl", c_b, se_b, "#8e44ad"))
    except Exception:
        pass
    for i, (name, c, se, color) in enumerate(specs):
        if c is not None and se is not None:
            ax.errorbar(c, i, xerr=1.96 * se,
                        fmt="none", ecolor=color + "88", elinewidth=1.4, capsize=4)
            ax.plot(c, i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([s[0] for s in specs], fontsize=8)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(lbl, fontsize=9, fontweight="bold")
plt.suptitle(
    "Test 2: Analyst Attention Channel\\n"
    "Coefficient shrinkage after adding IBES recommendation signals analyst mediation.",
    fontsize=10
)
plt.tight_layout()
plt.show()
""")]

# ── Test 3: Signal direction — M3 asymmetry ───────────────────────────────────
cells += [md("""\
### Test 3 — Signal Direction: M3 Asymmetry

**Logic:** Under the **signalling** hypothesis, both positive and negative
Business disclosures should command valuation premiums:
- Positive Mean (β > 0): growth-opportunity signals raise MC
- Negative Mean (β > 0): risk transparency signals management credibility

If negative disclosures are *penalised* (β < 0), the market treats them as bad
news rather than credibility signals. The sign pattern across topics reveals
which mechanism dominates for each Business dimension.
""")]

cells += [code("""\
# Collect M3 coefficients for all topics
m3_summary = []
for topic in ENV_TOPICS:
    lbl   = ENV_TOPIC_LABELS[topic]
    pm    = f"tend_{topic}_pos_mean_lag1"
    nm    = f"tend_{topic}_neg_mean_lag1"
    df_m3 = panel.dropna(subset=["log_mc", pm, nm]).copy()
    try:
        res_m3, n_m3, nc_m3 = panel_ols(df_m3, "log_mc", [pm, nm])
        c_p, se_p, st_p = get_coef(res_m3, pm)
        c_n, se_n, st_n = get_coef(res_m3, nm)
    except Exception:
        c_p = se_p = st_p = c_n = se_n = st_n = None

    # Mechanism interpretation
    if c_p is not None and c_n is not None:
        if c_p > 0 and c_n > 0:
            interp = "Signalling (both +)"
        elif c_p > 0 and c_n < 0:
            interp = "Growth signal / Risk penalty"
        elif c_p < 0 and c_n > 0:
            interp = "Risk transparency dominates"
        else:
            interp = "Ambiguous (both -)"
    else:
        interp = "n/a"

    m3_summary.append({
        "Topic": lbl,
        "\\u03b2 Pos Mean": f"{c_p:.4f}{st_p}" if c_p is not None else "\\u2014",
        "SE Pos": f"({se_p:.4f})"              if se_p is not None else "",
        "\\u03b2 Neg Mean": f"{c_n:.4f}{st_n}" if c_n is not None else "\\u2014",
        "SE Neg": f"({se_n:.4f})"              if se_n is not None else "",
        "Pattern": interp,
        "N obs": f"{n_m3:,}" if c_p is not None else "",
    })

df_m3_sum = pd.DataFrame(m3_summary).set_index("Topic")
print("M3 Signal Direction Summary — Positive Mean vs. Negative Mean")
print("Signalling: both β > 0  |  Risk penalty: β_pos > 0, β_neg < 0")
print("* p<0.10  ** p<0.05  *** p<0.01")
display(df_m3_sum)
""")]

cells += [code("""\
# M3 coefficient comparison plot across all 7 topics
fig, ax = plt.subplots(figsize=(10, 5))
pos_coefs, neg_coefs = [], []
pos_ses,   neg_ses   = [], []
labels = []
topic_cols = list(TOPIC_COLORS.values())

for topic in ENV_TOPICS:
    pm = f"tend_{topic}_pos_mean_lag1"
    nm = f"tend_{topic}_neg_mean_lag1"
    df_m3 = panel.dropna(subset=["log_mc", pm, nm]).copy()
    try:
        res_m3, _, _ = panel_ols(df_m3, "log_mc", [pm, nm])
        c_p, se_p, _ = get_coef(res_m3, pm)
        c_n, se_n, _ = get_coef(res_m3, nm)
    except Exception:
        c_p = se_p = c_n = se_n = None
    pos_coefs.append(c_p)
    pos_ses.append(se_p)
    neg_coefs.append(c_n)
    neg_ses.append(se_n)
    labels.append(ENV_TOPIC_LABELS[topic])

x = np.arange(len(ENV_TOPICS))
w = 0.35
for i, (c, se, color) in enumerate(zip(pos_coefs, pos_ses, topic_cols)):
    if c is not None and se is not None:
        ax.bar(x[i] - w/2, c, width=w, color="#27ae60", alpha=0.75, label="Positive Mean" if i==0 else "")
        ax.errorbar(x[i] - w/2, c, yerr=1.96*se, fmt="none", color="#27ae60", capsize=3, lw=1.2)
for i, (c, se, color) in enumerate(zip(neg_coefs, neg_ses, topic_cols)):
    if c is not None and se is not None:
        ax.bar(x[i] + w/2, c, width=w, color="#e74c3c", alpha=0.75, label="Negative Mean" if i==0 else "")
        ax.errorbar(x[i] + w/2, c, yerr=1.96*se, fmt="none", color="#e74c3c", capsize=3, lw=1.2)
ax.axhline(0, color="#555", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([l.replace(" ", "\\n") for l in labels], fontsize=8)
ax.set_ylabel("Coefficient  [TWFE, clustered SE]")
ax.set_title("Test 3: M3 Signal Direction — Business Disclosure Asymmetry\\n"
             "Both bars above zero → signalling; Neg bar below zero → risk penalty",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
""")]

# ── Test 4: ROA quartile heterogeneity ────────────────────────────────────────
cells += [md("""\
### Test 4 — Information Environment Heterogeneity (ROA Quartile)

**Logic:** Under the **signalling** interpretation, Business disclosure should
matter *more* for lower-quality firms (where the signal is more informative
because the market is less certain about value). Under **complementarity**,
higher-quality firms (higher ROA) would benefit more because their disclosures
are more credible.

Within-year ROA quartile split; M1 estimated separately per quartile.
""")]

cells += [code("""\
panel["roa_q"] = panel.groupby("year")["roa_pct"].transform(
    lambda x: pd.qcut(x, q=4, labels=["Q1 (Low ROA)", "Q2", "Q3", "Q4 (High ROA)"],
                      duplicates="drop")
)
roa_quartiles = ["Q1 (Low ROA)", "Q2", "Q3", "Q4 (High ROA)"]
q_colors = {"Q1 (Low ROA)": "#e74c3c", "Q2": "#e67e22",
            "Q3": "#27ae60", "Q4 (High ROA)": "#2980b9"}

for topic in ENV_TOPICS:
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"
    rows = []
    for q in roa_quartiles:
        df_q = panel[panel.roa_q == q].dropna(subset=["log_mc", match_var]).copy()
        nc_q = df_q.stock_code.nunique()
        try:
            res_q, n_q, _ = panel_ols(df_q, "log_mc", match_var)
            c, se, st = get_coef(res_q, match_var)
        except Exception:
            c, se, st, n_q = None, None, "", 0
        rows.append({"ROA Quartile": q,
                     "\\u03b2": f"{c:.4f}{st}" if c is not None else "\\u2014",
                     "SE": f"({se:.4f})" if se is not None else "",
                     "N obs": f"{n_q:,}", "N co.": f"{nc_q:,}"})
    df_roa = pd.DataFrame(rows).set_index("ROA Quartile")
    print(f"\\n=== {lbl} \\u2014 ROA Quartile M1 ===")
    display(df_roa)
""")]

cells += [code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    lbl       = ENV_TOPIC_LABELS[topic]
    match_var = f"match_{topic}_lag1"
    plot_rows = []
    for q in roa_quartiles:
        df_q = panel[panel.roa_q == q].dropna(subset=["log_mc", match_var]).copy()
        try:
            res_q, _, _ = panel_ols(df_q, "log_mc", match_var)
            c, se, st = get_coef(res_q, match_var)
            if c is not None:
                plot_rows.append({"q": q, "coef": c, "se": se})
        except Exception:
            pass
    for i, row in enumerate(plot_rows):
        color = q_colors[row["q"]]
        ax.errorbar(row["coef"], i, xerr=1.96 * row["se"],
                    fmt="none", ecolor=color + "99", elinewidth=1.4, capsize=4)
        ax.plot(row["coef"], i, "o", color=color, ms=7, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, ls="--")
    ax.set_yticks(range(len(plot_rows)))
    ax.set_yticklabels([r["q"] for r in plot_rows], fontsize=8)
    ax.set_xlabel("M1 Match coef  [95% CI]", fontsize=8)
    ax.set_title(lbl, fontsize=9, fontweight="bold")

legend_patches = [mpatches.Patch(color=q_colors[q], label=q) for q in roa_quartiles]
fig.legend(handles=legend_patches, loc="lower right", fontsize=8, bbox_to_anchor=(0.99, 0.02))
plt.suptitle(
    "Test 4: ROA Quartile Heterogeneity\\n"
    "Signalling: stronger effect for Q1 (low ROA)  |  Complementarity: stronger for Q4",
    fontsize=10
)
plt.tight_layout()
plt.show()
""")]

# ══════════════════════════════════════════════════════════════════════════════
# PART III: IDENTIFICATION ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
cells += [md("""\
---
## Part III — Identification Robustness

Tests whether the baseline association survives identification challenges.

**R1 — t-2 lag:** If the effect holds with a two-year lag, it is harder to
argue that the association reflects contemporaneous reverse causality
(high MC → better disclosure capacity). The causal chain now requires
MC at t to be driven by disclosure at t−2, separated by an additional year.

**R2–R5 — Standard checks:** Alternative outcome (ΔlogMC), exclude 2020 shock,
balanced panel, winsorised ΔlogMC.
""")]

cells += [code("""\
def run_identification_checks(df, topic):
    match_l1 = f"match_{topic}_lag1"
    match_l2 = f"match_{topic}_lag2"
    checks = []

    # Baseline M1 (t-1)
    df_m = df.dropna(subset=["log_mc", match_l1]).copy()
    res, n, nc = panel_ols(df_m, "log_mc", match_l1)
    c, se, st  = get_coef(res, match_l1)
    checks.append({"Check": "R0 Baseline M1 (t-1)", "var": match_l1,
                   "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n:,}", "N co.": f"{nc:,}",
                   "R\\u00b2": f"{res.rsquared:.3f}"})

    # R1: t-2 lag (reverse causality test)
    if match_l2 in df.columns:
        df_r1 = df.dropna(subset=["log_mc", match_l2]).copy()
        try:
            r1, n1, nc1 = panel_ols(df_r1, "log_mc", match_l2)
            c, se, st = get_coef(r1, match_l2)
            checks.append({"Check": "R1 t-2 lag (harder reverse causality)", "var": match_l2,
                           "coef": c, "se": se,
                           "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                           "Outcome": "log_mc", "N obs": f"{n1:,}", "N co.": f"{nc1:,}",
                           "R\\u00b2": f"{r1.rsquared:.3f}"})
        except Exception as e:
            checks.append({"Check": "R1 t-2 lag", "var": match_l2,
                           "coef": None, "se": None,
                           "\\u03b2": f"ERR: {e}", "SE": "",
                           "Outcome": "log_mc", "N obs": "", "N co.": "", "R\\u00b2": ""})

    # R2: \\u0394logMC
    df_r2 = df.dropna(subset=["dlog_mc", match_l1]).copy()
    r2, n2, nc2 = panel_ols(df_r2, "dlog_mc", match_l1)
    c, se, st = get_coef(r2, match_l1)
    checks.append({"Check": "R2 \\u0394logMC outcome", "var": match_l1,
                   "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "dlog_mc", "N obs": f"{n2:,}", "N co.": f"{nc2:,}",
                   "R\\u00b2": f"{r2.rsquared:.3f}"})

    # R3: Exclude 2020
    df_r3 = df_m[df_m.year != 2020].copy()
    r3, n3, nc3 = panel_ols(df_r3, "log_mc", match_l1)
    c, se, st = get_coef(r3, match_l1)
    checks.append({"Check": "R3 Excl. 2020 (COVID)", "var": match_l1,
                   "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n3:,}", "N co.": f"{nc3:,}",
                   "R\\u00b2": f"{r3.rsquared:.3f}"})

    # R4: Balanced panel
    yc    = df_m.groupby("stock_code")["year"].count()
    df_r4 = df_m[df_m.stock_code.isin(yc[yc == df_m.year.nunique()].index)].copy()
    r4, n4, nc4 = panel_ols(df_r4, "log_mc", match_l1)
    c, se, st = get_coef(r4, match_l1)
    checks.append({"Check": "R4 Balanced panel", "var": match_l1,
                   "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n4:,}", "N co.": f"{nc4:,}",
                   "R\\u00b2": f"{r4.rsquared:.3f}"})

    # R5: Winsorised \\u0394logMC
    df_r5 = df.dropna(subset=["dlog_mc", match_l1]).copy()
    p01, p99 = df_r5["dlog_mc"].quantile([0.01, 0.99])
    df_r5["dlog_win"] = df_r5["dlog_mc"].clip(p01, p99)
    r5, n5, nc5 = panel_ols(df_r5, "dlog_win", match_l1)
    c, se, st = get_coef(r5, match_l1)
    checks.append({"Check": "R5 Win. \\u0394logMC (1%/99%)", "var": match_l1,
                   "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "dlog_win", "N obs": f"{n5:,}", "N co.": f"{nc5:,}",
                   "R\\u00b2": f"{r5.rsquared:.3f}"})

    return pd.DataFrame(checks)


robust_results = {t: run_identification_checks(panel, t) for t in ENV_TOPICS}

for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    print(f"\\n=== {lbl} \\u2014 Identification Robustness ===")
    disp = robust_results[topic].set_index("Check")[
        ["\\u03b2", "SE", "Outcome", "N obs", "N co.", "R\\u00b2"]]
    display(disp)
""")]

cells += [code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    df_r = robust_results[topic].dropna(subset=["coef"])
    y    = range(len(df_r))
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_r["coef"].values, list(y),
                xerr=1.96 * df_r["se"].values,
                fmt="none", ecolor="#aaa", elinewidth=1.2, capsize=3)
    for i, (_, row) in enumerate(df_r.iterrows()):
        dot_color = color if "Baseline" in row["Check"] else (
            "#c0392b" if "t-2" in row["Check"] else "#555")
        ax.plot(row["coef"], i, "o", color=dot_color, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
    ax.set_yticks(list(range(len(df_r))))
    ax.set_yticklabels(df_r["Check"].values, fontsize=7.5)
    ax.set_xlabel("Match coef  [95% CI]", fontsize=8)
    ax.set_title(ENV_TOPIC_LABELS[topic], fontsize=9, fontweight="bold")
legend_handles = [
    mpatches.Patch(color=TOPIC_COLORS[ENV_TOPICS[0]], label="Baseline"),
    mpatches.Patch(color="#c0392b", label="t-2 lag"),
    mpatches.Patch(color="#555", label="Other checks"),
]
fig.legend(handles=legend_handles, loc="lower right", fontsize=8, bbox_to_anchor=(0.99, 0.02))
plt.suptitle("Part III: Identification Robustness\\nt-2 lag (red) is the sharpest reverse-causality test",
             fontsize=10)
plt.tight_layout()
plt.show()
""")]

# ══════════════════════════════════════════════════════════════════════════════
# PART IV: SECTOR HETEROGENEITY
# ══════════════════════════════════════════════════════════════════════════════
cells += [md("""\
---
## Part IV — Sector Heterogeneity

Applied research question: in which GICS sectors does Business disclosure carry
the largest valuation premium? Results use M1 / M2 / M3 separately.
""")]

cells += [code("""\
sectors = sorted(panel.dropna(subset=["sector"]).sector.unique())


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
                row[f"\\u03b2_{key}"] = f"{c:.4f}{stars}" if c is not None else "\\u2014"
                row[f"SE_{key}"]   = f"({se:.4f})" if se is not None else ""
            row["R\\u00b2"] = round(res_s.rsquared, 4)
        except Exception:
            for key in var_map:
                row[f"coef_{key}"] = row[f"se_{key}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


sector_results = {
    spec: {t: run_sector_spec(panel, t, spec) for t in ENV_TOPICS}
    for spec in ("M1", "M2", "M3")
}
print("Sector models complete.")
""")]

for spec_label, var_keys in [("M1", ["match"]),
                              ("M2", ["match", "sentiment"]),
                              ("M3", ["pos_mean", "neg_mean"])]:
    cells += [md(f"### Sector Results — {spec_label}")]
    cells += [code(f"""\
for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    df_s = sector_results["{spec_label}"][topic]
    disp_cols = (["Sector"]
                 + [c for k in {var_keys!r} for c in [f"\\u03b2_{{k}}", f"SE_{{k}}"]]
                 + ["R\\u00b2", "N obs", "N co."])
    disp_cols = [c for c in disp_cols if c in df_s.columns]
    print(f"\\n=== {{lbl}} ({spec_label}) ===")
    display(df_s[disp_cols].set_index("Sector"))
""")]

cells += [md("### Sector Forest Plots — M1")]
cells += [code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    df_s = sector_results["M1"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    y = range(len(df_s))
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_s["coef_match"].values, list(y),
                xerr=1.96 * df_s["se_match"].values,
                fmt="o", color=color, ms=5, ecolor="#aaa", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
    ax.set_xlabel("M1 Relatedness  [95% CI]", fontsize=8)
    ax.set_title(ENV_TOPIC_LABELS[topic], fontsize=9, fontweight="bold")
plt.suptitle("Sector Heterogeneity \\u2014 M1 (Coverage Volume)", fontsize=10)
plt.tight_layout()
plt.show()
""")]

cells += [md("### Sector Forest Plots — M3 (Positive & Negative Mean)")]
cells += [code("""\
for var_key, var_label, dot_color in [
    ("pos_mean", "Positive Mean Score", DIM_COLORS["pos"]),
    ("neg_mean", "Negative Mean Score", DIM_COLORS["neg"]),
]:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, topic in zip(axes.flatten(), ENV_TOPICS):
        df_s = (sector_results["M3"][topic]
                .dropna(subset=[f"coef_{var_key}"])
                .sort_values(f"coef_{var_key}"))
        y = range(len(df_s))
        ax.errorbar(df_s[f"coef_{var_key}"].values, list(y),
                    xerr=1.96 * df_s[f"se_{var_key}"].values,
                    fmt="o", color=dot_color, ms=5, ecolor="#aaa", elinewidth=1.2, capsize=3)
        ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
        ax.set_xlabel(f"M3 {var_label} coef  [95% CI]", fontsize=8)
        ax.set_title(ENV_TOPIC_LABELS[topic], fontsize=9, fontweight="bold")
    plt.suptitle(f"Sector Heterogeneity \\u2014 M3 ({var_label})", fontsize=10)
    plt.tight_layout()
    plt.show()
""")]

# ── Write notebook ─────────────────────────────────────────────────────────────
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
