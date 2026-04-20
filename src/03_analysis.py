"""
src/03_analysis.py
Generates src/03_analysis.ipynb.

Implements the 3 × 7 model grid from AnaSOP §5–6:
  - 7 governance topics (corp_governance, security, risk_compliance,
    management_ops, materiality, stakeholder_engagement, corporate_philosophy)
  - 3 disclosure dimension specifications (M1, M2, M3) per topic, run separately
  - Sector heterogeneity using all three specifications (M1, M2, M3)
  - Robustness checks for M1

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
# 03 Analysis — Governance Disclosure Dimensions and Market Capitalization

Implements the 3 × 7 model grid (AnaSOP §5–6).
Each governance topic is analysed separately across 3 disclosure-dimension
specifications (M1, M2, M3). Sector heterogeneity is analysed using all three
specifications, with results stored and displayed separately per specification.
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
    "coverage":  "#2980b9",   # blue   — M1/M2 match
    "sentiment": "#8e44ad",   # purple — M2 sentiment
    "pos":       "#27ae60",   # green  — M3 pos_mean
    "neg":       "#e74c3c",   # red    — M3 neg_mean
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
""")]

# ── Load data ─────────────────────────────────────────────────────────────────
cells += [code("""\
panel = pd.read_parquet(PROCESSED / "panel.parquet")

needed = [f"tend_{t}_{s}_lag1" for t in ENV_TOPICS for s in SUB_SCORE_SUFFIXES]
missing = [c for c in needed if c not in panel.columns]
if missing:
    print(f"WARNING: {len(missing)} sub-score lag columns missing. Re-run 01_data_cleaning.py")
    for c in missing[:5]: print(" ", c)
else:
    print(f"All {len(needed)} env sub-score lag columns present \\u2713")

print(f"Panel: {panel.shape[0]:,} obs | {panel.stock_code.nunique():,} companies")
print(f"Years : {sorted(panel.year.unique())}")
print(f"Sectors: {sorted(panel.dropna(subset=['sector']).sector.unique())}")
""")]

# ── Sector profile radar chart ────────────────────────────────────────────────
cells += [md("""\
## Sector Profile — Governance Disclosure Landscape

Descriptive radar chart per GICS sector. Each metric is standardised as
**(sector mean − global mean) / global σ** so spokes are directly comparable
across Relatedness, Positive Mean, and Negative Mean.
A dashed circle marks z = 0 (global average). Values annotated at each spoke.
""")]

cells += [code("""\
import numpy as np

def plot_sector_radar(panel, topics, topic_labels, colors):
    \"\"\"
    Polar radar chart: one panel per GICS sector.
    Each of the three metrics is z-scored: (sector_mean - global_mean) / global_sd.
    Positive values → above average; negative → below average.
    Values annotated at each data point.
    \"\"\"
    sectors  = sorted(panel.dropna(subset=["sector"]).sector.unique())
    n_topics = len(topics)
    angles   = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])

    LAYERS = [
        ("match",    "Relatedness",  "#2980b9", 0.20,
         [f"match_{t}" for t in topics]),
        ("pos_mean", "Positive Mean","#27ae60", 0.25,
         [f"tend_{t}_pos_mean" for t in topics]),
        ("neg_mean", "Neg Mean",     "#e74c3c", 0.25,
         [f"tend_{t}_neg_mean" for t in topics]),
    ]

    # ── global stats: per-column (each topic × metric has its own mean/std) ────
    # gstats[key][col] = (global_mean, global_std)
    gstats = {}
    for key, lbl, col, alpha, cols in LAYERS:
        gstats[key] = {}
        for c in cols:
            vals = panel[c].dropna().values
            gstats[key][c] = (float(vals.mean()), float(vals.std()))

    # ── z-scored sector means ─────────────────────────────────────────────────
    # z = (sector_mean[topic] - global_mean[topic]) / global_std[topic]
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

    # offset so all radii stay positive (add |min_z| + buffer)
    all_z  = [v for rec in records.values() for vals in rec.values() for v in vals]
    offset = max(0.0, -min(all_z)) + 0.4

    # ── layout ────────────────────────────────────────────────────────────────
    ncols = 3
    nrows = int(np.ceil(len(sectors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=dict(polar=True),
                             figsize=(15, nrows * 5.0))
    axes_flat = axes.flatten()

    short_labels = []
    for t in topics:
        words = topic_labels[t].split()
        short_labels.append("\\n".join(words) if len(words) > 1 else topic_labels[t])

    # annotations sit between 1σ and 2σ (independent of actual data values)
    r_ann_base = offset + 1.1   # innermost label at ~1.1σ
    DR = 0.33                   # gap → outermost label at ~1.1 + 2×0.33 ≈ 1.76σ

    for idx, sec in enumerate(sectors):
        ax  = axes_flat[idx]
        rec = records[sec]

        # reference circle at z = 0
        theta_ref = np.linspace(0, 2 * np.pi, 300)
        ax.plot(theta_ref, [offset] * 300,
                color="#999", lw=0.8, ls="--", alpha=0.6, zorder=1)

        for key, lbl, col, alpha, _ in LAYERS:
            z_vals = rec[key]
            r_vals = [z + offset for z in z_vals]
            r_c    = r_vals + [r_vals[0]]
            ax.plot(angles_c, r_c, color=col, lw=1.5, zorder=3)
            ax.fill(angles_c, r_c, color=col, alpha=alpha, zorder=2)

        # ── one annotation block per spoke: 3 stacked labels at fixed outer r ──
        for i, angle in enumerate(angles):
            for j, (key, lbl, col, alpha, _) in enumerate(LAYERS):
                z = rec[key][i]
                r = r_ann_base + j * DR
                ax.text(
                    angle, r, f"{z:+.2f}",
                    ha="center", va="center",
                    fontsize=5.0, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85),
                    zorder=6,
                )

        ax.set_xticks(angles)
        ax.set_xticklabels(short_labels, size=7)

        # radial ticks labelled as z-scores
        z_ticks = [-1, 0, 1, 2]
        r_ticks = [z + offset for z in z_ticks if z + offset >= 0]
        z_shown = [z for z in z_ticks if z + offset >= 0]
        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{z:+d}σ" for z in z_shown], size=6, color="#555")
        ax.set_ylim(0, None)

        ax.set_title(sec, size=9, fontweight="bold", pad=14)
        ax.spines["polar"].set_visible(False)
        ax.grid(color="grey", alpha=0.25, lw=0.5)

    for idx in range(len(sectors), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=col, lw=2, label=lbl)
        for _, lbl, col, _, _ in LAYERS
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=9,
               bbox_to_anchor=(0.98, 0.02))
    plt.suptitle(
        "Governance Disclosure Profile by GICS Sector\\n"
        "z-score: (sector mean − global mean) / global σ  |  dashed circle = global average (z=0)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


plot_sector_radar(panel, ENV_TOPICS, ENV_TOPIC_LABELS, TOPIC_COLORS)
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
    \"\"\"Return M1, M2, M3 spec dicts for one env topic.\"\"\"
    m  = f"match_{topic}_lag1"
    s  = "sentiment_mean_lag1"
    pm = f"tend_{topic}_pos_mean_lag1"
    nm = f"tend_{topic}_neg_mean_lag1"
    return [
        {"label": "M1", "name": "Relatedness",
         "rows": [(m,  "Relatedness",           "coverage")]},
        {"label": "M2", "name": "Relatedness + Overall Sentiment",
         "rows": [(m,  "Relatedness",           "coverage"),
                  (s,  "Overall Sentiment",     "sentiment")]},
        {"label": "M3", "name": "Positive Mean Score + Negative Mean Score",
         "rows": [(pm, "Positive Mean Score",   "pos"),
                  (nm, "Negative Mean Score",   "neg")]},
    ]


def run_topic_models(df, topic):
    \"\"\"Run M1-M3 for one env topic. Returns list of result dicts.\"\"\"
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
    \"\"\"Horizontal CI plot for M1-M3 (one env topic).\"\"\"
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

# ── Per-topic analysis (M1–M3) ────────────────────────────────────────────────
for topic in ENV_TOPICS:
    topic_label = ENV_TOPIC_LABELS[topic]

    cells += [md(f"""\
## {topic_label} — M1–M3 Specification Comparison

All models: company FE + year FE, SE clustered by company, ESG var lagged t−1.
""")]

    cells += [code(f"""\
results_{topic} = run_topic_models(panel, "{topic}")
table_{topic}   = build_topic_table(results_{topic})
print(f"=== {topic_label} ===")
print("* p<0.10  ** p<0.05  *** p<0.01   Company FE: Yes | Year FE: Yes | SE clustered by company")
display(table_{topic})
""")]

    cells += [code(f"""\
fig, ax = plt.subplots(figsize=(9, 4))
plot_topic_coefficients(ax, results_{topic}, title="{topic_label}")
plt.tight_layout()
plt.show()
""")]

# ── Sector heterogeneity (M1, M2, M3) ────────────────────────────────────────
cells += [md("""\
## Sector Heterogeneity — M1, M2, M3 by GICS Sector

Each specification (M1, M2, M3) is estimated within each GICS sector separately
for all seven governance topics. Results are stored in `sector_results[spec][topic]`.
Energy and Utilities have fewer than 50 companies — interpret with caution.
""")]

cells += [code("""\
sectors = sorted(panel.dropna(subset=["sector"]).sector.unique())


def run_sector_spec(df, topic, spec_label):
    \"\"\"Run one spec (M1, M2, or M3) by sector for one env topic.
    Returns DataFrame with sector rows and coef/se columns for each variable.\"\"\"
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
                row[f"\\u03b2_{key}"]    = f"{c:.4f}{stars}" if c is not None else "\\u2014"
                row[f"SE_{key}"]   = f"({se:.4f})"     if se is not None else ""
            row["R\\u00b2"] = round(res_s.rsquared, 4)
        except Exception as e:
            for key in var_map:
                row[f"coef_{key}"] = None
                row[f"se_{key}"]   = None
        rows.append(row)
    return pd.DataFrame(rows)


# Run all three specs for all three topics
sector_results = {
    spec: {t: run_sector_spec(panel, t, spec) for t in ENV_TOPICS}
    for spec in ("M1", "M2", "M3")
}
print("Sector models complete.")
""")]

# Display sector results per spec
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

# Sector forest plots
cells += [md("### Sector Forest Plots — M1 (Coverage Volume)")]
cells += [code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    df_s = sector_results["M1"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    y    = range(len(df_s))
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

cells += [md("### Sector Forest Plots — M2 (Match coef, controlling sentiment)")]
cells += [code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, topic in zip(axes.flatten(), ENV_TOPICS):
    df_s = sector_results["M2"][topic].dropna(subset=["coef_match"]).sort_values("coef_match")
    y    = range(len(df_s))
    color = TOPIC_COLORS[topic]
    ax.errorbar(df_s["coef_match"].values, list(y),
                xerr=1.96 * df_s["se_match"].values,
                fmt="o", color=color, ms=5, ecolor="#aaa", elinewidth=1.2, capsize=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
    ax.set_xlabel("M2 Relatedness (controlling Overall Sentiment)  [95% CI]", fontsize=8)
    ax.set_title(ENV_TOPIC_LABELS[topic], fontsize=9, fontweight="bold")
plt.suptitle("Sector Heterogeneity \\u2014 M2 (Relatedness + Overall Sentiment)", fontsize=10)
plt.tight_layout()
plt.show()
""")]

cells += [md("### Sector Forest Plots — M3 (pos\\_mean and neg\\_mean)")]
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
                    fmt="o", color=dot_color, ms=5,
                    ecolor="#aaa", elinewidth=1.2, capsize=3)
        ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
        ax.set_yticks(list(y))
        ax.set_yticklabels(df_s["Sector"].values, fontsize=8)
        ax.set_xlabel(f"M3 {var_label} coef  [95% CI]", fontsize=8)
        ax.set_title(f"{ENV_TOPIC_LABELS[topic]} \\u2014 {var_label}",
                     fontsize=9, fontweight="bold")
    plt.suptitle(f"Sector Heterogeneity \\u2014 M3 ({var_label})", fontsize=10)
    plt.tight_layout()
    plt.show()
""")]

# ── Robustness (M1) ───────────────────────────────────────────────────────────
cells += [md("""\
## Robustness Checks — M1 (Match Score)

Applied to all seven governance topics. Six checks per AnaSOP §6.3.
""")]

cells += [code("""\
def run_robustness_m1(df, topic):
    \"\"\"Run M1 (match) robustness checks for one env topic.\"\"\"
    match_var = f"match_{topic}_lag1"
    checks = []

    # Main M1
    df_m = df.dropna(subset=["log_mc", match_var]).copy()
    res, n, nc = panel_ols(df_m, "log_mc", match_var)
    c, se, st  = get_coef(res, match_var)
    checks.append({"Check": "Main M1", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n:,}", "N co.": f"{nc:,}",
                   "R\\u00b2": f"{res.rsquared:.3f}"})

    # R1: \\u0394logMC
    df_r1 = df.dropna(subset=["dlog_mc", match_var]).copy()
    r1, n1, nc1 = panel_ols(df_r1, "dlog_mc", match_var)
    c, se, st = get_coef(r1, match_var)
    checks.append({"Check": "R1 \\u0394logMC", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "dlog_mc", "N obs": f"{n1:,}", "N co.": f"{nc1:,}",
                   "R\\u00b2": f"{r1.rsquared:.3f}"})

    # R2: Rate ctrl (no year FE)
    df_r2 = df.dropna(subset=["log_mc", match_var, "interest_rate"]).copy()
    r2, n2, nc2 = panel_ols(df_r2, "log_mc", match_var, time_fe=False, add_rate=True)
    c, se, st = get_coef(r2, match_var)
    checks.append({"Check": "R2 Rate ctrl (no Yr FE)", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n2:,}", "N co.": f"{nc2:,}",
                   "R\\u00b2": f"{r2.rsquared:.3f}"})

    # R3: Exclude 2020
    df_r3 = df_m[df_m.year != 2020].copy()
    r3, n3, nc3 = panel_ols(df_r3, "log_mc", match_var)
    c, se, st = get_coef(r3, match_var)
    checks.append({"Check": "R3 Excl. 2020", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n3:,}", "N co.": f"{nc3:,}",
                   "R\\u00b2": f"{r3.rsquared:.3f}"})

    # R4: Balanced panel
    yc    = df_m.groupby("stock_code")["year"].count()
    df_r4 = df_m[df_m.stock_code.isin(yc[yc == df_m.year.nunique()].index)].copy()
    r4, n4, nc4 = panel_ols(df_r4, "log_mc", match_var)
    c, se, st = get_coef(r4, match_var)
    checks.append({"Check": "R4 Balanced panel", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "log_mc", "N obs": f"{n4:,}", "N co.": f"{nc4:,}",
                   "R\\u00b2": f"{r4.rsquared:.3f}"})

    # R5: Winsorized \\u0394logMC
    df_r5 = df.dropna(subset=["dlog_mc", match_var]).copy()
    p01, p99 = df_r5["dlog_mc"].quantile([0.01, 0.99])
    df_r5["dlog_win"] = df_r5["dlog_mc"].clip(p01, p99)
    r5, n5, nc5 = panel_ols(df_r5, "dlog_win", match_var)
    c, se, st = get_coef(r5, match_var)
    checks.append({"Check": "R5 Win. \\u0394logMC", "coef": c, "se": se,
                   "\\u03b2": f"{c:.4f}{st}", "SE": f"({se:.4f})",
                   "Outcome": "dlog_win", "N obs": f"{n5:,}", "N co.": f"{nc5:,}",
                   "R\\u00b2": f"{r5.rsquared:.3f}"})

    return pd.DataFrame(checks)


robust_results = {t: run_robustness_m1(panel, t) for t in ENV_TOPICS}

for topic in ENV_TOPICS:
    lbl = ENV_TOPIC_LABELS[topic]
    print(f"\\n=== {lbl} \\u2014 Robustness (M1) ===")
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
        c = color if row["Check"] == "Main M1" else "#555"
        ax.plot(row["coef"], i, "o", color=c, ms=6, zorder=3)
    ax.axvline(0, color="#e74c3c", lw=0.9, linestyle="--")
    ax.set_yticks(list(range(len(df_r))))
    ax.set_yticklabels(df_r["Check"].values, fontsize=8)
    ax.set_xlabel("M1 Relatedness  [95% CI]", fontsize=8)
    ax.set_title(ENV_TOPIC_LABELS[topic], fontsize=9, fontweight="bold")
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
