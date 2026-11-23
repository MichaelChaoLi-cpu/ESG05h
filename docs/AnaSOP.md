
# AnaSOP
Analysis Standard Operating Procedure

This document records the **human-designed analysis procedure** for the project.  
It explains the **research objective, conceptual framework, estimands, and analytical workflow**.

AnaSOP serves two purposes:

1. Guide researchers on how to reproduce and extend the analysis.
2. Provide contextual reasoning for automated research systems such as Jiazi.

This document may evolve as the research progresses.

---

# 1. Research Objective

Estimate conditional associations between Business-topic disclosure measures and market capitalization among Japanese listed companies. The seven topics are studied separately; coefficients do not identify each topic's independent contribution or a causal disclosure premium.

The empirical questions are: (1) how normalized topic-relatedness and overall sentiment associate with market capitalization; (2) how signed directional scores behave under an alternative specification; (3) how these associations change with observed fundamentals and analyst recommendations; and (4) how patterns vary across outcomes, samples, profitability groups and industries.

The 2026-09-05 audit correction replaces the earlier mechanism-identification framing. Channel interpretations are descriptive and exploratory. No causal mediation, formal cross-group difference, or multiple-testing-adjusted finding is claimed by the current outputs.

# 2. Theoretical Background

Disclosure may convey information about prospects, risk and operating conditions, but observed language can also reflect underlying company changes. Normalized semantic alignment differs from absolute disclosure quantity; favorable language need not be accurate or informative.

M1 uses Relatedness. M2 adds document-wide sentiment. M3 replaces these variables with positive and signed negative mean scores, and is not a nested extension controlling for coverage. Negative Mean is negative: a positive coefficient means more negative scores associate with lower market capitalization, holding the other M3 regressor fixed. Coefficient signs alone do not establish transparency or signalling mechanisms.

Fundamentals adjustment measures sensitivity to observed ROA and leverage. IBES recommendations measure analyst assessment, not attention. Different timing between the recommendation outcome and recommendation control prevents interpreting the reported stages as an identified mediation chain. No attenuation threshold determines which mechanism dominates.

ROA and sector comparisons explore heterogeneous associations. Separate significance is not a test of between-group differences. Longer lags may reflect persistence and do not exclude reverse causality. Log market capitalization is an equity-value level, not a valuation multiple; its annual change is not a total-return measure.

# 3. Data Overview

## 3.1 Data sources

| File | Content | Dimensions |
|------|---------|-----------|
| `match_scores.csv` | ESG topic match scores (23 themes) per company-year | 37,272 obs × 26 cols |
| `sentiment_scores.csv` | Mean sentiment of ESG disclosure per company-year | 37,272 obs × 4 cols |
| `tendency_scores.csv` | Tendency scores with 6 sub-scores per theme | 37,272 obs × 164 cols |
| `Market_cap_annual.xlsx` | Annual market cap (JPY) per company | 4,032 companies × 10 years |
| `MSCI_category_Japan_listed_companies.xlsx` | GICS sector/industry classification | 4,032 companies |
| `Japan_interest_rate_annual.xlsx` | Japan annual call rate (%) | 10 years |
| `annual_financial_raw.xlsx` | Annual fundamentals: ROA, leverage, R&D, revenue | 43,977 obs × 11 cols, 3,975 RICs, FY2015–2025 |
| `IBES_monthly_raw.xlsx` | Analyst consensus: EPS, revenue, recommendation, price target | 203,664 obs, 2,396 RICs, monthly 2015/01–2025/12 |

NLP scores are derived from Japanese listed companies' **securities reports (有価証券報告書)** sourced from the **Japan Financial Services Agency (金融庁, FSA)** EDGAR system (EDINET). Each company-year document is split into text fragments; match and tendency scores are computed at the fragment level and aggregated to company-year using a fragment-level NLP pipeline (SAPT model).

**Market cap and all fundamental data** are sourced from **LSEG (London Stock Exchange Group, formerly Refinitiv)**. Specifically:
- `Market_cap_annual.xlsx`: annual market capitalisation in JPY, downloaded from LSEG Workspace
- `annual_financial_raw.xlsx`: fiscal-year financial statement data (ROA, leverage, R&D, revenue, total assets/equity/debt), downloaded from LSEG Workspace using RIC identifiers. RICs follow the `.T` suffix convention for Tokyo Stock Exchange listings.
- `IBES_monthly_raw.xlsx`: IBES analyst consensus (EPS, revenue, recommendation, price target), FY1 horizon, downloaded from LSEG Workspace. Aggregated to annual by taking the **December month-end** snapshot for each RIC-year. All 16,972 RIC-year pairs carry exactly one December row. IBES covers ~33% of panel companies (2,396 of ~4,389), primarily larger firms with active analyst coverage.

## 3.2 Panel structure

- Units: Japanese listed companies (stock code as identifier)
- Time: 2016–2025 (annual, 10 periods)
- Panel is **unbalanced**: coverage grows from ~3,042 companies in 2016 to ~3,902 in 2025
- Total: 37,272 company-year observations

## 3.3 Business topic scope

All seven Business/Strategy topics are focal regressors. The remaining 16 topics (E / S / G) are excluded from the primary analysis; company and year fixed effects absorb the primary confounders.

| Topic (Japanese) | English code | Theoretical link |
|------------------|-------------|-----------------|
| 競争⼒ | `competitiveness` | Competitive moat, pricing power, market position signals |
| 他のステークホルダーとの共創 | `stakeholder_cocreation` | Value creation with partners, ecosystem strategy |
| 事業ポートフォリオ | `business_portfolio` | Portfolio restructuring, capital allocation strategy |
| 知的資本 | `intellectual_capital` | R&D, patents, human capital, intangible value |
| デジタル | `digital_transformation` | Technology investment, DX strategy, digital competitive advantage |
| 財務指標 | `financial_metrics` | Financial discipline, KPI transparency, capital efficiency |
| 企業価値 | `corporate_value` | Value creation narrative, long-term orientation |

## 3.4 Sector classification

11 GICS sectors; Business topic materiality by sector:

| Sector | N companies | Business materiality |
|--------|-------------|---------------------|
| Industrials | 1,052 | High (business portfolio restructuring, competitiveness) |
| Consumer Discretionary | 765 | High (digital transformation, corporate value) |
| Information Technology | 686 | Very High (digital transformation, intellectual capital, competitiveness) |
| Communication Services | 300 | High (digital transformation, intellectual capital) |
| Consumer Staples | 293 | Medium (corporate value, stakeholder co-creation) |
| Materials | 287 | Medium (business portfolio, financial metrics) |
| Health Care | 212 | High (intellectual capital, R&D disclosure) |
| Financials | 205 | High (financial metrics, corporate value, business portfolio) |
| Real Estate | 173 | Medium (corporate value, financial metrics) |
| Utilities | 33 — small sample | Medium (business portfolio, digital transformation) |
| Energy | 26 — small sample | Medium (business portfolio, financial metrics) |

---

# 4. Variable Construction

## 4.1 Outcome variable

**Primary outcome: log market cap level**

```
logMC_{it} = log(MC_{i,t})
```

Year fixed effects absorb the common annual valuation level, making log market cap comparable across years without first-differencing.

**Secondary outcome: annual market cap growth rate**

```
ΔlogMC_{it} = log(MC_{i,t}) − log(MC_{i,t−1})
```

Used as an alternative outcome addressing a different question from levels. Compute against the actual preceding calendar year, never the preceding available row.

---

## 4.2 Disclosure dimension variables (M1, M2, M3)

Three disclosure dimension specifications, constructed identically to ESG05e/f/g.

Let notation be defined per topic T and company-year (i, t):
- `rel_f = raw_match_f × 2 − 1` (centred fragment relatedness ∈ [−1, +1])
- A fragment is **related** if `rel_f > 0`
- `sentiment_f` = fragment-level SAPT sentiment score

| Model | Variable(s) | Formula | Empirical question |
|-------|-------------|---------|-------------------|
| **M1** | `match_T` | `mean_f [ max(rel_f, 0) ]` | Does **normalized semantic relatedness** associate with market cap? |
| **M2** | `match_T` + `sentiment` | M1 var + `mean_f [ sentiment_f ]` | Does overall document **tone** explain variance beyond coverage? |
| **M3** | `pos_mean_T` + `neg_mean_T` | Conditional means on related × positive/negative fragments | How do signed, similarity-weighted directional scores associate with market capitalization? |

All disclosure variables are **lagged by one year (t−1)**. A **t-2 lag** version is additionally computed for the timing-sensitivity comparison.

### 4.2.1 Formal variable definitions

**Relatedness** (normalized semantic relatedness):
$$\text{match}_{T,it} = \frac{1}{F} \sum_{f=1}^{F} \max\!\left(\, rel_{f,T},\ 0 \right)$$

**Overall Sentiment** (document-level tone):
$$\text{sentiment}_{it} = \frac{1}{F} \sum_{f=1}^{F} sent_f$$

**Positive Mean Score** (positive engagement intensity):
$$\text{pos\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f>0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f>0\}\right|}$$

**Negative Mean Score** (risk-acknowledgment intensity):
$$\text{neg\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f<0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f<0\}\right|}$$

> **Sign convention**: pos\_mean > 0 always; neg\_mean < 0 always (when defined). A positive coefficient on neg\_mean means companies with *deeper risk coverage* (more negative neg\_mean) are associated with *lower* market cap; this is an association, not an identified risk mechanism. A negative coefficient means deeper risk acknowledgment is associated with higher valuation; this does not by itself establish a transparency premium.

---

## 4.3 Fundamental control variables

Added to assess sensitivity to observed profitability and leverage.

| Variable | Source | Coverage | Role |
|----------|--------|----------|------|
| `roa_pct_lag1` | annual_financial_raw | ~93% | Profitability control — conditions on observed profitability |
| `ltdebt_assets_pct_lag1` | annual_financial_raw | ~93% | Leverage control — conditions on observed leverage |
| `rnd_share_pct_lag1` | annual_financial_raw | ~35% | R&D intensity — relevant for intellectual capital topic |
| `ibes_rec_mean_lag1` | IBES_monthly_raw | ~33% | Analyst assessment — descriptive recommendation comparison |

R&D is described but not included in the reported fundamentals regressions. IBES recommendations enter a separate restricted-sample comparison. Limited coverage motivates reporting sample selection; it does not by itself imply rank deficiency.

---

## 4.4 Sector profile standardisation

For the descriptive radar chart, each metric-topic combination is standardised independently using the full-sample distribution. Per topic T and metric m:

$$z_{m,T,s} = \frac{\bar{m}_{T,s} - \mu_{m,T}}{\sigma_{m,T}}$$

where $\mu_{m,T}$ and $\sigma_{m,T}$ are global mean and SD, $\bar{m}_{T,s}$ is the sector mean.

---

# 5. Identification Strategy

The design estimates conditional associations using joint company and year fixed effects. Both the outcome/regressors and year indicators are projected off company indicators, giving slopes equivalent to a full company/year dummy regression. The common implementation is `src/panel_methods.py`; numerical tests compare its coefficients, clustered standard errors and p-values against full LSDV on balanced and unbalanced panels.

Company-clustered CR1 covariance includes the absorbed company parameters in residual degrees of freedom. Tests use t(G−1), where G is the company cluster count. Plot intervals use the normal approximation coefficient ±1.96 SE and are labelled accordingly. Within R² denotes fit relative to the company-demeaned outcome, including year effects, not partial fit after removing both effects.

Lagged variables are matched on company and calendar year t−1 or t−2. Missing prior years remain missing. Annual log market-capitalization growth uses actual year t−1 market capitalization, including available external market-capitalization history. Reporting dates are not observed in the score panel, so calendar lags do not create an event-time identification design.

# 6. Main Estimation Framework

M1: logMC(t) on Relatedness(t−1). M2: logMC(t) on Relatedness(t−1) and Overall Sentiment(t−1). M3: logMC(t) on Positive Mean(t−1) and Negative Mean(t−1). All include joint company/year effects and company clustering. Each topic is estimated separately. Raw coefficient magnitudes are not a scale-free ranking of topic importance.

The fundamentals comparison adds lagged ROA and long-term debt/assets on a common complete-case sample. The analyst comparison estimates recommendation(t) on Relatedness(t−1), and separately compares logMC(t) regressions with/without recommendation(t−1) on a common second-stage sample. Absolute-coefficient attenuation is descriptive, not a causal proportion.

ROA quartiles use current within-year ROA; firms may change groups. Industry classifications are static. These analyses are exploratory and do not establish significant coefficient differences between groups. In particular, the 154 M1/M3 sector regressions involve multiple testing and some small company clusters.

R0 is baseline; R1 uses calendar t−2; R2 uses annual ΔlogMC; R3 excludes 2020; R4 restricts companies to all available estimation years; R5 winsorizes growth at pooled estimation-sample 1st/99th percentiles. No causal identification follows from survival of these checks.

# 7. Analytical Workflow

1. `src/preprocessing/01_data_cleaning.py` rebuilds the panel from raw inputs with exact year matching.
2. `tests/test_panel_methods.py` verifies joint fixed effects and calendar-gap handling.
3. `src/analyses/90_export.py` estimates the models and generates 11 figures, 16 tables and content metadata. `src/analyses/03_analysis.py` generates an optional notebook using the same estimator.
4. Compare revised coefficients, samples and significance with the archived pre-audit outputs.
5. Update the manuscript only from regenerated outputs; verify all numerical cells and inspect rendered pages.

The NLP measurement framework has prior published support: the TMPT methods study (Green Technologies and Sustainability 4, 100296) reports Japanese accuracy/precision/recall of 84.18%/82.16%/87.39%; the related governance study (Strategic Business Research 2, 100237) reports SAPT MSE 0.016 and MAE 0.085 on an independent multilingual sentiment dataset. These are prior evaluation results, not new performance estimates for the seven Business topics. TMPT matching scores are learned matcher outputs in [0,1], not raw cosine similarity. The downstream workflow does not reproduce the upstream model; documenting the actual model version, topic anchors, fragment and aggregation settings remains necessary for reproducibility.

# 8. Standard Table and Figure Structure

Main tables: Table 1 descriptive statistics; Table 2 cross-topic M1–M3 summary; Table 3 fundamentals comparison; Table 4 analyst recommendation comparison; Table 5 signed M3 coefficients; Table 6 ROA groups; Table 7 R0–R5; Tables 8–9 sector M1/M3. S1–S7 give per-topic M1–M3 output.

All regression tables require outcome, sample, fixed-effect, clustering, star and interval definitions. Pattern labels in Table 5 report coefficient signs only. Do not regenerate the former causal/signalling classifications.

# 9. Expected Outputs

Figures 1–4 describe relatedness and sentiment trends, positive/negative fragment ratios, market-capitalization distributions and sector score profiles. Figure 5 reports baseline coefficients; Figures 6–7 compare fundamentals and analyst adjustment; Figure 8 shows signed M3 coefficients; Figure 9 reports ROA quartiles; Figure 10 shows specification sensitivity; Figure 11 reports exploratory sector M1 results.

Preserve existing file identifiers to keep figure/table references stable. Figure 4 omits crowded numeric annotations; plotted standardized profiles retain the original definition. Graphical regression intervals are normal-approximation 95% intervals; significance stars follow company-cluster t tests.

# 10. Relationship with Jiazi

The analysis repository provides figures, tables, reproducible scripts and the content dictionary to the manuscript workflow.

Handoff artifacts include:
- Figures (PNG/SVG)
- Regression tables (CSV/LaTeX)
- Reproducible scripts
- actionbrief.yaml

Jiazi uses these artifacts to generate manuscript sections and assemble the research paper.

---

# 11. Human Debug Notes

- **2026-04-16**: ESG05g initialized from ESG05f. Research question confirmed as Governance ESG disclosure → market cap.
- **2026-04-16**: Scope set to Governance pillar (7 G topics). Analytical structure (M1/M2/M3, TWFE, sector heterogeneity) carried over from ESG05f.
- **2026-04-21**: Supplementary data added — `annual_financial_raw.xlsx` (ROA, leverage, R&D, revenue; 43,977 obs, 3,975 RICs, FY2015–2025) and `IBES_monthly_raw.xlsx` (EPS, revenue, recommendation, price target; 203,664 obs, 2,396 RICs, monthly 2015/01–2025/12). Both sourced from **LSEG Workspace**. IBES aggregated to annual via December month-end snapshot.
- **Data provenance confirmed**: NLP source documents (有価証券報告書) from **金融庁 EDINET**. Market cap, fundamentals, and IBES analyst consensus all from **LSEG Workspace** (RIC identifiers, `.T` suffix for TSE listings).
- **2026-04-21**: Research scope revised from Governance (G) pillar to **Business/Strategy (B) pillar** (7 B topics: competitiveness, stakeholder_cocreation, business_portfolio, intellectual_capital, digital_transformation, financial_metrics, corporate_value). vardict.py ENV_TOPICS updated accordingly. Panel rebuilt with B-pillar sub-score columns.
- **2026-04-21**: Research framing revised from **application study** to **mechanism study**. Analysis restructured: baseline association (Part I) + four mechanism tests (Part II) + identification robustness including t-2 lag (Part III) + sector heterogeneity (Part IV). AnaSOP updated to reflect mechanism design.
- Note: Utilities (N=33) and Energy (N=26) have small sector samples — interpret subsample results with caution.
- Note: IBES covers only ~33% of panel companies (larger firms with analyst coverage). IBES mediation results are not representative of the full sample.
- Note: R&D share has ~35% coverage due to limited voluntary disclosure in Japan. Intellectual capital topic analysis may benefit from sector-stratified R&D controls.
