
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

This project asks: **through what channel does Business-related ESG disclosure affect market capitalisation among Japanese listed companies?**

Unlike ESG05e (Environmental) and ESG05f/g (Governance), this is a **mechanism study**, not an application study. The central objective is not merely to document an association but to identify **which information channel** drives the valuation effect of Business disclosure and to **rule out competing explanations**.

The analysis covers all seven **Business/Strategy (B) topics**:
- 競争⼒ (Competitiveness)
- 他のステークホルダーとの共創 (Stakeholder Co-creation)
- 事業ポートフォリオ (Business Portfolio)
- 知的資本 (Intellectual Capital)
- デジタル (Digital Transformation)
- 財務指標 (Financial Metrics)
- 企業価値 (Corporate Value)

For each topic, corporate disclosure is decomposed into three specifications — coverage volume (M1), coverage with overall document tone (M2), and directional intensity of topic-related content (M3) — using the same fragment-level NLP scoring pipeline as ESG05e/f/g. Each specification is entered as the sole ESG regressor set in a separate TWFE model.

The four mechanism questions are:

1. **Information channel**: Does the association survive after controlling for firm fundamental quality (ROA, leverage)? If so, disclosure carries information *beyond* firm quality — consistent with an information asymmetry channel.
2. **Analyst attention channel**: Does Business disclosure improve analyst consensus forecasts, which in turn explains market cap? A mediation analysis tests whether the disclosure effect is partly transmitted through analyst attention.
3. **Signal direction**: Among topic-related content, do positive-intensity and risk-intensity disclosure carry the same or opposite valuation effects? This tests whether the market treats Business risk acknowledgment as a credibility signal (transparency premium) or as bad news (risk penalty).
4. **Information environment heterogeneity**: Does the disclosure premium vary by firm fundamental quality (ROA quartile)? Under signalling theory, the premium should be larger for weaker-fundamental firms where disclosure is more informative. Under complementarity, it should be larger for stronger firms whose disclosures are more credible.

---

# 2. Theoretical Background

## 2.1 Business disclosure and information asymmetry

Under voluntary disclosure theory (Verrecchia 1983; Dye 1985), firms disclose when expected benefits exceed disclosure costs. For Business/Strategy topics — competitiveness, digital transformation, intellectual capital, corporate value — disclosure signals management's assessment of future competitive position and cash flow potential. Because Business topics are forward-looking and inherently difficult for outsiders to verify, information asymmetry between managers and investors is particularly severe. Substantive Business disclosure reduces this asymmetry, lowering the required rate of return and raising valuation.

The key question is whether the association reflects the disclosure *per se* (information transmission) or merely the underlying firm quality (better firms both disclose more and are worth more). Controlling for ROA and leverage in the regression is the direct test of this distinction.

## 2.2 The three disclosure channels (M1, M2, M3)

The M1–M3 decomposition follows the ESG05e/f/g framework and maps to three distinct theoretical mechanisms:

- **M1 (coverage volume)**: Does the company talk about this Business topic at all? A pure quantity signal — easy to inflate without substance.
- **M2 (coverage + tone)**: Does the overall document positivity explain additional variance beyond coverage? Separates content quantity from general "good news" framing.
- **M3 (positive intensity + risk intensity)**: Among topic-related content, do positive signals (growth opportunities) and risk acknowledgments (strategic challenges) carry separable valuation effects? This is the primary research contribution.

If only M1 matters: the market rewards topic coverage volume regardless of quality.  
If M3 > M1: the market is sophisticated enough to distinguish substantive engagement from keyword density.  
If β_neg in M3 is positive (transparency premium): risk acknowledgment in Business discourse raises credibility and valuation.  
If β_neg is negative (risk penalty): the market treats negative Business content as bad news.

## 2.3 Analyst attention as a transmission mechanism

Analysts aggregate public information and translate it into earnings forecasts and recommendations. Business disclosure that provides new, verifiable information about strategy, innovation capacity, or competitive position should improve analyst forecast accuracy and raise consensus recommendations. This creates a two-step channel:

```
Business disclosure (t-1) → Analyst recommendation (t) → Market cap (t)
```

If the market cap association operates through analyst attention, the disclosure coefficient should attenuate when analyst recommendation is added as a control (Stage 2b vs. Stage 2a in the mediation test). The degree of attenuation quantifies the analyst-channel proportion.

## 2.4 Signalling theory and firm quality heterogeneity

Spence (1973) and the subsequent signalling literature predict that disclosure signals are most valuable when they convey information not otherwise available. For low-fundamental-quality firms (low ROA), the market faces greater uncertainty about true value — Business disclosure provides a stronger signal update. Under this view, the disclosure premium should be larger for Q1 (low ROA) firms.

Alternatively, under complementarity (Milgrom and Roberts 1990), high-quality firms' disclosures are more credible because they have the underlying reality to back them up. If so, the premium should be concentrated in Q4 (high ROA) firms. The ROA quartile test directly discriminates between these two hypotheses.

## 2.5 Reverse causality and the t-2 lag

A standard concern is that high market cap → more resources → better disclosure capacity, creating a spurious positive association. Two identification strategies address this:

1. **t-1 lag (baseline)**: ESG variable lagged one year. The reverse-causality chain requires market cap at t to be driven by resources acquired at t-1.
2. **t-2 lag (sharper test)**: With a two-year lag, the reverse-causality chain must operate across two periods. If the association holds at t-2, it is harder to attribute to contemporaneous resource effects.

This project uses both, with t-2 as the primary identification robustness check.

## 2.6 Japanese institutional context

Japan's institutional environment makes Business disclosure particularly salient for valuation:

- **TSE Prime Market restructuring (2022)**: Required stricter governance and strategic reporting. Companies with richer Business disclosure may have been better positioned for Prime Market listing or retention, creating a structural break in the disclosure-valuation relationship.
- **ISSB/TCFD integration**: As integrated reporting norms include strategy, business model, and competitive dynamics, Business disclosure quality increasingly enters institutional investor frameworks.
- **Near-zero interest rates (2016–2023)**: Japan's monetary environment compresses bond yields and forces valuation onto equity fundamental expectations. This amplifies the valuation relevance of any disclosure that credibly signals future cash flow potential — especially Business/Strategy content.
- **Stewardship Code engagement**: Institutional investors revised engagement frameworks to include strategy and competitive resilience, not just governance compliance. Business disclosure feeds directly into these engagement assessments.

---

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

Used as a robustness outcome (noisier than the level specification).

---

## 4.2 Disclosure dimension variables (M1, M2, M3)

Three disclosure dimension specifications, constructed identically to ESG05e/f/g.

Let notation be defined per topic T and company-year (i, t):
- `rel_f = raw_match_f × 2 − 1` (centred fragment relatedness ∈ [−1, +1])
- A fragment is **related** if `rel_f > 0`
- `sentiment_f` = fragment-level SAPT sentiment score

| Model | Variable(s) | Formula | Empirical question |
|-------|-------------|---------|-------------------|
| **M1** | `match_T` | `mean_f [ max(rel_f, 0) ]` | Does **coverage volume** associate with market cap? |
| **M2** | `match_T` + `sentiment` | M1 var + `mean_f [ sentiment_f ]` | Does overall document **tone** explain variance beyond coverage? |
| **M3** | `pos_mean_T` + `neg_mean_T` | Conditional means on related × positive/negative fragments | Do **positive signals** and **risk acknowledgment** carry separable valuation effects? |

All disclosure variables are **lagged by one year (t−1)**. A **t-2 lag** version is additionally computed for the identification robustness check (§5.5).

### 4.2.1 Formal variable definitions

**Relatedness** (coverage volume):
$$\text{match}_{T,it} = \frac{1}{F} \sum_{f=1}^{F} \max\!\left(\, rel_{f,T},\ 0 \right)$$

**Overall Sentiment** (document-level tone):
$$\text{sentiment}_{it} = \frac{1}{F} \sum_{f=1}^{F} sent_f$$

**Positive Mean Score** (positive engagement intensity):
$$\text{pos\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f>0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f>0\}\right|}$$

**Negative Mean Score** (risk-acknowledgment intensity):
$$\text{neg\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f<0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f<0\}\right|}$$

> **Sign convention**: pos\_mean > 0 always; neg\_mean < 0 always (when defined). A positive coefficient on neg\_mean means companies with *deeper risk coverage* (more negative neg\_mean) are associated with *lower* market cap — a risk penalty. A negative coefficient means deeper risk acknowledgment is associated with higher valuation — the **transparency premium**.

---

## 4.3 Fundamental control variables

Added to address the information channel vs. firm quality distinction (Mechanism Test 1).

| Variable | Source | Coverage | Role |
|----------|--------|----------|------|
| `roa_pct_lag1` | annual_financial_raw | ~93% | Profitability control — absorbs "good company" effect |
| `ltdebt_assets_pct_lag1` | annual_financial_raw | ~93% | Leverage control — absorbs capital structure effect |
| `rnd_share_pct_lag1` | annual_financial_raw | ~35% | R&D intensity — relevant for intellectual capital topic |
| `ibes_rec_mean_lag1` | IBES_monthly_raw | ~33% | Analyst attention — used in mediation test |

R&D share and IBES are included in the mechanism analysis only (not in the full-sample robustness), because their low coverage (~30% joint overlap after listwise deletion) would cause rank deficiency in the within-demeaned design matrix if required simultaneously.

---

## 4.4 Sector profile standardisation

For the descriptive radar chart, each metric-topic combination is standardised independently using the full-sample distribution. Per topic T and metric m:

$$z_{m,T,s} = \frac{\bar{m}_{T,s} - \mu_{m,T}}{\sigma_{m,T}}$$

where $\mu_{m,T}$ and $\sigma_{m,T}$ are global mean and SD, $\bar{m}_{T,s}$ is the sector mean.

---

# 5. Identification Strategy

## 5.1 Design principle: one-topic, one-dimension, one-model

Each regression contains exactly one Business topic and exactly one disclosure specification (M1, M2, or M3) plus company and year fixed effects. Topics are never entered simultaneously.

Rationale: the seven Business topics are positively correlated. Entering them jointly introduces multicollinearity. Running separately yields clean, interpretable estimates; cross-topic comparison is achieved by tabulating the coefficient matrix from the 21 separate regressions.

## 5.2 TWFE model specification

All baseline models:

```
logMC_{it} = α_i + γ_t + f(ESG_{T,i,t−1}) + ε_{it}
```

| Spec | f(ESG) |
|------|--------|
| M1 | β × match_T_{t−1} |
| M2 | β₁ × match_T_{t−1} + β₂ × sentiment_{t−1} |
| M3 | β₁ × pos_mean_T_{t−1} + β₂ × neg_mean_T_{t−1} |

**Implementation**: within-company entity demeaning + C(year) dummies (Frisch-Waugh-Lovell). Standard errors clustered by company.

**21 primary models**: 7 topics × 3 specifications.

## 5.3 Mechanism Test 1 — Information channel

Extended model adding fundamental controls (same sample as baseline):

```
logMC_{it} = α_i + γ_t + β₁ × match_T_{i,t−1} + β₂ × ROA_{i,t−1} + β₃ × LTDebt_{i,t−1} + ε_{it}
```

**Identification logic**: if β₁ is attenuated to zero after adding controls, the association reflects firm quality rather than information transmission. If β₁ survives, disclosure carries information beyond observable fundamental quality — consistent with an information asymmetry channel.

**Attenuation %** is reported: `(1 − |β₁_controlled| / |β₁_baseline|) × 100`.

## 5.4 Mechanism Test 2 — Analyst attention channel (mediation)

Three-stage Baron-Kenny procedure, IBES-covered sample:

- **Stage 1**: match_T (t−1) → IBES recommendation (t). Tests whether Business disclosure attracts analyst attention.
- **Stage 2a**: match_T (t−1) → logMC (t), IBES sample. Baseline within the IBES subsample.
- **Stage 2b**: match_T (t−1) + IBES rec (t−1) → logMC (t). Tests whether the disclosure coefficient attenuates when analyst recommendation is controlled.

Attenuation in Stage 2b relative to Stage 2a quantifies the analyst-channel proportion.

## 5.5 Mechanism Test 3 — Signal direction (M3 asymmetry)

M3 separates positive and negative Business disclosure intensity. The sign pattern of β_pos and β_neg across the seven topics tests the prevailing mechanism:

| Pattern | β_pos | β_neg | Interpretation |
|---------|-------|-------|---------------|
| Signalling (bilateral) | > 0 | > 0 | Growth signals raise MC; risk acknowledgment raises credibility |
| Risk penalty | > 0 | < 0 | Market rewards positive, penalises negative Business content |
| Risk transparency dominates | < 0 | > 0 | Counter-intuitive; acknowledging challenges is valued more than claiming growth |
| Ambiguous | < 0 | < 0 | Neither channel is operative |

## 5.6 Mechanism Test 4 — Information environment heterogeneity (ROA quartile)

Sample split by within-year ROA quartile. M1 estimated separately per quartile per topic.

Under **signalling theory**: β largest for Q1 (low ROA), where market uncertainty is highest and disclosure is most informative.  
Under **complementarity**: β largest for Q4 (high ROA), where disclosures are most credible.

## 5.7 Identification robustness: t-2 lag

Primary check against reverse causality:

```
logMC_{it} = α_i + γ_t + β × match_T_{i,t−2} + ε_{it}
```

If β remains significant with a two-year lag, the contemporaneous resource-driven disclosure story is substantially weakened. The t-2 lag columns are computed in the analysis notebook as `panel.groupby("stock_code")[match_lag1].shift(1)`.

## 5.8 Standard robustness checks (R2–R5)

Applied to M1:

| Check | Change |
|-------|--------|
| R2 ΔlogMC | First-differenced outcome |
| R3 Excl. 2020 | Remove COVID shock year |
| R4 Balanced panel | Companies observed all 10 years |
| R5 Winsorised ΔlogMC | Winsorise at 1st/99th percentiles |

## 5.9 Interpretation limits

This project provides **stronger identification evidence than a pure application study** through the four mechanism tests and the t-2 lag, but does **not claim causal identification**. Results should be described as:

- **Within-company conditional associations** (company FE absorbs cross-sectional heterogeneity)
- **Consistent with an information asymmetry channel** (if Test 1 survives, Test 2 shows mediation, Test 3 shows the expected sign pattern, Test 4 shows the expected heterogeneity)

Key residual limitations: unobserved time-varying confounders; IBES sample is non-random (larger, more-covered firms); ROA quartile split does not fully control for sector composition.

---

# 6. Main Estimation Framework

## 6.1 Notebook structure

The analysis notebook (`03_analysis.ipynb`) is organised in four parts:

**Part I — Baseline Association**  
Sector radar chart. Per-topic M1/M2/M3 tables and coefficient plots. 21 models total.

**Part II — Mechanism Tests**  
- Test 1: Fundamentals-controlled coefficient comparison (baseline vs. M1+ROA+leverage)
- Test 2: IBES mediation (three-stage Baron-Kenny, forest plot of S2a vs. S2b)
- Test 3: M3 signal direction summary table + bar chart across all 7 topics
- Test 4: ROA quartile heterogeneity forest plot

**Part III — Identification Robustness**  
R1 (t-2 lag) + R2–R5. Forest plot per topic with R1 highlighted.

**Part IV — Sector Heterogeneity**  
M1/M2/M3 by GICS sector for all topics.

## 6.2 Primary estimands

For each topic T ∈ {Competitiveness, Stakeholder Co-creation, Business Portfolio, Intellectual Capital, Digital Transformation, Financial Metrics, Corporate Value}:

1. **M1 β**: within-company association between coverage volume and logMC.
2. **M2 β₁, β₂**: separates topic coverage from overall document positivity.
3. **M3 β₁ (pos), β₂ (neg)**: primary research contribution — sign pattern reveals the operative mechanism.
4. **Attenuation from Test 1**: information channel proportion not explained by firm quality.
5. **Attenuation from Test 2**: analyst-channel proportion.
6. **ROA quartile gradient**: direction of heterogeneity (signalling vs. complementarity).

## 6.3 Cross-topic summary table

| Spec | Variable | Competitiveness | Stakeholder Co-creation | Business Portfolio | Intellectual Capital | Digital Transformation | Financial Metrics | Corporate Value |
|------|----------|----------------|------------------------|-------------------|---------------------|----------------------|------------------|----------------|
| M1 | match | | | | | | | |
| M2 | match | | | | | | | |
| M2 | sentiment | | | | | | | |
| M3 | pos\_mean | | | | | | | |
| M3 | neg\_mean | | | | | | | |

All coefficients from separate per-topic regressions.

---

# 7. Analytical Workflow

## Step 1 — Data Preparation (`01_data_cleaning.py`)

- Merge ESG scores, market cap, GICS, interest rate, **fundamentals** (ROA, leverage, R&D), **IBES** (December month-end consensus)
- Extract 6 sub-score columns for 7 Business topics from tendency_scores.csv
- Lag all ESG and control variables by one year (t−1)
- Output: `data/processed/panel.parquet` (37,272 obs × 328 cols)

## Step 2 — Data Inspection (`02_data_inspection.ipynb`)

- Sample coverage by year; missing data rates for all variable groups
- Market cap distributions by year
- Benchmark ESG score trends
- **Fundamentals trends**: ROA, leverage, R&D/revenue by year (mean ± 1 SD)
- **IBES trends**: recommendation, EPS, price target by year (IBES sample)
- **Control-outcome correlations**: Pearson r for all 7 control variables vs. logMC and ΔlogMC
- Topic-level ESG trends (heatmaps); sector profiles

## Step 3 — Baseline Association (`03_analysis.ipynb` Part I)

- Sector profile radar chart (Relatedness, Positive Mean, Negative Mean, z-scored)
- Per-topic M1/M2/M3 coefficient tables and horizontal CI plots
- Identify which specification shows the strongest signal; note cross-topic patterns

## Step 4 — Mechanism Tests (`03_analysis.ipynb` Part II)

- **Test 1**: Run M1 baseline and M1+fundamentals on the same sample; compute attenuation %; forest plot
- **Test 2**: Run three-stage mediation; report S1 (Stage 1) and S2 attenuation; forest plot
- **Test 3**: Collect M3 β_pos and β_neg for all 7 topics; classify pattern per topic; bar chart
- **Test 4**: Compute within-year ROA quartiles; run M1 per quartile; forest plot

## Step 5 — Identification Robustness (`03_analysis.ipynb` Part III)

- Compute t-2 lag variables; run R1 (t-2 lag) per topic
- Run R2–R5; produce joint forest plot with R1 highlighted
- Interpret: does the coefficient direction and approximate magnitude hold across all checks?

## Step 6 — Sector Heterogeneity (`03_analysis.ipynb` Part IV)

- Run M1/M2/M3 within each GICS sector per topic
- Produce forest plots per specification
- Interpret against Business materiality gradient in §3.4
- Flag Utilities (N=33) and Energy (N=26)

## Step 7 — Interpretation

Interpret findings in light of:
- **Information channel**: if Test 1 attenuation < 30%, the information channel dominates; if > 70%, firm quality explains most of the association
- **Analyst mediation**: if Stage 2b attenuation > 20%, analyst attention is a material transmission channel
- **Signal direction**: topics with bilateral positive β_pos and β_neg (signalling pattern) vs. asymmetric pattern
- **Digital transformation timing**: does the DX disclosure premium strengthen post-2022 (ISSB integration, Prime Market restructuring)?
- **Intellectual capital and R&D**: does the IC topic premium align with sectors where `rnd_share_pct` is high (IT, Health Care)?

---

# 8. Standard Table and Figure Structure

## 8.1 Main cross-topic coefficient table

Rows: M1 match; M2 match, sentiment; M3 pos\_mean, neg\_mean.  
Columns: 7 Business topics.  
All coefficients from separate per-topic regressions. Company FE: Yes; Year FE: Yes.

## 8.2 Mechanism Test 1 summary table

| Topic | β Baseline | SE | β + Controls | SE | Attenuation % | N obs |
|-------|------------|----|--------------|----|---------------|-------|

## 8.3 Mechanism Test 2 summary table (per topic)

| Stage | β | SE | Outcome |
|-------|---|----|---------|
| S1: Match → IBES Rec | | | ibes_rec_mean |
| S2a: Match → logMC (IBES) | | | log_mc |
| S2b: Match → logMC + Rec ctrl | | | log_mc |
| S2b: IBES Rec (t-1) → logMC | | | log_mc |

## 8.4 Mechanism Test 3 signal direction table

| Topic | β pos\_mean | SE | β neg\_mean | SE | Pattern |
|-------|------------|----|-----------|----|---------|

## 8.5 Robustness table (M1, all topics)

| Check | Competitiveness | Stakeholder Co-creation | Business Portfolio | Intellectual Capital | Digital Transformation | Financial Metrics | Corporate Value |
|-------|----------------|------------------------|-------------------|---------------------|----------------------|------------------|----------------|
| R0 Baseline (t-1) | | | | | | | |
| R1 t-2 lag | | | | | | | |
| R2 ΔlogMC | | | | | | | |
| R3 Excl. 2020 | | | | | | | |
| R4 Balanced panel | | | | | | | |
| R5 Win. ΔlogMC | | | | | | | |

## 8.6 Sector heterogeneity tables (M1, M2, M3 — separately)

One table per specification, all topics, with β and SE per sector.

---

# 9. Expected Outputs

**Figures:**
1. Business disclosure trends by year (match + pos\_mean per topic)
2. Positive vs. risk coverage trends (pos\_ratio vs. neg\_ratio per topic)
3. Market cap distributions: logMC and ΔlogMC by year
4. Sector profile radar: 11 GICS sectors × 7 Business topics (Relatedness / Positive Mean / Negative Mean)
5. Baseline coefficient plots: M1–M3 per topic (7 panels)
6. Test 1 forest plot: baseline vs. fundamentals-controlled (7 panels)
7. Test 2 forest plot: S2a vs. S2b attenuation (7 panels)
8. Test 3 bar chart: M3 signal direction across all 7 topics
9. Test 4 forest plot: ROA quartile heterogeneity (7 panels)
10. Identification robustness forest plot: R0–R5 with t-2 lag highlighted (7 panels)
11. Sector heterogeneity forest plots: M1, M3 pos\_mean, M3 neg\_mean (3 × 7 panels)

**Tables:**
1. Descriptive statistics: match, sentiment, pos\_mean, neg\_mean for 7 topics + logMC + controls
2–8. Per-topic coefficient tables (M1–M3, one per topic)
9. Cross-topic summary: M1–M3 primary coefficients, all topics
10. Mechanism Test 1: attenuation summary
11. Mechanism Test 2: mediation summary (per topic)
12. Mechanism Test 3: signal direction summary
13. Identification robustness: M1 across R0–R5

---

# 10. Relationship with Jiazi

The analysis repository provides artifacts to Jiazi through the `export/` interface.

Exported artifacts include:
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
