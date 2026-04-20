
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

This project asks: **which dimension of governance disclosure is associated with market capitalization among Japanese listed companies?**

The analysis covers all seven **Governance (G) topics**:
- コーポレートガバナンス (Corporate Governance)
- セキュリティ (Security / Cybersecurity)
- リスク・コンプライアンス (Risk & Compliance)
- 経営管理 (Management Operations)
- マテリアリティ (Materiality)
- ステークホルダーエンゲージメント (Stakeholder Engagement)
- 企業理念 (Corporate Philosophy)

For each topic, corporate disclosure is decomposed into three specifications — coverage volume (M1), coverage with overall document tone (M2), and directional intensity of topic-related content (M3) — using a fragment-level NLP scoring pipeline. Each specification is entered as the sole ESG regressor set in a separate TWFE model, holding the fixed effects structure constant.

The comparison of coefficients across the three specifications for the same topic reveals **through which channel the market prices governance disclosure**: quantity of coverage, the marginal role of overall document positivity beyond content, or the separable signals of positive engagement and risk acknowledgment.

The analysis is repeated across all seven topics and across **GICS sectors** to assess topic-level and sector-level heterogeneity.

The central research questions are:

1. Does the volume of governance disclosure (how much a company talks about a topic) associate with market capitalization within companies over time?
2. Does adding overall document sentiment to coverage volume change the estimated association, and what does that imply about the role of tone independent of content?
3. Among related governance content, do positive-intensity and negative-intensity (risk) disclosure carry distinct and separable market valuation signals?
4. How do these three channels vary across GICS sectors with differing governance exposure?

The project is designed as an **observational panel study**. Results should be interpreted as **within-company conditional associations**, not causal effects.

---

# 2. Theoretical Background

## 2.1 Voluntary disclosure and governance quality

Under voluntary disclosure theory (Verrecchia 1983; Dye 1985), firms choose to disclose when expected benefits exceed disclosure costs. For governance topics, disclosure signals the quality of internal controls, board oversight, and risk management culture. The key insight is that disclosure **quality** and **quantity** may carry distinct signals:

- **Coverage quantity** (match score, related\_ratio): signals that the company treats the governance topic as relevant. Cheap to inflate through keyword density without substantive engagement.
- **Commitment tone** (tendency score): requires both coverage AND positive sentiment — harder to produce without genuine engagement. A high tendency score implies the company discusses governance topics both frequently and positively.
- **Risk acknowledgment** (neg\_ratio, neg\_mean): signals that the company is transparent about governance challenges and compliance risks. Under voluntary disclosure, risk disclosure that is costly to make falsely may be more credible than positive disclosure alone.

## 2.2 Agency theory and governance disclosure

Jensen and Meckling (1976) and Shleifer and Vishny (1997) establish that agency costs arise from the separation of ownership and control. Governance disclosure reduces information asymmetry between managers and investors, potentially lowering the cost of capital and raising market valuation. Companies that signal strong governance through substantive, detailed disclosure may command a governance premium. The M1–M3 decomposition tests **which signal channel** — coverage, tone, or directional intensity — drives this premium.

## 2.3 Governancewashing and signal credibility

A growing concern in ESG research is that firms engage in "governancewashing" — producing formal but substantively empty disclosures on board structure, compliance, or risk management. If the market cannot distinguish genuine governance commitment from boilerplate, coverage volume and commitment quality should produce similar coefficient estimates. If the market penalises pure volume without quality, then M1 (match) should be weaker than M3 (pos\_mean + neg\_mean). Comparing M1 and M3 coefficients constitutes an implicit test of market sophistication in processing governance disclosure.

## 2.4 Risk transparency and governance credibility

A strand of the disclosure literature argues that voluntarily acknowledging negative information enhances firm credibility (Diamond & Verrecchia 1991; Healy & Palepu 2001). For governance topics, companies that explicitly discuss compliance failures, risk exposures, or control weaknesses may signal more credible governance than companies that only present positive achievements. If so, neg\_mean (M3, risk intensity) should be positively — not negatively — associated with market cap. This hypothesis is directly testable by examining the sign of β₂ in M3.

## 2.5 Materiality and sector heterogeneity

Khan, Serafeim and Yoon (2016) show that ESG information is most value-relevant when it concerns material issues for the company's industry. Governance topics such as security/cybersecurity are most material for IT, Financial, and Communication sectors; risk & compliance is material across all sectors but especially Financials; management operations quality matters most in capital-intensive Industrials and Utilities. The sector heterogeneity analysis tests whether the channel effects identified in the full sample are concentrated in sectors where governance topics are most material.

## 2.6 Japanese institutional context

Japan provides a particularly relevant setting for governance disclosure research:
- **Corporate Governance Code (CGC)**: Introduced by TSE in 2015, revised in 2018 and 2021. Prime Market companies must comply or explain against 73 principles, including independent director ratios, board diversity, audit committee structure, and cross-shareholding reduction. Governance disclosure intensity directly corresponds to CGC compliance depth.
- **J-SOX (Financial Instruments and Exchange Act, 2006)**: Requires management assessment of internal controls over financial reporting, analogous to US SOX Section 404. Risk & compliance disclosure in securities reports must include a management assessment report.
- **Cybersecurity governance**: FSA published cybersecurity guidelines for financial institutions (2018, updated 2022); METI published cyber-risk management guidelines for listed companies. `security` topic disclosure reflects the extent to which companies integrate cybersecurity into board-level governance.
- **Stewardship Code revision** (2017, 2020): Institutional investors are required to engage with portfolio companies on governance quality as a precondition of exercising voting rights.
- **TSE Prime Market requirements** (2022): Prime Market listing criteria include more stringent governance standards — at least one-third of board must be independent directors, cross-shareholding must be disclosed and reduced, and integrated governance reporting aligned with TCFD and ISSB standards is encouraged.
- **Near-zero interest rates**: Japan's monetary environment (rates near zero 2016–2023) makes equity valuation especially sensitive to long-run cash-flow expectations, amplifying the role of governance as a valuation signal (governance quality reduces discount-rate uncertainty).

These institutional pressures create a context where governance disclosure is not merely symbolic — it may directly affect investor perceptions of long-run governance risk, compliance stability, and strategic integrity.

---

# 3. Data Overview

## 3.1 Data sources

| File | Content | Dimensions |
|------|---------|-----------|
| `match_scores.csv` | ESG topic match scores (23 themes) per company-year | 37,272 obs × 26 cols |
| `sentiment_scores.csv` | Mean sentiment of ESG disclosure per company-year | 37,272 obs × 4 cols |
| `tendency_scores.csv` | Tendency scores with 7 sub-scores per theme | 37,272 obs × 164 cols |
| `Market_cap_annual.xlsx` | Annual market cap (JPY) per company | 4,032 companies × 10 years |
| `MSCI_category_Japan_listed_companies.xlsx` | GICS sector/industry classification | 4,032 companies |
| `Japan_interest_rate_annual.xlsx` | Japan annual call rate (%) | 10 years |

NLP scores are derived from Japanese listed companies' **securities reports (有価証券報告書)** using a fragment-level NLP pipeline (SAPT model). Each company-year document is split into text fragments; match and tendency scores are computed at the fragment level and aggregated to company-year.

## 3.2 Panel structure

- Units: Japanese listed companies (stock code as identifier)
- Time: 2016–2025 (annual, 10 periods)
- Panel is **unbalanced**: market cap coverage grows from ~3,042 companies in 2016 to ~3,902 in 2025

## 3.3 Governance topic scope

All seven Governance topics are focal regressors. The remaining 16 topics (E / S / B) are excluded from the main analysis; company and year fixed effects absorb the primary confounders.

| Topic (Japanese) | English code | Theoretical link |
|------------------|-------------|-----------------|
| コーポレートガバナンス | `corp_governance` | Board structure, independent directors, cross-shareholding reduction, CGC compliance |
| セキュリティ | `security` | Cybersecurity governance, IT risk management, FSA/METI cyber guidelines |
| リスク・コンプライアンス | `risk_compliance` | J-SOX internal controls, compliance culture, legal and regulatory risk |
| 経営管理 | `management_ops` | Operational efficiency, management quality, internal processes and controls |
| マテリアリティ | `materiality` | Materiality assessment, integrated reporting alignment, TCFD/ISSB governance |
| ステークホルダーエンゲージメント | `stakeholder_engagement` | Investor relations, shareholder engagement, Stewardship Code response |
| 企業理念 | `corporate_philosophy` | Corporate mission, values governance, long-term purpose and strategy alignment |

## 3.4 Sector classification

| Sector | N companies | Governance materiality |
|--------|-------------|-------------------|
| Industrials | 1,052 | High (management ops, risk compliance, large internal control scope) |
| Consumer Discretionary | 765 | Medium–High (compliance, brand governance) |
| Information Technology | 686 | High (cybersecurity, corp governance, management ops) |
| Communication Services | 300 | High (cybersecurity, regulatory compliance) |
| Consumer Staples | 293 | Medium (compliance, materiality reporting) |
| Materials | 287 | Medium–High (risk compliance, environmental governance link) |
| Health Care | 212 | High (compliance, risk management in regulated industry) |
| Financials | 205 | Very High (governance code strictest application, risk & compliance critical) |
| Real Estate | 173 | Medium (corp governance, management ops) |
| Utilities | 33 — small sample | Medium–High (regulatory compliance, security) |
| Energy | 26 — small sample | Medium–High (risk compliance, materiality) |

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

## 4.2 Disclosure dimension variables

Three disclosure dimension specifications are used, each designed to answer a distinct empirical question.

Let notation be defined per topic T and company-year (i, t):
- `rel_f = raw_match_f × 2 − 1` (raw fragment relatedness to topic T, range [−1, +1])
- A fragment is **related** if `rel_f > 0`
- `sentiment_f` = fragment-level sentiment score (SAPT model output)

| Model | Variable(s) | Formula | Empirical question |
|-------|-------------|---------|-------------------|
| **M1** | `match_T` | `mean_f [ max(rel_f, 0) ]` | Does **coverage volume** (how much a company discusses topic T) associate with market cap? |
| **M2** | `match_T` + `sentiment` | M1 var + `mean_f [ sentiment_f ]` | Does overall document **sentiment tone** explain additional variance beyond coverage? Separates content quantity from general positivity. |
| **M3** | `pos_mean_T` + `neg_mean_T` | `mean_f[rel_f×sent_f | rel_f>0, sent_f>0]` and `mean_f[rel_f×sent_f | rel_f>0, sent_f<0]` | Among the related content, do **positive-intensity** and **risk-intensity** disclosure carry separable market valuation signals? |

**Variable definitions**:

- `match_T` — coverage volume: weighted fraction of document related to topic T. Range [0, 1].
- `sentiment` — document-level overall tone: mean fragment sentiment across all text. Range ~[−0.04, +0.02]. Not topic-specific.
- `pos_mean_T` — positive intensity: mean tendency among fragments that are both related to topic T AND have positive sentiment. Captures depth of positive engagement.
- `neg_mean_T` — risk intensity: mean tendency among fragments that are related AND have negative sentiment. Captures depth of risk/challenge coverage. Typically negative in sign.

**Interpretation logic**:
- M1 isolates the pure volume effect — do companies that write more about the governance topic have higher market cap?
- M2 tests whether market cap correlates with general document positivity beyond topic coverage — if `β_sentiment` is significant while `β_match` is stable, both channels are operative.
- M3 tests whether the *directional quality* of related content matters: does positive governance discourse raise valuation (β_pos > 0) and does risk disclosure lower or raise it (sign of β_neg)?

All disclosure variables are **lagged by one year** (t−1) to reduce reverse causality.

---

### 4.2.1 Formal variable definitions

Let company-year document (i, t) consist of F fragments indexed f = 1, …, F.

**Fragment-level inputs**:

| Symbol | Definition |
|--------|-----------|
| `raw_match_{f,T}` | Raw cosine similarity of fragment f to topic T anchor ∈ [0, 1] |
| `rel_{f,T}` | Centred relatedness: `2 × raw_match_{f,T} − 1` ∈ [−1, +1] |
| `sent_f` | Fragment-level SAPT sentiment score ∈ ℝ (positive = favourable tone) |

A fragment is **topic-related** if `rel_{f,T} > 0`.

---

**Relatedness** (coverage volume for topic T):

$$\text{match}_{T,it} = \frac{1}{F} \sum_{f=1}^{F} \max\!\left(\, rel_{f,T},\ 0 \right)$$

This is the mean of clamped centred relatedness scores across all fragments. It equals zero for fragments with no topic affinity and increases with both the number and intensity of topic-related fragments.

---

**Overall Sentiment** (document-level tone, not topic-specific):

$$\text{sentiment}_{it} = \frac{1}{F} \sum_{f=1}^{F} sent_f$$

The arithmetic mean of fragment-level SAPT scores over the entire document, irrespective of topic relevance. Captures the general positive/negative tone of the securities report.

---

**Positive Mean Score** (positive engagement intensity for topic T):

$$\text{pos\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f>0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f>0\}\right|}$$

The mean product of centred relatedness and positive sentiment, restricted to fragments that are both topic-related and positively toned. Captures the depth of positive governance engagement. Undefined (set to NaN) if no such fragments exist for the company-year.

---

**Negative Mean Score** (risk-acknowledgment intensity for topic T):

$$\text{neg\_mean}_{T,it} = \frac{\displaystyle\sum_{f:\, rel_{f,T}>0,\; sent_f<0} rel_{f,T} \times sent_f}{\displaystyle\left|\{f : rel_{f,T}>0,\ sent_f<0\}\right|}$$

The mean product of centred relatedness and negative sentiment, restricted to topic-related fragments with negative tone. Because `rel_{f,T} > 0` and `sent_f < 0`, this quantity is **negative by construction**. A more negative value indicates deeper risk-related coverage of topic T. Undefined if no such fragments exist.

> **Note on sign conventions**: pos\_mean > 0 always; neg\_mean < 0 always (when defined). In regression tables, a positive coefficient on neg\_mean therefore indicates that companies with *deeper risk coverage* (more negative neg\_mean) are associated with *lower* market cap, i.e., a risk penalty. A negative coefficient indicates the opposite: deeper risk acknowledgment is associated with higher valuation — the **transparency premium**.

---

## 4.3 Control variables (fixed effects)

- **Company fixed effects (α_i)**: absorbs all time-invariant heterogeneity — sector, size class, business model, persistent disclosure practices
- **Year fixed effects (γ_t)**: absorbs common time trends — market cycles, macro valuation level, policy regime shifts, interest-rate environment
- **Japan call rate**: included in one robustness specification (rate control, no year FE) as an explicit macro test; not used as a deflator

---

## 4.4 Sector profile standardisation (Figure 4)

Figure 4 shows a descriptive radar chart comparing GICS sectors on their mean governance disclosure scores. To place all three metrics (Relatedness, Positive Mean Score, Negative Mean Score) and all seven topics on a common, interpretable scale, each metric-topic combination is standardised independently using the **full-sample** distribution.

**Step 1 — Global mean and standard deviation (per topic × metric cell)**:

For metric $m \in \{\text{match},\ \text{pos\_mean},\ \text{neg\_mean}\}$ and topic $T$:

$$\mu_{m,T} = \frac{1}{N_{m,T}} \sum_{i,t} m_{T,it} \qquad \sigma_{m,T} = \sqrt{\frac{1}{N_{m,T}} \sum_{i,t} \left(m_{T,it} - \mu_{m,T}\right)^2}$$

where the sum runs over all company-year observations with non-missing values of $m_{T,it}$ (global, not sector-specific).

**Step 2 — Sector mean**:

For GICS sector $s$:

$$\bar{m}_{T,s} = \frac{1}{N_{m,T,s}} \sum_{(i,t):\, \text{sector}_i = s} m_{T,it}$$

**Step 3 — z-score**:

$$z_{m,T,s} = \frac{\bar{m}_{T,s} - \mu_{m,T}}{\sigma_{m,T}}$$

**Interpretation**:

| z-score | Meaning |
|---------|---------|
| $z > 0$ | Sector mean is above the global average for this topic-metric |
| $z = 0$ | Sector mean equals the global average (dashed reference circle in figure) |
| $z < 0$ | Sector mean is below the global average |
| $z = \pm 1$ | Sector deviates by one global standard deviation |

Each spoke in the radar chart corresponds to one governance topic; the three overlaid filled areas correspond to the three metrics. Because standardisation is done separately for each topic-metric cell, the z-scores are directly comparable across topics within the same metric, but cross-metric comparisons should be made cautiously (the raw scales of match, pos\_mean, and neg\_mean differ).

---

# 5. Identification Strategy

## 5.1 Design principle: one-topic, one-dimension, one-model

Each regression model contains **exactly one governance topic** and **exactly one of the three disclosure dimension specifications** (M1, M2, M3), plus company and year fixed effects. Topics are never entered simultaneously in the primary analysis.

Rationale:
- The seven governance topics are positively correlated. Entering them jointly would introduce multicollinearity and make individual coefficients difficult to interpret.
- Running topics separately yields clean, interpretable estimates. Cross-topic comparison is achieved by tabulating the 3 × 7 coefficient matrix from the separate regressions.

## 5.2 Model specification (per topic T, per specification M)

All models share the two-way fixed effects structure:

```
logMC_{it} = α_i + γ_t + f(ESG_{T,i,t−1}) + ε_{it}
```

where `f(·)` is:

| Spec | f(ESG) | Question |
|------|--------|---------|
| M1 | β × match_T_{t−1} | Volume effect alone |
| M2 | β₁ × match_T_{t−1} + β₂ × sentiment_{t−1} | Volume + overall tone, separated |
| M3 | β₁ × pos_mean_T_{t−1} + β₂ × neg_mean_T_{t−1} | Positive vs risk intensity, separated |

**TWFE implementation**: within-company entity demeaning + C(year) dummies (Frisch-Waugh-Lovell). Standard errors clustered by company (stock_code).

This yields a **7 topics × 3 specifications = 21 core regression models** for the primary analysis.

## 5.3 Interpretation limits

This project does **not claim causal identification**.

Key limitations:

1. Endogeneity: financially strong companies invest more in governance disclosure.
2. Selection: companies with better governance may differ from weaker-governance firms on unobserved dimensions.
3. Market cap = price × shares; share issuance may contaminate the outcome.
4. Company FE absorbs cross-sectional variation — identification relies on **within-company changes over time**.
5. Lagging reduces but does not eliminate reverse causality.
6. pos_mean and neg_mean (M3) are collinear with overall document sentiment — interpret jointly with M2.

Results should be described as **conditional within-company associations**, not causal effects.

## 5.4 Primary estimands

For each topic T ∈ {Corporate Governance, Security, Risk & Compliance, Management Operations, Materiality, Stakeholder Engagement, Corporate Philosophy}:

1. **M1 estimand**: β — within-company association between coverage volume (match score, t−1) and logMC.
2. **M2 estimands**: β₁ (match), β₂ (sentiment) — separates topic coverage from overall document positivity; tests whether tone adds marginal information beyond content volume.
3. **M3 estimands**: β₁ (pos\_mean), β₂ (neg\_mean) — within related content, separates positive-engagement intensity from risk-coverage intensity. **Key test**: sign and significance of β₂ distinguishes transparency premium (β₂ > 0) from risk penalty (β₂ < 0).

**M1 is the primary cross-topic and sector comparison variable** (single coefficient, directly comparable across topics and sectors). M3 is the primary research contribution (novel decomposition of directional disclosure intensity).

---

# 6. Main Estimation Framework

## 6.1 Full 3 × 7 model grid (primary analysis)

For each topic T and each specification M1–M3, estimate the TWFE model as defined in §5.2.

The primary output is a **coefficient table and coefficient comparison plot** organised as:
- **Rows**: 5 coefficient rows (M1: match; M2: match, sentiment; M3: pos\_mean, neg\_mean)
- **Columns**: 7 governance topics

**Key diagnostic comparisons**:

| Comparison | What it tests |
|-----------|--------------|
| β_match (M1) vs β_match (M2) | Does sentiment absorb part of the volume effect? |
| β_sentiment (M2) | Does overall tone carry independent information? |
| β_pos\_mean vs β_neg\_mean (M3) | Do positive and risk coverage have opposite or similar signs? |
| Sign of β_neg\_mean | Positive → transparency premium; negative → risk penalty |

## 6.2 Sector heterogeneity (M1, M2, and M3)

All three specifications are run within each GICS sector separately for each topic T:

```
logMC_{it} = α_i + γ_t + f(ESG_{T,i,t−1}) + ε_{it}
```

where f(·) is M1, M2, or M3 as defined in §5.2.

Results are **saved separately per specification**:
- **M1 (match)**: single β_s per sector — cleanest for cross-sector volume comparison
- **M2 (match + sentiment)**: β_match_s and β_sentiment_s per sector — separates volume from tone within sector
- **M3 (pos_mean + neg_mean)**: β_pos_s and β_neg_s per sector — directional intensity within sector

Report β_s and 95% CI per sector. Forest plots per specification: sector on y-axis, coefficient on x-axis, separate panel per topic. Flag Utilities (N=33) and Energy (N=26).

**Expected gradient**: β_s largest for Financials, IT, Communication Services (high governance materiality), and Industrials (large management ops and risk compliance scope).

## 6.3 Robustness checks

Applied to M1 (primary) and M3 (pos\_mean only, to avoid table bloat):

1. **ΔlogMC outcome**: first-differenced outcome
2. **Interest rate control (no year FE)**: replace γ_t with Japan call rate
3. **Exclude 2020**: remove COVID shock year
4. **Balanced panel**: companies observed all 10 years
5. **Winsorized ΔlogMC**: winsorize at 1st / 99th percentiles

---

# 7. Analytical Workflow

## Step 1 — Data Preparation

- Reshape Market_cap_annual.xlsx from wide to long format
- Merge ESG scores (match, sentiment, all tendency sub-scores), market cap, GICS on (stock\_code, year)
- Compute ΔlogMC, log\_mc\_excess
- **Extract sub-score columns for 7 governance topics from tendency\_scores.csv**:
  - `{topic}_related_ratio`, `{topic}_pos_ratio`, `{topic}_neg_ratio`
  - `{topic}_related_mean`, `{topic}_pos_mean`, `{topic}_neg_mean`
- Lag all ESG variables by one year

---

## Step 2 — Descriptive Statistics

- Sample coverage by year (N companies with complete data)
- Summary statistics for all 7 disclosure dimensions per topic (mean, SD, min/max)
- Correlation matrix: 7 sub-scores for each topic plus logMC — shows internal structure
- Time trends: mean sub-scores by year per topic; identify post-2021 acceleration (CGC 2021 revision, Prime Market 2022)
- Sector profiles: z-scored mean scores by sector for each topic (see Figure 4 for standardisation definition)

---

## Step 3 — Governance Disclosure Trend Analysis

- Line chart: mean match score and tendency score per topic by year (2016–2025)
- Stacked area or dual-axis: pos\_ratio vs neg\_ratio per topic over time
  - Shows whether positive coverage is growing faster than risk coverage
  - Or whether risk coverage is rising (increased compliance transparency post-mandate)
- Sector heatmap: mean tend score by sector × year for each topic

---

## Step 4 — Primary 3 × 7 Regression Grid

- Estimate all 21 TWFE models (7 topics × 3 specifications M1, M2, M3)
- Produce the **coefficient comparison table** (5 rows × 7 topic columns) with SE and significance stars
- Produce the **coefficient comparison plot**: for each topic, plot all M1–M3 specification coefficients as a horizontal dot-and-CI chart
- Identify: which specification shows the strongest signal? Is it consistent across topics?

---

## Step 5 — Sector Heterogeneity

- Run all three specifications (M1, M2, M3) per sector for each topic (11 sectors × 7 topics × 3 specs = 231 models)
- Results stored and exported **separately per specification** (see §8.2)
- Produce **forest plots** per specification: sector on y-axis, coefficient on x-axis, separate panel per topic
- Flag small sectors; interpret against materiality gradient

---

## Step 6 — Robustness

- Execute checks in §6.3
- Report as a robustness table for M1 per topic (all seven topics combined)

---

## Step 7 — Interpretation

Interpret findings in light of:
- **Signaling vs governancewashing**: is M3 (commitment) stronger than M1 (coverage)?
- **Transparency premium**: is β₂ in M3 (neg channel) positive?
- **Topic heterogeneity**: do the seven topics show different channel patterns? Corp governance and risk compliance (directly mandated by CGC and J-SOX) may show stronger signals than corporate philosophy or materiality.
- **Sector materiality**: CGC 2021 revision and Prime Market launch (2022) — do corp_governance effects strengthen post-2022?
- **Cybersecurity timing**: FSA/METI guidelines escalation post-2020 — do security disclosure effects strengthen post-2021?

---

# 8. Standard Table and Figure Structure

## 8.1 Main cross-topic coefficient table (M1–M3, all topics)

| Spec | Variable | CorpGov β | SE | Security β | SE | RiskComp β | SE | MgmtOps β | SE | Materiality β | SE | StkhldEng β | SE | CorpPhil β | SE |
|------|----------|-----------|----|------------|----|-----------|----|------------|----|--------------|----|------------|----|-----------|----|
| M1 | match | | | | | | | | | | | | | | |
| M2 | match | | | | | | | | | | | | | | |
| M2 | sentiment | | | | | | | | | | | | | | |
| M3 | pos\_mean | | | | | | | | | | | | | | |
| M3 | neg\_mean | | | | | | | | | | | | | | |
| — | Company FE | Yes | | Yes | | Yes | | Yes | | Yes | | Yes | | Yes | |
| — | Year FE | Yes | | Yes | | Yes | | Yes | | Yes | | Yes | | Yes | |
| — | N obs | | | | | | | | | | | | | | |
| — | Within-R² | | | | | | | | | | | | | | |

All coefficients from separate per-topic regressions.

## 8.2 Sector heterogeneity tables (M1, M2, M3 — saved separately)

Three tables, one per specification, each covering all topics (Topic column + Sector rows).

**Table 6 (M1)** and **Table 7 (M2)** share the same structure (single β_match per sector per topic):

| Topic | Sector | coef\_match | se\_match | Within R² | N obs | N co. |
|-------|--------|------------|----------|-----------|-------|-------|
| Corporate Governance | Financials | | | | | |
| ... |

Table 7 (M2) additionally includes `coef_sentiment` and `se_sentiment` columns.

**Table 8 (M3)**: two coefficients per sector (pos\_mean and neg\_mean):

| Topic | Sector | coef\_pos\_mean | se\_pos\_mean | coef\_neg\_mean | se\_neg\_mean | Within R² | N obs | N co. |
|-------|--------|---------------|-------------|---------------|-------------|-----------|-------|-------|
| Corporate Governance | Financials | | | | | | | |
| ... |

⚠ = fewer than 50 companies (appended to Sector name).

## 8.3 Robustness table (M1 match coefficient, all topics)

| Robustness check | β Corporate Governance | β Security | β Risk & Compliance | β Management Operations | β Materiality | β Stakeholder Engagement | β Corporate Philosophy |
|-----------------|------------------------|------------|---------------------|------------------------|---------------|--------------------------|------------------------|
| Main M1 | | | | | | | |
| ΔlogMC | | | | | | | |
| Rate ctrl (no YFE) | | | | | | | |
| Excl. 2020 | | | | | | | |
| Balanced panel | | | | | | | |
| Win. ΔlogMC | | | | | | | |

---

# 9. Expected Outputs

Figures:
1. Governance disclosure trends by year (match + pos\_mean per topic, 4-panel)
2. Positive vs risk coverage trends (pos\_ratio vs neg\_ratio per topic, 4-panel)
3. Market cap distributions: logMC and ΔlogMC by year
4. **Sector profile radar chart**: one panel per GICS sector (11 panels); spokes = 7 governance topics; three overlaid filled areas for Relatedness (match), Positive Mean, and Negative Mean. Each metric is standardised **per topic column** as `z = (sector mean − global mean) / global σ`, where global mean and σ are computed over all companies and years for that specific topic-metric column. A dashed circle marks z = 0 (global average). Values annotated at fixed radial positions between 1σ and 2σ. Descriptive snapshot of sector disclosure landscape before regression.
5. Coefficient comparison plot: M1–M3 per topic (7-panel, 5 rows each)
6. Sector heterogeneity forest plot — M1: match by sector (7 topic panels)
7. Sector heterogeneity forest plot — M2: match coef by sector (7 topic panels)
8. Sector heterogeneity forest plot — M3: pos\_mean by sector (3×3, 7 topic panels)
9. Sector heterogeneity forest plot — M3: neg\_mean by sector (3×3, 7 topic panels)
10. Robustness comparison: M1 across checks (7 topic panels)

Tables:
1. Descriptive statistics: match, sentiment, pos\_mean, neg\_mean for 7 topics + logMC
2–8. Coefficient tables: M1–M3 for each of the 7 governance topics (one per topic)
9. Cross-topic summary: M1–M3 primary coefficients, all topics
10. Sector heterogeneity — M1: match by sector, all topics
11. Sector heterogeneity — M2: match + sentiment by sector, all topics
12. Sector heterogeneity — M3: pos\_mean + neg\_mean by sector, all topics
13. Robustness: M1 across checks, all topics

All outputs exported to `export/` directory.

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

This section records evolving research decisions.

- **2026-04-16**: ESG05g initialized from ESG05f. Research question confirmed: Governance ESG disclosure → market cap, with sector heterogeneity. Data covers Japanese listed companies × 2016–2025 from securities reports (有価証券報告書). Primary outcome: logMC with TWFE. Lagged ESG to reduce reverse causality.
- Note: Utilities (N=33) and Energy (N=26) have small sector samples — interpret subsample results with caution.
- Note: Market cap data has growing N over time (unbalanced panel) — check whether entry bias affects results.
- **2026-04-16**: Scope set to Governance pillar — all 7 Governance topics (corp_governance, security, risk_compliance, management_ops, materiality, stakeholder_engagement, corporate_philosophy). Analytical structure (M1/M2/M3, TWFE, sector heterogeneity) carried over intact from ESG05f. Model count: 7×3=21 primary models; sector heterogeneity: 11×7×3=231 models.
