# ESG05h: Business Disclosure and Market Capitalisation

A conditional-association study of Business/Strategy disclosure and market
capitalisation among Japanese listed companies. The project combines supplied
NLP scores with LSEG financial data and IBES recommendations.

The current manuscript is [ESG05h_8000words_revised.docx](article/show/ESG05h_8000words_revised.docx).
See [revision notes](article/show/ESG05h_revision_notes.md) for the 2026-09-05
statistical corrections, changed findings, and remaining evidence gaps.

## Research Question

How do topic relatedness and sentiment scores relate to market capitalisation
within companies over time, and how sensitive are these associations to financial
controls, analyst recommendations, sample restrictions, and outcome choice?
The comparisons do not identify causal mechanisms or valuation premia.

## Scope

- Universe: Japanese listed companies.
- Period: 2016-2025 annual panel.
- Panel size: 37,272 company-year observations before model-specific deletion.
- Main outcome: log annual market capitalisation.
- Main disclosure pillar: Business/Strategy topics.
- Estimator: two-way fixed effects panel OLS with company fixed effects, year
  fixed effects, and company-clustered standard errors.

The seven focal Business/Strategy topics are:

1. Competitiveness
2. Stakeholder Co-creation
3. Business Portfolio
4. Intellectual Capital
5. Digital Transformation
6. Financial Metrics
7. Corporate Value

## Data

Raw data are stored under `data/raw/` and the cleaned analysis panel is written
to `data/processed/panel.parquet`. The `data/` directory is Git-ignored and is
expected to be managed outside Git.

| Source | Files | Role |
| --- | --- | --- |
| EDINET securities reports + SAPT NLP pipeline | `match_scores.csv`, `sentiment_scores.csv`, `tendency_scores.csv` | Topic relatedness, document sentiment, and positive/negative tendency scores |
| LSEG Workspace | `Market_cap_annual.xlsx` | Annual market capitalisation |
| LSEG Workspace | `annual_financial_raw.xlsx` | ROA, leverage, R&D, revenue, assets, debt, equity |
| LSEG Workspace / IBES | `IBES_monthly_raw.xlsx` | Analyst EPS, revenue, recommendation, price target |
| MSCI/S&P via LSEG | `MSCI_category_Japan_listed_companies.xlsx` | GICS sector and industry classification |
| Japan interest rate data | `Japan_interest_rate_annual.xlsx` | Annual macro control/reference series |

## Variable Design

Each Business topic is estimated separately; these regressions do not identify
independent topic contributions. Lags use actual calendar years, including t−2.

| Specification | Variables | Interpretation |
| --- | --- | --- |
| M1 | `match_{topic}_lag1` | Normalized semantic relatedness |
| M2 | M1 + `sentiment_mean_lag1` | Relatedness and overall sentiment |
| M3 | `tend_{topic}_pos_mean_lag1` + `tend_{topic}_neg_mean_lag1` | Alternative signed-score model without Relatedness |

Company and year effects are absorbed jointly. Company-clustered CR1 standard
errors account for absorbed parameters; significance uses t(G−1) inference.
Within R² includes year effects relative to the company-demeaned outcome.
Plots show explicitly labelled normal-approximation 95% intervals.

## Main Findings

The corrected baseline uses 29,117 observations from 3,631 companies. M1
relatedness is significantly positive for Competitiveness, Stakeholder
Co-creation, Business Portfolio, and Digital Transformation; Corporate Value is
negative. Intellectual Capital and Financial Metrics are not significant in M1.
Overall Sentiment is positive in all M2 models. Fundamentals and analyst
comparisons show sample and specification sensitivity. Only Financial Metrics
is significant in the reported growth specifications. Subgroup results are
exploratory; there are no formal between-group contrasts or multiple-testing
adjustments. Prior studies support TMPT/SAPT validation; documenting the specific upstream measurement configuration remains necessary for replication.

## Reproduction

The project uses Python 3.12 and uv. Initialize the locked environment with:

```bash
uv sync --locked
```

The lockfile includes the analysis dependencies. The corrected pipeline was run
with Python 3.12.13. Validate the shared estimator with:

```bash
uv run python tests/test_panel_methods.py
```

Run from the repository root:

```bash
# Check canonical script paths without changing data
uv run python src/run_analysis.py --dry-run

# Rebuild the panel, figures, tables and content dictionary
uv run python src/run_analysis.py

# Optional: generate analysis notebook skeletons (overwrites their notebooks)
uv run python src/analyses/02_data_inspection.py
uv run python src/analyses/03_analysis.py
```

## Repository Structure

```text
ESG05h/
  src/
    preprocessing/01_data_cleaning.py
    analyses/02_data_inspection.py
    analyses/03_analysis.py
    analyses/90_export.py       # writes directly to data/results and docs
    run_analysis.py             # canonical pipeline entry point
    vardict.py                  # shared variable definitions
    exp/
  data/
    raw/
    processed/
    exp/
    results/
      figures/
      tables/
  docs/
    AnaSOP.md
    actionbrief.yaml            # figure/table paths relative to repository root
    metadata/
    framelog/
    frame-version.md
    stage-control.md
    LAWS/                      # historical schemas; current standards live in skills
  article/
    origin/
    show/
```

MiliFrame 3.2.2 is registered with current stage `mili`. DVC tracks the six
standard folders and the two Frame control documents separately. Its default
remote `storage` is `/Volumes/MikesDataBackup/dvc_remote`.

The old export bundle has been split into canonical code, documentation and
result directories. Duplicate scripts and the duplicate AnaSOP were removed.
AnaSOP and result metadata were aligned with the corrected estimation and interpretation in September 2026. This does not certify a complete Una submission audit.

## Key Artifacts

- [docs/AnaSOP.md](docs/AnaSOP.md): research objective, theory, variable
  construction, estimands, identification strategy, and workflow.
- [docs/actionbrief.yaml](docs/actionbrief.yaml): machine-readable summary
  of datasets, variables, tables, figures, and analysis structure.
- [data/results/tables/Table2_CrossTopic_Summary.xlsx](data/results/tables/Table2_CrossTopic_Summary.xlsx):
  baseline M1/M2/M3 coefficient matrix.
- [data/results/tables/Table3_Mechanism_InfoChannel.xlsx](data/results/tables/Table3_Mechanism_InfoChannel.xlsx):
  same-sample fundamentals comparison.
- [data/results/tables/Table4_Mechanism_IBESMediation.xlsx](data/results/tables/Table4_Mechanism_IBESMediation.xlsx):
  analyst recommendation comparison.
- [data/results/tables/Table5_Mechanism_SignalDirection.xlsx](data/results/tables/Table5_Mechanism_SignalDirection.xlsx):
  M3 signed-score coefficient summary.
- [data/results/tables/Table7_IdentificationRobustness.xlsx](data/results/tables/Table7_IdentificationRobustness.xlsx):
  t-2 lag and standard robustness checks.

## Notes

- IBES covers roughly one third of the panel and is concentrated among larger
  firms with analyst coverage; analyst comparisons should not be read as fully
  representative of the full sample.
- R&D coverage is limited, so R&D is used selectively rather than as a full
  baseline control.
- Utilities and Energy have small sector samples; sector heterogeneity results
  for those groups require caution.

## Dated manuscript build

`src/manuscript/build_8000words_revision.py` uses the archived pre-fix DOCX,
the reviewed September 5 table snapshot, and current figure PNGs. It is a dated
revision script: later model changes require a new review of prose and table
snapshots. Document authoring used the bundled Documents Python runtime with
python-docx; those authoring dependencies are separate from the analysis lockfile.
`src/manuscript/verify_8000words_revision.py` compares the delivered Word tables
and images with current results and checks that bibliography XML is unchanged.
