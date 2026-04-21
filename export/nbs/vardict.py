# Variable dictionary for ESG05g
# Maps raw dataset column names to readable English names and metadata.
# Import this module wherever column labels are needed.

# ── ESG theme: raw Japanese name → clean English snake_case name ─────────────
ESG_THEME_EN = {
    # Note: some characters use CJK Radical codepoints (e.g. ⽣ U+2F49, ⼈ U+2F08, ⼒ U+2F15)
    # exactly as they appear in the raw CSV files — do not normalise.
    "脱炭素":                       "decarbonization",
    "サーキュラー":                  "circular_economy",
    "⽣物多様性":                   "biodiversity",
    "健康経営":                     "employee_health",
    "顧客価値":                     "customer_value",
    "⼈権尊重":                     "human_rights",
    "⼈的資本経営":                  "human_capital",
    "DE&I":                         "dei",
    "地域共⽣":                     "community_coexistence",
    "コーポレートガバナンス":        "corp_governance",
    "セキュリティ":                  "security",
    "リスク・コンプライアンス":      "risk_compliance",
    "競争⼒":                       "competitiveness",
    "経営管理":                     "management_ops",
    "他のステークホルダーとの共創":  "stakeholder_cocreation",
    "事業ポートフォリオ":            "business_portfolio",
    "知的資本":                     "intellectual_capital",
    "デジタル":                     "digital_transformation",
    "財務指標":                     "financial_metrics",
    "企業価値":                     "corporate_value",
    "マテリアリティ":               "materiality",
    "ステークホルダーエンゲージメント": "stakeholder_engagement",
    "企業理念":                     "corporate_philosophy",
}

# ── ESG pillar assignment ─────────────────────────────────────────────────────
# E = Environmental, S = Social, G = Governance, B = Business/Strategy
ESG_THEME_PILLAR = {
    "decarbonization":          "E",
    "circular_economy":         "E",
    "biodiversity":             "E",
    "employee_health":          "S",
    "customer_value":           "S",
    "human_rights":             "S",
    "human_capital":            "S",
    "dei":                      "S",
    "community_coexistence":    "S",
    "stakeholder_engagement":   "G",
    "corp_governance":          "G",
    "security":                 "G",
    "risk_compliance":          "G",
    "management_ops":           "G",
    "materiality":              "G",
    "corporate_philosophy":     "G",
    "competitiveness":          "B",
    "stakeholder_cocreation":   "B",
    "business_portfolio":       "B",
    "intellectual_capital":     "B",
    "digital_transformation":   "B",
    "financial_metrics":        "B",
    "corporate_value":          "B",
}

# ── Pillar theme lists ────────────────────────────────────────────────────────
PILLAR_THEMES = {
    pillar: [t for t, p in ESG_THEME_PILLAR.items() if p == pillar]
    for pillar in ("E", "S", "G", "B")
}

# ── Readable labels for final panel columns ───────────────────────────────────
COLUMN_LABELS = {
    # identifiers
    "stock_code":                   "Stock Code",
    "year":                         "Year",
    "company_name":                 "Company Name",
    # GICS
    "sector":                       "GICS Sector",
    "industry_group":               "GICS Industry Group",
    "industry":                     "GICS Industry",
    "sub_industry":                 "GICS Sub-Industry",
    # market cap
    "market_cap":                   "Market Cap (JPY)",
    "log_mc":                       "Log Market Cap",
    "log_mc_excess":                "Excess Log Market Cap (market-demeaned)",
    "dlog_mc":                      "ΔLog Market Cap (YoY growth)",
    # macro
    "interest_rate":                "Japan Call Rate (%)",
    # fundamentals (annual financial)
    "roa_pct":                      "ROA (%)",
    "ltdebt_assets_pct":            "LT Debt / Assets (%)",
    "debt_equity_pct":              "Debt / Equity (%)",
    "total_debt":                   "Total Debt (JPY)",
    "total_equity":                 "Total Equity (JPY)",
    "total_assets":                 "Total Assets (JPY)",
    "rnd":                          "R&D Expenditure (JPY)",
    "total_revenue":                "Total Revenue (JPY)",
    "rnd_share_pct":                "R&D / Revenue (%)",
    # fundamentals lags
    "roa_pct_lag1":                 "ROA (%, t−1)",
    "ltdebt_assets_pct_lag1":       "LT Debt / Assets (%, t−1)",
    "debt_equity_pct_lag1":         "Debt / Equity (%, t−1)",
    "total_debt_lag1":              "Total Debt (JPY, t−1)",
    "total_equity_lag1":            "Total Equity (JPY, t−1)",
    "total_assets_lag1":            "Total Assets (JPY, t−1)",
    "rnd_lag1":                     "R&D Expenditure (JPY, t−1)",
    "total_revenue_lag1":           "Total Revenue (JPY, t−1)",
    "rnd_share_pct_lag1":           "R&D / Revenue (%, t−1)",
    # IBES analyst consensus (December month-end, FY1 horizon)
    "ibes_eps_mean":                "IBES EPS Consensus (mean)",
    "ibes_rev_mean":                "IBES Revenue Consensus (mean)",
    "ibes_rec_mean":                "IBES Recommendation (mean, 1=Strong Buy–5=Sell)",
    "ibes_pt_mean":                 "IBES Price Target (mean)",
    # IBES lags
    "ibes_eps_mean_lag1":           "IBES EPS Consensus (t−1)",
    "ibes_rev_mean_lag1":           "IBES Revenue Consensus (t−1)",
    "ibes_rec_mean_lag1":           "IBES Recommendation (t−1)",
    "ibes_pt_mean_lag1":            "IBES Price Target (t−1)",
    # ESG scores
    "sentiment_mean":               "ESG Sentiment (mean)",
    "esg_match_composite":          "ESG Match Score (composite)",
    "esg_match_e":                  "ESG Match Score — Environmental",
    "esg_match_s":                  "ESG Match Score — Social",
    "esg_match_g":                  "ESG Match Score — Governance",
    "esg_match_b":                  "ESG Match Score — Business/Strategy",
    "esg_tend_composite":           "ESG Tendency Score (composite)",
    "esg_tend_e":                   "ESG Tendency Score — Environmental",
    "esg_tend_s":                   "ESG Tendency Score — Social",
    "esg_tend_g":                   "ESG Tendency Score — Governance",
    "esg_tend_b":                   "ESG Tendency Score — Business/Strategy",
    # lagged (t-1)
    "esg_tend_composite_lag1":      "ESG Tendency Score (composite, t−1)",
    "esg_tend_e_lag1":              "ESG Tendency Score — E (t−1)",
    "esg_tend_s_lag1":              "ESG Tendency Score — S (t−1)",
    "esg_tend_g_lag1":              "ESG Tendency Score — G (t−1)",
    "esg_tend_b_lag1":              "ESG Tendency Score — B (t−1)",
    "esg_match_composite_lag1":     "ESG Match Score (composite, t−1)",
    "sentiment_mean_lag1":          "ESG Sentiment (t−1)",
}

# Reverse lookup: readable label → column name
LABEL_TO_COL = {v: k for k, v in COLUMN_LABELS.items()}

# ── Business/Strategy topic scope ─────────────────────────────────────────────
# Focal topics for this project (B pillar — all 7 business/strategy themes)
ENV_TOPICS = [
    "competitiveness",
    "stakeholder_cocreation",
    "business_portfolio",
    "intellectual_capital",
    "digital_transformation",
    "financial_metrics",
    "corporate_value",
]

ENV_TOPIC_LABELS = {
    "competitiveness":        "Competitiveness",
    "stakeholder_cocreation": "Stakeholder Co-creation",
    "business_portfolio":     "Business Portfolio",
    "intellectual_capital":   "Intellectual Capital",
    "digital_transformation": "Digital Transformation",
    "financial_metrics":      "Financial Metrics",
    "corporate_value":        "Corporate Value",
}

# Sub-score suffixes produced by compute_tendency.py for each theme
SUB_SCORE_SUFFIXES = [
    "related_ratio", "pos_ratio", "neg_ratio",
    "related_mean",  "pos_mean",  "neg_mean",
]
