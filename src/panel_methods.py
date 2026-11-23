"""Calendar-year lags and joint fixed effects shared by all analysis entry points."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def calendar_lag(df, column, periods=1):
    """Match the same company's value in year t-periods; never bridge gaps."""
    if df.duplicated(["stock_code", "year"]).any():
        raise ValueError("Company-year keys must be unique")
    lookup = df.set_index(["stock_code", "year"])[column]
    keys = pd.MultiIndex.from_arrays([df.stock_code, df.year - periods])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=df.index)


def panel_ols(df, y, x_vars, entity_fe=True, time_fe=True, add_rate=False):
    """Exact LSDV-equivalent slopes; company-clustered CR1 with full-model df.

    Both regressors and year indicators are projected off company indicators.
    R-squared measures fit of this regression relative to the company-demeaned
    outcome, including the contribution of year effects. It is not a partial
    R-squared after removing both fixed effects. Inference uses t(G-1).
    """
    x_vars = [x_vars] if isinstance(x_vars, str) else list(x_vars)
    regressors = x_vars + (["interest_rate"] if add_rate else [])
    d = df[[y] + regressors + ["stock_code", "year"]].dropna().reset_index(drop=True)
    n, groups = len(d), d.stock_code.nunique()
    X = d[regressors].astype(float).copy()
    if time_fe:
        X = pd.concat([X, pd.get_dummies(d.year, prefix="year", drop_first=True,
                                        dtype=float)], axis=1)
    Y = d[y].astype(float)
    if entity_fe:
        X = X - X.groupby(d.stock_code).transform("mean")
        Y = Y - Y.groupby(d.stock_code).transform("mean")
        X = X.rename(columns={v: f"w_{v}" for v in x_vars})
    else:
        X = sm.add_constant(X, has_constant="add")
    rank = np.linalg.matrix_rank(X.to_numpy())
    if rank != X.shape[1]:
        raise ValueError("Regressor design is rank deficient after fixed effects")
    full_rank = rank + (groups if entity_fe else 0)
    if groups < 2 or n <= full_rank:
        raise ValueError("Insufficient clusters or residual degrees of freedom")
    result = sm.OLS(Y, X).fit(cov_type="cluster", use_t=True,
                            cov_kwds={"groups": d.stock_code,
                                      "use_correction": False,
                                      "df_correction": True})
    # Match CR1 from the full company/year dummy regression, including absorbed df.
    result._results.cov_params_default *= groups / (groups - 1) * (n - 1) / (n - full_rank)
    result._results.df_resid = n - full_rank
    result._results.df_resid_inference = groups - 1
    result._results._cache.clear()
    return result, n, groups
