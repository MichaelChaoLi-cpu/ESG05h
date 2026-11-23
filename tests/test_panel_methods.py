"""Numerical equivalence to full LSDV and calendar-gap regression checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from panel_methods import panel_ols,calendar_lag

def test_fixed_effects():
 rng=np.random.default_rng(491)
 d=pd.DataFrame([(g,t) for g in range(14) for t in range(6)],columns=['stock_code','year'])
 d['x']=rng.normal(size=len(d));d['z']=rng.normal(size=len(d))
 d['y']=1.4*d.x-0.7*d.z+d.stock_code*.3+d.year*.8+rng.normal(size=len(d))
 for sample in [d,d.drop(index=[0,1,6,19,27,28,39,62])]:
  fit,n,g=panel_ols(sample,'y',['x','z'])
  ref=smf.ols('y ~ x + z + C(stock_code) + C(year)',sample).fit(cov_type='cluster',use_t=True,cov_kwds={'groups':sample.stock_code})
  np.testing.assert_allclose(fit.params[['w_x','w_z']],ref.params[['x','z']],rtol=1e-10)
  np.testing.assert_allclose(fit.bse[['w_x','w_z']],ref.bse[['x','z']],rtol=1e-10)
  np.testing.assert_allclose(fit.pvalues[['w_x','w_z']],ref.pvalues[['x','z']],rtol=1e-10)
  assert fit.df_resid==ref.df_resid

def test_calendar_lag():
 d=pd.DataFrame({'stock_code':['a','a','a','b','b'],'year':[2018,2020,2021,2020,2021],'v':[1,2,3,4,5]},index=[8,3,1,7,9])
 np.testing.assert_allclose(calendar_lag(d,'v'),[np.nan,np.nan,2,np.nan,4],equal_nan=True)
 np.testing.assert_allclose(calendar_lag(d,'v',2),[np.nan,1,np.nan,np.nan,np.nan],equal_nan=True)

if __name__=='__main__':
 test_fixed_effects();test_calendar_lag();print('LSDV coefficient / SE / p-value equivalence and calendar-gap tests passed')
