from pathlib import Path
import sys,json
import numpy as np,pandas as pd,openpyxl
sys.path.insert(0,str(Path.cwd()/'src'))
from panel_methods import calendar_lag
p=pd.read_parquet('data/processed/panel.parquet');old=pd.read_parquet('article/origin/pre_audit_fix_20260905/data/processed/panel.parquet')
checked=0
for c in p:
 if c.endswith('_lag1'):
  np.testing.assert_allclose(p[c],calendar_lag(p,c[:-5]),equal_nan=True);checked+=1
raw=pd.read_excel('data/raw/Market_cap_annual.xlsx',dtype={'Stock Code':str});cols=[c for c in raw if '(JPY)' in str(c)]
m=raw[['Stock Code']+cols].rename(columns={'Stock Code':'stock_code'}).melt(id_vars='stock_code',var_name='year',value_name='mc')
m.stock_code=m.stock_code.astype(str).str.strip();m.year=m.year.str.extract(r'(\d{4})')[0].astype(int);m['mc']=pd.to_numeric(m.mc,errors='coerce');m['log']=np.log(m.mc)
keys=pd.MultiIndex.from_arrays([p.stock_code,p.year-1]);prior=m.set_index(['stock_code','year'])['log'].reindex(keys).to_numpy()
np.testing.assert_allclose(p.dlog_mc,p.log_mc-prior,equal_nan=True)
result={'lag1_columns_verified':checked,'annual_growth_verified_against_raw_prior_year':True,'growth_values_changed':int((~np.isclose(p.dlog_mc,old.dlog_mc,equal_nan=True)).sum()),'rows':len(p),'baseline_n':int(p[['log_mc','match_competitiveness_lag1']].dropna().shape[0]),'r1_n':int((p.log_mc.notna() & calendar_lag(p,'match_competitiveness',2).notna()).sum())}
changes=[]
for f in Path('data/results/tables').glob('*.xlsx'):
 before=Path('article/origin/pre_audit_fix_20260905')/f
 a=list(openpyxl.load_workbook(before,data_only=True).active.values);b=list(openpyxl.load_workbook(f,data_only=True).active.values)
 assert len(a)==len(b)
 for i,(ar,br) in enumerate(zip(a,b),1):
  for j,(av,bv) in enumerate(zip(ar,br),1):
   if av!=bv:changes.append({'file':f.name,'row':i,'column':j,'before':av,'after':bv})
O=Path('article/show/audit_evidence/2026-09-05_fixed');pd.DataFrame(changes).to_csv(O/'table_changes.csv',index=False)
(O/'data_validation.json').write_text(json.dumps(result,indent=2));print(result)
