"""Generate a notebook that uses the canonical analysis/export pipeline.

Run: python src/analyses/03_analysis.py
The notebook reruns 90_export.py and displays its regenerated tables and figures.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/analyses/03_analysis.ipynb'

def markdown(text):
    return {'cell_type':'markdown','metadata':{},'source':text}

def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':text}

cells = [markdown('''# Business Disclosure and Market Capitalization

This notebook uses the same analysis implementation as the manuscript export.
Joint company/year fixed effects and exact calendar lags replace the earlier
row-shift and partially demeaned implementation. Results describe conditional
associations. Relatedness is normalized semantic alignment, M3 is an alternative
specification without Relatedness, and subgroup comparisons are exploratory.

Run data preprocessing first when raw inputs change. Running the cell below
regenerates tables, figures, and metadata in their standard project locations.
'''), code('''from pathlib import Path
import runpy
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'src/panel_methods.py').exists())
results = runpy.run_path(str(ROOT / 'src/analyses/90_export.py'))
'''), markdown('''## Interpretation and inference

Company-clustered CR1 covariance includes absorbed fixed-effect degrees of freedom;
p-values use t(G−1). Within R² includes year effects relative to the company-demeaned
outcome. Graphical intervals use the normal approximation ±1.96 SE. Stars do not
establish cross-group differences. Recommendations measure analyst assessment,
not attention; attenuation and longer lags do not identify causal mechanisms.
'''), code('''import pandas as pd
from IPython.display import display, Image
for path in sorted((ROOT / 'data/results/tables').glob('Table*.xlsx')):
    print(path.name)
    display(pd.read_excel(path))
'''), code('''for path in sorted((ROOT / 'data/results/figures').glob('fig*.png')):
    print(path.name)
    display(Image(filename=str(path)))
''')]
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':4}
OUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
print(f'Generated {OUT}')
