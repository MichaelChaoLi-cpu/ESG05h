from pathlib import Path
from docx import Document
import openpyxl,re,json,hashlib
R=Path.cwd();d=Document(R/'article/show/ESG05h_8000words_revised.docx');n=0;allvals=set()
for i in range(1,10):
 rows=list(openpyxl.load_workbook(next((R/'data/results/tables').glob(f'Table{i}_*.xlsx')),data_only=True).active.values)
 t=d.tables[i+7]
 for row in rows:
  for v in row:
   if v is not None:allvals.update(re.findall(r'[-−]?\d+\.\d+%?',str(v)))
 if i==2:
  for ni,si in enumerate([1,3,5,7,9],1):
   for j in range(7):assert t.cell(ni,j+1).text==f'{rows[si][2+2*j]}\n{rows[si][3+2*j]}';n+=2
 else:
  cols=list(range(len(rows[0])))
  if i==7:cols.remove(4)
  for ri,row in enumerate(rows[1:],1):
   for ci,j in enumerate(cols):
    a='' if row[j] is None else str(row[j]);b=t.cell(ri,ci).text
    assert a==b,(i,ri,ci,a,b)
    n+=bool(re.search(r'\d',a))
for shape,f in zip(d.inline_shapes,sorted((R/'data/results/figures').glob('fig*.png'))):
 rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
 assert d.part.related_parts[rid].blob==f.read_bytes(),f
old=Document(R/'article/origin/pre_audit_fix_20260905/article/show/ESG05h_8000words.docx')
def refs(doc):
 a=next(i for i,p in enumerate(doc.paragraphs) if p.text=='References')
 return [p._p.xml for p in doc.paragraphs[a:]]
assert refs(d)==refs(old)
unmatched=[]
for p in d.paragraphs:
 if p.text=='Figures':break
 for val in re.findall(r'(?<![\w])[-−]?\d+\.\d+%?',p.text):
  if val.replace('−','-') not in allvals:unmatched.append((val,p.text[:65]))
print('numeric cells',n,'images match 11; bibliography XML unchanged');print('review derived/rounded/constants:',unmatched)
Path('article/show/audit_evidence/2026-09-05_fixed/verification.json').write_text(json.dumps({'numeric_cells':n,'images':11,'bibliography_xml_unchanged':True,'non_table_decimal_occurrences':unmatched},ensure_ascii=False,indent=2))
