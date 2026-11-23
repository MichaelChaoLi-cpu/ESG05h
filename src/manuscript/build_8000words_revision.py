"""Rebuild the 2026-09-05 manuscript revision from its archived template and reviewed table snapshot."""
from pathlib import Path
import sys,json,copy,re
from docx import Document
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn,nsmap
from lxml import etree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from revisions_20260905 import R
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'article/origin/pre_audit_fix_20260905/article/show/ESG05h_8000words.docx'
out=ROOT/'article/show/ESG05h_8000words_revised.docx'
d=Document(source);ps=list(d.paragraphs);tables=json.loads((ROOT/'article/show/audit_evidence/2026-09-05_fixed/tables.json').read_text())
# Retain matching EndNote citation fields; untouched bibliography is copied verbatim.
fields={}
def xp(el,q):return etree.XPath(q,namespaces=nsmap)(el)
for para in d.paragraphs:
 children=list(para._p);depth=0;start=None
 for i,ch in enumerate(children):
  for kind in xp(ch,'.//w:fldChar/@w:fldCharType'):
   if kind=='begin':
    if depth==0:start=i
    depth+=1
   elif kind=='end':
    depth-=1
    if depth==0 and start is not None:
     block=children[start:i+1];ins=''.join(''.join(xp(c,'.//w:instrText/text()')) for c in block)
     txt=''.join(''.join(xp(c,'.//w:t/text()')) for c in block)
     if 'EN.CITE ' in ins:fields[txt]=[copy.deepcopy(c) for c in block]
def write_para(p,txt,size=12):
 for ch in list(p._p):
  if ch.tag!=qn('w:pPr'):p._p.remove(ch)
 def add(t):
  if t:
   r=p.add_run(t);r.font.name='Times New Roman';r.font.size=Pt(size)
 pos=0
 for m in re.finditer(r'\([^()]*\b(?:19|20)\d{2}[a-z]?[^()]*\)',txt):
  add(txt[pos:m.start()])
  if m.group() in fields:
   for el in fields[m.group()]:p._p.append(copy.deepcopy(el))
  else:add(m.group())
  pos=m.end()
 add(txt[pos:])
for pid,txt in R.items():write_para(ps[pid-1],txt)
# Low-risk text substitutions preserve remaining math and citation objects.
subs={'Mechanism Test 4: ROA Quartile Heterogeneity':'Comparison 4 ROA Quartile Heterogeneity',
 'Analyst attention':'Analyst assessment','analyst-channel comparison':'analyst recommendation comparison',
 'coverage volume':'normalized topic-relatedness',
 'It measures firm valuation':'It measures equity market value',
 'The intended fixed-effects model':'The fixed-effects model',
 'is consequently intended to describe':'consequently describes',
 'adding fundamentals reduces coefficient magnitudes without eliminating the associations':'adding fundamentals changes the precision and magnitude of selected associations'}
for p in d.paragraphs[:136]:
 for node in p._p.xpath('.//w:t'):
  for old,new in subs.items():node.text=(node.text or '').replace(old,new)
# Updated descriptive growth moments from the regenerated table.
oldrows=json.load(open(ROOT/'article/show/audit_evidence/2026-09-05_8000words/workbooks.json'))
old1=next(w['rows'] for w in oldrows if '/Table1_' in w['file'])
for old,new in zip(old1[1:],tables['1'][1:]):
 if old[0] and 'YoY' in old[0]:
  print('growth moments',old,new)
  for pid in [24,43]:
   for node in ps[pid-1]._p.xpath('.//w:t'):
    txt=node.text or ''
    for j in [2,3,6]:txt=txt.replace(f'{old[j]:.4f}',f'{new[j]:.4f}')
    node.text=txt
# Replace table cells while retaining their original paragraph/run formatting.
def setcell(c,txt):
 p=c.paragraphs[0];rpr=copy.deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
 for child in list(p._p):
  if child.tag!=qn('w:pPr'):p._p.remove(child)
 for extra in c.paragraphs[1:]:extra._p.getparent().remove(extra._p)
 r=p.add_run(txt)
 if rpr is not None:r._r.insert(0,rpr)
for i in range(1,10):
 rows=tables[str(i)];t=d.tables[i+7]
 if i==2:
  for ni,si in enumerate([1,3,5,7,9],1):
   for j in range(7):setcell(t.cell(ni,j+1),f'{rows[si][2+2*j]}\n{rows[si][3+2*j]}')
 else:
  cols=list(range(len(rows[0])))
  if i==7:cols.remove(4)
  for ri,row in enumerate(rows[1:],1):
   for ci,sourceci in enumerate(cols):setcell(t.cell(ri,ci),'' if row[sourceci] is None else str(row[sourceci]))
# Refresh media in their existing anchored positions.
figs=sorted((ROOT/'data/results/figures').glob('fig*.png'))
assert len(d.inline_shapes)==len(figs)==11
for shape,f in zip(d.inline_shapes,figs):
 rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
 d.part.related_parts[rid]._blob=f.read_bytes()
# Standalone notes; tables and formula bodies retain their layout.
from docx.text.paragraph import Paragraph
common='Notes: Joint company and year fixed effects. Standard errors in parentheses are clustered by company with CR1 adjustment including absorbed parameters. Stars use two-sided t(G−1) tests: * p < 0.10, ** p < 0.05, *** p < 0.01. Within R² includes year effects relative to the company-demeaned outcome.'
notes={2:'All models use 29,117 observations from 3,631 companies. Outcome: log market capitalization. Disclosure is matched to the preceding calendar year. M3 is an alternative specification without Relatedness. '+common,
3:'Outcome: log market capitalization. Both columns use the same complete-case sample; controls are calendar t−1 ROA and long-term debt/assets. Attenuation = 100 × (1 − |controlled coefficient / baseline coefficient|), using unrounded estimates. '+common,
4:'S1 outcome: current recommendation; S2 outcome: log market capitalization. S2a/S2b use a common 10,466-observation sample and S2b adds the calendar t−1 recommendation. S1 uses its own available sample. Standard errors are omitted from this summary. Attenuation uses unrounded coefficients and is unstable near a zero baseline; it is not a mediation proportion. '+common.replace('Standard errors in parentheses are clustered','Standard errors are clustered'),
5:'Outcome: log market capitalization. Pattern reports coefficient signs only, not statistical significance or a mechanism. Negative Mean is signed; a larger score is less negative. '+common,
6:'Outcome: log market capitalization. Quartiles use current ROA within each year, and firms may change groups. Comparisons are exploratory; no between-quartile coefficient tests are reported. '+common,
7:'R0, R1, R3 and R4 use log market capitalization; R2 uses annual log market-capitalization growth; R5 winsorizes growth at the pooled 1st/99th percentiles. R1 matches actual year t−2. R4 is balanced over the available estimation years. '+common,
8:'Outcome: log market capitalization. M1 is estimated separately by topic and industry. Industry comparisons are exploratory, without multiple-testing adjustment or formal between-sector contrasts. '+common,
9:'Outcome: log market capitalization. M3 includes calendar t−1 positive and signed negative mean scores, without Relatedness. Industry comparisons are exploratory. '+common}
for i,txt in notes.items():
 el=OxmlElement('w:p');d.tables[i+7]._tbl.addnext(el);p=Paragraph(el,d._body);write_para(p,txt,9)
 p.paragraph_format.space_after=Pt(6)
# Update all captions, including those outside the main body.
for p in d.paragraphs:
 if p.text.startswith(('Figure ', 'Table ')) and len(p.text)<180:
  for node in p._p.xpath('.//w:t'):
   txt=node.text or ''
   for old,new in [('Identification Robustness','Specification Sensitivity'),('Coverage Volume','Relatedness'),('Mechanism Test 1: Information Channel','Fundamentals Comparison'),('Mechanism Test 2: Analyst Attention Channel','Analyst Recommendation Comparison'),('Mechanism Test 4: ROA Quartile Heterogeneity','Exploratory ROA Quartile Comparison')]:txt=txt.replace(old,new)
   node.text=txt

# Figure notes distinguish normal approximation from table t tests.
for p in d.paragraphs:
 if p.text.startswith('Figure ') and re.match(r'Figure (?:[5-9]|10|11)\b',p.text):
  r=p.add_run((' ' if p.text.endswith('.') else '. ') + 'Error bars show normal-approximation 95% intervals (±1.96 company-clustered SE); subgroup comparisons are exploratory.');r.font.size=Pt(9)
# Limit claims in the policy discussion without disturbing citation fields.
p=ps[126]
for n in p._p.xpath('.//w:t'):n.text=(n.text or '').replace('the positive Relatedness association in six topics','the significant positive Relatedness associations in four baseline topics')
# Replace the former cosine label in all three native measurement equations.
for t in d.tables[:4]:
 for node in t._tbl.xpath('.//m:t'):
  if node.text:node.text=node.text.replace('2·sim','2·score')
# Add citation reminders as Word comments; the user will insert EndNote references.
comment_sources=[
 ('The TMPT methods study', 'Insert EndNote citation: Li, Keeley & Managi, ESG emphasis of top companies in East and Southeast Asian countries unveiled by deep learning, Green Technologies and Sustainability 4 (2026), 100296, Table 1, p. 9.'),
 ('The related Japanese securities-report study', 'Insert EndNote citation: Li, Keeley & Managi, A link between ESG topic emphases in security report and stock price, Journal of Cleaner Production 552 (2026), 148017, p. 5.'),
 ('The governance-disclosure study', 'Insert EndNote citation: Li, Keeley & Managi, Corporate governance disclosure volume, sentiment, and topic tone: Heterogeneous associations with firm valuation, Strategic Business Research 2 (2026), 100237, p. 7. SAPT metrics are reported here; they are not newly estimated in this manuscript.'),
 ('The environmental-disclosure study', 'Insert EndNote citation: Li, Keeley & Managi, Environmental Disclosure Dimensions and Japanese Firm Valuation: Volume, Sentiment, and Market Heterogeneity, Business Strategy and the Environment (2026), DOI 10.1002/bse.71323, section 2.1.')]
# Paragraph 34 contains the new evidence summary in one preserved prose block.
if hasattr(d,'add_comment'):
 d.add_comment(ps[33].runs, '\n\n'.join(text for _,text in comment_sources),author='Editorial note',initials='EN')
# Explicit normal math text prevents operator substitution in native math.
for t in d.tables[:4]:
 for mr in t._tbl.xpath('.//m:r'):
  val=''.join(xp(mr,'.//m:t/text()'))
  if 'score' in val or 'Neg' in val:
   pr=mr.find(qn('m:rPr'))
   if pr is None:pr=OxmlElement('m:rPr');mr.insert(0,pr)
   normal=OxmlElement('m:nor');normal.set(qn('m:val'),'1');pr.append(normal)
# Remove superseded table notes and refresh the retained figure sample note.
in_tables=False
for p in list(d.paragraphs):
 if p.text=='Tables':in_tables=True
 if p.text=='References':break
 if in_tables and p.text.startswith('Note:'):
  p._p.getparent().remove(p._p)
 elif p.text.startswith('Note: S2a and S2b'):
  for node in p._p.xpath('.//w:t'):node.text=(node.text or '').replace('10,477','10,466')
d.save(out)
text=lambda p:''.join(p._p.xpath('.//w:t/text() | .//m:t/text()'))
pp=d.paragraphs;a=next(i for i,p in enumerate(pp) if p.text=='Abstract');b=next(i for i,p in enumerate(pp) if p.text=='Figures')
wc=sum(len(text(p).split()) for p in pp[a:b]);print('BODY_WORDS',wc)
(ROOT/'article/show/audit_evidence/2026-09-05_fixed/doc_counts.json').write_text(json.dumps({'body_words':wc,'path':str(out)}))
