# MiliFrame directory migration — 2026-09-05

## Canonical locations

| Previous location | Current location |
| --- | --- |
| export/figures/ | data/results/figures/ |
| export/tables/ | data/results/tables/ |
| export/metadata/ | docs/metadata/ |
| export/actionbrief.yaml | docs/actionbrief.yaml |
| src/01_data_cleaning.py | src/preprocessing/01_data_cleaning.py |
| src/02_data_inspection.py | src/analyses/02_data_inspection.py |
| src/03_analysis.py | src/analyses/03_analysis.py |
| src/90_export.py | src/analyses/90_export.py |
| export/code/run_analysis.py | src/run_analysis.py |

All ten mirrored scripts in the old code/nbs folders matched their canonical
source files byte-for-byte before removal. The duplicate AnaSOP also matched.
Local notebooks were moved beside their corresponding scripts with outputs
preserved and setup paths adjusted. They remain Git-ignored.

The obsolete Conda hook and its empty etc directory were removed. The existing
requirements.txt remains as a legacy dependency input. Current environment
management instructions use uv. Historical docs/LAWS schemas are retained but
are not the active content standard.

## Verification

- SHA-256 comparison confirmed that all 11 figures, 16 workbooks and two metadata
  files retained their original bytes.
- All canonical Python scripts passed syntax parsing.
- The actionbrief schema passed the installed validator's schema checks after
  shortening two overlong captions. All 27 figure/table paths resolve relative
  to the repository root.
- Relocated script root/output paths were checked in an isolated copy. Both
  notebook generators ran there and their generated code cells parsed.
- The pipeline runner passed dry-run checks from outside the repository.
- DVC tracks all 27 result files under data/results.dvc and reports up to date.
- Git whitespace validation passed.

The statistical pipeline was not rerun: the current uv environment contains
DVC, while research dependencies still require migration. Existing numerical
results and models were not changed. Data has not been pushed to DVC storage.

## Remaining research-contract migration

The full installed content validator reports 18 AnaSOP issues, while the
content-dictionary schema passes. These are legacy contract issues outside
this directory migration:

- Eight expected level-two section headings are absent in the legacy layout.
- Three equation-format checks fail (dollar delimiters, Unicode math and
  expressions outside LaTeX delimiters).
- Five checks flag technical paths or file names inside AnaSOP.
- Two checks find missing explicit links from Sections 5–7 to Section 4 inputs
  and Section 8 outputs.

This is not a successful final content audit. Research-contract migration and
review are still required before manuscript handoff.
