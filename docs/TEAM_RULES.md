# Team Rules

Defines how we manage code, data, and experiments for reproducible, collaborative work.

---

## 1) Principles

1. **Data-centric** — code and structure exist to make data generation reproducible.
2. **exp for feasibility, dev for readability** — experiment on branches, refactor on `dev`.
3. **Tests are for data** — focus on shape, distributions, and key visualisations; keep records in `.ipynb`.
4. **Reuse, not redefine** — extract reusable logic into modules rather than duplicating it.

---

## 2) Repository Structure

```
myproj/
├── src/                # Python scripts; experimental *.ipynb allowed but not tracked
├── data/               # Datasets and reports (DVC-tracked, Git-ignored)
├── docs/               # Research contract, content dictionary and metadata
├── article/            # Manuscript originals and presentation materials
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked environment (requirements.txt is a legacy input)
├── .env                # Local environment variables (not committed)
└── .gitignore
```

---

## 3) Golden Rules

| What | Where |
|------|-------|
| Code | Git |
| Data & large outputs | DVC |
| Parameters | inline in scripts or `src/config.py` |
| Packages | `uv add / remove` (updates `pyproject.toml` and `uv.lock`) |
| Experiments | `git checkout -b exp/<id>-<topic>` + DVC |
| Results, code, content metadata | `data/results/`, `src/`, `docs/` |
| Current content standards | `build-content-dictionary/references/` in installed skills |

---

## 4) Git Practices

**Branches**

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reproducible. PR required, no direct push. |
| `dev` | Integration of validated experiments. |
| `exp/<id>-<topic>` | Active experiment (e.g. `exp/001-baseline`). |
| `feature/<scope>-<desc>` | Productionising a proven experiment. |
| `data/<dataset>-<change>` | Data ingestion / cleaning. |
| `spike/<topic>` | Throwaway prototype, may never merge. |
| `hotfix/<issue>` | Urgent fix off `main`. |

Flow: `spike/*` → `exp/*` → `feature/*` → `dev` → `main`

**Commit format:** `<type>(<scope>): <imperative summary ≤ 72 chars>`

| Type | Use |
|------|-----|
| `feat` | New feature or model |
| `fix` | Bug fix |
| `data` | Dataset changes |
| `exp` | Experiment config or result |
| `refactor` | Code restructure, no behaviour change |
| `docs` | Documentation only |
| `build` | Environment, dependencies, or build scripts |
| `chore` | Tooling / config / maintenance |

```
feat(model): add cosine LR scheduler
exp: 003 - grid search over regularisation strengths
data: add CMIP6 temperature 2000-2050
```

---

## 5) DVC Practices

```bash
uv run dvc add data/raw/
git add data/raw.dvc .gitignore
git commit -m "data: track dataset"
dvc push
```

- Always `dvc push` after updating data or artifacts.
- HPC shared cache (run once):
  ```bash
  dvc config cache.shared group
  dvc config cache.type hardlink,symlink
  ```

---

## 6) Script / Notebook Rules

Every script or notebook must start with:

```python
import os
from dotenv import load_dotenv

load_dotenv()
os.chdir(os.getenv("PROJECT_ROOT"))
```

Import order: standard library → third-party → internal, each group alphabetically sorted.

```python
import os

import numpy as np

from myproj.utils import helper
```

---

## 7) Experiment Workflow

```bash
git checkout dev && git pull
git checkout -b exp/001-description

# work ...

dvc add <large outputs>
git add . && git commit -m "exp: 001 - description"
dvc push

git checkout dev && git merge exp/001-description
git push && dvc push
```

To revisit: `git checkout exp/001-description && dvc pull`

---

## 8) Environment Management

See `COMMANDS.md` for full setup instructions.

```bash
uv add <package>       # updates pyproject.toml and uv.lock
uv remove <package>    # same
```

---

## 9) Collaboration Rhythm

- **Start of session:** `git pull && dvc pull`
- **End of session:** `git push && dvc push`
- **Never** commit raw data, credentials, or `.env` to Git.

---

## 10) Versioning (`pyproject.toml`)

`major.minor.patch`

| Level | Trigger |
|-------|---------|
| Major | Breaks or invalidates existing downstream components |
| Minor | Adds new components without breaking existing ones |
| Patch | Incremental improvement within the current segment |

Update the version whenever a workflow or structural change is merged to `dev`.

---

## 11) Security

- Store secrets in `.env` only — never commit them.
- Restrict DVC remote access to the project group (`chmod g+rwX`).

---

## 12) Research Outputs and Manuscript Handoff

Canonical figures and tables live in `data/results/figures/` and
`data/results/tables/`. Canonical scripts live in `src/preprocessing/` and
`src/analyses/`, with shared definitions in `src/vardict.py` and the pipeline
entry point in `src/run_analysis.py`.

`docs/AnaSOP.md` is the research contract. `docs/actionbrief.yaml` is the content
dictionary; all figure and table paths are relative to the repository root.
Dataset and variable metadata live in `docs/metadata/`. Manuscript materials
belong in `article/origin/` and `article/show/` when produced.

Current content standards are bundled with the installed
`build-content-dictionary` skill. `docs/LAWS/` is retained as historical material
and is not the active validator. The legacy AnaSOP still needs a separate
research-content review against the new eight-section standard.

Do not mirror source scripts or AnaSOP into another output bundle. DVC tracks
results as the existing `data/results/` unit; Git tracks source scripts,
metadata, the content dictionary and DVC pointers.
