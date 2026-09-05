# Commands Handbook

Run commands from the repository root. Python 3.12 and uv are the current
environment tools; the legacy Conda hook has been removed.

## Environment

```bash
uv sync --locked
uv run python --version
uv pip list
```

The current lockfile contains DVC, not the full research environment. To migrate
the retained research dependencies before running analyses:

```bash
uv add -r requirements.txt pandas statsmodels pyyaml
```

Review dependency changes in `pyproject.toml` and `uv.lock`. For later package
changes, use `uv add <package>` or `uv remove <package>`.

## Research pipeline

```bash
uv run python src/run_analysis.py --dry-run
uv run python src/run_analysis.py
```

The pipeline rebuilds `data/processed/panel.parquet`, writes figures and tables
under `data/results/`, and updates `docs/metadata/` and `docs/actionbrief.yaml`.
To run individual stages:

```bash
uv run python src/preprocessing/01_data_cleaning.py
uv run python src/analyses/90_export.py
```

Optional notebook generators overwrite their corresponding notebook skeletons:

```bash
uv run python src/analyses/02_data_inspection.py
uv run python src/analyses/03_analysis.py
```

## DVC

DVC is initialized. Default remote `storage` points to
`/Volumes/MikesDataBackup/dvc_remote`; do not initialize it again.

```bash
uv run dvc status
uv run dvc remote list
```

Update only changed standard units, for example after regenerating results:

```bash
uv run dvc add data/results/
```

Keep `data/raw/`, `data/processed/`, `data/exp/`, `data/results/`, `article/`,
`docs/framelog/`, `docs/frame-version.md`, and `docs/stage-control.md`
as eight separate units. Commit the changed pointers and corresponding source
or documentation changes through the Git workflow.

For an explicitly requested upload or restore:

```bash
uv run dvc push
uv run dvc status -c
# Restore the version referenced by the current pointers:
uv run dvc pull
```

## Git

Inspect the branch and remote before synchronization:

```bash
git status --short --branch
git remote -v
```

`origin` is ESG05h. The existing `main` tracking branch is `secondary/main`
(ZDP02k); use an explicitly chosen remote when synchronizing. Template remotes
are read-only sources. Follow `manage-git-workflow` for reviewed commits.
