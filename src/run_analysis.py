"""Regenerate the processed panel and research outputs from the repository raw data."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("src/preprocessing/01_data_cleaning.py", "src/analyses/90_export.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check script paths and show execution order without changing data.")
    args = parser.parse_args()
    for script in SCRIPTS:
        path = ROOT / script
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Running {script}...", flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(path)], check=True, cwd=ROOT)
    print("Path checks complete." if args.dry_run else "Processed data and research outputs regenerated.")


if __name__ == "__main__":
    main()
