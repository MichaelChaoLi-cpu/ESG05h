"""
run_analysis.py
Minimal reproducible script — regenerates all export/ artifacts.

Run from project root:
    python export/code/run_analysis.py
"""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for script in ["src/01_data_cleaning.py", "src/90_export.py"]:
    print(f"Running {script}...")
    subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=str(ROOT))
print("Done. All export/ artifacts regenerated.")
