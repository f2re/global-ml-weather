"""Run the analytic retrieval-to-training integration checks; not weather skill."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',
        'tests/test_products.py','tests/test_products_pipeline.py'],cwd=root))
