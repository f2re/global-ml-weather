"""Run the analytic product-to-forecast regression tests without network access."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    test = root/'tests'/'test_product_workflow.py'
    if not test.is_file():
        raise SystemExit('Нужна рабочая копия с тестами дополнения.')
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q',str(test)],cwd=root))
