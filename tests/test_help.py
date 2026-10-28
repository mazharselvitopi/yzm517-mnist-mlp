import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,expected', [
    ('train.py', '--limit-train'),
    ('evaluate.py', '--final-model-selected'),
    ('predict.py', '--polarity'),
])
def test_cli_help_exits_without_data_or_checkpoint(script, expected):
    result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                            capture_output=True, text=True, cwd=ROOT, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    assert '--output' in result.stdout
