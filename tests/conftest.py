"""Run every test against the mock provider and a throwaway database."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = tempfile.mkdtemp(prefix="modellab_tests_")
os.environ["MOCK"] = "1"
os.environ["MOCK_FAST"] = "1"
os.environ["MODEL_LAB_DATA_DIR"] = _TMP
# Importing app.main loads .env. Point it at a file that does not exist so the real keys
# never reach the test process (they would change results and could end up in test output).
os.environ["MODEL_LAB_ENV_FILE"] = str(Path(_TMP) / "no.env")
os.environ.pop("OPENROUTER_API_KEY", None)
os.environ.pop("HF_TOKEN", None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_mock_counters():
    """The mock's flaky, dropping, mute and roaming models count the requests they have had for each
    prompt, and that count outlives a test. Start it again before every test, in every file: a test that
    met a used-up counter failed or passed depending on which file had run before it
    (test_averages.py after test_cut_off.py)."""
    from app import mock
    mock.forget_hits()
