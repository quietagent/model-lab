"""The group of settings the Leaderboard opens on.

Live (2026-10-09): one pinned Standard run of two categories (#41) was the newest Standard run, so the
Standard Leaderboard opened on its group of one, and the six baseline runs were in the menu next to it.
Now a level opens on the group with the most runs (the newest of them on a tie), unless the viewer has
picked a group.
"""
import json
import shutil
import subprocess

import pytest

from app import config


def in_node(lines):
    result = subprocess.run(["node", "--input-type=module", "-"], input=chr(10).join(lines), capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_leaderboard_opens_on_the_group_with_the_most_runs():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    # As the server sends them: the group with the newest run first.
    live = [{"key": "max_tokens=8192;judge=none", "runs": 1, "newest": 1791500000},
            {"key": "max_tokens=8192;judge=apodex", "runs": 6, "newest": 1791400000}]
    tie = [{"key": "older", "runs": 3, "newest": 100}, {"key": "newer", "runs": 3, "newest": 200},
           {"key": "small", "runs": 2, "newest": 300}]
    cases = [live, tie, [live[0]], [], None]
    assert in_node([
        f"import {{ defaultGroup }} from {json.dumps(module)};",
        f"console.log(JSON.stringify({json.dumps(cases)}.map(defaultGroup)));",
    ]) == ["max_tokens=8192;judge=apodex", "newer", "max_tokens=8192;judge=none", None, None]
    board = (config.STATIC_DIR / "js" / "leaderboard.js").read_text(encoding="utf-8")
    # A group the viewer picked is kept while it exists; otherwise the default.
    assert "groups.some((group) => group.key === picked()) ? picked() : defaultGroup(groups)" in board
