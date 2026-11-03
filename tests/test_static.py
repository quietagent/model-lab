"""The front end has no build step, so nothing else would catch a syntax error in it.

One bad quote in any module stops the whole page from loading (seen while editing render.js),
and the Python tests alone would still pass.
"""
import json
import shutil
import subprocess

import pytest

from app import config

JS_FILES = sorted((config.STATIC_DIR / "js").glob("*.js"))


def test_the_page_loads_every_module_that_exists():
    assert len(JS_FILES) >= 15
    index = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/static/js/main.js" in index


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_javascript_module_parses(path):
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def render_markdown(sources):
    """Run the page's own Markdown renderer in Node and return the HTML for each source."""
    module = (config.STATIC_DIR / "js" / "markdown.js").as_uri()
    script = (f"import {{ renderMarkdown }} from {json.dumps(module)};\n"
              f"console.log(JSON.stringify({json.dumps(sources)}.map(renderMarkdown)));\n")
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_markdown_headings_underlined_with_equals_signs():
    # Llama 3.1 opened a live reply with "Recursion in Python" over a row of "=", shown as plain text.
    heading, two_lines, rule, bare, comparison = render_markdown([
        "Recursion in Python\n=======================\n\nRecursion is a technique.",
        "First line\nSecond line\n===\nBody",
        "A paragraph.\n---\nNext paragraph.",
        "===\nnot a heading",
        "a == b",
    ])
    assert heading == '<h1 dir="auto">Recursion in Python</h1><p dir="auto">Recursion is a technique.</p>'
    assert two_lines == '<p dir="auto">First line</p><h1 dir="auto">Second line</h1><p dir="auto">Body</p>'
    assert rule == '<p dir="auto">A paragraph.</p><hr><p dir="auto">Next paragraph.</p>'     # still a rule
    assert "<h1" not in bare and "<h1" not in comparison


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_markdown_never_lets_model_output_inject_html_or_load_images():
    script_tag, image, bad_link, good_link, fence = render_markdown([
        "<script>alert(1)</script> and <img src=x onerror=alert(1)>",
        "![chart](https://example.com/a.png)",
        "[click](javascript:alert(1))",
        "[docs](https://example.com/docs)",
        "```python\nprint('<b>hi</b>')\n```",
    ])
    assert "<script" not in script_tag and "<img" not in script_tag and "&lt;script&gt;" in script_tag
    assert "<img" not in image and 'href="https://example.com/a.png"' in image       # a link, never loaded
    assert "href" not in bad_link and "click" in bad_link
    assert 'href="https://example.com/docs"' in good_link and 'rel="noopener noreferrer"' in good_link
    assert "<pre" in fence and "<b>" not in fence and "&lt;b&gt;" in fence


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_elo_ratings_are_shown_as_sent_not_rounded_twice():
    """Live: after a tie the ratings were 1014.53 and 985.47, sent as 1014.5 and 985.5. The table rounded
    them again and showed 1015 and 986, which do not add up to 2000."""
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    script = "\n".join([f"import {{ fmtRating }} from {json.dumps(module)};",
                        "console.log(JSON.stringify([1014.5, 985.5, 1016, 984, null].map(fmtRating)));"])
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    shown = json.loads(result.stdout)
    assert shown[:4] == ["1014.5", "985.5", "1016.0", "984.0"] and shown[4] not in ("null", "NaN")
    assert float(shown[0]) + float(shown[1]) == 2000


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_a_retry_note_counts_down_and_then_says_the_retry_was_sent():
    """Live: a running test went on showing "retry 2 in 2.4s" for as long as the retried request took,
    because the note was written once, when the request failed. It now depends on the time."""
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    script = "\n".join([f"import {{ retryStatus }} from {json.dumps(module)};",
                        "const until = 1000;",
                        "console.log(JSON.stringify([997.6, 999, 999.96, 1000, 1000.5, 1090].map("
                        "(now) => retryStatus(2, until, now))));"])
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    waiting, soon, almost, due, started, long_after = json.loads(result.stdout)
    assert (waiting, soon) == ("retry 2 in 2.4s", "retry 2 in 1.0s")
    assert almost == due == started == long_after == "retry 2 sent - waiting for its reply"
    # The page keeps the note fresh: it stores when the wait ends, not a sentence.
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "until: event.at + event.delay" in page and "retryStatus(Number(node.dataset.attempt)" in page
    assert "`retry ${event.attempt} in ${event.delay}s`" not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_page_offers_a_retry_only_where_the_provider_failed_and_words_the_first_pass():
    module = (config.STATIC_DIR / "js" / "util.js").as_uri()
    rows = [{"status": "error", "score": None}, {"status": "cut_off", "score": None},
            {"status": "cut_off", "score": 0}, {"status": "fail", "score": 0}, {"status": "partial", "score": 0.5},
            {"status": "pass", "score": 1}, {"status": "empty", "score": 0}]
    runs = [
        {"score": 94.7, "retries": 1, "first_pass": {"score": 96.9, "n_error": 4, "n_cut_off": 0}},
        {"score": 80.3, "retries": 0, "first_pass": {"score": 80.3, "n_error": 1, "n_cut_off": 4}},
        {"score": 100, "retries": 2, "first_pass": {"score": None, "n_error": 0, "n_cut_off": 10}},
        {"score": 50, "retries": 0, "first_pass": None},
    ]
    script = "\n".join([f"import {{ isRetryable, firstPassText }} from {json.dumps(module)};",
                        f"console.log(JSON.stringify([{json.dumps(rows)}.map(isRetryable), "
                        f"{json.dumps(runs)}.map(firstPassText)]));"])
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    retryable, words = json.loads(result.stdout)
    # An API error and a reply the provider cut short. Never out of tokens (cut off, 0), a graded answer,
    # or a reply that was empty twice (0).
    assert retryable == [True, True, False, False, False, False, False]
    assert words[0] == {"first": "First pass 96.9", "left": "4 API errors",
                        "final": "Final 94.7 after one retry of failed requests"}
    assert words[1] == {"first": "First pass 80.3", "left": "1 API error, 4 cut off",
                        "final": "Final: the same - no failed request was retried"}
    assert words[2] == {"first": "First pass scored nothing", "left": "10 cut off",
                        "final": "Final 100 after 2 retries of failed requests"}
    assert words[3] is None
    page = (config.STATIC_DIR / "js" / "benchmark.js").read_text(encoding="utf-8")
    assert "results.filter(isRetryable)" in page and "firstPassText(run, labels)" in page
    # A reply that was empty twice has its own label, filter and colour: it is not shown as a plain "fail".
    assert "empty: 'empty reply'" in page and "['empty', 'Empty']" in page and "n_empty: count('empty')" in page
    styles = (config.STATIC_DIR / "css" / "app.css").read_text(encoding="utf-8")
    assert ".badge-empty" in styles and ".bg-empty" in styles


def saved_settings(saved):
    """The settings the page starts with when the browser holds `saved` (state.js, run in Node)."""
    module = (config.STATIC_DIR / "js" / "state.js").as_uri()
    lines = [f"const saved = {json.dumps(json.dumps(saved))};",
             "globalThis.localStorage = { getItem: (key) => (key === 'modellab.settings' ? saved : null), "
             "setItem() {}, removeItem() {} };",
             f"const {{ settings }} = await import({json.dumps(module)});",
             "console.log(JSON.stringify(settings));"]
    script = chr(10).join(lines)
    result = subprocess.run(["node", "--input-type=module", "-"], input=script, capture_output=True,
                            text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_a_saved_hard_token_limit_that_is_the_old_default_moves_up_to_the_new_one():
    """The Hard default went from 16,384 to 32,768. A browser that still holds the old default gets the
    new one; a value the user chose themselves is kept."""
    from app import main

    assert saved_settings({})["bench"]["maxTokensHard"] == main.HARD_MAX_TOKENS == 32768
    moved = saved_settings({"version": 3, "bench": {"maxTokens": 8192, "maxTokensHard": 16384, "concurrency": 3}})
    assert moved["bench"]["maxTokensHard"] == 32768 and moved["version"] == 4
    assert (moved["bench"]["maxTokens"], moved["bench"]["concurrency"]) == (8192, 3)     # nothing else changes
    assert saved_settings({"version": 3, "bench": {"maxTokensHard": 20000}})["bench"]["maxTokensHard"] == 20000
    # Chosen again after the change (settings already at version 4): it is the user's value now.
    assert saved_settings({"version": 4, "bench": {"maxTokensHard": 16384}})["bench"]["maxTokensHard"] == 16384
