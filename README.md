# Model Lab

A local web app for testing and scoring LLMs through **OpenRouter** and **Hugging Face**.
Chat with a model, run an auto-graded benchmark, compare two models side by side, measure speed,
and keep a leaderboard - all from one page served on your own machine.

- Backend: Python + FastAPI + httpx (async), SQLite for storage
- Frontend: one page of plain HTML / CSS / JavaScript (no build step)
- Every provider call goes through the local backend; replies stream to the browser over SSE

## Run it

**Double-click `run.bat`** (or run `.\run.bat` in a terminal).

The first run creates `.venv`, installs the dependencies, starts the server and opens
<http://localhost:8100>. Later runs start straight away. Press `Ctrl+C` in the window to stop.

| Command | What it does |
|---|---|
| `.\run.bat` | Live providers (needs an API key) |
| `.\run.bat --mock` | Built-in mock provider - no key, nothing leaves your machine |
| `.\run.bat --port 8010` | Use another port |

Requires Python 3.10 or newer. Without the batch file:

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m app --open          (add --mock to try it without a key)
```

## API keys

Paste a key into the sidebar (one per provider), or copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY` / `HF_TOKEN`. A key typed in the app overrides the one in `.env`.

- Keys are sent only to your local server, which forwards them to the provider in the
  `Authorization` header. They are never logged and never written to the database.
- "Remember on this device" stores them in this browser's `localStorage`. Leave it off and the
  key is forgotten when you close or reload the page.

## The sidebar

- **Provider** - OpenRouter or Hugging Face (both OpenAI-compatible).
- **Model** - type a model ID (`openai/gpt-4o-mini`, `meta-llama/Llama-3.1-8B-Instruct`) or paste a
  link from openrouter.ai or huggingface.co; the provider is selected for you. OpenRouter models
  autocomplete from the public list (cached for 24 hours) with context length, price per 1M tokens
  and a "free" badge. The badge is only shown when every price OpenRouter lists for the model is
  zero (per request, per image, cached input and so on, not just the token prices) and the model
  outputs plain text; a model with free tokens that is billed another way, such as a music model
  charged per clip, gets an "other fees" badge instead. Hugging Face models autocomplete from its
  router's public list (also cached for 24 hours). There, one model is served by several
  providers, each with its own context size and price, so the list shows "from $x / $y" and the
  number of providers, and the line under the box shows the largest context and the cheapest
  price. A Hugging Face ID can end in a route such as `:fastest`, `:cheapest` or a provider name
  like `:novita`; with a provider name, that provider's own context and price are shown.
- **Provider (optional, OpenRouter only)** - under the model. OpenRouter sends each request to a
  provider of its choice, and price, speed and even the answers change with that choice (one model
  was served at $0.02, $0.05 and $0.22 per million tokens in one afternoon). The box lists the
  model's providers from OpenRouter's public endpoint list, each with its slug (`deepinfra/fp8`),
  price and the longest reply it allows. Pick one and every request for that model asks OpenRouter
  for that provider only, with fallbacks off. If it cannot serve, the request fails with an error
  that names the pin: it is never sent to another provider, and a reply that claims to come from
  another provider is refused. Leave it empty and OpenRouter chooses, as before. The pin belongs to
  the model it was set for; the judge is never pinned. On the Leaderboard a pinned model is a row of
  its own ("meta-llama/llama-3.1-8b-instruct · Groq"), ranked with the other models.
- **Test connection** - sends a tiny request and shows the latency and the provider that answered,
  or a clear error (401 bad key, 402 no credits, 404 model not found, 429 rate limited).
- **Parameters** - temperature, max tokens and system prompt (used by Chat, Arena and Speed).
  Max tokens defaults to 8,192. It is a cap, not a target: a reasoning model spends most of it
  thinking before the answer starts (a small one needed 4,100 tokens for a 150-word paragraph).
- **Judge model** - optional second model that grades open-ended answers (Writing and Honesty
  tests, custom judge tests, Arena verdicts). Before a benchmark (or a "Retry failed requests" that
  needs it) starts, the app checks with one tiny request that the judge answers: a judge that is
  gone (404), refuses the key (401, 403), has no credits (402) or rejects the request (400) stops the
  run before it starts, with a message that says so. A judge that is only busy (429, 5xx) is let
  through. Since 2026-10-18 the runs on the Leaderboard are judged by `openai/gpt-4o-mini`.

## The tabs

**Chat** - streaming Markdown (tables, code blocks with a copy button), a collapsible thinking
block, and under every reply: time to first token, tokens/sec, total time, tokens in/out and cost.
Thumbs up/down build a session chat score. `Enter` sends, `Shift+Enter` adds a line.

**Benchmark** - auto-graded tests at temperature 0, grouped into categories you can switch on/off.
There are two levels, chosen with the **Standard / Hard** switch above the categories. A run is at
one level, and the two are stored, listed and ranked apart: a Hard score is never averaged or
compared with a Standard one.

*Standard* (43 tests - most current models pass nearly all of them):

| Category | How it is graded |
|---|---|
| Math | The number after `Answer:` (it must be equal: 64.80 equals 64.8, 65 does not). When `Answer:` ends its line, the number is read from the next line that has text; lines that only open or close a display formula (`\[`, `\]`, `$$`) are skipped, so `\boxed{347}` there is read |
| Reasoning | Logic puzzles, sequences, letter counting, day of the week |
| Instructions | Code checks: exactly 5 lines with no blank lines, valid JSON with given keys, lowercase with no punctuation, exact format, exactly 3 sentences |
| Coding | The model's Python function is run against unit tests in a subprocess; score = passed / total |
| Knowledge | Facts with known answers |
| Arabic | Answer in Arabic, translate Arabic to English, reply only in Arabic script |
| Honesty | Made-up books and unknowable questions - the model must admit it does not know. With a judge model set, the judge decides (did it admit it, without inventing details?); without one, a phrase check is used |
| Long context | A passphrase hidden in about 5k tokens of filler |
| Writing | A strict judge (needs a judge model): it starts from 10, lists each deduction and uses the whole 1-10 scale. The length rules of each brief (100-150 words, under 150 words, one paragraph, four sentences) are measured by code and given to the judge as facts; each one broken caps the score 2 points lower |
| Custom | Your own tests: contains / exact / regex / number / judge (a plain 0-10 score against your rubric, more lenient than the built-in Writing judge). They run with either level |

*Hard* (67 more tests, built to tell strong models from weak ones; nine each for Reasoning and
Coding, 13 for Knowledge, 12 for Long context, four or five for the others):

| Category | What is asked | How it is graded |
|---|---|---|
| Math | Multi-step, competition-style problems | One exact number after `Answer:` |
| Reasoning | Famous puzzles with a twist (the memorised answer is wrong), two logic grids, and four puzzles that take a search: five hikers crossing a bridge three at a time, lockers with the primes missing, eight statements about themselves, a birthday worked out from who knows what | The number, name or date after `Answer:` |
| Instructions | Three to seven rules in one reply: word counts, a format, banned words, a required order, and rules about letters (no letter e; word n has n letters) | Every rule is checked by code; each broken rule costs half the marks |
| Coding | Harder functions with tricky edge cases. In four of them the first idea is wrong: exact decimal rounding (not float), text justification with changed rules, a longest subsequence with a tie-break, shell brace expansion | 13 to 17 unit tests each; score = passed / total |
| Knowledge | Less common facts with one exact answer, several with a better-known wrong number next to the right one. Four take a whole table or a rule rather than one remembered number: the dots on the Arabic alphabet, the abjad value of a word, a Julian date after 2100, the element names that do not end in "ium" | The name or number after `Answer:` |
| Arabic | Grammar (إعراب), a Saudi-dialect sentence to translate, word problems in dialect and in Standard Arabic | The grammar term and case; key meanings in the translation; the number after `الإجابة:` |
| Honesty | Questions built on a false premise | The judge decides whether the reply corrects the premise without inventing anything (a phrase check without a judge) |
| Long context | About 30k tokens with three to ten facts to combine (add them up after a correction and a transfer, follow a chain of three or five, skip a decoy, count people, apply prices that changed, work out an order from clues, keep a stock in three units, date two lines of a family, count a vote under two rules) | The combined answer: for the first four, found anywhere in the reply; for the eight that take working out, on the `Answer:` line |
| Writing | Briefs with several requirements | A strict judge: it starts from 10, lists each deduction and uses the whole 1-10 scale (10 is rare). Word, sentence and banned-word rules are measured by code and given to the judge as facts; each one broken caps the score 2 points lower |

Every Hard answer key is worked out a second time, independently, in `tests/test_hard_keys.py`:
brute force for the maths and the puzzles, a second implementation for the code (and a real shell
for the brace expansion), plain counting for the rest.

Scoring: every test scores 0-1. A category's score is the average of its tests; the overall
0-100 score is the average of the categories, so each category counts equally. Grades: A 90+,
B 80+, C 70+, D 60+, F below. Thinking is stripped before grading. "Max tokens per test" defaults
to 8,192 at the Standard level and 32,768 at the Hard level, so that reasoning models have room to
think (the judge gets 8,192). Each level keeps its own value. Lower it to bound the cost of a
model that rambles. The Hard default was 16,384 until 2026-10-08: at that limit five tests in
every strong-model run ended at the limit, and which five changed from run to run. Runs made at
the old limit stay in their own group of settings on the Leaderboard.
**API errors are shown but left out of the score.** Requests that hit 429, 5xx or a timeout are
retried with backoff, and a test that needed a retry says in its notes why the earlier attempt
failed. Every run is saved to SQLite.

**Every test gets one attempt.** The first reply that arrives whole is the one that is graded. A
second request is made only when no whole reply arrived although there were tokens to spare:

| What happened | What the app does | Score |
|---|---|---|
| The request failed (429, 5xx, timeout), also after the retries with backoff | Shown as an API error | not counted |
| The provider cut the reply short with tokens to spare: it ends on a comma, a colon or an open bracket, in an unclosed code block, or on a word a sentence cannot end on ("the", "and", ...) | Asks once more, automatically. A whole second reply is graded. Cut short again, or empty: marked **cut off** | not counted |
| The reply looks cut short, and comes back the same, word for word, when it is asked for again | Graded as written: a provider does not drop the end of a stream at the same word twice, so this is how the model ends its reply. The notes say so | as graded |
| Nothing arrived at all, with tokens to spare (the model thought and then wrote nothing, or the reply was simply empty) | Asks once more, automatically. A whole second reply is graded. Empty a second time: marked **empty reply**, and not asked again | 0: an empty reply, twice, is the model's answer |
| The model used up "Max tokens per test" before its answer was there (the provider says so, or the token count does) | Marked **cut off**. Not asked again | 0: the limit is part of the test |
| The reply only might be unfinished: no `Answer:` line and no full stop, a last sentence without an ending, a stream that ended without a `finish_reason` | Graded as written; the notes say it may be unfinished | as graded |

A reply at the token limit that already has its answer (an `Answer:` line, a finished code block,
or text that passes a check done by code) is graded as written. Every request counts towards the cost.

**Retry failed requests** appears on a finished run where the provider failed some tests: API
errors, and replies it cut short. It runs only those tests again (same model, judge and token
limit), replaces their results and updates the run's score, keeping what the earlier attempts
cost. A test that ran out of tokens, came back empty twice or got a wrong answer had its attempt
and is never run again.
When the model's reply arrived and only its grading failed (the judge call, or an error in a
grader), the stored reply is graded again and the model is not asked. The same goes for an Honesty
reply that the phrase check graded because the judge's call failed (its notes say so): the judge is
asked again about the stored reply; if it fails again, the phrase-check grade and its note stay. A
reply the phrase check graded because no judge was set is not touched. The button says how many of
each it will do: "Retry failed requests (22 failed, 2 to re-judge)". It is refused for a run made
with older tests, because that would mix two test sets in one score.

**First pass and final score.** A run keeps the score it had the first time it ended, before any
"Retry failed requests". The results page and the Leaderboard show it next to the final score, and
the exports have `first_pass_score` and `retries` columns. A first pass in which a whole category
got no score (every one of its tests failed at the provider, or the judge failed on all of them)
is the average of the other categories only: it is marked "no Writing score" and is left out of the
model's first-pass average and range, which then say "2 of 3 runs".

**A final score with a whole category missing** is marked the same way ("no Long context and
Writing"), on the results page and on the Leaderboard. It is left out of the model's average and
range, it is not ranked, and "Best per model" takes a complete run over it. When every run of a model
at those settings lacks a category, the model's row is listed below the ranked rows, marked, without
a rank, and it is left out of the chart.

**Scored of asked.** A test that ended in an API error, or whose reply the provider cut short, has
no score and is left out of its category's average. Where that happened the category says what its
score is over: "6 of 12 scored" on the results page and in a run's row on the Leaderboard, and
"6–7 of 12 scored" in the average of a model's runs.

A benchmark runs as a **background job on the server**, not in the page. You can switch tabs, hide
the window or close the page and it carries on; reopening the Benchmark tab reconnects to it and
replays what you missed, or shows the result if it ended in the meantime. While it runs, each test
shows how long it has been waiting for a reply (amber after a minute), and the count is also in
the page title and on the Benchmark tab, so it keeps moving while the page is hidden. **Cancel**
stops the job and keeps the tests that had finished (the run is saved as "cancelled"). One
benchmark runs at a time. If the server itself stops mid-run, the run is saved as "interrupted"
with what it had finished, instead of being left looking as if it were still running.

A running run records the process that is working on it (its ID and start time, since the system
reuses IDs). Only a run whose owner is gone is marked "interrupted", so a second server, or a
script, on the same database leaves the first one's job alone: it reports that run as in progress
elsewhere, and marks it "interrupted" itself if the owner dies. Each server still runs one
benchmark at a time; two servers can run one each.

**Arena** - one prompt, two models, streamed side by side. Vote A / Tie / B / Both bad: your votes
feed an Elo table with win / loss / tie records. "Ask judge" gets a second opinion on the same
round, before or after you vote (the A/B order sent to the judge is shuffled to avoid position
bias). The judge's verdict and its reason are shown next to your vote, marked "agrees with you" or
"disagrees with you", and are **never counted as a vote**: only your own votes move the ratings.
The record shows "Judge agrees with you: N%", over the rounds where you both gave a verdict (a tie
and "both bad" count as the same answer).

**Speed** - runs one prompt 1-10 times and reports average / min / max time to first token,
tokens/sec and total time, the success rate, and a bar chart per run.

**Leaderboard** - every saved benchmark run (or the best run per model), one level at a time
(Standard / Hard switch), sortable, with category columns, speed, cost and date, a chart of the
top models, CSV / JSON export (with a `level` column) and delete. Click a row to reopen that
run's full results. Cancelled and interrupted runs are left out unless "Include unfinished runs"
is on, and are never used for the chart or for "best per model" when a completed run exists.

**Several runs of one model.** One run says little about a category: live, two runs of the same
model an hour apart differed by 0.3 points overall and by 40 points in Instructions. When a model
has more than one finished run of the same tests at the same settings (the same fingerprint), the
Leaderboard shows them as one row: the **average**, with the **range from the lowest run to the
highest** under it, for the overall score and for every category, and a "3 runs" label. Click the
row to list the runs it is made of (each opens as before). The chart draws the average, with a band
from the lowest run to the highest. Cancelled and interrupted runs, runs with another token limit
or judge, and runs made with older tests are never part of an average. A model pinned to a
provider is a row of its own, named after the provider ("meta-llama/llama-3.1-8b-instruct ·
DeepInfra"): its runs are averaged apart from the same model's unpinned runs (and from runs pinned
to another provider), and ranked in the same table as every other model. With "Best per model only"
nothing is averaged: each row is the model's best run, labelled "best of 3".

The **Out of tokens** column counts the tests in a run that used up "Max tokens per test" before
their answer was there (cut off, scored 0); for a model with several runs it is the average per
run, with the range. Replies the provider cut short (not counted, can be retried) are shown as a
"cut short" label instead, and replies that were empty twice as an "empty" label.

Every run saves a **fingerprint of what it measured**: a hash of each test's prompt and grader
spec (which holds the answer key, unit tests, rubric and limits), the grader version
(`GRADER_VERSION` in `app/graders.py`, bumped whenever the same reply could be scored
differently) and its settings: the token limit per test, and the judge model.

- **Older tests.** A run counts as current when the grader version is the same, every test it ran
  is unchanged, and it ran every test its categories have today. The Leaderboard shows only current
  runs by default; the others (and runs saved before fingerprints existed) are hidden, counted under
  the table, and shown with an "older tests" label when "Include older tests" is on.
- **Settings.** Runs are only ranked against runs with the same token limit and the same judge: a
  model that runs out of tokens at 8,192 can pass at 16,384, and two judges do not give the same
  marks. The Leaderboard shows one group of settings at a time and names it above the table; when
  runs with other settings exist, a menu switches between the groups. It opens on the group with
  the most runs at that level (the newest of them on a tie), unless you picked a group, which is
  remembered until the next run ends. "Best per model" is best per model, level and settings. The
  judge only counts for runs it graded something in (Writing, Honesty, custom judge tests): the
  judge selected in the sidebar does nothing to a Math-only run. A **pinned provider** is not a
  setting: a pinned run has the settings, key and fingerprint of an unpinned one, and is ranked with
  it; only its row is its own (see "Several runs of one model"). Until 2026-10-18 a pin was a setting
  and a pinned run was a group of its own; such a run is moved when the app starts.
- **Judge again.** When a judge disappears (as `apodex/apodex-1.1-mini:free` did on 2026-10-18), a
  group judged by it can be judged again: set the new judge in the sidebar, show the group on the
  Leaderboard and click **Judge again…** next to its settings. The dialog says how many judge calls it
  takes and what they should cost. Then the app checks that the new judge answers, backs up the
  database (`data/backups/modellab-<date>-<time>-before-judge-again.db`), and asks the new judge about
  every judged result of the group's runs from the stored replies: Writing, custom judge tests, and
  Honesty in a run that had a judge, including a reply whose judge call failed. A call that fails is
  asked once more at the end. Then, in one transaction, it writes the new marks (each result's notes
  say "Judged again with ... before, ..."), works out each run's score and first pass again, and moves
  the runs to the new judge: judge, settings, key and fingerprint, so a new run with that judge is
  ranked and averaged with them. No model is asked again, and replies and code-graded results are
  never touched. Runs with older tests stay as they are. Nothing is written if the new judge gives no
  verdict at all. A run's `params` record each change of judge (`judged_again`).
- **Who served it.** Every reply names the provider that served it (OpenRouter in the reply itself,
  Hugging Face in a header), and the app records it for every request of a run. The results page
  shows "Served by DeepInfra", or "Served by mixed: Groq 60, DeepInfra 9, CoreWeave 2" when more
  than one provider answered; each test names its own, and a mixed run carries a "mixed providers"
  label on the Leaderboard. Runs saved before 2026-10-09 have no record and say nothing.

The exports carry the same filters and have `max_tokens`, `judge`, `pinned_provider`, `served_by`,
`tests_fingerprint` and `older_tests` columns, and `cut_off`, `out_of_tokens` and `empty` counts. They have one row per run:
the averages are worked out by the page's own request (`averages` in `/api/runs`). `tests/test_fingerprint.py` fails when the grading code changes, as a
reminder to decide whether the grader version needs a bump.

## Mock mode

`.\run.bat --mock` (or `MOCK=1`) swaps the network for a built-in fake provider so every feature
can be tried without a key. Mock runs are stored in their own database. Useful model IDs:

| Model | Behaviour |
|---|---|
| `mock/smart`, `mock/average`, `mock/weak` | Different benchmark skill and speed |
| `mock/thinker`, `mock/think-tags` | Thinking via a reasoning field / `<think>` tags |
| `mock/slow` | Slow stream - good for trying Stop |
| `mock/free:free` | Zero price (free badge) |
| `mock/per-request`, `mock/music` | Free tokens but billed per request / per audio clip ("other fees" badge, not "free") |
| `mock/flaky` | Returns 429 twice per prompt, then succeeds (shows retries) |
| `mock/mute`, `mock/mute-once` | Thinks, then writes nothing: every time (each test ends as "empty reply", 0), or only the first time it is asked something (asked once more, then graded) |
| `mock/cutoff` | Always runs out of tokens (`finish_reason=length`): cut off, scored 0, nothing to retry |
| `mock/hiccup` | The first reply to each prompt is cut short mid-sentence; the automatic retry fixes it |
| `mock/dropper` | The first two replies are cut short: the run ends with cut-off tests, and "Retry failed requests" fixes them |
| `mock/flaky-hiccup` | Returns 429 once, then a reply that is cut short: the reply is asked for twice, and the notes list each request's attempts apart |
| `mock/outage` | Returns 429 four times per prompt: the run ends with API errors, and "Retry failed requests" fixes them |
| `mock/error-401`, `-402`, `-404`, `-429`, `-500` | Always that error |
| `mock/roaming` | Served by another provider from one request to the next: the run says "Served by mixed" |
| `mock/switcher` | Ignores a pinned provider and answers from another one: the app refuses the reply |

On OpenRouter every mock model has two providers to pin, `alpha` and `beta-cloud/fp8` (twice the
price); a pin to any other provider gets OpenRouter's "no allowed providers" error.

A prompt containing `[[mock:some text]]` makes the mock reply with exactly `some text`, which is
handy for trying custom tests.

## Tests

```
.venv\Scripts\python -m pytest
```

The suite covers every grader, the code sandbox (with one escape attempt per route: files, network,
other programs, time, memory, forged results, tampering with the hook), model-URL parsing, `<think>` handling, error
mapping, retries, the localhost-only API guard, and full benchmark / arena runs against the mock
provider. It also checks that each built-in test's reference answer scores 1.0 and its wrong answer
scores less, and (when Node.js is installed) that every JavaScript module parses. The tests never
load your `.env`, so no real key is used.

## Project layout

```
app/
  main.py         FastAPI routes
  security.py     localhost-only check for /api/* (Host, Origin and Sec-Fetch-Site headers)
  providers.py    provider calls, streaming, error mapping, retries, model list cache
  mock.py         mock provider (an httpx transport, so the real code paths run)
  bench_tests.py  built-in benchmark tests: the Standard level, categories and levels
  bench_hard.py   the Hard level (UTF-8, with real Arabic text)
  bench_base.py   helpers shared by both, and the long-context filler generator
  graders.py      code-based graders
  sandbox.py      runs model-written Python against unit tests in a locked-down process
  sandbox_runner.py  the part that runs inside that process (audit hook, calls the model's function)
  benchmark.py    benchmark jobs (background runner, event replay, cancel) and judge prompts
  rejudge.py      "Judge again": a group's judged results graded by another judge, from the stored replies
  db.py           SQLite storage, scoring, Elo
  process.py      is a process still running? (the owner of a running benchmark)
  fingerprint.py  fingerprints of the tests a run used (current vs "older tests")
static/           index.html, css/app.css, js/*.js
tests/            pytest suite
data/             created on first run: modellab.db and the cached model list
```

## Good to know

- **Coding tests run model-written code on your machine**, in a sandbox with four layers
  (`app/sandbox.py`, `app/sandbox_runner.py`). It is as strong as Windows allows without Docker
  or administrator rights, and it is not a container: see the limits below.

  | Layer | What it does |
  |---|---|
  | Static check | Before anything runs: only 26 standard-library modules may be imported; `open`, `eval`, `exec`, `globals` and the attributes used to reach the interpreter's internals (`__subclasses__`, `__globals__`, `f_back` ...) are refused |
  | Separate process | Python's base interpreter in isolated mode (no site-packages, no user settings), in a throwaway temp folder, with an environment that holds no key or token |
  | Audit hook (Python) | Refuses by default. Reading: only the sandbox folder and the source of allowed library modules. No writing, deleting or renaming anywhere. No sockets, no other programs, no threads, no registry, no ctypes, no hand-made bytecode, no second interpreter, and no further built-in modules once it is set. The process never shuts down normally, because Python drops its hooks part-way through shutting down: it ends hard, or the server ends it |
  | Windows limits | A Job Object: 256 MB of memory, a CPU-time limit, no child process, outgoing network traffic throttled to 1 byte per second, no clipboard or desktop access, and the process ends with the server. Low integrity level: Windows refuses its writes to your files and registry |

  The time limit is 5 seconds per test, output is cut at 1 MB, and the expected results never enter
  the sandbox: the comparison is done outside, so code that prints "passed" scores nothing.

  **What it does not stop** (each of these is a test in `tests/test_sandbox.py`):
  - **Reading files and reaching this machine's own servers depend on the audit hook alone.** The hook
    is Python code inside the same process, which Python's own documentation says is not a security
    boundary. If code got around it (through an interpreter bug, say), it could read any file you can
    read, including `.env`, and reach servers on localhost, Model Lab included. Windows would still
    refuse writes, child processes, more memory, and nearly all traffic to other machines.
  - The network throttle does not cover name lookups, which a Windows service makes for the
    process. A lookup can carry a few dozen bytes out.
  - It can tell whether a file exists and how big it is (`os.stat` raises no audit event), and it
    can read its own folder and the allowed library modules.
  - The modules Python loads at start-up (`os`, `nt`, `winreg`, `marshal` ...) stay within reach.
    What they offer that matters raises an audit event and is refused; a function in them that
    raises no event is not covered. (`_winapi` is emptied, because it has such functions.)
  - The static check is a coarse first filter, not a barrier: `open = open` makes a builtin's name
    "the code's own". What keeps hostile code in is the process and its limits.
  - Results are compared the way a unit test's `assertEqual` would, so `8.0` passes for `8` and
    `True` for `1`.
  - It can use one CPU core fully for the 5 seconds.
  - Off Windows, the operating-system limits are `resource` limits and have not been tested.

  For models you do not trust at all, turn the Coding category off or run Model Lab in a VM.
- Benchmarks cost tokens: a full Standard run is about 40 requests, including four long-context
  prompts of roughly 5k tokens each. A full Hard run sends about 130k input tokens, nearly all of
  it in its four long-context prompts (about 30k tokens each), so the model needs a context
  window of 65k tokens or more (the prompt plus up to 32,768 tokens of reply). Free OpenRouter
  models are rate limited - use concurrency 1 or 2.
- The cost shown is the amount OpenRouter reports it charged for each request (checked against the
  account's usage counter). On Hugging Face it is the serving provider's own estimate when it sends
  one; otherwise it is worked out from the router's price list, using the price of the provider
  that actually served the request (the router names it in a response header). Both are marked
  with `~`. A dash means that provider lists no price. A benchmark run's total is the tested model's
  requests plus the judge's calls; the two parts are shown separately under the total, on each
  judged test and in the CSV export (`model_cost_usd`, `judge_cost_usd`). "Test connection" is not
  included.
- The server listens on `127.0.0.1` only, and `/api/*` only answers requests whose `Host` (and
  `Origin`, when the browser sends one) is `localhost` or `127.0.0.1` on the server's own port.
  A request the browser labels `Sec-Fetch-Site: cross-site` or `same-site` is refused as well,
  which covers cross-site GETs that carry no `Origin`. Anything else gets a 403, so another
  website open in your browser cannot use the keys in `.env` or read your results (cross-site
  requests, DNS rebinding). Open the app as `http://localhost:<port>` or
  `http://127.0.0.1:<port>`; other names for the same machine are refused.
- To delete all saved results, stop the server and remove `data/`.
- The HTTP client connects over IPv4 first and falls back to the system default automatically,
  because some networks have an IPv6 route to `router.huggingface.co` that stalls.
