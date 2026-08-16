# ARCHIE

An agentic content workflow that turns a one-line campaign brief into reviewed,
safety-screened ad copy — and knows how to recover when something goes wrong.

Built with Python, Pydantic and Streamlit, running Llama 3.3 70B on
[Groq](https://console.groq.com).

---

## What makes it more than an API wrapper

Most LLM demos are a single call wrapped in a UI. The interesting problems start
when a call fails, when the model returns something malformed, or when the
output simply isn't good enough. ARCHIE is built around those three cases.

**Self-correction.** Stage 5 reviews the copy and scores it out of ten. A
rejected draft is not retried blindly — the reviewer's specific complaint is fed
back into the copywriting prompt, so the rewrite targets the stated problem.
After three rejections the run stops and escalates to a human rather than
looping forever.

**Resumability.** State is persisted after every stage. If the process dies
during stage four, the next invocation reloads from disk and starts at stage
four. The first three stages are never recomputed.

**Idempotency.** Every stage is guarded by a completion check, so a duplicate
request against a finished workflow costs nothing. Retrying is free rather than
expensive.

**Typed boundaries.** Every model response is validated against a Pydantic
schema before it is stored. A malformed response is retried, then fails loudly —
it never propagates through the pipeline as an untyped dictionary.

---

## The pipeline

| Stage | Name | Produces |
|-------|------|----------|
| 1 | Brief extraction | A structured `StrategicBrief` from free text |
| 2 | Creative angles | Three scored `AngleOption`s; the highest wins |
| 3 | Content drafts | Ad copy written against the winning angle |
| 4 | Safety check | Screening for unverifiable claims and policy risk |
| 5 | Quality assurance | A scored verdict; rejection triggers a rewrite |
| 6 | Final package | A Markdown handover with visual directions |

A run ends in one of five states: `COMPLETED`, `FAILED` (resumable),
`FLAGGED_FOR_HUMAN_REVIEW` (QA budget exhausted), `FLAGGED_SAFETY_RISK`
(halted before review), or `RUNNING`.

---

## Architecture

```
app.py / main.py        Streamlit UI and CLI — two front ends, one core
      │
orchestrator.py         Stage sequencing, the QA loop, resume logic
      │
      ├── llm_service.py    Groq client: retries, JSON parsing, schema validation
      ├── statestore.py     Persistence behind a swappable interface
      └── schemas.py        Pydantic models; WorkflowState is the unit of resume
            │
config.py                 Settings from environment or .env
logging_config.py         Structured logging shared by every entry point
```

`llm_service` imports nothing from Streamlit, so the core is usable from the
CLI, from tests, and from any future web framework. `StateStore` exposes only
save / load / list / delete, so swapping the JSON files for SQLite or Postgres
would not touch the orchestrator.

---

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your Groq API key
```

Web UI:

```bash
streamlit run app.py
```

Command line:

```bash
python main.py "A budgeting app for university students" --id demo
python main.py "..." --id demo --json     # full state as JSON
```

Reuse an id to resume an interrupted run:

```
Status   : FAILED
  [x] Step 1: Brief extraction
  [ ] Step 2: Creative angles
Error    : Could not obtain a valid CreativeAngles response after 3 attempts
Resume with: --id crashed
```

Docker:

```bash
docker build -t archie . && docker run -p 8501:8501 --env-file .env archie
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

28 tests, no network access — a fake client stands in for the Groq SDK. They
cover the properties that matter rather than line coverage:

- a completed workflow re-run makes zero model calls
- a stage failure preserves earlier results and resumes from that stage
- state survives a process restart with a fresh orchestrator and client
- a QA rejection reaches the next drafting call as explicit feedback
- repeated rejection escalates instead of looping
- unsafe copy halts before review and packaging
- malformed JSON and schema violations are retried, then raised as `LLMError`
- identifiers that could escape the state directory are rejected

CI runs `ruff` and `pytest` on Python 3.10 and 3.12 for every push.

---

## Configuration

All settings load from the environment or `.env`; nothing reads `os.environ`
directly outside `config.py`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Required |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model id |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `LLM_MAX_RETRIES` | `3` | Attempts before a stage fails |
| `STATE_DIR` | `states` | Where runs are persisted |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Known limits

State lives in JSON files on local disk, so concurrent writers to the same
workflow id would race — SQLite is the natural next step. Prompts are embedded
in `llm_service.py`; moving them to versioned YAML would make them reviewable
without a code change. There is no per-run token or cost accounting yet.
