# ChatKit Starter

Minimal Vite + React UI paired with a FastAPI backend that forwards chat
requests to OpenAI through the ChatKit server library.

## Quick start

```bash
npm install
npm run dev
```

What happens:

- `npm run dev` starts the FastAPI backend on `127.0.0.1:8000` and the Vite
  frontend on `127.0.0.1:3000` with a proxy at `/chatkit`.

## Required environment

- `OPENAI_API_KEY` (backend)
- `VITE_CHATKIT_API_URL` (optional, defaults to `/chatkit`)
- `VITE_CHATKIT_API_DOMAIN_KEY` (optional, defaults to `domain_pk_localhost_dev`)
- `INVESTIGATION_ENGINE_ENABLED` (optional, defaults to `1`)

Set `OPENAI_API_KEY` in your shell or in `.env.local` at the repo root before
running the backend. Register a production domain key in the OpenAI dashboard
and set `VITE_CHATKIT_API_DOMAIN_KEY` when deploying.

## Customize

- Update UI and connection settings in `frontend/src/lib/config.ts`.
- Adjust layout in `frontend/src/components/ChatKitPanel.tsx`.
- `backend/app/server.py` uses a process-local in-memory store so old threads
  and messages are not retained after restart.

## DS Chat Investigation Runtime

- The backend now uses a knowledge-driven multi-agent pipeline:
  - Multi-Agent Orchestrator
  - Investigation Operator Agent
  - Codebase Explanation Agent
- Local KB files are editable and used by investigation planning.
- Per-turn datasets are materialized under `backend/.work/sessions/...` and cleaned up after each response turn.
- Autonomous deep EDA is available for table prompts (for example: `can you do a EDA of the table combined_audit`).
- `threevictors` must be available in the backend Python runtime for Redshift/MySQL/S3 access.
- Basic live connectivity smoke test script: `backend/scripts/smoke_threevictors.py --profile 3VDEV`

## DS Chat Investigation Verification

One command to run everything (unit tests + connectivity + E2E agent smokes):

```bash
npm run backend:verify
```

Available shortcuts:

- Unit tests only:

```bash
npm run backend:test
```

- Live smoke tests only (requires `OPENAI_API_KEY` and granted profile access):

```bash
npm run backend:smoke
```

Direct script usage (more control):

```bash
backend/scripts/verify_investigation.sh --help
backend/scripts/verify_investigation.sh --profile 3VDEV --model gpt-5.5-mini --max-turns 40
backend/scripts/verify_investigation.sh --scenarios top_site_issues,market_anomalies_distribution
backend/scripts/verify_investigation.sh --profile 3VDEV --model gpt-5.5 --skip-unit
```

Direct interactive CLI chat (no frontend required):

```bash
cd backend
.venv/bin/python tests/cli_chat.py --model gpt-5-mini
```

CLI notes:

- Tool calls are logged in the terminal by default (`[tool-call ...]`, `[tool-output ...]`).
- Use `--no-log-tools` to disable tool-call logging.
- Built-in commands: `/thread`, `/reset`, `/quit`.

E2E smoke reports with full model output and debug steps are written under:

- `backend/.runtime/e2e_reports/*.md`
- `backend/.runtime/e2e_reports/*.json`

## Runtime Flags

- Default mode (`INVESTIGATION_ENGINE_ENABLED=1`) routes internal data tasks to the
  unified `Investigation Operator Agent`.
- Knowledge sources are local and editable:
  - `backend/app/investigation/knowledge/tables.md`
  - `backend/app/investigation/knowledge/common_codes.json`
  - `backend/app/investigation/knowledge/task_recipes.json`
  - `backend/app/investigation/knowledge/sql_best_practices.md`

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │  Browser  :3000  (React + Vite)                                                           │
 │  ┌───────────────────────────────────────┐   ┌──────────────────────────────────────┐    │
 │  │  ChatKitPanel.tsx                     │   │  SessionStateBar.tsx                 │    │
 │  │  · model selector (gpt-5.3 / mini)   │   │  polls GET /session/{thread_id}      │    │
 │  │  · message input + file attach       │   │  shows shell cwd + idle secs         │    │
 │  │  · SSE stream renderer               │   └──────────────────────────────────────┘    │
 │  │  · Card widget display (plots/diffs) │                                                │
 │  └──────────────────┬────────────────── ┘                                                │
 └─────────────────────┼────────────────────────────────────────────────────────────────────┘
                       │  POST /chatkit (SSE)   PUT/GET /chatkit/uploads/{id}
                       ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │  FastAPI :8000  ·  StarterChatServer  (server.py)                                         │
 │  · InMemoryStore  (active threads only; no retained history records)                     │
 │  · load last 50 items → agent_input[]  ·  pick model from inference_options              │
 │                                    │ build_agent(model)   ds_agent.py                    │
 │                                    ▼                                                     │
 │  ┌─────────────────────────────────────────────────────────────────────────────────────┐ │
 │  │  DS Chat Agent  (OpenAI Agents SDK · Runner.run_streamed · max 50 turns)            │ │
 │  │                                                                                     │ │
 │  │  system prompt =  coding identity + tool guide + git repos + AWS guide             │ │
 │  │                 + investigation_agent instructions (assembled fresh each request):  │ │
 │  │                   · 405-table live metadata (tier, freshness, partitions)          │ │
 │  │                   · provider/site/customer code aliases                            │ │
 │  │                   · 18 investigation patterns + SQL rules                          │ │
 │  │                                                                                     │ │
 │  │  ┌──── SHELL TOOLS (shell_tools.py) ──────────────────┐  ┌─ INVESTIGATION TOOLS ─┐ │ │
 │  │  │                                                     │  │  (investigation_       │ │ │
 │  │  │  bash ──────────────────────────────────────────┐  │  │   tools_core())        │ │ │
 │  │  │  · persistent PTY per thread (shell_session.py) │  │  │                        │ │ │
 │  │  │  · cd / export / bg jobs persist across calls   │  │  │  execute_sql           │ │ │
 │  │  │  · streams live chunks → Terminal Card widget   │  │  │  · SqlGuard (read-only)│ │ │
 │  │  │  · PROMPT_COMMAND sentinel tracks cwd           │  │  │  · PartitionGuard      │ │ │
 │  │  │  · auto-evicted after 1hr idle                  │  │  │  · LIMIT ≤ 120k        │ │ │
 │  │  │                                                 │  │  │  · routes by prefix    │ │ │
 │  │  │  read_file  · cat -n (~/git/-relative)          │  │  │    → DatasourceRegistry│ │ │
 │  │  │  list_dir   · glob filter, up to 200 entries    │  │  │                        │ │ │
 │  │  │  edit_file  · exact-match replace, diff Card    │  │  │  fetch_s3              │ │ │
 │  │  │  git        · any subcommand (push --force       │  │  │  · list prefix → read  │ │ │
 │  │  │               blocked); cwd ~/git/               │  │  │  · CSV/Parquet/JSONL   │ │ │
 │  │  │  fetch_url  · HTTP GET, HTML stripped            │  │  │  · up to 30 files      │ │ │
 │  │  │  run_parallel · up to 8 cmds concurrently        │  │  │                        │ │ │
 │  │  │                                                  │  │  │  inspect_table         │ │ │
 │  │  │  plan_task  · planner sub-agent (gpt-5-mini)     │  │  │  · svv_columns / DESC  │ │ │
 │  │  │               generates numbered execution plan  │  │  │  · masked sample row   │ │ │
 │  │  │  web_search · WebSearchTool (built-in)           │  │  │                        │ │ │
 │  │  └──────────┬──────────────────────────────────────┘  │  │  search_kb             │ │ │
 │  │             │                                          │  │  · FTS knowledge.sqlite│ │ │
 │  │             │                                          │  │  · 27 docs + tables.md │ │ │
 │  │             │                                          │  │                        │ │ │
 │  │             │                                          │  │  resolve_codes         │ │ │
 │  │             │                                          │  │  · common_codes.json   │ │ │
 │  │             │                                          │  │  · MySQL priceeye.*    │ │ │
 │  │             │                                          │  └────────────┬───────────┘ │ │
 │  └─────────────┼──────────────────────────────────────────────────────── │─────────────┘ │
 │                │                                                          │               │
 │                ▼                                                          ▼               │
 │  ┌─────────────────────────────────┐        ┌──────────────────────────────────────────┐ │
 │  │  PersistentShell  (PTY / bash)  │        │  DatasourceRegistry                      │ │
 │  │  shell_session.py               │        │  ensure_credentials():                   │ │
 │  │  · one PTY per thread_id        │        │    zsh -lc "assume 3VDEV; env -0"        │ │
 │  │  · SENTINEL marks cmd end + cwd │        │    → AWS_* injected into env (once)      │ │
 │  │  · 1hr idle TTL, auto-evict     │        │                                          │ │
 │  └──────────────┬──────────────────┘        │  routing:                                │ │
 │                 │                           │  billing_db.* / prod.monitoring.*  ──▶   │ │
 │                 │                           │    CoreRedshiftReader                    │ │
 │                 │                           │  prod.analytics.* / (default)      ──▶   │ │
 │                 │                           │    AnalyticsRedshiftReader               │ │
 │                 │                           │  priceeye.* / taxregression.*      ──▶   │ │
 │                 │                           │    PriceEyeMySQLReader                   │ │
 │                 │                           └──────────┬──────────────┬────────────────┘ │
 └─────────────────┼────────────────────────────────────── ┼─────────────┼──────────────────┘
                   │                                        │             │
                   ▼                                        ▼             ▼
 ┌─────────────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐  ┌──────────┐
 │  Local machine / EC2        │  │  Redshift Analytics  │  │  Redshift Core    │  │  MySQL   │
 │                             │  │  prod.analytics.*    │  │  prod.monitoring.*│  │  price-  │
 │  ~/git/  (source repos)     │  │  prod.common_output.*│  │  prod.site_       │  │  eye.*   │
 │  · ds-priceeye-analytics    │  │  prod.tax_reg.*  etc.│  │  metrics.*  etc.  │  │  taxreg.*│
 │  · ds-internal-monitoring   │  └─────────────────────┘  └───────────────────┘  └──────────┘
 │  · ds-priceeye-data-        │
 │    collection               │  ┌──────────────────────────────────────────────────────────┐
 │  · priceeye-v2  etc.        │  │  S3  (fetch_s3 only · threevictors s3_util.S3Util)       │
 │                             │  │  anomaly-datasets · derived-common-output                │
 │  /tmp/  (scripts, plots)    │  │  collection-anomalies · competitive-position  etc.       │
 │  aws CLI  (read-only)       │  └──────────────────────────────────────────────────────────┘
 └─────────────────────────────┘
```
