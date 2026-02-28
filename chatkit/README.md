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
- Swap the in-memory store in `backend/app/server.py` for persistence.

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
backend/scripts/verify_investigation.sh --profile 3VDEV --max-turns 40
backend/scripts/verify_investigation.sh --scenarios top_site_issues,market_anomalies_distribution
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

- `backend/.runtime/smoke_reports/*.md`
- `backend/.runtime/smoke_reports/*.json`

## Runtime Flags

- Default mode (`INVESTIGATION_ENGINE_ENABLED=1`) routes internal data tasks to the
  unified `Investigation Operator Agent`.
- Knowledge sources are local and editable:
  - `backend/app/investigation/knowledge/tables.md`
  - `backend/app/investigation/knowledge/common_codes.json`
  - `backend/app/investigation/knowledge/task_recipes.json`
  - `backend/app/investigation/knowledge/sql_best_practices.md`
