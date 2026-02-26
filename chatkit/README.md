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
- `NEXT_GEN_INVESTIGATION` (optional, defaults to `1`)

Set `OPENAI_API_KEY` in your shell or in `.env.local` at the repo root before
running the backend. Register a production domain key in the OpenAI dashboard
and set `VITE_CHATKIT_API_DOMAIN_KEY` when deploying.

## Customize

- Update UI and connection settings in `frontend/src/lib/config.ts`.
- Adjust layout in `frontend/src/components/ChatKitPanel.tsx`.
- Swap the in-memory store in `backend/app/server.py` for persistence.

## DS Chat Next-Gen Notes

- The backend now uses a knowledge-driven multi-agent pipeline:
  - Orchestrator
  - Knowledge Planner
  - Data Access
  - Analysis
  - Synthesis
- Local KB files live in `backend/knowledgebase/` and are editable on demand.
- Query execution is partition-enforced from KB table metadata.
- Per-turn datasets are materialized under `backend/.runtime/workspaces/...` and cleaned up after each response turn.
- `threevictors` must be available in the backend Python runtime for Redshift/MySQL/S3 access.
- Basic live connectivity smoke test script: `backend/scripts/smoke_threevictors.py --profile 3VDEV`

## DS Chat Next-Gen Verification

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
backend/scripts/verify_nextgen.sh --help
backend/scripts/verify_nextgen.sh --profile 3VDEV --max-turns 40
backend/scripts/verify_nextgen.sh --scenarios top_site_issues,market_anomalies_distribution
```

E2E smoke reports with full model output and debug steps are written under:

- `backend/.runtime/smoke_reports/*.md`
- `backend/.runtime/smoke_reports/*.json`

## Next-gen investigation runtime

- Default mode (`NEXT_GEN_INVESTIGATION=1`) routes internal data tasks to the
  unified `Investigation Operator Agent`.
- Legacy split monitoring/anomalies agents can be forced with
  `NEXT_GEN_INVESTIGATION=0` during migration.
- Knowledge sources are local and editable:
  - `tables.md`
  - `backend/app/investigation/common_codes.json`
  - `backend/app/investigation/task_recipes.json`
  - `backend/app/investigation/sql_best_practices.md`
