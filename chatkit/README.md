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

## Repository Layout

```
ds-chat-2/
├── .gitignore
├── .claude/
│   └── settings.local.json                         # Claude Code local settings
├── .codex/
│   └── environments/
│       └── environment.toml                        # OpenAI Codex cloud agent config (runs assume 3VDEV; npm run dev)
├── package.json                                    # Root workspace – delegates to chatkit/
├── package-lock.json
├── tables.md                                       # Root-level table quick reference
│
└── chatkit/                                        # Main application workspace
    ├── README.md                                   # ← this file
    ├── package.json                                # npm scripts: dev, backend:verify, backend:smoke, etc.
    ├── package-lock.json
    ├── .env.example                                # Required env vars (OPENAI_API_KEY, VITE_CHATKIT_API_URL, etc.)
    ├── .gitignore
    │
    ├── backend/                                    # Python FastAPI service (:8000)
    │   ├── pyproject.toml                          # Package config; deps: fastapi, openai-agents, openai-chatkit, threevictors
    │   ├── .gitignore
    │   ├── docs/
    │   │   └── investigation-runtime.md            # Internal doc: investigation runtime architecture
    │   │
    │   ├── scripts/                                # Maintenance & smoke scripts
    │   │   ├── run.sh                              # Dev launcher (uvicorn + hot-reload)
    │   │   ├── run-production.sh                   # Production launcher
    │   │   ├── verify_investigation.sh             # Full verify: unit tests + connectivity + E2E smokes
    │   │   ├── refresh_table_metadata.py           # Sweeps Redshift/MySQL/S3 → common_table_live_metadata.json
    │   │   ├── refresh_aws_infra.py                # Regenerates aws_infrastructure.md KB doc on demand
    │   │   ├── enrich_common_tables.py             # Adds tier/freshness metadata to common tables JSON
    │   │   ├── smoke_threevictors.py               # Live connectivity smoke (Redshift + MySQL + S3)
    │   │   └── smoke_e2e.py                        # End-to-end agent smoke runner
    │   │
    │   ├── tests/                                  # Test suite
    │   │   ├── cli_chat.py                         # Interactive CLI chat (no frontend needed)
    │   │   ├── run_e2e_smoke.py                    # Runs e2e_investigation_cases.json against live agent
    │   │   ├── e2e_investigation_cases.json        # Declarative E2E test cases (site issues, anomalies, billing, etc.)
    │   │   ├── test_investigation_runtime.py       # Unit tests: executor, workspace, catalog, datasources
    │   │   ├── test_shell_tools.py                 # Unit tests: shell_tools sandbox
    │   │   └── test_table_sweep.py                 # Unit tests: metadata sweep logic
    │   │
    │   └── app/                                    # Application source
    │       ├── __init__.py
    │       ├── main.py                             # FastAPI entry: chatkit_endpoint(), mounts ChatKit server
    │       ├── server.py                           # StarterChatServer: SQLite thread store, stream_agent_response()
    │       ├── persistent_store.py                 # SQLite-backed thread/message persistence
    │       ├── attachment_store.py                 # File attachment upload/retrieval
    │       ├── chatkit.sqlite                      # SQLite DB (thread + message store, gitignored)
    │       │
    │       ├── agents/                             # Agent definitions (OpenAI Agents SDK)
    │       │   ├── __init__.py
    │       │   ├── investigation_agent.py          # Main agent: _build_instructions() with KB injection, 8 tools
    │       │   └── ds_agent.py                     # DS agent: AWS/infra-aware variant with lean prompt
    │       │
    │       ├── tools/                              # Tool implementations exposed to agents
    │       │   ├── __init__.py
    │       │   ├── investigation_tools.py          # 8 tools: extract_sql_to_dataset, extract_s3_to_dataset,
    │       │   │                                   #   run_python, resolve_codes, search_kb,
    │       │   │                                   #   inspect_table_metadata, run_table_eda, publish_image
    │       │   └── shell_tools.py                  # Sandboxed shell execution tools
    │       │
    │       └── investigation/                      # Core investigation pipeline
    │           ├── __init__.py
    │           ├── catalog.py                      # Dataset catalog: register/lookup artifacts by dataset_id
    │           ├── datasources.py                  # Connectors: Redshift (analytics+core), MySQL, S3 via threevictors
    │           ├── entity_resolution.py            # Resolves "JetBlue"→B6, "American"→AA via common_codes.json
    │           ├── executor.py                     # SQL executor (SqlGuard read-only, PartitionGuard warnings, LIMIT clamp)
    │           ├── runtime.py                      # Investigation pipeline runtime: orchestrates tools per turn
    │           ├── shell_session.py                # Persistent shell session for run_python sandbox
    │           ├── tools.py                        # Core tool logic (lower-level, called by investigation_tools.py)
    │           ├── workspace.py                    # Per-thread per-run artifact storage at .work/sessions/{thread}/{run}/
    │           │
    │           └── knowledge/                      # Static knowledge base (editable, loaded at request time)
    │               ├── tables.md                   # Human-readable table reference
    │               ├── sql_best_practices.md       # SQL safety rules, partition filter requirements
    │               ├── common_codes.json           # Provider/site/customer code→name aliases
    │               ├── common_table_live_metadata.json  # 405-table live metadata: tier, max_sales_date,
    │               │                               #   partitions, columns — regenerated by refresh_table_metadata.py
    │               └── docs/                       # System documentation KB (27 files, ~15 KB each)
    │                   ├── README.md               # KB index
    │                   ├── overview.md             # PriceEye platform overview
    │                   ├── priceeye_system.md      # 15KB comprehensive doc: 18 investigation patterns, process→table map
    │                   ├── aws_infrastructure.md   # 3VDEV+3VPROD AWS inventory (Redshift, S3, Glue, Lambda, EMR)
    │                   ├── priceeye-v2.md          # priceeye-v2 collection engine
    │                   ├── priceeye-analytics.md   # Analytics pipeline overview
    │                   ├── priceeye-monitoring.md  # Monitoring pipeline
    │                   ├── priceeye-scheduling.md  # Scheduling system
    │                   ├── priceeye-providers.md   # Provider configs and site codes
    │                   ├── priceeye-customers.md   # Customer configs
    │                   ├── priceeye-applications.md
    │                   ├── priceeye-api.md
    │                   ├── priceeye-vacations.md
    │                   ├── ds-priceeye-analytics.md    # ds-priceeye-analytics repo (anomaly models)
    │                   ├── ds-priceeye-data-collection.md  # ds-priceeye-data-collection repo (site metrics, ingest TTL)
    │                   ├── ds-priceeye-enrichment.md   # ds-priceeye-enrichment repo (tax regression)
    │                   ├── ds-customer-monitoring.md   # ds-customer-monitoring repo (billing)
    │                   ├── ds-internal-monitoring.md   # ds-internal-monitoring repo (combined_audit)
    │                   ├── ds-threevictors.md          # threevictors connector library
    │                   ├── 3v-build-deploy.md          # Build and deploy patterns
    │                   ├── ingest.md                   # Ingest pipeline
    │                   ├── ingest-cache.md             # Cache ingest
    │                   ├── ingest-sources.md           # Ingest source configs
    │                   ├── federated_schemas.md        # Redshift Spectrum federated schemas
    │                   ├── partition-creator.md        # Partition creator Lambda
    │                   ├── event-launcher.md           # Event launcher
    │                   ├── emr.md                      # EMR Spark cluster config
    │                   └── spark-v3.md                 # Spark v3 job patterns
    │
    └── frontend/                                   # React+TypeScript UI (:3000)
        ├── package.json
        ├── package-lock.json
        ├── index.html
        ├── vite.config.ts                          # Vite config: proxies /chatkit → :8000
        ├── tsconfig.json
        ├── tsconfig.node.json
        ├── postcss.config.mjs
        ├── eslint.config.mjs
        ├── .gitignore
        ├── public/
        │   └── favicon.ico
        └── src/
            ├── main.tsx                            # React entry point
            ├── App.tsx                             # Root component
            ├── index.css                           # Tailwind base styles
            ├── vite-env.d.ts
            ├── lib/
            │   └── config.ts                       # VITE_CHATKIT_API_URL, VITE_CHATKIT_API_DOMAIN_KEY
            └── components/
                ├── ChatKitPanel.tsx                # Main chat UI: model selector, widget handlers, SSE streaming
                └── SessionStateBar.tsx             # Session status bar (thread ID, model, connection state)
```
