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

## Architecture

### Request / Response Flow

```
Browser (React)
│  ChatKitPanel.tsx — model selector, message input, widget renderer
│  SessionStateBar.tsx — polls GET /chatkit/session/{thread_id} for shell state
│
│  POST /chatkit   (SSE text/event-stream)
▼
FastAPI  app/main.py
│  chatkit_endpoint()  →  StarterChatServer.process(payload)
│                                │
│                                ├─ load thread items from SQLite (last 50)
│                                ├─ DSChatThreadItemConverter → agent_input[]
│                                ├─ read model from inference_options
│                                │
│                                └─ build_agent(model)          ← ds_agent.py
│                                       │  _build_instructions()
│                                       │    loads at request time:
│                                       │      common_table_live_metadata.json  (405 tables, tier+freshness)
│                                       │      common_codes.json                (provider/site/customer aliases)
│                                       │      priceeye_system.md               (18 investigation patterns)
│                                       │      sql_best_practices.md
│                                       │      today's date, current sales_date
│                                       │
│                                       └─ Agent(model, instructions, tools=[7])
│
│  Runner.run_streamed(agent, agent_input, max_turns=50)
│    ↓ agent reasons, calls tools, reasons again … (up to 50 turns)
│  stream_agent_response() → SSE chunks → StreamingResponse → browser
│
│  [post-turn cleanup]
│    cleanup_thread_workspace(thread_id, mode="ephemeral_manifest")
│      → deletes .work/sessions/{thread_id}/{run_id}/*.parquet  (datasets)
│      → retains manifest.json
│    close_session(thread_id)      → tears down persistent shell
│    _maybe_set_thread_title()     → gpt-5-mini generates thread title → SQLite
│
▼
Browser renders:  text • ProgressUpdate events • Card widgets (charts, tables)
```

---

### Agent Tool Layer

```
Investigation Agent  (OpenAI Agents SDK)
│  System prompt: ~8 KB domain instructions + KB injected at runtime
│  Model: gpt-5.3 (default) | gpt-5-mini (fast)  — user-selectable
│
├─[1] execute_sql(query, datasource?)
│       │  SqlGuard.validate()      → blocks INSERT/UPDATE/DELETE/DROP; enforces LIMIT ≤ 120k
│       │  PartitionGuard.check()   → warns if sales_date / customer filter missing
│       │  datasource_for_table()   → routes by table prefix:
│       │      prod.monitoring.*  / local.*  / billing_db.*  / collection_optimizer.*
│       │          → redshift_core   (CoreRedshiftReader via threevictors)
│       │      priceeye.*  / taxregression.*
│       │          → mysql_priceeye  (PriceEyeMySQLReader via threevictors)
│       │      everything else (prod.analytics.*, prod.common_output.*, prod.tax_reg.*, …)
│       │          → redshift_analytics  (AnalyticsRedshiftReader via threevictors)
│       │  result → DataFrame → WorkspaceManager.save_dataset()
│       │      .work/sessions/{thread_id}/{run_id}/{dataset_id}.parquet
│       └─ returns: dataset_id, row_count, columns, preview (20 rows), partition_warnings
│
├─[2] fetch_s3(bucket, key_or_prefix)
│       │  DatasourceRegistry.fetch_s3_data()
│       │    → ensure_credentials() (see credential flow below)
│       │    → s3_util.S3Util (threevictors) lists objects under prefix
│       │    → reads CSV / Parquet / JSONL / JSON; auto-detects delimiter
│       │    → pd.concat() all files (up to 30) into one DataFrame
│       │  result → WorkspaceManager.save_dataset()
│       └─ returns: dataset_id, row_count, columns, preview, s3_keys[]
│
├─[3] run_python(code)
│       │  OperatorRuntime.run_python()
│       │    sandboxed exec scope provides:
│       │      load_dataset(dataset_id)  → loads .parquet from workspace
│       │      save_dataframe(df, name)  → writes new .parquet to workspace
│       │      save_plot(fig, name)      → saves matplotlib fig to /tmp
│       │      pd, np, plt, sns, json, Path
│       │    no os.system / subprocess / shutil access
│       │  auto-publish: scans result for image paths → _publish_image_widget()
│       │    → reads file → saves as Attachment → streams Card widget to browser
│       │        Card contains: Title, Image (base64 inline), Caption,
│       │                       "Open Full Size" button, "Download PNG" button
│       └─ returns: stdout, created_datasets[], created_analyses[], published_images[]
│
├─[4] search_kb(query)
│       │  KnowledgeBase.retrieve()  →  knowledge.sqlite  (FTS index)
│       │    indexes: tables.md, sql_best_practices.md, docs/*.md (27 files)
│       └─ returns: candidate_tables[], table_hints (partition info)
│
├─[5] inspect_table(table_name, datasource?)
│       │  DatasourceRegistry.inspect_table_metadata()
│       │    Redshift: SELECT FROM svv_columns WHERE table_schema=... AND table_name=...
│       │    MySQL:    DESCRIBE {schema}.{table}
│       │  fetches 1 masked sample row  (first 2 + last 2 chars of each value)
│       │  upserts result into knowledge.sqlite
│       └─ returns: columns[], partitions[], sample_row_masked, tier
│
├─[6] resolve_codes(text)
│       │  EntityResolver.resolve()
│       │    1. LocalCodeCatalog  →  common_codes.json  (static aliases)
│       │    2. MySQL fallback    →  priceeye.provider / priceeye.site / priceeye.customer
│       └─ returns: providers[], sites[], customers[], unknown_tokens[]
│
└─[7] browse_repo_files(path_or_glob)
        │  reads files under ~/git/  (source repos + documentation)
        │  supports glob patterns:  'ds-priceeye-analytics/src/**/*.py'
        └─ returns: count, files[{path, size, content (first 8 KB), truncated}]
```

---

### Data & Credential Layer

```
DatasourceRegistry  (singleton, initialized once per process)
│
├─ ensure_credentials()   [called before every DB/S3 access]
│    runs: zsh -lc "assume 3VDEV >/dev/null 2>&1; env -0"
│    parses output → sets AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
│                          AWS_SESSION_TOKEN / AWS_REGION in os.environ
│    fallback: granted credential-process --profile 3VDEV --auto-login
│    cached: _creds_ready flag (bootstraps once per process lifetime)
│
├─ redshift_analytics  →  AnalyticsRedshiftReader  (threevictors RedshiftConnector)
│    properties: database-analytics-redshift-serverless-reader.properties
│    tables: prod.analytics.*, prod.common_output.*, prod.data_lakes.*,
│            prod.flight_summary.*, prod.midt_external.*, prod.federated_*,
│            prod.billing.*, prod.tax_reg.*, prod.priceeye_output.*
│
├─ redshift_core  →  CoreRedshiftReader  (threevictors RedshiftConnector)
│    properties: database-core-redshift-serverless-reader.properties
│    tables: prod.monitoring.*, prod.site_metrics.*, prod.scheduling.*,
│            billing_db.* (Glue/Spectrum external schema)
│
├─ mysql_priceeye  →  PriceEyeMySQLReader  (threevictors MySQLConnector)
│    properties: database-priceeye-reader.properties
│    tables: priceeye.*, sales_poc.*, taxregression.*
│
└─ S3  →  s3_util.S3Util  (threevictors)
     buckets (examples):
       s3-atp-3victors-3vdev-use1-anomaly-datasets
       s3-atp-3victors-3vdev-use1-derived-common-output
       s3-atp-3victors-3vdev-use1-collection-anomalies
       s3-atp-3victors-3vprod-use1-pe-common-output
       s3-atp-3victors-3vdev-use1-competitive-position

WorkspaceManager  →  .work/sessions/{thread_id}/{run_id}/
│  {dataset_id}.parquet   — materialized query / S3 result (deleted post-turn)
│  manifest.json          — query log, source log, event log  (retained)
└─ /tmp/ds-chat-investigation/{thread_id}/{run_id}/activity.jsonl  — mirror log

SQLiteStore  →  app/chatkit.sqlite
  threads table   — thread_id, title, created_at
  items table     — message history (user + assistant turns)
  attachments     — attachment metadata; payloads on LocalDiskAttachmentStore

KnowledgeBase  →  .work/knowledge/knowledge.sqlite
  FTS index over: tables.md, sql_best_practices.md, docs/*.md
  table_metadata  — upserted by inspect_table() calls
```

---

### Repository Layout

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
