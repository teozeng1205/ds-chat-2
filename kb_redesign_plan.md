# DS Chat Knowledge Base — Redesign Plan

A proposal to replace the current "KB V2" with a simpler, more powerful design, and to fix how table metadata (`tables.md` and friends) is maintained.

> **Status (2026-06-08):** Diagnosis re-verified against the live code and validated against current (2026) retrieval research. Locked decisions: **local embeddings** (sentence-transformers, no API). Open: build order (catalog generator vs. hybrid KB). This revision adds the concrete rot found in the table-metadata path and a feasibility preflight.

---

## 1. What's wrong with the current design

The current KB (`app/investigation/kb/`) is a SQLite store with four tables — `items`, `chunks`, `edges`, `tasks` — plus a hand-rolled retriever. It works, but the design has aged badly in five concrete ways.

**Retrieval is a hand-written lexical scorer, not real search.** `store.search_chunks()` loads *every* chunk via a full `chunks JOIN items` table scan on every query, then loops in Python adding magic weights (`+2.0` per exact token, `+0.7` per substring, `×` source-weight, `×` confidence). There is no BM25, no embeddings, and therefore no semantic matching — a query for "carrier revenue anomalies" can't match a chunk that says "airline yield outliers." It also doesn't scale: retrieval is O(n) over the whole corpus per call.

**It's overfit to the eval set.** The scorer and retriever are littered with hardcoded query rules: `+6.0` if the query contains "priceeye" and "work" and the item is `doc_overview:priceeye`; string-injecting `prod.monitoring.*` table names when the query mentions "monitoring"; a per-schema block of hand-written `useful_lines` for `prod.monitoring`; e2e-case-specific `preferred_tables`. These make the benchmarks pass but don't generalize and are a maintenance trap — every new question risks needing another `if`.

**The output contract is over-engineered.** `SearchResult` has ~14 fields (`items`, `verified_items`, `hints`, `tables`, `lineage`, `tool_plan`, `citations`, `authority_trace`, `retrieval_trace`, `source_policy`, `verification_required`, …). The agent consuming it is a strong model that mostly needs "here are the relevant facts and where they came from." Most of these fields are bookkeeping the model doesn't act on.

**The KB does the agent's job (tool planning).** `tasks` / `tool_plan` / `match_tasks` try to pre-plan which tools to call (`["search_kb","inspect_table","execute_sql"]`). That's orchestration baked into a retrieval layer. A capable agent should choose tools; coupling planning into the KB means two places reason about strategy and they drift.

**Duplication and manual upkeep of table metadata — and it's already rotting.** Table knowledge lives in at least two places, and a re-check of the live files (2026-06-08) shows concrete decay:

- **The snapshot is stale and structurally impoverished.** `common_table_live_metadata.json` (~74k lines, 1197 tables) was generated `2026-03-11` with `stale_days: 30` — i.e. ~3 months stale. Worse, every table is *bare schema only*: `column_name`, `data_type`, `nullable`, `is_key`. There are **no descriptions, no sample values, no per-table freshness/row counts, and no lineage** — exactly the fields that research says drive text-to-SQL accuracy. This is the impoverished payload the KB indexes today.
- **Two crawlers that disagree, one of them dead.** `scripts/refresh_table_metadata.py` bulk-dumps every schema via system catalog views (→1197 tables). `scripts/enrich_common_tables.py` instead parses table names out of the hand-written `tables.md` and calls `runtime._datasource_for_table(...)`, `runtime.inspect_table_metadata(...)`, and `runtime.refresh_knowledge_index(...)` — **none of which exist on the current `InvestigationRuntime`** (the real methods are module-level `datasource_for_table`, `runtime.inspect_table`, and `kb.ensure_ready`). So the "enrich" path silently broke against an API refactor and produces nothing.
- **`tables.md` is hand-edited prose that isn't the source of truth** — the JSON is. It drifts, and `enrich_common_tables.py` even treats it as an *input* (parsing table names from it), inverting the dependency.

Meanwhile the repo already has `pipelines/discover_code.py`, `discover_modules.py`, and a lineage graph (`pipelines/graph_store.py`, exposed via `trace_pipeline`) that *could* derive the repo→module→table mappings automatically. The fix is to generate one catalog and delete both ad-hoc crawlers.

---

## 2. Design principles for the replacement

1. **One document model, one search function.** No bespoke scorers, no per-query `if`-branches.
2. **Use the database for search.** SQLite FTS5 (BM25) + `sqlite-vec` (embeddings) + Reciprocal Rank Fusion. This is now the standard "local RAG in one binary" stack.
3. **Retrieve facts; let the agent plan.** Drop task recipes / tool_plan from the KB. Playbooks stay as *skills*, indexed like any other document.
4. **Generate the catalog, don't hand-write it.** `tables.md` becomes a build artifact, not a source file.
5. **Config over code.** Source-authority weighting becomes a small weights table, not magic numbers scattered through the scorer.

---

## 3. New knowledge store: hybrid search in one table

Collapse `items` + `chunks` into a single `doc` table, keep a thin optional graph, and add FTS + vector indexes.

```sql
CREATE TABLE doc (
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,     -- table_card | column | doc | skill | code
  source_type TEXT NOT NULL,     -- catalog | code | doc | skill   (drives weighting)
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,     -- the searchable text
  ref         TEXT,              -- citation: file path, table name, URL
  meta        TEXT,              -- JSON: datasource, columns, freshness, repo, etc.
  updated_at  REAL NOT NULL
);

-- Keyword search (BM25), kept in sync with `doc`
CREATE VIRTUAL TABLE doc_fts USING fts5(
  title, body, content='doc', content_rowid='rowid'
);

-- Semantic search (embeddings) via sqlite-vec
CREATE VIRTUAL TABLE doc_vec USING vec0(embedding float[384]);

-- Source-authority weights as DATA, not code
CREATE TABLE source_weight (source_type TEXT PRIMARY KEY, weight REAL NOT NULL);
```

`edges`/lineage is **removed from the KB** — the repo already has a dedicated pipeline graph (`pipelines/graph_store.py`, exposed via `trace_pipeline`). Don't maintain a second copy. Tables and their producing repos are linked by fields inside `meta` instead.

> **Revision (2026-06-08): this stance is reversed — see §9.** Once *debugging*, *cross-connector flow* (MySQL→federated→Redshift→S3), *producer ordering*, and *full coverage of unused tables* became hard requirements, lineage stopped being a side concern. The redesign **unifies the catalog and the graph into one store** and makes lineage the core. §3's `doc` table and §4's catalog still hold, but the node set and edge model below supersede the "edges removed" decision.

### The one search function

```python
def search(query: str, k: int = 8) -> list[Hit]:
    q_vec = embed(query)                       # local small model, see §5
    bm25  = fts_topn(query, n=50)              # SELECT rowid, rank FROM doc_fts WHERE doc_fts MATCH ?
    vec   = vec_topn(q_vec, n=50)              # SELECT rowid, distance FROM doc_vec ... ORDER BY distance
    fused = rrf_merge(bm25, vec,               # reciprocal rank fusion
                      weight_by=source_weight) # authority applied here, once
    return hydrate(fused[:k])                  # join back to doc for title/body/ref/meta
```

Reciprocal Rank Fusion (`1/(rank+60)` summed across lists) is the standard, parameter-light way to combine keyword and vector results, and it removes every hand-tuned `+2.0`/`+0.7`/`×weight` term. Optional **reranking** (a cross-encoder or a cheap LLM rerank) can be added later for ambiguous queries, but start without it.

That's the entire retrieval surface. The 14-field `SearchResult` shrinks to:

```python
@dataclass
class Hit:
    title: str
    body: str          # the matched text
    ref: str           # citation
    kind: str
    source_type: str
    score: float
    meta: dict         # columns/freshness/repo when kind == table_card
```

### Why this is "more powerful" despite being smaller

- Semantic recall: synonyms/paraphrases now match (the current scorer literally cannot).
- Real ranking: BM25 is a tuned, well-understood relevance model vs. ad-hoc token counting.
- Generalizes: no per-question `if`s, so new questions work without code changes.
- Faster at scale: FTS5 and vec0 use indexes instead of scanning every row in Python.

**What the 2026 research measures (validation, not aspiration):** running BM25 + vector in parallel and fusing with RRF lifts recall@10 from ~65–78% (single method) to ~91%, for roughly +6ms p50 latency over dense-only; reported answer-quality gains are +15–30% on RAGAS-style metrics. The two knobs that matter are `rrf_k` (leave at 60) and per-retriever top-k (start at ~20–50, raise only if pre-rerank recall is low). Fixed 50/50 weighting is discouraged; RRF being rank-based is precisely what removes the score-scale problem our magic weights were fighting. Reranking (cross-encoder or cheap LLM) is a later, optional add for ambiguous queries — start without it.

---

## 4. The table catalog ("those tables") — generate it

Replace the hand-maintained `tables.md` + manually-curated `common_table_live_metadata.json` with **one generated source of truth**, built the way modern text-to-SQL semantic layers are: rich per-column descriptions, sample values, freshness, and lineage, because column-level descriptions measurably improve LLM SQL accuracy.

```
app/investigation/catalog/
  build_catalog.py     # the generator (run nightly / on demand)
  catalog.json         # GENERATED source of truth, one object per table
  overrides.yaml       # human edits that survive regeneration
```

Each table object:

```jsonc
{
  "name": "prod.monitoring.provider_combined_audit",
  "datasource": "redshift",
  "env": "prod",
  "tier": "...",
  "description": "Provider/site-level collection-issue audit ...",   // LLM-synthesized, overridable
  "columns": [
    {"name": "issue_sources", "type": "super", "description": "...", "sample": ["TIMEOUT"], "nullable": true}
  ],
  "partitions": ["sales_date"],
  "freshness": {"max_sales_date": "2026-06-07", "row_estimate": 1234567},
  "s3_location": "s3://...",
  "produced_by": [{"repo": "ds-priceeye-data-collection", "module": "site-metrics/import-metrics-generator"}],
  "read_by":    [{"repo": "ds-priceeye-analytics", "module": "alerts"}]
}
```

`build_catalog.py` assembles each field from a source that's already in the repo, so nothing is hand-typed:

| Field | Source (already exists) |
|---|---|
| name, columns, types, partitions | live Glue catalog (`glue_get_table`) + DB introspection |
| freshness (max date, row counts) | a bounded `execute_sql` probe per table |
| `produced_by` / `read_by` | `pipelines/discover_code.py` + `discover_modules.py` lineage — this is exactly what `tables.md` lists by hand today |
| column/table `description` | one LLM pass over name + samples + code context, written once, cached |
| human corrections | `overrides.yaml`, deep-merged last so regeneration never clobbers them |

`tables.md` is then **emitted from `catalog.json`** for humans to read, never hand-edited. The KB indexes a compact "table card" per table (name + description + key columns + freshness) for *discovery*; full schemas are fetched on demand by `inspect_table` reading `catalog.json`. This is the standard catalog/semantic-layer split: small searchable cards for routing, full detail on request.

---

## 5. Embeddings — local model (DECIDED), no new infra

**Decision:** local sentence-embedding model. Run a small local model (e.g. `bge-small-en` / `gte-small`, 384-dim) — no external API, no network dependency, fits the single-EC2 deployment. `sqlite-vec` is a single loadable extension; the whole stack stays "one binary, one file" with zero new services. `text-embedding-3-small` via the existing OpenAI client remains a drop-in fallback (only `embed()` changes), but is **not** the chosen path.

Embeddings are computed at **ingest** time (once per chunk) and cached; query embedding is one call per `search_kb`.

### Feasibility preflight (verified 2026-06-08 in `chatkit/backend/.venv`)

| Dependency | Status | Action |
|---|---|---|
| `sentence-transformers` | ✅ installed | none |
| `torch` + `huggingface_hub` | ✅ installed | none |
| SQLite `fts5` | ✅ available in the bundled sqlite3 | none |
| `sqlite-vec` | ❌ **not installed** | `pip install sqlite-vec` (Phase 0) |
| `threevictors` (DB/S3 access) | ✅ importable | none |
| AWS creds (3VDEV) | ⚠️ **currently expired** (`InvalidClientTokenId`) | re-auth (`assume 3VDEV` / SSO) before any live recrawl or catalog build |

So the only new package is `sqlite-vec`; the embedding model and DB connectors are already present. Note the live-data steps (catalog freshness probes, recrawl) are blocked until 3VDEV creds are refreshed — a human auth step, not a code change.

---

## 6. Retrieval philosophy: adaptive, not "agentic KB"

Current best practice is *adaptive* RAG — match pipeline complexity to query complexity rather than always running an expensive multi-step retrieval loop. In this codebase the **agent already is the control loop**: it calls `search_kb`, reads results, and decides whether to `inspect_table` / `execute_sql` / re-search. So the KB itself should stay a fast, stateless, one-shot hybrid lookup. Don't build an agentic-retrieval layer *inside* the KB; that just adds cost and a second planner. Keep planning in the agent, keep `search_kb` cheap.

---

## 7. Migration plan (incremental, low-risk)

**Phase 0 — preflight + scaffold (no behavior change).** Preflight: `pip install sqlite-vec` (the one missing dep) and confirm 3VDEV creds for the later live steps. Then add the local embed model behind an `embed()` helper, create `doc`, `doc_fts`, `doc_vec`, `source_weight` tables alongside the existing KB, and write an adapter so `search_kb` can run *either* engine behind a flag.

**Phase 1 — reindex into the new store.** Reuse the existing `_ingest_*` functions (docs, tables, codes, skills) but emit `doc` rows + embeddings instead of items/chunks/edges/tasks. Delete the bespoke scorer; implement `search()` = FTS5 + vec + RRF. Keep the old store readable for comparison.

**Phase 2 — generate the catalog.** Build `catalog/build_catalog.py` from Glue + lineage + an LLM description pass; emit `catalog.json` and a generated `tables.md`. Point `inspect_table` and the table-card ingest at `catalog.json`. Retire the hand-maintained `common_table_live_metadata.json`.

**Phase 3 — cut over and delete.** Run the existing e2e investigation cases against both engines; tune only the `source_weight` table (data, not code) and `k`. When the new engine matches/beats the old on the eval set, delete `tasks`/`edges`/`match_tasks`/the legacy scorer and all hardcoded query branches. Move task "playbooks" fully into skills.

**Phase 4 — automate freshness.** Schedule `build_catalog.py` (nightly) and KB reindex on source-hash change (the existing `_source_hash` mechanism already supports this).

### What gets deleted

- `store.search_chunks()` scorer (incl. the `_SOURCE_WEIGHT` magic numbers) and every hardcoded query rule in it and in `retriever.py`.
- `tasks` table, `match_tasks`, `tool_plan` generation, `TaskRecipe`, e2e/skill task ingest.
- `edges` table (lineage lives in the pipeline graph already).
- `tables.md` as a hand-edited file (becomes generated).
- ~10 of the 14 `SearchResult` fields.
- **The two ad-hoc crawlers** — `scripts/enrich_common_tables.py` (already broken against the current runtime API) and the bulk-dump path in `scripts/refresh_table_metadata.py` — replaced by the single `catalog/build_catalog.py`.
- `common_table_live_metadata.json` (bare-schema snapshot) — superseded by the generated `catalog.json`.

Net: fewer tables, one search path, no magic weights, no eval-overfit branches, and genuine semantic + keyword retrieval.

---

## 8. Effort & risk

Roughly: Phase 0–1 is the core lift (new store + hybrid search + reindex) — a few days. Phase 2 (catalog generator) is the larger piece because of the LLM description pass and lineage wiring, but it removes ongoing manual maintenance forever. Risk is contained because the old engine stays runnable behind a flag until the e2e cases confirm parity.

---

## 9. Revised core: one code-derived lineage graph, built for debugging

The §1–§8 design optimizes *retrieval* (find the right table/doc). But the real requirements are broader: the only source of truth is the (messy) code; **every** table must be discoverable even if unused; processes produce one external table then another (ordering); data crosses connectors (MySQL → federated → Redshift external → S3); and the system must **debug** these flows, not just answer questions. That makes lineage the core, and it changes three things.

### 9.1 Unify the catalog and the graph

Collapse "the KB" and "the pipeline graph" into **one store**: nodes are entities, edges are lineage, and every node also carries a searchable card (§3's `doc` row = the card for a node). No second copy, no drift between `search_kb` and `trace_pipeline`.

- **Nodes — every entity, every connector.** Built from the **union** of (a) live introspection of *all* datasources (redshift analytics/core/monitoring, mysql priceeye/metadata/aurora, `federated_*`, glue/external, S3 prefixes) and (b) code/config references. (a) guarantees full coverage: a table with zero code references still gets a node — it's just flagged `no known producer`, which is itself the debugging answer (e.g. `travelport_carriers`, empty in all envs because its EMR producer was deleted years ago).
- **Edges — typed, ordered, cross-connector:** `writes` / `reads`, `federated_as` (mysql ↔ redshift external schema), `backs` (s3 prefix / glue db ↔ external table), `runs` / `triggers` / `then` (orchestration + ordering).

### 9.2 Extract lineage from authoritative declarative sources, not regex alone

Today's edges come from regex over `.py/.java/.scala/.sql/.sh` (`UNLOAD`/`COPY`/`INSERT`/`FROM`). On messy code this is too noisy — the live graph contains edges like `redshift_table:r6gd.8xlarge → redshift_table:m6g.8xlarge` (EC2 instance types parsed as tables). Rank extraction by reliability:

| Tier | Source | Gives us |
|---|---|---|
| 1 — declarative (authoritative) | external-schema SQL (`*-redshift.sql` in `3v-build-deploy` / `priceeye-v2`) | the **cross-connector map**: `federated_priceeye → MySQL priceeye`, `monitoring → Glue monitoring_db`, etc. |
| 1 | `partition_details` MySQL table (`bucket`, `pattern`, `partition_order`, `destination_database`, `destination_table`, `emit_event`) | S3-prefix → external-table edges, **producer ordering** (`partition_order`), and whether a completion event fires (`emit_event`) |
| 1 | Step Function ASL definitions | **ordered** `runs`/`then` edges (e.g. SegmentLevel Spark → market + segment Python) |
| 1 | EventBridge `data-pipeline` rules | `triggers` edges (`Task Completed`, `Price Anomaly`) |
| 2 — code regex (validated) | `UNLOAD`/`COPY`/`INSERT`/`FROM` | producer/consumer edges, **validated against the live catalog** to drop garbage, tagged with confidence + `file:line` provenance |
| 3 — LLM gap-fill | messy/ambiguous code | resolve read-vs-write & step order; synthesize table/column descriptions; cached, overridable |

Ordering ("A then B") is first-class — edges carry `step`/`order` sourced from Step Function state order + `partition_order` + EventBridge chains, never guessed.

### 9.3 Debugging as a first-class operation

Add `diagnose(entity)` that composes the static graph with live state:

1. Resolve the entity (any connector).
2. Walk upstream to the **ordered** producer chain.
3. For each producer: identify the job (SFN / Lambda / ECS / Glue / EMR), pull its last-run status + the table's live freshness/row-count, and check whether it emits its expected event.
4. Pinpoint the break: *"B is empty; producer job X has no recent successful run / B has no producer at all / upstream A is N days stale / `emit_event=0` so nothing alerted."*
5. Attach institutional context (deprecation notes, prior incidents) from the doc/PR/chat side of the store.

This directly covers the real incidents: `travelport_carriers` / `valid_market_carriers` (producer deleted, empty everywhere) and `sales_poc.input_request` (cleared in DEV only). The retrieval surface is then three composable tools — `search_kb` (hybrid card search), `trace_lineage` (ordered cross-connector walk), `diagnose` (the composition above) — with the agent as planner.

### 9.4 Build

One `build_graph.py` (nightly + on source-hash change): introspect all connectors (full coverage) → parse declarative configs (Tier 1) → regex + catalog-validate (Tier 2) → LLM enrich (Tier 3) → write nodes + edges + cards + embeddings into one SQLite (FTS5 + sqlite-vec). `overrides.yaml` deep-merged last. This **replaces** the two divergent crawlers (§1) and the separate pipeline-graph build.

---

## 10. KB content model + table-query protocol (requirements, 2026-06-08)

### 10.1 Three-tier knowledge content (replaces one-file-per-repo)

`docs/` today is ~30 per-repo/per-topic markdown files (`ds-priceeye-analytics.md`, `priceeye-api.md`, …) plus a thin `priceeye_system.md`. Replace that sprawl with three tiers, consulted **in order**:

1. **Tier A — one end-to-end system doc** (`priceeye_overview.md`): the single canonical narrative of how PriceEye works end to end — ingestion → collection → DCO → analytics → anomalies → alerting, the connectors (Redshift core/analytics/monitoring, MySQL/Aurora, federated, Glue/S3), and the prod data stores. Generated from code + configs + lineage (per §9), human-overridable via `overrides.yaml`. This is what `search_kb` returns first for "how does X work."
2. **Tier B — workflow docs, one per process** (`workflows/<process>.md`): each defines a *single* process/workflow (e.g. `market-level-anomalies`, `site-metrics-ingest`, `partition-creation`) — its trigger, **ordered** steps, the tables/buckets it reads and writes, the job that runs it (SFN/Lambda/ECS/Glue/EMR), and how to tell if it's healthy. These are the operational/debugging unit and map 1:1 to §9's process nodes.
3. **Tier C — live code fallback** (`~/git/`): if A and B don't answer, the agent explores the repos directly with `bash`/`grep`/`read_file`/`git`. Already available via the shell tools — make it the *explicit, instructed last resort*, not the default.

Contract: prefer **A → B → C**. The per-repo files are retired (their content folds into A + B or is regenerated). A and B are `doc` cards in the hybrid store; C is not pre-indexed (it is exploration). The agent prompt + `search_kb` ranking must encode this ordering so the model doesn't jump to a shell crawl when a workflow doc already answers the question.

### 10.2 Table-query protocol (`execute_sql`)

Every table query follows a fixed procedure:

1. **Partition check first — HARD GATE (decided).** Resolve the table's partition keys (Glue live → static fallback; already in `PartitionGuard`). If the table is partitioned and the query lacks a predicate on **any** of its partition key(s), **reject the query** with an actionable error naming the missing key(s) and the available partitions. This is a behavior change: today `runtime.execute_sql` only attaches `partition_warnings` and runs anyway — it must instead raise before execution. Tables with no resolvable partition keys (Glue miss + not in static map) are unaffected and run normally.
2. **LIMIT always.** `SqlGuard._apply_limit` already appends `LIMIT <default>` when absent and clamps to `max_limit`. Keep it; ensure it targets the outermost `SELECT`.
3. **GROUP BY for multi-partition — PROMPT-GUIDED (decided).** Not enforced in the guard (detecting "multi-partition intent" in arbitrary SQL is brittle and risks false rejections). Instead, an agent prompt rule instructs: when scanning more than one partition, aggregate with `GROUP BY <partition_key> + measures` rather than returning raw rows. A warning may still flag a raw multi-partition scan, but the query is not blocked on this basis.

**Implementation notes:**
- The hard gate belongs in the `execute_sql` path (runtime) so it raises before hitting the datasource; reuse `PartitionGuard.check_live` to get the missing-key list, but change the caller to raise instead of warn.
- Because the gate is strict, the partition-key resolution must be reliable — Glue-live first, static map fallback — and the error must always list the partition keys so the agent can immediately retry with a valid predicate.

---

## Sources

- [Hybrid full-text + vector search with SQLite (sqlite-vec, Alex Garcia)](https://alexgarcia.xyz/blog/2024/sqlite-vec-hybrid-search/index.html)
- [Hybrid Search: FTS5 + Vector + RRF](https://ceaksan.com/en/hybrid-search-fts5-vector-rrf)
- [Inside SQLite + FTS5 + Vectors hybrid memory](https://zeroclaws.io/blog/zeroclaw-sqlite-fts5-vector-hybrid-memory-explained/)
- [RAG Techniques Compared: Best Practices Guide (2026) — adaptive RAG](https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide)
- [Agentic RAG vs Classic RAG: from pipeline to control loop (Towards Data Science)](https://towardsdatascience.com/agentic-rag-vs-classic-rag-from-a-pipeline-to-a-control-loop/)
- [Optimizing RAG with Hybrid Search & Reranking (Superlinked VectorHub)](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Semantic Layers in 2025: Catalog Owner & Data Leader Playbook (Coalesce)](https://coalesce.io/data-insights/semantic-layers-2025-catalog-owner-data-leader-playbook/)
- [Synthetic SQL Column Descriptions and Their Impact on Text-to-SQL Performance (arXiv)](https://arxiv.org/html/2408.04691)

### Added 2026-06-08 (re-validation pass)

- [Hybrid Search Guide: Vectors & Full-Text, April 2026 (Supermemory)](https://blog.supermemory.ai/hybrid-search-guide/) — recall 65–78% → 91% with BM25+vector+RRF; ~6ms p50 cost.
- [Hybrid Search for RAG: Vector + Keyword + Reranking, 2026 (BuildMVPFast)](https://www.buildmvpfast.com/blog/hybrid-search-rag-vector-keyword-reranking-2026) — `rrf_k=60`, per-retriever top-k ~20–50, reranking as optional later step.
- [The Database Has a New User — LLMs (Tiger Data)](https://www.tigerdata.com/blog/the-database-new-user-llms-need-a-different-database) — LLM-generated semantic catalog improved SQL accuracy up to 27%.
- [Agentic Semantic Model Improvement: Text-to-SQL (Snowflake)](https://www.snowflake.com/en/blog/engineering/agentic-semantic-model-text-to-sql/) — rich semantic spec + validation ≈ +20% SQL accuracy.

### Internal artifacts grounding §9 (authoritative lineage sources)

- `ds-priceeye-analytics/README.md` — pipeline flow, Step Functions ordering (SegmentLevel Spark → market/segment Python), EventBridge `data-pipeline` bus, `partition_details` ordering.
- `3v-build-deploy/databases/sql/prod-core-redshift.sql` + `priceeye-v2/docs/redshift/*.sql` — external-schema → Glue/MySQL cross-connector map (`CREATE EXTERNAL SCHEMA … FROM MYSQL / FROM DATA CATALOG`); note the `-- Not working` `swav` cross-account schema.
- Confluence "GOLD" release pages — `partition_details` INSERTs (`partition_order`, `emit_event`, `destination_database/table`).
- Slack #team 2024-11-21 (`travelport_carriers` / `valid_market_carriers` empty — producer deleted) and #ask-data-science 2025-11-24 (`sales_poc.input_request` cleared in DEV only) — the debugging scenarios `diagnose()` must cover.
