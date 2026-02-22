# Providers Overview

## Provider Types
- `API`: real-time or near-real-time partner API calls.
- `Ingest`: streaming/lookup support to reduce repeated external calls.
- `Batch`: file-based exchange through AWS S3 drop-off workflows.

## Operational Characteristics
- API providers depend on endpoint stability and auth lifecycle.
- Ingest providers improve response speed for previously-seen requests.
- Batch providers favor throughput and scheduled processing over immediacy.

## Integration Considerations
- Map each provider to valid site_code values.
- Keep per-provider transaction limits current for scheduling.
- Track provider-specific failure modes for fallback routing decisions.
