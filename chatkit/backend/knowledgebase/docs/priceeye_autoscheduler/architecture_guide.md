# PriceEye Architecture Guide (Ops Summary)

## Processing Model
- Queue-driven architecture centered on SQS message exchange.
- Worker runtime mix includes Lambda and Fargate services.
- Components are decoupled into request handling, persistence, and analytics stages.

## Storage Roles
- Aurora: configuration and auditing workloads.
- Redshift: reporting and analytics workloads.
- S3: staged datasets and batch exchange surfaces.

## Reliability and Scale Signals
- FIFO queues are used where ordering/dedup semantics matter.
- Audit persistence is asynchronous and queue-backed.
- Partitioned external datasets are surfaced for scalable query access.

## Investigation Relevance
- For backlog incidents: inspect queue depth, worker throughput, and rate caps.
- For data consistency incidents: check async persistence boundaries and retry paths.
- For reporting gaps: verify Redshift/Spectrum partition freshness and registration.
