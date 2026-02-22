# Provider Integration Skeleton

## Supported Patterns
- API provider flow.
- Batch provider flow (S3 drop-off).
- Ingest-assisted flow for repeat lookup optimization.

## Lambda-Oriented Pipeline
- Request formatter builds provider-specific payload from expanded requests.
- HTTP service sends requests and receives raw responses.
- Response parser normalizes provider responses to PriceEye output format.

## Messaging and Config
- Components communicate through provider-specific SQS queues.
- Provider behavior is driven by queue naming and config in S3/runtime stores.
- Transaction rate constraints are tracked in provider rate configuration tables.

## Data and Auth Expectations
- Typical payload formats: JSON or XML.
- Common auth model: token/API key credentials.
- Error handling must tolerate retries and duplicate message delivery.
