# Customer Site Codes

## Intent
- Restrict allowed customer site codes to a controlled set.
- Remove free-form codes and reduce manual mapping overhead.

## Hierarchy-Based Routing
- A customer_site_code maps to ordered routes:
- Primary provider + site_code.
- Secondary provider + site_code.
- Tertiary provider + site_code.

## Capacity Interaction
- Scheduler first tries primary route within configured capacity.
- Excess requests are shifted to lower-priority routes or later windows.
- Capacity is transaction-rate based (`TPS`/`TPM`/`TPH`).

## Expected Benefits
- Fewer manual runtime records.
- More consistent routing behavior across customers.
- Better recovery path when one route cannot absorb volume.
