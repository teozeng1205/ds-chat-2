# AutoScheduler Implementation

## Glossary
- `provider`: integration source (for example `QL2`).
- `site_code`: provider-specific endpoint or website variant.
- `customer_site_code`: canonical input code used for scheduling/routing.

## Why AutoScheduler Exists
- Replace manual site-mapping/runtime setup per customer.
- Enforce throughput constraints across provider + site_code.
- Use deterministic fallback when a primary route fails or is saturated.

## Core Scheduling Model
- Input requests are grouped by customer_site_code.
- Each customer_site_code resolves to a hierarchy of provider+site_code routes.
- Scheduler allocates work against route capacity.
- Overflow is re-routed by hierarchy or moved to later execution windows.

## Required Inputs / Config
- Standardized customer_site_code dictionary.
- Site hierarchy definitions per customer_site_code.
- Capacity values per provider+site_code (`TPS` / `TPM` / `TPH`).
- Validation path for file submissions before runtime scheduling.

## Operational Signals
- Capacity misconfiguration causes delays/backlog.
- Weak hierarchy coverage causes routing gaps.
- API/scraper limitations can reduce field completeness vs expected output.
