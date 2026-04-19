-- Fixture — SQL file with FROM + INSERT INTO patterns.
INSERT INTO analytics.daily_summary
SELECT customer, count(*) FROM prod.monitoring.provider_combined_audit
WHERE sales_date = 20260419
GROUP BY customer;

UPDATE analytics.daily_summary SET updated_at = now();
