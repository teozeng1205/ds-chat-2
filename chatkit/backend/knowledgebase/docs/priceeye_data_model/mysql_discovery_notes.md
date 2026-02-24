# MySQL Discovery Notes

- Total `priceeye` tables discovered: 89.
- New KB table specs generated: 86 (existing `priceeye_provider`, `priceeye_site`, `priceeye_customer` preserved and corrected).
- Entity key correction: `priceeye.customer` canonical customer column is `name`.
- Autoscheduler-related tables identified include `customer_site_code`, `site_hierarchy`, `site_metrics`, `transaction_rates`, `runtime_archive`, and `auto_schedule_trigger`.
