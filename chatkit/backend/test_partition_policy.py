from __future__ import annotations

import pytest

from app.nextgen_types import PartitionPolicy, PlanFilter, TableSpec
from app.partition_policy import ensure_partition_filters


def test_partition_policy_requires_declared_predicates() -> None:
    spec = TableSpec(
        table_id="monitoring_provider_combined_audit",
        physical_name="prod.monitoring.provider_combined_audit",
        source_system="redshift",
        partition_policy=PartitionPolicy(
            partition_columns=["customers", "sales_date"],
            required_predicates=["customers", "sales_date"],
        ),
    )

    filters = [PlanFilter(column="sales_date", operator="=", value="20260211")]
    with pytest.raises(ValueError, match="Missing required partition predicates"):
        ensure_partition_filters(spec, filters)

    ok_filters = [
        PlanFilter(column="sales_date", operator="=", value="20260211"),
        PlanFilter(column="customers", operator="=", value="AA"),
    ]
    ensure_partition_filters(spec, ok_filters)
