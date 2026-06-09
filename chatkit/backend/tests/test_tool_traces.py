"""Tests for workflow tool-call traces (app.tools._common).

trace_begin adds a 'loading' CustomTask and returns its index; trace_finish
flips that exact index to 'complete' even when other tasks were appended in
between; trace_done is a one-shot 'complete'. A bad icon must not drop the
trace (it retries without the icon).
"""

from __future__ import annotations

import asyncio

from chatkit.types import Workflow

from app.tools._common import trace_begin, trace_done, trace_finish


class _FakeAgentCtx:
    def __init__(self) -> None:
        self.workflow_item = type("WF", (), {"workflow": Workflow(type="custom", tasks=[])})()
        self.adds = 0
        self.updates = 0

    async def add_workflow_task(self, task):
        self.workflow_item.workflow.tasks.append(task)
        self.adds += 1

    async def update_workflow_task(self, task, index):
        self.workflow_item.workflow.tasks[index] = task
        self.updates += 1


class _Wrapper:
    def __init__(self, ctx):
        self.context = ctx


def test_trace_begin_finish_roundtrip() -> None:
    async def run():
        inner = _FakeAgentCtx()
        ctx = _Wrapper(inner)
        idx = await trace_begin(ctx, title="Running SQL…", content="SELECT 1 FROM t", icon="search")
        assert idx == 0
        assert inner.workflow_item.workflow.tasks[0].status_indicator == "loading"
        # interleave another task — must not shift idx 0
        await trace_done(ctx, title="Searched KB · 3 items", content="anomalies", icon="book-open")
        await trace_finish(ctx, idx, title="Ran SQL · core · 8 rows", content="SELECT 1 FROM t", icon="search")
        assert inner.workflow_item.workflow.tasks[0].status_indicator == "complete"
        assert inner.workflow_item.workflow.tasks[0].title == "Ran SQL · core · 8 rows"
        assert inner.adds == 2 and inner.updates == 1
    asyncio.run(run())


def test_bad_icon_still_emits() -> None:
    async def run():
        inner = _FakeAgentCtx()
        ctx = _Wrapper(inner)
        idx = await trace_begin(ctx, title="x", content="y", icon="definitely-not-an-icon")
        assert idx == 0
        assert len(inner.workflow_item.workflow.tasks) == 1
        assert inner.workflow_item.workflow.tasks[0].icon is None
    asyncio.run(run())


def test_trace_is_noop_without_context() -> None:
    async def run():
        # No .context / no workflow API → must not raise, returns None.
        idx = await trace_begin(object(), title="x", content="y")
        assert idx is None
        await trace_finish(object(), None, title="x")  # no-op
    asyncio.run(run())
