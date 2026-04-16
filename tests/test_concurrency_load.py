"""Concurrency load test for Phase 2 scalability guarantees.

Exercises the three fixes we just landed without hitting a real LLM:
  1. SQLite WAL + batched commits under concurrent writers
  2. BaseLLMClient semaphore + atomic budget reservation under contention
  3. SimulationManager bounded SSE queue with overflow handling

Run with:
  pytest tests/test_concurrency_load.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import random
import statistics
import tempfile
import time
import pytest

from core.llm.base_client import BaseLLMClient
from core.llm.json_parser import BudgetExceededError
from core.platform.platform_engine import PlatformEngine


class _FakeLLM(BaseLLMClient):
    """In-process LLM stub: pretends to call an API, records deterministic
    per-call cost, can inject failures to validate retry/backoff paths."""

    def __init__(self, budget: float = 10.0, max_concurrent: int = 10,
                 latency_ms: float = 30.0, failure_rate: float = 0.0,
                 cost_per_call: float = 0.002):
        super().__init__(budget=budget, max_concurrent=max_concurrent,
                         rate_limit_delay=0.0)
        self.latency_s = latency_ms / 1000.0
        self.failure_rate = failure_rate
        self.cost_per_call = cost_per_call
        self.in_flight = 0
        self.peak_in_flight = 0
        self._lock = asyncio.Lock()

    async def _call_api(self, prompt, system_prompt, temperature,
                        max_output_tokens, response_json, component):
        async with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.latency_s * random.uniform(0.8, 1.2))
            if random.random() < self.failure_rate:
                raise RuntimeError("429 rate limit (synthetic)")
            # Record synthetic usage — keeps the atomic-budget path honest.
            self.stats.record(
                model="fake",
                input_tokens=200,
                output_tokens=400,
                cost_per_1m_input=self.cost_per_call * 1_000_000 / 2 / 200,
                cost_per_1m_output=self.cost_per_call * 1_000_000 / 2 / 400,
                component=component,
            )
            return '{"ok": true}'
        finally:
            async with self._lock:
                self.in_flight -= 1


@pytest.mark.asyncio
async def test_platform_engine_concurrent_writers():
    """50 concurrent writers must not deadlock or lose rows in WAL mode."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "load.db")
    engine = PlatformEngine(db_path)

    async def writer(worker_id: int, n: int):
        for i in range(n):
            pid = await engine.aadd_post(
                {
                    "author_id": f"w{worker_id}",
                    "author_tier": 1,
                    "platform": "x",
                    "text": f"post {i} from worker {worker_id}",
                },
                round_num=1 + (i % 5),
            )
            await engine.aadd_reaction(pid, f"reader{worker_id}", "like", 1)

    start = time.perf_counter()
    await asyncio.gather(*[writer(i, 100) for i in range(50)])
    engine.flush()
    elapsed = time.perf_counter() - start

    stats = engine.get_total_stats()
    print(
        f"\n[platform] 50 writers × 100 posts = {stats['total_posts']} rows, "
        f"{stats['total_reactions']} reactions in {elapsed:.2f}s "
        f"({stats['total_posts']/elapsed:.0f} posts/s)"
    )
    assert stats["total_posts"] == 5000
    assert stats["total_reactions"] == 5000
    engine.close()


@pytest.mark.asyncio
async def test_llm_semaphore_and_budget_cap():
    """Under a tight budget + high concurrency the atomic reservation must
    prevent overshooting, and peak in-flight must respect max_concurrent."""
    llm = _FakeLLM(budget=0.05, max_concurrent=8, latency_ms=20,
                   failure_rate=0.0, cost_per_call=0.005)

    async def call_once(i: int):
        try:
            await llm.generate(f"hello {i}", component="stress", retries=1)
            return "ok"
        except BudgetExceededError:
            return "budget"
        except Exception:
            return "err"

    # 200 concurrent callers, budget only allows ~10 calls.
    results = await asyncio.gather(*[call_once(i) for i in range(200)])
    ok = sum(r == "ok" for r in results)
    blocked = sum(r == "budget" for r in results)

    print(
        f"\n[llm] budget=$0.05 concurrent=200 → ok={ok} blocked={blocked} "
        f"cost=${llm.stats.total_cost:.4f} peak_in_flight={llm.peak_in_flight}"
    )
    # Reservation must stop us *before* we blow the budget by more than one
    # in-flight slot.
    assert llm.stats.total_cost <= 0.05 + 0.005
    # Semaphore must cap actual concurrency.
    assert llm.peak_in_flight <= 8
    # Some requests must have been blocked, others must have succeeded.
    assert ok >= 5 and blocked >= 5


@pytest.mark.asyncio
async def test_llm_retry_backoff_survives_rate_limit_storm():
    """20% synthetic 429 rate — retries + jittered backoff should still
    complete the workload."""
    llm = _FakeLLM(budget=5.0, max_concurrent=16, latency_ms=15,
                   failure_rate=0.2, cost_per_call=0.001)

    async def call_once(i: int):
        try:
            await llm.generate(f"req {i}", component="stress", retries=4)
            return "ok"
        except Exception:
            return "fail"

    start = time.perf_counter()
    results = await asyncio.gather(*[call_once(i) for i in range(100)])
    elapsed = time.perf_counter() - start
    ok = sum(r == "ok" for r in results)

    print(
        f"\n[llm] 100 reqs @ 20% synthetic 429s → ok={ok}/100 in {elapsed:.2f}s "
        f"(errors logged={llm.stats.errors})"
    )
    # With 4 retries and 20% base failure, per-request failure ≈ 0.2^4 ≈ 0.16%.
    # Be forgiving — we just need the happy path to dominate.
    assert ok >= 90


@pytest.mark.asyncio
async def test_sse_queue_bounded_drop_oldest():
    """SSE queue must cap memory and drop oldest under sustained producer."""
    from api.simulation_manager import SimulationManager, SSE_QUEUE_MAX
    from api.models import SimulationRequest, ProgressEvent

    mgr = SimulationManager()
    sim_id = "loadtest"
    req = SimulationRequest(brief="x", provider="gemini")
    from api.simulation_manager import SimulationState
    state = SimulationState(sim_id, req)
    mgr.simulations[sim_id] = state

    # Producer: hammer _emit faster than any consumer can drain.
    total = SSE_QUEUE_MAX * 3
    for i in range(total):
        await mgr._emit(sim_id, ProgressEvent(type="round_phase",
                                              message=f"e{i}"))

    assert state.event_queue.qsize() <= SSE_QUEUE_MAX, (
        f"queue exceeded cap: {state.event_queue.qsize()} > {SSE_QUEUE_MAX}"
    )
    dropped = state.events_dropped
    print(
        f"\n[sse] emitted {total}, queued={state.event_queue.qsize()}, "
        f"dropped={dropped} (cap={SSE_QUEUE_MAX})"
    )
    assert dropped >= total - SSE_QUEUE_MAX

    # Terminal event must go through even under backpressure.
    await mgr._emit(sim_id, ProgressEvent(type="completed", message="done"))
    events = []
    while not state.event_queue.empty():
        events.append(state.event_queue.get_nowait())
    types = {e.type for e in events}
    assert "completed" in types


@pytest.mark.asyncio
async def test_safe_name_hash_unique_under_collision():
    """Two scenarios whose sanitized name collides must still produce
    different directory names once the sim_id hash is appended."""
    from api.simulation_manager import SimulationManager
    mgr = SimulationManager()
    a = mgr._make_safe_name("Foo/Bar!", "aaaaaaaa")
    b = mgr._make_safe_name("Foo Bar.", "bbbbbbbb")
    assert a != b, f"safe_name collision: {a} == {b}"
    assert a.startswith("Foo_Bar_")
    assert b.startswith("Foo_Bar_")


@pytest.mark.asyncio
async def test_concurrent_streams_throughput_10_20_50():
    """End-to-end mini stress: simulate 10/20/50 concurrent 'streams' each
    doing a burst of 50 LLM calls. Prints a summary table so we have
    repeatable baselines for future regressions."""
    results = {}
    for n_streams in (10, 20, 50):
        llm = _FakeLLM(budget=50.0, max_concurrent=16, latency_ms=10,
                       failure_rate=0.05, cost_per_call=0.001)
        latencies: list[float] = []

        async def stream(i: int):
            for j in range(50):
                t0 = time.perf_counter()
                try:
                    await llm.generate(f"s{i}-{j}", component="bench", retries=3)
                except Exception:
                    pass
                latencies.append((time.perf_counter() - t0) * 1000)

        start = time.perf_counter()
        await asyncio.gather(*[stream(i) for i in range(n_streams)])
        elapsed = time.perf_counter() - start

        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95)]
        p99 = lat_sorted[int(len(lat_sorted) * 0.99)]
        rps = llm.stats.call_count / elapsed if elapsed else 0

        results[n_streams] = {
            "elapsed_s": round(elapsed, 2),
            "calls": llm.stats.call_count,
            "errors": llm.stats.errors,
            "rps": round(rps, 1),
            "peak_in_flight": llm.peak_in_flight,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
        }

    print("\n[concurrency] load test summary")
    print(f"{'streams':>8}  {'rps':>7}  {'peak':>5}  {'p50':>6}  {'p95':>6}  {'p99':>6}  {'errs':>5}")
    for n, m in results.items():
        print(f"{n:>8}  {m['rps']:>7}  {m['peak_in_flight']:>5}  "
              f"{m['p50_ms']:>6}  {m['p95_ms']:>6}  {m['p99_ms']:>6}  "
              f"{m['errors']:>5}")

    # Sanity: 50-stream throughput shouldn't collapse by >5× vs 10-stream,
    # which would signal semaphore/lock degeneration.
    assert results[50]["rps"] >= results[10]["rps"] / 5
    # Semaphore must bound peak in-flight at 16 regardless of stream count.
    for m in results.values():
        assert m["peak_in_flight"] <= 16
