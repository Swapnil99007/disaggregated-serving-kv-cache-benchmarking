"""Telemetry samplers that run alongside a benchmark.

* :class:`NvmlSampler` — GPU memory used/total and utilization via NVML
  (``pip install nvidia-ml-py``). Runs in a thread; silently disabled when NVML
  or an NVIDIA GPU is unavailable (e.g. on a laptop with the simulator).
* :class:`RouterStateSampler` — polls the router's ``/admin/state`` to record
  per-worker KV-cache usage, engine queue depths, router in-flight counts, and
  (for the simulator) modeled GPU memory.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx


class NvmlSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.rows: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False
        self.error: str | None = None
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self.available = len(self._handles) > 0
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    def _run(self, t0: float) -> None:
        nv = self._nvml
        while not self._stop.is_set():
            ts = time.perf_counter() - t0
            for i, h in enumerate(self._handles):
                try:
                    mem = nv.nvmlDeviceGetMemoryInfo(h)
                    util = nv.nvmlDeviceGetUtilizationRates(h)
                    self.rows.append({
                        "ts": ts, "gpu": i,
                        "mem_used_bytes": mem.used, "mem_total_bytes": mem.total,
                        "util_gpu_pct": util.gpu, "util_mem_pct": util.memory,
                    })
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(self.interval_s)

    def start(self, t0: float) -> None:
        if not self.available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(t0,), daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=5)
        return self.rows


class RouterStateSampler:
    def __init__(self, router_url: str, interval_s: float = 0.5):
        self.url = router_url.rstrip("/") + "/admin/state"
        self.interval_s = interval_s
        self.rows: list[dict] = []
        self._task: asyncio.Task | None = None

    async def _run(self, t0: float) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    st = (await client.get(self.url)).json()
                    ts = time.perf_counter() - t0
                    for w in st.get("workers", []):
                        self.rows.append({
                            "ts": ts,
                            "worker": w["name"],
                            "role": w["role"],
                            "kv_usage": w.get("kv_usage"),
                            "engine_running": w.get("engine_running"),
                            "engine_waiting": w.get("engine_waiting"),
                            "router_inflight": w.get("inflight"),
                            "pending_prefill_tokens": w.get("pending_prefill_tokens"),
                            "gpu_mem_used": w.get("gpu_mem_used"),
                            "gpu_mem_total": w.get("gpu_mem_total"),
                            "index_chunks": w.get("index_chunks"),
                        })
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(self.interval_s)

    def start(self, t0: float) -> None:
        self._task = asyncio.create_task(self._run(t0))

    async def stop(self) -> list[dict]:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self.rows
