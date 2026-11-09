"""Launch a local simulated cluster (sim workers + router) as subprocesses — no Docker needed."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from kvbench.router.config import RouterConfig


def _wait_http(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False


def launch_cluster(router_config: str, sim_overrides: dict | None = None, router_port: int | None = None) -> int:
    cfg = RouterConfig.load(router_config)
    port = router_port or cfg.port
    procs: list[subprocess.Popen] = []

    def shutdown(*_):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    try:
        for w in cfg.workers:
            if w.backend != "sim":
                print(f"skipping {w.name}: backend={w.backend} (start it separately)")
                continue
            u = urlparse(w.url)
            if u.hostname not in ("localhost", "127.0.0.1"):
                print(f"skipping {w.name}: {w.url} is not local")
                continue
            args = [sys.executable, "-m", "kvbench.cli", "sim-worker", "--host", "127.0.0.1", "--port", str(u.port),
                    "--engine-id", w.name, "--role", w.role, "--model-name", cfg.model]
            if w.sim_config:
                args += ["--config-file", w.sim_config]
            for k, v in {**(w.sim_overrides or {}), **(sim_overrides or {})}.items():
                args += [f"--{k.replace('_', '-')}", str(v)]
            procs.append(subprocess.Popen(args))
            print(f"started sim worker {w.name} ({w.role}) on {w.url}")
        for w in cfg.workers:
            if w.backend == "sim" and not _wait_http(w.url.rstrip("/") + "/health", 30):
                print(f"worker {w.name} did not become healthy", file=sys.stderr)
                shutdown()
                return 1
        procs.append(subprocess.Popen([sys.executable, "-m", "kvbench.cli", "router", "--config", router_config,
                                       "--host", "127.0.0.1", "--port", str(port)]))
        if not _wait_http(f"http://127.0.0.1:{port}/health", 30):
            print("router did not become healthy", file=sys.stderr)
            shutdown()
            return 1
        print(f"\ncluster ready: router http://localhost:{port}  (mode={cfg.mode}, policy={cfg.policy})")
        print("Ctrl-C to stop.\n")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        while all(p.poll() is None for p in procs):
            time.sleep(1.0)
        print("a process exited; shutting down", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        shutdown()
