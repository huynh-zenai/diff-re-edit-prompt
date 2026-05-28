from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from logger_config import logger
from modules.base import BaseModule
from modules.renderer.settings import RendererConfig
from pipeline.task import PipelineTask

_RUNNER_JS = Path(__file__).parent / "render_service" / "render_runner.mjs"


@dataclass
class _Sidecar:
    idx: int
    port: int
    proc: asyncio.subprocess.Process
    client: httpx.AsyncClient
    log_tasks: list[asyncio.Task] = field(default_factory=list)


class RendererModule(BaseModule):
    def __init__(self, config: RendererConfig) -> None:
        self.config = config
        self._sidecars: list[_Sidecar] = []
        self._dispatch_counter = 0
        self._dispatch_lock = asyncio.Lock()

    async def startup(self) -> None:
        if not _RUNNER_JS.exists():
            logger.warning(f"[RENDERER] runner missing at {_RUNNER_JS}, disabling")
            return

        node_cwd = self._resolve_node_cwd()
        count = max(1, self.config.sidecar_count)
        logger.info(
            f"[RENDERER] starting {count} sidecar(s) | node={self.config.node_binary} "
            f"runner={_RUNNER_JS} base_port={self.config.sidecar_port} cwd={node_cwd} "
            f"pool_size_per_sidecar={self.config.pool_size}"
        )

        async def _spawn(idx: int) -> _Sidecar:
            port = self.config.sidecar_port + idx
            static_port = self.config.static_port_base + idx
            env = {
                **os.environ,
                "PORT": str(port),
                "STATIC_PORT": str(static_port),
                "RENDER_POOL_SIZE": str(self.config.pool_size),
                "RENDER_TIMEOUT_MS": str(self.config.render_timeout_ms),
                "PROTOCOL_TIMEOUT_MS": str(self.config.protocol_timeout_ms),
            }
            proc = await asyncio.create_subprocess_exec(
                self.config.node_binary,
                str(_RUNNER_JS),
                env=env,
                cwd=node_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_task = asyncio.create_task(
                self._pipe_logger(proc.stdout, f"#{idx}/stdout")
            )
            stderr_task = asyncio.create_task(
                self._pipe_logger(proc.stderr, f"#{idx}/stderr")
            )
            client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
            await self._wait_ready_for(proc, client, port, idx)
            logger.info(f"[RENDERER] sidecar #{idx} ready on port {port}")
            return _Sidecar(idx=idx, port=port, proc=proc, client=client,
                            log_tasks=[stdout_task, stderr_task])

        try:
            self._sidecars = await asyncio.gather(*[_spawn(i) for i in range(count)])
        except Exception:
            await self.shutdown()
            raise

        logger.info(
            f"[RENDERER] {len(self._sidecars)} sidecar(s) ready, total slots="
            f"{len(self._sidecars) * self.config.pool_size}"
        )

    async def shutdown(self) -> None:
        sidecars = self._sidecars
        self._sidecars = []

        for sc in sidecars:
            try:
                await sc.client.aclose()
            except Exception:
                pass

        for sc in sidecars:
            if sc.proc.returncode is None:
                logger.info(f"[RENDERER] terminating sidecar #{sc.idx}")
                try:
                    sc.proc.terminate()
                    try:
                        await asyncio.wait_for(sc.proc.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"[RENDERER] sidecar #{sc.idx} did not exit on SIGTERM, sending SIGKILL"
                        )
                        sc.proc.kill()
                        await sc.proc.wait()
                except ProcessLookupError:
                    pass

        for sc in sidecars:
            for t in sc.log_tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

    async def _next_sidecar(self) -> _Sidecar | None:
        async with self._dispatch_lock:
            if not self._sidecars:
                return None
            sc = self._sidecars[self._dispatch_counter % len(self._sidecars)]
            self._dispatch_counter += 1
            return sc

    async def process(self, task: PipelineTask) -> PipelineTask:
        if task.failed or not task.js_code:
            logger.debug(
                f"[RENDERER] '{task.stem}' skip "
                f"(failed={task.failed}, has_code={bool(task.js_code)})"
            )
            return task

        sc = await self._next_sidecar()
        if sc is None:
            task.render_errors = ["no sidecars available"]
            logger.warning(f"[RENDERER] '{task.stem}' skip — no sidecars available")
            return task

        logger.info(
            f"[RENDERER] '{task.stem}' start | sidecar=#{sc.idx} "
            f"| js_code={len(task.js_code)} bytes"
        )

        payload = {
            "source": task.js_code,
            "options": {
                "imgSize": self.config.img_size,
                "gap": self.config.grid_gap,
            },
        }
        if self.config.bg_color:
            payload["options"]["bgColor"] = self.config.bg_color

        url = f"http://{self.config.sidecar_host}:{sc.port}/render/grid"
        t0 = time.monotonic()
        try:
            resp = await sc.client.post(url, json=payload)
        except Exception as exc:
            task.render_errors = [f"{type(exc).__name__}: {exc}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (http) sidecar=#{sc.idx} | "
                f"{task.render_errors[0]}"
            )
            return task

        task.render_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code == 200:
            task.rendered_png = resp.content
            task.refinement_rendered_pngs.append(resp.content)
            logger.info(
                f"[RENDERER] '{task.stem}' PASS sidecar=#{sc.idx} | "
                f"png={len(task.rendered_png)}B render={task.render_ms/1000:.1f}s"
            )
        else:
            detail = resp.text[:200] if resp.text else ""
            task.render_errors = [f"HTTP {resp.status_code}: {detail}"]
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (status) sidecar=#{sc.idx} | "
                f"{task.render_errors[0]} | render={task.render_ms/1000:.1f}s"
            )
            return task

        return task

    def _resolve_node_cwd(self) -> str:
        preferred = os.environ.get("NODE_CWD")
        candidates = [preferred, "/workspace", str(_RUNNER_JS.parent.parent.parent.parent)]
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "node_modules")):
                return c
        return str(_RUNNER_JS.parent)

    async def _wait_ready_for(
        self,
        proc: asyncio.subprocess.Process,
        client: httpx.AsyncClient,
        port: int,
        idx: int,
    ) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        ping_url = f"http://{self.config.sidecar_host}:{port}/ping"
        last_err: str = ""
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(
                    f"[RENDERER] sidecar #{idx} exited early: rc={proc.returncode}"
                )
            try:
                r = await client.get(ping_url, timeout=2.0)
                if r.status_code == 200:
                    return
                last_err = f"status={r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"[RENDERER] sidecar #{idx} not ready after {self.config.startup_timeout_s}s "
            f"(last: {last_err})"
        )

    @staticmethod
    async def _pipe_logger(stream: asyncio.StreamReader | None, which: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    if "stderr" in which:
                        logger.warning(f"[renderer-sidecar {which}] {text}")
                    else:
                        logger.debug(f"[renderer-sidecar {which}] {text}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[renderer-sidecar {which}] pipe error: {exc}")
