#!/usr/bin/env python3
"""
Lightweight live dashboard server for ML-Agents runs.

Serves a small static UI and exposes JSON endpoints to read scalar metrics
from TensorBoard event files produced by `mlagents-learn`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:  # pragma: no cover - handled at runtime with a friendly error
    event_accumulator = None


def _default_size_guidance():
    if event_accumulator is None:
        return {}
    return {
        event_accumulator.SCALARS: 0,  # keep all scalars
        event_accumulator.TENSORS: 0,
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
    }


@dataclass
class BehaviorEventFile:
    run_id: str
    behavior: str
    event_path: Path
    updated_at: float


def discover_runs(results_dir: Path) -> List[BehaviorEventFile]:
    """Find TensorBoard event files under the ML-Agents results directory."""
    runs: List[BehaviorEventFile] = []
    if not results_dir.exists():
        return runs

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for behavior_dir in run_dir.iterdir():
            if not behavior_dir.is_dir():
                continue
            event_files = list(behavior_dir.glob("events.out.tfevents.*"))
            if not event_files:
                continue
            latest = max(event_files, key=lambda p: p.stat().st_mtime)
            runs.append(
                BehaviorEventFile(
                    run_id=run_dir.name,
                    behavior=behavior_dir.name,
                    event_path=latest,
                    updated_at=latest.stat().st_mtime,
                )
            )
    return runs


class EventCache:
    """Caches EventAccumulators so we do not re-open files for every request."""

    def __init__(self) -> None:
        self._cache: Dict[Path, event_accumulator.EventAccumulator] = {}
        self._lock = threading.Lock()

    def get(self, event_path: Path) -> event_accumulator.EventAccumulator:
        if event_accumulator is None:
            raise RuntimeError(
                "tensorboard is not installed. Install it with `pip install tensorboard`."
            )
        with self._lock:
            acc = self._cache.get(event_path)
            if acc is None:
                acc = event_accumulator.EventAccumulator(
                    str(event_path), size_guidance=_default_size_guidance()
                )
                acc.Reload()
                self._cache[event_path] = acc
            else:
                acc.Reload()
            return acc


class DashboardHandler(SimpleHTTPRequestHandler):
    """Handles API requests and serves the static dashboard."""

    def __init__(
        self,
        *args,
        results_dir: Path,
        event_cache: EventCache,
        static_dir: Path,
        **kwargs,
    ) -> None:
        self.results_dir = results_dir
        self.event_cache = event_cache
        super().__init__(*args, directory=str(static_dir), **kwargs)

    # Silence the default noisy log.
    def log_message(self, fmt: str, *args) -> None:  # pragma: no cover - keep quiet
        sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            return self.handle_runs()
        if parsed.path == "/api/metrics":
            return self.handle_metrics(parsed.query)
        return super().do_GET()

    def handle_runs(self) -> None:
        runs = discover_runs(self.results_dir)
        grouped: Dict[str, dict] = {}
        for r in runs:
            if r.run_id not in grouped:
                grouped[r.run_id] = {
                    "id": r.run_id,
                    "behaviors": [],
                    "latest_update": r.updated_at,
                }
            grouped[r.run_id]["behaviors"].append(
                {
                    "name": r.behavior,
                    "event_path": str(r.event_path),
                    "updated_at": r.updated_at,
                }
            )
            grouped[r.run_id]["latest_update"] = max(
                grouped[r.run_id]["latest_update"], r.updated_at
            )
        payload = {"runs": sorted(grouped.values(), key=lambda x: x["latest_update"])}
        self._write_json(payload)

    def handle_metrics(self, query: str) -> None:
        params = parse_qs(query)
        run_id = params.get("run", [None])[0]
        behavior = params.get("behavior", [None])[0]
        limit_raw = params.get("limit", [200])[0]
        try:
            limit = max(1, min(int(limit_raw), 2000))
        except ValueError:
            limit = 200

        if not run_id or not behavior:
            self._write_json(
                {"error": "run and behavior are required"}, status=HTTPStatus.BAD_REQUEST
            )
            return

        event_file = self._find_event_file(run_id, behavior)
        if not event_file:
            self._write_json(
                {"error": f"No event file found for run '{run_id}' and behavior '{behavior}'"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            acc = self.event_cache.get(event_file.event_path)
            tags = acc.Tags().get("scalars", [])
            data: Dict[str, List[dict]] = {}
            for tag in tags:
                events = acc.Scalars(tag)[-limit:]
                data[tag] = [
                    {"step": ev.step, "value": float(ev.value), "wall_time": ev.wall_time}
                    for ev in events
                ]
            payload = {
                "run": run_id,
                "behavior": behavior,
                "event_path": str(event_file.event_path),
                "updated_at": event_file.updated_at,
                "tags": tags,
                "data": data,
            }
            self._write_json(payload)
        except RuntimeError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _find_event_file(self, run_id: str, behavior: str) -> Optional[BehaviorEventFile]:
        for run in discover_runs(self.results_dir):
            if run.run_id == run_id and run.behavior == behavior:
                return run
        return None

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a live dashboard for ML-Agents training metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to the ML-Agents results directory (defaults to ./results).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)."
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path(__file__).parent / "web",
        help="Path to static assets (default: monitor/web).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    static_dir = args.static_dir.resolve()
    static_dir.mkdir(parents=True, exist_ok=True)

    event_cache = EventCache()

    handler = lambda *h_args, **h_kwargs: DashboardHandler(
        *h_args,
        results_dir=results_dir,
        event_cache=event_cache,
        static_dir=static_dir,
        **h_kwargs,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Live dashboard ready: http://{args.host}:{args.port} "
        f"(monitoring runs under {results_dir})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
