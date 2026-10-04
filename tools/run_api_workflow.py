#!/usr/bin/env python3
"""Submit a ComfyUI API workflow and wait for it to finish.

Usage: python tools/run_api_workflow.py WORKFLOW.json [--base URL] [--timeout SECONDS]

Prints the prompt id, then tails the websocket-free status by polling /history.
Exits non-zero when the prompt fails or validation is rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def submit(base: str, workflow: dict) -> str:
    payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
    try:
        result = post_json(f"{base}/prompt", payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"prompt rejected ({exc.code}):\n{body}") from exc
    if "error" in result:
        raise SystemExit(f"prompt error: {json.dumps(result, indent=2)}")
    return result["prompt_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow")
    parser.add_argument("--base", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    base = args.base.rstrip("/")
    workflow = json.loads(Path(args.workflow).read_text(encoding="utf-8"))
    prompt_id = submit(base, workflow)
    print(f"submitted {args.workflow} -> prompt_id {prompt_id}", flush=True)

    started = time.time()
    last_note = 0.0
    while True:
        if time.time() - started > args.timeout:
            raise SystemExit(f"timed out after {args.timeout:.0f}s waiting for {prompt_id}")
        history = get_json(f"{base}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in ("success", "error"):
                messages = status.get("messages", [])
                if status.get("status_str") == "error" or status.get("completed") is False:
                    print(json.dumps(status, indent=2)[:8000])
                    return 1
                print(f"completed in {time.time() - started:.1f}s")
                for node_id, output in (entry.get("outputs") or {}).items():
                    if output:
                        print(f"  node {node_id}: {json.dumps(output)[:400]}")
                return 0
        if time.time() - last_note > 30:
            last_note = time.time()
            queue = get_json(f"{base}/queue")
            running = len(queue.get("queue_running") or [])
            print(f"  ...{time.time() - started:.0f}s elapsed (running={running})", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main())
