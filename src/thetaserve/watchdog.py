"""Stop the pod when vLLM is idle or the maximum lifetime has passed (DESIGN.md section 15).

The launch script writes this file to the host and starts it next to vLLM, so it must use only
the standard library. Usage: python3 watchdog.py PORT IDLE_S MAX_LIFE_S

The pod stops itself with the RUNPOD_API_KEY and RUNPOD_POD_ID that Runpod puts in every pod
(M3 probe, 2026-10-22). Stopping keeps the disk and the cache volume; `thetaserve start` resumes.
"""

import json
import os
import sys
import time
import urllib.request

POLL_S = 60
STOP_URL = "https://api.runpod.io/v2/pods/{pod}/action"


def activity(metrics: str) -> tuple[float, float]:
    """(requests completed, requests running or waiting) from vLLM's Prometheus text."""
    done = busy = 0.0
    for line in metrics.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name == "vllm:request_success_total":
            done += float(line.rsplit(" ", 1)[1])
        elif name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            busy += float(line.rsplit(" ", 1)[1])
    return done, busy


def reason_to_stop(
    now: float, started: float, last_active: float | None, idle_s: int, max_life_s: int
) -> str | None:
    """Why to stop now, or None. The idle clock starts once vLLM answers, not while it loads."""
    if now - started >= max_life_s:
        return f"maximum lifetime of {max_life_s} s reached"
    if last_active is not None and now - last_active >= idle_s:
        return f"idle for {idle_s} s"
    return None


def main() -> None:
    port, idle_s, max_life_s = (int(arg) for arg in sys.argv[1:4])
    metrics = urllib.request.Request(
        f"http://127.0.0.1:{port}/metrics",
        headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"},
    )
    # ponytail: the clocks restart with the container, so a vLLM crash loop never reaches
    # max-life. Keep the start time on disk if that shows up in practice.
    started = time.monotonic()
    last_active: float | None = None
    seen: tuple[float, float] | None = None
    while True:
        time.sleep(POLL_S)
        now = time.monotonic()
        try:
            with urllib.request.urlopen(metrics, timeout=10) as resp:  # noqa: S310 - fixed http URL
                current = activity(resp.read().decode())
            if current != seen or current[1] > 0:
                last_active, seen = now, current
        except (OSError, ValueError):
            pass  # vLLM still loading, or restarting; only max-life counts meanwhile
        if why := reason_to_stop(now, started, last_active, idle_s, max_life_s):
            stop(why)


def stop(why: str) -> None:
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    print(f"thetaserve watchdog: {why}; stopping pod {pod}", flush=True)
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        STOP_URL.format(pod=pod),
        method="POST",
        data=json.dumps({"action": "stop"}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - fixed https URL
            print(f"thetaserve watchdog: stop answered {resp.status}", flush=True)
    except OSError as exc:  # HTTPError included
        print(f"thetaserve watchdog: stop failed ({exc}); retrying next minute", flush=True)


if __name__ == "__main__":
    main()
