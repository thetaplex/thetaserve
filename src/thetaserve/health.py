"""Readiness polling and start-up log parsing for the vLLM server."""

import re
import time
from collections.abc import Callable, Iterable

import httpx

POLL_S = 10.0
# vLLM logs e.g. "GPU KV cache size: 147,456 tokens" once memory profiling is done.
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


class NotReady(Exception):
    """The server did not become ready in time, or refused our key."""


def wait_ready(
    base_url: str,
    api_key: str,
    timeout_s: int,
    *,
    on_wait: Callable[[float, str], None] | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Poll GET {base_url}/models until it lists a model; returns the served model IDs.

    Until vLLM is up the provider's proxy answers with 5xx or the connection fails; those
    just mean "keep waiting". A 401 means the key is wrong, which waiting will not fix.
    """
    if client is None:
        with httpx.Client(timeout=httpx.Timeout(15.0)) as owned:
            return wait_ready(
                base_url, api_key, timeout_s, on_wait=on_wait, client=owned, sleep=sleep
            )
    http = client
    start = time.monotonic()
    while True:
        try:
            resp = http.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code == 401:
                raise NotReady("the server rejected the API key (401)")
            if resp.is_success and (ids := [m["id"] for m in resp.json().get("data", [])]):
                return ids
            seen = f"HTTP {resp.status_code}"
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            seen = type(exc).__name__
        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            raise NotReady(f"no answer from {base_url} after {timeout_s} s (last: {seen})")
        if on_wait:
            on_wait(elapsed, seen)
        sleep(POLL_S)


def kv_cache_tokens(lines: Iterable[str]) -> int | None:
    """Tokens of KV cache vLLM allocated, from its start-up log; the last report wins."""
    found = None
    for line in lines:
        if match := _KV_TOKENS.search(line):
            found = int(match[1].replace(",", ""))
    return found
