import httpx
import pytest
import respx

from thetaserve import health

BASE = "https://pod1-8000.proxy.runpod.net/v1"


@respx.mock
def test_waits_through_proxy_errors_until_model_listed() -> None:
    route = respx.get(f"{BASE}/models").mock(
        side_effect=[
            httpx.ConnectError("refused"),
            httpx.Response(502),
            httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.8-27B-FP8"}]}),
        ]
    )
    seen: list[str] = []
    ids = health.wait_ready(
        BASE, "key", 600, on_wait=lambda _e, s: seen.append(s), sleep=lambda _s: None
    )
    assert ids == ["Qwen/Qwen3.8-27B-FP8"]
    assert seen == ["ConnectError", "HTTP 502"]
    assert route.calls.last.request.headers["Authorization"] == "Bearer key"


@respx.mock
def test_wrong_key_fails_fast() -> None:
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(401))
    with pytest.raises(health.NotReady, match="401"):
        health.wait_ready(BASE, "key", 600, sleep=lambda _s: None)


@respx.mock
def test_times_out() -> None:
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(503))
    with pytest.raises(health.NotReady, match=r"after 0 s \(last: HTTP 503\)"):
        health.wait_ready(BASE, "key", 0, sleep=lambda _s: None)


def test_kv_cache_tokens_parses_vllm_log() -> None:
    lines = [
        "INFO 11-05 10:00:00 [loader.py:12] Loading weights took 41.2 seconds",
        "INFO 11-05 10:01:02 [kv_cache_utils.py:716] GPU KV cache size: 147,456 tokens",
        "INFO Maximum concurrency for 131,072 tokens per request: 1.12x",
    ]
    assert health.kv_cache_tokens(lines) == 147456
    assert health.kv_cache_tokens(["nothing here"]) is None


def test_crash_line_finds_root_cause_in_vllm_log() -> None:
    # From the M1 pilot (2026-10-22): an engine crash, then the API server's follow-up error.
    lines = [
        "(EngineCore pid=480) ERROR 10-22 10:15:38 [core.py:1510] Traceback (most recent call",
        "ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (187).",
        "RuntimeError: Engine core initialization failed. See root cause above.",
    ]
    assert health.crash_line(lines) == lines[1]
    assert health.crash_line(["vllm serve: error: argument --api-key: expected"]) is not None
    assert health.crash_line(["(APIServer pid=56) INFO 10-22 [entry.py:140] Starting"]) is None
