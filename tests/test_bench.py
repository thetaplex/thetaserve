import json

import httpx
import respx

from thetaserve import bench

URL = "https://pod-8000.proxy.runpod.net/v1"


def sse(*chunks: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_parse_startup_reads_memory_and_download_lines() -> None:
    lines = [
        "INFO [weight_utils.py:532] Time spent downloading weights for x/y: 103.49 seconds",
        "INFO [model_runner.py:424] Model loading took 31.07 GiB memory and 111.8 seconds",
        "Actual usage is 31.29 GiB for consumed memory (weights + non-torch), 2.84 GiB for peak "
        "activation, and 0.08 GiB for CUDAGraph memory.",
        "INFO [gpu_worker.py:759] Available KV cache memory: 9.46 GiB",
        "INFO [kv_cache_utils.py:2327] GPU KV cache size: 162,789 tokens",
    ]
    assert bench.parse_startup(lines) == {
        "weights_gib": 31.07,
        "kv_cache_gib": 9.46,
        "kv_cache_tokens": 162789,
        "activation_peak_gib": 2.84,
        "cuda_graph_gib": 0.08,
        "download_s": 103.49,
    }
    assert bench.parse_startup([])["download_s"] is None  # weights came from the cache


def test_stream_chat_times_first_token_and_reads_usage() -> None:
    with respx.mock as router:
        router.post(f"{URL}/chat/completions").respond(
            text=sse(
                {"choices": [{"delta": {"role": "assistant"}}]},  # no token yet
                {"choices": [{"delta": {"reasoning": "hm"}}]},
                {"choices": [{"delta": {"content": "1"}}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 256}},
            ),
            headers={"Content-Type": "text/event-stream"},
        )
        with httpx.Client() as http:
            s = bench.stream_chat(http, URL, "m", "hi", 256)
        sent = json.loads(router.calls.last.request.content)
    assert s.error is None
    assert (s.prompt_tokens, s.completion_tokens) == (12, 256)
    assert s.ttft_s is not None
    assert s.ttft_s <= s.total_s
    assert sent["ignore_eos"] is True
    assert sent["max_tokens"] == 256


def test_stream_chat_reports_http_errors() -> None:
    with respx.mock as router:
        router.post(f"{URL}/chat/completions").respond(400, text="too long")
        with httpx.Client() as http:
            s = bench.stream_chat(http, URL, "m", "hi", 16)
    assert s.error == "HTTP 400: too long"
    assert s.decode_tps is None


def test_ttft_skips_prompts_longer_than_the_context() -> None:
    with respx.mock as router:
        router.post(f"{URL}/chat/completions").respond(
            text=sse({"choices": [{"delta": {"content": "x"}}]}),
            headers={"Content-Type": "text/event-stream"},
        )
        with httpx.Client() as http:
            results = bench.time_to_first_token(http, URL, "m", context=32768)
        first, second = (
            json.loads(c.request.content)["messages"][0]["content"] for c in router.calls
        )
    assert [r["target_tokens"] for r in results] == [8192, 32768]
    assert first[:40] != second[:40]  # fresh prefix: no prefix-cache hit


def test_tool_calls_need_the_exact_function_and_arguments() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        city = "Oslo" if "Oslo" in prompt else "Bergen"  # wrong for every other case
        call = {"function": {"name": "get_weather", "arguments": json.dumps({"city": city})}}
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [call]}}]})

    with respx.mock as router:
        router.post(f"{URL}/chat/completions").mock(side_effect=answer)
        with httpx.Client() as http:
            result = bench.tool_calls(http, URL, "m")
    repeats = bench.TOOL_REPEATS
    assert (result["passed"], result["total"]) == (repeats, len(bench.TOOL_CASES) * repeats)


def test_cost_per_million_output_tokens() -> None:
    assert bench.cost_per_million(0.59, 20.0) == 0.59 / 72000 * 1e6
    assert bench.cost_per_million(None, 20.0) is None
