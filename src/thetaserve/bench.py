"""Benchmark workload and result file (DESIGN.md section 10)."""

import hashlib
import json
import re
import statistics
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from thetaserve.spec import LaunchSpec

PROMPT_LENGTHS = (8192, 32768, 131072)  # time to first token; those above the context are skipped
DECODE_TOKENS = 256
ROUNDS = 3
TOOL_REPEATS = 5  # at temperature 0.7: 50 calls, enough to see a few-percent failure rate
# vLLM start-up log lines (v0.32.0) -> measured memory and download time.
_STARTUP = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "activation_peak_gib": re.compile(r"([\d.]+) GiB for peak activation"),
    "cuda_graph_gib": re.compile(r"([\d.]+) GiB for CUDAGraph memory"),
    "download_s": re.compile(r"Time spent downloading weights for \S+: ([\d.]+) seconds"),
}
# Tool-call reliability, the check 4-bit weights are known to fail first (DESIGN.md 5.1, 10):
# (prompt, function, expected arguments).
TOOL_CASES = [
    ("What's the weather in Oslo?", "get_weather", {"city": "Oslo"}),
    ("Is it raining in Tokyo right now?", "get_weather", {"city": "Tokyo"}),
    ("Read the file src/main.py.", "read_file", {"path": "src/main.py"}),
    ("Show me what is in README.md", "read_file", {"path": "README.md"}),
    ("Run the command `ls -la`.", "run_command", {"command": "ls -la"}),
    ("Execute pytest -q in the shell.", "run_command", {"command": "pytest -q"}),
    ("Replace foo with bar in app.py.", "edit_file",
     {"path": "app.py", "old": "foo", "new": "bar"}),
    ("In config.yaml change 8000 to 9000.", "edit_file",
     {"path": "config.yaml", "old": "8000", "new": "9000"}),
    ("Search the code for TODO.", "search", {"pattern": "TODO"}),
    ("Find every use of parse_duration.", "search", {"pattern": "parse_duration"}),
]  # fmt: skip
_STRING = {"type": "string"}
TOOLS = [
    {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": dict.fromkeys(args, _STRING), "required": list(args)}}}
    for name, desc, args in [
        ("get_weather", "Current weather for a city", ["city"]),
        ("read_file", "Read a file in the repository", ["path"]),
        ("run_command", "Run a shell command", ["command"]),
        ("edit_file", "Replace the text `old` with `new` in a file", ["path", "old", "new"]),
        ("search", "Search the repository for a regular expression", ["pattern"]),
    ]
]  # fmt: skip


@dataclass
class Stream:
    """One streamed completion, timed on this machine (proxy latency included)."""

    ttft_s: float | None
    total_s: float
    prompt_tokens: int
    completion_tokens: int
    start: float
    error: str | None = None

    @property
    def decode_tps(self) -> float | None:
        """Tokens per second after the first token."""
        if self.ttft_s is None or self.completion_tokens < 2 or self.total_s <= self.ttft_s:
            return None
        return (self.completion_tokens - 1) / (self.total_s - self.ttft_s)


def parse_startup(lines: Iterable[str]) -> dict[str, float | None]:
    """Measured memory and download time from vLLM's start-up log; the last report wins."""
    found: dict[str, float | None] = dict.fromkeys(_STARTUP)
    for line in lines:
        for key, pattern in _STARTUP.items():
            if match := pattern.search(line):
                found[key] = float(match[1].replace(",", ""))
    return found


def stream_chat(
    http: httpx.Client, base_url: str, model: str, prompt: str, max_tokens: int
) -> Stream:
    """Stream one chat completion of exactly `max_tokens` tokens (vLLM's ignore_eos)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.monotonic()
    first: float | None = None
    usage: dict[str, int] = {}
    try:
        with http.stream("POST", f"{base_url}/chat/completions", json=body) as resp:
            if not resp.is_success:
                resp.read()
                return Stream(None, time.monotonic() - start, 0, 0, start,
                              f"HTTP {resp.status_code}: {resp.text[:200]}")  # fmt: skip
            for raw in resp.iter_lines():
                if not raw.startswith("data:") or raw.strip() == "data: [DONE]":
                    continue
                chunk = json.loads(raw[5:])
                usage = chunk.get("usage") or usage
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if first is None and any(
                    delta.get(k)
                    for k in ("content", "reasoning", "reasoning_content", "tool_calls")
                ):
                    first = time.monotonic()  # fmt: skip
    except httpx.HTTPError as exc:
        return Stream(None, time.monotonic() - start, 0, 0, start, type(exc).__name__)
    return Stream(
        None if first is None else first - start,
        time.monotonic() - start,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        start,
    )


def time_to_first_token(
    http: httpx.Client, base_url: str, model: str, context: int
) -> list[dict[str, Any]]:
    """Time to first token for long prompts. A fresh prefix per prompt defeats prefix caching."""
    results = []
    for target in (n for n in PROMPT_LENGTHS if n <= context):
        # " the" is one token in both launch models' tokenizers; leave room for the template.
        prompt = f"[{uuid.uuid4()}] Summarise:" + " the" * (min(target, context) - 128)
        s = stream_chat(http, base_url, model, prompt, max_tokens=16)
        results.append({"target_tokens": target, "prompt_tokens": s.prompt_tokens,
                        "ttft_s": s.ttft_s, "error": s.error})  # fmt: skip
    return results


def decode(http: httpx.Client, base_url: str, model: str, concurrency: int) -> dict[str, Any]:
    """Decode speed with `concurrency` parallel requests, ROUNDS times: per request and total."""

    def one(prompt: str) -> Stream:
        return stream_chat(http, base_url, model, prompt, DECODE_TOKENS)

    rounds: list[dict[str, Any]] = []
    for _ in range(ROUNDS):
        prompts = [f"[{uuid.uuid4()}] Count upward from 1." for _ in range(concurrency)]
        with ThreadPoolExecutor(concurrency) as pool:
            streams = list(pool.map(one, prompts))
        if errors := [s.error for s in streams if s.error]:
            rounds.append({"error": errors[0]})
            continue
        wall = max(s.start + s.total_s for s in streams) - min(s.start for s in streams)
        rounds.append(
            {
                "per_request_tps": _median(s.decode_tps for s in streams),
                "total_tps": sum(s.completion_tokens for s in streams) / wall,
                "ttft_s": _median(s.ttft_s for s in streams),
            }
        )
    ok = [r for r in rounds if "error" not in r]
    return {
        "concurrency": concurrency,
        "per_request_tps": _median(r["per_request_tps"] for r in ok),
        "total_tps": _median(r["total_tps"] for r in ok),
        "rounds": rounds,
    }


def tool_calls(http: httpx.Client, base_url: str, model: str) -> dict[str, Any]:
    """Share of TOOL_CASES x TOOL_REPEATS answered with the right function and arguments."""
    cases = []
    for prompt, name, expected in TOOL_CASES * TOOL_REPEATS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "temperature": 0.7,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        got: Any = None
        try:
            resp = http.post(f"{base_url}/chat/completions", json=body)
            calls = resp.json()["choices"][0]["message"].get("tool_calls") or []
            got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]
            passed = got == [(name, expected)]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            got, passed = type(exc).__name__, False
        cases.append({"prompt": prompt, "passed": passed, "got": got})
    return {"passed": sum(c["passed"] for c in cases), "total": len(cases), "cases": cases}


def image_digest(image: str) -> str | None:
    """The Docker Hub digest of an image tag, or None if Docker Hub does not answer."""
    repo, _, tag = image.partition(":")
    try:
        resp = httpx.get(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag or 'latest'}", timeout=15.0
        )
        return str(resp.json()["digest"]) if resp.is_success else None
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def config_hash(spec: LaunchSpec, cloud_type: str) -> str:
    """Same hash for the same image, weights, vLLM arguments, GPU and cloud."""
    parts = [spec.image, spec.repo, spec.device, cloud_type, *spec.vllm_args]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def cost_per_million(cost_per_hr: float | None, tokens_per_s: float | None) -> float | None:
    if not cost_per_hr or not tokens_per_s:
        return None
    return cost_per_hr / (tokens_per_s * 3600) * 1e6


def result_path(out: Path, spec: LaunchSpec, provider: str, date: str) -> Path:
    device = re.sub(r"[^a-z0-9.]+", "-", spec.device.lower()).strip("-")
    return out / spec.model.replace("/", "--") / f"{device}-{spec.weights}-{provider}-{date}.json"


def write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n", encoding="utf-8")


def _median(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _jsonable(value: Any) -> Any:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else str(value)
