"""Launch spec, memory estimate and vLLM command shared by every launch (DESIGN.md section 6)."""

import shlex
from importlib.resources import files

from pydantic import BaseModel, ConfigDict

from thetaserve.config import KvCacheDtype

GIB = 1024**3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_OVERHEAD_GB = 3.2  # largest measured, Gemma 4 31B (DESIGN.md section 13.2)
API_KEY_ENV = "VLLM_API_KEY"
# Expanded by the shell on the host, so the key never appears in the rendered command.
_API_KEY_REF = f'"${API_KEY_ENV}"'
# guard.py on the host, as the module vLLM imports for --middleware. A path in the Linux
# container, not on this machine.
GUARD_DIR = "/tmp/thetaserve"  # noqa: S108
GUARD = "thetaserve_guard.RequireKey"


class ResolveError(Exception):
    """The requested combination cannot be launched."""


class Fit(BaseModel):
    """Estimated GPU memory use in GiB."""

    model_config = ConfigDict(frozen=True)

    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    budget_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.overhead_gb

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.budget_gb


class LaunchSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    device: str
    model: str
    weights: str  # label of the weight format, e.g. fp8, awq-int4
    sku: dict[str, str]
    image: str
    repo: str
    port: int
    max_model_len: int
    kv_cache_dtype: KvCacheDtype
    gpu_memory_utilization: float
    vllm_args: list[str]
    secret_env: list[str]
    idle_s: int
    max_life_s: int
    fit: Fit

    def command(self) -> str:
        """The vLLM command line, with the API key left as a shell variable."""
        quoted = (arg if arg == _API_KEY_REF else shlex.quote(arg) for arg in self.vllm_args)
        return " ".join(["vllm", "serve", *quoted])

    def script(self) -> str:
        """Bash script for the host: write the key guard and the watchdog, start the watchdog,
        then exec vLLM (DESIGN.md sections 9 and 15).

        exec makes vLLM the process that receives stop signals.
        """
        return (
            f"mkdir -p {GUARD_DIR}\n{_host_files('guard', 'watchdog')}"
            f"python3 {GUARD_DIR}/thetaserve_watchdog.py {self.port} {self.idle_s} "
            f"{self.max_life_s} &\n"
            f"PYTHONPATH={GUARD_DIR}${{PYTHONPATH:+:$PYTHONPATH}} exec {self.command()}"
        )

    def worker_script(self, health_port: int) -> str:
        """Bash script for a Runpod serverless worker: start the health answer for the load
        balancer, then exec vLLM (DESIGN.md section 8.3).

        No key guard and no watchdog: Runpod's gateway checks the key, and Runpod scales an idle
        worker to zero.
        """
        return (
            f"mkdir -p {GUARD_DIR}\n{_host_files('healthcheck')}"
            f"python3 {GUARD_DIR}/thetaserve_healthcheck.py {self.port} {health_port} &\n"
            f"exec {self.command()}"
        )


def _host_files(*names: str) -> str:
    """Shell here-documents that write these package modules to the host."""
    return "".join(
        f"cat > {GUARD_DIR}/thetaserve_{name}.py <<'EOF'\n"
        f"{files('thetaserve').joinpath(f'{name}.py').read_text('utf-8')}EOF\n"
        for name in names
    )


def estimate(
    weights_gb: float,
    kv_bytes_per_token: int,
    kv_fixed_bytes: int,
    context: int,
    kv_dtype: KvCacheDtype,
    vram_gb: float,
) -> Fit:
    """Memory estimate: weights + KV cache for one full-length sequence + fixed overhead."""
    kv_factor = 0.5 if kv_dtype.startswith("fp8") else 1.0
    kv_bytes = (kv_bytes_per_token * context + kv_fixed_bytes) * kv_factor
    return Fit(
        weights_gb=weights_gb,
        kv_cache_gb=kv_bytes / GIB,
        overhead_gb=DEFAULT_OVERHEAD_GB,
        budget_gb=vram_gb * DEFAULT_GPU_MEMORY_UTILIZATION,
    )


def vllm_args(
    repo: str,
    port: int,
    context: int,
    kv_dtype: KvCacheDtype,
    extra: list[str],
    *,
    guarded: bool = True,
) -> list[str]:
    """`vllm serve` arguments. `guarded`: vLLM checks the per-launch key, not the provider."""
    key = ["--api-key", _API_KEY_REF, "--middleware", GUARD] if guarded else []
    return [
        repo,
        "--port", str(port),
        *key,
        "--max-model-len", str(context),
        "--kv-cache-dtype", kv_dtype,
        "--gpu-memory-utilization", str(DEFAULT_GPU_MEMORY_UTILIZATION),
        *extra,
    ]  # fmt: skip
