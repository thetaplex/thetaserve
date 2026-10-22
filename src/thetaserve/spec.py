"""Launch spec, memory estimate and vLLM command shared by every launch (DESIGN.md section 6)."""

import shlex
from importlib.resources import files

from pydantic import BaseModel, ConfigDict

from thetaserve.config import KvCacheDtype

GIB = 1024**3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_OVERHEAD_GB = 2.0
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
        """Bash script for the host: write the key guard, then exec vLLM (DESIGN.md section 9).

        exec makes vLLM the process that receives stop signals.
        """
        guard = files("thetaserve").joinpath("guard.py").read_text("utf-8")
        return (
            f"mkdir -p {GUARD_DIR} && cat > {GUARD_DIR}/thetaserve_guard.py <<'EOF'\n{guard}EOF\n"
            f"PYTHONPATH={GUARD_DIR}${{PYTHONPATH:+:$PYTHONPATH}} exec {self.command()}"
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
    repo: str, port: int, context: int, kv_dtype: KvCacheDtype, extra: list[str]
) -> list[str]:
    return [
        repo,
        "--port", str(port),
        "--api-key", _API_KEY_REF,
        "--middleware", GUARD,
        "--max-model-len", str(context),
        "--kv-cache-dtype", kv_dtype,
        "--gpu-memory-utilization", str(DEFAULT_GPU_MEMORY_UTILIZATION),
        *extra,
    ]  # fmt: skip
