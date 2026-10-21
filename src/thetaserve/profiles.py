"""Resolve provider x device x model into a concrete LaunchSpec (DESIGN.md section 6)."""

import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from thetaserve.config import Arch, Catalog, KvCacheDtype, Tier

GIB = 1024**3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_OVERHEAD_GB = 2.0
API_KEY_ENV = "VLLM_API_KEY"
# Expanded by the shell on the host, so the key never appears in the rendered command.
_API_KEY_REF = f'"${API_KEY_ENV}"'


class ResolveError(Exception):
    """The requested combination cannot be launched."""


@dataclass(frozen=True)
class Overrides:
    """CLI flags; they take precedence over profile values."""

    context: int | None = None
    kv_cache_dtype: KvCacheDtype | None = None
    extra_args: list[str] = field(default_factory=list)
    force: bool = False


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

    @property
    def sequences(self) -> float:
        """Full-length sequences the memory left after weights and overhead holds as KV cache."""
        room = self.budget_gb - self.weights_gb - self.overhead_gb
        return room / self.kv_cache_gb if self.kv_cache_gb else float("inf")


class LaunchSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    device: str
    model: str
    tier: str
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


def resolve(
    catalog: Catalog,
    *,
    provider: str,
    device: str,
    model: str,
    port: int,
    idle_s: int,
    max_life_s: int,
    name: str | None = None,
    overrides: Overrides | None = None,
) -> LaunchSpec:
    overrides = overrides or Overrides()
    if provider not in catalog.providers.names():
        raise ResolveError(_unknown("provider", provider, catalog.providers.names()))
    if device not in catalog.devices:
        raise ResolveError(_unknown("device", device, catalog.devices))
    if model not in catalog.models:
        raise ResolveError(_unknown("model", model, catalog.models))

    dev = catalog.devices[device]
    mod = catalog.models[model]
    sku = dev.providers.get(provider)
    if sku is None:
        raise ResolveError(
            f"device {device!r} is not offered on {provider!r}; "
            f"it is available on: {', '.join(sorted(dev.providers))}"
        )

    # 1. Largest tier that fits the device's memory.
    fitting = [(size, key, t) for size, key, t in mod.tiers_by_size() if size <= dev.vram_gb]
    if not fitting:
        smallest = mod.tiers_by_size()[0][1]
        raise ResolveError(
            f"{model!r} needs at least the {smallest} tier; {device!r} has {dev.vram_gb} GiB"
        )
    _, tier_key, tier = fitting[-1]

    # 2. Architecture overrides, then 3. CLI flags.
    tier = _apply_arch(tier, dev.arch)
    max_model_len = overrides.context or tier.max_model_len
    kv_dtype = overrides.kv_cache_dtype or tier.kv_cache_dtype
    utilization = DEFAULT_GPU_MEMORY_UTILIZATION

    # 4. Memory-fit check.
    fit = estimate(
        tier.weights_gb,
        mod.kv_bytes_per_token,
        mod.kv_fixed_bytes,
        max_model_len,
        kv_dtype,
        dev.vram_gb,
    )
    if not fit.fits and not overrides.force:
        raise ResolveError(
            f"{model!r} at {max_model_len} tokens does not fit {device!r}: "
            f"weights {fit.weights_gb:.1f} + KV cache {fit.kv_cache_gb:.1f} "
            f"+ overhead {fit.overhead_gb:.1f} = {fit.total_gb:.1f} GiB "
            f"> {fit.budget_gb:.1f} GiB usable. Lower --context or pass --force."
        )

    # 5. vLLM arguments.
    args = vllm_args(
        tier.repo,
        port,
        max_model_len,
        kv_dtype,
        [*mod.common_args, *tier.extra_args, *overrides.extra_args],
    )

    return LaunchSpec(
        name=name or f"{model}-{device}",
        provider=provider,
        device=device,
        model=model,
        tier=tier_key,
        sku=sku,
        image=catalog.vllm_image,
        repo=tier.repo,
        port=port,
        max_model_len=max_model_len,
        kv_cache_dtype=kv_dtype,
        gpu_memory_utilization=utilization,
        vllm_args=args,
        secret_env=[API_KEY_ENV],
        idle_s=idle_s,
        max_life_s=max_life_s,
        fit=fit,
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
        "--max-model-len", str(context),
        "--kv-cache-dtype", kv_dtype,
        "--gpu-memory-utilization", str(DEFAULT_GPU_MEMORY_UTILIZATION),
        *extra,
    ]  # fmt: skip


def _apply_arch(tier: Tier, arch: Arch) -> Tier:
    override = tier.arch_overrides.get(arch)
    if override is None:
        return tier
    return tier.model_copy(update=override.model_dump(exclude_none=True))


def _unknown(kind: str, value: str, options: Iterable[str]) -> str:
    return f"unknown {kind} {value!r}; choose one of: {', '.join(sorted(options))}"
