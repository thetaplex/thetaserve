"""Suggest launch options from live model and GPU data (docs/adr/0001-derive-launch-options.md)."""

import re
from dataclasses import dataclass
from typing import Any

import httpx

from thetaserve.config import Catalog, KvCacheDtype, Preset
from thetaserve.profiles import (
    API_KEY_ENV,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    GIB,
    Fit,
    LaunchSpec,
    ResolveError,
    estimate,
    vllm_args,
)
from thetaserve.providers.base import Gpu

HF_URL = "https://huggingface.co"
# ponytail: share of peak memory bandwidth one vLLM request reaches while decoding. A guess
# until `bench` measures it per GPU; replace with calibrated values then.
DECODE_EFFICIENCY = 0.6
_VARIANTS = 40  # most-downloaded quantisations of the base model to consider


@dataclass(frozen=True)
class Weights:
    repo: str
    label: str  # bf16, fp8, awq-int4, int4, nvfp4, ...
    bits: int
    gb: float  # GiB of .safetensors files
    blackwell_only: bool = False


@dataclass(frozen=True)
class ModelFacts:
    repo: str
    max_context: int
    kv_bytes_per_token: int  # BF16, full-attention layers
    kv_fixed_bytes: int  # BF16, sliding-window layers at full window


@dataclass(frozen=True)
class Option:
    gpu: Gpu
    weights: Weights
    context: int
    kv_cache_dtype: KvCacheDtype
    fit: Fit


def fetch_model(repo: str, token: str | None = None) -> tuple[ModelFacts, list[Weights]]:
    """Memory facts of `repo` and the weight variants vLLM can serve, best first per format."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(base_url=HF_URL, headers=headers, follow_redirects=True, timeout=30.0) as hub:
        config = _get(hub, f"/{repo}/resolve/main/config.json", repo)
        listed = _get(
            hub,
            "/api/models",
            repo,
            params=[
                ("filter", f"base_model:quantized:{repo}"),
                ("sort", "downloads"),
                ("limit", str(_VARIANTS)),
                ("expand[]", "config"),
                ("expand[]", "tags"),
            ],
        )
        candidates = [(repo, config.get("quantization_config"))] + [
            (m["id"], (m.get("config") or {}).get("quantization_config"))
            for m in listed
            if not {"gguf", "mlx"} & set(m.get("tags") or [])
            and (m.get("config") or {}).get("quantization_config")
        ]
        best: dict[str, tuple[str, int, bool]] = {}
        repos: dict[str, str] = {}
        for name, qc in candidates:
            if (q := quantization(qc)) and q[0] not in best:
                best[q[0]], repos[q[0]] = q, name
        weights = []
        for label, (_, bits, blackwell_only) in best.items():
            files = _get(hub, f"/api/models/{repos[label]}/tree/main", repos[label])
            size = sum(f.get("size", 0) for f in files if f["path"].endswith(".safetensors"))
            if size:
                weights.append(Weights(repos[label], label, bits, size / GIB, blackwell_only))
    return model_facts(repo, config), weights


def model_facts(repo: str, config: dict[str, Any]) -> ModelFacts:
    """KV-cache size per token and maximum context from a Hugging Face `config.json`."""
    t = config.get("text_config") or config
    try:
        heads = t["num_attention_heads"]
        head_dim = t.get("head_dim") or t["hidden_size"] // heads
        per_layer = 2 * t.get("num_key_value_heads", heads) * head_dim * 2  # K and V, 2 B each
        types = t.get("layer_types") or ["full_attention"] * t["num_hidden_layers"]
        max_context = t["max_position_embeddings"]
    except KeyError as exc:
        raise ResolveError(f"{repo}: config.json has no {exc}; pin --device instead") from None
    # ponytail: linear-attention (Mamba, DeltaNet) state is ignored as small and per-sequence;
    # add it if calibration shows the estimate running short for such models.
    full = sum(kind == "full_attention" for kind in types)
    sliding = sum(kind == "sliding_attention" for kind in types)
    window = t.get("sliding_window") or 0
    return ModelFacts(repo, max_context, per_layer * full, per_layer * sliding * window)


def quantization(qc: dict[str, Any] | None) -> tuple[str, int, bool] | None:
    """(label, weight bits, needs Blackwell) for a `quantization_config`; None if unsupported."""
    if not qc:
        return "bf16", 16, False
    method = qc.get("quant_method")
    if method == "fp8":
        return "fp8", 8, False
    if method in ("awq", "gptq"):
        bits = int(qc.get("bits", 4))
        return f"{method}-int{bits}", bits, False
    if method == "compressed-tensors":
        groups = list((qc.get("config_groups") or {}).values())
        if not groups:
            return None
        bits = min(int((g.get("weights") or {}).get("num_bits", 16)) for g in groups)
        # Hub listings drop `weights.type` but keep the group `format`, e.g. "nvfp4-pack-quantized".
        floating = any(
            (g.get("weights") or {}).get("type") == "float"
            or g.get("format", "").startswith(("float", "nvfp4", "mxfp"))
            for g in groups
            if (g.get("weights") or {}).get("num_bits") == bits
        )
        if bits == 4 and floating:
            return "nvfp4", 4, True
        return f"{'fp' if floating else 'int'}{bits}", bits, False
    return None


def suggest(
    facts: ModelFacts,
    weights: list[Weights],
    gpus: list[Gpu],
    preset: Preset,
    *,
    context: int | None = None,
    kv_cache_dtype: KvCacheDtype | None = None,
    min_bits: int | None = None,
) -> list[Option]:
    """The cheapest GPU in stock for each weight precision x KV-cache precision that fits.

    Highest weight precision first, BF16 KV before FP8 KV. `min_bits` drops lower precisions;
    the preset's floor only decides which option is recommended (see `recommended`).
    """
    context = context or min(preset.min_context or facts.max_context, facts.max_context)
    kv_choices: list[KvCacheDtype] = [kv_cache_dtype] if kv_cache_dtype else ["auto", "fp8"]
    found = []
    for bits in sorted({w.bits for w in weights if w.bits >= (min_bits or 0)}, reverse=True):
        for kv in kv_choices:
            fits = [
                option
                for gpu in gpus
                if gpu.stock and (option := _fit(facts, weights, gpu, bits, kv, context))
            ]
            if fits:
                found.append(min(fits, key=lambda o: (o.gpu.price, -o.gpu.stock)))
    return found


def recommended(found: list[Option], preset: Preset) -> int:
    """Index of the cheapest option meeting the preset's precision floor, else the cheapest."""
    meeting = [o for o in found if o.weights.bits >= preset.min_bits] or found
    return found.index(min(meeting, key=lambda o: (o.gpu.price, -o.weights.bits)))


def tokens_per_s(
    option: Option, bandwidth: dict[str, float], gpus: list[Gpu], weights: list[Weights]
) -> float | None:
    """Estimated decode speed of one request: each token reads every weight once.

    Ignores KV-cache reads, so it is an upper bound that falls as the context fills. Weights
    below 8 bits are timed as the smallest 8-bit variant: unpacking them costs about what the
    smaller read saves, and INT4 measured slower than FP8 on vLLM (imec, 2026-10).
    """
    gpu = option.gpu
    gbps = bandwidth.get(gpu.id)
    if gbps is None and " MIG " in gpu.id:  # a slice gets its share of the parent's bandwidth
        parent_id = gpu.id.split(" MIG ")[0]
        parent = next((g for g in gpus if g.id == parent_id), None)
        if parent and parent_id in bandwidth:
            gbps = bandwidth[parent_id] * gpu.vram_gb / parent.vram_gb
    if gbps is None:
        return None
    read_gb = option.weights.gb
    if option.weights.bits < 8:
        read_gb = min((w.gb for w in weights if w.bits >= 8), default=read_gb)
    return DECODE_EFFICIENCY * gbps * 1e9 / (read_gb * GIB)


def no_options(facts: ModelFacts, preset: Preset, context: int | None) -> str:
    context = context or min(preset.min_context or facts.max_context, facts.max_context)
    return (
        f"no GPU in stock fits {facts.repo} at {context:,} tokens. Try a smaller --context or "
        "another --use, or pin --device."
    )


def to_spec(
    option: Option,
    catalog: Catalog,
    *,
    provider: str,
    model: str,
    common_args: list[str],
    extra_args: list[str],
    port: int,
    idle_s: int,
    max_life_s: int,
    name: str | None,
) -> LaunchSpec:
    o = option
    device = _slug(o.gpu.id.removeprefix("NVIDIA ").removeprefix("GeForce "))
    return LaunchSpec(
        name=name or f"{_slug(model.split('/')[-1])}-{device}",
        provider=provider,
        device=o.gpu.id,
        model=model,
        tier=o.weights.label,
        sku={"gpu_type_id": o.gpu.id},
        image=catalog.vllm_image,
        repo=o.weights.repo,
        port=port,
        max_model_len=o.context,
        kv_cache_dtype=o.kv_cache_dtype,
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
        vllm_args=vllm_args(
            o.weights.repo, port, o.context, o.kv_cache_dtype, [*common_args, *extra_args]
        ),
        secret_env=[API_KEY_ENV],
        idle_s=idle_s,
        max_life_s=max_life_s,
        fit=o.fit,
    )


def _fit(
    facts: ModelFacts,
    weights: list[Weights],
    gpu: Gpu,
    bits: int,
    kv: KvCacheDtype,
    context: int,
) -> Option | None:
    """The first `bits`-bit weights (most downloaded first) that fit on `gpu` with `kv`."""
    # FP8 KV cache on Ampere is unverified (DESIGN.md section 13), so it is never offered there.
    if kv != "auto" and gpu.arch == "ampere":
        return None
    for w in weights:
        if w.bits != bits or (w.blackwell_only and gpu.arch != "blackwell"):
            continue
        fit = estimate(
            w.gb, facts.kv_bytes_per_token, facts.kv_fixed_bytes, context, kv, gpu.vram_gb
        )
        if fit.fits:
            return Option(gpu, w, context, kv, fit)
    return None


def _get(hub: httpx.Client, url: str, repo: str, **kwargs: Any) -> Any:
    try:
        resp = hub.get(url, **kwargs)
    except httpx.HTTPError as exc:
        raise ResolveError(f"Hugging Face request failed: {exc}") from exc
    if resp.status_code in (401, 403, 404):
        raise ResolveError(
            f"Hugging Face has no readable {repo!r} ({resp.status_code}); check the name, or "
            "store a token with `thetaserve setup` for a gated model"
        )
    if not resp.is_success:
        raise ResolveError(f"Hugging Face returned {resp.status_code} for {repo!r}")
    return resp.json()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")
