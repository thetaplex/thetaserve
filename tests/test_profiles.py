import pytest

from thetaserve.config import Catalog, load_catalog
from thetaserve.profiles import LaunchSpec, Overrides, ResolveError, resolve


@pytest.fixture
def catalog() -> Catalog:
    return load_catalog()


def _resolve(catalog: Catalog, device: str, model: str, **kw: object) -> LaunchSpec:
    return resolve(
        catalog,
        provider=str(kw.pop("provider", "runpod")),
        device=device,
        model=model,
        port=8000,
        idle_s=1800,
        max_life_s=36000,
        overrides=Overrides(**kw),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("device", "tier"),
    [("rtx4090-24g", "24g"), ("a6000-48g", "48g"), ("h100-80g", "80g")],
)
def test_picks_largest_tier_that_fits(catalog: Catalog, device: str, tier: str) -> None:
    assert _resolve(catalog, device, "qwen3.8-27b").tier == tier


def test_arch_override_applies_on_ampere_only(catalog: Catalog) -> None:
    assert _resolve(catalog, "a6000-48g", "gemma4-31b").kv_cache_dtype == "auto"
    assert _resolve(catalog, "l40s-48g", "gemma4-31b").kv_cache_dtype == "fp8"


def test_cli_overrides_win(catalog: Catalog) -> None:
    spec = _resolve(
        catalog,
        "a6000-48g",
        "qwen3.8-27b",
        context=32768,
        kv_cache_dtype="fp8",
        extra_args=["--seed", "1"],
    )
    assert spec.max_model_len == 32768
    assert spec.kv_cache_dtype == "fp8"
    assert spec.vllm_args[-2:] == ["--seed", "1"]


def test_kv_cache_estimate_matches_design_table(catalog: Catalog) -> None:
    # DESIGN.md section 6: Qwen3.8-27B at 128K is 8.0 GiB in BF16, 4.0 GiB in FP8.
    bf16 = _resolve(catalog, "a6000-48g", "qwen3.8-27b", kv_cache_dtype="auto")
    fp8 = _resolve(catalog, "a6000-48g", "qwen3.8-27b", kv_cache_dtype="fp8")
    assert bf16.fit.kv_cache_gb == pytest.approx(8.0)
    assert fp8.fit.kv_cache_gb == pytest.approx(4.0)


def test_too_much_context_fails_fit_check(catalog: Catalog) -> None:
    with pytest.raises(ResolveError, match=r"does not fit .* GiB usable"):
        _resolve(catalog, "rtx4090-24g", "qwen3.8-27b", context=262144)


def test_force_skips_fit_check(catalog: Catalog) -> None:
    spec = _resolve(catalog, "rtx4090-24g", "qwen3.8-27b", context=262144, force=True)
    assert not spec.fit.fits


def test_device_not_on_provider(catalog: Catalog) -> None:
    with pytest.raises(ResolveError, match="not offered on 'aws'; it is available on: runpod"):
        _resolve(catalog, "a6000-48g", "qwen3.8-27b", provider="aws")


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"provider": "vast", "device": "a6000-48g", "model": "qwen3.8-27b"}, "unknown provider"),
        ({"device": "tpu", "model": "qwen3.8-27b"}, "unknown device 'tpu'; choose one of: a100"),
        ({"device": "a6000-48g", "model": "llama"}, "unknown model 'llama'"),
    ],
)
def test_unknown_names_list_choices(catalog: Catalog, kw: dict[str, str], message: str) -> None:
    device, model = kw.pop("device"), kw.pop("model")
    with pytest.raises(ResolveError, match=message):
        _resolve(catalog, device, model, **kw)


def test_no_tier_small_enough(catalog: Catalog) -> None:
    small = catalog.model_copy(
        update={
            "devices": {"t4-16g": catalog.devices["rtx4090-24g"].model_copy(update={"vram_gb": 16})}
        }
    )
    with pytest.raises(ResolveError, match="needs at least the 24g tier"):
        _resolve(small, "t4-16g", "qwen3.8-27b")
