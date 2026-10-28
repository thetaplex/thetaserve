"""Launch options from model metadata and the GPU catalogue. Hub and Runpod are mocked."""

import json
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from tests.conftest import QWEN_CONFIG
from thetaserve.cli import app
from thetaserve.config import Preset
from thetaserve.options import (
    DECODE_EFFICIENCY,
    ModelFacts,
    Weights,
    model_facts,
    quantization,
    recommended,
    suggest,
    tokens_per_s,
)
from thetaserve.providers.base import Gpu
from thetaserve.spec import DEFAULT_OVERHEAD_GB

GIB = 1024**3
CODING = Preset(min_context=131072, tools=True, min_bits=8)
CHAT = Preset(min_context=32768, tools=False, min_bits=4)
FACTS = ModelFacts("Qwen/Qwen3.8-27B", 262144, 65536, 0)
WEIGHTS = [
    Weights("Qwen/Qwen3.8-27B", "bf16", 16, 51.7),
    Weights("Qwen/Qwen3.8-27B-FP8", "fp8", 8, 28.75),
    Weights("x/Qwen3.8-27B-INT4", "int4", 4, 18.12),
    Weights("x/Qwen3.8-27B-NVFP4", "nvfp4", 4, 14.0, blackwell_only=True),
]


def test_kv_counts_only_full_attention_layers() -> None:
    facts = model_facts("Qwen/Qwen3.8-27B", QWEN_CONFIG)
    # 16 full-attention layers x 4 KV heads x head dim 256 x K,V x 2 B: DESIGN.md section 6.2.
    assert facts.kv_bytes_per_token == 65536
    assert facts.max_context == 262144


def test_sliding_window_layers_add_a_fixed_amount() -> None:
    config = {
        "num_hidden_layers": 3, "num_attention_heads": 8, "num_key_value_heads": 2,
        "hidden_size": 1024, "max_position_embeddings": 8192, "sliding_window": 512,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    }  # fmt: skip
    facts = model_facts("m", config)
    per_layer = 2 * 2 * 128 * 2
    assert (facts.kv_bytes_per_token, facts.kv_fixed_bytes) == (per_layer, per_layer * 2 * 512)


def test_gemma4_full_attention_layers_use_their_own_kv_heads_and_head_size() -> None:
    config = {
        "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16,
        "head_dim": 256, "num_global_key_value_heads": 4, "global_head_dim": 512,
        "sliding_window": 1024, "max_position_embeddings": 262144,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
    }  # fmt: skip
    facts = model_facts("google/gemma-4-31B-it", {"text_config": config})
    # As in models.yaml: 10 x 4 x 512 x K,V x 2 B per token; 50 x 1024 x 16 x 256 x 2 x 2 fixed.
    assert (facts.kv_bytes_per_token, facts.kv_fixed_bytes) == (81920, 838860800)


@pytest.mark.parametrize(
    ("qc", "expected"),
    [
        (None, ("bf16", 16, False)),
        ({"quant_method": "fp8"}, ("fp8", 8, False)),
        ({"quant_method": "mxfp4"}, ("mxfp4", 4, False)),  # gpt-oss; runs on Ampere too
        ({"quant_method": "awq", "bits": 4}, ("awq-int4", 4, False)),
        (  # Hub listings keep only `format` per group
            {"quant_method": "compressed-tensors", "config_groups": {
                "g0": {"format": "float-quantized", "weights": {"num_bits": 8}},
                "g1": {"format": "nvfp4-pack-quantized", "weights": {"num_bits": 4}}}},
            ("nvfp4", 4, True),
        ),
        (
            {"quant_method": "modelopt", "config_groups": {"g": {"weights": {"num_bits": 4}}}},
            ("nvfp4", 4, True),
        ),
        ({"quant_method": "modelopt", "quant_algo": "FP8"}, ("fp8", 8, False)),
        (
            {"quant_method": "modelopt", "quant_algo": "W4A8_AWQ"},
            None,
        ),
        (
            {"quant_method": "compressed-tensors", "config_groups": {
                "g0": {"format": "pack-quantized", "weights": {"num_bits": 4}}}},
            ("int4", 4, False),
        ),
        ({"quant_method": "bitsandbytes"}, None),
    ],
)  # fmt: skip
def test_quantization_labels(qc: dict[str, object] | None, expected: object) -> None:
    assert quantization(qc) == expected


GPUS = [
    Gpu("NVIDIA L40S", 48, "ada", 1.09, 2),
    Gpu("NVIDIA A40", 48, "ampere", 0.59, 3),
    Gpu("NVIDIA RTX A6000", 48, "ampere", 0.49, 0),  # sold out
    Gpu("NVIDIA A100 80GB PCIe", 80, "ampere", 1.79, 1),
    Gpu("NVIDIA H100 80GB HBM3", 80, "hopper", 2.99, 3),
    Gpu("NVIDIA RTX A5000", 24, "ampere", 0.27, 1),
    Gpu("NVIDIA GeForce RTX 4090", 24, "ada", 0.34, 3),
]


def test_cheapest_gpu_for_every_weight_and_kv_precision() -> None:
    found = suggest(FACTS, WEIGHTS, GPUS, CODING)
    assert [(o.weights.label, o.kv_cache_dtype, o.gpu.id) for o in found] == [
        ("fp8", "auto", "NVIDIA A40"),  # the A6000 is cheaper but sold out
        ("int4", "auto", "NVIDIA A40"),  # same price: higher precision first
        ("fp8", "fp8", "NVIDIA L40S"),
        ("int4", "fp8", "NVIDIA L40S"),  # 18.1 + 4.0 + 3.2 GiB is over a 24 GB card's 22.1
        ("bf16", "auto", "NVIDIA A100 80GB PCIe"),
        ("bf16", "fp8", "NVIDIA H100 80GB HBM3"),  # no FP8 KV on Ampere
    ]
    assert found[0].fit.total_gb == pytest.approx(28.75 + 8 + DEFAULT_OVERHEAD_GB)
    assert recommended(found, CODING) == 0
    # Chat allows 4-bit, but on the same A40 at the same price 8-bit wins.
    assert recommended(found, CHAT) == 0
    int4_only = [o for o in found if o.weights.bits == 4]
    assert recommended(int4_only, CODING) == 0  # nothing meets the floor: cheapest


def test_min_bits_drops_lower_precisions_and_fp4_needs_blackwell() -> None:
    blackwell = Gpu("NVIDIA RTX PRO 4000 Blackwell", 24, "blackwell", 0.57, 1)
    found = suggest(FACTS, WEIGHTS, [*GPUS, blackwell], CHAT, min_bits=8)
    assert {o.weights.bits for o in found} == {16, 8}
    four_bit = [o for o in suggest(FACTS, WEIGHTS, [blackwell], CHAT) if o.weights.bits == 4]
    # At 32K only the smaller FP4 weights fit 24 GB: INT4 with FP8 KV is 18.1 + 1.0 + 3.2 GiB,
    # over the 22.1 usable.
    assert [(o.weights.label, o.kv_cache_dtype) for o in four_bit] == [
        ("nvfp4", "auto"),
        ("nvfp4", "fp8"),
    ]
    assert suggest(FACTS, WEIGHTS[-1:], GPUS, CHAT) == []  # NVFP4 only, no Blackwell


def test_up_lists_options_and_dry_runs_the_recommended_one(
    user_dir: Path, monkeypatch: pytest.MonkeyPatch, hub: respx.MockRouter
) -> None:
    (user_dir / "models.yaml").write_text("vllm_image: vllm/vllm-openai:v0.32.0\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    result = CliRunner().invoke(
        app,
        ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--dry-run"],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0, result.output
    assert hub["gpus"].called
    assert "NVIDIA RTX A6000" in result.output
    assert "tok/s" in result.output
    assert "MI300X" not in result.output
    assert "sample" not in result.output
    assert "vllm serve Qwen/Qwen3.8-27B-FP8" in result.output
    assert f"--revision {len('Qwen/Qwen3.8-27B-FP8'):040x}" in result.output
    assert "--tool-call-parser qwen3_xml" in result.output
    body = json.loads(result.output[result.output.index("{") :])
    assert body["gpu"] == {"id": "NVIDIA RTX A6000", "count": 1, "minCudaVersion": "13.0"}
    assert body["name"] == "thetaserve-qwen3.8-27b-rtx-a6000"


def test_untrusted_publishers_only_with_any_publisher(
    monkeypatch: pytest.MonkeyPatch, hub: respx.MockRouter
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    args = ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--use", "chat", "--dry-run"]
    trusted_only = CliRunner().invoke(app, args, env={"COLUMNS": "200"})
    assert trusted_only.exit_code == 0, trusted_only.output
    assert "someone/" not in trusted_only.output

    anyone = CliRunner().invoke(app, [*args, "--any-publisher"], env={"COLUMNS": "200"})
    assert anyone.exit_code == 0, anyone.output
    assert "someone/Qwen3.8-27B-AWQ (untrusted publisher)" in anyone.output


def test_dry_run_without_key_uses_the_packaged_sample(hub: respx.MockRouter) -> None:
    result = CliRunner().invoke(
        app,
        ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--dry-run"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert not hub.calls  # neither Runpod nor the Hugging Face Hub
    assert "sample GPU prices, stock and model data from 20" in result.output
    assert "vllm serve Qwen/Qwen3.8-27B-FP8" in result.output
    assert "--revision " in result.output


def test_launch_without_key_fails_before_listing(hub: respx.MockRouter) -> None:
    result = CliRunner().invoke(
        app, ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--yes"]
    )
    assert result.exit_code == 1
    assert "no Runpod API key" in result.output


@pytest.mark.parametrize(
    ("device", "message"),
    [("NVIDIA H900", "unknown GPU 'NVIDIA H900'"), ("NVIDIA RTX 4090", "is sold out")],
)
def test_device_must_be_a_listed_gpu_in_stock(
    monkeypatch: pytest.MonkeyPatch, hub: respx.MockRouter, device: str, message: str
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    result = CliRunner().invoke(
        app,
        ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--device", device, "--dry-run"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert message in result.output


def test_device_limits_the_options_to_that_gpu(
    monkeypatch: pytest.MonkeyPatch, hub: respx.MockRouter
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    result = CliRunner().invoke(
        app,
        ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--device", "NVIDIA A40",
         "--dry-run"],
        env={"COLUMNS": "200"},
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "RTX A6000" not in result.output
    assert json.loads(result.output[result.output.index("{") :])["gpu"]["id"] == "NVIDIA A40"


def test_tokens_per_s_reads_every_weight_once_4bit_as_8bit_and_slices_share_bandwidth() -> None:
    a40 = suggest(FACTS, WEIGHTS[1:2], [Gpu("NVIDIA A40", 48, "ampere", 0.59, 3)], CODING)[0]
    rate = tokens_per_s(a40, {"NVIDIA A40": 696}, [a40.gpu], WEIGHTS)
    assert rate == pytest.approx(DECODE_EFFICIENCY * 696e9 / (28.75 * GIB))

    parent = Gpu("NVIDIA B300 SXM6 AC", 288, "blackwell", 8.99, 0)
    piece = Gpu("NVIDIA B300 SXM6 AC MIG 1g.36gb", 36, "blackwell", 0.5, 1)
    sliced = suggest(FACTS, WEIGHTS[2:3], [parent, piece], CHAT)[0]
    bandwidth = {"NVIDIA B300 SXM6 AC": 8000.0}
    assert tokens_per_s(sliced, bandwidth, [parent, piece], WEIGHTS) == pytest.approx(
        DECODE_EFFICIENCY * 1000e9 / (28.75 * GIB)
    )
    assert tokens_per_s(sliced, {}, [parent, piece], WEIGHTS) is None
    alone = tokens_per_s(sliced, bandwidth, [parent, piece], WEIGHTS[2:3])  # no 8-bit to compare
    assert alone == pytest.approx(DECODE_EFFICIENCY * 1000e9 / (18.12 * GIB))


def test_kv_scale_from_the_catalogue_raises_the_kv_estimate(hub: respx.MockRouter) -> None:
    result = CliRunner().invoke(
        app, ["up", "--provider", "runpod", "--model", "gemma4-31b", "--dry-run"],
        env={"COLUMNS": "200"},
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    # Sample facts: 80 KiB/token + 0.8 GiB fixed; x1.5 per token at 131,072 tokens, FP8 KV.
    assert "KV 7.9 " in result.output
