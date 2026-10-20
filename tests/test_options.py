"""Launch options from model metadata and the GPU catalogue. Hub and Runpod are mocked."""

import json
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from thetaserve.cli import app
from thetaserve.config import Preset
from thetaserve.options import HF_URL, ModelFacts, Weights, model_facts, quantization, suggest
from thetaserve.providers.base import Gpu
from thetaserve.providers.runpod import API_URL

GIB = 1024**3
CODING = Preset(min_context=131072, tools=True, min_bits=8)
CHAT = Preset(min_context=32768, tools=False, min_bits=4)
QWEN_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16,
        "max_position_embeddings": 262144,
    },
}
FACTS = ModelFacts("Qwen/Qwen3.8-27B", 262144, 65536, 0)
WEIGHTS = [
    Weights("Qwen/Qwen3.8-27B", "bf16", 16, 51.7),
    Weights("Qwen/Qwen3.8-27B-FP8", "fp8", 8, 28.75),
    Weights("x/Qwen3.8-27B-INT4", "int4", 4, 18.12),
    Weights("x/Qwen3.8-27B-NVFP4", "nvfp4", 4, 14.0, blackwell_only=True),
]


def test_kv_counts_only_full_attention_layers() -> None:
    facts = model_facts("Qwen/Qwen3.8-27B", QWEN_CONFIG)
    # 16 full-attention layers x 4 KV heads x head dim 256 x K,V x 2 B: DESIGN.md section 6.
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


@pytest.mark.parametrize(
    ("qc", "expected"),
    [
        (None, ("bf16", 16, False)),
        ({"quant_method": "fp8"}, ("fp8", 8, False)),
        ({"quant_method": "awq", "bits": 4}, ("awq-int4", 4, False)),
        (  # Hub listings keep only `format` per group
            {"quant_method": "compressed-tensors", "config_groups": {
                "g0": {"format": "float-quantized", "weights": {"num_bits": 8}},
                "g1": {"format": "nvfp4-pack-quantized", "weights": {"num_bits": 4}}}},
            ("nvfp4", 4, True),
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


def test_suggest_picks_best_precision_per_gpu_cheapest_first() -> None:
    gpus = [
        Gpu("NVIDIA L40S", 48, "ada", 1.09, 2),
        Gpu("NVIDIA A40", 48, "ampere", 0.59, 3),
        Gpu("NVIDIA RTX A6000", 48, "ampere", 0.59, 0),  # sold out
        Gpu("NVIDIA A100 80GB PCIe", 80, "ampere", 1.79, 1),
        Gpu("NVIDIA GeForce RTX 4090", 24, "ada", 0.34, 3),  # only 4-bit fits; below floor
    ]
    found = suggest(FACTS, WEIGHTS, gpus, CODING)
    assert [(o.gpu.id, o.weights.label) for o in found] == [
        ("NVIDIA A40", "fp8"),
        ("NVIDIA L40S", "fp8"),
        ("NVIDIA A100 80GB PCIe", "bf16"),
    ]
    assert found[0].fit.total_gb == pytest.approx(28.75 + 8 + 2)


def test_fp8_kv_and_fp4_weights_only_where_supported() -> None:
    gpus = [
        Gpu("NVIDIA RTX A5000", 24, "ampere", 0.27, 1),
        Gpu("NVIDIA GeForce RTX 4090", 24, "ada", 0.34, 3),
        Gpu("NVIDIA RTX PRO 4000 Blackwell", 24, "blackwell", 0.57, 1),
    ]
    found = {o.gpu.id: o for o in suggest(FACTS, WEIGHTS, gpus, CHAT)}
    # INT4 + BF16 KV at 32K is 22.1 GiB, just over 24 x 0.92; Ampere may not use FP8 KV.
    assert "NVIDIA RTX A5000" not in found
    rtx4090 = found["NVIDIA GeForce RTX 4090"]
    assert (rtx4090.weights.label, rtx4090.kv_cache_dtype) == ("int4", "fp8")
    assert found["NVIDIA RTX PRO 4000 Blackwell"].weights.label == "int4"


def test_up_without_device_lists_options_and_dry_runs_the_first(
    user_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (user_dir / "models.yaml").write_text("vllm_image: vllm/vllm-openai:v0.32.0\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{HF_URL}/Qwen/Qwen3.8-27B/resolve/main/config.json").respond(json=QWEN_CONFIG)
        router.get(f"{HF_URL}/api/models").respond(
            json=[
                {"id": "unsloth/Qwen3.8-27B-GGUF", "tags": ["gguf"], "config": {}},
                {
                    "id": "Qwen/Qwen3.8-27B-FP8",
                    "tags": [],
                    "config": {"quantization_config": {"quant_method": "fp8"}},
                },
            ]
        )
        for repo, gb in (("Qwen/Qwen3.8-27B", 51.7), ("Qwen/Qwen3.8-27B-FP8", 28.75)):
            router.get(f"{HF_URL}/api/models/{repo}/tree/main").respond(
                json=[{"path": "model.safetensors", "size": int(gb * GIB)}, {"path": "x.json"}]
            )
        router.get(f"{API_URL}/v2/catalog/gpus").respond(
            json={
                "gpus": [
                    {
                        "id": "NVIDIA A40",
                        "manufacturer": "NVIDIA",
                        "memory": 48,
                        "pool": "AMPERE_48",
                        "price": {"secure": 0.59},
                        "availability": "HIGH",
                    },
                    {
                        "id": "AMD MI300X",
                        "manufacturer": "AMD",
                        "memory": 192,
                        "pool": None,
                        "price": {"secure": 2.39},
                        "availability": "HIGH",
                    },
                ]
            }
        )
        result = CliRunner().invoke(
            app, ["up", "--provider", "runpod", "--model", "qwen3.8-27b", "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    assert "NVIDIA A40" in result.output
    assert "MI300X" not in result.output
    assert "vllm serve Qwen/Qwen3.8-27B-FP8" in result.output
    assert "--tool-call-parser qwen3_xml" in result.output
    body = json.loads(result.output[result.output.index("{") :])
    assert body["gpu"] == {"id": "NVIDIA A40", "count": 1}
    assert body["name"] == "thetaserve-qwen3.8-27b-a40"
