from collections.abc import Iterator
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest
import respx

from thetaserve import config, state
from thetaserve.config import Preset, load_catalog
from thetaserve.options import HF_URL, ModelFacts, Weights, suggest, to_spec
from thetaserve.providers.base import Gpu
from thetaserve.providers.runpod import API_URL
from thetaserve.spec import LaunchSpec

GIB = 1024**3
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
RUNPOD_GPUS = {
    "gpus": [
        {"id": "NVIDIA A40", "manufacturer": "NVIDIA", "memory": 48, "pool": "AMPERE_48",
         "secure": True, "community": False,  # Runpod still quotes a community price
         "price": {"secure": 0.59, "community": 0.21}, "availability": "HIGH"},
        {"id": "NVIDIA RTX A6000", "manufacturer": "NVIDIA", "memory": 48, "pool": "AMPERE_48",
         "secure": True, "community": True,
         "price": {"secure": 0.49, "community": 0.33}, "availability": "HIGH"},
        {"id": "NVIDIA RTX 4090", "manufacturer": "NVIDIA", "memory": 24, "pool": "ADA_24",
         "secure": True, "community": True,
         "price": {"secure": 0.69, "community": 0.48}, "availability": "NONE"},
        {"id": "AMD MI300X", "manufacturer": "AMD", "memory": 192, "pool": None,
         "secure": True, "community": False, "price": {"secure": 2.39}, "availability": "HIGH"},
    ]
}  # fmt: skip


class MemoryKeyring(keyring.backend.KeyringBackend):
    """In-process keyring so tests never read or write the real system keyring."""

    priority = 1

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.items.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.items[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.items.pop((service, username), None) is None:
            raise keyring.errors.PasswordDeleteError(username)


@pytest.fixture(autouse=True)
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config and state at empty temp dirs so real user files never leak in."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr(config, "user_config_dir", lambda: cfg)
    monkeypatch.setattr(state, "state_dir", lambda: tmp_path / "state")
    for var in ("RUNPOD_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return cfg


@pytest.fixture(autouse=True)
def memory_keyring() -> Iterator[MemoryKeyring]:
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)


@pytest.fixture
def spec() -> LaunchSpec:
    """Qwen3.8-27B FP8 on an RTX A6000, named dev."""
    gpu = Gpu("NVIDIA RTX A6000", 48, "ampere", 0.49, 3)
    weights = Weights("Qwen/Qwen3.8-27B-FP8", "fp8", 8, 28.75, revision="a" * 40)
    facts = ModelFacts("Qwen/Qwen3.8-27B", 262144, 65536, 0)
    option = suggest(facts, [weights], [gpu], Preset(min_context=131072, tools=True, min_bits=8))
    return to_spec(
        option[0],
        load_catalog(),
        provider="runpod",
        model="qwen3.8-27b",
        common_args=[],
        extra_args=[],
        port=8000,
        idle_s=1800,
        max_life_s=36000,
        name="dev",
    )


@pytest.fixture
def hub() -> Iterator[respx.MockRouter]:
    """Hub with Qwen3.8-27B (BF16, FP8, untrusted AWQ, GGUF to skip) and the Runpod catalogue."""
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{HF_URL}/Qwen/Qwen3.8-27B/resolve/main/config.json").respond(json=QWEN_CONFIG)
        router.get(f"{HF_URL}/api/models").respond(
            json=[
                {"id": "unsloth/Qwen3.8-27B-GGUF", "tags": ["gguf"], "config": {}},
                {
                    "id": "someone/Qwen3.8-27B-AWQ",
                    "tags": [],
                    "config": {"quantization_config": {"quant_method": "awq", "bits": 4}},
                },
                {
                    "id": "Qwen/Qwen3.8-27B-FP8",
                    "tags": [],
                    "config": {"quantization_config": {"quant_method": "fp8"}},
                },
            ]
        )
        for repo, gb in (
            ("Qwen/Qwen3.8-27B", 51.7),
            ("Qwen/Qwen3.8-27B-FP8", 28.75),
            ("someone/Qwen3.8-27B-AWQ", 18.1),
        ):
            router.get(f"{HF_URL}/api/models/{repo}/revision/main").respond(
                json={
                    "sha": f"{len(repo):040x}",
                    "siblings": [
                        {"rfilename": "model.safetensors", "size": int(gb * GIB)},
                        {"rfilename": "x.json"},
                    ],
                }
            )
        router.get(f"{API_URL}/v2/catalog/gpus", name="gpus").respond(json=RUNPOD_GPUS)
        yield router
