"""Packaged sample data for dry runs without a Runpod key (DESIGN.md section 5.6).

`python -m thetaserve.samples` fetches it again; this needs the Runpod key and network access.
"""

import json
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

from thetaserve.options import ModelFacts, Weights, fetch_model
from thetaserve.providers.base import Gpu
from thetaserve.providers.runpod import parse_gpus

FILE = "samples.json"
_GPU_FIELDS = ("id", "manufacturer", "memory", "pool", "price", "availability")


def load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(files("thetaserve.catalog").joinpath(FILE).read_text("utf-8"))
    return data


def gpus(data: dict[str, Any], cloud_type: str) -> list[Gpu]:
    return parse_gpus(data["runpod"], cloud_type)


def model(data: dict[str, Any], repo: str) -> tuple[ModelFacts, list[Weights]] | None:
    """Facts and weight variants of `repo`, or None if the sample does not hold it."""
    if (m := data["huggingface"].get(repo)) is None:
        return None
    return ModelFacts(**m["facts"]), [Weights(**w) for w in m["weights"]]


def main() -> None:
    from thetaserve import secrets
    from thetaserve.config import load_catalog
    from thetaserve.providers import RUNPOD_SECRET
    from thetaserve.providers.runpod import RunPod

    with tempfile.TemporaryDirectory() as empty:  # built-in models only, not the user's
        catalog = load_catalog(Path(empty))
    cfg = catalog.providers.runpod
    listing = RunPod(cfg, secrets.get(RUNPOD_SECRET, env=cfg.api_key_env)).gpu_listing()
    hf: dict[str, Any] = {}
    for m in catalog.models.values():
        # ponytail: trusted publishers only, so `up --any-publisher` offline shows the same
        # variants; store every publisher too if offline dry runs need them.
        facts, weights = fetch_model(
            m.hf_family, secrets.get("hf", env="HF_TOKEN"), catalog.publishers
        )
        hf[m.hf_family] = {"facts": asdict(facts), "weights": [asdict(w) for w in weights]}
    data: dict[str, Any] = {
        "fetched": datetime.now(UTC).date().isoformat(),
        "runpod": {
            "gpus": [
                {k: g.get(k) for k in _GPU_FIELDS}
                for g in listing["gpus"]
                if g.get("manufacturer") == "NVIDIA"
            ]
        },
        "huggingface": hf,
    }
    path = Path(__file__).parent / "catalog" / FILE
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}: {len(data['runpod']['gpus'])} GPUs, {len(hf)} models")


if __name__ == "__main__":
    main()
