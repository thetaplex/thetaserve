"""Command-line interface."""

import re
from enum import StrEnum
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from thetaserve import __version__
from thetaserve.config import Catalog, ConfigError, load_catalog
from thetaserve.profiles import LaunchSpec, Overrides, ResolveError, resolve

app = typer.Typer(
    help="Launch open-weight models with vLLM on rented GPUs.",
    no_args_is_help=True,
    add_completion=False,
)
out = Console()
err = Console(stderr=True)

_DURATION = re.compile(r"(\d+)([smh])")
_UNIT_S = {"s": 1, "m": 60, "h": 3600}
MASK = "****"


class KvDtype(StrEnum):
    auto = "auto"
    fp8 = "fp8"
    fp8_e4m3 = "fp8_e4m3"
    fp8_e5m2 = "fp8_e5m2"


class Kind(StrEnum):
    models = "models"
    devices = "devices"
    providers = "providers"


def _fail(message: str, code: int = 1) -> NoReturn:
    err.print(f"[red]error:[/red] {message}", highlight=False)
    raise typer.Exit(code)


def _catalog() -> Catalog:
    try:
        return load_catalog()
    except ConfigError as exc:
        _fail(str(exc), code=2)


def parse_duration(value: str) -> int:
    """'90s', '30m', '10h' or combinations like '1h30m' -> seconds."""
    parts = _DURATION.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        raise typer.BadParameter(f"{value!r} is not a duration like 30m, 10h or 1h30m")
    return sum(int(n) * _UNIT_S[u] for n, u in parts)


def _version(show: bool) -> None:
    if show:
        out.print(f"thetaserve {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    pass


@app.command("list")
def list_(kind: Annotated[Kind, typer.Argument(help="What to list.")]) -> None:
    """List the catalogue of models, devices or providers."""
    catalog = _catalog()
    match kind:
        case Kind.models:
            table = Table("model", "hf family", "tiers", "min vLLM")
            for key, m in sorted(catalog.models.items()):
                tiers = ", ".join(k for _, k, _ in m.tiers_by_size())
                table.add_row(key, m.hf_family, tiers, m.min_vllm)
        case Kind.devices:
            table = Table("device", "VRAM (GiB)", "arch", "providers")
            for key, d in sorted(catalog.devices.items()):
                table.add_row(key, str(d.vram_gb), d.arch, ", ".join(sorted(d.providers)))
        case Kind.providers:
            table = Table("provider", "devices")
            for name in catalog.providers.names():
                devices = sorted(k for k, d in catalog.devices.items() if name in d.providers)
                table.add_row(name, ", ".join(devices))
    out.print(table)


@app.command()
def up(
    provider: Annotated[str, typer.Option(help="Provider, e.g. runpod.")],
    device: Annotated[str, typer.Option(help="Device (GPU class), e.g. a6000-48g.")],
    model: Annotated[str, typer.Option(help="Model, e.g. qwen3.8-27b.")],
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port vLLM listens on.")] = 8000,
    context: Annotated[
        int | None, typer.Option(min=1, help="Override the profile's max context (tokens).")
    ] = None,
    kv_cache_dtype: Annotated[
        KvDtype | None, typer.Option(help="Override the profile's KV-cache dtype.")
    ] = None,
    extra_arg: Annotated[
        list[str] | None, typer.Option(help="Extra vLLM argument; repeat for several.")
    ] = None,
    idle: Annotated[str, typer.Option(help="Stop after this long idle.")] = "30m",
    max_life: Annotated[str, typer.Option(help="Stop after this long regardless.")] = "10h",
    name: Annotated[str | None, typer.Option(help="Deployment name.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would be launched; create nothing.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Skip the memory-fit check.")] = False,
) -> None:
    """Launch a model on a rented GPU."""
    catalog = _catalog()
    try:
        spec = resolve(
            catalog,
            provider=provider,
            device=device,
            model=model,
            port=port,
            idle_s=parse_duration(idle),
            max_life_s=parse_duration(max_life),
            name=name,
            overrides=Overrides(
                context=context,
                kv_cache_dtype=kv_cache_dtype.value if kv_cache_dtype else None,
                extra_args=extra_arg or [],
                force=force,
            ),
        )
    except ResolveError as exc:
        _fail(str(exc))

    if not dry_run:
        _fail("launching is not implemented yet (milestone M1); use --dry-run")
    _print_dry_run(spec)


def _print_dry_run(spec: LaunchSpec) -> None:
    fit = spec.fit
    summary = Table.grid(padding=(0, 2))
    for label, value in [
        ("name", spec.name),
        ("provider", f"{spec.provider} ({', '.join(f'{k}={v}' for k, v in spec.sku.items())})"),
        ("device", spec.device),
        ("model", f"{spec.model} (tier {spec.tier}, repo {spec.repo})"),
        ("image", spec.image),
        ("port", str(spec.port)),
        ("context", f"{spec.max_model_len} tokens, KV cache {spec.kv_cache_dtype}"),
        (
            "memory",
            f"weights {fit.weights_gb:.1f} + KV {fit.kv_cache_gb:.1f} + overhead "
            f"{fit.overhead_gb:.1f} = {fit.total_gb:.1f} of {fit.budget_gb:.1f} GiB usable"
            + ("" if fit.fits else "  (does not fit; --force)"),
        ),
        ("idle stop", f"{spec.idle_s // 60} min"),
        ("max life", f"{spec.max_life_s / 3600:g} h"),
        ("secrets", ", ".join(f"{k}={MASK}" for k in spec.secret_env)),
    ]:
        summary.add_row(f"[bold]{label}[/bold]", escape(value))

    out.print("[bold]Dry run[/bold]: nothing will be created.\n")
    out.print(summary)
    out.print("\n[bold]vLLM command[/bold]")
    out.print(spec.command(), markup=False, highlight=False, soft_wrap=True)
    out.print("\n[dim]Provider request body: available with the provider adapters (M1).[/dim]")
