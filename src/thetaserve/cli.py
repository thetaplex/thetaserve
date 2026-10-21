"""Command-line interface."""

import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from thetaserve import __version__, health, options, secrets, state
from thetaserve.config import Catalog, ConfigError, KvCacheDtype, load_catalog
from thetaserve.profiles import (
    API_KEY_ENV,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_OVERHEAD_GB,
    LaunchSpec,
    Overrides,
    ResolveError,
    resolve,
)
from thetaserve.providers import RUNPOD_SECRET, Provider, ProviderError, get_provider
from thetaserve.providers.runpod import RunPod
from thetaserve.state import Deployment, DeploymentNotFound

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
HF_SECRET = "hf"  # noqa: S105 - name of a secrets-store entry, not a secret
HF_ENV = "HF_TOKEN"
RUNNING_TIMEOUT_S = 20 * 60
READY_TIMEOUT_S = 40 * 60
_STOCK = {1: "low", 2: "medium", 3: "high"}

NameArg = Annotated[str | None, typer.Argument(help="Deployment name; optional if only one.")]


class KvDtype(StrEnum):
    auto = "auto"
    fp8 = "fp8"
    fp8_e4m3 = "fp8_e4m3"
    fp8_e5m2 = "fp8_e5m2"


class Precision(StrEnum):
    bit4 = "4bit"
    bit8 = "8bit"
    bit16 = "16bit"


class Kind(StrEnum):
    models = "models"
    devices = "devices"
    providers = "providers"


class Shell(StrEnum):
    posix = "posix"
    fish = "fish"
    powershell = "powershell"
    cmd = "cmd"


def _fail(message: str, code: int = 1) -> NoReturn:
    err.print(f"[red]error:[/red] {escape(message)}", highlight=False)
    raise typer.Exit(code)


@contextmanager
def _errors() -> Iterator[None]:
    """Turn expected failures into a one-line message and a non-zero exit."""
    try:
        yield
    except ConfigError as exc:
        _fail(str(exc), code=2)
    except (ResolveError, ProviderError, health.NotReady, DeploymentNotFound) as exc:
        _fail(str(exc))


def _catalog() -> Catalog:
    with _errors():
        return load_catalog()


def parse_duration(value: str) -> int:
    """'90s', '30m', '10h' or combinations like '1h30m' -> seconds."""
    parts = _DURATION.findall(value)
    if not parts or "".join(n + u for n, u in parts) != value:
        raise typer.BadParameter(f"{value!r} is not a duration like 30m, 10h or 1h30m")
    return sum(int(n) * _UNIT_S[u] for n, u in parts)


def _human(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _confirm(question: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        _fail(f"{question} Pass --yes to confirm when not running interactively.")
    if not typer.confirm(question, default=False):
        raise typer.Exit(1)


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


@app.command()
def setup() -> None:
    """Store provider credentials and a Hugging Face token."""
    catalog = _catalog()
    env = catalog.providers.runpod.api_key_env
    if os.environ.get(env):
        out.print(f"Runpod API key: using ${env} from the environment.")
    else:
        existing = secrets.get(RUNPOD_SECRET)
        hint = " (Enter keeps the stored one)" if existing else ""
        key = typer.prompt(f"Runpod API key{hint}", default="", hide_input=True, show_default=False)
        if key:
            _stored("Runpod API key", secrets.put(RUNPOD_SECRET, key))
        elif not existing:
            out.print(f"[yellow]No Runpod key stored; set ${env} or run setup again.[/yellow]")

    if os.environ.get(HF_ENV):
        out.print(f"Hugging Face token: using ${HF_ENV} from the environment.")
    else:
        token = typer.prompt(
            "Hugging Face token (optional, Enter to skip)",
            default="",
            hide_input=True,
            show_default=False,
        )
        if token:
            _stored("Hugging Face token", secrets.put(HF_SECRET, token))


def _stored(what: str, where: secrets.Store) -> None:
    if where == "keyring":
        out.print(f"{what} stored in the system keyring.")
    else:
        out.print(
            f"[yellow]{what} stored in {secrets.file_path()} as plaintext (mode 600); "
            "no system keyring is available here.[/yellow]",
            highlight=False,
        )


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
    model: Annotated[
        str, typer.Option(help="Catalogue model such as qwen3.8-27b, or a Hugging Face repo.")
    ],
    device: Annotated[
        str | None,
        typer.Option(help="Pin a device from `list devices`; omit to choose from live options."),
    ] = None,
    use: Annotated[
        str, typer.Option(help="Use case for the options: coding, chat or research.")
    ] = "coding",
    min_precision: Annotated[
        Precision | None, typer.Option(help="Lowest weight precision; overrides the use case.")
    ] = None,
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
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before launching.")] = False,
) -> None:
    """Launch a model on a rented GPU.

    Without --device, lists the cheapest GPUs in stock that fit the model for --use, and asks
    which to launch.
    """
    catalog = _catalog()
    kv: KvCacheDtype | None = kv_cache_dtype.value if kv_cache_dtype else None
    with _errors():
        if device is not None:
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
                    context=context, kv_cache_dtype=kv, extra_args=extra_arg or [], force=force
                ),
            )
            adapter = get_provider(spec.provider, catalog)
        else:
            adapter = get_provider(provider, catalog)
            spec = _pick(
                catalog,
                adapter,
                provider=provider,
                model=model,
                use=use,
                min_bits=int(min_precision.value.removesuffix("bit")) if min_precision else None,
                context=context,
                kv_cache_dtype=kv,
                extra_args=extra_arg or [],
                port=port,
                idle_s=parse_duration(idle),
                max_life_s=parse_duration(max_life),
                name=name,
                choose=not (yes or dry_run),
            )
            yes = True  # choosing an option is the go-ahead

    if dry_run:
        _print_dry_run(spec, adapter)
        return

    with _errors():
        _launch(spec, adapter, yes)


def _pick(
    catalog: Catalog,
    adapter: Provider,
    *,
    provider: str,
    model: str,
    use: str,
    min_bits: int | None,
    context: int | None,
    kv_cache_dtype: KvCacheDtype | None,
    extra_args: list[str],
    port: int,
    idle_s: int,
    max_life_s: int,
    name: str | None,
    choose: bool,
) -> LaunchSpec:
    """Show the cheapest options for `model` and return the chosen one as a LaunchSpec."""
    if not isinstance(adapter, RunPod):
        raise ResolveError(f"{provider} has no live GPU catalogue; pass --device")
    if use not in catalog.presets:
        raise ResolveError(f"unknown use {use!r}; choose one of: {', '.join(catalog.presets)}")
    preset = catalog.presets[use]
    repo, common_args = _model_source(catalog, model)
    if choose and not sys.stdin.isatty():
        _fail("Choosing an option needs a terminal. Pass --yes to take the first option.")

    with out.status(f"Finding GPUs and weights for {repo}…"):
        facts, weights = options.fetch_model(repo, secrets.get(HF_SECRET, env=HF_ENV))
        gpus = adapter.gpus()
        found = options.suggest(
            facts,
            weights,
            gpus,
            preset,
            context=context,
            kv_cache_dtype=kv_cache_dtype,
            min_bits=min_bits,
        )
    if not found:
        raise ResolveError(options.no_options(facts, preset, context))
    best = options.recommended(found, preset)

    table = Table(title=f"{repo} for {use}, {catalog.providers.runpod.cloud_type} cloud")
    table.add_column("#", justify="right")
    table.add_column("GPU", no_wrap=True)
    table.add_column("weights")
    table.add_column("KV cache", no_wrap=True)
    table.add_column("memory (GiB)", no_wrap=True)
    table.add_column("tok/s", justify="right")
    table.add_column("$/h", justify="right")
    table.add_column("stock")
    table.add_column("note")
    for i, o in enumerate(found):
        note = "recommended" if i == best else ""
        if o.weights.bits < preset.min_bits:
            note = f"below {use}'s {preset.min_bits}-bit floor"
        rate = options.tokens_per_s(o, catalog.bandwidth, gpus)
        table.add_row(
            str(i + 1),
            f"{o.gpu.id} ({o.gpu.vram_gb:g} GB)",
            f"{o.weights.bits}-bit {o.weights.label}  {o.weights.repo}",
            "8-bit fp8" if o.kv_cache_dtype.startswith("fp8") else "16-bit",
            f"{o.fit.total_gb:.1f} of {o.fit.budget_gb:.1f}",
            f"~{rate:.0f}" if rate else "?",
            f"{o.gpu.price:.2f}",
            _STOCK[o.gpu.stock],
            note,
        )
    out.print(table)
    out.print(
        f"memory = weights + KV cache for one {found[0].context:,}-token sequence + "
        f"{DEFAULT_OVERHEAD_GB:g} GiB overhead, of VRAM x "
        f"{DEFAULT_GPU_MEMORY_UTILIZATION:g} usable.\n"
        f"tok/s = one request at short context, {options.DECODE_EFFICIENCY:.0%} of memory "
        "bandwidth / weight size; an estimate, not yet measured."
    )
    if preset.tools and "--tool-call-parser" not in common_args:
        out.print(
            f"[yellow]No tool-call parser is known for {repo}; tool calling will not work.[/yellow]"
        )

    choice = best + 1
    while choose:
        choice = typer.prompt(
            "Launch which option (billed until stopped; Ctrl-C to cancel)",
            default=best + 1,
            type=int,
        )
        if 1 <= choice <= len(found):
            break
        err.print(f"Choose 1 to {len(found)}.")
    return options.to_spec(
        found[choice - 1],
        catalog,
        provider=provider,
        model=model,
        common_args=common_args,
        extra_args=extra_args,
        port=port,
        idle_s=idle_s,
        max_life_s=max_life_s,
        name=name,
    )


def _model_source(catalog: Catalog, model: str) -> tuple[str, list[str]]:
    """Base repo and family vLLM flags for a catalogue key or a Hugging Face repo."""
    # ponytail: family flags come from the catalogue entry with the same base repo; ADR 0001
    # keys them by architecture instead, so other repos of a family pick them up. Do that once
    # a second model of a known family is wanted.
    if entry := catalog.models.get(model):
        return entry.hf_family, entry.common_args
    if "/" not in model:
        raise ResolveError(
            f"unknown model {model!r}; use a Hugging Face repo such as Qwen/Qwen3.8-27B or one "
            f"of: {', '.join(sorted(catalog.models))}"
        )
    family = next((m for m in catalog.models.values() if m.hf_family == model), None)
    return model, family.common_args if family else []


def _launch(spec: LaunchSpec, adapter: Provider, yes: bool) -> None:
    if untested := [v for v in (spec.image, spec.repo) if "TODO" in v]:
        _fail(
            f"this profile is not tested yet ({', '.join(untested)}). Set real values in a "
            "user models.yaml (see README) or wait for the M2 pilot."
        )
    if spec.name in state.load().deployments:
        _fail(f"a deployment named {spec.name!r} exists; pass --name or `thetaserve down` it")

    rate = adapter.price_per_hour(spec)
    rate_text = f"${rate:.2f}/h" if rate is not None else "unknown rate"
    out.print(f"{spec.model} on {spec.provider} {spec.device}: {rate_text}, billed until stopped.")
    out.print(
        "[yellow]Automatic idle stop is not implemented yet (M3); "
        f"run `thetaserve stop {spec.name}` when done.[/yellow]"
    )
    _confirm("Launch?", yes)

    api_key = secrets.new_api_key()
    env = {API_KEY_ENV: api_key}
    if hf := secrets.get(HF_SECRET, env=HF_ENV):
        env[HF_ENV] = hf

    # Store the key before creating anything, so a crash never leaves an unreachable host.
    placeholder = Deployment(
        name=spec.name, provider=spec.provider, provider_id="", spec=spec, status="NEW"
    )
    secrets.put(placeholder.secret_name, api_key)

    with out.status("Creating host…"):
        try:
            d = adapter.create(spec, env)
        except BaseException:
            secrets.delete(placeholder.secret_name)
            raise
    state.put(d)
    out.print(f"Created {spec.provider} pod {d.provider_id}.")

    try:
        with out.status("Waiting for the host to start…") as status:
            d = adapter.wait_running(d, RUNNING_TIMEOUT_S)
            state.put(d)
            status.update("Loading the model (first start downloads the weights)…")
            endpoint = adapter.endpoint(d)

            def on_wait(elapsed: float, seen: str) -> None:
                status.update(f"Loading the model… {_human(elapsed)} ({seen})")

            health.wait_ready(endpoint.base_url, api_key, READY_TIMEOUT_S, on_wait=on_wait)
    except (ProviderError, health.NotReady, KeyboardInterrupt) as exc:
        err.print(
            f"[red]error:[/red] {escape(str(exc) or 'interrupted')}\n"
            f"The host is still running and billing. Inspect with `thetaserve logs {d.name}`; "
            f"remove with `thetaserve down {d.name}`."
        )
        raise typer.Exit(1) from None

    kv_tokens = health.kv_cache_tokens(adapter.logs(d, follow=False, tail=2000))
    d.max_context = min(spec.max_model_len, kv_tokens) if kv_tokens else None
    d.cost_per_hr = d.cost_per_hr or rate
    state.put(d)

    out.print(f"\n[green]Ready.[/green] {d.spec.repo} is serving at {endpoint.base_url}")
    if kv_tokens is not None:
        out.print(
            f"KV cache holds {kv_tokens:,} tokens (configured context {spec.max_model_len:,})."
        )
    out.print(f"API key {secrets.mask(api_key)}. For export lines: thetaserve env {d.name}")


def _print_dry_run(spec: LaunchSpec, adapter: Provider) -> None:
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
        ("secrets", ", ".join(f"{k}={MASK}" for k in spec.secret_env) + f", {HF_ENV} if set"),
    ]:
        summary.add_row(f"[bold]{label}[/bold]", escape(value))

    out.print("[bold]Dry run[/bold]: nothing will be created.\n")
    out.print(summary)
    out.print("\n[bold]vLLM command[/bold]")
    out.print(spec.command(), markup=False, highlight=False, soft_wrap=True)
    out.print(f"\n[bold]{spec.provider} request[/bold]")
    out.print_json(json.dumps(adapter.plan(spec)))


def _deployment(name: str | None) -> tuple[Deployment, Provider]:
    d = state.resolve_name(state.load(), name)
    return d, get_provider(d.provider, _catalog())


@app.command()
def status(name: NameArg = None) -> None:
    """Show deployments: status, endpoint, uptime and cost so far."""
    with _errors():
        deployments = state.load().deployments
        if name is not None:
            deployments = {name: state.resolve_name(state.load(), name)}
    if not deployments:
        out.print("No deployments. Launch one with `thetaserve up`.")
        return

    catalog = _catalog()
    table = Table("name", "model / device", "status", "uptime", "rate", "cost so far", "endpoint")
    now = datetime.now(UTC)
    for d in deployments.values():
        try:
            d = get_provider(d.provider, catalog).refresh(d)
            state.put(d)
            shown = d.status
        except ProviderError as exc:
            shown = f"{d.status} (stale: {exc})"
        up_s = (now - d.started_at).total_seconds() if d.started_at and d.status == "RUNNING" else 0
        rate = f"${d.cost_per_hr:.2f}/h" if d.cost_per_hr else "-"
        cost = f"${d.cost_per_hr * up_s / 3600:.2f}" if d.cost_per_hr and up_s else "-"
        table.add_row(
            d.name,
            f"{d.spec.model} / {d.spec.device}",
            shown,
            _human(up_s) if up_s else "-",
            rate,
            cost,
            d.base_url or "-",
        )
    out.print(table)
    out.print("[dim]Cost so far counts the current run only, at the pod's current rate.[/dim]")


@app.command()
def env(
    name: NameArg = None,
    shell: Annotated[
        Shell | None, typer.Option(help="Output syntax; defaults to your platform's shell.")
    ] = None,
) -> None:
    """Print OPENAI_BASE_URL / OPENAI_API_KEY lines for clients and coding agents."""
    with _errors():
        d = state.resolve_name(state.load(), name)
    key = secrets.get(d.secret_name)
    if key is None or d.base_url is None:
        _fail(f"no stored endpoint or key for {d.name!r}")
    shell = shell or (Shell.powershell if os.name == "nt" else Shell.posix)
    pairs = {"OPENAI_BASE_URL": d.base_url, "OPENAI_API_KEY": key}
    for k, v in pairs.items():
        match shell:
            case Shell.posix:
                line = f"export {k}='{v}'"
            case Shell.fish:
                line = f"set -gx {k} '{v}'"
            case Shell.powershell:
                line = f"$env:{k} = '{v}'"
            case Shell.cmd:
                line = f"set {k}={v}"
        print(line)  # plain print: this output is meant to be eval'd, not styled


@app.command()
def logs(
    name: NameArg = None,
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Keep streaming.")] = False,
    tail: Annotated[int, typer.Option(min=0, max=5000, help="Lines of history.")] = 100,
) -> None:
    """Show the vLLM container's logs."""
    with _errors():
        d, adapter = _deployment(name)
        try:
            for line in adapter.logs(d, follow=follow, tail=tail):
                print(line)
        except KeyboardInterrupt:
            pass


@app.command()
def stop(name: NameArg = None) -> None:
    """Stop compute and keep the disk (cheap pause)."""
    with _errors():
        d, adapter = _deployment(name)
        d = adapter.stop(d)
        state.put(d)
    out.print(f"Stopped {d.name} ({d.status}). Resume with `thetaserve start {d.name}`.")


@app.command()
def start(name: NameArg = None) -> None:
    """Resume a stopped deployment."""
    with _errors():
        d, adapter = _deployment(name)
        d = adapter.start(d)
        state.put(d)
    out.print(f"Starting {d.name} ({d.status}). Check with `thetaserve status {d.name}`.")


@app.command()
def down(
    name: NameArg = None,
    purge: Annotated[
        bool, typer.Option("--purge", help="Also delete the weights cache volume.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask first.")] = False,
) -> None:
    """Destroy the deployment's compute; the cache volume is kept unless --purge."""
    with _errors():
        d, adapter = _deployment(name)
        if purge:
            volume = d.extra.get("volume_id")
            sharing = [
                o.name
                for o in state.load().deployments.values()
                if o.name != d.name and volume and o.extra.get("volume_id") == volume
            ]
            if sharing:
                _fail(f"cannot purge: the cache volume is also used by {', '.join(sharing)}")
        what = "the pod and its cache volume" if purge else "the pod (cache volume kept)"
        _confirm(f"Destroy {d.name}: {what}?", yes)
        adapter.destroy(d, purge=purge)
        state.remove(d.name)
        secrets.delete(d.secret_name)
    out.print(f"Destroyed {d.name}.")
