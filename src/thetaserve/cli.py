"""Command-line interface."""

import json
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from thetaserve import __version__, bench, health, options, samples, secrets, state
from thetaserve._files import write_private
from thetaserve.config import Catalog, ConfigError, KvCacheDtype, load_catalog
from thetaserve.providers import RUNPOD_SECRET, Provider, ProviderError, get_provider
from thetaserve.providers.base import Gpu
from thetaserve.providers.runpod import RunPod
from thetaserve.spec import (
    API_KEY_ENV,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_OVERHEAD_GB,
    LaunchSpec,
    ResolveError,
)
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
CRASH_CHECK_S = 60
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
    providers = "providers"


class Shell(StrEnum):
    posix = "posix"
    fish = "fish"
    powershell = "powershell"
    cmd = "cmd"


class Cloud(StrEnum):
    secure = "SECURE"  # Runpod-owned data-centre hardware
    community = "COMMUNITY"  # community-hosted hardware


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


def _with_cloud(catalog: Catalog, cloud: Cloud) -> Catalog:
    """Switch the Runpod cloud tier for this run without touching the catalog files."""
    runpod = catalog.providers.runpod.model_copy(update={"cloud_type": cloud.value})
    providers = catalog.providers.model_copy(update={"runpod": runpod})
    return catalog.model_copy(update={"providers": providers})


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
    _credential("Runpod API key", RUNPOD_SECRET, _catalog().providers.runpod.api_key_env)
    _credential("Hugging Face token", HF_SECRET, HF_ENV, optional=True)


def _credential(what: str, name: str, env: str, *, optional: bool = False) -> None:
    """Use the env value if set; otherwise prompt and store it. Enter keeps a stored one."""
    if os.environ.get(env):
        out.print(f"{what}: using ${env} from the environment.")
        return
    existing = secrets.get(name)
    prompt = what
    if existing:
        prompt += " (Enter keeps the stored one)"
    elif optional:
        prompt += " (optional, Enter to skip)"
    value = typer.prompt(prompt, default="", hide_input=True, show_default=False)
    if value:
        _stored(what, secrets.put(name, value))
    elif not existing and not optional:
        out.print(f"[yellow]No {what} stored; set ${env} or run setup again.[/yellow]")


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
    """List the built-in and user models or providers."""
    catalog = _catalog()
    match kind:
        case Kind.models:
            table = Table("model", "hf family", "vLLM flags")
            for key, m in sorted(catalog.models.items()):
                table.add_row(key, m.hf_family, " ".join(m.common_args))
        case Kind.providers:
            table = Table("provider")
            for name in catalog.providers.names():
                table.add_row(name)
    out.print(table)


@app.command()
def up(
    provider: Annotated[str, typer.Option(help="Provider, e.g. runpod.")],
    model: Annotated[
        str, typer.Option(help="Catalogue model such as qwen3.8-27b, or a Hugging Face repo.")
    ],
    device: Annotated[
        str | None,
        typer.Option(help="Provider GPU ID such as 'NVIDIA A40'; show options on it only."),
    ] = None,
    use: Annotated[
        str, typer.Option(help="Use case for the options: coding, chat or research.")
    ] = "coding",
    min_precision: Annotated[
        Precision | None, typer.Option(help="Lowest weight precision; overrides the use case.")
    ] = None,
    cloud: Annotated[
        Cloud | None,
        typer.Option(
            "--cloud",
            case_sensitive=False,
            help="Runpod cloud tier; default secure, or cloud_type in providers.yaml.",
        ),
    ] = None,
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port vLLM listens on.")] = 8000,
    context: Annotated[
        int | None, typer.Option(min=1, help="Context length in tokens; overrides --use.")
    ] = None,
    kv_cache_dtype: Annotated[
        KvDtype | None, typer.Option(help="KV-cache dtype; default shows auto and fp8.")
    ] = None,
    any_publisher: Annotated[
        bool,
        typer.Option(
            "--any-publisher", help="Also list weights from publishers not in publishers.yaml."
        ),
    ] = False,
    extra_arg: Annotated[
        list[str] | None, typer.Option(help="Extra vLLM argument; repeat for several.")
    ] = None,
    idle: Annotated[str, typer.Option(help="Stop after this long idle.")] = "30m",
    max_life: Annotated[str, typer.Option(help="Stop after this long regardless.")] = "10h",
    name: Annotated[str | None, typer.Option(help="Deployment name.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would be launched; create nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before launching.")] = False,
) -> None:
    """Launch a model on a rented GPU.

    Lists the cheapest GPU in stock that fits the model for every weight and KV-cache precision,
    and asks which to launch.
    """
    catalog = _catalog()
    if cloud is not None:
        catalog = _with_cloud(catalog, cloud)
    with _errors():
        adapter = get_provider(provider, catalog)
        spec = _pick(
            catalog,
            adapter,
            provider=provider,
            model=model,
            device=device,
            use=use,
            min_bits=int(min_precision.value.removesuffix("bit")) if min_precision else None,
            context=context,
            kv_cache_dtype=kv_cache_dtype.value if kv_cache_dtype else None,
            any_publisher=any_publisher,
            extra_args=extra_arg or [],
            port=port,
            idle_s=parse_duration(idle),
            max_life_s=parse_duration(max_life),
            name=name,
            choose=not (yes or dry_run),
            dry_run=dry_run,
        )

    if dry_run:
        _print_dry_run(spec, adapter)
        return

    with _errors():
        _launch(spec, adapter)


def _pick(
    catalog: Catalog,
    adapter: Provider,
    *,
    provider: str,
    model: str,
    device: str | None,
    use: str,
    min_bits: int | None,
    context: int | None,
    kv_cache_dtype: KvCacheDtype | None,
    any_publisher: bool,
    extra_args: list[str],
    port: int,
    idle_s: int,
    max_life_s: int,
    name: str | None,
    choose: bool,
    dry_run: bool,
    bits: int | None = None,
) -> LaunchSpec:
    """Show the cheapest options for `model` and return the chosen one as a LaunchSpec.

    Choosing an option is the go-ahead to launch it.
    """
    if not isinstance(adapter, RunPod):
        raise ResolveError(f"{provider} has no GPU catalogue")
    if use not in catalog.presets:
        raise ResolveError(f"unknown use {use!r}; choose one of: {', '.join(catalog.presets)}")
    preset = catalog.presets[use]
    repo, common_args, kv_scale = _model_source(catalog, model)
    if choose and not sys.stdin.isatty():
        _fail("Choosing an option needs a terminal. Pass --yes to take the first option.")

    sample = samples.load() if dry_run and not adapter.has_key else None
    if sample:
        err.print(
            f"[yellow]No Runpod key: sample GPU prices, stock and model data from "
            f"{sample['fetched']}, not live. Run `thetaserve setup` for live options.[/yellow]"
        )
    with out.status(f"Finding GPUs and weights for {repo}…"):
        sampled = samples.model(sample, repo) if sample else None
        facts, weights = sampled or options.fetch_model(
            repo,
            secrets.get(HF_SECRET, env=HF_ENV),
            None if any_publisher else catalog.publishers,
        )
        facts = replace(facts, kv_bytes_per_token=round(facts.kv_bytes_per_token * kv_scale))
        if bits is not None:
            weights = [w for w in weights if w.bits == bits]
        gpus = samples.gpus(sample, adapter.cfg.cloud_type) if sample else adapter.gpus()
        if device is not None:
            gpus = _on_device(gpus, device)
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
    for i, o in enumerate(found):
        rate = options.tokens_per_s(o, catalog.bandwidth, gpus, weights)
        untrusted = not options.trusted(o.weights.repo, repo, catalog.publishers)
        table.add_row(
            str(i + 1),
            f"{o.gpu.id} ({o.gpu.vram_gb:g} GB)",
            f"{o.weights.bits}-bit {o.weights.label}  {o.weights.repo}"
            + (" [yellow](untrusted publisher)[/yellow]" if untrusted else ""),
            "8-bit fp8" if o.kv_cache_dtype.startswith("fp8") else "16-bit",
            f"{o.fit.total_gb:.1f} of {o.fit.budget_gb:.1f}",
            f"~{rate:.0f}" if rate else "?",
            f"{o.gpu.price:.2f}",
            _STOCK[o.gpu.stock],
        )
    out.print(table)
    if catalog.providers.runpod.network_volume and not adapter.volume:
        err.print(
            "[yellow]Community Cloud pods cannot mount a network volume: no weights cache, so "
            "every start downloads the weights.[/yellow]"
        )
    out.print(
        f"memory = weights + KV cache for one {found[0].context:,}-token sequence + "
        f"{DEFAULT_OVERHEAD_GB:g} GiB overhead, of VRAM x "
        f"{DEFAULT_GPU_MEMORY_UTILIZATION:g} usable.\n"
        f"tok/s = one request at short context, {options.DECODE_EFFICIENCY:.0%} of memory "
        "bandwidth / weight size (4-bit timed as 8-bit); an estimate, not yet measured."
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


def _model_source(catalog: Catalog, model: str) -> tuple[str, list[str], float]:
    """Base repo, family vLLM flags and KV scale for a catalogue key or a Hugging Face repo."""
    # ponytail: family flags come from the catalogue entry with the same base repo; DESIGN.md 5.4
    # keys them by architecture instead, so other repos of a family pick them up. Do that once
    # a second model of a known family is wanted.
    if entry := catalog.models.get(model):
        return entry.hf_family, entry.common_args, entry.kv_scale
    if "/" not in model:
        raise ResolveError(
            f"unknown model {model!r}; use a Hugging Face repo such as Qwen/Qwen3.8-27B or one "
            f"of: {', '.join(sorted(catalog.models))}"
        )
    family = next((m for m in catalog.models.values() if m.hf_family == model), None)
    return model, *((family.common_args, family.kv_scale) if family else ([], 1.0))


def _on_device(gpus: list[Gpu], device: str) -> list[Gpu]:
    on = [g for g in gpus if g.id == device]
    if not on:
        raise ResolveError(
            f"unknown GPU {device!r}; use a Runpod GPU ID such as 'NVIDIA A40'. "
            "Leave out --device to see the options."
        )
    if not on[0].stock:
        raise ResolveError(f"{device} is sold out; leave out --device to see other GPUs")
    return on


def _launch(spec: LaunchSpec, adapter: Provider) -> tuple[Deployment, str]:
    """Create the host and wait until vLLM answers; returns the deployment and its API key."""
    if "TODO" in spec.image:
        _fail(
            f"the vLLM image is not pinned ({spec.image}). Set vllm_image to a tag in "
            "your user models.yaml."
        )
    if spec.name in state.load().deployments:
        _fail(f"a deployment named {spec.name!r} exists; pass --name or `thetaserve down` it")

    rate = adapter.price_per_hour(spec)
    rate_text = f"${rate:.2f}/h" if rate is not None else "unknown rate"
    out.print(f"{spec.model} on {spec.provider} {spec.device}: {rate_text}, billed until stopped.")
    out.print(
        f"It stops itself after {_human(spec.idle_s)} idle or {_human(spec.max_life_s)} in "
        f"total; `thetaserve start {spec.name}` resumes it."
    )
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

            checked = 0.0

            def on_wait(elapsed: float, seen: str) -> None:
                nonlocal checked
                status.update(f"Loading the model… {_human(elapsed)} ({seen})")
                if elapsed - checked >= CRASH_CHECK_S:  # vLLM restarts in a loop on a crash
                    checked = elapsed
                    lines = list(adapter.logs(d, follow=False, tail=500))
                    if line := health.crash_line(lines):
                        saved = save_log(d.name, lines, api_key)
                        raise health.NotReady(f"vLLM crashed: {line}\nFull log: {saved}")

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
    return d, api_key


@app.command("bench")
def bench_(
    provider: Annotated[str, typer.Option(help="Provider, e.g. runpod.")],
    model: Annotated[str, typer.Option(help="Catalogue model or Hugging Face repo.")],
    device: Annotated[str, typer.Option(help="Provider GPU ID such as 'NVIDIA A40'.")],
    use: Annotated[str, typer.Option(help="Use case that sets the context.")] = "coding",
    precision: Annotated[
        Precision | None, typer.Option(help="Weight precision; default the recommended one.")
    ] = None,
    cloud: Annotated[
        Cloud | None,
        typer.Option(
            "--cloud",
            case_sensitive=False,
            help="Runpod cloud tier; default secure, or cloud_type in providers.yaml.",
        ),
    ] = None,
    context: Annotated[int | None, typer.Option(min=1, help="Context length in tokens.")] = None,
    kv_cache_dtype: Annotated[KvDtype | None, typer.Option(help="KV-cache dtype.")] = None,
    out_dir: Annotated[Path, typer.Option("--out", help="Results directory.")] = Path("results"),
    keep: Annotated[bool, typer.Option("--keep", help="Keep the deployment afterwards.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before launching.")] = False,
) -> None:
    """Launch the recommended option on one GPU, run the fixed workload, write one JSON result,
    then remove the deployment (DESIGN.md section 10)."""
    catalog = _catalog()
    if cloud is not None:
        catalog = _with_cloud(catalog, cloud)
    with _errors():
        spec = _pick(
            catalog,
            get_provider(provider, catalog),
            provider=provider,
            model=model,
            device=device,
            use=use,
            min_bits=None,
            bits=int(precision.value.removesuffix("bit")) if precision else None,
            context=context,
            kv_cache_dtype=kv_cache_dtype.value if kv_cache_dtype else None,
            any_publisher=False,
            extra_args=[],
            port=8000,
            idle_s=30 * 60,
            max_life_s=3 * 3600,  # a stuck run stops itself
            name=None,
            choose=False,
            dry_run=False,
        )
        spec = spec.model_copy(update={"name": f"bench-{spec.name}-{spec.weights}"})
        # No cache volume, so the pod can go to any data centre with the GPU; the adapter sizes
        # the container disk to hold the weights.
        runpod = catalog.providers.runpod.model_copy(update={"network_volume": None})
        catalog = catalog.model_copy(
            update={"providers": catalog.providers.model_copy(update={"runpod": runpod})}
        )
        adapter = get_provider(provider, catalog)
    _confirm(f"Benchmark option 1 ({spec.repo} on {spec.device}), billed until done?", yes)
    started = datetime.now(UTC)
    try:
        with _errors():
            t0 = time.monotonic()
            d, api_key = _launch(spec, adapter)
            result = _bench_run(d, api_key, adapter, catalog, time.monotonic() - t0)
    finally:
        if not keep and spec.name in state.load().deployments:
            d = state.load().deployments[spec.name]
            with _errors():
                adapter.destroy(d)
                state.remove(d.name)
                secrets.delete(d.secret_name)
            out.print(f"Destroyed {d.name}.")
    path = bench.result_path(out_dir, spec, provider, started.date().isoformat())
    bench.write(path, {"started": started.isoformat(timespec="seconds"), **result})
    _print_bench(result)
    out.print(f"Wrote {path}")


def _bench_run(
    d: Deployment, api_key: str, adapter: Provider, catalog: Catalog, ready_s: float
) -> dict[str, object]:
    spec, base_url = d.spec, d.base_url or ""
    args = spec.vllm_args
    cloud = catalog.providers.runpod.cloud_type
    measured = bench.parse_startup(adapter.logs(d, follow=False, tail=2000))
    timeout = httpx.Timeout(30.0, read=300.0)
    auth = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(headers=auth, timeout=timeout) as http, out.status("Benchmark…") as status:
        version = http.get(f"{base_url.removesuffix('/v1')}/version").json().get("version")
        status.update("Benchmark: time to first token…")
        ttft = bench.time_to_first_token(http, base_url, spec.repo, spec.max_model_len)
        status.update("Benchmark: decode at 1 and 4 requests…")
        decode = [bench.decode(http, base_url, spec.repo, n) for n in (1, 4)]
        status.update("Benchmark: tool calls…")
        tools = bench.tool_calls(http, base_url, spec.repo)
    return {
        "schema_version": 1,
        "thetaserve_version": __version__,
        "config_hash": bench.config_hash(spec, cloud),
        "provider": d.provider,
        "cloud_type": cloud,
        "device": spec.device,
        "vram_gb": round(spec.fit.budget_gb / spec.gpu_memory_utilization, 1),
        "data_center": d.extra.get("data_center"),
        "cost_per_hr": d.cost_per_hr,
        "model": spec.model,
        "repo": spec.repo,
        "revision": args[args.index("--revision") + 1] if "--revision" in args else "main",
        "weights": spec.weights,
        "context": spec.max_model_len,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "image": spec.image,
        "image_digest": bench.image_digest(spec.image),
        "vllm_version": version,
        "vllm_args": args,
        "estimate": {**spec.fit.model_dump(), "kv_scale": _model_source(catalog, spec.model)[2]},
        "measured": measured,
        "time_to_ready_s": round(ready_s, 1),
        "ttft": ttft,
        "decode": decode,
        "cost_per_million_output_tokens": {
            f"concurrency_{r['concurrency']}": bench.cost_per_million(d.cost_per_hr, r["total_tps"])
            for r in decode
        },
        "tool_calls": tools,
    }


def _print_bench(r: dict[str, object]) -> None:
    table = Table.grid(padding=(0, 2))
    decode = {d["concurrency"]: d for d in r["decode"]}  # type: ignore[attr-defined]
    for label, value in [
        ("run", f"{r['repo']} on {r['device']}, ${r['cost_per_hr']}/h"),
        ("decode", " / ".join(f"{n} req: {_num(d['per_request_tps'])} tok/s each, "
                              f"{_num(d['total_tps'])} total" for n, d in decode.items())),
        ("ttft", ", ".join(f"{t['target_tokens'] // 1024}K: {_num(t['ttft_s'], 's')}"
                           for t in r["ttft"])),  # type: ignore[attr-defined]
        ("$/M out", ", ".join(f"{k.split('_')[1]} req: {_num(v, '', '$')}"
                              for k, v in r["cost_per_million_output_tokens"].items())),  # type: ignore[attr-defined]
        ("tool calls", f"{r['tool_calls']['passed']}/{r['tool_calls']['total']}"),  # type: ignore[index]
        ("ready", f"{r['time_to_ready_s']} s"),
    ]:  # fmt: skip
        table.add_row(label, escape(str(value)))
    out.print(table)


def _num(value: object, unit: str = "", prefix: str = "") -> str:
    return f"{prefix}{value:.1f}{unit}" if isinstance(value, float | int) else "-"


def save_log(name: str, lines: list[str], api_key: str) -> Path:
    """Keep a crashed server's log, key masked: `down` removes the host and its log."""
    path = state.state_dir() / "logs" / f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    write_private(
        path, "".join(f"{line.rstrip(chr(10))}\n" for line in lines).replace(api_key, MASK)
    )
    return path


def _print_dry_run(spec: LaunchSpec, adapter: Provider) -> None:
    fit = spec.fit
    summary = Table.grid(padding=(0, 2))
    for label, value in [
        ("name", spec.name),
        ("provider", f"{spec.provider} ({', '.join(f'{k}={v}' for k, v in spec.sku.items())})"),
        ("device", spec.device),
        ("model", f"{spec.model} ({spec.weights}, repo {spec.repo})"),
        ("image", spec.image),
        ("port", str(spec.port)),
        ("context", f"{spec.max_model_len} tokens, KV cache {spec.kv_cache_dtype}"),
        (
            "memory",
            f"weights {fit.weights_gb:.1f} + KV {fit.kv_cache_gb:.1f} + overhead "
            f"{fit.overhead_gb:.1f} = {fit.total_gb:.1f} of {fit.budget_gb:.1f} GiB usable",
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
    """Print OPENAI_BASE_URL / OPENAI_API_KEY lines for clients and coding agents.

    Model details and URLs go to stderr, so `eval "$(thetaserve env)"` sees only the exports.
    """
    with _errors():
        d = state.resolve_name(state.load(), name)
    key = secrets.get(d.secret_name)
    if key is None or d.base_url is None:
        _fail(f"no stored endpoint or key for {d.name!r}")
    args, root = d.spec.vllm_args, d.base_url.removesuffix("/v1")
    values = {
        "name": d.name,
        "model": d.spec.repo,  # the "model" field of requests
        "catalogue_model": d.spec.model,
        "weights": d.spec.weights,
        "revision": args[args.index("--revision") + 1] if "--revision" in args else "main",
        "context": d.max_context or d.spec.max_model_len,
        "device": d.spec.device,
        "provider": d.provider,
        "base_url": d.base_url,
        "chat_url": f"{d.base_url}/chat/completions",
        "completions_url": f"{d.base_url}/completions",
        "models_url": f"{d.base_url}/models",
        "metrics_url": f"{root}/metrics",
        "health_url": f"{root}/health",  # the only URL that needs no key
    }
    info = Table.grid(padding=(0, 2))
    for label, value in values.items():
        info.add_row(label, escape(str(value)))
    err.print(info)
    err.print("[dim]Every URL except health_url needs the key (OPENAI_API_KEY).[/dim]")
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
        # codeql[py/clear-text-logging-sensitive-data]
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
