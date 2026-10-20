# Override the catalogue

Use this when a GPU you want is missing from `list devices`, or a built-in model profile still
holds a `TODO` value (`up` refuses those).

1. Find your user config directory:

   ```sh
   uv run python -c "from thetaserve.config import user_config_dir; print(user_config_dir())"
   ```

2. Create `devices.yaml`, `models.yaml` or `providers.yaml` there. Each top-level key replaces
   the built-in entry with the same key **whole**; fields are not merged within an entry.

## Add a GPU

Look up the Runpod GPU ID (for example in the Runpod console's GPU list), then:

```yaml
# devices.yaml
a40-48g:
  vram_gb: 48
  arch: ampere
  providers:
    runpod: { gpu_type_id: "NVIDIA A40" }
```

## Fill a model profile

Copy the model's whole entry from the built-in `src/thetaserve/catalog/models.yaml`, then
replace the `TODO` values for the tier you need. A minimal 48 GB profile:

```yaml
# models.yaml
vllm_image: vllm/vllm-openai:v0.32.0

qwen3.8-27b:
  hf_family: Qwen/Qwen3.8-27B
  min_vllm: "0.17"
  common_args: [--enable-prefix-caching, --reasoning-parser, qwen3,
                --enable-auto-tool-choice, --tool-call-parser, qwen3_xml]
  kv_bytes_per_token: 65536
  kv_fixed_bytes: 0
  tiers:
    48g: { repo: Qwen/Qwen3.8-27B-FP8, weights_gb: 28.75, max_model_len: 131072, kv_cache_dtype: auto }
```

`weights_gb` is the total size of the repository's `.safetensors` files in GiB.

## Check

```sh
uv run thetaserve list devices
uv run thetaserve up --provider runpod --device a40-48g --model qwen3.8-27b --dry-run
```

A YAML or validation error names the file and the field.
