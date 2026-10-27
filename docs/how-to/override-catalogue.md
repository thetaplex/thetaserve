# Override the catalogue

Use this procedure to change a built-in file. For example: add a use-case preset, use a
different vLLM image, or give a model a short name and its vLLM flags.

1. Find your user config directory in the
   [locations table](../reference/configuration.md#locations). On Linux it is
   `~/.config/thetaserve/`.

2. In that directory, create a file with the name of the built-in file: `presets.yaml`,
   `bandwidth.yaml`, `models.yaml`, `providers.yaml` or `publishers.yaml`. Each top-level key
   replaces **all** of the built-in entry with the same key. thetaserve does not merge the fields of an entry.

## Trust another weight publisher

```yaml
# publishers.yaml
cyankiwi: AWQ quantisations I checked myself
```

## Add a use-case preset

```yaml
# presets.yaml
agent-long: { min_context: 262144, tools: true, min_bits: 8 }
```

Use it with `up --use agent-long`.

## Use a different vLLM image

The built-in image is the one measured in the last pilot. To try a newer one:

```yaml
# models.yaml
vllm_image: vllm/vllm-openai:latest   # not measured; pin a tag for repeat launches
```

A user `models.yaml` replaces only the keys it contains, so the built-in models stay.

## Add a model

Give the base repository a short name, and the vLLM flags of its family:

```yaml
# models.yaml
qwen3.8-27b:
  hf_family: Qwen/Qwen3.8-27B
  common_args: [--enable-prefix-caching, --reasoning-parser, qwen3,
                --enable-auto-tool-choice, --tool-call-parser, qwen3_xml,
                --max-num-seqs, '32']
```

thetaserve reads the sizes and variants from Hugging Face. Do not add them.

## Check

```sh
thetaserve list models
thetaserve up --provider runpod --model qwen3.8-27b --dry-run
```

A YAML or validation error shows the file and the field, and the command stops with exit
status 2.
