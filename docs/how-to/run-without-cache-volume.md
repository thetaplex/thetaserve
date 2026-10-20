# Run without a cache volume

By default the first `up` creates a 100 GB Runpod network volume, `thetaserve-cache`, that holds
downloaded weights and is billed monthly until deleted. To launch without it:

1. In your user config directory, create `providers.yaml`:

   ```yaml
   runpod:
     api_key_env: RUNPOD_API_KEY
     cloud_type: SECURE
     network_volume: null
     container_disk_gb: 50
   ```

   The entry replaces the built-in `runpod` entry whole, so keep the other fields. Size
   `container_disk_gb` to hold the weights plus about 20 GB.

2. Check the request no longer has `mounts` or `dataCenterIds`:

   ```sh
   uv run thetaserve up --provider runpod --device a40-48g --model qwen3.8-27b --dry-run
   ```

3. Launch as usual. Runpod places the pod in any data centre with the GPU in stock.

End each run with `thetaserve down`, not `stop`: the container disk is wiped on stop, so
`start` downloads the weights again while you pay for the GPU.

To return to the cache, delete `providers.yaml` or set `network_volume: thetaserve-cache`.
