# Run without a cache volume

By default, the first `up` creates a 100 GB Runpod network volume, `thetaserve-cache`. The
volume holds the downloaded weights. Runpod charges for it each month until you delete it. To
launch without it:

1. In your user config directory, create `providers.yaml`:

   ```yaml
   runpod:
     api_key_env: RUNPOD_API_KEY
     cloud_type: SECURE
     network_volume: null
     container_disk_gb: 50
   ```

   This entry replaces all of the built-in `runpod` entry, so keep the other fields. Set
   `container_disk_gb` to the weight size plus approximately 20 GB.

2. Make sure that the request has no `mounts` and no `dataCenterIds`:

   ```sh
   thetaserve up --provider runpod --model qwen3.8-27b --dry-run
   ```

3. Launch as usual. Runpod puts the pod in a data centre that has the GPU in stock.

End each run with `thetaserve down`, not `stop`. `stop` erases the container disk, so `start`
downloads the weights again while you pay for the GPU.

To use the cache again, delete `providers.yaml` or set `network_volume: thetaserve-cache`.
