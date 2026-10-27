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

2. Do a dry run:

   ```sh
   thetaserve up --provider runpod --model qwen3.8-27b --dry-run
   ```

   Make sure that the request has no `mounts` and no `dataCenterIds`.

3. Launch as usual. Runpod puts the pod in a data centre that has the GPU in stock.

End each run with `thetaserve down`. Without a cache volume, `stop` and `start` do not make the
next launch faster: Runpod erases the container disk when the pod stops, so `start` downloads
the weights again, the same as a new `up`. Either way, you pay the GPU rate during the download.

`down` and `up` are also more reliable. A stopped pod without a cache volume bills nothing, but it
keeps its host machine, and `start` fails when a different user has that GPU. `up` selects the cheapest GPU in
stock in all data centres. `down` leaves nothing on your Runpod account that bills.

After each `up`, the endpoint URL and the API key are new. Run `eval "$(thetaserve env)"` again.

To use the cache again, delete `providers.yaml` or set `network_volume: thetaserve-cache`.
