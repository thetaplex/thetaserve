# Launch on Runpod serverless

A serverless endpoint costs nothing while no request comes. A request starts a worker; the
worker stops after `--idle` without requests. Use it for occasional use. For long sessions, a
pod (`--provider runpod`) has no cold starts.

This provider has not yet been run on real hardware (`docs/DESIGN.md` section 8.3).

1. In the Runpod console, under **Settings → API Keys**, create a key with **Restricted**
   permission for your clients. After step 3, give it access to the new endpoint only.

2. Look at the options and the request, without creating anything:

   ```sh
   thetaserve up --provider runpod-serverless --model qwen3.8-27b --idle 15m --dry-run
   ```

   The rates are per GPU-hour while a worker runs. The idle timeout is at most `1h`.

3. Launch:

   ```sh
   thetaserve up --provider runpod-serverless --model qwen3.8-27b --idle 15m
   ```

   `up` waits until the first worker has loaded the model. The first start downloads the
   weights.

4. Point your client at the endpoint with the restricted key, not the account key:

   ```sh
   eval "$(thetaserve env)"
   export OPENAI_API_KEY='<restricted key from step 1>'
   ```

5. After an idle gap, the first request waits for a cold start of some minutes. Runpod ends a
   request that waits more than 2 minutes for a worker (`no workers available`), and a request
   that runs more than 5.5 minutes. Retry the first request, and stream long generations.

`thetaserve stop` blocks new workers; `start` allows them again. `thetaserve down` deletes the
endpoint.
