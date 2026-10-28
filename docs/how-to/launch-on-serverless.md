# Launch on Runpod serverless

A serverless endpoint costs nothing while no request comes. A request starts a worker; the
worker stops after `--idle` without requests. Use it for occasional use. For long sessions, a
pod (`--provider runpod`) has no cold starts.

This provider has not yet been run on real hardware (`docs/DESIGN.md` section 8.3).

1. Create a client key. Clients send it to Runpod's gateway, so it must not be your account
   key. In the Runpod console, open **Settings → API Keys → Create API Key**, name it
   `thetaserve-client`, select **Restricted**, and set:

   | Setting | Value |
   | --- | --- |
   | `api.runpod.io/graphql` | None |
   | Serverless endpoints (`api.runpod.ai`) | Read/Write |

   Store it. Use a real terminal, and press Enter at the first two prompts to keep the stored
   values:

   ```sh
   thetaserve setup
   ```

   Or set `RUNPOD_CLIENT_KEY`. `up` stops before it creates anything when there is no client
   key, or when the client key is the account key.

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

4. Point your client at the endpoint. `env` exports the client key:

   ```sh
   eval "$(thetaserve env)"
   ```

   The client key can call all serverless endpoints of your account, but it cannot create or
   delete anything outside them.

5. After an idle gap, the first request waits for a cold start of some minutes. Runpod ends a
   request that waits more than 2 minutes for a worker (`no workers available`), and a request
   that runs more than 5.5 minutes. Retry the first request, and stream long generations.

`thetaserve stop` blocks new workers; `start` allows them again. `thetaserve down` deletes the
endpoint.
