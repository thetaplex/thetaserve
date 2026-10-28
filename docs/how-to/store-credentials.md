# Store credentials

## Create a Runpod API key

1. In the Runpod console, open **Settings → API Keys → Create API Key** and name it `thetaserve`.
2. Set the permissions:

   | Setting | Value |
   | --- | --- |
   | `api.runpod.io/graphql` | Read/Write |
   | `api.runpod.ai` | None |

   Read Only is not sufficient, because `up` creates pods. Runpod warns that GraphQL Read/Write
   gives full account access. No narrower setting can create pods (checked 2026-10-20). If the
   console shows no GraphQL line, select **All**.
3. Set a spend limit on the account. The limit sets the maximum cost of a leaked key.

## Store it

Use a real terminal. An agent shell or a `!` prompt cannot hide input.

```sh
thetaserve setup
```

Paste the key. The terminal does not show what you type. At the second prompt, paste a Hugging
Face token, or press Enter to skip it. To keep a stored value when you run `setup` again, press
Enter.

`setup` shows where it stored each secret. This is the system keyring or, when no keyring
backend is available (headless Linux, containers), a mode-600 `secrets.json` in the user config
directory.

## Use environment variables instead

For CI or a single shell, do not use `setup`. Set the variables:

```sh
export RUNPOD_API_KEY=...
export HF_TOKEN=...        # optional
```

Environment variables have priority over stored secrets. For `runpod-serverless`, the
client key is `RUNPOD_CLIENT_KEY` ([Launch on Runpod serverless](launch-on-serverless.md)).

## Revoke

Delete the key in the Runpod console. thetaserve keeps no copy outside the keyring or the
secrets file.
