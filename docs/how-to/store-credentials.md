# Store credentials

## Create a Runpod API key

1. In the Runpod console, open **Settings → API Keys → Create API Key** and name it `thetaserve`.
2. Set the permissions:

   | Setting | Value |
   | --- | --- |
   | `api.runpod.io/graphql` | Read/Write |
   | `api.runpod.ai` | None |

   Read Only does not work: `up` creates pods. Runpod warns that GraphQL Read/Write gives full
   account access; no narrower setting can create pods (checked 2026-10-20). Use **All** if the
   console offers no GraphQL line.
3. Set a spend limit on the account. It caps what a leaked key can cost.

## Store it

In a real terminal (not an agent shell or `!` prompt, which cannot hide input):

```sh
uv run thetaserve setup
```

Paste the key; nothing is shown as you type. Optionally paste a Hugging Face token at the second
prompt, or press Enter to skip. Run `setup` again and press Enter to keep a stored value.

`setup` reports where each secret went: the system keyring, or, where no keyring backend exists
(headless Linux, containers), a mode-600 `secrets.json` in the user config directory.

## Use environment variables instead

For CI or a one-off shell, skip `setup`:

```sh
export RUNPOD_API_KEY=...
export HF_TOKEN=...        # optional
```

Environment variables take precedence over stored secrets.

## Revoke

Delete the key in the Runpod console. thetaserve keeps no copy outside the keyring or the
secrets file.
