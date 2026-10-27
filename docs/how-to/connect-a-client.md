# Connect a client or coding agent

You need a deployment that `up` showed as `Ready.`

## Set the environment

POSIX shells:

```sh
eval "$(thetaserve env)"
```

PowerShell:

```powershell
thetaserve env | Invoke-Expression
```

fish: `thetaserve env --shell fish | source`. If you have more than one deployment, give
its name: `thetaserve env dev`.

This sets `OPENAI_BASE_URL` (`https://<pod-id>-<port>.proxy.runpod.net/v1`) and
`OPENAI_API_KEY` (the per-launch key). Start your client in the same shell.

## Check the endpoint

```sh
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

In your client, set the model name to the `id` in the response (for example
`Qwen/Qwen3.8-27B-FP8`). `thetaserve env` also shows it on stderr, as `model`.

## Stream long generations

Runpod's proxy stops a request after 100 s with HTTP 524. If a request can take longer, turn
on streaming in the client. Most coding agents use streaming already.

## If it fails

- `401`: the key in your shell is from an older launch. Run `thetaserve env` again.
- `404` or a connection error: run `thetaserve status`. Then run `thetaserve logs -f`.
