# Connect a client or coding agent

Requires a deployment that `up` reported as `Ready.`

## Set the environment

POSIX shells:

```sh
eval "$(uv run thetaserve env)"
```

PowerShell:

```powershell
uv run thetaserve env | Invoke-Expression
```

fish: `uv run thetaserve env --shell fish | source`. With several deployments, name one:
`thetaserve env dev`.

This sets `OPENAI_BASE_URL` (`https://<pod-id>-<port>.proxy.runpod.net/v1`) and
`OPENAI_API_KEY` (the per-launch key). Start your client from the same shell.

## Check the endpoint

```sh
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

Use the model `id` from the response (for example `Qwen/Qwen3.8-27B-FP8`) as the model name in
your client.

## Stream long generations

Runpod's proxy cuts a request off after 100 s with HTTP 524. Turn streaming on in the client for
anything that can run longer; coding agents usually stream already.

## If it fails

- `401`: the key in your shell is from an older launch. Run `thetaserve env` again.
- `404` or connection errors: check `thetaserve status`, then `thetaserve logs -f`.
