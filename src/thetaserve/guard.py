"""vLLM middleware that requires the API key on every path except /health (DESIGN.md section 9).

vLLM's `--api-key` guards only /v1 and a few other prefixes; /metrics, /load and others stay
open. The launch script writes this file to the host and passes it to `vllm serve
--middleware`, so it must use only the standard library.
"""

import hmac
import os
from typing import Any


class RequireKey:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.expected = f"Bearer {os.environ['VLLM_API_KEY']}".encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] == "OPTIONS"
            or scope["path"].removeprefix(scope.get("root_path", "")) == "/health"
            or hmac.compare_digest(dict(scope["headers"]).get(b"authorization", b""), self.expected)
        ):
            await self.app(scope, receive, send)
            return
        headers = [(b"content-type", b"application/json")]
        await send({"type": "http.response.start", "status": 401, "headers": headers})
        await send({"type": "http.response.body", "body": b'{"error": "Unauthorized"}'})
