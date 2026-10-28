"""Health answer for the Runpod serverless load balancer while vLLM loads (DESIGN.md section 8.3).

The load balancer polls the worker's PORT_HEALTH: 200 is healthy, 204 is initializing, anything
else is unhealthy, and an unhealthy worker is removed and restarted. vLLM opens its port only
after the weights are loaded, minutes after the worker starts. This server answers 204 until
vLLM's /health answers 200, then 200.

The launch script writes this file to the host and starts it next to vLLM, so it must use only
the standard library. Usage: python3 healthcheck.py VLLM_PORT HEALTH_PORT
"""

import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def status(vllm_health_url: str) -> int:
    """200 when vLLM answers its /health, else 204 (still initializing)."""
    try:
        with urllib.request.urlopen(vllm_health_url, timeout=5) as resp:  # noqa: S310 - fixed http URL
            return 200 if resp.status == 200 else 204
    except OSError:  # not listening yet, or HTTPError
        return 204


def main() -> None:
    vllm_port, health_port = (int(arg) for arg in sys.argv[1:3])
    url = f"http://127.0.0.1:{vllm_port}/health"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status(url))
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass  # polled every few seconds; keep the worker log for vLLM

    ThreadingHTTPServer(("0.0.0.0", health_port), Handler).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
