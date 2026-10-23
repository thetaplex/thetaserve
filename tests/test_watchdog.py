import json
import urllib.request
from typing import Any

import pytest

from thetaserve import watchdog
from thetaserve.watchdog import activity, reason_to_stop

METRICS = """\
# HELP vllm:request_success_total Count of successfully processed requests.
vllm:request_success_total{finished_reason="stop",model_name="m"} 3.0
vllm:request_success_total{finished_reason="length",model_name="m"} 1.0
vllm:num_requests_running{model_name="m"} 2.0
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:request_success_total_created{finished_reason="stop",model_name="m"} 1.7e9
"""


def test_activity_sums_completed_and_busy_requests() -> None:
    assert activity(METRICS) == (4.0, 3.0)
    assert activity("") == (0.0, 0.0)


@pytest.mark.parametrize(
    ("now", "last_active", "expected"),
    [
        (100, None, None),  # still loading: the idle clock has not started
        (100, 50, None),
        (100, 40, "idle for 60 s"),
        (1000, 999, "maximum lifetime of 1000 s reached"),
        (1000, None, "maximum lifetime of 1000 s reached"),  # loading never ends
    ],
)
def test_reason_to_stop(now: float, last_active: float | None, expected: str | None) -> None:
    assert reason_to_stop(now, 0, last_active, idle_s=60, max_life_s=1000) == expected


def test_stop_asks_runpod_to_stop_this_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    class Resp:
        status = 200

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def urlopen(request: Any, timeout: float) -> Resp:
        sent.append(request)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod1")
    monkeypatch.setenv("RUNPOD_API_KEY", "pod-key")
    watchdog.stop("idle for 60 s")
    assert sent[0].full_url == "https://api.runpod.io/v2/pods/pod1/action"
    assert json.loads(sent[0].data) == {"action": "stop"}
    assert sent[0].get_header("Authorization") == "Bearer pod-key"
