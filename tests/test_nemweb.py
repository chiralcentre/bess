"""Tests for the NEMWEB client's retry behaviour.

Retrying is delegated to the ``backoff`` library, so what is worth testing is
the policy rather than the mechanism: which failures are retried, which are
given up on immediately, and what the caller sees when the attempts run out.

The retry schedule is a module constant, so the waits are skipped by patching
``time.sleep`` -- which is what ``backoff`` calls -- keeping the suite fast
without making production code configurable purely for the tests.
"""

from __future__ import annotations

import time

import pytest
import requests

from bess.marketdata.nemweb import MAX_TRIES, NemwebClient, NemwebError


class FakeResponse:
    def __init__(self, status: int, content: bytes = b"") -> None:
        self.status_code = status
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error


class RecordingSession:
    """Stands in for requests.Session, returning a scripted sequence."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float | None = None):
        self.calls.append(url)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Skip the real backoff waits; backoff calls ``time.sleep``."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def _client(tmp_path, session) -> NemwebClient:
    client = NemwebClient(cache_dir=tmp_path)
    client.session = session
    return client


class TestRetryPolicy:
    def test_succeeds_without_retrying(self, tmp_path) -> None:
        session = RecordingSession(FakeResponse(200, b"payload"))
        client = _client(tmp_path, session)

        assert client._get("https://example.test/a") == b"payload"
        assert len(session.calls) == 1

    def test_retries_a_connection_failure_then_succeeds(self, tmp_path) -> None:
        session = RecordingSession(
            requests.ConnectionError("dropped"),
            FakeResponse(200, b"payload"),
        )
        client = _client(tmp_path, session)

        assert client._get("https://example.test/a") == b"payload"
        assert len(session.calls) == 2

    def test_retries_a_server_error(self, tmp_path) -> None:
        """A 503 is usually transient, so it is worth another attempt."""
        session = RecordingSession(FakeResponse(503), FakeResponse(200, b"ok"))
        client = _client(tmp_path, session)

        assert client._get("https://example.test/a") == b"ok"
        assert len(session.calls) == 2

    def test_gives_up_immediately_on_a_404(self, tmp_path) -> None:
        """A day the archive does not hold will not appear by waiting.

        The hand-rolled loop this replaced burned every attempt on a 404.
        """
        session = RecordingSession(FakeResponse(404))
        client = _client(tmp_path, session)

        with pytest.raises(NemwebError, match="could not retrieve"):
            client._get("https://example.test/missing")
        assert len(session.calls) == 1

    def test_stops_after_max_tries(self, tmp_path) -> None:
        session = RecordingSession(requests.ConnectionError("dropped"))
        client = _client(tmp_path, session)

        with pytest.raises(NemwebError):
            client._get("https://example.test/a")
        assert len(session.calls) == MAX_TRIES

    def test_failure_is_reported_as_a_nemweb_error(self, tmp_path) -> None:
        """Callers see the project's own error type, with the cause attached."""
        session = RecordingSession(requests.ConnectionError("dropped"))
        client = _client(tmp_path, session)

        with pytest.raises(NemwebError) as caught:
            client._get("https://example.test/a")

        assert "example.test" in str(caught.value)
        assert isinstance(caught.value.__cause__, requests.RequestException)
