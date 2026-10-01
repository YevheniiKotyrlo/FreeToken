"""--bind-when-ready: the API port opens only once the backend serves, so a listening port means a loaded model."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI

import freetoken.server.api_server as api
import freetoken.server.supervisor as supervisor
from freetoken.distributed import DistributedInfo
from freetoken.server.args import ServerArgs


def _config(bind_when_ready: bool) -> ServerArgs:
    return ServerArgs(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
        use_dummy_weight=True,
        bind_when_ready=bind_when_ready,
    )


# Converts a supervisor that never finishes into a failure; the fakes below finish in milliseconds.
HANG_GUARD_SECONDS = 10.0
BOUND = threading.Event()
SUPERVISED = threading.Event()


@pytest.fixture
def events(monkeypatch):
    recorded: list[str] = []
    BOUND.clear()
    SUPERVISED.clear()

    def bind(*args, **kwargs) -> None:
        recorded.append("bind")
        BOUND.set()

    monkeypatch.setattr(api, "_GLOBAL_STATE", None)
    # run_api_server installs its middleware on the module's app, which another test's client
    # may already have started; each run here gets an app nothing has started.
    monkeypatch.setattr(api, "app", FastAPI())
    monkeypatch.setattr(api, "_exit_after_backend_death", lambda grace_s: recorded.append("exit"))
    monkeypatch.setattr(api.uvicorn, "run", bind)
    yield recorded
    # the supervisor thread reports into module state, which monkeypatch restores right after
    assert SUPERVISED.wait(HANG_GUARD_SECONDS), "the backend supervisor outlived its test"


def _supervisor_that(outcome: str, events: list[str], release: threading.Event):
    def fake_supervisor(handle, progress, on_ready, *, on_failure, on_meta, is_shutting_down):
        try:
            events.append("loading")
            release.wait(HANG_GUARD_SECONDS)
            # A server that binds early does so now, while this backend still reports loading; one that
            # holds the port never sets BOUND, and the window only bounds how long that absence is watched.
            BOUND.wait(2.0)
            if outcome == "ready":
                events.append("ready")
                on_ready()
            else:
                events.append("failed")
                on_failure("weights did not fit")
        finally:
            SUPERVISED.set()

    return fake_supervisor


def _no_workers():
    return SimpleNamespace(processes=[])


def test_the_port_opens_only_after_the_backend_is_serving(monkeypatch, events):
    release = threading.Event()
    release.set()
    monkeypatch.setattr(supervisor, "run_backend_supervisor", _supervisor_that("ready", events, release))
    api.run_api_server(_config(bind_when_ready=True), _no_workers, run_shell=False)
    assert events == ["loading", "ready", "bind"]


def test_a_failed_load_exits_without_ever_binding(monkeypatch, events):
    release = threading.Event()
    release.set()
    monkeypatch.setattr(supervisor, "run_backend_supervisor", _supervisor_that("failed", events, release))
    with pytest.raises(SystemExit, match="weights did not fit"):
        api.run_api_server(_config(bind_when_ready=True), _no_workers, run_shell=False)
    assert "bind" not in events


def test_without_the_flag_the_port_opens_while_loading(monkeypatch, events):
    release = threading.Event()
    monkeypatch.setattr(supervisor, "run_backend_supervisor", _supervisor_that("ready", events, release))
    api.run_api_server(_config(bind_when_ready=False), _no_workers, run_shell=False)
    bound_while = list(events)
    release.set()
    # the supervisor thread reports "loading" whenever the scheduler runs it; what matters is
    # that the port was open before the backend was ready
    assert "bind" in bound_while and "ready" not in bound_while
