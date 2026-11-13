# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_op/_numba_callback_gate.py``."""

import ctypes
import time

import pytest

pytest.importorskip('llvmlite')

from brainevent._op._numba_callback_gate import _FfiCallbackGate
from brainevent._op.numba_ffi import XLA_FFI_Api, XLA_FFI_CallFrame, XLA_FFI_Error_Create_Args


def _gate(entrants=0):
    """A gate whose shared state starts with ``entrants`` admitted calls."""
    state = ctypes.c_uint64(2 * entrants)
    gate = _FfiCallbackGate(
        frame_api_offset=XLA_FFI_CallFrame.api.offset,
        error_create_offset=XLA_FFI_Api.XLA_FFI_Error_Create.offset,
        error_args=XLA_FFI_Error_Create_Args(
            struct_size=ctypes.sizeof(XLA_FFI_Error_Create_Args),
            extension_start=None, message=b'test gate closed', errc=1,
        ),
        state=state,
    )
    return gate, state


def test_close_without_entrants_returns_at_once():
    gate, state = _gate()
    assert gate.in_flight() == 0
    assert gate.close(timeout=0) is True
    assert state.value == 1  # closed, nobody inside


def test_close_gives_up_after_the_timeout_while_a_call_is_admitted():
    # An admitted callback that never returns must not hang interpreter exit.
    gate, state = _gate(entrants=1)
    start = time.monotonic()
    assert gate.close(timeout=0.05) is False
    assert 0.05 <= time.monotonic() - start < 5
    assert state.value & 1 and gate.in_flight() == 1  # stays closed
    state.value -= 2  # the admitted call returns
    assert gate.close(timeout=1) is True


def test_close_waits_without_a_timeout_until_admitted_calls_return():
    import threading
    gate, state = _gate(entrants=1)
    timer = threading.Timer(0.05, lambda: setattr(state, 'value', state.value - 2))
    timer.start()
    try:
        assert gate.close() is True
    finally:
        timer.join()


def test_wrapped_entry_rejects_calls_after_close():
    calls = []
    callback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(lambda frame: calls.append(frame) or None)
    gate, state = _gate()
    entry = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(gate.wrap(callback))
    assert entry(None) is None and len(calls) == 1
    gate.close(timeout=0)
    assert entry(None) is None  # a null frame has no error API: plain rejection
    assert len(calls) == 1 and state.value == 1
