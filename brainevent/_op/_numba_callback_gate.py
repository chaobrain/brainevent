# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Native admission control for CPU callbacks during interpreter shutdown."""

import ctypes
import threading
import time
from typing import Any, Optional


class _CpuCallbackGate:
    """Keep late XLA workers outside Python after the exit guard has run.

    The owner must outlive interpreter finalization: its engine owns executable
    code and its state, error arguments, and callbacks are embedded as raw
    addresses in that code. The caller is responsible for retaining the owner.
    """

    def __init__(
        self,
        *,
        frame_api_offset: int,
        error_create_offset: int,
        error_args: ctypes.Structure,
        state: Optional[ctypes.c_uint64] = None,
    ) -> None:
        # Numba depends on llvmlite, but importing this module alone must not
        # initialize LLVM or require the optional CPU backend dependencies.
        from llvmlite import binding, ir

        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        self._binding = binding
        self._ir = ir
        self._state = state if state is not None else ctypes.c_uint64(0)
        self._error_args = error_args
        self._frame_api_offset = frame_api_offset
        self._error_create_offset = error_create_offset
        self.callbacks: list[Any] = []
        self._wrap_lock = threading.Lock()
        self._pointer_integer = ir.IntType(ctypes.sizeof(ctypes.c_void_p) * 8)
        self._byte_pointer = ir.IntType(8).as_pointer()
        self._state_pointer = self._pointer_constant(
            ctypes.addressof(self._state), ir.IntType(64).as_pointer()
        )
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        self._data_layout = str(target_machine.target_data)

        module = self._new_module('control')
        close_name = f'brainevent_cpu_gate_close_{id(self)}'
        close = ir.Function(
            module, ir.FunctionType(ir.VoidType(), []), name=close_name
        )
        builder = ir.IRBuilder(close.append_basic_block('entry'))
        builder.atomic_rmw(
            'or', self._state_pointer,
            ir.Constant(ir.IntType(64), 1), 'seq_cst'
        )
        builder.ret_void()

        read_name = f'brainevent_cpu_gate_read_{id(self)}'
        read = ir.Function(
            module, ir.FunctionType(ir.IntType(64), []), name=read_name
        )
        builder = ir.IRBuilder(read.append_basic_block('entry'))
        builder.ret(builder.load_atomic(self._state_pointer, 'seq_cst', 8))

        native_module = binding.parse_assembly(str(module))
        native_module.verify()
        self._engine = binding.create_mcjit_compiler(
            native_module, target_machine
        )
        self._engine.finalize_object()
        self._shutdown = ctypes.CFUNCTYPE(None)(
            self._function_address(close_name)
        )
        self._read_state = ctypes.CFUNCTYPE(ctypes.c_uint64)(
            self._function_address(read_name)
        )

    def _new_module(self, suffix: str) -> Any:
        name = f'brainevent_cpu_gate_{id(self)}_{suffix}'
        module = self._ir.Module(name=name)
        module.triple = self._binding.get_default_triple()
        module.data_layout = self._data_layout
        return module

    def _pointer_constant(self, address: int, pointer_type: Any) -> Any:
        integer = self._ir.Constant(self._pointer_integer, address)
        return integer.inttoptr(pointer_type)

    def _function_address(self, name: str) -> int:
        address = self._engine.get_function_address(name)
        if not address:
            raise RuntimeError(
                f'Could not compile CPU callback gate function {name!r}.'
            )
        return address

    def wrap(self, callback: Any) -> int:
        """Return a native FFI target that admits calls through this gate."""
        callback_address = ctypes.cast(callback, ctypes.c_void_p).value
        if callback_address is None:
            raise ValueError('CPU callback gate cannot wrap a null callback.')
        with self._wrap_lock:
            ir = self._ir
            pointer = self._byte_pointer
            i64 = ir.IntType(64)
            zero_pointer = ir.Constant(pointer, None)
            module = self._new_module(str(len(self.callbacks)))
            function_type = ir.FunctionType(pointer, [pointer])
            name = f'brainevent_gate_callback_{id(self)}_{len(self.callbacks)}'
            function = ir.Function(module, function_type, name=name)
            frame = function.args[0]
            entry = function.append_basic_block('entry')
            admitted = function.append_basic_block('admitted')
            closed = function.append_basic_block('closed')
            read_api = function.append_basic_block('read_api')
            read_error_create = function.append_basic_block('read_error_fn')
            create_error = function.append_basic_block('create_error')
            closed_null = function.append_basic_block('closed_null')

            # A separate enabled flag and counter would have a check/increment
            # race. One atomic word linearizes admission against closing: bit
            # zero means closed, and each entrant contributes two until return.
            builder = ir.IRBuilder(entry)
            previous = builder.atomic_rmw(
                'add', self._state_pointer, ir.Constant(i64, 2), 'seq_cst'
            )
            was_closed = builder.icmp_unsigned(
                '!=', builder.and_(previous, ir.Constant(i64, 1)),
                ir.Constant(i64, 0)
            )
            builder.cbranch(was_closed, closed, admitted)

            builder.position_at_end(admitted)
            original = builder.inttoptr(
                ir.Constant(self._pointer_integer, callback_address),
                function_type.as_pointer()
            )
            result = builder.call(original, [frame])
            builder.atomic_rmw(
                'sub', self._state_pointer, ir.Constant(i64, 2), 'seq_cst'
            )
            builder.ret(result)

            # This path must be entirely native: even constructing arguments
            # for a ctypes callback would first enter PyGILState_Ensure(). Use
            # XLA's own C error API so a later atexit caller cannot receive an
            # successful result whose output buffers were not filled.
            builder.position_at_end(closed)
            builder.cbranch(
                builder.icmp_unsigned('==', frame, zero_pointer),
                closed_null, read_api
            )

            builder.position_at_end(read_api)
            api_slot = builder.gep(
                frame,
                [ir.Constant(self._pointer_integer, self._frame_api_offset)],
            )
            api = builder.load(builder.bitcast(api_slot, pointer.as_pointer()))
            builder.cbranch(
                builder.icmp_unsigned('==', api, zero_pointer),
                closed_null, read_error_create
            )

            builder.position_at_end(read_error_create)
            error_slot = builder.gep(
                api,
                [ir.Constant(
                    self._pointer_integer, self._error_create_offset
                )],
            )
            error_create = builder.load(
                builder.bitcast(error_slot, pointer.as_pointer())
            )
            builder.cbranch(
                builder.icmp_unsigned('==', error_create, zero_pointer),
                closed_null, create_error
            )

            builder.position_at_end(create_error)
            error_args = self._pointer_constant(
                ctypes.addressof(self._error_args), pointer
            )
            error_function = builder.bitcast(
                error_create, function_type.as_pointer()
            )
            error = builder.call(error_function, [error_args])
            builder.atomic_rmw(
                'sub', self._state_pointer, ir.Constant(i64, 2), 'seq_cst'
            )
            builder.ret(error)

            builder.position_at_end(closed_null)
            builder.atomic_rmw(
                'sub', self._state_pointer, ir.Constant(i64, 2), 'seq_cst'
            )
            builder.ret(zero_pointer)

            native_module = self._binding.parse_assembly(str(module))
            native_module.verify()
            self._engine.add_module(native_module)
            self._engine.finalize_object()
            address = self._function_address(name)
            self.callbacks.append(callback)
            return address

    def close(self) -> None:
        """Reject Python entries and wait for admitted calls to return."""
        self._shutdown()
        # These reads are atomic too: callbacks execute without the GIL. Sleep
        # releases the GIL so callbacks already admitted can finish their work.
        while self._read_state() != 1:
            time.sleep(0.001)
