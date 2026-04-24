"""Explicit COBA benchmark built on FixedPostNumConn and ``@`` operators."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import brainpy
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from brainevent._event.binary import BinaryArray
from brainevent._event.bitpack_binary import BitPackedBinary
from brainevent._event.compact_binary import CompactBinary
from brainevent._fcn.main import FixedPostNumConn

conn_num_base = 80

_FCN_DATA_TYPES = ('binary', 'float', 'bitpack', 'bitpack_a0', 'bitpack_a1', 'compact')
_FCN_LAYOUTS = ('col_scatter', 'row_gather', 'auto')
_FCN_TARGETS = ('pre', 'post')


def _validate_args(
    *,
    data_type: str,
    efferent_target: str,
    mv_layout: str,
):
    if data_type not in _FCN_DATA_TYPES:
        raise ValueError(
            'data_type must be one of '
            '"binary", "float", "bitpack", "bitpack_a0", "bitpack_a1", "compact".'
        )
    if efferent_target not in _FCN_TARGETS:
        raise ValueError('The target of the connection must be either "pre" or "post".')
    if mv_layout not in _FCN_LAYOUTS:
        raise ValueError("`mv_layout` must be one of 'col_scatter', 'row_gather', or 'auto'.")


def _resolve_conn_num(
    conn_num: Union[int, float],
    source_size: int,
    target_size: int,
    *,
    efferent_target: str,
) -> int:
    if isinstance(conn_num, float):
        assert 0.0 <= conn_num <= 1.0, 'Connection probability must be in [0, 1].'
        conn_num = int(target_size * conn_num) if efferent_target == 'post' else int(source_size * conn_num)
    assert isinstance(conn_num, int), 'Connection number must be an integer.'
    return conn_num


def _make_post_conn(
    source_size: int,
    target_size: int,
    conn_num: Union[int, float],
    *,
    efferent_target: str,
    data_type: str,
    homo: bool,
    conn_weight_base: u.Quantity,
    mv_layout: str,
) -> FixedPostNumConn:
    conn_num = _resolve_conn_num(conn_num, source_size, target_size, efferent_target=efferent_target)

    if efferent_target == 'post':
        shape = (source_size, target_size)
        n_rows = source_size
        n_cols = target_size
    else:
        shape = (target_size, source_size)
        n_rows = target_size
        n_cols = source_size

    # Dual row/col layout is only needed for the pre-synaptic column-scatter
    # path. Post-synaptic updates use transpose=True scatter and do not consume
    # the maintained CSC mirror.
    maintain_dual_layout = (
        efferent_target == 'pre'
        and data_type in ('binary', 'compact')
        and mv_layout == 'col_scatter'
    )

    with jax.ensure_compile_time_eval():
        indices_np = np.random.randint(0, n_cols, size=(n_rows, conn_num)).astype(np.int32, copy=False)
        conn_weight = conn_weight_base * conn_num_base / conn_num
        if homo:
            weight = u.math.asarray(conn_weight, dtype=brainstate.environ.dftype())
        else:
            weight = u.math.asarray(
                u.math.full((n_rows, conn_num), conn_weight),
                dtype=brainstate.environ.dftype(),
            )

    return FixedPostNumConn(
        (weight, u.math.asarray(indices_np, dtype=np.int32)),
        shape=shape,
        maintain_dual_layout=maintain_dual_layout,
    )


def _prepare_operand(spikes, *, data_type: str):
    spikes = u.math.asarray(spikes, dtype=jnp.bool_)
    if data_type == 'binary':
        return BinaryArray(spikes)
    if data_type == 'compact':
        # 1D compact matmul through ``@`` needs active_ids/n_active on both
        # left- and right-matmul paths, so the full constructor is required.
        return CompactBinary.from_array(spikes)
    if data_type in ('bitpack', 'bitpack_a0', 'bitpack_a1'):
        return BitPackedBinary(spikes)
    if data_type == 'float':
        return u.math.asarray(spikes, dtype=brainstate.environ.dftype())
    raise ValueError(f'Unsupported data_type: {data_type}')


def _apply_conn(spikes, conn: FixedPostNumConn, *, data_type: str, efferent_target: str):
    operand = _prepare_operand(spikes, data_type=data_type)
    if efferent_target == 'post':
        return operand @ conn
    return conn @ operand


class EINet(brainstate.nn.Module):
    def __init__(
        self,
        scale: float,
        data_type: str,
        efferent_target: str,
        conn_num: Union[int, float] = 80,
        homo: bool = True,
        mv_layout: str = 'row_gather',
    ):
        super().__init__()
        _validate_args(data_type=data_type, efferent_target=efferent_target, mv_layout=mv_layout)

        self.data_type = data_type
        self.efferent_target = efferent_target
        self.mv_layout = mv_layout

        self.n_exc = int(3200 * scale)
        self.n_inh = int(800 * scale)
        self.num = self.n_exc + self.n_inh

        self.N = brainpy.state.LIFRef(
            self.num,
            V_rest=-60. * u.mV,
            V_th=-50. * u.mV,
            V_reset=-60. * u.mV,
            tau=20. * u.ms,
            tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55., 2., unit=u.mV),
        )

        self.exc_conn = _make_post_conn(
            self.n_exc,
            self.num,
            conn_num,
            efferent_target=efferent_target,
            data_type=data_type,
            homo=homo,
            conn_weight_base=0.6 * u.mS,
            mv_layout=mv_layout,
        )
        self.inh_conn = _make_post_conn(
            self.n_inh,
            self.num,
            conn_num,
            efferent_target=efferent_target,
            data_type=data_type,
            homo=homo,
            conn_weight_base=6.7 * u.mS,
            mv_layout=mv_layout,
        )

        self.exc_syn = brainpy.state.Expon(self.num, tau=5. * u.ms)
        self.inh_syn = brainpy.state.Expon(self.num, tau=10. * u.ms)

        self.exc_out = brainpy.state.COBA(E=0. * u.mV)
        self.inh_out = brainpy.state.COBA(E=-80. * u.mV)

        self.N.add_current_input('exc_coba', self.exc_out)
        self.N.add_current_input('inh_coba', self.inh_out)

    def init_state(self, *args, **kwargs):
        self.rate = brainstate.ShortTermState(u.math.zeros(self.num))

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            exc_spk = spk[:self.n_exc]
            inh_spk = spk[self.n_exc:]

            delta_g_exc = _apply_conn(
                exc_spk,
                self.exc_conn,
                data_type=self.data_type,
                efferent_target=self.efferent_target,
            )
            delta_g_inh = _apply_conn(
                inh_spk,
                self.inh_conn,
                data_type=self.data_type,
                efferent_target=self.efferent_target,
            )

            g_exc = self.exc_syn(delta_g_exc)
            g_inh = self.inh_syn(delta_g_inh)

            self.exc_out.bind_cond(g_exc)
            self.inh_out.bind_cond(g_inh)

            self.N(inp)
            self.rate.value += self.N.get_spike()
            return self.N.get_spike()


def make_simulation_run(
    scale: float,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: Union[int, float] = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    # Construct the network outside JIT so FixedPostNumConn can validate
    # indices and materialize any dual-layout buffers from concrete arrays.
    net = EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )

    @brainstate.transform.jit
    def run():
        net.init_all_states()

        def fn(t):
            return net.update(t, 20 * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second)

    return run


def make_training_run(
    scale: float,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: Union[int, float] = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    def run():
        raise NotImplementedError(
            'make_training_run() is not implemented for the explicit FixedPostNumConn benchmark.'
        )

    return run


def make_simulation_batch_run(
    scale: float,
    batch_size: int = 16,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: Union[int, float] = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    net = EINet(
        scale,
        data_type=data_type,
        efferent_target=efferent_target,
        conn_num=conn_num,
        homo=homo,
        mv_layout=mv_layout,
    )
    mapper = brainstate.nn.Map(net, init_map_size=batch_size)

    @brainstate.transform.jit
    def run():
        mapper.init_all_states()

        def fn(t):
            ts = jax.numpy.ones(batch_size) * t
            return mapper.map('update', in_axes=(0, None))(ts, 20. * u.mA)

        with brainstate.environ.context(dt=0.1 * u.ms):
            times = u.math.arange(0. * u.ms, duration, brainstate.environ.get_dt())
            brainstate.transform.for_loop(fn, times)

        return net.num, net.rate.value.sum() / net.num / duration.to_decimal(u.second) / batch_size

    return run


def make_training_batch_run(
    scale: float,
    batch_size: int = 16,
    data_type: str = 'binary',
    efferent_target: str = 'post',
    duration: u.Quantity = 1e4 * u.ms,
    conn_num: Union[int, float] = 80,
    homo: bool = True,
    mv_layout: str = 'row_gather',
):
    def run():
        raise NotImplementedError(
            'make_training_batch_run() is not implemented for the explicit FixedPostNumConn benchmark.'
        )

    return run
