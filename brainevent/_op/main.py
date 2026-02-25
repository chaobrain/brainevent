# -*- coding: utf-8 -*-
# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import jax
from jax.interpreters import xla, batching, ad, mlir

from brainevent._compatible_import import Primitive
from brainevent._error import KernelFallbackExhaustedError
from brainevent._typing import KernelGenerator
from brainevent.config import get_backend
from .benchmark import BenchmarkRecord, BenchmarkResult, benchmark_function
from .util import (
    general_batching_rule, defjvp, OutType,
    abstract_arguments, check_pallas_jax_version,
    check_warp_installed
)

__all__ = [
    'XLACustomKernel',
    'KernelEntry',
]


@dataclass
class KernelEntry:
    """A registered kernel implementation for a specific backend and platform.

    ``KernelEntry`` is a lightweight data class used internally by
    :class:`XLACustomKernel` to store a single kernel implementation together
    with the backend and platform it targets.

    Parameters
    ----------
    backend : str
        The backend name (e.g., ``'numba'``, ``'warp'``, ``'pallas'``,
        ``'triton'``, ``'cuda_raw'``, ``'numba_cuda'``).
    platform : str
        The hardware platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
    kernel_generator : KernelGenerator
        A callable that accepts keyword arguments (forwarded from the
        primitive ``bind`` call) and returns a concrete kernel function
        ready to be invoked with the input arrays.

    See Also
    --------
    XLACustomKernel : The kernel manager that creates and stores
        ``KernelEntry`` instances.
    XLACustomKernel.def_kernel : Method used to register a new
        ``KernelEntry``.

    Notes
    -----
    ``KernelEntry`` instances are created internally by
    :meth:`XLACustomKernel.def_kernel` and stored in a nested
    dictionary keyed by ``(platform, backend)``.  Users typically do
    not need to instantiate this class directly.

    Examples
    --------
    .. code-block:: python

        >>> entry = KernelEntry(
        ...     backend='numba',
        ...     platform='cpu',
        ...     kernel_generator=my_kernel_generator,
        ... )
        >>> entry.backend
        'numba'
        >>> entry.platform
        'cpu'
    """
    backend: str
    platform: str
    kernel_generator: KernelGenerator


class XLACustomKernel:
    """Creates and manages a custom JAX primitive for XLA custom calls.

    This class provides a high-level interface to define custom operations
    that can be executed efficiently on different backends (CPU, GPU, TPU)
    via XLA custom calls. It handles the registration of the JAX primitive,
    its abstract evaluation rule, backend-specific kernel implementations,
    and JAX transformation rules like batching, JVP (forward-mode AD), and
    transpose (reverse-mode AD).

    Supported backends by platform:

    - **CPU**: Numba, CUDA
    - **GPU**: Pallas, CUDA, Numba CUDA, Warp, Triton
    - **TPU**: Pallas

    The workflow for using this class is:

    1. Create an instance with a unique primitive name
    2. Register kernel implementations using ``def_kernel`` or convenience
       methods like ``def_numba_kernel``, ``def_pallas_kernel``, etc.
    3. Optionally set default backends using ``set_default`` or ``asdefault=True``
    4. Define JAX transformation rules (batching, JVP, transpose) as needed
    5. Call the instance with input arrays and output specifications

    The first kernel registered for a platform automatically becomes the default.
    You can override this by calling ``set_default(platform, backend)`` or by
    passing ``asdefault=True`` when registering a kernel.

    If a kernel fails, the error message shows alternative backends available
    for the platform and how to switch to them.

    Instance attributes:

    - ``primitive``: The underlying JAX primitive created.
    - ``name``: The name assigned to the primitive.

    Parameters
    ----------
    name : str
        The unique name for the custom JAX primitive. This name is used
        to identify the primitive in JAX's internal registry and in
        error messages.

    See Also
    --------
    KernelEntry : Data class representing a single registered kernel.
    defjvp : Utility to define JVP rules for primitives with multiple
        results.
    general_batching_rule : Default batching rule applied to new
        ``XLACustomKernel`` instances.

    Examples
    --------
    .. code-block:: python

        >>> kernel = XLACustomKernel('my_custom_op')
        >>> kernel.def_numba_kernel(numba_kernel_generator)  # CPU default
        >>> kernel.def_pallas_kernel('gpu', pallas_kernel_generator, asdefault=True)
        >>> kernel.def_warp_kernel(warp_kernel_generator)  # Alternative GPU backend
        >>> print(kernel.defaults)  # {'cpu': 'numba', 'gpu': 'pallas'}
        >>> kernel.set_default('gpu', 'warp')  # Change GPU default
        >>> result = kernel(input_array, outs=[jax.ShapeDtypeStruct((10,), jnp.float32)])
    """

    __module__ = 'brainevent'

    def __init__(self, name: str, doc: str = None):
        # primitive
        self.name = name
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True
        if doc is not None:
            self.__doc__ = doc

        # abstract evaluation
        self.primitive.def_impl(functools.partial(xla.apply_primitive, self.primitive))
        self.primitive.def_abstract_eval(self._abstract_eval)

        # batching rule
        self.register_general_batching()

        # kernel storage: platform -> backend -> KernelEntry
        self._kernels: Dict[str, Dict[str, KernelEntry]] = {}
        # default backends per platform: platform -> backend_name
        self._defaults: Dict[str, str] = {}
        # tracks which platforms have had lowering registered
        self._registered_platforms: set = set()

        # call function for benchmarking
        self._call_fn: Optional[Callable] = None

        # categorization tags (e.g., {'csr', 'binary'})
        self._tags: set = set()
        # benchmark data generator function
        self._benchmark_data_fn: Optional[Callable] = None
        # Auto-register in global registry
        from brainevent._registry import register_primitive
        register_primitive(name, self)

    def _abstract_eval(self, *ins, outs: OutType, **kwargs):
        """Compute the abstract output types for the primitive.

        This method defines how JAX determines the shape and dtype of the
        primitive's outputs based on its inputs, without performing the
        actual computation.  In this implementation the output shapes and
        dtypes are explicitly provided via the ``outs`` parameter during
        the ``primitive.bind`` call and are returned directly.

        Parameters
        ----------
        *ins : jax.core.ShapedArray
            Abstract values corresponding to the input operands.  Not
            directly used because the output specification is pre-determined.
        outs : OutType
            A sequence of ``jax.core.ShapedArray`` objects specifying the
            expected shape and dtype of each output.
        **kwargs
            Additional keyword arguments forwarded during ``primitive.bind``.
            Not used in this abstract evaluation rule.

        Returns
        -------
        tuple of jax.core.ShapedArray
            The abstract output types, identical to the ``outs`` parameter.
        """
        return tuple(outs)

    def __call__(self, *ins, outs: OutType, **kwargs):
        """Invoke the primitive with the given inputs and output specification.

        Binds the input arrays and keyword arguments to the underlying JAX
        primitive.  The ``outs`` parameter describes the shapes and dtypes
        of the expected outputs so that JAX can allocate output buffers and
        perform abstract evaluation.

        Parameters
        ----------
        *ins : array_like
            Input arrays to the primitive.
        outs : OutType
            Output specification.  Can be a single object with ``shape``
            and ``dtype`` attributes (e.g., ``jax.ShapeDtypeStruct``), or
            a sequence / pytree of such objects for multiple outputs.
        **kwargs
            Additional keyword arguments forwarded to the kernel.  A
            ``backend`` keyword, if present, selects a specific backend
            implementation instead of the platform default.

        Returns
        -------
        result
            The output(s) of the primitive.  The pytree structure matches
            the structure of ``outs``.

        Raises
        ------
        AssertionError
            If the number of results returned by the primitive does not
            match the number of output specifications.

        See Also
        --------
        call : Invoke the high-level call function registered via
            :meth:`def_call`.

        Notes
        -----
        The ``outs`` specification is flattened into a list of
        ``jax.core.ShapedArray`` objects via :func:`abstract_arguments`
        and passed as the ``outs`` keyword to ``primitive.bind``.  The
        results are then unflattened to match the original pytree
        structure.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> kernel = XLACustomKernel('my_op')
            >>> # After registering kernels...
            >>> out = kernel(x, outs=[jax.ShapeDtypeStruct((10,), jnp.float32)])  # doctest: +SKIP
        """
        outs, tree_def = abstract_arguments(outs)
        r = self.primitive.bind(*ins, **kwargs, outs=tuple(outs))
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def def_kernel(
        self,
        backend: str,
        platform: str,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a kernel implementation for a specific backend and platform.

        Creates a :class:`KernelEntry` and stores it in the internal kernel
        registry.  If this is the first kernel registered for the given
        *platform*, it automatically becomes the default.  Pass
        ``asdefault=True`` to override an existing default.

        A JAX lowering rule is registered for the platform the first time
        any kernel targets it.

        Parameters
        ----------
        backend : str
            The backend name (e.g., ``'numba'``, ``'warp'``, ``'pallas'``,
            ``'triton'``, ``'cuda_raw'``, ``'numba_cuda'``).
        platform : str
            The hardware platform (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
        kg : KernelGenerator
            A callable that accepts keyword arguments (from the primitive
            ``bind`` call) and returns a concrete kernel function.
        asdefault : bool, optional
            If ``True``, set this backend as the default for *platform*
            even if a default already exists.  Default is ``False``.

        Raises
        ------
        AssertionError
            If *backend* or *platform* is not a string, or if *kg* is not
            callable.

        See Also
        --------
        def_numba_kernel : Shorthand for ``def_kernel('numba', 'cpu', ...)``.
        def_warp_kernel : Shorthand for ``def_kernel('warp', 'gpu', ...)``.
        def_pallas_kernel : Shorthand for Pallas kernels on GPU or TPU.
        set_default : Change the default backend for a platform after
            registration.
        """
        assert isinstance(backend, str), f'The `backend` should be a string, but got {type(backend)}.'
        assert isinstance(platform, str), f'The `platform` should be a string, but got {type(platform)}.'
        assert callable(kg), f'The `kg` should be a callable, but got {type(kg)}.'

        # Create kernel entry
        entry = KernelEntry(
            backend=backend,
            platform=platform,
            kernel_generator=kg,
        )

        # Store kernel in the platform's dict
        if platform not in self._kernels:
            self._kernels[platform] = {}
        self._kernels[platform][backend] = entry

        # Default logic:
        # 1. First kernel for a platform becomes the default automatically
        # 2. Explicit asdefault=True overrides any existing default
        if asdefault or platform not in self._defaults:
            self._defaults[platform] = backend

        # Register fallback lowering once per platform
        if platform not in self._registered_platforms:
            self._register_fallback_lowering(platform)
            self._registered_platforms.add(platform)

    def _register_fallback_lowering(self, platform: str):
        """Register a MLIR lowering function that dispatches to the default backend.

        Creates a lowering function for the given *platform* that, when
        invoked by JAX during compilation, selects the appropriate kernel
        backend (either the default or one explicitly requested via the
        ``backend`` keyword argument) and calls it.

        Parameters
        ----------
        platform : str
            The platform to register the lowering for (e.g., ``'cpu'``,
            ``'gpu'``, ``'tpu'``).

        Raises
        ------
        KernelFallbackExhaustedError
            If no kernels are registered for the platform, or if the
            explicitly requested backend is not available.
        """

        def fallback_kernel_fn(*args, **kwargs):
            # Get kernels dict for this platform
            kernels = self._kernels.get(platform, {})
            if not kernels:
                raise KernelFallbackExhaustedError(
                    f"No kernels registered for platform '{platform}' in primitive '{self.name}'."
                )

            # Determine which backend to use
            # Priority: per-call > global > per-primitive > first registered
            backend_to_use = kwargs.pop('backend', None)
            if backend_to_use is not None:
                if isinstance(backend_to_use, str) and backend_to_use == '':
                    raise ValueError(
                        f"backend cannot be an empty string in primitive '{self.name}'."
                    )
                if backend_to_use not in kernels:
                    raise KernelFallbackExhaustedError(
                        f'{backend_to_use} not available for platform {platform} in primitive '
                        f'{self.name}.'
                    )
            else:
                # Check global backend setting
                global_be = get_backend(platform)
                if global_be is not None and global_be in kernels:
                    backend_to_use = global_be
                else:
                    backend_to_use = self._defaults.get(platform)

            # Get the kernel entry
            if backend_to_use and backend_to_use in kernels:
                pass
            else:
                # Fallback to first registered kernel
                backend_to_use = next(iter(kernels))

            if backend_to_use == 'pallas':
                check_pallas_jax_version()
            elif backend_to_use == 'warp':
                check_warp_installed()

            entry = kernels[backend_to_use]
            kernel = entry.kernel_generator(**kwargs)
            return kernel(*args)

        # Register the lowering with JAX
        lower = mlir.lower_fun(fallback_kernel_fn, multiple_results=True)
        mlir.register_lowering(self.primitive, lower, platform=platform)

    def def_numba_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Numba kernel for the CPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='numba'`` and ``platform='cpu'``.

        Parameters
        ----------
        kg : KernelGenerator
            A callable that generates the Numba kernel function.
        asdefault : bool, optional
            If ``True``, set Numba as the default CPU backend.  Default
            is ``False`` (the first registered CPU kernel becomes the
            default automatically).

        See Also
        --------
        def_kernel : General kernel registration method.
        def_warp_kernel : Register a Warp kernel for GPU.
        def_pallas_kernel : Register a Pallas kernel for GPU or TPU.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_numba_kernel(my_numba_kernel_gen)  # doctest: +SKIP
        """
        self.def_kernel(backend='numba', platform='cpu', kg=kg, asdefault=asdefault)

    def def_warp_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Warp kernel for the GPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='warp'`` and ``platform='gpu'``.

        Parameters
        ----------
        kg : KernelGenerator
            A callable that generates the Warp kernel function.
        asdefault : bool, optional
            If ``True``, set Warp as the default GPU backend.  Default
            is ``False``.

        See Also
        --------
        def_kernel : General kernel registration method.
        def_numba_kernel : Register a Numba kernel for CPU.
        def_pallas_kernel : Register a Pallas kernel for GPU or TPU.
        def_triton_kernel : Register a Triton kernel for GPU.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_warp_kernel(my_warp_kernel_gen)  # doctest: +SKIP
        """
        self.def_kernel(backend='warp', platform='gpu', kg=kg, asdefault=asdefault)

    def def_triton_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Triton kernel for the GPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='triton'`` and ``platform='gpu'``.

        Parameters
        ----------
        kg : KernelGenerator
            A callable that generates the Triton kernel function.
        asdefault : bool, optional
            If ``True``, set Triton as the default GPU backend.  Default
            is ``False``.

        See Also
        --------
        def_kernel : General kernel registration method.
        def_warp_kernel : Register a Warp kernel for GPU.
        def_pallas_kernel : Register a Pallas kernel for GPU or TPU.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_triton_kernel(my_triton_kernel_gen)  # doctest: +SKIP
        """
        self.def_kernel(backend='triton', platform='gpu', kg=kg, asdefault=asdefault)

    def def_pallas_kernel(
        self,
        platform: str,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Pallas kernel for the GPU or TPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='pallas'``.

        Parameters
        ----------
        platform : str
            Target platform.  Must be ``'gpu'`` or ``'tpu'``.
        kg : KernelGenerator
            A callable that generates the Pallas kernel function.
        asdefault : bool, optional
            If ``True``, set Pallas as the default backend for the
            given platform.  Default is ``False``.

        Raises
        ------
        AssertionError
            If *platform* is not ``'gpu'`` or ``'tpu'``.

        See Also
        --------
        def_kernel : General kernel registration method.
        def_warp_kernel : Register a Warp kernel for GPU.
        def_numba_kernel : Register a Numba kernel for CPU.

        Notes
        -----
        Pallas kernels require JAX >= 0.7.1.  The version check is
        performed lazily at dispatch time, not at registration time.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_pallas_kernel('gpu', my_pallas_gen, asdefault=True)  # doctest: +SKIP
        """
        assert platform in ['gpu', 'tpu'], f'The `platform` should be either `gpu` or `tpu`, but got {platform}.'
        self.def_kernel(backend='pallas', platform=platform, kg=kg, asdefault=asdefault)

    def def_cuda_raw_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a cuda_raw (nvcc-compiled) kernel for the CPU or GPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='cuda_raw'``.  The kernel generator function should
        call :func:`brainevent.kernix.load_cuda_file` or
        :func:`brainevent.kernix.load_cuda_inline` to compile and
        register the CUDA kernel, then return a closure that calls it via
        ``jax.ffi.ffi_call``.

        Parameters
        ----------
        platform : str
            Target platform.  Must be ``'cpu'`` or ``'gpu'``.
        kg : KernelGenerator
            A callable that compiles and returns the kernel function.
        asdefault : bool, optional
            If ``True``, set cuda_raw as the default backend for the
            given platform.  Default is ``False``.

        See Also
        --------
        def_kernel : General kernel registration method.
        """
        self.def_kernel(backend='cuda_raw', platform='gpu', kg=kg, asdefault=asdefault)

    def def_numba_cuda_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Numba CUDA kernel for the GPU platform.

        Convenience wrapper around :meth:`def_kernel` with
        ``backend='numba_cuda'`` and ``platform='gpu'``.

        Parameters
        ----------
        kg : KernelGenerator
            A callable that generates the Numba CUDA kernel function.
        asdefault : bool, optional
            If ``True``, set Numba CUDA as the default GPU backend.
            Default is ``False``.

        See Also
        --------
        def_kernel : General kernel registration method.
        def_warp_kernel : Register a Warp kernel for GPU.
        numba_cuda_kernel : Standalone function for wrapping a single
            Numba CUDA kernel.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_numba_cuda_kernel(my_numba_cuda_gen)  # doctest: +SKIP
        """
        self.def_kernel(backend='numba_cuda', platform='gpu', kg=kg, asdefault=asdefault)

    def set_default(self, platform: str, backend: str):
        """Set the default backend for a platform.

        After this call, all subsequent dispatches on the given
        *platform* will use the specified *backend* unless overridden
        by an explicit ``backend=`` keyword argument at call time.

        Parameters
        ----------
        platform : str
            The platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
        backend : str
            The backend name to set as default.  Must already be
            registered for the given platform.

        Raises
        ------
        ValueError
            If no kernels are registered for *platform*, or if *backend*
            is not registered for *platform*.

        See Also
        --------
        get_default : Retrieve the current default backend for a
            platform.
        defaults : Property returning all default backends.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_warp_kernel(warp_gen)  # doctest: +SKIP
            >>> kernel.def_pallas_kernel('gpu', pallas_gen)  # doctest: +SKIP
            >>> kernel.set_default('gpu', 'pallas')  # doctest: +SKIP
        """
        if platform not in self._kernels:
            raise ValueError(f"No kernels registered for platform '{platform}'")
        if backend not in self._kernels[platform]:
            available = list(self._kernels[platform].keys())
            raise ValueError(
                f"Backend '{backend}' not registered for platform '{platform}'. "
                f"Available: {available}"
            )
        self._defaults[platform] = backend

    def get_default(self, platform: str) -> Optional[str]:
        """Get the current default backend for a platform.

        Parameters
        ----------
        platform : str
            The platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).

        Returns
        -------
        str or None
            The default backend name, or ``None`` if no default is set
            for the given platform.

        See Also
        --------
        set_default : Set the default backend for a platform.
        defaults : Property returning all default backends.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_numba_kernel(numba_gen)  # doctest: +SKIP
            >>> kernel.get_default('cpu')  # doctest: +SKIP
            'numba'
        """
        return self._defaults.get(platform)

    @property
    def defaults(self) -> Dict[str, str]:
        """Return a copy of all default backends.

        Returns
        -------
        dict of str to str
            A dictionary mapping platform names to their default backend
            names.  Modifying the returned dictionary does not affect the
            internal state.

        See Also
        --------
        get_default : Retrieve the default for a single platform.
        set_default : Change the default backend for a platform.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_numba_kernel(numba_gen)  # doctest: +SKIP
            >>> kernel.def_pallas_kernel('gpu', pallas_gen)  # doctest: +SKIP
            >>> kernel.defaults  # doctest: +SKIP
            {'cpu': 'numba', 'gpu': 'pallas'}
        """
        return self._defaults.copy()

    def def_batching_rule(self, fun: Callable):
        """Define a custom batching rule for the primitive.

        The batching rule specifies how the primitive should behave when
        applied to batched inputs (i.e., inputs with a leading batch
        dimension introduced by ``jax.vmap``).

        Parameters
        ----------
        fun : callable
            A function implementing the batching logic.  It receives
            batched arguments and per-argument batch dimension indices,
            and must return ``(batched_outputs, output_batch_dims)``.
            See JAX documentation for
            ``jax.interpreters.batching.primitive_batchers``.

        See Also
        --------
        register_general_batching : Register the default general-purpose
            batching rule.
        general_batching_rule : The general-purpose batching
            implementation used by default.

        Examples
        --------
        .. code-block:: python

            >>> def my_batching(args, axes, **kwargs):
            ...     # Custom batching logic
            ...     return batched_out, out_dims
            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_batching_rule(my_batching)  # doctest: +SKIP
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun: Callable):
        """Define a custom JVP (Jacobian-vector product) rule.

        This rule is used for forward-mode automatic differentiation.  It
        specifies how to compute the directional derivative of the
        primitive's outputs with respect to its inputs.

        Parameters
        ----------
        fun : callable
            A function implementing the JVP logic.  See JAX documentation
            for ``jax.interpreters.ad.primitive_jvps``.

        See Also
        --------
        def_jvp_rule2 : Convenience method for defining per-input JVP
            rules.
        def_transpose_rule : Define a transpose rule for reverse-mode AD.

        Examples
        --------
        .. code-block:: python

            >>> def my_jvp(primals, tangents, **params):
            ...     val_out = kernel.primitive.bind(*primals, **params)
            ...     tangent_out = ...  # compute tangent
            ...     return val_out, tangent_out
            >>> kernel.def_jvp_rule(my_jvp)  # doctest: +SKIP
        """
        ad.primitive_jvps[self.primitive] = fun

    def def_jvp_rule2(self, *jvp_rules):
        """Define per-input JVP rules for the primitive.

        This is a convenience method similar to ``jax.interpreters.ad.defjvp``
        but adapted for primitives that return multiple results.  Each rule
        corresponds to one input primal and computes the tangent
        contribution from that input.

        Parameters
        ----------
        *jvp_rules : callable or None
            One callable per input primal.  Each callable has the
            signature ``rule(tangent, *primals, **params) -> tangent_out``.
            Pass ``None`` for inputs whose JVP contribution is zero.

        See Also
        --------
        def_jvp_rule : Define a single monolithic JVP rule.
        defjvp : The underlying utility function.

        Examples
        --------
        .. code-block:: python

            >>> def jvp_input0(t, x, y, **kw):
            ...     return t * y
            >>> def jvp_input1(t, x, y, **kw):
            ...     return t * x
            >>> kernel.def_jvp_rule2(jvp_input0, jvp_input1)  # doctest: +SKIP
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun: Callable):
        """Define a custom transpose rule for reverse-mode AD.

        The transpose rule is invoked during ``jax.linear_transpose`` and
        defines how to propagate cotangent vectors (gradients) backward
        through the primitive.

        Parameters
        ----------
        fun : callable
            A function implementing the transpose logic.  See JAX
            documentation for
            ``jax.interpreters.ad.primitive_transposes``.

        See Also
        --------
        def_jvp_rule : Define a JVP rule for forward-mode AD.
        def_jvp_rule2 : Define per-input JVP rules.

        Examples
        --------
        .. code-block:: python

            >>> def my_transpose(ct, *args, **params):
            ...     # Propagate cotangent backward
            ...     return (ct_input0, ct_input1)
            >>> kernel.def_transpose_rule(my_transpose)  # doctest: +SKIP
        """
        ad.primitive_transposes[self.primitive] = fun

    def register_general_batching(self):
        """Register the default general-purpose batching rule.

        This method applies a common batching pattern that handles most
        custom operators by using ``jax.lax.scan`` to map the kernel
        over the batch dimension.  It is called automatically during
        ``__init__``; call it again to restore the default after
        overriding with :meth:`def_batching_rule`.

        See Also
        --------
        def_batching_rule : Override with a custom batching rule.
        general_batching_rule : The underlying batching implementation.

        Notes
        -----
        The general batching rule moves all batch dimensions to axis 0
        and uses ``jax.lax.scan`` to iterate over the batch.  This is
        correct but may be slower than a hand-written batching rule for
        operations that can natively handle batched inputs.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_batching_rule(custom_rule)  # doctest: +SKIP
            >>> kernel.register_general_batching()  # Restore default  # doctest: +SKIP
        """
        prim = self.primitive
        batching.primitive_batchers[prim] = functools.partial(general_batching_rule, prim)

    def def_call(self, fn: Callable):
        """Associate a high-level call function with this primitive.

        The call function is the user-facing Python function that
        prepares arguments and invokes the primitive.  It is stored
        so that :meth:`call` and :meth:`benchmark` can use it.

        Parameters
        ----------
        fn : callable
            The call function (e.g., ``binary_csrmv_p_call``).  It
            should accept the same positional and keyword arguments
            that the end user would pass.

        See Also
        --------
        call : Invoke the registered call function.
        benchmark : Benchmark the registered call function.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_call(my_call_fn)  # doctest: +SKIP
            >>> kernel.call(x, y)  # delegates to my_call_fn  # doctest: +SKIP
        """
        self._call_fn = fn

    def call(self, *args, **kwargs):
        """Invoke the registered call function.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the call function.
        **kwargs
            Keyword arguments forwarded to the call function.

        Returns
        -------
        result
            The return value of the call function.

        Raises
        ------
        ValueError
            If no call function has been registered via :meth:`def_call`.

        See Also
        --------
        def_call : Register a call function.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_call(my_call_fn)  # doctest: +SKIP
            >>> result = kernel.call(x, y, backend='pallas')  # doctest: +SKIP
        """
        if self._call_fn is None:
            raise ValueError(
                f"No call function registered for '{self.name}'. "
                "Use def_call() to register one before calling."
            )
        return self._call_fn(*args, **kwargs)

    def def_tags(self, *tags: str):
        """Set categorization tags for this primitive.

        Tags are used by the CLI and the global primitive registry to
        filter primitives by sparse format (e.g., ``'csr'``, ``'coo'``)
        and value type (e.g., ``'binary'``, ``'float'``).

        Parameters
        ----------
        *tags : str
            Tag strings (e.g., ``'csr'``, ``'binary'``).

        See Also
        --------
        def_benchmark_data : Register a benchmark data generator.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('binary_csrmv')
            >>> kernel.def_tags('csr', 'binary', 'mv')  # doctest: +SKIP
        """
        self._tags = set(tags)

    def def_benchmark_data(self, fn: Callable):
        """Register a benchmark data generator function.

        The generator produces a list of :class:`BenchmarkConfig`
        instances that define the parameter combinations to benchmark
        for this primitive.

        Parameters
        ----------
        fn : callable
            A callable with signature
            ``fn(*, platform: str) -> List[BenchmarkConfig]``.

        See Also
        --------
        benchmark : Run benchmarks using the registered call function.
        def_tags : Set categorization tags for filtering.

        Examples
        --------
        .. code-block:: python

            >>> def my_data_gen(*, platform):
            ...     return [BenchmarkConfig(n=100), BenchmarkConfig(n=1000)]
            >>> kernel.def_benchmark_data(my_data_gen)  # doctest: +SKIP
        """
        self._benchmark_data_fn = fn

    def available_backends(self, platform: str) -> List[str]:
        """Return the list of registered backend names for a platform.

        Parameters
        ----------
        platform : str
            The platform name (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).

        Returns
        -------
        list of str
            Backend names registered for *platform*.  Returns an empty
            list if no kernels are registered for the platform.

        See Also
        --------
        defaults : Property returning the default backend for each
            platform.

        Examples
        --------
        .. code-block:: python

            >>> kernel = XLACustomKernel('my_op')
            >>> kernel.def_numba_kernel(numba_gen)  # doctest: +SKIP
            >>> kernel.available_backends('cpu')  # doctest: +SKIP
            ['numba']
        """
        if platform not in self._kernels:
            return []
        return list(self._kernels[platform].keys())

    def benchmark(
        self,
        *,
        platform: str,
        n_warmup: int = 5,
        n_runs: int = 20,
        n_batch_per_run: int = 1,
        compare_results: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        verbose: bool = False,
        catch_errors: bool = True,
        backends: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Benchmark all registered backends across every configured data config.

        Iterates over the :class:`~brainevent._op.benchmark.BenchmarkConfig`
        instances produced by the registered benchmark data function, runs
        the call function on every registered backend for the given
        *platform*, collects timing statistics, and returns a unified
        :class:`~brainevent._op.benchmark.BenchmarkResult`.

        Parameters
        ----------
        platform : str
            Target platform (``'cpu'``, ``'gpu'``, or ``'tpu'``).
        n_warmup : int, optional
            Number of warmup runs before timing.  Default is ``5``.
        n_runs : int, optional
            Number of timed runs per backend per config.  Default is
            ``20``.
        n_batch_per_run : int, optional
            Number of back-to-back kernel calls issued within each timed
            interval before blocking.  Default is ``1`` (per-call
            latency).  Higher values amortise blocking overhead, useful
            for measuring throughput on asynchronous GPU/TPU execution.
            Reported times are always **per-call** values.
        compare_results : bool, optional
            If ``True`` (default), verify that outputs match across
            backends for each config using ``jnp.allclose``.  Mismatches
            are printed as warnings.
        verbose : bool, optional
            If ``True``, print a one-line timing summary after each
            (config, backend) pair completes.  Useful for monitoring
            progress when the config list is large.  Default is
            ``False``.
        rtol : float, optional
            Relative tolerance for output comparison when *compare_results*
            is ``True``.  Default is ``1e-3``.
        atol : float, optional
            Absolute tolerance for output comparison when *compare_results*
            is ``True``.  Default is ``1e-3``.
        catch_errors : bool, optional
            If ``True`` (default), runtime errors raised by a backend during
            warmup or timed runs are caught and stored in the returned
            :class:`~brainevent._op.benchmark.BenchmarkRecord` as
            ``success=False`` with the exception message in ``error``.
            The benchmark continues with the remaining backends.
            Set to ``False`` to let exceptions propagate immediately, which
            is useful for debugging a specific backend failure.

        Returns
        -------
        BenchmarkResult
            A :class:`~brainevent._op.benchmark.BenchmarkResult`
            containing one :class:`~brainevent._op.benchmark.BenchmarkRecord`
            per (config × backend) pair.  Failed runs (when
            *catch_errors* is ``True``) are included with ``success=False``.

        Raises
        ------
        BenchmarkDataFnNotProvidedError
            If no benchmark data function has been registered.  Use
            :meth:`def_benchmark_data` first.
        ValueError
            If no call function has been registered (use :meth:`def_call`
            first), or if no backends are registered for *platform*.
        Exception
            Any backend runtime error, when *catch_errors* is ``False``.

        See Also
        --------
        def_call : Register the call function to benchmark.
        def_benchmark_data : Register a data generator for automated
            benchmarking.
        BenchmarkResult : Unified result container.
        """
        from brainevent._error import BenchmarkDataFnNotProvidedError

        if self._call_fn is None:
            raise ValueError(
                f"No call function registered for '{self.name}'. "
                "Use def_call() to register one before benchmarking."
            )

        if self._benchmark_data_fn is None:
            raise BenchmarkDataFnNotProvidedError(
                f"No benchmark data function registered for '{self.name}'. "
                "Use def_benchmark_data() to register one before benchmarking."
            )

        if backends is not None:
            backends_to_test = backends
        else:
            backends_to_test = self.available_backends(platform)
        if not backends_to_test:
            raise ValueError(
                f"No backends registered for platform '{platform}' in primitive '{self.name}'."
            )

        configs = self._benchmark_data_fn(platform=platform)

        records: List[BenchmarkRecord] = []

        for config in configs:
            config_outputs = {}  # backend -> output for cross-backend comparison
            config = config.put_args()

            for be in backends_to_test:

                @jax.jit
                def run_fn(*args):
                    return self._call_fn(*args, backend=be, **config.kernel_kwargs)

                if catch_errors:
                    try:
                        mean_s, std_s, min_s, _max_s, output = benchmark_function(
                            run_fn, n_warmup, n_runs, n_batch_per_run=n_batch_per_run, data=config.args,
                        )
                        record = BenchmarkRecord(
                            platform=platform,
                            backend=be,
                            label=config.name,
                            mean_ms=mean_s * 1000.0,
                            std_ms=std_s * 1000.0,
                            min_ms=min_s * 1000.0,
                            throughput=None,
                            success=True,
                            error=None,
                            kernel_kwargs=dict(config.kernel_kwargs),
                            data_kwargs=dict(config.data_kwargs),
                        )
                        config_outputs[be] = output
                    except Exception as exc:
                        error_msg = f"{type(exc).__name__}: {exc}"
                        record = BenchmarkRecord(
                            platform=platform,
                            backend=be,
                            label=config.name,
                            mean_ms=float('nan'),
                            std_ms=float('nan'),
                            min_ms=float('nan'),
                            throughput=None,
                            success=False,
                            error=error_msg,
                            kernel_kwargs=dict(config.kernel_kwargs),
                            data_kwargs=dict(config.data_kwargs),
                        )
                        if verbose:
                            print(
                                f"[{self.name}] [{platform}|{be}] {config.name}: "
                                f"FAILED — {error_msg}"
                            )
                else:
                    mean_s, std_s, min_s, _max_s, output = benchmark_function(
                        run_fn, n_warmup, n_runs, n_batch_per_run=n_batch_per_run, data=config.args,
                    )
                    record = BenchmarkRecord(
                        platform=platform,
                        backend=be,
                        label=config.name,
                        mean_ms=mean_s * 1000.0,
                        std_ms=std_s * 1000.0,
                        min_ms=min_s * 1000.0,
                        throughput=None,
                        success=True,
                        error=None,
                        kernel_kwargs=dict(config.kernel_kwargs),
                        data_kwargs=dict(config.data_kwargs),
                    )
                    config_outputs[be] = output

                records.append(record)
                if verbose and record.success:
                    print(
                        f"[{self.name}] [{platform}|{be}] {config.name}: "
                        f"mean={record.mean_ms:.3f}ms  "
                        f"std={record.std_ms:.3f}ms  "
                        f"min={record.min_ms:.3f}ms"
                    )

            # Optionally compare outputs across backends for this config
            if compare_results and len(config_outputs) > 1:
                import jax.numpy as jnp
                be_list = list(config_outputs.keys())
                ref_be = be_list[0]
                ref_out = config_outputs[ref_be]
                for other_be in be_list[1:]:
                    other_out = config_outputs[other_be]
                    try:
                        if isinstance(ref_out, (list, tuple)):
                            for i, (r, o) in enumerate(zip(ref_out, other_out)):
                                if not jnp.allclose(r, o, rtol=rtol, atol=atol):
                                    print(
                                        f"[{self.name}][{config.name}] "
                                        f"{ref_be} vs {other_be}: output[{i}] mismatch"
                                    )
                        else:
                            if not jnp.allclose(ref_out, other_out, rtol=rtol, atol=atol):
                                print(
                                    f"[{self.name}][{config.name}] "
                                    f"{ref_be} vs {other_be}: output mismatch"
                                )
                    except Exception as cmp_exc:
                        print(
                            f"[{self.name}][{config.name}] "
                            f"{ref_be} vs {other_be}: comparison error: {cmp_exc}"
                        )

        return BenchmarkResult(records=records, primitive_name=self.name)
