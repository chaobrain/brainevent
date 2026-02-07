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
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import jax
from jax.interpreters import xla, batching, ad, mlir

from brainevent._compatible_import import Primitive
from brainevent._error import KernelFallbackExhaustedError, KernelExecutionError
from brainevent._typing import KernelGenerator
from .benchmark import BenchmarkResult, BenchmarkReport, benchmark_function
from .util import general_batching_rule, defjvp, OutType, abstract_arguments

__all__ = [
    'XLACustomKernel',
    'KernelEntry',
]


@dataclass
class KernelEntry:
    """Represents a registered kernel for a specific backend and platform.

    Attributes:
        backend: The backend name (e.g., 'numba', 'warp', 'pallas', 'triton').
        platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
        kernel_generator: A callable that generates the kernel function.
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

    - **CPU**: Numba, TVM FFI
    - **GPU**: Pallas, TVM FFI, Numba CUDA, Warp, Triton
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

    Args:
        name: The unique name for the custom JAX primitive.

    Example:
        >>> kernel = XLACustomKernel('my_custom_op')
        >>> kernel.def_numba_kernel(numba_kernel_generator)  # CPU default
        >>> kernel.def_pallas_kernel('gpu', pallas_kernel_generator, asdefault=True)
        >>> kernel.def_warp_kernel(warp_kernel_generator)  # Alternative GPU backend
        >>> print(kernel.defaults)  # {'cpu': 'numba', 'gpu': 'pallas'}
        >>> kernel.set_default('gpu', 'warp')  # Change GPU default
        >>> result = kernel(input_array, outs=[jax.ShapeDtypeStruct((10,), jnp.float32)])

    See Also:
        :class:`KernelEntry`: Represents a registered kernel.
    """

    __module__ = 'brainevent'

    def __init__(self, name: str):
        # primitive
        self.name = name
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True

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
        # lazy flag for user defaults from config file
        self._user_defaults_applied: bool = False

        # Auto-register in global registry
        from brainevent._registry import register_primitive
        register_primitive(name, self)

    def _abstract_eval(self, *ins, outs: OutType, **kwargs):
        """
        Abstract evaluation rule for the JAX primitive.

        This method defines how JAX should determine the shape and dtype of the
        primitive's output(s) based on the shapes and dtypes of the inputs,
        without performing the actual computation. In this specific implementation,
        the output shapes and dtypes are explicitly provided via the `outs`
        parameter during the `primitive.bind` call and are simply returned here.

        Args:
            *ins: Abstract values (e.g., `jax.core.ShapedArray`) corresponding
                  to the input operands. Not directly used in this implementation
                  as output shapes are pre-determined.
            outs: A sequence of `jax.core.ShapedArray` objects specifying the
                  expected shape and dtype of each output. This is passed as a
                  parameter to the primitive binding.
            **kwargs: Additional keyword arguments passed during primitive binding.
                      Not used in this abstract evaluation rule.

        Returns:
            A tuple containing the `jax.core.ShapedArray` objects passed in `outs`,
            representing the abstract value of the primitive's output(s).
        """
        return tuple(outs)

    def __call__(self, *ins, outs: OutType, **kwargs):
        """Call the primitive with the given inputs.

        Args:
            *ins: Input arrays to the primitive.
            outs: Output specification (shapes and dtypes).
            backend: Optional backend to use. If specified and a kernel
                with that backend is available for the target platform,
                it will be prioritized. If None, kernels are tried in
                default priority order.
            **kwargs: Additional keyword arguments passed to the kernel.

        Returns:
            The output(s) of the primitive.
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
        """Register a kernel for a specific backend and platform.

        Args:
            backend: The backend name (e.g., 'numba', 'warp', 'pallas').
            platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
            kg: A kernel generator callable that creates the kernel function.
            asdefault: If True, set this backend as the default for the platform.
                The first kernel registered for a platform automatically becomes
                the default unless another kernel is explicitly set as default.
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
        """Register a lowering function that uses the default backend.

        This creates a lowering function that uses the default backend for the
        platform, or a specific backend if requested via _preferred_backend.
        On failure, it shows alternative backends available.

        Args:
            platform: The platform to register the lowering for.
        """

        def fallback_kernel_fn(*args, **kwargs):
            # Apply user defaults lazily on first dispatch
            self._apply_user_defaults()

            # Get kernels dict for this platform
            kernels = self._kernels.get(platform, {})
            if not kernels:
                raise KernelFallbackExhaustedError(
                    f"No kernels registered for platform '{platform}' in primitive '{self.name}'."
                )

            # Determine which backend to use
            backend_to_use = self._defaults.get(platform)

            # Get the kernel entry
            if backend_to_use and backend_to_use in kernels:
                entry = kernels[backend_to_use]
            else:
                # Fallback to first registered kernel
                backend_to_use = next(iter(kernels))
                entry = kernels[backend_to_use]

            try:
                kernel = entry.kernel_generator(**kwargs)
                return kernel(*args)
            except Exception as e:
                # Build helpful error message with alternatives
                alternatives = [b for b in kernels.keys() if b != backend_to_use]
                alt_msg = ""
                if alternatives:
                    alt_msg = (
                        f"\n\nAlternative backends available for '{platform}':\n"
                        + "\n".join(f"  - {b}" for b in alternatives)
                        + f"\n\nTo use an alternative:\n"
                          f"  1. Call with backend='{alternatives[0]}'\n"
                          f"  2. Or set default: kernel.set_default('{platform}', '{alternatives[0]}')"
                    )
                raise KernelExecutionError(
                    f"Backend '{backend_to_use}' failed on platform '{platform}':\n"
                    f"  {type(e).__name__}: {e}{alt_msg}"
                ) from e

        # Register the lowering with JAX
        lower = mlir.lower_fun(fallback_kernel_fn, multiple_results=True)
        mlir.register_lowering(self.primitive, lower, platform=platform)

    def def_numba_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Numba kernel for CPU platform.

        This is a convenience method equivalent to:
        ``self.def_kernel(backend='numba', platform='cpu', kg=kg, asdefault=asdefault)``

        Args:
            kg: A kernel generator callable that creates the Numba kernel function.
            asdefault: If True, set this as the default backend for CPU.
        """
        self.def_kernel(backend='numba', platform='cpu', kg=kg, asdefault=asdefault)

    def def_warp_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Warp kernel for GPU platform.

        This is a convenience method equivalent to:
        ``self.def_kernel(backend='warp', platform='gpu', kg=kg, asdefault=asdefault)``

        Args:
            kg: A kernel generator callable that creates the Warp kernel function.
            asdefault: If True, set this as the default backend for GPU.
        """
        self.def_kernel(backend='warp', platform='gpu', kg=kg, asdefault=asdefault)

    def def_triton_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Triton kernel for GPU platform.

        This is a convenience method equivalent to:
        ``self.def_kernel(backend='triton', platform='gpu', kg=kg, asdefault=asdefault)``

        Args:
            kg: A kernel generator callable that creates the Triton kernel function.
            asdefault: If True, set this as the default backend for GPU.
        """
        self.def_kernel(backend='triton', platform='gpu', kg=kg, asdefault=asdefault)

    def def_pallas_kernel(
        self,
        platform: str,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Pallas kernel for GPU or TPU platform.

        This is a convenience method equivalent to:
        ``self.def_kernel(backend='pallas', platform=platform, kg=kg, asdefault=asdefault)``

        Args:
            platform: The target platform, must be either 'gpu' or 'tpu'.
            kg: A kernel generator callable that creates the Pallas kernel function.
            asdefault: If True, set this as the default backend for the platform.
        """
        assert platform in ['gpu', 'tpu'], f'The `platform` should be either `gpu` or `tpu`, but got {platform}.'
        self.def_kernel(backend='pallas', platform=platform, kg=kg, asdefault=asdefault)

    def def_tvmffi_kernel(
        self,
        platform: str,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a TVM FFI kernel for CPU or GPU platform.

        This is a convenience method equivalent to:
        ``self.def_kernel(backend='tvmffi', platform=platform, kg=kg, asdefault=asdefault)``

        Args:
            platform: The target platform, must be either 'cpu' or 'gpu'.
            kg: A kernel generator callable that creates the TVM FFI kernel function.
            asdefault: If True, set this as the default backend for the platform.
        """
        assert platform in ['cpu', 'gpu'], f'The `platform` should be either `cpu` or `gpu`, but got {platform}.'
        self.def_kernel(backend='tvmffi', platform=platform, kg=kg, asdefault=asdefault)

    def def_numba_cuda_kernel(
        self,
        kg: KernelGenerator,
        asdefault: bool = False
    ):
        """Register a Numba CUDA kernel for GPU platform.

        Args:
            kg: A kernel generator callable that creates the Numba CUDA kernel function.
            asdefault: If True, set this as the default backend for GPU.
        """
        self.def_kernel(backend='numba_cuda', platform='gpu', kg=kg, asdefault=asdefault)

    def set_default(self, platform: str, backend: str, persist: bool = False):
        """Set the default backend for a platform.

        Args:
            platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
            backend: The backend name to set as default.
            persist: If True, save the default to the user config file
                so it persists across sessions.

        Raises:
            ValueError: If no kernels registered for the platform or
                the specified backend is not registered for the platform.
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
        if persist:
            from brainevent._config import set_user_default
            set_user_default(self.name, platform, backend)

    def get_default(self, platform: str) -> Optional[str]:
        """Get the default backend for a platform.

        Args:
            platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').

        Returns:
            The default backend name, or None if no default is set.
        """
        return self._defaults.get(platform)

    @property
    def defaults(self) -> Dict[str, str]:
        """Get all default backends.

        Returns:
            A dictionary mapping platform names to their default backend names.
        """
        return self._defaults.copy()

    def def_batching_rule(self, fun: Callable):
        """
        Defines a custom batching rule for the JAX primitive.

        This rule specifies how the primitive should behave when applied to
        batched inputs (inputs with a leading batch dimension).

        Args:
            fun: A callable that implements the batching logic. It typically
                 takes batched arguments and batch dimensions as input and returns
                 batched outputs and output batch dimensions. See JAX documentation
                 for `batching.primitive_batchers`.
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun: Callable):
        """
        Defines a custom JVP (Jacobian-vector product) rule for the primitive.

        This rule is used for forward-mode automatic differentiation (AD). It
        specifies how to compute the directional derivative of the primitive's
        output with respect to its inputs.

        Args:
            fun: A callable that implements the JVP logic. See JAX documentation
                 for `ad.primitive_jvps`.
        """
        ad.primitive_jvps[self.primitive] = fun

    def def_jvp_rule2(self, *jvp_rules):
        """
        Defines the JVP (Jacobian-vector product) rules for the primitive.

        This is a convenience method similar to `jax.interpreters.ad.defjvp`,
        but specifically adapted to handle primitives that may have multiple
        output values. It registers the JVP rules necessary for forward-mode
        automatic differentiation.

        Args:
            *jvp_rules: A sequence of callables, each defining the JVP rule for
                        a corresponding input primal. See the implementation of
                        `brainevent._xla_custom_op_util.defjvp` and JAX AD
                        documentation for details.
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun: Callable):
        """
        Defines a custom transpose rule for the primitive.

        This rule is used for reverse-mode automatic differentiation (AD),
        specifically within the context of `jax.linear_transpose`. It defines
        how to propagate gradients backward through the primitive.

        Args:
            fun: A callable that implements the transpose logic. See JAX
                 documentation for `ad.primitive_transposes`.
        """
        ad.primitive_transposes[self.primitive] = fun

    def register_general_batching(self):
        """
        Registers a predefined general-purpose batching rule for the primitive.

        This method applies a common batching pattern suitable for many custom
        operators, likely handling element-wise operations or operations where
        batching involves mapping the kernel over the batch dimension. It uses
        the `general_batching_rule` function internally.
        """
        prim = self.primitive
        batching.primitive_batchers[prim] = functools.partial(general_batching_rule, prim)

    def def_call(self, fn: Callable):
        """Associate a call function with this primitive.

        The call function is automatically JIT-compiled with all keyword-only
        arguments treated as static.

        Args:
            fn: The call function (e.g., binary_csrmv_p_call).
        """

        # Get all keyword-only parameter names from the function signature
        sig = inspect.signature(fn)
        static_argnames = [
            name for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.KEYWORD_ONLY
        ]

        # Wrap with JIT, treating all kwargs as static
        self._call_fn = jax.jit(fn, static_argnames=static_argnames)

    def call(self, *args, **kwargs):
        """Call the associated call function.

        Args:
            *args: Positional arguments for the call function.
            **kwargs: Keyword arguments for the call function.

        Returns:
            The result of the call function.
        """
        if self._call_fn is None:
            raise ValueError(
                f"No call function registered for '{self.name}'. "
                "Use def_call() to register one before calling."
            )
        return self._call_fn(*args, **kwargs)

    def def_tags(self, *tags: str):
        """Set categorization tags for this primitive.

        Tags are used by the CLI and registry to filter primitives
        by format (e.g., 'csr', 'coo') and type (e.g., 'binary', 'float').

        Args:
            *tags: Tag strings (e.g., 'csr', 'binary').
        """
        self._tags = set(tags)

    def def_benchmark_data(self, fn: Callable):
        """Register a benchmark data generator function.

        The function should return a list of :class:`BenchmarkConfig`
        instances, each covering a specific parameter combination for
        this primitive.

        Args:
            fn: A callable with signature
                ``fn(*, platform) -> List[BenchmarkConfig]``.
        """
        self._benchmark_data_fn = fn

    def _apply_user_defaults(self):
        """Lazily apply user defaults from the config file.

        Called once before the first kernel dispatch. Reads the user config
        and sets defaults for any platforms that have a user preference and
        a matching registered backend.
        """
        if self._user_defaults_applied:
            return
        self._user_defaults_applied = True
        from brainevent._config import load_user_defaults
        prim_defaults = load_user_defaults().get(self.name, {})
        for plat, backend in prim_defaults.items():
            if plat in self._kernels and backend in self._kernels[plat]:
                self._defaults[plat] = backend

    def available_backends(self, platform: str) -> List[str]:
        """Return list of registered backend names for a platform.

        Args:
            platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').

        Returns:
            A list of backend names registered for the given platform.
        """
        if platform not in self._kernels:
            return []
        return list(self._kernels[platform].keys())

    def benchmark(
        self,
        *args,
        platform: str,
        backend: Optional[str] = None,
        n_warmup: int = 5,
        n_runs: int = 20,
        batch_mode: bool = False,
        compare_results: bool = True,
        **kwargs
    ) -> Union[BenchmarkResult, BenchmarkReport]:
        """Benchmark kernel execution using the registered call function.

        Args:
            *args: Positional args passed to the call function.
            platform: Target platform ('cpu', 'gpu', 'tpu').
            backend: Specific backend to benchmark, or None for all.
            n_warmup: Number of warmup runs.
            n_runs: Number of timed runs.
            batch_mode: If False (default), block after each function call and time
                each run individually (measures per-call latency). If True, run all
                n_runs calls first, then block once at the end and measure total time
                (measures throughput, useful for async GPU/TPU execution).
            compare_results: Verify outputs match across backends (not yet implemented).
            **kwargs: Keyword args passed to the call function.

        Returns:
            BenchmarkResult if backend specified, else BenchmarkReport.

        Raises:
            ValueError: If no call function registered (use def_call first).
        """
        if self._call_fn is None:
            raise ValueError(
                f"No call function registered for '{self.name}'. "
                "Use def_call() to register one before benchmarking."
            )

        # Get backends to benchmark
        if backend is not None:
            backends_to_test = [backend]
        else:
            backends_to_test = self.available_backends(platform)

        if not backends_to_test:
            raise ValueError(
                f"No backends registered for platform '{platform}' in primitive '{self.name}'."
            )

        results = []
        outputs = {}  # backend -> output for comparison
        for be in backends_to_test:
            try:
                # Create a function that calls the primitive with the specified backend
                def run_fn(be=be):
                    return self._call_fn(*args, backend=be, **kwargs)

                mean_time, std_time, min_time, max_time, output = benchmark_function(
                    run_fn, n_warmup, n_runs, batch_mode=batch_mode
                )

                results.append(
                    BenchmarkResult(
                        backend=be,
                        platform=platform,
                        mean_time=mean_time,
                        std_time=std_time,
                        min_time=min_time,
                        max_time=max_time,
                        n_runs=n_runs,
                        success=True,
                        error=None,
                    )
                )
                outputs[be] = output
            except Exception as e:
                results.append(
                    BenchmarkResult(
                        backend=be,
                        platform=platform,
                        mean_time=0.0,
                        std_time=0.0,
                        min_time=0.0,
                        max_time=0.0,
                        n_runs=0,
                        success=False,
                        error=str(e),
                    )
                )

        # Compare results across backends if requested
        mismatches = []
        if compare_results and len(outputs) > 1:
            import jax.numpy as jnp
            backends_list = list(outputs.keys())
            ref_backend = backends_list[0]
            ref_output = outputs[ref_backend]

            for other_backend in backends_list[1:]:
                other_output = outputs[other_backend]
                try:
                    # Handle tuple/list outputs
                    if isinstance(ref_output, (list, tuple)):
                        for i, (r, o) in enumerate(zip(ref_output, other_output)):
                            if not jnp.allclose(r, o, rtol=1e-5, atol=1e-5):
                                mismatches.append(f"{ref_backend} vs {other_backend}: output[{i}] mismatch")
                    else:
                        if not jnp.allclose(ref_output, other_output, rtol=1e-5, atol=1e-5):
                            mismatches.append(f"{ref_backend} vs {other_backend}: output mismatch")
                except Exception as e:
                    mismatches.append(f"{ref_backend} vs {other_backend}: comparison error: {e}")

        # Return single result if specific backend was requested
        if backend is not None:
            return results[0]

        return BenchmarkReport(
            primitive_name=self.name,
            platform=platform,
            results=results,
            mismatches=mismatches,
        )
