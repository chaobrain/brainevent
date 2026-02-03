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

# -*- coding: utf-8 -*-

import functools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from jax.interpreters import xla, batching, ad, mlir

from brainevent._compatible_import import Primitive
from brainevent._error import KernelNotAvailableError, KernelCompilationError, KernelFallbackExhaustedError
from brainevent._typing import KernelGenerator
from .util import general_batching_rule, defjvp, OutType, abstract_arguments

__all__ = [
    'XLACustomKernel',
    'KernelEntry',
    'DEFAULT_PRIORITIES',
]

# Default priority for each (backend, platform) combination.
# Lower values = higher priority (tried first).
DEFAULT_PRIORITIES: Dict[tuple, int] = {

    # CPU
    ('numba', 'cpu'): 100,
    ('tvmffi', 'cpu'): 200,

    # GPU
    ('pallas', 'gpu'): 100,
    ('tvmffi', 'gpu'): 150,
    ('numba_cuda', 'gpu'): 200,
    ('warp', 'gpu'): 250,
    ('triton', 'gpu'): 300,

    # TPU
    ('pallas', 'tpu'): 100,
}


@dataclass
class KernelEntry:
    """Represents a registered kernel for a specific backend and platform.

    Attributes:
        backend: The backend name (e.g., 'numba', 'warp', 'pallas', 'triton').
        platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
        kernel_generator: A callable that generates the kernel function.
        priority: Execution priority. Lower values are tried first.
    """
    backend: str
    platform: str
    kernel_generator: KernelGenerator
    priority: int = field(default=0)

    def __post_init__(self):
        if self.priority == 0:
            self.priority = DEFAULT_PRIORITIES.get((self.backend, self.platform), 500)


class XLACustomKernel:
    """Creates and manages a custom JAX primitive for XLA custom calls.

    This class provides a high-level interface to define custom operations
    that can be executed efficiently on different backends (CPU, GPU, TPU)
    via XLA custom calls. It handles the registration of the JAX primitive,
    its abstract evaluation rule, backend-specific kernel implementations
    (using Numba for CPU, Pallas or Warp for GPU/TPU), and JAX transformation
    rules like batching, JVP (forward-mode AD), and transpose (reverse-mode AD).

    The core idea is to define the computation logic once for each relevant
    backend using specialized kernel generators (:class:`KernelGenerator`,
    :class:`KernelGenerator`, :class:`KernelGenerator`) and then use this class
    to bind everything together into a callable JAX operation.

    Attributes:
        primitive: The underlying JAX primitive created.
        name: The name assigned to the primitive.

    Args:
        name: The unique name for the custom JAX primitive.

    """

    __module__ = 'brainevent'

    def __init__(self, name: str, fallback_enabled: bool = True):
        # primitive
        self.name = name
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_impl(functools.partial(xla.apply_primitive, self.primitive))
        self.primitive.def_abstract_eval(self._abstract_eval)

        # batching rule
        self.register_general_batching()

        # kernel storage: platform -> list of KernelEntry (sorted by priority)
        self._kernels: Dict[str, List[KernelEntry]] = {}
        # tracks which platforms have had lowering registered
        self._registered_platforms: set = set()

        # setting
        self.fallback_enabled = fallback_enabled

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

    def __call__(self, *ins, outs: OutType, backend: Optional[str] = None, **kwargs):
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
        # Pass backend hint to the lowering via kwargs if specified
        bind_kwargs = dict(kwargs)
        if backend is not None:
            bind_kwargs['_preferred_backend'] = backend
        r = self.primitive.bind(*ins, **bind_kwargs, outs=tuple(outs))
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def def_kernel(
        self,
        backend: str,
        platform: str,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        """Register a kernel for a specific backend and platform.

        Args:
            backend: The backend name (e.g., 'numba', 'warp', 'pallas').
            platform: The platform name (e.g., 'cpu', 'gpu', 'tpu').
            kg: A kernel generator callable that creates the kernel function.
            priority: Priority of this kernel. Lower values are tried first.
                If not specified, uses DEFAULT_PRIORITIES or 500 as fallback.
            priority: Optional priority override. Lower values are tried first.
                If not specified, uses DEFAULT_PRIORITIES or 500 as fallback.
        """
        assert isinstance(backend, str), f'The `backend` should be a string, but got {type(backend)}.'
        assert isinstance(platform, str), f'The `platform` should be a string, but got {type(platform)}.'
        assert callable(kg), f'The `kg` should be a callable, but got {type(kg)}.'

        # Create kernel entry
        entry = KernelEntry(
            backend=backend,
            platform=platform,
            kernel_generator=kg,
            priority=priority if priority is not None else 0  # 0 triggers __post_init__ default
        )
        if priority is not None:
            entry.priority = priority  # Override post_init default

        # Store kernel in the platform's list
        if platform not in self._kernels:
            self._kernels[platform] = []
        self._kernels[platform].append(entry)
        # Sort by priority (lower = higher priority)
        self._kernels[platform].sort(key=lambda e: e.priority)

        # Register fallback lowering once per platform
        if platform not in self._registered_platforms:
            self._register_fallback_lowering(platform)
            self._registered_platforms.add(platform)

    def _register_fallback_lowering(self, platform: str):
        """Register a lowering function that implements the fallback mechanism.

        This creates a lowering function that will try kernels in priority order,
        catching expected errors and falling back to the next kernel.

        Args:
            platform: The platform to register the lowering for.
        """
        primitive = self.primitive
        kernels_dict = self._kernels
        name = self.name

        def fallback_kernel_fn(*args, **kwargs):
            # Extract preferred backend hint if provided
            preferred_backend = kwargs.pop('_preferred_backend', None)

            # Get kernels for this platform
            kernels = kernels_dict.get(platform, [])
            if not kernels:
                raise KernelFallbackExhaustedError(
                    f"No kernels registered for platform '{platform}' in primitive '{name}'."
                )

            # Reorder kernels if a preferred backend is specified
            if preferred_backend is not None:
                # Put preferred backend kernels first, maintaining priority within groups
                preferred = [k for k in kernels if k.backend == preferred_backend]
                others = [k for k in kernels if k.backend != preferred_backend]
                kernels = preferred + others

            # Check if fallback is enabled
            errors = []
            for entry in kernels:
                try:
                    kernel = entry.kernel_generator(**kwargs)
                    return kernel(*args)
                except (ImportError, ModuleNotFoundError) as e:
                    errors.append((entry.backend, type(e).__name__, str(e)))
                    if not self.fallback_enabled:
                        raise
                    continue
                except KernelNotAvailableError as e:
                    errors.append((entry.backend, type(e).__name__, str(e)))
                    if not self.fallback_enabled:
                        raise
                    continue
                except KernelCompilationError as e:
                    errors.append((entry.backend, type(e).__name__, str(e)))
                    if not self.fallback_enabled:
                        raise
                    continue
                except Exception as e:
                    errors.append((entry.backend, type(e).__name__, str(e)))
                    if not self.fallback_enabled:
                        raise
                    continue

            # All kernels failed
            error_details = "\n".join(
                f"  - {backend} ({err_type}): {msg}"
                for backend, err_type, msg in errors
            )
            raise KernelFallbackExhaustedError(
                f"All kernels failed for platform '{platform}' in primitive '{name}'.\n"
                f"Attempted kernels (in order):\n{error_details}"
            )

        # Register the lowering with JAX
        lower = mlir.lower_fun(fallback_kernel_fn, multiple_results=True)
        mlir.register_lowering(primitive, lower, platform=platform)

    def def_numba_kernel(
        self,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        self.def_kernel(backend='numba', platform='cpu', kg=kg, priority=priority)

    def def_warp_kernel(
        self,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        self.def_kernel(backend='warp', platform='gpu', kg=kg, priority=priority)

    def def_triton_kernel(
        self,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        self.def_kernel(backend='triton', platform='gpu', kg=kg, priority=priority)

    def def_pallas_kernel(
        self,
        platform: str,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        assert platform in ['gpu', 'tpu'], f'The `platform` should be either `gpu` or `tpu`, but got {platform}.'
        self.def_kernel(backend='pallas', platform=platform, kg=kg, priority=priority)

    def def_tvmffi_kernel(
        self,
        platform: str,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        assert platform in ['cpu', 'gpu'], f'The `platform` should be either `cpu` or `gpu`, but got {platform}.'
        self.def_kernel(backend='tvmffi', platform=platform, kg=kg, priority=priority)

    def def_numba_cuda_kernel(
        self,
        kg: KernelGenerator,
        priority: Optional[int] = None
    ):
        """Register a Numba CUDA kernel for GPU platform.

        Args:
            kg: A kernel generator callable that creates the Numba CUDA kernel function.
            priority: Priority of this kernel. Lower values are tried first.
                If not specified, uses DEFAULT_PRIORITIES (150 for numba_cuda on gpu).
        """
        self.def_kernel(backend='numba_cuda', platform='gpu', kg=kg, priority=priority)

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
