# Enabling Event-driven Computation in CPU/GPU/TPU

<p align="center">
  	<img alt="Header image of brainevent." src="https://brainx.chaobrain.com/images/brainevent.webp" width=50%>
</p> 

<p align="center">
	<a href="https://pypi.org/project/brainevent/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainevent"></a>
	<a href="https://github.com/chaobrain/brainevent/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  	<a href='https://brainx.chaobrain.com/brainevent/'>
        <img src='https://readthedocs.org/projects/brainevent/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://badge.fury.io/py/brainevent"><img alt="PyPI version" src="https://badge.fury.io/py/brainevent.svg"></a>
    <a href="https://github.com/chaobrain/brainevent/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainevent/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://github.com/chaobrain/brainevent/actions/workflows/CI-daily.yml"><img alt="Daily CI Tests" src="https://github.com/chaobrain/brainevent/actions/workflows/CI-daily.yml/badge.svg"></a>
    <a href="https://pepy.tech/projects/brainevent"><img src="https://static.pepy.tech/badge/brainevent" alt="PyPI Downloads"></a>
    <a href="https://doi.org/10.5281/zenodo.15324450"><img src="https://zenodo.org/badge/921610544.svg" alt="DOI"></a>
</p>


Brain is characterized by the discrete spiking events, which are the fundamental units of computation in the brain.

`BrainEvent` provides a set of data structures and algorithms for such event-driven computation on
**CPUs**, **GPUs**, **TPUs**, and maybe more, which can be used to model the brain dynamics in an
efficient and biologically plausible way.

Particularly, it provides the following classes to represent binary (spiking) events in the brain:

- ``BinaryArray``: an array wrapping a vector/matrix of binary events (spikes).
- ``BitPackedBinary``: a memory-efficient representation that packs binary events into bits
  (see also the ``bitpack`` helper).
- ``CompactBinary``: a compact representation that stores only the indices of the active (non-zero) events.

Furthermore, it implements the following commonly used data structures for event-driven computation
of the above classes. Most structures come in row-oriented (``R``) and column-oriented (``C``) variants:

- ``CSR`` / ``CSC``: sparse matrices in CSR / CSC format for sparse and event-driven computation.
- ``JITCScalarR`` / ``JITCScalarC``: a just-in-time connectivity matrix with homogeneous (scalar)
  weight for sparse and event-driven computation.
- ``JITCNormalR`` / ``JITCNormalC``: a just-in-time connectivity matrix with normal-distribution
  weights for sparse and event-driven computation.
- ``JITCUniformR`` / ``JITCUniformC``: a just-in-time connectivity matrix with uniform-distribution
  weights for sparse and event-driven computation.
- ``FixedNumConn`` / ``FixedNumPerPre`` / ``FixedNumPerPost``: fixed-number connectivity matrices,
  where each neuron has a fixed number of synaptic connections.
- ...

`BrainEvent` is fully compatible with physical units and unit-aware computations provided
in [BrainUnit](https://github.com/chaobrain/brainunit).

## Usage

If you want to take advantage of event-driven computations, you must warp your data with ``brainevent.BinaryArray``:

```python
import brainevent

# wrap your array with BinaryArray
event_array = brainevent.BinaryArray(your_array)
```

Then, the matrix multiplication with the following data structures, $\mathrm{event\ array} @ \mathrm{data}$,
will take advantage of event-driven computations:

- Sparse data structures provided by ``brainevent``, like:
    - ``brainevent.CSR``
    - ``brainevent.JITCScalarR``
    - ``brainevent.FixedNumPerPre``
    - ...
- Dense data structures provided by JAX/NumPy, like:
    - ``jax.numpy.ndarray``
    - ``numpy.ndarray``

```python
data = jax.random.rand(...)  # normal dense array
data = brainevent.CSR(...)  # CSR structure
data = brainevent.JITCScalarR(...)  # JIT connectivity
data = brainevent.FixedNumPerPre(...)  # fixed number of post-synaptic connections per pre-neuron

# event-driven matrix multiplication
r = event_array @ data
r = data @ event_array
```

## Installation

You can install ``brainevent`` via pip:

```bash
pip install brainevent -U
```

Alternatively, you can install `BrainX`, which bundles `brainevent` with other compatible packages for a comprehensive brain modeling ecosystem:

```bash
pip install BrainX -U
```

### GPU compile dependencies

The first time a kernel runs on a GPU, `brainevent` compiles its CUDA source on the fly. This needs three things:

1. **NVIDIA driver** (provides `libcuda` and `nvidia-smi`) — a system-level requirement for any approach.
2. **`jax[cuda12]` or `jax[cuda13]`** — installing it pulls in the `nvidia-*` pip packages, which already
   bundle `nvcc`/`ptxas`/CUDA runtime/headers. **A separate system CUDA Toolkit is therefore not required.**
3. **A host C++ compiler (`g++`/`clang++`)** — pip does not provide one. Install it via
   `conda install -c conda-forge gxx`, `sudo apt-get install g++`, or `sudo dnf install gcc-c++`.

Optional configuration:

- `brainevent.config.prefer_system_nvcc()` — prefer the system `PATH` nvcc instead of the pip-bundled one (pip is the default).
- Environment variables: `BRAINEVENT_NVCC_PREFER=pip|system`, `BRAINEVENT_NVCC_PATH`, `CUDA_HOME`, `CXX`.
- `BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1` — force compilation when the host gcc is newer than nvcc supports.
- `BRAINEVENT_COMPUTE_CAPABILITIES=8.6,8.0` — skip `nvidia-smi` auto-detection.
- `BRAINEVENT_TOOLCHAIN_DEBUG=1` — append a "toolchain snapshot" to every toolchain error for easier debugging.


## Documentation

The official documentation is hosted on Read the Docs: [https://brainx.chaobrain.com/brainevent/](https://brainx.chaobrain.com/brainevent/)


## See also the ecosystem

``brainevent`` is one part of our brain modeling ecosystem: https://brainx.chaobrain.com/

