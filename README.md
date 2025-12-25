# Enabling Event-driven Computation in Brain Dynamics

<p align="center">
  	<img alt="Header image of brainevent." src="https://raw.githubusercontent.com/chaobrain/brainevent/main/docs/_static/brainevent.png" width=50%>
</p> 

<p align="center">
	<a href="https://pypi.org/project/brainevent/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainevent"></a>
	<a href="https://github.com/chaobrain/brainevent/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  	<a href='https://brainevent.readthedocs.io/en/latest/?badge=latest'>
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

Particularly, it provides the following class to represent binary events in the brain:

- ``EventArray``: representing array with a vector/matrix of events.

Furthermore, it implements the following commonly used data structures for event-driven computation
of the above class:

- ``COO``: a sparse matrix in COO format for sparse and event-driven computation.
- ``CSR``: a sparse matrix in CSR format for sparse and event-driven computation.
- ``CSC``: a sparse matrix in CSC format for sparse and event-driven computation.
- ``JITCHomoR``: a just-in-time connectivity matrix with homogenous weight for sparse and event-driven computation.
- ``JITCNormalR``: a just-in-time connectivity matrix with normal distribution weight for sparse and event-driven
  computation.
- ``JITCUniformR``: a just-in-time connectivity matrix with uniform distribution weight for sparse and event-driven
  computation.
- ``FixedPreNumConn``: a fixed number of pre-synaptic connections for sparse and event-driven computation.
- ``FixedPostNumConn``: a fixed number of post-synaptic connections for sparse and event-driven computation.
- ...

`BrainEvent` is fully compatible with physical units and unit-aware computations provided
in [BrainUnit](https://github.com/chaobrain/brainunit).

## Usage

If you want to take advantage of event-driven computations, you must warp your data with ``brainevent.EventArray``:

```python
import brainevent

# wrap your array with EventArray
event_array = brainevent.EventArray(your_array)
```

Then, the matrix multiplication with the following data structures, $\mathrm{event\ array} @ \mathrm{data}$,
will take advantage of event-driven computations:

- Sparse data structures provided by ``brainevent``, like:
    - ``brainevent.CSR``
    - ``brainevent.JITCHomoR``
    - ``brainevent.FixedPostNumConn``
    - ...
- Dense data structures provided by JAX/NumPy, like:
    - ``jax.numpy.ndarray``
    - ``numpy.ndarray``


```python
data = jax.random.rand(...)  # normal dense array
data = brainevent.CSR(...)  # CSR structure
data = brainevent.JITCHomoR(...)  # JIT connectivity
data = brainevent.FixedPostNumConn(...)  # fixed number of post-synaptic connections

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


## Documentation

The official documentation is hosted on Read the Docs: [https://brainevent.readthedocs.io/](https://brainevent.readthedocs.io/)


## See also the ecosystem

``brainevent`` is one part of our brain modeling ecosystem: https://brainmodeling.readthedocs.io/

