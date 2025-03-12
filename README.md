# Enabling Event-driven Computation in Brain Dynamics

[//]: # (<p align="center">)

[//]: # (  	<img alt="Header image of brainevent." src="https://github.com/chaobrain/brainevent/blob/main/docs/_static/brainevent.png" width=50%>)

[//]: # (</p> )



<p align="center">
	<a href="https://pypi.org/project/brainevent/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainevent"></a>
	<a href="https://github.com/chaobrain/brainevent/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  	<a href='https://brainevent.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/brainevent/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://badge.fury.io/py/brainevent"><img alt="PyPI version" src="https://badge.fury.io/py/brainevent.svg"></a>
    <a href="https://github.com/chaobrain/brainevent/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainevent/actions/workflows/CI.yml/badge.svg"></a>
    <a href="https://pepy.tech/projects/brainevent"><img src="https://static.pepy.tech/badge/brainevent" alt="PyPI Downloads"></a>
</p>




Brain is characterized by the discrete spiking events, which are the fundamental units of computation in the brain.

`BrainEvent` provides a set of data structures and algorithms for such event-driven computation, which can be used to
model the brain dynamics in a more efficient and biologically plausible way.

Particularly, it provides the following class to represent binary events in the brain:

- ``EventArray``: representing array with a vector/matrix of events.

Furthermore, it implements the following commonly used data structures for event-driven computation
of the above class:

- ``COO``: a sparse matrix in COO format for sparse and event-driven computation.
- ``CSR``: a sparse matrix in CSR format for sparse and event-driven computation.
- ``CSC``: a sparse matrix in CSC format for sparse and event-driven computation.
- ``BlockCSR``: a block sparse matrix in CSR format for sparse and event-driven computation.
- ``BlockELL``: a block sparse matrix in ELL format for sparse and event-driven computation.
- ``JITC_CSR``: a just-in-time connectivity sparse matrix in CSR format for sparse and event-driven computation.
- ``JITC_CSC``: a just-in-time connectivity sparse matrix in CSC format for sparse and event-driven computation.
- ``FixedPreNumConn``: a fixed number of pre-synaptic connections for sparse and event-driven computation.
- ``FixedPostNumConn``: a fixed number of post-synaptic connections for sparse and event-driven computation.
- ...


`BrainEvent` is fully compatible with physical units and unit-aware computations provided in [BrainUnit](https://github.com/chaobrain/brainunit).


## Installation

You can install ``brainevent`` via pip:

```bash
pip install brainevent --upgrade
```

## Documentation

The official documentation is hosted on Read the
Docs: [https://brainevent.readthedocs.io/](https://brainevent.readthedocs.io/)

## See also the brain modeling ecosystem

We are building the Brain Modeling ecosystem: https://brainmodeling.readthedocs.io/

