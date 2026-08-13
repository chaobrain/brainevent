---
orphan: true
---

# Tutorials Data/Events Restructure Specification

**Status:** Approved design

**Date:** 2026-08-13

**Branch:** `docs/tutorials-data-events-restructure`

## Objective

Restructure the BrainEvent learning documentation so that its navigation and
tutorial responsibilities reflect the package model:

- **Data** describes connectivity representations and their storage or
  generation semantics.
- **Events** describes binary event representations and event-driven
  operations.
- **Custom operators** describes extension mechanisms and remains unchanged.

The restructure must also separate introductory material from the current
oversized `Getting Started with BrainEvent` notebook without duplicating the
existing Quickstart.

## Current State

The Tutorials navigation currently contains two groups:

1. `Data structures & operators`
2. `Custom operators`

The first group combines five notebooks with different responsibilities:

| Current source | Current responsibility |
| --- | --- |
| `docs/tutorials/data-structures/01_eventarray_basics.ipynb` | Installation, event representation, event operations, examples, time series, and visualization |
| `docs/tutorials/data-structures/02_sparse_matrices.ipynb` | CSR and CSC data representations |
| `docs/tutorials/data-structures/03_jit_connectivity.ipynb` | Generated random connectivity data |
| `docs/tutorials/data-structures/04_fixed_connections.ipynb` | Fixed fan-in and fan-out connectivity data |
| `docs/tutorials/data-structures/05_synaptic_plasticity.ipynb` | Event-triggered synaptic updates |

This organization obscures the Data/Events boundary. It also gives the first
notebook too many teaching objectives and places synaptic plasticity under a
data-structure heading.

## Information Architecture

The target navigation is:

```text
Getting Started
├── Installation
├── Quickstart
└── Getting Started with BrainEvent

Tutorials
├── Data
│   ├── CSR and CSC Sparse Matrices
│   ├── Fixed Connection Count Structures
│   └── Just-in-Time Connection Matrices
├── Events
│   ├── Binary Events and Event-Driven Operations
│   └── Event-Driven Synaptic Plasticity
└── Custom operators
    ├── Custom CPU Operators with Numba
    ├── Custom GPU Operators with Numba CUDA
    ├── Custom GPU Operators with Warp
    ├── Custom C++ (CPU) Kernels
    └── Custom CUDA (GPU) Kernels
```

### Category definitions

**Data** covers structures that represent, store, or generate connectivity.
This includes CSR, CSC, fixed-connection-count structures, and JIT-generated
connectivity.

**Events** covers representations of discrete activity and the operations
whose execution is driven by active events. This includes `BinaryArray`,
event-driven matrix multiplication, time-series spike processing, and
event-triggered synaptic updates.

The tutorial label **Events** is reader-facing. It does not claim that every
operation in the package belongs to an `Events` implementation layer. Regular
array operations may appear only as correctness or performance baselines.

### Data tutorial order

The Data sequence is:

1. CSR and CSC sparse matrices
2. Fixed connection count structures
3. Just-in-time connection matrices

This order moves from explicit general sparse storage, to explicit constrained
storage, to generated connectivity that does not materialize the full matrix.

## Tutorial Numbering Policy

Tutorial titles must not use global `Tutorial 1`, `Tutorial 2`, and similar
prefixes.

Descriptive titles and `toctree` order define the learning path. This avoids a
false global sequence after the material is divided among Getting Started,
Data, and Events, and prevents future additions from forcing broad
renumbering.

For this restructure, existing Data notebook filenames `02_sparse_matrices`,
`03_jit_connectivity`, and `04_fixed_connections` remain unchanged to limit
unnecessary URL churn. Their page titles lose the numeric prefixes. New files
use descriptive, unnumbered names.

## Complete Content Migration

### Current Tutorial 1: Getting Started with BrainEvent

Source:
`docs/tutorials/data-structures/01_eventarray_basics.ipynb`

The source is split into two documents.

#### Target: Getting Started with BrainEvent

Target:
`docs/getting-started/getting-started-with-brainevent.ipynb`

| Current content | Target heading or action |
| --- | --- |
| Opening introduction | Rewrite around BrainEvent's Data/Events model |
| Section 1, `What is BinaryArray?` | `What BrainEvent Computes` |
| `Core Advantages` | `Why Event-Driven Computation?`; qualify performance claims |
| Section 2, `Installation and Import` | `Import BrainEvent`; link to Installation rather than repeat installation instructions |
| Section 8, `Visualizing Spike Patterns` | `Visualize Your First Spike Pattern` |
| Section 9, `Summary` | Rewrite to summarize only the new Getting Started document |
| `Next steps` | Link separately to Data, Events, and Custom operators |
| `References` | Preserve relevant references and repair link text |

This document introduces the conceptual split and produces one visible result.
It must not repeat the Quickstart's survey of CSR, JITC, and fixed-count
connectivity.

#### Target: Binary Events and Event-Driven Operations

Target:
`docs/tutorials/events/binary-events.ipynb`

| Current content | Target heading or action |
| --- | --- |
| Section 3, `Creating BinaryArray` | `Creating Binary Events` |
| Section 3.1 | `Creating Events from Array-Like Inputs` |
| Section 3.2 | `Representing Simulated Spikes` |
| Section 4 | `Inspecting and Transforming Binary Events` |
| Section 4.1 | `Indexing` |
| Section 4.2 | `Reductions and Logical Operations` |
| Section 5 | `Event-Driven Matrix Multiplication` |
| Section 5.1 | `Binary Events with Dense Data` |
| Section 5.2 | `Correctness and Performance Comparison` |
| Section 6 | `A Small Event-Driven Feedforward Network` |
| Section 7 | `Processing Time-Series Events` |
| Relevant items from the old summary | Rewrite as `Summary and Next Steps` |

Section 7 belongs to Events because it processes temporal spike events.
Section 8 belongs to Getting Started because it provides an immediate,
reader-visible introduction. Therefore the approved boundary is:

- Getting Started: old sections 1-2 and 8, plus a rewritten section 9.
- Events: old sections 3-7, plus a new summary.

### Current Tutorial 2: Sparse Data Structures - CSR and CSC

Source retained:
`docs/tutorials/data-structures/02_sparse_matrices.ipynb`

New page title: `CSR and CSC Sparse Matrices`

| Current heading | Target heading |
| --- | --- |
| `Why do we need sparse matrices?` | `Why Use Sparse Connectivity Data?` |
| `Coordinate triplets and the compressed formats` | `COO Input, CSR Storage, and CSC Storage` |
| `Creating and using sparse matrices` | `Constructing CSR and CSC Data` |
| `Using with BinaryArray` | `Combining Sparse Data with Binary Events` |
| `Performance comparison` | `Memory, Correctness, and Performance` |
| `Practice: Building a sparse connection neural network` | `Build a Sparse Event-Driven Network` |
| `Visualizing sparse connection structures` | `Inspect the Connectivity Structure` |
| `Format selection guide` | `Choosing CSR or CSC` |
| `Summary` and `Next Steps` | `Summary and Next Steps` |

CSR and CSC remain together as one complete chapter.

### Current Tutorial 3: JIT Connection Matrices

Source retained:
`docs/tutorials/data-structures/03_jit_connectivity.ipynb`

New page title: `Just-in-Time Connection Matrices`

| Current heading | Target heading |
| --- | --- |
| `Core concepts of JIT connections` | `Why Generate Connections Just in Time?` |
| `JITCHomoR - Homogeneous weight connections` | `Homogeneous-Weight JIT Connectivity` |
| `JITCNormalR - Normal distribution weights` | `Normally Distributed JIT Connectivity` |
| `JITCUniformR - Uniform distribution weights` | `Uniformly Distributed JIT Connectivity` |
| `Memory and performance comparison` | `Memory and Performance Trade-offs` |
| `Practice: Ultra-large-scale spiking neural network` | `Build a Large Random Network` |
| `Advanced techniques: Row-oriented vs Column-oriented` | `Row- and Column-Oriented Connectivity` |
| `Summary` and `Next Steps` | `Summary and Next Steps` |

All prose, code, and stored output must use the current `JITCScalarR` and
`JITCScalarC` API names instead of deprecated `JITCHomoR` and `JITCHomoC`
names.

### Current Tutorial 4: Fixed Connection Count Structures

Source retained:
`docs/tutorials/data-structures/04_fixed_connections.ipynb`

New page title: `Fixed Connection Count Structures`

| Current heading | Target heading |
| --- | --- |
| `Biological significance of fixed connection counts` | `Why Fix the Number of Connections?` |
| `FixedNumPerPre - Fixed output connection count` | `Fixed Fan-Out with FixedNumPerPre` |
| `FixedNumPerPost - Fixed input connection count` | `Fixed Fan-In with FixedNumPerPost` |
| `Using with BinaryArray` | `Combining Fixed Connectivity with Binary Events` |
| `Practice: Biologically realistic cortical network` | `Build a Fixed-Degree Network` |
| `Performance and memory analysis` | `Memory and Performance Characteristics` |
| `Usage recommendations` | `Choosing Fan-In or Fan-Out Constraints` |
| `Summary` and `Next Steps` | `Summary and Next Steps` |

The title drops `Biologically Realistic Network Topology`. Fixed-degree
connectivity can express one biological constraint, but does not by itself
establish that a complete network topology is biologically realistic.

### Current Tutorial 5: Synaptic Plasticity Modeling

Source:
`docs/tutorials/data-structures/05_synaptic_plasticity.ipynb`

Target:
`docs/tutorials/events/synaptic-plasticity.ipynb`

New page title: `Event-Driven Synaptic Plasticity`

| Current heading | Target heading or action |
| --- | --- |
| `Biological Background of Synaptic Plasticity` | `From Spike Events to Synaptic Updates` |
| `Hebb's Rule` | Retain as concise background, not a general neuroscience review |
| `STDP (Spike-Timing-Dependent Plasticity)` | `Spike-Timing-Dependent Plasticity` |
| `Mathematical Expression` | `The Update Rule` |
| `Implementation in BrainEvent` | `BrainEvent Update Operations` |
| `Basic STDP Implementation` | `Implement a Minimal STDP Rule` |
| `Visualize STDP Window` | `Visualize the Learning Window` |
| `Implementing STDP on CSR Format` | `Update CSR Weights from Pre- and Postsynaptic Events` |
| `Practice: Building a Self-Learning Neural Network` | `Apply Event-Driven Updates in a Network` |
| `Summary` | `Summary and Next Steps` |

The hand-written Contents block must be regenerated from the actual section
structure. It must not advertise dense STDP, advanced rules, or other sections
that are absent from the document.

## File Operations

### Create

- `docs/getting-started/getting-started-with-brainevent.ipynb`
- `docs/tutorials/events/index.rst`
- `docs/tutorials/events/binary-events.ipynb`
- `docs/tutorials/events/synaptic-plasticity.ipynb`

### Modify

- `docs/index.rst`
- `docs/getting-started/quickstart.rst`
- `docs/tutorials/data-structures/index.rst`
- `docs/tutorials/data-structures/02_sparse_matrices.ipynb`
- `docs/tutorials/data-structures/03_jit_connectivity.ipynb`
- `docs/tutorials/data-structures/04_fixed_connections.ipynb`
- `docs/how-to/data-structures/index.rst`
- `docs/how-to/data-structures/choosing-a-sparse-format.rst`
- `docs/how-to/data-structures/synaptic-plasticity.rst`
- `docs/reference/apis/index.rst`

### Remove after replacements are complete

- `docs/tutorials/data-structures/01_eventarray_basics.ipynb`
- `docs/tutorials/data-structures/05_synaptic_plasticity.ipynb`

The implementation must search the entire `docs/` tree for old paths before
removing either source notebook.

### Unchanged

The following group remains unchanged in content, order, and filenames:

- `docs/tutorials/custom-operators/index.rst`
- `docs/tutorials/custom-operators/01_numba.ipynb`
- `docs/tutorials/custom-operators/02_numba_cuda.ipynb`
- `docs/tutorials/custom-operators/03_warp.ipynb`
- `docs/tutorials/custom-operators/04_cpp.ipynb`
- `docs/tutorials/custom-operators/05_cuda.ipynb`

## Cross-Reference Migration

At minimum, the implementation must update old tutorial paths consumed by:

- `docs/index.rst`
- `docs/getting-started/quickstart.rst`
- `docs/how-to/data-structures/index.rst`
- `docs/how-to/data-structures/choosing-a-sparse-format.rst`
- `docs/how-to/data-structures/synaptic-plasticity.rst`
- `docs/reference/apis/index.rst`

The implementation must perform a final repository-wide search for
`/tutorials/data-structures/01_eventarray_basics`,
`/tutorials/data-structures/05_synaptic_plasticity`, and the retired
`Data structures & operators` navigation label.

## Content Quality Constraints

1. Getting Started must not duplicate the Quickstart's connectivity-format
   survey.
2. Data tutorials may demonstrate `BinaryArray`, but their teaching objective
   remains the data representation.
3. Events tutorials may use dense or sparse data, but their teaching objective
   remains event representation or event-driven behavior.
4. Performance claims must distinguish correctness, compilation/warm-up time,
   steady-state execution, memory use, and hardware dependence.
5. Benchmarks must warm up compiled functions and synchronize device work
   before timing.
6. Repeated model or time-step execution must not use a bare Python `for` or
   `while` loop. Use the appropriate compiled transformation available to the
   project.
7. Notebook imports, prose, code, and stored outputs must agree with the
   supported public API.
8. Notebook summaries and hand-written contents must describe only material
   actually present in the notebook.
9. Claims of biological realism must be scoped to the represented constraint;
   they must not infer whole-network realism from fixed degree alone.

## Verification and Acceptance Criteria

The restructure is complete only when all of the following hold:

1. `docs/index.rst` exposes three Tutorial groups: Data, Events, and Custom
   operators.
2. Getting Started precedes Tutorials and includes the new introductory
   notebook.
3. No tutorial page title contains a global `Tutorial N` prefix.
4. The Data and Events landing pages define their responsibilities and learning
   order.
5. CSR/CSC remains one complete Data chapter.
6. Fixed connection count and JIT connectivity each remain one complete Data
   chapter.
7. Binary event fundamentals and synaptic plasticity appear under Events.
8. Custom operators has no content, order, or filename changes.
9. No internal link targets a removed notebook path.
10. Notebook JSON parses successfully.
11. Every modified notebook executes from a clean kernel in document order.
12. The strict Sphinx documentation build completes without broken references,
    duplicate labels, orphan documents, or new warnings.
13. A repository-wide search finds no stale global tutorial numbering in the
    migrated five notebooks.
14. A repository-wide search finds no deprecated `JITCHomoR` or `JITCHomoC`
    usage in the migrated JIT tutorial.
15. The Git worktree remains on `docs/tutorials-data-events-restructure`, not
    `main`, for every modification and commit.

## Out of Scope

- Changes to BrainEvent runtime APIs or kernels.
- Rewriting the Custom operators tutorials.
- Adding new connectivity structures beyond the five existing tutorial topics.
- Turning the Tutorials section into API reference documentation.
- Broad rewrites of How-to, Explanation, or Reference pages beyond link and
  navigation consistency required by this restructure.
