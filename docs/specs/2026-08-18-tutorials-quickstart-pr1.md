# Tutorial Information Architecture and Quickstart Merge

**Status:** Approved for implementation

**Date:** 2026-08-18

## Objective

Prepare a tutorial-only pull request that reorganizes BrainEvent documentation
around Data and Events while exposing one unambiguous introductory path:

```text
Getting Started
├── Installation
└── Quickstart
```

The display title is `Quickstart`; the stable Sphinx document name and public
URL remain `getting-started/quickstart` and
`getting-started/quickstart.html`.

## Scope

The pull request includes the tutorial information architecture, the Data and
Events notebook migration, the Quickstart merge, and the static tests that lock
the navigation and selected notebook-execution scope.

It excludes the Brette 2007 COBA example, local-preview HTML rewriting,
generated static assets, and HTML build output.

## Quickstart Source and Content

Replace both existing introductory sources:

- `docs/getting-started/quickstart.rst`
- `docs/getting-started/getting-started-with-brainevent.ipynb`

with the single executable source:

- `docs/getting-started/quickstart.ipynb`

The notebook teaches, in order:

1. BrainEvent's Data and Events model.
2. What `BinaryArray` represents and why event-driven kernels can save work.
3. Importing BrainEvent and reporting the active JAX backend.
4. Creating and visualizing a binary spike train.
5. Performing one dense event-driven matrix multiplication and checking it
   against the ordinary JAX result.
6. Using the operation inside `jax.jit`.
7. Continuing separately to Data, Events, and Custom operators.

CSR/CSC, fixed-count, and JIT connectivity remain dedicated Data chapters.
Quickstart links to them instead of duplicating their construction examples.

## Navigation and Link Compatibility

The hidden Getting Started toctree contains only Installation followed by
Quickstart. All active references to
`getting-started/getting-started-with-brainevent` change to
`getting-started/quickstart`.

Because the document name `getting-started/quickstart` is preserved while its
source suffix changes from `.rst` to `.ipynb`, existing generated HTML links do
not require a redirect.

Global `Tutorial N` numbering remains retired. Data, Events, and Custom
operators remain the three tutorial groups, and Custom operators content is not
changed.

## Notebook Execution

Keep `nb_execution_mode = "off"` globally and opt only these notebooks into
MyST-NB execution:

- `getting-started/quickstart.ipynb`
- `tutorials/data-structures/02_sparse_matrices.ipynb`
- `tutorials/data-structures/03_jit_connectivity.ipynb`
- `tutorials/data-structures/04_fixed_connections.ipynb`
- `tutorials/events/binary-events.ipynb`
- `tutorials/events/synaptic-plasticity.ipynb`

Each uses:

```json
"mystnb": {
  "execution_mode": "force",
  "execution_timeout": 120
}
```

All other notebooks, including Custom operators, remain non-executing.

## Acceptance Criteria

1. Installation and Quickstart are the only Getting Started navigation entries.
2. `quickstart.ipynb` is the only Quickstart/Getting Started lesson source.
3. The Quickstart docname and HTML URL remain stable.
4. Quickstart contains a visible spike plot, a dense event-driven multiplication,
   a numerical equivalence check, and a `jax.jit` example.
5. Data, Events, and Custom operators links are local Sphinx document links.
6. Exactly the six approved notebooks declare forced build-time execution.
7. Tutorial structure tests, execution-scope tests, notebook validation, and
   Ruff pass without building HTML or executing notebook cells.
8. No push, PR creation, or HTML build is performed during implementation.

