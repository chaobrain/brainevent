# Tutorial Information Architecture and Quickstart Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a tutorial-only change set with Installation and one executable Quickstart entry, while retaining Data, Events, and Custom operators as the tutorial groups.

**Architecture:** Preserve the stable `getting-started/quickstart` document name by replacing its RST source with a notebook, then migrate the useful introductory cells from the retired Getting Started notebook. Lock the information architecture and exact notebook execution whitelist with static tests.

**Tech Stack:** Sphinx, MyST-NB, Jupyter Notebook v4, pytest, nbformat, Ruff.

## Global Constraints

- Do not push or create a pull request.
- Do not build HTML or execute notebook cells.
- Do not modify or remove COBA, local-preview, generated-static, or unrelated user changes.
- Display the page as `Quickstart`; retain the `getting-started/quickstart` docname.
- Keep global notebook execution off and force only Quickstart, Data, and Events notebooks.

---

### Task 1: Lock the final information architecture

**Files:**
- Modify: `docs/tutorials/tutorial_structure_test.py`

- [ ] Require `quickstart.ipynb` and reject both superseded introductory sources.
- [ ] Require Installation followed by Quickstart in the Getting Started toctree.
- [ ] Require the Quickstart title, teaching sections, and local next-step targets.
- [ ] Run the focused test and confirm it fails because the source migration has not occurred.

### Task 2: Lock the selected execution scope

**Files:**
- Create: `docs/getting-started/quickstart_execution_test.py`

- [ ] Require `quickstart.ipynb` and the five Data/Events notebooks to use the exact execution metadata.
- [ ] Run the focused test and confirm it fails because Quickstart is not yet a notebook.

### Task 3: Merge the introductory sources

**Files:**
- Create: `docs/getting-started/quickstart.ipynb`
- Delete: `docs/getting-started/quickstart.rst`
- Delete: `docs/getting-started/getting-started-with-brainevent.ipynb`

- [ ] Create the approved notebook structure and portable CPU-compatible code cells.
- [ ] Preserve one dense multiplication, numerical check, spike visualization, and `jax.jit` example.
- [ ] Add the approved MyST-NB force-execution metadata.

### Task 4: Repair navigation and active links

**Files:**
- Modify: `docs/index.rst`
- Modify: `docs/tutorials/events/index.rst`

- [ ] Remove the retired Getting Started entry from the top-level toctree.
- [ ] Point Events to Quickstart.
- [ ] Scan active documentation for retired links.

### Task 5: Complete the execution whitelist

**Files:**
- Modify: `docs/tutorials/data-structures/02_sparse_matrices.ipynb`
- Modify: `docs/tutorials/data-structures/03_jit_connectivity.ipynb`
- Modify: `docs/tutorials/data-structures/04_fixed_connections.ipynb`
- Modify: `docs/tutorials/events/binary-events.ipynb`
- Modify: `docs/tutorials/events/synaptic-plasticity.ipynb`

- [ ] Ensure every approved notebook has the exact force/120 metadata.
- [ ] Ensure no Custom operators notebook declares an executing mode.

### Task 6: Verify without push, PR, notebook execution, or HTML

**Files:**
- Test: `docs/tutorials/tutorial_structure_test.py`
- Test: `docs/getting-started/quickstart_execution_test.py`

- [ ] Run both static pytest files.
- [ ] Run Ruff on both Python test files.
- [ ] Validate all source notebooks with nbformat without executing cells.
- [ ] Scan active sources for the retired document path and globally numbered tutorial titles.
- [ ] Report changed files, edge cases, and user-run HTML checks without pushing or creating a PR.

