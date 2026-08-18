---
orphan: true
---

# Tutorials Data/Events Restructure Implementation Plan

> **Superseded for the introductory path:** The final Quickstart merge and
> execution scope are defined in
> `2026-08-18-tutorials-quickstart-pr1.md`. This plan remains as the historical
> record for the Data/Events migration only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize BrainEvent tutorials into unnumbered Data, Events, and unchanged Custom operators learning paths, with a separate Getting Started notebook.

**Architecture:** Preserve the three existing Data notebook URLs while changing their titles and order, split the old event-array notebook into Getting Started and Events documents, and move synaptic plasticity into Events. Add a structural regression test that treats navigation, titles, retired paths, and Custom operators immutability as contracts; validate executable notebooks separately from the Sphinx build because `nb_execution_mode` is `off`.

**Tech Stack:** reStructuredText/Sphinx, MyST-NB/Jupyter notebooks, Python 3.11, `nbformat`, `nbclient`, pytest.

## Global Constraints

- Work only on `docs/tutorials-data-events-restructure`, never `main`.
- Tutorial page titles must not use global `Tutorial N` prefixes.
- Tutorials navigation must contain Data, Events, and Custom operators.
- Keep CSR/CSC, fixed-count, and JIT connectivity as separate Data chapters in that order.
- Keep Custom operators content, order, and filenames unchanged.
- Keep `02_sparse_matrices.ipynb`, `03_jit_connectivity.ipynb`, and `04_fixed_connections.ipynb` paths stable.
- Remove the old `01_eventarray_basics.ipynb` and `05_synaptic_plasticity.ipynb` only after replacements and references exist.
- Use current public names `JITCScalarR` and `JITCScalarC`, not `JITCHomoR` or `JITCHomoC`.
- Benchmarks must warm up and synchronize device work.
- Repeated time-step work must not use a bare Python `for` or `while` loop.
- Modified notebooks must parse, execute from a clean kernel, and render in a strict Sphinx build.

---

### Task 1: Encode the documentation structure contract

**Files:**
- Create: `docs/tutorials/tutorial_structure_test.py`
- Read: `docs/specs/2026-08-13-tutorials-data-events-restructure.md`

**Interfaces:**
- Consumes: repository-root-relative documentation paths.
- Produces: pytest checks for navigation, unnumbered titles, retired references, notebook JSON validity, and unchanged Custom operators SHA-256 hashes recorded from commit `7a4d5fc`.

- [ ] **Step 1: Add failing structural tests**

Create tests that assert the exact target paths and toctree order, reject `^# Tutorial [0-9]+:`, reject retired paths and `JITCHomoR/C`, parse every migrated notebook with `nbformat`, and compare every file under `docs/tutorials/custom-operators` with `git show 7a4d5fc:<path>`.

- [ ] **Step 2: Run the tests to verify the pre-migration failure**

Run:

```powershell
python -m pytest docs/tutorials/tutorial_structure_test.py -q
```

Expected: failures for missing Getting Started/Events files, old navigation, and numbered titles; Custom operators integrity passes.

- [ ] **Step 3: Commit the contract test**

```powershell
git add docs/tutorials/tutorial_structure_test.py
git commit -m "test(docs): define tutorial restructure contract"
```

### Task 2: Build the navigation skeleton and migrate cross-references

**Files:**
- Modify: `docs/index.rst`
- Modify: `docs/getting-started/quickstart.rst`
- Modify: `docs/tutorials/data-structures/index.rst`
- Create: `docs/tutorials/events/index.rst`
- Modify: `docs/how-to/data-structures/index.rst`
- Modify: `docs/how-to/data-structures/choosing-a-sparse-format.rst`
- Modify: `docs/how-to/data-structures/synaptic-plasticity.rst`
- Modify: `docs/reference/apis/index.rst`

**Interfaces:**
- Consumes: target document paths from the approved spec.
- Produces: Sphinx toctrees and links for Getting Started, Data, Events, and Custom operators.

- [ ] **Step 1: Add the target toctrees**

Add `getting-started/getting-started-with-brainevent` under Getting Started. Replace the old Tutorials entry with Data, Events, and Custom operators entries. Rename the Data landing page and order it as `02_sparse_matrices`, `04_fixed_connections`, `03_jit_connectivity`.

- [ ] **Step 2: Define Events and Data responsibilities**

Write concise landing-page prose matching the category definitions in the spec. Events lists `binary-events` followed by `synaptic-plasticity`.

- [ ] **Step 3: Update every known old-path consumer**

Point general tutorial links to the relevant Data or Events landing page and point synaptic-plasticity links to `/tutorials/events/synaptic-plasticity`.

- [ ] **Step 4: Run the structural test**

```powershell
python -m pytest docs/tutorials/tutorial_structure_test.py -q
```

Expected: navigation assertions pass; missing notebook and numbered-title assertions still fail.

- [ ] **Step 5: Commit navigation**

```powershell
git add docs/index.rst docs/getting-started/quickstart.rst docs/tutorials/data-structures/index.rst docs/tutorials/events/index.rst docs/how-to/data-structures/index.rst docs/how-to/data-structures/choosing-a-sparse-format.rst docs/how-to/data-structures/synaptic-plasticity.rst docs/reference/apis/index.rst
git commit -m "docs: split tutorials navigation into data and events"
```

### Task 3: Split event fundamentals and move synaptic plasticity

**Files:**
- Create: `docs/getting-started/getting-started-with-brainevent.ipynb`
- Create: `docs/tutorials/events/binary-events.ipynb`
- Create: `docs/tutorials/events/synaptic-plasticity.ipynb`
- Remove: `docs/tutorials/data-structures/01_eventarray_basics.ipynb`
- Remove: `docs/tutorials/data-structures/05_synaptic_plasticity.ipynb`

**Interfaces:**
- Consumes: cells and examples from the two old source notebooks.
- Produces: three valid notebooks with unnumbered descriptive headings and current APIs.

- [ ] **Step 1: Create Getting Started**

Build a short notebook containing the Data/Events mental model, import/version check, one reproducible spike-pattern example, one raster visualization, a scoped summary, and links to Data, Events, and Custom operators. Link to Installation instead of duplicating installation commands; do not repeat Quickstart's format survey.

- [ ] **Step 2: Create Binary Events and Event-Driven Operations**

Migrate old sections 3-7 with the approved headings. Replace the old time-step Python loop with a batched `BinaryArray` operation or an appropriate compiled transform. Correctness comparisons must use `jax.block_until_ready`; any timing comparison must include warm-up and synchronized repetitions.

- [ ] **Step 3: Create Event-Driven Synaptic Plasticity**

Migrate the actual old sections 1-5, regenerate the hand-written contents from those sections, and remove claims about absent dense STDP or advanced-rule sections. Keep biological claims scoped and use current `update_csr_on_binary_pre/post` operations.

- [ ] **Step 4: Validate the three notebooks**

Run:

```powershell
python -c "import nbformat; from pathlib import Path; paths=[Path('docs/getting-started/getting-started-with-brainevent.ipynb'),Path('docs/tutorials/events/binary-events.ipynb'),Path('docs/tutorials/events/synaptic-plasticity.ipynb')]; [nbformat.validate(nbformat.read(p, as_version=4)) for p in paths]; print('validated', len(paths))"
```

Expected: `validated 3`.

- [ ] **Step 5: Remove the two retired notebooks and run structural tests**

```powershell
python -m pytest docs/tutorials/tutorial_structure_test.py -q
```

Expected: event migration and retired-path checks pass; Data title checks remain failing until Task 4.

- [ ] **Step 6: Commit event migration**

```powershell
git add docs/getting-started/getting-started-with-brainevent.ipynb docs/tutorials/events/binary-events.ipynb docs/tutorials/events/synaptic-plasticity.ipynb docs/tutorials/data-structures/01_eventarray_basics.ipynb docs/tutorials/data-structures/05_synaptic_plasticity.ipynb
git commit -m "docs: separate getting started and event tutorials"
```

### Task 4: Retitle and harden the three Data notebooks

**Files:**
- Modify: `docs/tutorials/data-structures/02_sparse_matrices.ipynb`
- Modify: `docs/tutorials/data-structures/03_jit_connectivity.ipynb`
- Modify: `docs/tutorials/data-structures/04_fixed_connections.ipynb`

**Interfaces:**
- Consumes: existing runnable notebook examples and approved heading map.
- Produces: unnumbered, responsibility-focused Data chapters with current names and qualified claims.

- [ ] **Step 1: Update CSR/CSC headings and claims**

Apply the complete heading map from the spec, keep CSR and CSC in one chapter, and distinguish storage, correctness, memory, and steady-state timing claims.

- [ ] **Step 2: Update fixed-count headings and claims**

Apply the heading map, remove whole-topology biological-realism claims, and describe fixed fan-in/fan-out as explicit structural constraints.

- [ ] **Step 3: Update JIT headings and public API names**

Apply the heading map and replace `JITCHomoR/C` in prose, code, and stored outputs with `JITCScalarR/C`.

- [ ] **Step 4: Validate notebook JSON and run structural tests**

```powershell
python -m pytest docs/tutorials/tutorial_structure_test.py -q
```

Expected: all structural tests pass.

- [ ] **Step 5: Commit Data tutorials**

```powershell
git add docs/tutorials/data-structures/02_sparse_matrices.ipynb docs/tutorials/data-structures/03_jit_connectivity.ipynb docs/tutorials/data-structures/04_fixed_connections.ipynb
git commit -m "docs: align data tutorials with connectivity model"
```

### Task 5: Execute notebooks and build documentation strictly

**Files:**
- Verify: all modified `.ipynb` files
- Verify: complete `docs/` tree

**Interfaces:**
- Consumes: final migrated documentation.
- Produces: execution and Sphinx-build evidence.

- [ ] **Step 1: Execute each modified notebook in a clean kernel**

Use `nbclient.NotebookClient(timeout=600, kernel_name='python3')` for the six modified notebooks. Execute copies in a temporary directory so stored outputs in source files are not rewritten merely by verification.

Expected: all six complete without cell exceptions. If hardware-specific examples cannot run on CPU, revise them to use a documented CPU-safe scale or mark only genuinely optional hardware cells with MyST-compatible skip metadata.

- [ ] **Step 2: Run the structural test suite**

```powershell
python -m pytest docs/tutorials/tutorial_structure_test.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Search for retired labels, paths, and names**

```powershell
rg -n "^# Tutorial [0-9]+:|Data structures & operators|01_eventarray_basics|05_synaptic_plasticity|JITCHomo[RC]" docs
```

Expected: no matches except historical text in `docs/specs/` and `docs/superpowers/plans/`; no matches in active `.rst` or `.ipynb` documentation.

- [ ] **Step 4: Run a strict Sphinx build**

```powershell
$env:BRAINX_HEADER_TTL='0'; sphinx-build -W --keep-going -b html docs docs/_build/html
```

Expected: exit code 0 and `docs/_build/html/index.html` exists.

- [ ] **Step 5: Verify branch, diff scope, and Custom operators immutability**

```powershell
git branch --show-current
git status --short
git diff 7a4d5fc -- docs/tutorials/custom-operators
```

Expected: branch is `docs/tutorials-data-events-restructure`; Custom operators diff is empty.

- [ ] **Step 6: Commit verification-driven corrections**

```powershell
git add docs
git commit -m "docs: verify restructured tutorial learning paths"
```

Skip this commit if verification required no corrections and the worktree is clean.
