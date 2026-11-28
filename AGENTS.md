# AGENTS.md


## Working agreement

Follow the repository's [CONTRIBUTING.md](CONTRIBUTING.md) when making changes.

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must happen on the worktree branch, not main. 
7. Design notes, task specs and the roadmap are kept by the maintainers
   outside this repository. Do not cite their paths in code, tests or docs
   here: name a task by its ID and state the decision you rely on in the text
   itself (pre-commit `no-private-references` checks).
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. Typing tests should pass mypy with no errors, but prioritize meaningful type annotations that improve readability and maintainability over exhaustive coverage of every variable.
10. Support jax version >=0.9.0
11. Keep the `foo_test.py` suffix style (never the `test_*.py` prefix). For `brainevent/_csr`, `brainevent/_tcsr`, `brainevent/_sddmm`, `brainevent/_dense`, `brainevent/_fcn`, `brainevent/_jit_normal`, `brainevent/_jit_scalar`, and `brainevent/_jit_uniform`, place each functional area's tests in its `test/` subdirectory; root-level tests belong in that package's `test/` directory. Other packages keep tests as siblings of the code under test.

## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Rules for the Examples section

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

