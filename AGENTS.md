# AGENTS.md


## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must happen on the worktree branch, not main. 
7. Write a spec only when it will actually guide the implementation: the work
   spans several modules or repos, changes a public API or the architecture,
   has real design choices to settle first, or will take multiple sessions.
   Skip it for bug fixes, small features, local refactors, and docs edits.
   A spec goes in `docs/specs/YYYY-MM-DD-<slug>.md`, one file (no separate
   -spec/-plan pairs), and stays short: goal, decisions with reasons, step
   plan, how to verify. Specs are dated records: don't rewrite old ones to
   match later changes.
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. Typing tests should pass mypy with no errors, but prioritize meaningful type annotations that improve readability and maintainability over exhaustive coverage of every variable.
10. Support jax version >=0.9.0
11. Co-locate tests with the code under test: each module `foo.py` has its tests in a sibling `foo_test.py` (suffix style — never a separate `tests/` directory, never the `test_*.py` prefix). 

## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Rules for the Examples section

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

