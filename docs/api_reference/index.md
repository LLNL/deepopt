# DeepOpt API Reference

DeepOpt's detailed API reference is generated at documentation-build time from source docstrings. The MkDocs build runs `docs/gen_ref_pages.py`, which creates one reference page per Python module, then `mkdocstrings-python` renders the public classes, functions, methods, and parameters from those docstrings.

Use the generated API pages when you need exact call signatures or return types. Use the User Guide when you need workflow examples and compatibility notes.

Important modules:

- `deepopt.deepopt_cli` — Click entrypoint, `learn` and `optimize` commands, CLI parsing contracts.
- `deepopt.configuration` and `deepopt.defaults` — model configuration, defaults, and optimization profiles.
- `deepopt.models` — high-level wrappers, checkpoint loading, candidate generation, constraints, risk methods, and GP model helpers.
- `deepopt.acquisition` — MaxValEntropy and GIBBON acquisition implementations.
- `deepopt.input_scaling` and `deepopt.output_scaling` — input/output scaling contracts and checkpoint state.
- `deepopt.deltaenc`, `deepopt.nn_ensemble`, and `deepopt.surrogate_utils` — neural surrogate models and MLP utilities.

Because these pages are generated from docstrings, public behavior that users call from Python should be documented in the corresponding source docstring as well as, when appropriate, in the narrative guide pages.
