# DeepOpt Documentation Guide

The documentation for the DeepOpt library uses [MkDocs](https://www.mkdocs.org/).

## Prerequisites

Before you run the docs:
1. Ensure you're at the top-level `deepopt_release/` directory
2. Install the necessary libraries for the documentation by running `pip install -r docs/requirements.txt`.

## How to Run

To get a live view of your docs that updates on each save, run `mkdocs serve` from the `deepopt_release/` directory. Once `mkdocs serve` has been run, you can view the docs by pressing `ctrl+click` on the url provided by MkDocs (this url will likely look something like https://127.0.0.1:800x).

## API Reference Guide

The API Reference Guide will be updated each time a new module, function, or class is added to the codebase. To create accurate documentation for these new additions you *must* add module headers, docstrings, and class descriptions to newly added code.

If you're updating the API Reference Guide you'll have to stop and restart `mkdocs serve` each time you make a change. This happens since our API docs are auto-generated upon docs initiliazation.
