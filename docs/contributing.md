---
hide:
  - navigation
---

# Contributing

Contributing to DeepOpt is as easy as following the steps below:

1. Create a fork of the DeepOpt repository

2. Clone your forked repository to your local system

3. Create a virtual environment with DeepOpt installed and activate it:

    ```bash
    python3 -m venv venv_deepopt
    source venv_deepopt/bin/activate # or activate.csh
    ```

4. Move into the local clone of the forked repository:

    ```bash
    cd deepopt
    ```

5. Install *all* requirements for the DeepOpt library:

    ```bash
    pip install -r requirements/requirements.txt
    pip install -r requirements/dev.txt
    pip install -r docs/requirements.txt
    ```

6. Install DeepOpt in an editable mode:

    ```bash
    pip install -e .
    ```

7. Initialize the [pre-commit](https://pre-commit.com/) library:

    ```bash
    pre-commit install
    ```

8. Create a new branch and get going:

    ```bash
    git checkout develop
    git switch -c <branch prefix>/<your branch name>
    ```

See the sections below for additional information on the contributing process.

## Branch Naming Guide

DeepOpt branches *must* start with one of the following prefixes:

- `feature/`: for new features
- `bugfix/`: for bugfixes
- `hotfix/`: for bugfixes that need to go directly into the `main` branch
- `refactor/`: for any sort of code refactoring
- `docs/`: for any large-scale changes to the documentation

## Best Practices

When contributing, please conform to the common docstring format used throughout the codebase. This helps keep the API docs up-to-date and the codebase syntactically clean.

Linters will be ran after each commit, ensure there are no errors with the linters before you create a merge request.

## Making Commits

The [pre-commit](https://pre-commit.com/) library will be run on each commit you make. For all checks except flake8, changes will be made for you. If flake8 fails it will be your job to find the issue and fix it.

If _any_ lint failures occur, you will have to run `git add ...` for the files that were changed and then re-run `git commit ...`.

## Merge Requests

When you believe you're ready for a merge request, create the merge from your forked branch into the `develop` branch of the DeepOpt repository. Approval will be required by one of the maintainers.