# Installation

## Installing From PyPI

!!! note

    Before installing from PyPI, it's recommended that you use a virtual environment. To see instructions on how to set one up see [Setting Up Your Virtual Environment](#setting-up-your-environment).

[//]: <> (TODO: add a link to the PyPI page here when it's released instead of the link to the pypi home page)
The DeepOpt library is released as a [PyPI package](https://pypi.org/) which makes it easy to install via [pip](https://pip.pypa.io/en/stable/):

```bash
pip install deepopt
```

To verify that the DeepOpt library installed properly, run:

```bash
deepopt -h
```

## Installing From GitHub

!!! note

    Before installing from GitHub, it's recommended that you use a virtual environment. To see instructions on how to set one up see [Setting Up Your Virtual Environment](#setting-up-your-environment).

[//]: <> (TODO: add a link to the GitHub DeepOpt repo once we release it there)
To install from GitHub, you'll first need to clone the DeepOpt repository:

[//]: <> (TODO: this might need to be modified depending on what the actual repo link is)
```bash
git clone https://github.com/LLNL/deepopt.git
```

Once cloned, enter the repo with:

```bash
cd deepopt
```

From here, install all of the requirements for the DeepOpt library using:

```bash
pip install -r requirements/requirements.txt
```

Then install DeepOpt by running:

```bash
pip install -e .
```

To verify that the DeepOpt library installed properly, run:

```bash
deepopt -h
```

## Setting Up Your Environment

It is highly recommended to set up a virtual environment to ensure a clean and isolated environment for your project. Virtual environments allow you to manage and isolate project dependencies, preventing potential conflicts with other projects and system-wide packages.

By using virtual environments, you can create an isolated space where you can install specific versions of packages without affecting the system-wide Python installation. This approach enables you to work on multiple projects simultaneously, each with its own set of dependencies.

We'll talk about two ways to set up virtual environments here: [Python Virtualenvs](#python-virtualenvs) and [Conda Environments](#conda-environments).

### Python Virtualenvs

Install Python by following the instructions [here](https://www.python.org/downloads/). After following the instructions on Python's website, you can verify that Python was installed by running:

```bash
python3 --help
```

Once installed, create a python virtual environment using:

```bash
python3 -m venv deepopt_venv
```

Now that the environment has been created, activate this environment with:

```bash
source deepopt_venv/bin/activate
```

This environment will automatically have [pip](https://pip.pypa.io/en/stable/) installed. To check that you have access to pip you can run:

```bash
which pip
```

If you have access to pip, you should see: `/path/to/deepopt_venv/bin/pip`.

To deactivate this environment when you're done using it:

```bash
deactivate
```

### Conda Environments

Install Conda by following the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). After following the instructions on Conda's website you can verify that Conda was installed by running:

```bash
conda --help
```

Once installed, create a Conda virtual environment with [pip](https://pip.pypa.io/en/stable/) installed:

```bash
conda create --name deepopt_venv pip
```

Now that we have a Conda environment, let's activate it:

```bash
conda activate deepopt_venv
```

To ensure pip was installed in the environment use:

```bash
which pip
```

If installed correctly this should show: `/path/to/deepopt_venv/bin/pip`.

To disable this environment when you're done using it:

```bash
conda deactivate
```