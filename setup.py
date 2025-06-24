import os
import re

from setuptools import find_packages, setup


def _strip_comments(line):
    return line.split("#", 1)[0].strip()


def _pip_requirement(req):
    if req.startswith("-r "):
        _, path = req.split()
        return reqs(*path.split("/"))
    return [req]


def _reqs(*f):
    return [
        _pip_requirement(req)
        for req in (_strip_comments(line) for line in open(os.path.join(os.getcwd(), "requirements", *f)).readlines())
        if req
    ]


def reqs(*f):
    """Parse requirement file.
    Example:
        reqs('default.txt')          # requirements/default.txt
        reqs('extras', 'redis.txt')  # requirements/extras/redis.txt
    Returns:
        List[str]: list of requirements specified in the file.
    """
    return [req for subreq in _reqs(*f) for req in subreq]


def install_requires():
    """Get list of requirements required for installation."""
    return reqs("requirements.txt")


with open("README.md", "r") as readme:
    long_description = readme.read()


with open("deepopt/__init__.py", "r") as version_file:
    pattern = re.compile(r'__version__ = "\d+(\.\d+){2}"')
    version_line = pattern.search(version_file.read())[0]
    version = version_line.split(" ")[-1].replace('"', "")


setup(
    name="deepopt",
    author="DeepOpt Dev Team",
    author_email="deepopt@llnl.gov",
    classifiers=[        
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Easy-to-use library for Bayesian optimization, with support for neural network surrogates.",
    long_description=long_description,
    url="https://github.com/LLNL/deepopt",
    license="MIT",
    install_requires=install_requires(),
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "deepopt = deepopt.deepopt_cli:main",
        ]
    },
    version=version,
)
