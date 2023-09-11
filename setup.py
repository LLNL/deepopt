import re
import os
from setuptools import find_packages, setup


def _strip_comments(l):
    return l.split('#', 1)[0].strip()


def _pip_requirement(req):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*path.split('/'))
    return [req]


def _reqs(*f):
    return [
        _pip_requirement(r) for r in (
            _strip_comments(l) for l in open(
                os.path.join(os.getcwd(), 'requirements', *f)).readlines()
        ) if r]


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
    author="Jayaraman Thiagarajan",
    author_email="jayaramanthi1@llnl.gov",
    classifiers=[
        "Programming Language :: Python3.8"
    ],
    description="A design optimization framework based on deep neural network surrogates.",
    long_description=long_description,
    url="https://lc.llnl.gov/gitlab/idesign/deepopt",
    install_requires=install_requires(),
    packages=find_packages(),
    # package_data={
    #     "deepopt": [
    #         "data/*",
    #         "examples/*",
    #     ],
    # },
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'deepopt-c = deepopt.deepopt_cli:main',
            # 'deepopt-learner = deepopt.bayes_opt_example.scripts.learner:main',
            # 'deepopt-optimize = deepopt.bayes_opt_example.scripts.get_candidates:main',
            # 'deepopt-learner-mf = deepopt.multi_fidelity_bayes_opt_example.scripts.learner_mf:main',
            # 'deepopt-optimize-mf = deepopt.multi_fidelity_bayes_opt_example.scripts.get_candidates_mf:main',
        ]
    },
    version=version,
)
