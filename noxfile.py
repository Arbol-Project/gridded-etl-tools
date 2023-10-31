import json
import pathlib
import subprocess

import nox

# numba is not supported on Python 3.12
ALL_INTERPRETERS = (
    "3.10",
    "3.11",
)
CODE = "gridded_etl_tools"
DEFAULT_INTERPRETER = "3.10"
HERE = pathlib.Path(__file__).parent


@nox.session(py=ALL_INTERPRETERS)
def unit(session):
    session.install("-e", ".[testing]")
    session.run(
        "pytest",
        f"--cov={CODE}",
        "--cov=tests.unit",
        "--cov-append",
        "--cov-config",
        HERE / ".coveragerc",
        "--cov-report=term-missing",
        "tests/unit",
    )


@nox.session(py=DEFAULT_INTERPRETER)
def cover(session):
    session.install("coverage")
    session.run("coverage", "report", "--fail-under=100", "--show-missing")
    session.run("coverage", "erase")


@nox.session(py=DEFAULT_INTERPRETER)
def lint(session):
    session.install("black", "flake8")
    run_black(session, check=True)
    session.run("flake8", CODE, "tests")


@nox.session(py=DEFAULT_INTERPRETER)
def blacken(session):
    # Install all dependencies.
    session.install("black")
    run_black(session)


@nox.session(py=DEFAULT_INTERPRETER)
def system(session):
    if not check_kubo():
        session.skip("No IPFS server running")

    session.install("-e", ".[testing]")
    session.run(
        "pytest",
        "--cov=tests.system",
        "--cov-config",
        HERE / ".coveragerc",
        "--cov-report=term-missing",
        "tests/system",
    )


def run_black(session, check=False):
    args = ["black"]
    if check:
        args.append("--check")
    args.extend(["noxfile.py", CODE, "tests"])
    session.run(*args)


def check_kubo():
    """Check whether Kubo IPFS server is running locally.

    Required for System tests.
    """
    result = subprocess.run(["ipfs", "id"], capture_output=True, check=True)
    result = json.loads(result.stdout)
    return bool(result["Addresses"])
