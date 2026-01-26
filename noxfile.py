import pathlib
import platform

import nox

# numba is not supported on Python 3.12
ALL_INTERPRETERS = "3.12"


CODE = "gridded_etl_tools"
DEFAULT_INTERPRETER = "3.12"
HERE = pathlib.Path(__file__).parent


def _get_local_constraints(session):
    """Return constraint args to skip problematic packages on local macOS builds."""
    if platform.system() == "Darwin":
        # numba/llvmlite require specific LLVM versions that are hard to match locally.
        # Exclude them on macOS - they're optional dependencies via xarray[complete].
        return [
            "--constraint",
            f"{HERE}/constraints-local.txt",
        ]
    return []


@nox.session(py=ALL_INTERPRETERS)
def unit(session):
    session.install(*_get_local_constraints(session), "-e", ".[testing]")
    session.run(
        "pytest",
        "--log-level=INFO",
        "--log-cli-level=INFO",
        f"--cov={CODE}",
        "--cov=tests.unit",
        "--cov-append",
        "--cov-config",
        HERE / ".coveragerc",
        "--cov-report=term-missing",
        "tests/unit",
    )


@nox.session(py=DEFAULT_INTERPRETER)
def lint(session):
    session.install("black", "flake8", "flake8-pyproject")
    run_black(session, check=True)
    session.run("flake8", CODE, "tests", "examples")


@nox.session(py=DEFAULT_INTERPRETER)
def blacken(session):
    # Install all dependencies.
    session.install("black")
    run_black(session)


@nox.session(py=DEFAULT_INTERPRETER)
def system(session):
    session.install(*_get_local_constraints(session), "-e", ".[testing]")
    session.run(
        "pytest",
        "--log-level=INFO",
        "--log-cli-level=INFO",
        "--cov=tests.system",
        "--cov-config",
        HERE / ".coveragerc",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        "tests/system",
    )


@nox.session(py=DEFAULT_INTERPRETER)
def cover(session):
    session.install("coverage")
    session.run("coverage", "report", "--fail-under=100", "--show-missing")
    session.run("coverage", "erase")


def run_black(session, check=False):
    args = ["black"]
    if check:
        args.append("--check")
    args.extend(["noxfile.py", CODE, "tests", "examples"])
    session.run(*args)
