# -*- coding: utf-8 -*-
"""Minimal pytest-free runner for V5-0a tests.

Discovers test_* functions, runs them, reports pass/fail.
Handles pytest.raises via a thin compatibility shim.
"""

import sys
from pathlib import Path
import traceback
import importlib.util

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"

sys.path.insert(0, str(SRC))


# Thin pytest compatibility shim
class _RaisesContext:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.exc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            name = (
                self.expected.__name__
                if not isinstance(self.expected, tuple)
                else " or ".join(e.__name__ for e in self.expected)
            )
            raise AssertionError(f"Expected {name}, no exception raised")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None:
            import re
            if not re.search(self.match, str(exc_val)):
                raise AssertionError(
                    f"Exception {exc_type.__name__}({exc_val}) does not match '{self.match}'"
                )
        self.exc = exc_val
        return True


class _PytestShim:
    @staticmethod
    def raises(expected, match=None):
        return _RaisesContext(expected, match=match)

    @staticmethod
    def main(args):
        # No-op: we drive ourselves
        pass

    @staticmethod
    def approx(expected, rel=None, abs=None):
        return _ApproxValue(expected, rel=rel, abs_=abs)

    @staticmethod
    def skip(reason=""):
        raise _Skip(reason)


class _Skip(Exception):
    pass


class _ApproxValue:
    """Minimal pytest.approx replacement for ==/!= comparison."""

    def __init__(self, expected, rel=None, abs_=None):
        self.expected = expected
        self.rel = rel if rel is not None else 1e-6
        self.abs = abs_ if abs_ is not None else 1e-12

    def __eq__(self, other):
        try:
            import numpy as _np
            return bool(_np.isclose(other, self.expected, rtol=self.rel, atol=self.abs))
        except Exception:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected})"


sys.modules["pytest"] = _PytestShim()


def run_test_file(path: Path) -> tuple[int, int, list[str]]:
    """Run all test_* functions in a file. Return (n_pass, n_fail, failures)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n_pass = 0
    n_fail = 0
    failures = []

    for name in dir(mod):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        try:
            fn()
            n_pass += 1
            print(f"  PASS  {path.stem}::{name}")
        except _Skip as e:
            print(f"  SKIP  {path.stem}::{name}: {e}")
        except Exception as e:
            n_fail += 1
            tb = traceback.format_exc()
            failures.append(f"{path.stem}::{name}\n{tb}")
            print(f"  FAIL  {path.stem}::{name}: {type(e).__name__}: {e}")

    return n_pass, n_fail, failures


def main():
    test_files = sorted(TESTS.glob("test_*.py"))
    if len(sys.argv) > 1:
        # Filter by name
        wanted = sys.argv[1]
        test_files = [p for p in test_files if wanted in p.stem]

    total_pass = 0
    total_fail = 0
    all_failures = []

    for tf in test_files:
        print(f"\n=== {tf.name} ===")
        n_pass, n_fail, fails = run_test_file(tf)
        total_pass += n_pass
        total_fail += n_fail
        all_failures.extend(fails)

    print(f"\n=== Summary ===")
    print(f"  Total: {total_pass + total_fail}")
    print(f"  Pass:  {total_pass}")
    print(f"  Fail:  {total_fail}")

    if all_failures:
        print(f"\n=== Failure details ===")
        for f in all_failures:
            print(f"\n--- {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
