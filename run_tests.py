"""
run_tests.py
------------
Master test runner. Discovers and executes all tests under tests/ in isolation.
Production logs and models are never touched (tests use temp paths).

Usage
-----
    python run_tests.py
"""

import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Exit with non-zero code on failure (useful for CI pipelines)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
