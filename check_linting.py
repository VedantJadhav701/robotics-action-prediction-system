#!/usr/bin/env python3
"""Check remaining linting errors and show summary"""

import importlib.util
import subprocess
import sys


def run_flake8_check():
    """Run flake8 on src directory and report"""
    # Check if flake8 is available
    if importlib.util.find_spec("flake8") is None:
        print("Installing flake8...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flake8", "-q"])

    print("Running flake8 check on src/...\n")
    result = subprocess.run(
        [sys.executable, "-m", "flake8", "src/", "--max-line-length=79"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
        print("\n⚠️  Found linting issues above")
        return False
    else:
        print("✅ No linting errors found!")
        return True


if __name__ == "__main__":
    success = run_flake8_check()
    sys.exit(0 if success else 1)
