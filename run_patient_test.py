#!/usr/bin/env python3
"""
Simple script to run Patient_test_1 test following KISS principle.

Usage:
    python run_patient_test.py

This script:
1. Sets the required environment variable
2. Runs the Patient_test_1 test with proper timeout
3. Reports results clearly

Follows KISS principle: One simple task, clear execution.
"""

import os
import subprocess
import sys

def main():
    """Run Patient_test_1 test with proper environment setup."""

    print("🧬 Running Patient_test_1 Large File Test")
    print("=========================================")
    print()
    print("Configuration:")
    print("- Tile size: 2000×2000px (19.1MB each)")
    print("- Overlap: 400px (20%)")
    print("- Expected: 15,332 cells in ~35 minutes")
    print()

    # Set environment variable to enable the test
    env = os.environ.copy()
    env["RUN_PATIENT_TEST_1"] = "1"

    # Run the test
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_large_files.py::TestLargeFileProcessing::test_patient_test_1_processing",
        "-v", "-s"  # verbose and no capture for real-time output
    ]

    print("Starting test...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, env=env, timeout=4000)  # 1+ hour timeout

        if result.returncode == 0:
            print()
            print("✅ Patient_test_1 test PASSED!")
            print("Large file processing is working correctly.")
        else:
            print()
            print("❌ Patient_test_1 test FAILED!")
            print("Check the output above for details.")

        return result.returncode

    except subprocess.TimeoutExpired:
        print()
        print("⏰ Test timed out after 1+ hour")
        print("This may indicate a performance issue.")
        return 1
    except KeyboardInterrupt:
        print()
        print("🛑 Test interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)