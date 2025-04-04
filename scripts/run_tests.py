import os
import sys
import pytest
from pathlib import Path
import logging
from typing import List, Optional

def setup_logging():
    """Setup logging for test runner"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_tests(
    test_paths: Optional[List[str]] = None,
    verbose: bool = False,
    coverage: bool = False
) -> int:
    """Run tests with specified options
    
    Args:
        test_paths: List of paths to test files or directories. If None, runs all tests.
        verbose: Whether to run tests in verbose mode
        coverage: Whether to generate coverage report
        
    Returns:
        int: Exit code from pytest
    """
    logger = setup_logging()
    
    # Default to running all tests if no paths specified
    if test_paths is None:
        test_paths = ["tests"]
    
    # Build pytest arguments
    pytest_args = []
    
    # Add test paths
    pytest_args.extend(test_paths)
    
    # Add verbose flag if requested
    if verbose:
        pytest_args.append("-v")
    
    # Add coverage if requested
    if coverage:
        pytest_args.extend([
            "--cov=lfa",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report"
        ])
    
    # Add additional pytest configuration
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--durations=10",  # Show 10 slowest tests
        "-p", "no:warnings"  # Suppress warnings
    ])
    
    logger.info("Starting test run...")
    logger.info(f"Test paths: {test_paths}")
    logger.info(f"Verbose mode: {verbose}")
    logger.info(f"Coverage report: {coverage}")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code: {exit_code}")
    
    return exit_code

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LFA project tests")
    parser.add_argument(
        "test_paths",
        nargs="*",
        help="Paths to test files or directories (default: all tests)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Run tests with provided arguments
    exit_code = run_tests(
        test_paths=args.test_paths if args.test_paths else None,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    # Exit with the same code as pytest
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 