#!/usr/bin/env python3
"""
Security Audit Script for FinancePortfolio

This script runs comprehensive security checks including:
1. Dependency vulnerability scanning (pip-audit)
2. Static security analysis (bandit)
3. Code linting with security rules (ruff S rules)

Usage:
    uv run python scripts/security_audit.py [--fix] [--verbose]

Options:
    --fix       Attempt to fix fixable vulnerabilities
    --verbose   Show detailed output
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(
    cmd: list[str], description: str, verbose: bool = False
) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        capture_output=not verbose,
        text=True,
    )

    if not verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

    return result.returncode, result.stdout or "", result.stderr or ""


def _filter_requirements(raw_output: str) -> list[str]:
    """Filter requirements to exclude local packages and invalid lines."""
    filtered: list[str] = []
    skip_prefixes = ("#", "-e")
    skip_patterns = ("file://", "financeportfolio")

    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in skip_prefixes):
            continue
        if any(p in stripped.lower() for p in skip_patterns):
            continue
        filtered.append(stripped)

    return filtered


def _export_requirements() -> tuple[bool, str]:
    """Export requirements using uv and return (success, output/error)."""
    result = subprocess.run(
        ["uv", "export", "--no-hashes"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr
    return True, result.stdout


def run_pip_audit(fix: bool = False, verbose: bool = False) -> int:
    """Run pip-audit to scan for dependency vulnerabilities."""
    import tempfile

    # Export requirements
    success, output = _export_requirements()
    if not success:
        print(f"Failed to export requirements: {output}")
        return 1

    # Filter and write to temp file
    filtered_lines = _filter_requirements(output)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as req_file:
        req_path = req_file.name
        req_file.write("\n".join(filtered_lines))

    cmd = ["uv", "run", "pip-audit", "-r", req_path, "--desc", "on", "--strict"]
    if fix:
        cmd.append("--fix")
    if verbose:
        cmd.append("--progress-spinner=off")

    return_code, _, _ = run_command(
        cmd, "Dependency Vulnerability Scan (pip-audit)", verbose
    )

    # Clean up temp file
    try:
        Path(req_path).unlink()
    except OSError:
        pass

    if return_code == 0:
        print("[PASS] No known vulnerabilities found in dependencies")
    else:
        print("[FAIL] Vulnerabilities detected in dependencies")

    return return_code


def run_bandit(verbose: bool = False) -> int:
    """Run bandit for static security analysis."""
    cmd = [
        "uv",
        "run",
        "bandit",
        "-c",
        "pyproject.toml",
        "-r",
        ".",
        "--severity-level",
        "medium",
    ]

    if verbose:
        cmd.append("-v")

    return_code, stdout, stderr = run_command(
        cmd, "Static Security Analysis (Bandit)", verbose
    )

    if return_code == 0:
        print("[PASS] No medium or high severity security issues found")
    else:
        print("[FAIL] Security issues detected by Bandit")

    return return_code


def run_ruff_security(verbose: bool = False) -> int:
    """Run ruff with security rules only."""
    cmd = ["uv", "run", "ruff", "check", ".", "--select", "S"]

    return_code, stdout, stderr = run_command(
        cmd, "Security Linting (Ruff S rules)", verbose
    )

    if return_code == 0:
        print("[PASS] No security issues found by Ruff")
    else:
        print("[FAIL] Security issues detected by Ruff")

    return return_code


def main() -> int:
    """Run all security checks and return overall status."""
    parser = argparse.ArgumentParser(
        description="Run security audits for FinancePortfolio"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix fixable vulnerabilities",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only run dependency vulnerability scan",
    )
    args = parser.parse_args()

    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    results = {}

    # Always run dependency audit
    print("\n" + "#" * 60)
    print("# DEPENDENCY VULNERABILITY SCAN")
    print("#" * 60)
    results["pip-audit"] = run_pip_audit(fix=args.fix, verbose=args.verbose)

    if not args.deps_only:
        # Run static analysis
        print("\n" + "#" * 60)
        print("# STATIC SECURITY ANALYSIS")
        print("#" * 60)
        results["bandit"] = run_bandit(verbose=args.verbose)

        print("\n" + "#" * 60)
        print("# SECURITY LINTING")
        print("#" * 60)
        results["ruff-security"] = run_ruff_security(verbose=args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("SECURITY AUDIT SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, code in results.items():
        status = "PASS" if code == 0 else "FAIL"
        symbol = "[+]" if code == 0 else "[!]"
        print(f"{symbol} {check}: {status}")
        if code != 0:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All security checks passed!")
        return 0
    else:
        print("\n[WARNING] Some security checks failed. Please review and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
