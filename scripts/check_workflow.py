#!/usr/bin/env python3
"""Check that architecture-facing edits carry design and verification changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def changed_files() -> set[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        line.strip()
        for line in (tracked.stdout + untracked.stdout).splitlines()
        if line.strip()
    }


def has_prefix(files: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefixes) for path in files)


def require(
    condition: bool,
    message: str,
    errors: list[str],
) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    files = changed_files()
    if not files:
        print("PASS workflow check: clean worktree")
        return 0

    errors: list[str] = []
    design_changed = has_prefix(files, ("docs/design/", "docs/target_architecture.md"))
    tests_changed = has_prefix(files, ("test/",)) or any(
        "/tb/" in path for path in files
    )
    transformer_design_changed = has_prefix(files, ("docs/design/transformer/",))
    ppa_contract_test_changed = has_prefix(files, ("test/ppa_contract/",))

    npu_rtl_changed = has_prefix(
        files,
        (
            "hw/npu_core/rtl/",
            "hw/npu_wrapper/rtl/",
            "hw/npu_subsystem/rtl/",
        ),
    )
    architecture_config_changed = has_prefix(
        files,
        (
            "arch/configs/npu",
        ),
    )
    architecture_spec_changed = has_prefix(files, ("arch/specs/",))
    compiler_submit_changed = has_prefix(
        files,
        (
            "sw/tools/npu_compiler/",
            "sw/soc_cpu/runtime/",
            "sw/soc_cpu/apps/",
        ),
    )
    ppa_changed = has_prefix(
        files,
        (
            "sw/tools/ppa/",
            "ppa/schema/",
            "arch/configs/ppa/",
        ),
    )

    if npu_rtl_changed:
        require(
            design_changed,
            "NPU RTL changed without an owning docs/design update.",
            errors,
        )
        require(
            tests_changed,
            "NPU RTL changed without a test or RTL testbench update.",
            errors,
        )

    if architecture_spec_changed:
        require(
            design_changed,
            "Architecture spec/config changed without a docs/design update.",
            errors,
        )

    if architecture_config_changed:
        require(
            design_changed,
            "Executable architecture config changed without a docs/design update.",
            errors,
        )
        require(
            tests_changed,
            "Executable architecture config changed without a verification update.",
            errors,
        )

    if compiler_submit_changed:
        require(
            transformer_design_changed,
            "Compiler/submitter changed without a Transformer owning-design update.",
            errors,
        )
        require(
            tests_changed,
            "Compiler/submitter changed without a verification update.",
            errors,
        )

    if ppa_changed:
        require(
            "docs/design/ppa_methodology.md" in files
            or "docs/design/performance_instrumentation.md" in files,
            "PPA contract/tooling changed without a PPA/performance design update.",
            errors,
        )
        require(
            ppa_contract_test_changed,
            "PPA contract/tooling changed without a PPA contract test update.",
            errors,
        )

    if errors:
        print("FAIL workflow check:")
        for error in errors:
            print(f"- {error}")
        print("See AGENTS.md and docs/work_rules.md.")
        return 1

    print(f"PASS workflow check: reviewed {len(files)} changed files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
