#!/usr/bin/env python3
"""Compare rxbyak's public mnemonic names with an Xbyak checkout.

This is deliberately a name-level inventory. It does not claim that matching
names have matching operand forms or byte encodings; those require the focused
conformance tests used by each implementation commit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


UPSTREAM_METHOD = re.compile(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
RUST_METHOD = re.compile(r"^\s*pub fn\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
GENERATED_METHOD = re.compile(r'Insn::[A-Za-z_]+\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')

RUST_SUFFIXES = (
    "_st0_st",
    "_st_st0",
    "_string",
    "_store",
    "_load",
    "_imm",
    "_reg",
    "_cl",
    "_xmm",
    "_m32",
    "_m64",
    "_m80",
)


def git_description(path: Path) -> str:
    try:
        tag = subprocess.check_output(
            ["git", "-C", str(path), "describe", "--tags", "--always"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"], text=True
        ).strip()
        return f"{tag} ({commit})"
    except (OSError, subprocess.CalledProcessError):
        return "unknown revision"


def upstream_methods(root: Path) -> tuple[set[str], int]:
    header = root / "xbyak" / "xbyak_mnemonic.h"
    text = header.read_text(encoding="utf-8")
    forms = UPSTREAM_METHOD.findall(text)
    return set(forms), len(forms)


def rxbyak_methods(root: Path) -> tuple[set[str], int, int]:
    assembler = (root / "src" / "assembler.rs").read_text(encoding="utf-8")
    instruction_section = assembler.split("x86 Instructions", maxsplit=1)[1]
    handwritten = set(RUST_METHOD.findall(instruction_section))

    generated: set[str] = set()
    for source in (root / "gen").glob("*.rs"):
        generated.update(GENERATED_METHOD.findall(source.read_text(encoding="utf-8")))
    return handwritten | generated, len(handwritten), len(generated)


def upstream_spellings(methods: set[str]) -> set[str]:
    spellings = set(methods)
    for method in methods:
        for suffix in RUST_SUFFIXES:
            if method.endswith(suffix):
                spellings.add(method[: -len(suffix)])
                break

    for rust_name, xbyak_names in {
        "and_": ("and", "and_"),
        "not_": ("not", "not_"),
        "or_": ("or", "or_"),
        "std_": ("std",),
        "xor_": ("xor", "xor_"),
        "cmpsd_xmm": ("cmpsd",),
        "movsd_string": ("movsd",),
    }.items():
        if rust_name in methods:
            spellings.update(xbyak_names)
    return spellings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xbyak", type=Path, help="path to the Xbyak checkout")
    parser.add_argument(
        "--rxbyak",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="path to the rxbyak checkout",
    )
    args = parser.parse_args()

    upstream, upstream_form_count = upstream_methods(args.xbyak)
    rust, handwritten_count, generated_count = rxbyak_methods(args.rxbyak)
    covered = upstream & upstream_spellings(rust)
    unmatched = sorted(upstream - covered)

    print(f"Xbyak reference: {git_description(args.xbyak)}")
    print(f"Xbyak unique mnemonic names: {len(upstream)}")
    print(f"Xbyak mnemonic overloads: {upstream_form_count}")
    print(f"rxbyak handwritten public methods: {handwritten_count}")
    print(f"rxbyak generated unique methods: {generated_count}")
    print(f"Xbyak names matched after Rust API mapping: {len(covered)}")
    print(f"Xbyak names not matched: {len(unmatched)}")
    print()
    print("Unmatched Xbyak names (includes aliases and 32-bit-only instructions):")
    for name in unmatched:
        print(name)


if __name__ == "__main__":
    main()
