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
UPSTREAM_FORWARDER = re.compile(
    r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^{}]*\)\s*"
    r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^{};]*\);\s*\}"
)
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

ACTIVE_XBYAK_MACROS = {
    "XBYAK64": True,
    "XBYAK_ENABLE_OMITTED_OPERAND": False,
    "XBYAK_DISABLE_AVX512": False,
    "XBYAK_NO_OP_NAMES": False,
}

# These Xbyak spellings emit exactly the same instruction as an rxbyak method
# that already exists. They remain API-compatibility gaps until wrappers are
# added, but they are not missing encoder families.
MANUAL_EQUIVALENT_ALIASES = {
    "jna": "jbe",
    "jnae": "jb",
    "jng": "jle",
    "jnge": "jl",
    "jpe": "jp",
    "jpo": "jnp",
    "popfq": "popf",
    "pushfq": "pushf",
    "sal": "shl",
    "wait": "fwait",
}


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


def active_xbyak_text(text: str) -> str:
    """Evaluate the small preprocessor surface in xbyak_mnemonic.h.

    The audit targets rxbyak's supported 64-bit configuration, with AVX-512
    enabled and optional omitted-operand overloads disabled.
    """

    active = True
    stack: list[tuple[bool, bool]] = []
    output: list[str] = []
    for line in text.splitlines():
        directive = re.match(r"\s*#(ifdef|ifndef)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if directive:
            kind, macro = directive.groups()
            value = ACTIVE_XBYAK_MACROS.get(macro, False)
            condition = value if kind == "ifdef" else not value
            stack.append((active, condition))
            active = active and condition
            continue
        if re.match(r"\s*#else\b", line):
            parent, condition = stack[-1]
            stack[-1] = (parent, not condition)
            active = parent and not condition
            continue
        if re.match(r"\s*#endif\b", line):
            parent, _ = stack.pop()
            active = parent
            continue
        if active:
            output.append(line)
    if stack:
        raise ValueError("unterminated preprocessor block in xbyak_mnemonic.h")
    return "\n".join(output)


def upstream_methods(root: Path) -> tuple[set[str], int, dict[str, set[str]]]:
    header = root / "xbyak" / "xbyak_mnemonic.h"
    text = active_xbyak_text(header.read_text(encoding="utf-8"))
    forms = UPSTREAM_METHOD.findall(text)
    forwarders: dict[str, set[str]] = {}
    for source, target in UPSTREAM_FORWARDER.findall(text):
        if source != target:
            forwarders.setdefault(source, set()).add(target)
    return set(forms), len(forms), forwarders


def rxbyak_methods(root: Path) -> tuple[set[str], int, int]:
    assembler = (root / "src" / "assembler.rs").read_text(encoding="utf-8")
    instruction_section = assembler.split("x86 Instructions", maxsplit=1)[1]
    handwritten = set(RUST_METHOD.findall(instruction_section))

    generated: set[str] = set()
    for source in (root / "gen").glob("*.rs"):
        generated.update(GENERATED_METHOD.findall(source.read_text(encoding="utf-8")))

    codegen = (root / "gen" / "codegen.rs").read_text(encoding="utf-8")
    predicate_block = re.search(
        r"const COMPARE_PREDICATES:\s*\[&str;\s*32\]\s*=\s*\[(.*?)\];",
        codegen,
        re.DOTALL,
    )
    if predicate_block:
        predicates = re.findall(r'"([a-z0-9_]+)"', predicate_block.group(1))
        for suffix in ("pd", "ps", "sd", "ss"):
            generated.update(f"vcmp{predicate}{suffix}" for predicate in predicates)
            generated.update(f"cmp{predicate}{suffix}" for predicate in predicates[:8])

    pclmul_block = re.search(
        r"const PCLMUL_ALIASES:.*?=\s*\[(.*?)\];", codegen, re.DOTALL
    )
    if pclmul_block:
        aliases = re.findall(r'\("([a-z0-9_]+)"\s*,', pclmul_block.group(1))
        generated.update(aliases)
        generated.update(f"v{name}" for name in aliases)
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

    upstream, upstream_form_count, forwarders = upstream_methods(args.xbyak)
    rust, handwritten_count, generated_count = rxbyak_methods(args.rxbyak)
    rust_spellings = upstream_spellings(rust)
    covered = upstream & rust_spellings
    unmatched = upstream - covered

    equivalent_aliases = {
        name: sorted(target for target in targets if target in rust_spellings)
        for name, targets in forwarders.items()
        if name in unmatched and any(target in rust_spellings for target in targets)
    }
    for name, target in MANUAL_EQUIVALENT_ALIASES.items():
        if name in unmatched and target in rust_spellings:
            equivalent_aliases[name] = [target]
    missing = sorted(unmatched - equivalent_aliases.keys())

    print(f"Xbyak reference: {git_description(args.xbyak)}")
    print(f"Xbyak active 64-bit mnemonic names: {len(upstream)}")
    print(f"Xbyak active 64-bit mnemonic overloads: {upstream_form_count}")
    print(f"rxbyak handwritten public methods: {handwritten_count}")
    print(f"rxbyak generated unique methods: {generated_count}")
    print(f"Xbyak names matched after Rust API mapping: {len(covered)}")
    print(f"Xbyak names not matched: {len(unmatched)}")
    print(f"  compatibility aliases with an equivalent encoder: {len(equivalent_aliases)}")
    print(f"  missing encoder/API names: {len(missing)}")
    print()
    print("Missing compatibility aliases (Xbyak name -> existing rxbyak encoder):")
    for name in sorted(equivalent_aliases):
        print(f"{name} -> {', '.join(equivalent_aliases[name])}")
    print()
    print("Missing encoder/API names:")
    for name in missing:
        print(name)


if __name__ == "__main__":
    main()
