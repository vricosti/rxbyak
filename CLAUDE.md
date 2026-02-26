# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rxbyak is a pure Rust port of [xbyak](https://github.com/herumi/xbyak), an x86/x64 JIT assembler library. It enables runtime machine code generation with full instruction encoding support including SSE, AVX, AVX-512, FMA, AES-NI, and Intel APX extensions. Cross-platform: Linux (mmap/mprotect) and Windows (VirtualAlloc/VirtualProtect).

## Environment

On Windows (WSL), cargo is installed in the current user's profile: `C:\Users\<user>\.cargo\bin`. Use the full path or ensure it is in `PATH`.

## Build & Test Commands

```bash
cargo build                    # Debug build (triggers build.rs code generation)
cargo build --release          # Optimized build
cargo test                     # Run all 560+ tests
cargo test test_mov            # Run tests matching a pattern
cargo test -- --nocapture      # Show stdout during tests
cargo clippy                   # Lint
cargo fmt                      # Format
cargo clean && cargo build     # Force regeneration of instruction methods
```

## Architecture

### Layer Diagram

```
User API: CodeAssembler methods (mov, add, vaddps, ...)
    │
    ├── Hand-written (assembler.rs) — GPR, basic SSE/AVX, jumps, labels
    ├── Auto-generated (mnemonic.rs ← build.rs) — 683+ SIMD instructions
    └── StackFrame (util/stack_frame.rs) — prologue/epilogue, ABI register mapping
            │
Encoding Helpers (encode.rs)
    op_rr, op_mr, op_sse, op_vex, op_avx_x_x_xm
    REX/VEX/EVEX prefix emission, ModRM/SIB generation
            │
Byte Buffer (code_array.rs)
    db/dw/dd/dq emission, memory management (UserBuf|Alloc|AutoGrow)
            │
Platform (platform/) — mmap/VirtualAlloc, memory protection (RW→RX)
```

### Build-Time Code Generation

`build.rs` reads instruction tables from `gen/` and generates `mnemonics.rs` into `OUT_DIR`:
- `gen/instructions.rs` — SSE, AVX, FMA tables
- `gen/avx512.rs` — AVX-512/EVEX tables (FP16, BF16)
- `gen/codegen.rs` — Transforms tables into `impl CodeAssembler` methods
- `gen/mod.rs` — `Insn` and `Pattern` enum definitions

Instructions listed in `gen/codegen.rs::HANDWRITTEN` are skipped (already in `assembler.rs`). Rust keywords get `_` suffix (`and_`, `or_`, `not_`).

### Key Types

- **`CodeAssembler`** (`assembler.rs`) — Main API. Wraps `CodeBuffer`. Call `ready()` to resolve labels and set memory executable.
- **`CodeBuffer`** (`code_array.rs`) — Low-level byte emission + label tracking. Three allocation modes: `UserBuf`, `Alloc` (fixed), `AutoGrow` (dynamic).
- **`Reg`** (`operand.rs`) — Unified register: GPR/XMM/YMM/ZMM/MMX/FPU/Opmask/Tmm with EVEX modifiers (mask, rounding, zeroing, NF, ZU).
- **`RegMem`** / **`RegMemImm`** (`operand.rs`) — Operand enums. `Into<>` trait impls enable ergonomic overloading.
- **`Address`** / **`RegExp`** (`address.rs`) — Memory addressing: `[base + index*scale + disp]`, RIP-relative. Helper functions: `ptr()`, `byte_ptr()`, `dword_ptr()`, `qword_ptr()`, `xmmword_ptr()`, etc.
- **`TypeFlags`** (`encoding_flags.rs`) — 64-bit bitfield controlling encoding: prefix (T_66/T_F2/T_F3), opcode map (T_0F/T_0F38/T_0F3A), operand width (T_W0/T_W1), VEX/EVEX mode, broadcast, masking, rounding.
- **`LabelManager`** (`label.rs`) — Forward reference resolution. Labels are patched in-place during `ready()`.
- **`StackFrame`** (`util/stack_frame.rs`) — Automatic function prologue/epilogue generator. Handles callee-saved register preservation, 16-byte stack alignment, and parameter/temporary register mapping per platform ABI.

### Operand Overloading Pattern

Methods accept `impl Into<RegMem>` / `impl Into<RegMemImm>` so callers can pass `Reg`, `Address`, or integer immediates directly:
```rust
asm.mov(EAX, ECX)?;           // reg, reg
asm.mov(EAX, dword_ptr(RBX))?; // reg, mem
asm.mov(EAX, 42)?;             // reg, imm
```

### Encoding Pipeline (encode.rs)

Instruction encoding follows x86 format: `[Prefix][Opcode][ModRM][SIB][Disp][Imm]`
- **Legacy**: REX prefix → opcode → ModRM/SIB
- **VEX** (AVX/AVX2): 2-3 byte VEX → opcode → ModRM
- **EVEX** (AVX-512): 4-byte EVEX → opcode → ModRM (adds opmask, broadcast, rounding)

### Test Pattern

Tests validate byte-level encoding output:
```rust
fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

#[test]
fn test_mov_eax_ecx() {
    let code = assemble(|a| a.mov(EAX, ECX));
    assert_eq!(code, [0x89, 0xC8]);
}
```

### Adding a New Instruction

1. **Generated (SIMD/AVX)**: Add entry to the appropriate table in `gen/instructions.rs` or `gen/avx512.rs` with correct `Pattern`, `TypeFlags`, and opcode. Run `cargo build` to regenerate.
2. **Hand-written (GPR/complex)**: Add method directly to `assembler.rs` and add name to `HANDWRITTEN` in `gen/codegen.rs`.
3. **Add test** in `tests/` validating exact byte output against a reference (e.g., objdump, Intel manual).

### StackFrame (util/stack_frame.rs)

Port of xbyak's `Xbyak::util::StackFrame`. Generates calling-convention-aware function prologue/epilogue for JIT-compiled functions callable from C/Rust.

**API**:
- `StackFrame::new(asm, pNum, tNum, stackSize)` — emit prologue, return register mapping
- `sf.close(asm)` — emit epilogue (`add rsp` + pop callee-saved + `ret`)
- `sf.p[0..pNum]` — parameter registers, `sf.t[0..tNum]` — temporary registers
- `USE_RCX` / `USE_RDX` — OR with `tNum` to request RCX/RDX as free regs (original param values preserved to R10/R11)

**Platform ABI** (compile-time `cfg(target_os)`):

| | Windows x64 | System V AMD64 |
|---|---|---|
| Param regs | RCX, RDX, R8, R9 | RDI, RSI, RDX, RCX, R8, R9 |
| Scratch (NO_SAVE) | 6 (+ RDI, RSI callee-saved) | 8 |
| Callee-saved | RDI, RSI, RBX, RBP, R12-R15 | RBX, RBP, R12-R15 |

**Register allocation**: Walks a 14-entry order table (params → scratch → callee-saved), skipping RCX/RDX when USE flags set. `allRegNum = pNum + tNum + useRcx + useRdx`, max 14.

**Stack alignment**: `P = ceil(stackSize/8)` slots; if `(P & 1) == (saveNum & 1)` add one slot to maintain 16-byte RSP alignment.

**Rust adaptation**: xbyak uses RAII destructor to auto-emit epilogue. In Rust, the user must explicitly call `sf.close(&mut asm)?`.

```rust
use rxbyak::*;
use rxbyak::util::stack_frame::StackFrame;

let mut asm = CodeAssembler::new(4096)?;
let sf = StackFrame::new(&mut asm, 2, 1, 0)?;  // 2 params, 1 temp
asm.mov(RAX, sf.p[0])?;
asm.add(RAX, sf.p[1])?;
sf.close(&mut asm)?;
asm.ready()?;
let f: fn(i64, i64) -> i64 = unsafe { asm.get_code() };
assert_eq!(f(10, 32), 42);
```

### Error System

54 error variants in `error.rs` matching xbyak error codes. Covers operand validation, encoding constraints, label resolution, memory allocation, and AVX-512/APX-specific errors.
