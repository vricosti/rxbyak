# rxbyak

*A JIT assembler for x86(64) written in Rust, based on [xbyak](https://github.com/herumi/xbyak)*

## Abstract

rxbyak is a pure Rust port of [xbyak](https://github.com/herumi/xbyak), a C++ JIT assembler for x86/x64 architectures. It enables runtime machine code generation with an ergonomic, Intel/MASM-like API.

The name combines **r** (Rust) with **xbyak** (from the Japanese word [開闢](https://translate.google.com/?hl=ja&sl=ja&tl=en&text=%E9%96%8B%E9%97%A2&op=translate), kaibyaku, meaning "creation").

## Features

- Pure Rust, no C/C++ dependencies
- Intel/MASM-like syntax via method calls
- Full support for SSE, AVX, AVX-512 (EVEX), FMA, AES-NI, and AMX
- APX extensions (r16-r31, NF, ZU)
- 683+ auto-generated SIMD instruction methods + hand-written GPR/control flow
- Cross-platform: Linux (`mmap`/`mprotect`) and Windows (`VirtualAlloc`/`VirtualProtect`)
- `Result<T>`-based error handling (no exceptions/panics in the API)
- 522 tests including byte-level NASM validation

### Supported OS

- Windows (64-bit)
- Linux (64-bit)

### Differences from xbyak

rxbyak is **64-bit only**. The following xbyak preprocessor defines have no equivalent:

| xbyak | rxbyak | Reason |
|---|---|---|
| `XBYAK32` | N/A | 32-bit mode not supported |
| `XBYAK64` / `XBYAK64_WIN` / `XBYAK64_GCC` | Automatic | Rust `cfg(target_os)` handles platform detection |
| `XBYAK_USE_OP_NAMES` | N/A | Rust keywords get `_` suffix automatically (`and_()`, `or_()`, `not_()`) |
| `XBYAK_ENABLE_OMITTED_OPERAND` | N/A | All operands must be explicit |
| `XBYAK_UNDEF_JNL` | N/A | No C macro namespace pollution in Rust |
| `XBYAK_NO_EXCEPTION` | N/A | Rust has no exceptions; all methods return `Result<()>` |
| `XBYAK_USE_MEMFD` | Not implemented | Could be added to the Linux backend |
| `XBYAK_OLD_DISP_CHECK` | N/A | Deprecated in xbyak, not ported |

Other differences:

- **Single `Reg` type** instead of a C++ class hierarchy — one flat struct for GPR, XMM, YMM, ZMM, MMX, FPU, opmask, and TMM registers.
- **EVEX modifiers via method chaining** — `ZMM0.k(1).z()` instead of `zmm0 | k1 | T_z`.
- **Build-time code generation** — `build.rs` reads instruction tables from `gen/` and generates method implementations, avoiding manual boilerplate for 683+ SIMD instructions.
- **No `setProtectModeRE()`** — `ready()` handles label resolution and memory protection (RW -> RX) in one call.

## Usage

### Basic Example

```rust
use rxbyak::*;

fn main() -> Result<()> {
    let mut asm = CodeAssembler::new(4096)?;
    asm.mov(EAX, 42i32)?;
    asm.ret()?;
    asm.ready()?;

    let f: extern "C" fn() -> i32 = unsafe { asm.get_code() };
    assert_eq!(f(), 42);
    Ok(())
}
```

### Syntax

Method calls mirror Intel/MASM syntax:

```
NASM                    rxbyak
mov eax, ebx        --> asm.mov(EAX, EBX)?;
inc ecx             --> asm.inc(ECX)?;
ret                 --> asm.ret()?;
```

### Memory Addressing

Size-specific pointer helpers construct `Address` operands:

```rust
asm.mov(EAX, dword_ptr(RBX + RCX * 4))?;          // [rbx+rcx*4]
asm.mov(EAX, dword_ptr(RBP - 8))?;                 // [rbp-8]
asm.inc(byte_ptr(RSI + 0x10))?;                     // byte [rsi+0x10]
asm.mov(RAX, qword_ptr(RDI + R12 * 8 + 0x100))?;   // qword [rdi+r12*8+0x100]
```

Available size helpers: `ptr()`, `byte_ptr()`, `word_ptr()`, `dword_ptr()`, `qword_ptr()`, `xmmword_ptr()`, `ymmword_ptr()`, `zmmword_ptr()`, `broadcast_ptr()`.

### Operand Overloading

Methods accept `impl Into<RegMem>` and `impl Into<RegMemImm>`, so registers, addresses, and immediates can be passed directly:

```rust
asm.mov(EAX, ECX)?;              // reg, reg
asm.mov(EAX, dword_ptr(RBX))?;   // reg, mem
asm.mov(EAX, 42)?;               // reg, imm
asm.mov(dword_ptr(RCX), EAX)?;   // mem, reg
```

### AVX (VEX 3-operand)

```rust
asm.vaddps(XMM1, XMM2, XMM3)?;                     // xmm1 <- xmm2 + xmm3
asm.vaddps(YMM0, YMM1, ymmword_ptr(RAX))?;          // ymm with memory
asm.vfmadd213ps(XMM0, XMM1, XMM2)?;                 // FMA
```

### AVX-512 (EVEX)

```rust
asm.vaddps(ZMM0, ZMM1, ZMM2)?;                      // zmm basic
asm.vaddps(ZMM0.k(1), ZMM1, ZMM2)?;                 // opmask {k1}
asm.vaddps(ZMM0.k(1).z(), ZMM1, ZMM2)?;             // opmask + zeroing {k1}{z}
asm.vaddps(ZMM0.k(2).rounding(Rounding::RnSae),     // rounding {rn-sae}
           ZMM1, ZMM2)?;
```

EVEX modifier methods (return a modified copy of the register):

| Modifier | Method | NASM equivalent |
|---|---|---|
| Opmask | `.k(n)` | `{k1}`..`{k7}` |
| Zeroing | `.z()` | `{z}` |
| Rounding | `.rounding(Rounding::RnSae)` | `{rn-sae}` |
| SAE | `.rounding(Rounding::Sae)` | `{sae}` |
| APX NF | `.nf()` | `{nf}` |
| APX ZU | `.zu()` | `{zu}` |

### Labels

```rust
// Forward reference
let label = asm.create_label();
asm.jmp(&label, JmpType::Near)?;
asm.nop()?;
asm.bind(&label)?;  // label defined here, forward ref patched

// Backward reference
let top = asm.create_label();
asm.bind(&top)?;
// ... loop body ...
asm.jmp(&top, JmpType::Short)?;

// Local label scopes (named)
asm.enter_local();
asm.named_label(".lp")?;
// ...
asm.leave_local()?;
```

All forward references are resolved when `ready()` is called.

### Memory Management

**Fixed allocation** (default):

```rust
let mut asm = CodeAssembler::new(4096)?;  // 4KB code buffer
```

**Auto-growing**:

```rust
let mut asm = CodeAssembler::new_auto_grow(4096)?;  // starts at 4KB, grows as needed
// ... emit lots of code ...
asm.ready()?;  // resolves labels and sets RX protection
```

After `ready()`, the memory is set to read+execute (RX). Use `get_code()` to obtain a typed function pointer:

```rust
asm.ready()?;
let f: extern "C" fn(i64, i64) -> i64 = unsafe { asm.get_code() };
let result = f(10, 20);
```

### Raw Data Emission

```rust
asm.db(0x90)?;           // 1 byte (nop)
asm.dw(0x1234)?;         // 2 bytes (little-endian)
asm.dd(0xDEADBEEF)?;     // 4 bytes
asm.dq(0x123456789ABCDEF0)?;  // 8 bytes
```

### Rust Keyword Handling

Rust keywords used as x86 mnemonics get an `_` suffix:

```rust
asm.and_(EAX, EBX)?;     // AND
asm.or_(EAX, ECX)?;      // OR
asm.not_(EDX)?;           // NOT
asm.xor_(EAX, EAX)?;     // XOR
```

### Error Handling

All instruction methods return `Result<()>`. Errors cover operand validation, encoding constraints, label resolution, and memory allocation (54 error variants matching xbyak error codes):

```rust
match asm.mov(EAX, XMM0) {
    Ok(()) => { /* success */ }
    Err(Error::BadCombination) => { /* invalid operand combination */ }
    Err(e) => { /* other error */ }
}
```

## Supported Instruction Sets

| Category | Examples |
|---|---|
| General Purpose | mov, add, sub, and, or, xor, cmp, test, lea, push, pop, inc, dec, mul, div, imul, neg, not, shl, shr, sar, rol, ror, bsf, bsr, bt, bts, btr, btc, popcnt, lzcnt, tzcnt, cmovcc, setcc, cmpxchg, xadd, bswap, shld, shrd |
| Control Flow | jmp, jcc (all conditions), call, ret, enter, leave, loop |
| String | rep/repe/repne + lodsb/w/d/q, stosb/w/d/q, movsb/w/d/q, scasb/w/d/q, cmpsb/w/d/q |
| SSE/SSE2/SSE3/SSSE3/SSE4 | addps/pd/ss/sd, movaps/dqa, paddd, pxor, cvtsi2ss, comiss, pextrb/w/d/q, pinsrb/w/d/q, blendvps/pd, pshufb, palignr, roundps/pd, ... |
| AVX/AVX2 | vaddps/pd, vmovaps/dqa, vpaddd, vpxor, vfmadd132/213/231ps/pd/ss/sd, vextractf128, vbroadcastss, vinsertf128, vpshufb, ... |
| AVX-512 | All of the above with ZMM/opmask/zeroing/rounding/broadcast, vmovdqa32/64, vmovdqu8/16/32/64, vpternlogd/q, vpermi2*, ... |
| FPU (x87) | fld, fst, fstp, fadd, fsub, fmul, fdiv, fcom, fucom, fcomi, fchs, fabs, fsqrt, fsin, fcos, fxch, fldz, fld1, finit, fldcw, fnstcw, fnstsw, fcmovcc, fiadd, fisub, fimul, fidiv |
| Opmask | kmovw/b/d/q, kandw/b/d/q, korw/b/d/q, kxorw/b/d/q, knotw/b/d/q, kortestw/b, kshiftlw, kshiftrw, kunpckbw, ... |
| AMX | tdpbssd, tdpbsud, tdpbusd, tdpbuud, tdpbf16ps, tdpfp16ps, tileloadd, tileloaddt1, tilestored, tilezero, tilerelease, ldtilecfg, sttilecfg |
| Misc | cpuid, rdtsc, rdtscp, pause, lfence, mfence, sfence, clflush, clflushopt, prefetchnta/t0/t1/t2, stmxcsr, ldmxcsr, movnti, movntps/pd/dq, vzeroall, vzeroupper, vcvtps2ph |

## Architecture

```
User API: CodeAssembler methods (mov, add, vaddps, ...)
    |
    +-- Hand-written (assembler.rs) -- GPR, basic SSE/AVX, jumps, labels
    +-- Auto-generated (mnemonic.rs <- build.rs) -- 683+ SIMD instructions
            |
Encoding Helpers (encode.rs)
    op_rr, op_mr, op_sse, op_vex, op_avx_x_x_xm
    REX/VEX/EVEX prefix emission, ModRM/SIB generation
            |
Byte Buffer (code_array.rs)
    db/dw/dd/dq emission, memory management (Alloc | AutoGrow)
            |
Platform (platform/) -- mmap/VirtualAlloc, memory protection (RW -> RX)
```

Build-time code generation: `build.rs` reads instruction tables from `gen/` (SSE, AVX, FMA, AVX-512, FP16, BF16) and generates `mnemonic.rs` into `OUT_DIR`. Hand-written instructions are listed in `gen/codegen.rs::HANDWRITTEN` to avoid duplication.

## Building

```bash
cargo build                # Debug build
cargo build --release      # Optimized build
cargo test                 # Run all 522 tests
cargo test nm_             # Run NASM conformance tests only
```

## References

- [Intel 64 and IA-32 Architectures Software Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [xbyak](https://github.com/herumi/xbyak) -- Original C++ library by MITSUNARI Shigeo

## License

[BSD-3-Clause](http://opensource.org/licenses/BSD-3-Clause)
