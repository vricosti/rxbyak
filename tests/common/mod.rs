use rxbyak::*;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Try to locate the NASM executable.
/// Checks PATH first, then the known Windows location via WSL.
pub fn find_nasm() -> Option<String> {
    // Check if nasm is on PATH
    if let Ok(output) = Command::new("nasm").arg("-v").output() {
        if output.status.success() {
            return Some("nasm".to_string());
        }
    }
    // Check WSL path to Windows NASM
    let wsl_path = "/mnt/c/Apps/NASM/nasm.exe";
    if std::path::Path::new(wsl_path).exists() {
        return Some(wsl_path.to_string());
    }
    None
}

/// Returns true if `nasm_path` is a Windows .exe accessed via WSL.
fn is_windows_nasm(nasm_path: &str) -> bool {
    nasm_path.ends_with(".exe")
}

/// Convert a WSL path like `/mnt/c/foo/bar` to a Windows path like `C:\foo\bar`.
fn wsl_to_windows_path(wsl_path: &str) -> String {
    if let Some(rest) = wsl_path.strip_prefix("/mnt/") {
        if rest.len() >= 2 {
            let drive = rest.chars().next().unwrap().to_uppercase().to_string();
            let remainder = &rest[1..]; // starts with '/'
            return format!("{}:{}", drive, remainder.replace('/', "\\"));
        }
    }
    wsl_path.to_string()
}

/// Get a unique temp directory for NASM files.
fn temp_dir() -> PathBuf {
    let dir = PathBuf::from(
        std::env::var("CARGO_TARGET_TMPDIR")
            .unwrap_or_else(|_| "/tmp".to_string()),
    )
    .join("nasm_tmp");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// Generate a unique filename stem for thread safety.
fn unique_name() -> String {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tid = std::thread::current().id();
    format!("nasm_{:?}_{}", tid, id)
        .replace(['(', ')', ' '], "_")
}

/// Assemble NASM source into raw binary bytes.
/// `bits` should be 32 or 64.
/// Returns None if NASM is not available, panics on assembly errors.
pub fn nasm_assemble(nasm_path: &str, code: &str, bits: u32) -> Vec<u8> {
    let dir = temp_dir();
    let name = unique_name();
    let asm_path = dir.join(format!("{}.asm", name));
    let bin_path = dir.join(format!("{}.bin", name));

    let full_source = format!("bits {}\n{}", bits, code);
    std::fs::write(&asm_path, &full_source).expect("failed to write NASM source");

    let (asm_arg, bin_arg) = if is_windows_nasm(nasm_path) {
        (
            wsl_to_windows_path(asm_path.to_str().unwrap()),
            wsl_to_windows_path(bin_path.to_str().unwrap()),
        )
    } else {
        (
            asm_path.to_str().unwrap().to_string(),
            bin_path.to_str().unwrap().to_string(),
        )
    };

    let output = Command::new(nasm_path)
        .args(["-f", "bin", "-o", &bin_arg, &asm_arg])
        .output()
        .expect("failed to run NASM");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "NASM assembly failed:\n--- source ---\n{}\n--- stderr ---\n{}",
            full_source, stderr
        );
    }

    let bytes = std::fs::read(&bin_path).expect("failed to read NASM output");

    // Clean up temp files
    std::fs::remove_file(&asm_path).ok();
    std::fs::remove_file(&bin_path).ok();

    bytes
}

/// Assemble NASM source and return per-line byte output via listing.
/// Each entry is (source_line, bytes_for_that_line).
pub fn nasm_listing(nasm_path: &str, code: &str, bits: u32) -> Vec<(String, Vec<u8>)> {
    let dir = temp_dir();
    let name = unique_name();
    let asm_path = dir.join(format!("{}.asm", name));
    let bin_path = dir.join(format!("{}.bin", name));
    let lst_path = dir.join(format!("{}.lst", name));

    let full_source = format!("bits {}\n{}", bits, code);
    std::fs::write(&asm_path, &full_source).expect("failed to write NASM source");

    let (asm_arg, bin_arg, lst_arg) = if is_windows_nasm(nasm_path) {
        (
            wsl_to_windows_path(asm_path.to_str().unwrap()),
            wsl_to_windows_path(bin_path.to_str().unwrap()),
            wsl_to_windows_path(lst_path.to_str().unwrap()),
        )
    } else {
        (
            asm_path.to_str().unwrap().to_string(),
            bin_path.to_str().unwrap().to_string(),
            lst_path.to_str().unwrap().to_string(),
        )
    };

    let output = Command::new(nasm_path)
        .args(["-f", "bin", "-o", &bin_arg, "-l", &lst_arg, &asm_arg])
        .output()
        .expect("failed to run NASM");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "NASM listing failed:\n--- source ---\n{}\n--- stderr ---\n{}",
            full_source, stderr
        );
    }

    let listing = std::fs::read_to_string(&lst_path).expect("failed to read NASM listing");
    let result = parse_listing(&listing);

    // Clean up
    std::fs::remove_file(&asm_path).ok();
    std::fs::remove_file(&bin_path).ok();
    std::fs::remove_file(&lst_path).ok();

    result
}

/// Parse a NASM listing file into (source_line, bytes) pairs.
/// Handles continuation lines where NASM splits long instructions across multiple
/// listing lines (e.g. EVEX instructions >9 bytes).
///
/// NASM listing format:
///   line_num  offset  hex_bytes[-]  source
/// Continuation line (same line number, no source):
///   line_num  offset  hex_bytes
fn parse_listing(listing: &str) -> Vec<(String, Vec<u8>)> {
    let mut results: Vec<(String, Vec<u8>)> = Vec::new();
    let mut prev_line_num: Option<&str> = None;

    for line in listing.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let non_empty: Vec<&str> = trimmed.split_whitespace().collect();
        if non_empty.len() < 3 {
            continue;
        }

        let line_num = non_empty[0];
        let offset_str = non_empty[1];

        // Offset should be 8 hex digits
        if offset_str.len() != 8 || !offset_str.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }

        // The hex bytes field — may end with '-' indicating continuation
        let hex_field = non_empty[2];
        let hex_str = hex_field.trim_end_matches('-');
        if hex_str.is_empty() || !hex_str.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }

        let bytes = hex_to_bytes(hex_str);

        // Extract source: everything after the hex field
        let source = if non_empty.len() > 3 {
            non_empty[3..].join(" ")
        } else {
            String::new()
        };

        // Skip the "bits 64/32" directive
        if source.starts_with("bits ") {
            prev_line_num = Some(line_num);
            continue;
        }

        // Check if this is a continuation line (same line number, no source)
        let is_continuation = source.is_empty()
            && prev_line_num == Some(line_num)
            && !results.is_empty();

        if is_continuation {
            // Append bytes to the previous entry
            results.last_mut().unwrap().1.extend_from_slice(&bytes);
        } else if !bytes.is_empty() {
            results.push((source, bytes));
        }

        prev_line_num = Some(line_num);
    }
    results
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut chars = hex.chars();
    while let (Some(h), Some(l)) = (chars.next(), chars.next()) {
        if let Ok(b) = u8::from_str_radix(&format!("{}{}", h, l), 16) {
            bytes.push(b);
        }
    }
    bytes
}

/// Assemble rxbyak instructions and return the generated bytes.
pub fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(65536).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

/// Compare rxbyak output against NASM for a batch of instructions.
/// `instructions` is a list of (nasm_text, rxbyak_closure) pairs.
/// All NASM instructions are assembled in one invocation for efficiency.
pub fn compare_nasm_batch(
    nasm_path: &str,
    bits: u32,
    instructions: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)>,
) {
    // Build NASM source
    let nasm_source: String = instructions
        .iter()
        .map(|(asm, _)| asm.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    // Get per-instruction NASM bytes via listing
    let nasm_results = nasm_listing(nasm_path, &nasm_source, bits);

    // Assemble each instruction individually with rxbyak
    let mut failures = Vec::new();
    for (i, (nasm_text, rxbyak_fn)) in instructions.into_iter().enumerate() {
        let rxbyak_bytes = assemble(rxbyak_fn);
        let nasm_entry = nasm_results.get(i);

        match nasm_entry {
            Some((_, nasm_bytes)) => {
                if rxbyak_bytes != *nasm_bytes {
                    failures.push(format!(
                        "  [{}] {}\n    NASM:   {:02X?}\n    rxbyak: {:02X?}",
                        i, nasm_text, nasm_bytes, rxbyak_bytes
                    ));
                }
            }
            None => {
                failures.push(format!(
                    "  [{}] {} — no NASM listing entry (got {} entries)",
                    i, nasm_text, nasm_results.len()
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "NASM comparison failures ({}/{}):\n{}",
            failures.len(),
            nasm_results.len(),
            failures.join("\n")
        );
    }
}

// ─── Register name tables ────────────────────────────────────────

pub const REGS8: &[(Reg, &str)] = &[
    (AL, "al"), (CL, "cl"), (DL, "dl"), (BL, "bl"),
];

pub const REGS8_EXT: &[(Reg, &str)] = &[
    (R8B, "r8b"), (R9B, "r9b"), (R10B, "r10b"), (R11B, "r11b"),
    (R12B, "r12b"), (R13B, "r13b"), (R14B, "r14b"), (R15B, "r15b"),
];

pub const REGS16: &[(Reg, &str)] = &[
    (AX, "ax"), (CX, "cx"), (DX, "dx"), (BX, "bx"),
    (SP, "sp"), (BP, "bp"), (SI, "si"), (DI, "di"),
];

pub const REGS32: &[(Reg, &str)] = &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (ESP, "esp"), (EBP, "ebp"), (ESI, "esi"), (EDI, "edi"),
];

pub const REGS32_EXT: &[(Reg, &str)] = &[
    (R8D, "r8d"), (R9D, "r9d"), (R10D, "r10d"), (R11D, "r11d"),
    (R12D, "r12d"), (R13D, "r13d"), (R14D, "r14d"), (R15D, "r15d"),
];

pub const REGS64: &[(Reg, &str)] = &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (RSP, "rsp"), (RBP, "rbp"), (RSI, "rsi"), (RDI, "rdi"),
];

pub const REGS64_EXT: &[(Reg, &str)] = &[
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R11, "r11"),
    (R12, "r12"), (R13, "r13"), (R14, "r14"), (R15, "r15"),
];

pub const XMMS: &[(Reg, &str)] = &[
    (XMM0, "xmm0"), (XMM1, "xmm1"), (XMM2, "xmm2"), (XMM3, "xmm3"),
    (XMM4, "xmm4"), (XMM5, "xmm5"), (XMM6, "xmm6"), (XMM7, "xmm7"),
];

pub const XMMS_EXT: &[(Reg, &str)] = &[
    (XMM8, "xmm8"), (XMM9, "xmm9"), (XMM10, "xmm10"), (XMM11, "xmm11"),
    (XMM12, "xmm12"), (XMM13, "xmm13"), (XMM14, "xmm14"), (XMM15, "xmm15"),
];

pub const YMMS: &[(Reg, &str)] = &[
    (YMM0, "ymm0"), (YMM1, "ymm1"), (YMM2, "ymm2"), (YMM3, "ymm3"),
    (YMM4, "ymm4"), (YMM5, "ymm5"), (YMM6, "ymm6"), (YMM7, "ymm7"),
];

pub const YMMS_EXT: &[(Reg, &str)] = &[
    (YMM8, "ymm8"), (YMM9, "ymm9"), (YMM10, "ymm10"), (YMM11, "ymm11"),
    (YMM12, "ymm12"), (YMM13, "ymm13"), (YMM14, "ymm14"), (YMM15, "ymm15"),
];

pub const ZMMS: &[(Reg, &str)] = &[
    (ZMM0, "zmm0"), (ZMM1, "zmm1"), (ZMM2, "zmm2"), (ZMM3, "zmm3"),
    (ZMM4, "zmm4"), (ZMM5, "zmm5"), (ZMM6, "zmm6"), (ZMM7, "zmm7"),
];

pub const ZMMS_EXT: &[(Reg, &str)] = &[
    (ZMM8, "zmm8"), (ZMM9, "zmm9"), (ZMM10, "zmm10"), (ZMM11, "zmm11"),
    (ZMM12, "zmm12"), (ZMM13, "zmm13"), (ZMM14, "zmm14"), (ZMM15, "zmm15"),
    (ZMM16, "zmm16"), (ZMM17, "zmm17"), (ZMM18, "zmm18"), (ZMM19, "zmm19"),
    (ZMM20, "zmm20"), (ZMM21, "zmm21"), (ZMM22, "zmm22"), (ZMM23, "zmm23"),
    (ZMM24, "zmm24"), (ZMM25, "zmm25"), (ZMM26, "zmm26"), (ZMM27, "zmm27"),
    (ZMM28, "zmm28"), (ZMM29, "zmm29"), (ZMM30, "zmm30"), (ZMM31, "zmm31"),
];

/// GPR64 registers that can be used as base (all 16).
pub const BASES64: &[(Reg, &str)] = &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (RSP, "rsp"), (RBP, "rbp"), (RSI, "rsi"), (RDI, "rdi"),
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R11, "r11"),
    (R12, "r12"), (R13, "r13"), (R14, "r14"), (R15, "r15"),
];

/// GPR64 registers that can be used as index (all except RSP).
pub const INDICES64: &[(Reg, &str)] = &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (RBP, "rbp"), (RSI, "rsi"), (RDI, "rdi"),
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R11, "r11"),
    (R12, "r12"), (R13, "r13"), (R14, "r14"), (R15, "r15"),
];

// ─── Additional register tables ──────────────────────────────────

pub const REGS8_HIGH: &[(Reg, &str)] = &[
    (AH, "ah"), (CH, "ch"), (DH, "dh"), (BH, "bh"),
];

pub const REGS8_SPL: &[(Reg, &str)] = &[
    (SPL, "spl"), (BPL, "bpl"), (SIL, "sil"), (DIL, "dil"),
];

pub const REGS16_EXT: &[(Reg, &str)] = &[
    (R8W, "r8w"), (R9W, "r9w"), (R10W, "r10w"), (R11W, "r11w"),
    (R12W, "r12w"), (R13W, "r13w"), (R14W, "r14w"), (R15W, "r15w"),
];

pub const MMXS: &[(Reg, &str)] = &[
    (MM0, "mm0"), (MM1, "mm1"), (MM2, "mm2"), (MM3, "mm3"),
    (MM4, "mm4"), (MM5, "mm5"), (MM6, "mm6"), (MM7, "mm7"),
];

pub const OPMASKS: &[(Reg, &str)] = &[
    (K0, "k0"), (K1, "k1"), (K2, "k2"), (K3, "k3"),
    (K4, "k4"), (K5, "k5"), (K6, "k6"), (K7, "k7"),
];

pub const OPMASKS_NONZERO: &[(Reg, &str)] = &[
    (K1, "k1"), (K2, "k2"), (K3, "k3"),
    (K4, "k4"), (K5, "k5"), (K6, "k6"), (K7, "k7"),
];

pub const FPUS: &[(Reg, &str)] = &[
    (ST0, "st0"), (ST1, "st1"), (ST2, "st2"), (ST3, "st3"),
    (ST4, "st4"), (ST5, "st5"), (ST6, "st6"), (ST7, "st7"),
];

// ─── Memory operand helpers ──────────────────────────────────────

/// Build representative address expressions (no size) for testing.
/// Returns (RegExp, nasm_bracket_string) pairs.
fn addr_exprs() -> Vec<(RegExp, String)> {
    vec![
        (RAX.into(), "[rax]".to_string()),
        ((RCX + 0x10), "[rcx+0x10]".to_string()),
        (RBP.into(), "[rbp]".to_string()),
        (R13.into(), "[r13]".to_string()),
        ((RAX + RCX * 4 + 0x100), "[rax+rcx*4+0x100]".to_string()),
        ((R13 + R14 * 2 + 0x20), "[r13+r14*2+0x20]".to_string()),
    ]
}

/// Generate sized memory operands using a given address constructor and NASM size prefix.
fn mems_sized(
    make_addr: fn(RegExp) -> Address,
    nasm_prefix: &str,
) -> Vec<(Address, String)> {
    addr_exprs()
        .into_iter()
        .map(|(exp, bracket)| {
            (make_addr(exp), format!("{} {}", nasm_prefix, bracket))
        })
        .collect()
}

pub fn mems8() -> Vec<(Address, String)> { mems_sized(byte_ptr, "byte") }
pub fn mems16() -> Vec<(Address, String)> { mems_sized(word_ptr, "word") }
pub fn mems32() -> Vec<(Address, String)> { mems_sized(dword_ptr, "dword") }
pub fn mems64() -> Vec<(Address, String)> { mems_sized(qword_ptr, "qword") }
pub fn mems128() -> Vec<(Address, String)> { mems_sized(xmmword_ptr, "oword") }
pub fn mems256() -> Vec<(Address, String)> { mems_sized(ymmword_ptr, "yword") }
pub fn mems512() -> Vec<(Address, String)> { mems_sized(zmmword_ptr, "zword") }

/// Untyped memory operands (no size prefix, for instructions where size is implicit).
pub fn mems_nosizeptr() -> Vec<(Address, String)> {
    addr_exprs()
        .into_iter()
        .map(|(exp, bracket)| (ptr(exp), bracket))
        .collect()
}

// ─── Prefix normalization ────────────────────────────────────────

/// Normalize legacy prefix ordering.
/// NASM and rxbyak may emit legacy prefixes (0x66, 0x67, 0xF2, 0xF3) in
/// different orders. Sort leading prefixes so both sides match.
pub fn normalize_prefix(bytes: &[u8]) -> Vec<u8> {
    const LEGACY_PREFIXES: &[u8] = &[0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3];
    let prefix_end = bytes.iter().position(|b| !LEGACY_PREFIXES.contains(b)).unwrap_or(bytes.len());
    let mut prefixes = bytes[..prefix_end].to_vec();
    prefixes.sort_unstable();
    let mut result = prefixes;
    result.extend_from_slice(&bytes[prefix_end..]);
    result
}

/// Like `compare_nasm_batch` but normalizes legacy prefix order before comparing.
pub fn compare_nasm_batch_normalized(
    nasm_path: &str,
    bits: u32,
    instructions: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)>,
) {
    let nasm_source: String = instructions
        .iter()
        .map(|(asm, _)| asm.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let nasm_results = nasm_listing(nasm_path, &nasm_source, bits);

    let mut failures = Vec::new();
    for (i, (nasm_text, rxbyak_fn)) in instructions.into_iter().enumerate() {
        let rxbyak_bytes = normalize_prefix(&assemble(rxbyak_fn));
        let nasm_entry = nasm_results.get(i);

        match nasm_entry {
            Some((_, nasm_bytes)) => {
                let nasm_norm = normalize_prefix(nasm_bytes);
                if rxbyak_bytes != nasm_norm {
                    failures.push(format!(
                        "  [{}] {}\n    NASM:   {:02X?}\n    rxbyak: {:02X?}",
                        i, nasm_text, nasm_norm, rxbyak_bytes
                    ));
                }
            }
            None => {
                failures.push(format!(
                    "  [{}] {} — no NASM listing entry (got {} entries)",
                    i, nasm_text, nasm_results.len()
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "NASM comparison failures ({}/{}):\n{}",
            failures.len(),
            nasm_results.len(),
            failures.join("\n")
        );
    }
}
