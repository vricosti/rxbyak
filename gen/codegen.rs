/// Code generation logic: transforms instruction tables into Rust method implementations.

use super::{Insn, Pattern};
use std::io::Write;

/// Names of instructions that are already hand-written in assembler.rs.
/// These will be skipped during code generation to avoid conflicts.
const HANDWRITTEN: &[&str] = &[
    // GPR instructions
    "nop", "ret", "push", "pop", "mov", "add", "or_", "adc", "sbb", "and_", "sub", "xor_",
    "cmp", "lea", "test", "inc", "dec", "neg", "not_", "jmp", "call", "int3", "xchg",
    "movzx", "movsx", "movsxd", "cdq", "cqo", "imul", "shl", "shr", "sar",
    // Conditional jumps (hand-written with label support)
    "jo", "jno", "jb", "jnb", "jz", "jnz", "jbe", "jnbe", "js", "jns", "jp", "jnp",
    "jl", "jnl", "jle", "jnle", "je", "jne", "jc", "jnc", "ja", "jae", "jg", "jge",
    // SSE (hand-written)
    "addps", "addpd", "addss", "addsd", "subps", "subpd", "subss", "subsd",
    "mulps", "mulpd", "mulss", "mulsd", "divps", "divpd", "divss", "divsd",
    "xorps", "xorpd", "andps", "andpd", "orps", "orpd",
    "sqrtps", "sqrtpd", "sqrtss", "sqrtsd",
    "movaps", "movups", "movapd", "movupd", "movdqa", "movdqu",
    "paddd", "psubd", "pxor", "pand", "por", "movd", "movq",
    "cvtsi2ss", "cvtsi2sd", "cvtss2sd", "cvtsd2ss",
    "comiss", "comisd", "ucomiss", "ucomisd",
    // AVX (hand-written)
    "vaddps", "vaddpd", "vaddss", "vaddsd", "vsubps", "vsubpd",
    "vmulps", "vmulpd", "vdivps", "vdivpd",
    "vxorps", "vxorpd", "vandps", "vandpd", "vorps", "vorpd",
    "vmovaps", "vmovups", "vmovapd", "vmovupd", "vmovdqa", "vmovdqu",
    "vpaddd", "vpsubd", "vpxor", "vpand", "vpor",
    // CMOVcc (hand-written)
    "cmovo", "cmovno", "cmovb", "cmovc", "cmovnae", "cmovae", "cmovnb", "cmovnc",
    "cmove", "cmovz", "cmovne", "cmovnz", "cmovbe", "cmovna", "cmova", "cmovnbe",
    "cmovs", "cmovns", "cmovp", "cmovpe", "cmovnp", "cmovpo", "cmovl", "cmovnge",
    "cmovge", "cmovnl", "cmovle", "cmovng", "cmovg", "cmovnle",
    // SETcc (hand-written)
    "seto", "setno", "setb", "setc", "setnae", "setae", "setnb", "setnc",
    "sete", "setz", "setne", "setnz", "setbe", "setna", "seta", "setnbe",
    "sets", "setns", "setp", "setpe", "setnp", "setpo", "setl", "setnge",
    "setge", "setnl", "setle", "setng", "setg", "setnle",
    // Bit operations (hand-written)
    "bsf", "bsr", "popcnt", "lzcnt", "tzcnt", "bt", "bts", "btr", "btc",
    // Rotate (hand-written)
    "rol", "ror", "rcl", "rcr",
    // Single-operand GPR (hand-written)
    "mul", "div", "idiv", "leave", "enter",
    // Flag and misc operations (hand-written)
    "clc", "stc", "cld", "std_", "cmc", "cli", "sti", "sahf", "lahf",
    "hlt", "ud2", "cpuid", "rdtsc", "rdtscp", "pause", "lock",
    "lfence", "mfence", "sfence", "emms", "cbw", "cwde", "cwd", "cdqe",
    "popf", "pushf", "stmxcsr", "ldmxcsr",
    // String operations (hand-written)
    "rep", "repe", "repz", "repne", "repnz",
    "lodsb", "lodsw", "lodsd", "lodsq",
    "stosb", "stosw", "stosd", "stosq",
    "movsb", "movsw", "movsd_string", "movsq",
    "scasb", "scasw", "scasd", "scasq",
    "cmpsb", "cmpsw", "cmpsq",
    // CMPXCHG / XADD (hand-written)
    "cmpxchg", "xadd",
    // VEX misc (hand-written)
    "vzeroall", "vzeroupper",
    // Non-temporal stores (hand-written)
    "movntps", "movntpd", "movntdq", "movnti",
    "vmovntps", "vmovntpd", "vmovntdq",
    // Partial register loads/stores (hand-written)
    "movhps", "movlps", "movhpd", "movlpd",
    "vmovhps", "vmovlps", "vmovhpd", "vmovlpd",
    // Extract scalar (hand-written)
    "pextrb", "pextrw", "pextrd", "pextrq", "extractps",
    "vpextrb", "vpextrw", "vpextrd", "vpextrq", "vextractps",
    // Insert scalar (hand-written)
    "pinsrb", "pinsrw", "pinsrd", "pinsrq", "insertps",
    "vpinsrb", "vpinsrw", "vpinsrd", "vpinsrq", "vinsertps",
    // Extract vector (hand-written)
    "vextractf128", "vextracti128",
    "vextractf32x4", "vextracti32x4", "vextractf64x2", "vextracti64x2",
    "vextractf32x8", "vextracti32x8", "vextractf64x4", "vextracti64x4",
    // Variable blend (hand-written)
    "blendvps", "blendvpd", "pblendvb",
    // Float conversion (hand-written)
    "vcvtps2ph",
    // SHLD/SHRD/BSWAP (hand-written)
    "shld", "shrd", "bswap",
    // MOVSS/MOVSD (hand-written — bidirectional)
    "movss", "movsd", "vmovss", "vmovsd",
    // MOVAPS/MOVUPS already hand-written above
    // Cache prefetch (hand-written)
    "prefetchnta", "prefetcht0", "prefetcht1", "prefetcht2",
    "clflush", "clflushopt",
    // MOVMSKPS/PD (hand-written)
    "movmskps", "movmskpd", "vmovmskps", "vmovmskpd", "vpmovmskb",
    // CVT scalar (hand-written)
    "cvttss2si", "cvttsd2si", "cvtss2si", "cvtsd2si",
    "cvtdq2ps", "cvtps2dq", "cvttps2dq",
    // PUNPCK/PACK/UNPACK (hand-written)
    "punpcklbw", "punpcklwd", "punpckldq", "punpcklqdq",
    "punpckhbw", "punpckhwd", "punpckhdq", "punpckhqdq",
    "packsswb", "packssdw", "packuswb", "packusdw",
    "unpcklps", "unpckhps", "unpcklpd", "unpckhpd",
    // PSLL/PSRL/PSRA immediate shifts (hand-written, suffixed _imm)
    "pslld_imm", "psllq_imm", "psrld_imm", "psrlq_imm", "psrad_imm",
    "psllw_imm", "psrlw_imm", "psraw_imm",
    // Opmask (k-register) instructions (hand-written)
    "kmovw", "kmovb", "kmovd", "kmovq",
    "kandw", "kandb", "kandd", "kandq",
    "kandnw", "kandnb", "kandnd", "kandnq",
    "korw", "korb", "kord", "korq",
    "kxorw", "kxorb", "kxord", "kxorq",
    "kxnorw", "kxnorb", "kxnord", "kxnorq",
    "kaddw", "kaddb", "kaddd", "kaddq",
    "knotw", "knotb", "knotd", "knotq",
    "kortestw", "kortestb", "kortestd", "kortestq",
    "ktestw", "ktestb", "ktestd", "ktestq",
    "kunpckbw", "kunpckwd", "kunpckdq",
    "kshiftlw", "kshiftlb", "kshiftld", "kshiftlq",
    "kshiftrw", "kshiftrb", "kshiftrd", "kshiftrq",
    // x87 FPU instructions (hand-written)
    "fld", "fst", "fstp", "fild", "fist", "fistp", "fisttp",
    "fadd", "fsub", "fsubr", "fmul", "fdiv", "fdivr",
    "faddp", "fsubp", "fsubrp", "fmulp", "fdivp", "fdivrp",
    "fcom", "fcomp", "fcompp", "fucom", "fucomp", "fucompp",
    "fucomi", "fucomip", "fcomi", "fcomip",
    "fchs", "fabs", "fsqrt", "fsin", "fcos", "fptan", "fpatan",
    "frndint", "fscale", "f2xm1", "fyl2x", "fyl2xp1",
    "fprem", "fprem1", "fxtract", "ftst", "fxam",
    "fxch", "fldz", "fld1", "fldpi", "fldl2t", "fldl2e", "fldlg2", "fldln2",
    "fwait", "finit", "fninit", "fldcw", "fnstcw", "fstcw",
    "fnstsw", "fstsw", "fclex", "fnclex", "fnop", "fdecstp", "fincstp", "ffree",
    "fiadd", "fisub", "fimul", "fidiv", "ficom", "ficomp",
    "fcmovb", "fcmove", "fcmovbe", "fcmovu",
    "fcmovnb", "fcmovne", "fcmovnbe", "fcmovnu",
    // AMX tile instructions (hand-written)
    "tilerelease", "tilezero",
    "tdpbssd", "tdpbsud", "tdpbusd", "tdpbuud",
    "tdpbf16ps", "tdpfp16ps",
    "tileloadd", "tileloaddt1", "tilestored",
    "ldtilecfg", "sttilecfg",
];

/// Rust keywords and identifiers that need an underscore suffix.
const RUST_KEYWORDS: &[&str] = &["and", "or", "not", "xor", "in", "loop", "mod", "ref", "type", "yield"];

/// Sanitize a mnemonic name for use as a Rust identifier.
fn sanitize_name(name: &str) -> String {
    if RUST_KEYWORDS.contains(&name) {
        format!("{}_", name)
    } else {
        name.to_string()
    }
}

/// Format TypeFlags as a Rust expression.
fn format_flags(flags: u64) -> String {
    if flags == 0 {
        return "TypeFlags::NONE".to_string();
    }

    // Map each known bit to its constant name
    let flag_defs: &[(u64, &str)] = &[
        (1 << 3, "TypeFlags::T_N_VL"),
        (1 << 4, "TypeFlags::T_APX"),
        (1 << 5, "TypeFlags::T_66"),
        (1 << 6, "TypeFlags::T_F3"),
        (1 << 7, "TypeFlags::T_ER_R"),
        (1 << 8, "TypeFlags::T_0F"),
        (1 << 9, "TypeFlags::T_0F38"),
        (1 << 10, "TypeFlags::T_0F3A"),
        (1 << 11, "TypeFlags::T_MAP5"),
        (1 << 12, "TypeFlags::T_L1"),
        (1 << 13, "TypeFlags::T_W0"),
        (1 << 14, "TypeFlags::T_W1"),
        (1 << 16, "TypeFlags::T_EW1"),
        (1 << 17, "TypeFlags::T_YMM"),
        (1 << 18, "TypeFlags::T_EVEX"),
        (1 << 19, "TypeFlags::T_ER_X"),
        (1 << 20, "TypeFlags::T_ER_Y"),
        (1 << 21, "TypeFlags::T_ER_Z"),
        (1 << 22, "TypeFlags::T_SAE_X"),
        (1 << 23, "TypeFlags::T_SAE_Y"),
        (1 << 24, "TypeFlags::T_SAE_Z"),
        (1 << 25, "TypeFlags::T_MUST_EVEX"),
        (1 << 26, "TypeFlags::T_B32"),
        (1 << 27, "TypeFlags::T_B64"),
        (1 << 28, "TypeFlags::T_M_K"),
        (1 << 29, "TypeFlags::T_VSIB"),
        (1 << 30, "TypeFlags::T_MEM_EVEX"),
        (1 << 31, "TypeFlags::T_MAP6"),
        (1 << 32, "TypeFlags::T_NF"),
        (1 << 33, "TypeFlags::T_CODE1_IF1"),
        (1 << 35, "TypeFlags::T_ND1"),
        (1 << 36, "TypeFlags::T_ZU"),
        (1 << 37, "TypeFlags::T_F2"),
    ];

    let mut parts = Vec::new();
    let mut remaining = flags;

    // Handle low 3 bits (N value) specially
    let n_val = remaining & 7;
    if n_val > 0 {
        let n_name = match n_val {
            1 => "TypeFlags::T_N1",
            2 => "TypeFlags::T_N2",
            3 => "TypeFlags::T_N4",
            4 => "TypeFlags::T_N8",
            5 => "TypeFlags::T_N16",
            6 => "TypeFlags::T_N32",
            7 => "TypeFlags::T_DUP",
            _ => unreachable!(),
        };
        parts.push(n_name.to_string());
        remaining &= !7;
    }

    // Handle T_B16 = T_B32 | T_B64 specially
    let b32 = 1u64 << 26;
    let b64 = 1u64 << 27;
    if (remaining & (b32 | b64)) == (b32 | b64) {
        parts.push("TypeFlags::T_B16".to_string());
        remaining &= !(b32 | b64);
    }

    for &(bit, name) in flag_defs {
        if (remaining & bit) == bit {
            parts.push(name.to_string());
            remaining &= !bit;
        }
    }

    if remaining != 0 {
        parts.push(format!("TypeFlags(0x{:X})", remaining));
    }

    if parts.is_empty() {
        "TypeFlags::NONE".to_string()
    } else {
        parts.join(" | ")
    }
}

/// Generate Rust code for all instruction tables.
pub fn generate<W: Write>(out: &mut W, tables: &[&[Insn]]) -> std::io::Result<()> {
    writeln!(out, "// Auto-generated by build.rs — DO NOT EDIT")?;
    writeln!(out, "")?;
    writeln!(out, "use crate::encoding_flags::TypeFlags;")?;
    writeln!(out, "use crate::operand::{{Reg, RegMem}};")?;
    writeln!(out, "use crate::error::Result;")?;
    writeln!(out, "")?;
    writeln!(out, "impl crate::assembler::CodeAssembler {{")?;

    let mut generated_count = 0;
    let mut skipped_count = 0;
    let mut duplicate_count = 0;
    let mut seen_names = std::collections::HashSet::new();

    for table in tables {
        for insn in *table {
            let name = sanitize_name(insn.name);
            if HANDWRITTEN.contains(&insn.name) {
                skipped_count += 1;
                continue;
            }
            // Skip duplicate method names (e.g., vcmppd has both VEX 3-op and EVEX K-result forms)
            if !seen_names.insert(name.clone()) {
                duplicate_count += 1;
                continue;
            }
            let flags = format_flags(insn.type_flags);
            let opcode = insn.opcode;

            match insn.pattern {
                Pattern::Sse => {
                    writeln!(out, "    /// `{} xmm, xmm/m`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {{", name)?;
                    writeln!(out, "        self.buf.op_sse(&dst, &src.into(), {}, 0x{:02X}, None)", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::SseImm => {
                    writeln!(out, "    /// `{} xmm, xmm/m, imm8`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {{", name)?;
                    writeln!(out, "        self.buf.op_sse(&dst, &src.into(), {}, 0x{:02X}, Some(imm))", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::AvxXXXm => {
                    writeln!(out, "    /// `{} xmm/ymm/zmm, xmm/ymm/zmm, xmm/ymm/zmm/m`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {{", name)?;
                    writeln!(out, "        self.op_avx_x_x_xm(x1, x2, op, {}, 0x{:02X}, None)", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::AvxXXXmImm => {
                    writeln!(out, "    /// `{} xmm/ymm/zmm, xmm/ymm/zmm, xmm/ymm/zmm/m, imm8`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>, imm: u8) -> Result<()> {{", name)?;
                    writeln!(out, "        self.op_avx_x_x_xm(x1, x2, op, {}, 0x{:02X}, Some(imm))", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::AvxKXXm => {
                    writeln!(out, "    /// `{} k, xmm/ymm/zmm, xmm/ymm/zmm/m`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, k: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {{", name)?;
                    writeln!(out, "        self.op_avx_k_x_xm(k, x2, op, {}, 0x{:02X}, None)", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::AvxKXXmImm => {
                    writeln!(out, "    /// `{} k, xmm/ymm/zmm, xmm/ymm/zmm/m, imm8`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, k: Reg, x2: Reg, op: impl Into<RegMem>, imm: u8) -> Result<()> {{", name)?;
                    writeln!(out, "        self.op_avx_k_x_xm(k, x2, op, {}, 0x{:02X}, Some(imm))", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::VexXXm => {
                    writeln!(out, "    /// `{} xmm/ymm/zmm, xmm/ymm/zmm/m`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {{", name)?;
                    writeln!(out, "        self.buf.op_vex(&dst, None, &src.into(), {}, 0x{:02X}, None)", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::VexXXmImm => {
                    writeln!(out, "    /// `{} xmm/ymm/zmm, xmm/ymm/zmm/m, imm8`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {{", name)?;
                    writeln!(out, "        self.buf.op_vex(&dst, None, &src.into(), {}, 0x{:02X}, Some(imm))", flags, opcode)?;
                    writeln!(out, "    }}")?;
                }
                Pattern::VexMov => {
                    let store_opcode = insn.store_opcode;
                    writeln!(out, "    /// `{} xmm/ymm/zmm, xmm/ymm/zmm/m` or `{0} m, xmm/ymm/zmm`", insn.name)?;
                    writeln!(out, "    pub fn {}(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {{", name)?;
                    writeln!(out, "        let dst = dst.into();")?;
                    writeln!(out, "        let src = src.into();")?;
                    writeln!(out, "        let type_ = {}; ", flags)?;
                    writeln!(out, "        match (&dst, &src) {{")?;
                    writeln!(out, "            (RegMem::Reg(d), _) => {{")?;
                    writeln!(out, "                self.buf.op_vex(d, None, &src, type_, 0x{:02X}, None)", opcode)?;
                    writeln!(out, "            }}")?;
                    writeln!(out, "            (RegMem::Mem(_), RegMem::Reg(s)) => {{")?;
                    writeln!(out, "                self.buf.op_vex(s, None, &dst, type_, 0x{:02X}, None)", store_opcode)?;
                    writeln!(out, "            }}")?;
                    writeln!(out, "            _ => Err(crate::error::Error::BadCombination),")?;
                    writeln!(out, "        }}")?;
                    writeln!(out, "    }}")?;
                }
            }
            writeln!(out)?;
            generated_count += 1;
        }
    }

    writeln!(out, "}}")?;
    writeln!(out)?;

    // Write a summary as a comment
    writeln!(out, "// Generated {} methods ({} skipped as hand-written, {} skipped as duplicate)",
             generated_count, skipped_count, duplicate_count)?;

    Ok(())
}
