/// Instruction table data structures for code generation.
///
/// Each instruction entry describes a mnemonic, its encoding flags,
/// opcode, and the dispatch pattern (which internal helper to call).

/// Dispatch pattern: which encoding helper the generated method calls.
#[derive(Clone, Copy, Debug)]
pub enum Pattern {
    /// SSE 2-operand: `fn name(dst: Reg, src: impl Into<RegMem>)`
    /// Calls `self.buf.op_sse(&dst, &src, flags, opcode, None)`
    Sse,
    /// SSE 2-operand with immediate: `fn name(dst: Reg, src: impl Into<RegMem>, imm: u8)`
    /// Calls `self.buf.op_sse(&dst, &src, flags, opcode, Some(imm))`
    SseImm,
    /// VEX/EVEX 3-operand: `fn name(x1: Reg, x2: Reg, op: impl Into<RegMem>)`
    /// Calls `self.op_avx_x_x_xm(x1, x2, op, flags, opcode, None)`
    AvxXXXm,
    /// VEX/EVEX 3-operand with immediate: `fn name(x1: Reg, x2: Reg, op: impl Into<RegMem>, imm: u8)`
    /// Calls `self.op_avx_x_x_xm(x1, x2, op, flags, opcode, Some(imm))`
    AvxXXXmImm,
    /// AVX-512 opmask result: `fn name(k: Reg, x2: Reg, op: impl Into<RegMem>)`
    /// Calls `self.op_avx_k_x_xm(k, x2, op, flags, opcode, None)`
    AvxKXXm,
    /// AVX-512 opmask with imm: `fn name(k: Reg, x2: Reg, op: impl Into<RegMem>, imm: u8)`
    /// Calls `self.op_avx_k_x_xm(k, x2, op, flags, opcode, Some(imm))`
    AvxKXXmImm,
    /// VEX/EVEX 2-operand (no vvvv): `fn name(dst: Reg, src: impl Into<RegMem>)`
    /// Calls `self.buf.op_vex(&dst, None, &src, flags, opcode, None)`
    VexXXm,
    /// VEX/EVEX 2-operand with imm: `fn name(dst: Reg, src: impl Into<RegMem>, imm: u8)`
    /// Calls `self.buf.op_vex(&dst, None, &src, flags, opcode, Some(imm))`
    VexXXmImm,
    /// VEX/EVEX move pattern (bidirectional): `fn name(dst: impl Into<RegMem>, src: impl Into<RegMem>)`
    /// load_code for reg←mem, store_code for mem←reg
    VexMov,
}

/// A single instruction entry for code generation.
#[derive(Clone, Debug)]
pub struct Insn {
    /// Mnemonic name (e.g., "vaddps")
    pub name: &'static str,
    /// TypeFlags as raw u64 (combined T_* constants)
    pub type_flags: u64,
    /// Primary opcode byte
    pub opcode: u8,
    /// Dispatch pattern
    pub pattern: Pattern,
    /// Store opcode for VexMov pattern (0 if not applicable)
    pub store_opcode: u8,
}

impl Insn {
    pub const fn sse(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::Sse, store_opcode: 0 }
    }
    pub const fn sse_imm(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::SseImm, store_opcode: 0 }
    }
    pub const fn avx(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::AvxXXXm, store_opcode: 0 }
    }
    pub const fn avx_imm(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::AvxXXXmImm, store_opcode: 0 }
    }
    pub const fn avx_k(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::AvxKXXm, store_opcode: 0 }
    }
    pub const fn avx_k_imm(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::AvxKXXmImm, store_opcode: 0 }
    }
    pub const fn vex_xm(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::VexXXm, store_opcode: 0 }
    }
    pub const fn vex_xm_imm(name: &'static str, flags: u64, opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode, pattern: Pattern::VexXXmImm, store_opcode: 0 }
    }
    pub const fn vex_mov(name: &'static str, flags: u64, load_opcode: u8, store_opcode: u8) -> Self {
        Self { name, type_flags: flags, opcode: load_opcode, pattern: Pattern::VexMov, store_opcode }
    }
}

pub mod instructions;
pub mod avx512;
pub mod codegen;
