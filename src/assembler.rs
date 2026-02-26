use crate::address::Address;
use crate::code_array::{AllocMode, CodeBuffer, LabelMode};
use crate::encoding_flags::TypeFlags;
use crate::error::{Error, Result};
use crate::label::{JmpLabel, JmpType, Label, LabelId, LabelManager};
use crate::operand::{Reg, RegMem, RegMemImm};

/// The main assembler struct. Users create an instance, emit instructions,
/// then call `ready()` to finalize and obtain executable code.
pub struct CodeAssembler {
    pub(crate) buf: CodeBuffer,
    label_mgr: LabelManager,
}

impl CodeAssembler {
    /// Create a new assembler with the given maximum code size.
    pub fn new(max_size: usize) -> Result<Self> {
        Ok(Self {
            buf: CodeBuffer::new(max_size, AllocMode::Alloc)?,
            label_mgr: LabelManager::new(),
        })
    }

    /// Create a new assembler with auto-growing buffer.
    pub fn new_auto_grow(initial_size: usize) -> Result<Self> {
        Ok(Self {
            buf: CodeBuffer::new(initial_size, AllocMode::AutoGrow)?,
            label_mgr: LabelManager::new(),
        })
    }

    // ─── Buffer access ─────────────────────────────────────────

    /// Get current code size.
    pub fn size(&self) -> usize { self.buf.size() }

    /// Get the generated code as a byte slice.
    pub fn code(&self) -> &[u8] { self.buf.as_slice() }

    /// Finalize the code: resolve labels, set memory protection to RX.
    pub fn ready(&mut self) -> Result<()> {
        if self.label_mgr.has_undef_labels() {
            return Err(Error::LabelIsNotFound);
        }
        if self.buf.alloc_mode() == AllocMode::AutoGrow {
            self.buf.calc_jmp_address()?;
        }
        self.buf.protect_rx()
    }

    /// Get a typed function pointer to the generated code.
    ///
    /// # Safety
    /// The caller must ensure the generated code matches the expected
    /// calling convention and function signature.
    pub unsafe fn get_code<F>(&self) -> F {
        self.buf.as_fn()
    }

    // ─── Raw byte emission ─────────────────────────────────────

    /// Emit a single byte.
    pub fn db(&mut self, v: u8) -> Result<()> { self.buf.db(v) }
    /// Emit a 16-bit word.
    pub fn dw(&mut self, v: u16) -> Result<()> { self.buf.dw(v) }
    /// Emit a 32-bit dword.
    pub fn dd(&mut self, v: u32) -> Result<()> { self.buf.dd(v) }
    /// Emit a 64-bit qword.
    pub fn dq(&mut self, v: u64) -> Result<()> { self.buf.dq(v) }

    // ─── Label management ──────────────────────────────────────

    /// Create a new anonymous label.
    pub fn create_label(&mut self) -> Label {
        self.label_mgr.create_label()
    }

    /// Bind a label to the current code position.
    pub fn bind(&mut self, label: &Label) -> Result<()> {
        let offset = self.buf.size();
        let id = label.id();
        let is_auto_grow = self.buf.alloc_mode() == AllocMode::AutoGrow;

        self.label_mgr.define_label(label, offset)?;
        let patches = self.label_mgr.resolve_label(id, offset, is_auto_grow)?;

        for (patch_offset, disp, size, mode) in patches {
            if is_auto_grow {
                self.buf.save(patch_offset, disp, size, mode);
            } else {
                self.buf.rewrite(patch_offset, disp, size as usize);
            }
        }
        Ok(())
    }

    /// Define a named label at the current position.
    pub fn named_label(&mut self, name: &str) -> Result<LabelId> {
        let offset = self.buf.size();
        self.label_mgr.define_named_label(name, offset)
    }

    /// Enter a local label scope.
    pub fn enter_local(&mut self) {
        self.label_mgr.enter_local();
    }

    /// Leave a local label scope.
    pub fn leave_local(&mut self) -> Result<()> {
        self.label_mgr.leave_local()
    }

    // ─── Internal helpers ──────────────────────────────────────

    fn put_label(&mut self, label: &Label, jmp_size: u8, relative: bool, disp: i64) -> Result<()> {
        let id = label.id();
        if let Some(offset) = self.label_mgr.get_offset(label) {
            if relative {
                let d = offset as i64 + disp - self.buf.size() as i64 - jmp_size as i64;
                if !(-2147483648..=2147483647).contains(&d) {
                    return Err(Error::OffsetIsTooBig);
                }
                self.buf.dd(d as u32)?;
            } else if self.buf.alloc_mode() == AllocMode::AutoGrow {
                self.buf.dd(0)?;
                self.buf.save(self.buf.size() - 4, offset as u64, 4, LabelMode::AddTop);
            } else {
                let addr = self.buf.top() as u64 + offset as u64;
                self.buf.dq(addr)?;
            }
        } else {
            // Forward reference
            self.buf.dd(0)?;
            let mode = if relative {
                LabelMode::AsIs
            } else if self.buf.alloc_mode() == AllocMode::AutoGrow {
                LabelMode::AddTop
            } else {
                LabelMode::Abs
            };
            self.label_mgr.add_undef(id, JmpLabel {
                end_of_jmp: self.buf.size(),
                jmp_size,
                mode,
                disp,
            });
        }
        Ok(())
    }

    /// Get immediate bit size for arithmetic operations.
    fn get_imm_bit(reg_bit: u16, imm: i64) -> u8 {
        if reg_bit == 8 { return 8; }
        if (-128..=127).contains(&imm) { return 8; }
        if reg_bit == 16 { return 16; }
        32
    }

    // ─── x86 Instructions (manually implemented) ───────────────

    /// `nop` — No operation.
    pub fn nop(&mut self) -> Result<()> {
        self.buf.db(0x90)
    }

    /// `ret` — Return from procedure.
    pub fn ret(&mut self) -> Result<()> {
        self.buf.db(0xC3)
    }

    /// `ret imm16` — Return and pop imm16 bytes from stack.
    pub fn ret_imm(&mut self, imm: u16) -> Result<()> {
        self.buf.db(0xC2)?;
        self.buf.dw(imm)
    }

    /// `push reg` — Push register onto stack.
    pub fn push(&mut self, reg: Reg) -> Result<()> {
        if reg.has_rex2() {
            let default = Reg::default();
            self.buf.emit_rex2(false, crate::encode::rex_rxb(3, false, &default, &reg, &default), &default, &reg, &default)?;
            self.buf.db(0x50 | (reg.get_idx() & 7))
        } else {
            let bit = reg.get_bit();
            if bit == 16 { self.buf.db(0x66)?; }
            if bit == 16 || bit == 64 {
                if reg.get_idx() >= 8 { self.buf.db(0x41)?; }
                self.buf.db(0x50 | (reg.get_idx() & 7))
            } else {
                Err(Error::BadCombination)
            }
        }
    }

    /// `push imm` — Push immediate onto stack.
    pub fn push_imm(&mut self, imm: i32) -> Result<()> {
        if (-128..=127).contains(&imm) {
            self.buf.db(0x6A)?;
            self.buf.db(imm as u8)
        } else {
            self.buf.db(0x68)?;
            self.buf.dd(imm as u32)
        }
    }

    /// `pop reg` — Pop from stack into register.
    pub fn pop(&mut self, reg: Reg) -> Result<()> {
        if reg.has_rex2() {
            let default = Reg::default();
            self.buf.emit_rex2(false, crate::encode::rex_rxb(3, false, &default, &reg, &default), &default, &reg, &default)?;
            self.buf.db(0x58 | (reg.get_idx() & 7))
        } else {
            let bit = reg.get_bit();
            if bit == 16 { self.buf.db(0x66)?; }
            if bit == 16 || bit == 64 {
                if reg.get_idx() >= 8 { self.buf.db(0x41)?; }
                self.buf.db(0x58 | (reg.get_idx() & 7))
            } else {
                Err(Error::BadCombination)
            }
        }
    }

    /// `mov dst, src` — Move data.
    pub fn mov(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (dst, src) {
            // mov reg, reg
            (RegMem::Reg(d), RegMemImm::Reg(s)) => {
                if d.get_bit() != s.get_bit() {
                    return Err(Error::BadSizeOfRegister);
                }
                let code = if d.is_bit(8) { 0x88u8 } else { 0x89u8 };
                self.buf.op_rr(&s, &d, TypeFlags::NONE, code)
            }
            // mov reg, imm
            (RegMem::Reg(d), RegMemImm::Imm(imm)) => {
                self.mov_reg_imm(&d, imm as u64)
            }
            // mov reg, mem
            (RegMem::Reg(d), RegMemImm::Mem(m)) => {
                let code = if d.is_bit(8) { 0x8Au8 } else { 0x8Bu8 };
                self.buf.op_mr(&m, &d, TypeFlags::NONE, code)
            }
            // mov mem, reg
            (RegMem::Mem(m), RegMemImm::Reg(s)) => {
                let code = if s.is_bit(8) { 0x88u8 } else { 0x89u8 };
                self.buf.op_mr(&m, &s, TypeFlags::NONE, code)
            }
            // mov mem, imm
            (RegMem::Mem(m), RegMemImm::Imm(imm)) => {
                let bit = m.get_bit();
                if bit == 0 { return Err(Error::MemSizeIsNotSpecified); }
                let code = if bit == 8 { 0xC6u8 } else { 0xC7u8 };
                self.buf.op_rext(&RegMem::Mem(m), 0, TypeFlags::NONE, code, if bit == 8 { 1 } else { (bit.min(32) / 8) as u8 })?;
                let imm_bytes = (bit.min(32) / 8) as usize;
                self.buf.db_n(imm as u64, imm_bytes)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// Internal: mov reg, imm with optimal encoding.
    fn mov_reg_imm(&mut self, reg: &Reg, imm: u64) -> Result<()> {
        let bit = reg.get_bit();
        let idx = reg.get_idx();

        if bit == 64 && (imm & !0xFFFFFFFFu64) == 0 {
            // Use 32-bit mov which zero-extends
            let r32 = Reg::gpr32(idx);
            let default = Reg::default();
            self.buf.emit_rex_for_reg_reg(&r32, &default, TypeFlags::NONE)?;
            self.buf.db(0xB8 | (idx & 7))?;
            self.buf.dd(imm as u32)?;
        } else if bit == 64 && crate::encode::is_in_int32(imm) {
            // Use sign-extending mov r/m64, imm32
            let default = Reg::default();
            self.buf.emit_rex_for_reg_reg(reg, &default, TypeFlags::NONE)?;
            self.buf.db(0xC7)?;
            self.buf.db(0xC0 | (idx & 7))?;
            self.buf.dd(imm as u32)?;
        } else {
            // Full-width immediate
            let default = Reg::default();
            self.buf.emit_rex_for_reg_reg(reg, &default, TypeFlags::NONE)?;
            let code = 0xB0u8 | (if bit == 8 { 0 } else { 8 }) | (idx & 7);
            self.buf.db(code)?;
            self.buf.db_n(imm, (bit / 8) as usize)?;
        }
        Ok(())
    }

    /// Generic arithmetic operation (add/or/adc/sbb/and/sub/xor/cmp).
    fn arith_op(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>, ext: u8) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let base_code = ext << 3;

        match (dst, src) {
            (RegMem::Reg(d), RegMemImm::Reg(s)) => {
                if d.get_bit() != s.get_bit() {
                    return Err(Error::BadSizeOfRegister);
                }
                let code = base_code | if d.is_bit(8) { 0 } else { 1 };
                self.buf.op_rr(&s, &d, TypeFlags::NONE, code)
            }
            (RegMem::Reg(d), RegMemImm::Imm(imm)) => {
                let imm_bit = Self::get_imm_bit(d.get_bit(), imm);
                // Special short form for eax/ax/al
                if d.get_idx() == 0 && (d.get_bit() == imm_bit as u16 || (d.is_bit(64) && imm_bit == 32)) {
                    let default = Reg::default();
                    self.buf.emit_rex_for_reg_reg(&d, &default, TypeFlags::NONE)?;
                    self.buf.db(base_code | 4 | if imm_bit == 8 { 0 } else { 1 })?;
                } else {
                    let tmp = if (imm_bit as u16) < d.get_bit().min(32) { 2u8 } else { 0 };
                    self.buf.op_rext(&RegMem::Reg(d), ext, TypeFlags::NONE, 0x80 | tmp, imm_bit / 8)?;
                }
                self.buf.db_n(imm as u64, (imm_bit / 8) as usize)
            }
            (RegMem::Reg(d), RegMemImm::Mem(m)) => {
                let code = base_code | if d.is_bit(8) { 2 } else { 3 };
                self.buf.op_mr(&m, &d, TypeFlags::NONE, code)
            }
            (RegMem::Mem(m), RegMemImm::Reg(s)) => {
                let code = base_code | if s.is_bit(8) { 0 } else { 1 };
                self.buf.op_mr(&m, &s, TypeFlags::NONE, code)
            }
            (RegMem::Mem(m), RegMemImm::Imm(imm)) => {
                let bit = m.get_bit();
                if bit == 0 { return Err(Error::MemSizeIsNotSpecified); }
                let imm_bit = Self::get_imm_bit(bit, imm);
                let tmp = if (imm_bit as u16) < bit.min(32) { 2u8 } else { 0 };
                self.buf.op_rext(&RegMem::Mem(m), ext, TypeFlags::NONE, 0x80 | tmp, imm_bit / 8)?;
                self.buf.db_n(imm as u64, (imm_bit / 8) as usize)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `add dst, src`
    pub fn add(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 0)
    }

    /// `or dst, src`
    pub fn or_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 1)
    }

    /// `adc dst, src`
    pub fn adc(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 2)
    }

    /// `sbb dst, src`
    pub fn sbb(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 3)
    }

    /// `and dst, src`
    pub fn and_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 4)
    }

    /// `sub dst, src`
    pub fn sub(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 5)
    }

    /// `xor dst, src`
    pub fn xor_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 6)
    }

    /// `cmp dst, src`
    pub fn cmp(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 7)
    }

    /// `lea dst, src` — Load effective address.
    pub fn lea(&mut self, dst: Reg, src: Address) -> Result<()> {
        if dst.is_bit(8) { return Err(Error::BadCombination); }
        self.buf.op_mr(&src, &dst, TypeFlags::NONE, 0x8D)
    }

    /// `test dst, src`
    pub fn test(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (dst, src) {
            (RegMem::Reg(d), RegMemImm::Reg(s)) => {
                let code = if d.is_bit(8) { 0x84u8 } else { 0x85u8 };
                self.buf.op_rr(&s, &d, TypeFlags::NONE, code)
            }
            (RegMem::Reg(d), RegMemImm::Imm(imm)) => {
                // test eax, imm → short form
                if d.get_idx() == 0 {
                    let default = Reg::default();
                    self.buf.emit_rex_for_reg_reg(&d, &default, TypeFlags::NONE)?;
                    let code = if d.is_bit(8) { 0xA8u8 } else { 0xA9u8 };
                    self.buf.db(code)?;
                } else {
                    self.buf.op_rext(&RegMem::Reg(d), 0, TypeFlags::NONE, 0xF6, if d.is_bit(8) { 1 } else { (d.get_bit().min(32) / 8) as u8 })?;
                }
                let n = if d.is_bit(8) { 1 } else { (d.get_bit().min(32) / 8) as usize };
                self.buf.db_n(imm as u64, n)
            }
            (RegMem::Mem(m), RegMemImm::Reg(s)) => {
                let code = if s.is_bit(8) { 0x84u8 } else { 0x85u8 };
                self.buf.op_mr(&m, &s, TypeFlags::NONE, code)
            }
            (RegMem::Mem(m), RegMemImm::Imm(imm)) => {
                let bit = m.get_bit();
                if bit == 0 { return Err(Error::MemSizeIsNotSpecified); }
                let n = if bit == 8 { 1usize } else { (bit.min(32) / 8) as usize };
                self.buf.op_rext(&RegMem::Mem(m), 0, TypeFlags::NONE, 0xF6, n as u8)?;
                self.buf.db_n(imm as u64, n)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `inc r/m`
    pub fn inc(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 0, TypeFlags::NONE, 0xFE, 0)
    }

    /// `dec r/m`
    pub fn dec(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 1, TypeFlags::NONE, 0xFE, 0)
    }

    /// `neg r/m`
    pub fn neg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 3, TypeFlags::NONE, 0xF6, 0)
    }

    /// `not r/m`
    pub fn not_(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 2, TypeFlags::NONE, 0xF6, 0)
    }

    // ─── Jump / Call ───────────────────────────────────────────

    /// `jmp label` — Jump to label.
    pub fn jmp(&mut self, label: &Label, jmp_type: JmpType) -> Result<()> {
        match jmp_type {
            JmpType::Short => {
                self.buf.db(0xEB)?;
                // Emit 1-byte placeholder
                if let Some(offset) = self.label_mgr.get_offset(label) {
                    let d = offset as i64 - self.buf.size() as i64 - 1;
                    if !(-128..=127).contains(&d) {
                        return Err(Error::LabelIsTooFar);
                    }
                    self.buf.db(d as u8)?;
                } else {
                    self.buf.db(0)?;
                    self.label_mgr.add_undef(label.id(), JmpLabel {
                        end_of_jmp: self.buf.size(),
                        jmp_size: 1,
                        mode: LabelMode::AsIs,
                        disp: 0,
                    });
                }
                Ok(())
            }
            JmpType::Near | JmpType::Auto => {
                self.buf.db(0xE9)?;
                self.put_label(label, 4, true, 0)
            }
        }
    }

    /// `jmp reg` — Jump to address in register.
    pub fn jmp_reg(&mut self, reg: Reg) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(reg), 4, TypeFlags::NONE, 0xFE, 0)
    }

    /// `call label` — Call subroutine.
    pub fn call(&mut self, label: &Label) -> Result<()> {
        self.buf.db(0xE8)?;
        self.put_label(label, 4, true, 0)
    }

    /// `call reg` — Call address in register.
    pub fn call_reg(&mut self, reg: Reg) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(reg), 2, TypeFlags::NONE, 0xFE, 0)
    }

    /// Conditional jump helper.
    fn jcc(&mut self, cc: u8, label: &Label, jmp_type: JmpType) -> Result<()> {
        match jmp_type {
            JmpType::Short => {
                self.buf.db(0x70 | cc)?;
                if let Some(offset) = self.label_mgr.get_offset(label) {
                    let d = offset as i64 - self.buf.size() as i64 - 1;
                    if !(-128..=127).contains(&d) {
                        return Err(Error::LabelIsTooFar);
                    }
                    self.buf.db(d as u8)?;
                } else {
                    self.buf.db(0)?;
                    self.label_mgr.add_undef(label.id(), JmpLabel {
                        end_of_jmp: self.buf.size(),
                        jmp_size: 1,
                        mode: LabelMode::AsIs,
                        disp: 0,
                    });
                }
                Ok(())
            }
            JmpType::Near | JmpType::Auto => {
                self.buf.db(0x0F)?;
                self.buf.db(0x80 | cc)?;
                self.put_label(label, 4, true, 0)
            }
        }
    }

    // Conditional jumps
    pub fn jo(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0, label, t) }
    pub fn jno(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(1, label, t) }
    pub fn jb(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(2, label, t) }
    pub fn jnb(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(3, label, t) }
    pub fn jz(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(4, label, t) }
    pub fn jnz(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(5, label, t) }
    pub fn jbe(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(6, label, t) }
    pub fn jnbe(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(7, label, t) }
    pub fn js(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(8, label, t) }
    pub fn jns(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(9, label, t) }
    pub fn jp(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xA, label, t) }
    pub fn jnp(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xB, label, t) }
    pub fn jl(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xC, label, t) }
    pub fn jnl(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xD, label, t) }
    pub fn jle(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xE, label, t) }
    pub fn jnle(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jcc(0xF, label, t) }

    // Aliases
    pub fn je(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jz(label, t) }
    pub fn jne(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnz(label, t) }
    pub fn jc(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jb(label, t) }
    pub fn jnc(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnb(label, t) }
    pub fn ja(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnbe(label, t) }
    pub fn jae(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnb(label, t) }
    pub fn jg(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnle(label, t) }
    pub fn jge(&mut self, label: &Label, t: JmpType) -> Result<()> { self.jnl(label, t) }

    /// `int3` — Software breakpoint.
    pub fn int3(&mut self) -> Result<()> {
        self.buf.db(0xCC)
    }

    /// `xchg dst, src` — Exchange register values.
    pub fn xchg(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if dst.get_bit() != src.get_bit() {
            return Err(Error::BadSizeOfRegister);
        }
        // Short form: xchg eax, reg or xchg reg, eax
        if !dst.is_bit(8) && (dst.get_idx() == 0 || src.get_idx() == 0) {
            let (reg, _) = if dst.get_idx() == 0 { (src, dst) } else { (dst, src) };
            let default = Reg::default();
            self.buf.emit_rex_for_reg_reg(&reg, &default, TypeFlags::NONE)?;
            self.buf.db(0x90 | (reg.get_idx() & 7))
        } else {
            let code = if dst.is_bit(8) { 0x86u8 } else { 0x87u8 };
            self.buf.op_rr(&dst, &src, TypeFlags::NONE, code)
        }
    }

    /// `movzx dst, src` — Move with zero-extend.
    pub fn movzx(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        let src_bit = src.get_bit();
        if src_bit >= 32 { return Err(Error::BadCombination); }
        if dst.get_bit() <= src_bit { return Err(Error::BadCombination); }
        let w = if src_bit == 16 { 1u8 } else { 0 };
        match src {
            RegMem::Reg(s) => {
                self.buf.op_rr(&dst, &s, TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0xB6 | w)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(&m, &dst, TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0xB6 | w)
            }
        }
    }

    /// `movsx dst, src` — Move with sign-extend.
    pub fn movsx(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        let src_bit = src.get_bit();
        if src_bit >= 32 { return Err(Error::BadCombination); }
        if dst.get_bit() <= src_bit { return Err(Error::BadCombination); }
        let w = if src_bit == 16 { 1u8 } else { 0 };
        match src {
            RegMem::Reg(s) => {
                self.buf.op_rr(&dst, &s, TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0xBE | w)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(&m, &dst, TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0xBE | w)
            }
        }
    }

    /// `movsxd dst, src` — Move with sign-extend (32→64).
    pub fn movsxd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        if !dst.is_bit(64) { return Err(Error::BadCombination); }
        let src = src.into();
        match src {
            RegMem::Reg(s) => {
                self.buf.op_rr(&dst, &s, TypeFlags::T_ALLOW_DIFF_SIZE, 0x63)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(&m, &dst, TypeFlags::T_ALLOW_DIFF_SIZE, 0x63)
            }
        }
    }

    /// `cdq` — Convert doubleword to quadword (sign-extend eax into edx:eax).
    pub fn cdq(&mut self) -> Result<()> { self.buf.db(0x99) }
    /// `cqo` — Convert quadword to double-quadword (sign-extend rax into rdx:rax).
    pub fn cqo(&mut self) -> Result<()> {
        self.buf.db(0x48)?; // REX.W
        self.buf.db(0x99)
    }

    /// `imul dst, src` — Signed multiply (2-operand form).
    pub fn imul(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match src {
            RegMem::Reg(s) => {
                self.buf.op_rr(&dst, &s, TypeFlags::T_0F, 0xAF)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(&m, &dst, TypeFlags::T_0F, 0xAF)
            }
        }
    }

    // ─── Shift operations ──────────────────────────────────────

    /// `shl r/m, imm`
    pub fn shl(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 4)
    }

    /// `shr r/m, imm`
    pub fn shr(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 5)
    }

    /// `sar r/m, imm`
    pub fn sar(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 7)
    }

    fn shift_op(&mut self, op: RegMem, imm: u8, ext: u8) -> Result<()> {
        let code = if imm == 1 { 0xD0u8 } else { 0xC0u8 };
        let bit = match &op {
            RegMem::Reg(r) => r.get_bit(),
            RegMem::Mem(m) => {
                if m.get_bit() == 0 { return Err(Error::MemSizeIsNotSpecified); }
                m.get_bit()
            }
        };
        let code = code | if bit == 8 { 0 } else { 1 };
        self.buf.op_rext(&op, ext, TypeFlags::NONE, code, if imm == 1 { 0 } else { 1 })?;
        if imm != 1 {
            self.buf.db(imm)?;
        }
        Ok(())
    }

    // ─── VEX/EVEX dispatch helpers ──────────────────────────────

    /// AVX 3-operand form: (dst, src1, src2) where src2 is reg or mem.
    /// If src2 is None-like, collapses to 2-operand form (dst, src1) with vvvv=dst.
    pub(crate) fn op_avx_x_x_xm(
        &mut self,
        x1: Reg,
        x2: Reg,
        op: impl Into<RegMem>,
        type_: TypeFlags,
        code: u8,
        imm8: Option<u8>,
    ) -> Result<()> {
        // Validate register combination
        let ok = (x1.is_xmm() && x2.is_xmm())
            || (type_.contains(TypeFlags::T_YMM) && (
                (x1.is_ymm() && x2.is_ymm()) || (x1.is_zmm() && x2.is_zmm())
            ));
        if !ok {
            return Err(Error::BadCombination);
        }
        let op = op.into();
        self.buf.op_vex(&x1, Some(&x2), &op, type_, code, imm8)
    }

    /// AVX-512 form with opmask: (k, xmm, xmm/m)
    pub(crate) fn op_avx_k_x_xm(
        &mut self,
        k: Reg,
        x2: Reg,
        op: impl Into<RegMem>,
        type_: TypeFlags,
        code: u8,
        imm8: Option<u8>,
    ) -> Result<()> {
        let op = op.into();
        if let RegMem::Reg(r) = &op {
            if x2.get_kind() as u16 != r.get_kind() as u16 {
                return Err(Error::BadCombination);
            }
        }
        self.buf.op_vex(&k, Some(&x2), &op, type_, code, imm8)
    }

    // ─── SSE Instructions ───────────────────────────────────────

    /// `addps xmm, xmm/m128`
    pub fn addps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x58, None)
    }

    /// `addpd xmm, xmm/m128`
    pub fn addpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x58, None)
    }

    /// `addss xmm, xmm/m32`
    pub fn addss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x58, None)
    }

    /// `addsd xmm, xmm/m64`
    pub fn addsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x58, None)
    }

    /// `subps xmm, xmm/m128`
    pub fn subps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5C, None)
    }

    /// `subpd xmm, xmm/m128`
    pub fn subpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x5C, None)
    }

    /// `subss xmm, xmm/m32`
    pub fn subss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x5C, None)
    }

    /// `subsd xmm, xmm/m64`
    pub fn subsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x5C, None)
    }

    /// `mulps xmm, xmm/m128`
    pub fn mulps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x59, None)
    }

    /// `mulpd xmm, xmm/m128`
    pub fn mulpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x59, None)
    }

    /// `mulss xmm, xmm/m32`
    pub fn mulss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x59, None)
    }

    /// `mulsd xmm, xmm/m64`
    pub fn mulsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x59, None)
    }

    /// `divps xmm, xmm/m128`
    pub fn divps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5E, None)
    }

    /// `divpd xmm, xmm/m128`
    pub fn divpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x5E, None)
    }

    /// `divss xmm, xmm/m32`
    pub fn divss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x5E, None)
    }

    /// `divsd xmm, xmm/m64`
    pub fn divsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x5E, None)
    }

    /// `xorps xmm, xmm/m128`
    pub fn xorps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x57, None)
    }

    /// `xorpd xmm, xmm/m128`
    pub fn xorpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x57, None)
    }

    /// `andps xmm, xmm/m128`
    pub fn andps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x54, None)
    }

    /// `andpd xmm, xmm/m128`
    pub fn andpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x54, None)
    }

    /// `orps xmm, xmm/m128`
    pub fn orps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x56, None)
    }

    /// `orpd xmm, xmm/m128`
    pub fn orpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x56, None)
    }

    /// `sqrtps xmm, xmm/m128`
    pub fn sqrtps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x51, None)
    }

    /// `sqrtpd xmm, xmm/m128`
    pub fn sqrtpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x51, None)
    }

    /// `sqrtss xmm, xmm/m32`
    pub fn sqrtss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x51, None)
    }

    /// `sqrtsd xmm, xmm/m64`
    pub fn sqrtsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x51, None)
    }

    /// `movaps xmm, xmm/m128`
    pub fn movaps(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_0F, 0x28, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_0F, 0x29)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movups xmm, xmm/m128`
    pub fn movups(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movapd xmm, xmm/m128`
    pub fn movapd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x28, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x29)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movupd xmm, xmm/m128`
    pub fn movupd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movdqa xmm, xmm/m128`
    pub fn movdqa(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x6F, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x7F)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movdqu xmm, xmm/m128`
    pub fn movdqu(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x6F, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0x7F)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `paddd xmm, xmm/m128`
    pub fn paddd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xFE, None)
    }

    /// `psubd xmm, xmm/m128`
    pub fn psubd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xFA, None)
    }

    /// `pxor xmm, xmm/m128`
    pub fn pxor(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xEF, None)
    }

    /// `pand xmm, xmm/m128`
    pub fn pand(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xDB, None)
    }

    /// `por xmm, xmm/m128`
    pub fn por(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xEB, None)
    }

    /// `movd xmm, r/m32` or `movd r/m32, xmm`
    pub fn movd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            // movd xmm, r/m32
            (RegMem::Reg(d), _) if d.is_xmm() => {
                self.buf.op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x6E, None)
            }
            // movd r/m32, xmm
            (_, RegMem::Reg(s)) if s.is_xmm() => {
                match &dst {
                    RegMem::Reg(d) => {
                        self.buf.op_rr(s, d, TypeFlags::T_66 | TypeFlags::T_0F, 0x7E)
                    }
                    RegMem::Mem(m) => {
                        self.buf.op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x7E)
                    }
                }
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movq xmm, xmm/m64` or `movq m64, xmm`
    pub fn movq(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            // movq xmm, xmm/m64
            (RegMem::Reg(d), _) if d.is_xmm() => {
                self.buf.op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x7E, None)
            }
            // movq m64, xmm
            (RegMem::Mem(m), RegMem::Reg(s)) if s.is_xmm() => {
                self.buf.op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0xD6)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `cvtsi2ss xmm, r/m32`
    pub fn cvtsi2ss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x2A, None)
    }

    /// `cvtsi2sd xmm, r/m32`
    pub fn cvtsi2sd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x2A, None)
    }

    /// `cvtss2sd xmm, xmm/m32`
    pub fn cvtss2sd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x5A, None)
    }

    /// `cvtsd2ss xmm, xmm/m64`
    pub fn cvtsd2ss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x5A, None)
    }

    /// `comiss xmm, xmm/m32`
    pub fn comiss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x2F, None)
    }

    /// `comisd xmm, xmm/m64`
    pub fn comisd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x2F, None)
    }

    /// `ucomiss xmm, xmm/m32`
    pub fn ucomiss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x2E, None)
    }

    /// `ucomisd xmm, xmm/m64`
    pub fn ucomisd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x2E, None)
    }

    // ─── AVX Instructions (VEX-encoded) ─────────────────────────

    /// `vaddps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vaddps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x58, None)
    }

    /// `vaddpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vaddpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x58, None)
    }

    /// `vaddss xmm, xmm, xmm/m32`
    pub fn vaddss(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_EW0 | TypeFlags::T_N4 | TypeFlags::T_ER_X, 0x58, None)
    }

    /// `vaddsd xmm, xmm, xmm/m64`
    pub fn vaddsd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_EW1 | TypeFlags::T_N8 | TypeFlags::T_ER_X, 0x58, None)
    }

    /// `vsubps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vsubps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x5C, None)
    }

    /// `vsubpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vsubpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x5C, None)
    }

    /// `vmulps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vmulps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x59, None)
    }

    /// `vmulpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vmulpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x59, None)
    }

    /// `vdivps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vdivps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x5E, None)
    }

    /// `vdivpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vdivpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x5E, None)
    }

    /// `vxorps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vxorps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x57, None)
    }

    /// `vxorpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vxorpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x57, None)
    }

    /// `vandps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vandps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x54, None)
    }

    /// `vandpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vandpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x54, None)
    }

    /// `vorps xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vorps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x56, None)
    }

    /// `vorpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vorpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_B64 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0x56, None)
    }

    /// `vmovaps xmm/ymm, xmm/ymm/m` or `vmovaps m, xmm/ymm`
    pub fn vmovaps(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_N16 | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x28, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x29, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovups xmm/ymm, xmm/ymm/m` or `vmovups m, xmm/ymm`
    pub fn vmovups(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_N16 | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x10, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x11, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovapd xmm/ymm, xmm/ymm/m` or `vmovapd m, xmm/ymm`
    pub fn vmovapd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_N16 | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x28, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x29, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovupd xmm/ymm, xmm/ymm/m` or `vmovupd m, xmm/ymm`
    pub fn vmovupd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW1 | TypeFlags::T_N16 | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x10, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x11, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovdqa xmm/ymm, xmm/ymm/m` or `vmovdqa m, xmm/ymm`
    pub fn vmovdqa(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x6F, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x7F, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovdqu xmm/ymm, xmm/ymm/m` or `vmovdqu m, xmm/ymm`
    pub fn vmovdqu(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_YMM;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_vex(d, None, &src, type_, 0x6F, None)
            }
            (RegMem::Mem(_), RegMem::Reg(s)) => {
                self.buf.op_vex(s, None, &dst, type_, 0x7F, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `vpaddd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vpaddd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0xFE, None)
    }

    /// `vpsubd xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vpsubd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_YMM | TypeFlags::T_EW0 | TypeFlags::T_B32 | TypeFlags::T_N16 | TypeFlags::T_N_VL, 0xFA, None)
    }

    /// `vpxor xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vpxor(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM, 0xEF, None)
    }

    /// `vpand xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vpand(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM, 0xDB, None)
    }

    /// `vpor xmm/ymm, xmm/ymm, xmm/ymm/m`
    pub fn vpor(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(x1, x2, op, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM, 0xEB, None)
    }
}
