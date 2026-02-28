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

    /// Create an assembler backed by a user-provided buffer.
    ///
    /// # Safety
    /// The buffer must remain valid for the lifetime of this CodeAssembler.
    /// The caller is responsible for setting memory protection (e.g., RX before execution).
    pub unsafe fn from_user_buf(buf: *mut u8, size: usize) -> Self {
        Self {
            buf: CodeBuffer::from_user_buf(buf, size),
            label_mgr: LabelManager::new(),
        }
    }

    // ─── Buffer access ─────────────────────────────────────────

    /// Get current code size.
    pub fn size(&self) -> usize { self.buf.size() }

    /// Get the generated code as a byte slice.
    pub fn code(&self) -> &[u8] { self.buf.as_slice() }

    /// Reset the code size to zero (for re-generating code in the same buffer).
    pub fn reset_size(&mut self) { self.buf.reset_size(); }

    /// Set the code size to a specific value.
    ///
    /// Used to reset the code pointer back to after the prelude when clearing
    /// the block cache while preserving the dispatcher stubs.
    pub fn set_size(&mut self, size: usize) { self.buf.set_size(size); }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize { self.buf.capacity() }

    /// Get pointer to start of code buffer.
    pub fn top(&self) -> *const u8 { self.buf.top() }

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

    /// Finalize the code for read+execute (resolve labels, set RX protection).
    /// Same as `ready()` — resolves labels and sets memory protection to RX.
    pub fn ready_re(&mut self) -> Result<()> {
        self.ready()
    }

    /// Set memory protection to Read+Execute.
    pub fn set_protect_mode_re(&mut self) -> Result<()> {
        self.buf.protect_rx()
    }

    /// Set memory protection to Read+Write.
    pub fn set_protect_mode_rw(&mut self) -> Result<()> {
        self.buf.protect_rw()
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

    /// Align the code to a boundary by emitting NOP bytes (0x90).
    pub fn align(&mut self, n: usize) -> Result<()> {
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(Error::BadParameter);
        }
        while !self.buf.size().is_multiple_of(n) {
            self.buf.db(0x90)?;
        }
        Ok(())
    }

    /// Embed an absolute label address (8 bytes) in the code stream.
    /// Used for building jump tables with absolute addresses.
    pub fn put_l(&mut self, label: &Label) -> Result<()> {
        self.put_label(label, 8, false, 0)
    }

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
            } else if mode == LabelMode::Abs {
                // Absolute address: top + label_offset
                let addr = self.buf.top() as u64 + offset as u64;
                self.buf.rewrite(patch_offset, addr, size as usize);
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
        let is_auto_grow = self.buf.alloc_mode() == AllocMode::AutoGrow;

        if let Some(offset) = self.label_mgr.get_offset(label) {
            if relative {
                let d = offset as i64 + disp - self.buf.size() as i64 - jmp_size as i64;
                if !(-2147483648..=2147483647).contains(&d) {
                    return Err(Error::OffsetIsTooBig);
                }
                self.buf.dd(d as u32)?;
            } else if is_auto_grow {
                // In AutoGrow mode, emit 8 bytes for absolute label addresses
                // The value will be resolved during calc_jmp_address
                self.buf.dq(0)?;
                self.buf.save(self.buf.size() - 8, offset as u64, 8, LabelMode::AddTop);
            } else {
                let addr = self.buf.top() as u64 + offset as u64;
                self.buf.dq(addr)?;
            }
        } else {
            // Forward reference
            if relative {
                self.buf.dd(0)?;
            } else {
                // Both AutoGrow and fixed mode use 8 bytes for absolute addresses
                self.buf.dq(0)?;
            }
            let mode = if relative {
                LabelMode::AsIs
            } else if is_auto_grow {
                LabelMode::AddTop
            } else {
                LabelMode::Abs
            };
            self.label_mgr.add_undef(id, JmpLabel {
                end_of_jmp: self.buf.size(),
                jmp_size: if relative { jmp_size } else { 8 },
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

    /// `lea dst, [rip + label]` — Load label address via RIP-relative addressing.
    ///
    /// This is equivalent to xbyak's `mov(reg, label)` which generates
    /// `lea reg, [rip + disp32]` for 64-bit code.
    pub fn lea_label(&mut self, dst: Reg, label: &Label) -> Result<()> {
        if !dst.is_bit(64) { return Err(Error::BadCombination); }
        // Emit: REX.W + 8D /r with RIP-relative ModRM (mod=0, rm=5)
        // REX.W prefix
        let rex = 0x48 | if dst.get_idx() >= 8 { 0x04 } else { 0 };
        self.buf.db(rex)?;
        // opcode for LEA
        self.buf.db(0x8D)?;
        // ModRM: mod=00, reg=dst, rm=101 (RIP-relative)
        self.buf.db(((dst.get_idx() & 7) << 3) | 5)?;
        // 32-bit displacement (relative to end of instruction)
        self.put_label(label, 4, true, 0)
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
    pub fn jmp_reg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 4, TypeFlags::NONE, 0xFE, 0)
    }

    /// `call label` — Call subroutine.
    pub fn call(&mut self, label: &Label) -> Result<()> {
        self.buf.db(0xE8)?;
        self.put_label(label, 4, true, 0)
    }

    /// `call reg` — Call address in register.
    pub fn call_reg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 2, TypeFlags::NONE, 0xFE, 0)
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

    /// `xchg op1, op2` — Exchange values. At most one operand may be memory.
    pub fn xchg(&mut self, op1: impl Into<RegMem>, op2: impl Into<RegMem>) -> Result<()> {
        let op1 = op1.into();
        let op2 = op2.into();
        // Normalize: ensure p1 is always the register.
        // Swap if p1 is memory, or if p2 is a non-8-bit register with idx=0 (eax/rax).
        let (p1, p2) = if op1.is_mem()
            || (op2.is_reg() && !op2.as_reg().unwrap().is_bit(8) && op2.as_reg().unwrap().get_idx() == 0)
        {
            (op2, op1)
        } else {
            (op1, op2)
        };
        // After normalization, p1 must be a register (mem-mem is invalid).
        let r1 = match p1 {
            RegMem::Reg(r) => r,
            RegMem::Mem(_) => return Err(Error::BadCombination),
        };
        // Size check
        if r1.get_bit() != p2.get_bit() {
            return Err(Error::BadSizeOfRegister);
        }
        // Short form (0x90+reg): both registers, p1 idx=0, not 8-bit,
        // and NOT xchg eax,eax (which would encode as NOP 0x90 in 64-bit mode).
        if let RegMem::Reg(r2) = &p2 {
            if r1.get_idx() == 0 && !r1.is_bit(8) && (r2.get_idx() != 0 || !r1.is_bit(32)) {
                let default = Reg::default();
                self.buf.emit_rex_for_reg_reg(r2, &default, TypeFlags::NONE)?;
                return self.buf.db(0x90 | (r2.get_idx() & 7));
            }
        }
        // General form: 0x86 for 8-bit, 0x87 for 16/32/64-bit
        let code = if r1.is_bit(8) { 0x86u8 } else { 0x87u8 };
        match p2 {
            RegMem::Reg(r2) => self.buf.op_rr(&r1, &r2, TypeFlags::NONE, code),
            RegMem::Mem(m) => self.buf.op_mr(&m, &r1, TypeFlags::NONE, code),
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

    fn shift_op_cl(&mut self, op: RegMem, ext: u8) -> Result<()> {
        self.buf.op_rext(&op, ext, TypeFlags::T_CODE1_IF1, 0xD2, 0)
    }

    /// `shl r/m, CL`
    pub fn shl_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shift_op_cl(op.into(), 4)
    }

    /// `shr r/m, CL`
    pub fn shr_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shift_op_cl(op.into(), 5)
    }

    /// `sar r/m, CL`
    pub fn sar_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shift_op_cl(op.into(), 7)
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
            // movq xmm, r64 — 66 REX.W 0F 6E /r
            (RegMem::Reg(d), RegMem::Reg(s)) if d.is_xmm() && s.is_reg_bit(64) => {
                self.buf.op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0x6E, None)
            }
            // movq xmm, xmm/m64 — F3 0F 7E /r
            (RegMem::Reg(d), _) if d.is_xmm() => {
                self.buf.op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x7E, None)
            }
            // movq r64, xmm — 66 REX.W 0F 7E /r
            (RegMem::Reg(d), RegMem::Reg(s)) if d.is_reg_bit(64) && s.is_xmm() => {
                self.buf.op_rr(s, d, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0x7E)
            }
            // movq m64, xmm — 66 0F D6 /r
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

    // ── CMOVcc ────────────────────────────────────────────────
    fn cmovcc(&mut self, cc: u8, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0x40 | cc),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0x40 | cc),
        }
    }
    pub fn cmovo(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(0, dst, src) }
    pub fn cmovno(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(1, dst, src) }
    pub fn cmovb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(2, dst, src) }
    pub fn cmovc(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(2, dst, src) }
    pub fn cmovnae(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(2, dst, src) }
    pub fn cmovae(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(3, dst, src) }
    pub fn cmovnb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(3, dst, src) }
    pub fn cmovnc(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(3, dst, src) }
    pub fn cmove(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(4, dst, src) }
    pub fn cmovz(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(4, dst, src) }
    pub fn cmovne(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(5, dst, src) }
    pub fn cmovnz(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(5, dst, src) }
    pub fn cmovbe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(6, dst, src) }
    pub fn cmovna(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(6, dst, src) }
    pub fn cmova(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(7, dst, src) }
    pub fn cmovnbe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(7, dst, src) }
    pub fn cmovs(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(8, dst, src) }
    pub fn cmovns(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(9, dst, src) }
    pub fn cmovp(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(10, dst, src) }
    pub fn cmovpe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(10, dst, src) }
    pub fn cmovnp(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(11, dst, src) }
    pub fn cmovpo(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(11, dst, src) }
    pub fn cmovl(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(12, dst, src) }
    pub fn cmovnge(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(12, dst, src) }
    pub fn cmovge(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(13, dst, src) }
    pub fn cmovnl(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(13, dst, src) }
    pub fn cmovle(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(14, dst, src) }
    pub fn cmovng(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(14, dst, src) }
    pub fn cmovg(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(15, dst, src) }
    pub fn cmovnle(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> { self.cmovcc(15, dst, src) }

    // ── SETcc ─────────────────────────────────────────────────
    fn setcc(&mut self, cc: u8, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        // 0F 90+cc /0
        self.buf.op_rext(&op, 0, TypeFlags::T_0F, 0x90 | cc, 0)
    }
    pub fn seto(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(0, op) }
    pub fn setno(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(1, op) }
    pub fn setb(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(2, op) }
    pub fn setc(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(2, op) }
    pub fn setnae(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(2, op) }
    pub fn setae(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(3, op) }
    pub fn setnb(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(3, op) }
    pub fn setnc(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(3, op) }
    pub fn sete(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(4, op) }
    pub fn setz(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(4, op) }
    pub fn setne(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(5, op) }
    pub fn setnz(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(5, op) }
    pub fn setbe(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(6, op) }
    pub fn setna(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(6, op) }
    pub fn seta(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(7, op) }
    pub fn setnbe(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(7, op) }
    pub fn sets(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(8, op) }
    pub fn setns(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(9, op) }
    pub fn setp(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(10, op) }
    pub fn setpe(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(10, op) }
    pub fn setnp(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(11, op) }
    pub fn setpo(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(11, op) }
    pub fn setl(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(12, op) }
    pub fn setnge(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(12, op) }
    pub fn setge(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(13, op) }
    pub fn setnl(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(13, op) }
    pub fn setle(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(14, op) }
    pub fn setng(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(14, op) }
    pub fn setg(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(15, op) }
    pub fn setnle(&mut self, op: impl Into<RegMem>) -> Result<()> { self.setcc(15, op) }

    // ── Bit operations ────────────────────────────────────────
    /// BSF - Bit Scan Forward: 0F BC /r
    pub fn bsf(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0xBC),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0xBC),
        }
    }
    /// BSR - Bit Scan Reverse: 0F BD /r
    pub fn bsr(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0xBD),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0xBD),
        }
    }
    /// POPCNT: F3 0F B8 /r
    pub fn popcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xB8),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xB8),
        }
    }
    /// LZCNT: F3 0F BD /r
    pub fn lzcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBD),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBD),
        }
    }
    /// TZCNT: F3 0F BC /r
    pub fn tzcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBC),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBC),
        }
    }
    /// `crc32 r32/r64, r/m8/16/32/64` — F2 0F 38 F0/F1 /r
    pub fn crc32(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        let src_bit = src.get_bit();
        let dst_bit = dst.get_bit();
        // r32 accepts 8/16/32 source; r64 accepts only 8/64 source
        if !((dst_bit == 32 && (src_bit == 8 || src_bit == 16 || src_bit == 32))
            || (dst_bit == 64 && (src_bit == 8 || src_bit == 64)))
        {
            return Err(Error::BadSizeOfRegister);
        }
        let code = if src_bit == 8 { 0xF0u8 } else { 0xF1u8 };
        let mut type_ = TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_ALLOW_DIFF_SIZE;
        if src_bit == 16 {
            type_ = type_ | TypeFlags::T_66;
        }
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, type_, code),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, type_, code),
        }
    }
    /// BT - Bit Test: 0F A3 /r
    pub fn bt(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xA3),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xA3),
        }
    }
    /// BTS - Bit Test and Set: 0F AB /r
    pub fn bts(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xAB),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xAB),
        }
    }
    /// BTR - Bit Test and Reset: 0F B3 /r
    pub fn btr(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xB3),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xB3),
        }
    }
    /// BTC - Bit Test and Complement: 0F BB /r
    pub fn btc(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xBB),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xBB),
        }
    }

    fn bt_imm_op(&mut self, op: RegMem, ext: u8, imm: u8) -> Result<()> {
        self.buf.op_rext(&op, ext, TypeFlags::T_0F, 0xBA, 1)?;
        self.buf.db(imm)
    }
    /// `bt r/m, imm8` — 0F BA /4 ib
    pub fn bt_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 4, imm)
    }
    /// `bts r/m, imm8` — 0F BA /5 ib
    pub fn bts_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 5, imm)
    }
    /// `btr r/m, imm8` — 0F BA /6 ib
    pub fn btr_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 6, imm)
    }
    /// `btc r/m, imm8` — 0F BA /7 ib
    pub fn btc_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 7, imm)
    }

    // ── Rotate ────────────────────────────────────────────────
    fn rotate_op(&mut self, op: &RegMem, ext: u8, count: u8) -> Result<()> {
        if count == 1 {
            self.buf.op_rext(op, ext, TypeFlags::T_CODE1_IF1, 0xD0, 0)
        } else {
            self.buf.op_rext(op, ext, TypeFlags::T_CODE1_IF1, 0xC0, 1)?;
            self.buf.db(count)
        }
    }
    pub fn rol(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 0, count)
    }
    pub fn ror(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 1, count)
    }
    pub fn rcl(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 2, count)
    }
    pub fn rcr(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 3, count)
    }

    fn rotate_op_cl(&mut self, op: &RegMem, ext: u8) -> Result<()> {
        self.buf.op_rext(op, ext, TypeFlags::T_CODE1_IF1, 0xD2, 0)
    }
    /// `rol r/m, CL`
    pub fn rol_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 0)
    }
    /// `ror r/m, CL`
    pub fn ror_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 1)
    }
    /// `rcl r/m, CL`
    pub fn rcl_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 2)
    }
    /// `rcr r/m, CL`
    pub fn rcr_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 3)
    }

    // ── Single-operand GPR ────────────────────────────────────
    /// MUL: F6 /4 (8-bit), F7 /4 (16/32/64)
    pub fn mul(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 4, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// DIV: F6 /6 (8-bit), F7 /6 (16/32/64)
    pub fn div(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 6, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// IDIV: F6 /7 (8-bit), F7 /7 (16/32/64)
    pub fn idiv(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 7, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// LEAVE: C9
    pub fn leave(&mut self) -> Result<()> {
        self.buf.db(0xC9)
    }
    /// ENTER: C8 iw ib
    pub fn enter(&mut self, alloc_size: u16, nesting: u8) -> Result<()> {
        self.buf.db(0xC8)?;
        self.buf.dw(alloc_size)?;
        self.buf.db(nesting)
    }

    // ── Flag operations ───────────────────────────────────────
    pub fn clc(&mut self) -> Result<()> { self.buf.db(0xF8) }
    pub fn stc(&mut self) -> Result<()> { self.buf.db(0xF9) }
    pub fn cld(&mut self) -> Result<()> { self.buf.db(0xFC) }
    pub fn std_(&mut self) -> Result<()> { self.buf.db(0xFD) }
    pub fn cmc(&mut self) -> Result<()> { self.buf.db(0xF5) }
    pub fn cli(&mut self) -> Result<()> { self.buf.db(0xFA) }
    pub fn sti(&mut self) -> Result<()> { self.buf.db(0xFB) }
    pub fn sahf(&mut self) -> Result<()> { self.buf.db(0x9E) }
    pub fn lahf(&mut self) -> Result<()> { self.buf.db(0x9F) }
    pub fn hlt(&mut self) -> Result<()> { self.buf.db(0xF4) }
    pub fn ud2(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0x0B) }
    pub fn cpuid(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0xA2) }
    pub fn rdtsc(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0x31) }
    pub fn rdtscp(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0x01)?; self.buf.db(0xF9) }
    pub fn pause(&mut self) -> Result<()> { self.buf.db(0xF3)?; self.buf.db(0x90) }
    pub fn lock(&mut self) -> Result<()> { self.buf.db(0xF0) }
    pub fn lfence(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0xAE)?; self.buf.db(0xE8) }
    pub fn mfence(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0xAE)?; self.buf.db(0xF0) }
    pub fn sfence(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0xAE)?; self.buf.db(0xF8) }
    pub fn emms(&mut self) -> Result<()> { self.buf.db(0x0F)?; self.buf.db(0x77) }
    pub fn cbw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0x98) }
    pub fn cwde(&mut self) -> Result<()> { self.buf.db(0x98) }
    pub fn cwd(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0x99) }
    pub fn cdqe(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0x98) }
    pub fn popf(&mut self) -> Result<()> { self.buf.db(0x9D) }
    pub fn pushf(&mut self) -> Result<()> { self.buf.db(0x9C) }
    pub fn stmxcsr(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &Reg::new(3, crate::operand::Kind::Reg, 32), TypeFlags::T_0F, 0xAE)
    }
    pub fn ldmxcsr(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &Reg::new(2, crate::operand::Kind::Reg, 32), TypeFlags::T_0F, 0xAE)
    }

    // ── String operations ─────────────────────────────────────
    pub fn rep(&mut self) -> Result<()> { self.buf.db(0xF3) }
    pub fn repe(&mut self) -> Result<()> { self.buf.db(0xF3) }
    pub fn repz(&mut self) -> Result<()> { self.buf.db(0xF3) }
    pub fn repne(&mut self) -> Result<()> { self.buf.db(0xF2) }
    pub fn repnz(&mut self) -> Result<()> { self.buf.db(0xF2) }
    pub fn lodsb(&mut self) -> Result<()> { self.buf.db(0xAC) }
    pub fn lodsw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0xAD) }
    pub fn lodsd(&mut self) -> Result<()> { self.buf.db(0xAD) }
    pub fn lodsq(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0xAD) }
    pub fn stosb(&mut self) -> Result<()> { self.buf.db(0xAA) }
    pub fn stosw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0xAB) }
    pub fn stosd(&mut self) -> Result<()> { self.buf.db(0xAB) }
    pub fn stosq(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0xAB) }
    pub fn movsb(&mut self) -> Result<()> { self.buf.db(0xA4) }
    pub fn movsw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0xA5) }
    pub fn movsd_string(&mut self) -> Result<()> { self.buf.db(0xA5) }
    pub fn movsq(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0xA5) }
    pub fn scasb(&mut self) -> Result<()> { self.buf.db(0xAE) }
    pub fn scasw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0xAF) }
    pub fn scasd(&mut self) -> Result<()> { self.buf.db(0xAF) }
    pub fn scasq(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0xAF) }
    pub fn cmpsb(&mut self) -> Result<()> { self.buf.db(0xA6) }
    pub fn cmpsw(&mut self) -> Result<()> { self.buf.db(0x66)?; self.buf.db(0xA7) }
    pub fn cmpsq(&mut self) -> Result<()> { self.buf.db(0x48)?; self.buf.db(0xA7) }

    // ── CMPXCHG ───────────────────────────────────────────────
    pub fn cmpxchg(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let code = if src.get_bit() == 8 { 0xB0u8 } else { 0xB1u8 };
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, code),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, code),
        }
    }
    /// XADD: 0F C0/C1 /r
    pub fn xadd(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let code = if src.get_bit() == 8 { 0xC0u8 } else { 0xC1u8 };
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, code),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, code),
        }
    }

    // ── VZEROALL / VZEROUPPER ─────────────────────────────────
    pub fn vzeroall(&mut self) -> Result<()> {
        self.buf.db(0xC5)?; self.buf.db(0xFC)?; self.buf.db(0x77)
    }
    pub fn vzeroupper(&mut self) -> Result<()> {
        self.buf.db(0xC5)?; self.buf.db(0xF8)?; self.buf.db(0x77)
    }

    // ── Non-temporal stores ───────────────────────────────────
    /// `movntps m128, xmm` — 0F 2B /r
    pub fn movntps(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x2B)
    }
    /// `movntpd m128, xmm` — 66 0F 2B /r
    pub fn movntpd(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x2B)
    }
    /// `movntdq m128, xmm` — 66 0F E7 /r
    pub fn movntdq(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0xE7)
    }
    /// `movnti m32/m64, r32/r64` — 0F C3 /r
    pub fn movnti(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0xC3)
    }
    /// `vmovntps m, xmm/ymm/zmm` — VEX.128/256.0F.WIG 2B /r
    pub fn vmovntps(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), t, 0x2B, None)
    }
    /// `vmovntpd m, xmm/ymm/zmm` — VEX.128/256.66.0F.WIG 2B /r
    pub fn vmovntpd(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), t, 0x2B, None)
    }
    /// `vmovntdq m, xmm/ymm/zmm` — VEX.128/256.66.0F.WIG E7 /r
    pub fn vmovntdq(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), t, 0xE7, None)
    }

    // ── Partial register loads/stores ─────────────────────────
    /// `movhps xmm, m64` — 0F 16 /r (load)
    pub fn movhps_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_0F, 0x16)
    }
    /// `movhps m64, xmm` — 0F 17 /r (store)
    pub fn movhps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x17)
    }
    /// `movlps xmm, m64` — 0F 12 /r (load)
    pub fn movlps_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_0F, 0x12)
    }
    /// `movlps m64, xmm` — 0F 13 /r (store)
    pub fn movlps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x13)
    }
    /// `movhpd xmm, m64` — 66 0F 16 /r (load)
    pub fn movhpd_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_66 | TypeFlags::T_0F, 0x16)
    }
    /// `movhpd m64, xmm` — 66 0F 17 /r (store)
    pub fn movhpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x17)
    }
    /// `movlpd xmm, m64` — 66 0F 12 /r (load)
    pub fn movlpd_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_66 | TypeFlags::T_0F, 0x12)
    }
    /// `movlpd m64, xmm` — 66 0F 13 /r (store)
    pub fn movlpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x13)
    }
    /// `vmovhps xmm, xmm, m64` — VEX 0F 16 /r (load, 3-op)
    pub fn vmovhps_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &RegMem::Mem(addr), TypeFlags::T_0F, 0x16, None)
    }
    /// `vmovhps m64, xmm` — VEX 0F 17 /r (store)
    pub fn vmovhps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F, 0x17, None)
    }
    /// `vmovlps xmm, xmm, m64` — VEX 0F 12 /r (load, 3-op)
    pub fn vmovlps_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &RegMem::Mem(addr), TypeFlags::T_0F, 0x12, None)
    }
    /// `vmovlps m64, xmm` — VEX 0F 13 /r (store)
    pub fn vmovlps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F, 0x13, None)
    }
    /// `vmovhpd xmm, xmm, m64` — VEX.66 0F 16 /r (load, 3-op)
    pub fn vmovhpd_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F, 0x16, None)
    }
    /// `vmovhpd m64, xmm` — VEX.66 0F 17 /r (store)
    pub fn vmovhpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F, 0x17, None)
    }
    /// `vmovlpd xmm, xmm, m64` — VEX.66 0F 12 /r (load, 3-op)
    pub fn vmovlpd_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F, 0x12, None)
    }
    /// `vmovlpd m64, xmm` — VEX.66 0F 13 /r (store)
    pub fn vmovlpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F, 0x13, None)
    }

    // ── SSE4.1 Extract scalar ─────────────────────────────────
    /// `pextrb r/m8, xmm, imm8` — 66 0F 3A 14 /r ib
    pub fn pextrb(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf.op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x14)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x14)?;
                self.buf.db(imm)
            }
        }
    }
    /// `pextrw r32, xmm, imm8` — 66 0F C5 /r ib (reg form)
    pub fn pextrw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_rr(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0xC5)?;
        self.buf.db(imm)
    }
    /// `pextrd r/m32, xmm, imm8` — 66 0F 3A 16 /r ib
    pub fn pextrd(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf.op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
        }
    }
    /// `pextrq r/m64, xmm, imm8` — 66 REX.W 0F 3A 16 /r ib
    pub fn pextrq(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        // Need REX.W: use a 64-bit dummy reg to force W bit
        match &dst {
            RegMem::Reg(d) => {
                let d64 = Reg::new(d.get_idx(), crate::operand::Kind::Reg, 64);
                self.buf.op_rr(&src, &d64, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                let src64 = Reg::new(src.get_idx(), crate::operand::Kind::Xmm, 64);
                self.buf.op_mr(m, &src64, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
        }
    }
    /// `extractps r/m32, xmm, imm8` — 66 0F 3A 17 /r ib
    pub fn extractps(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf.op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x17)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf.op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x17)?;
                self.buf.db(imm)
            }
        }
    }

    // ── SSE4.1 Insert scalar ──────────────────────────────────
    /// `pinsrb xmm, r/m8, imm8` — 66 0F 3A 20 /r ib
    pub fn pinsrb(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x20, Some(imm))
    }
    /// `pinsrw xmm, r/m16, imm8` — 66 0F C4 /r ib
    pub fn pinsrw(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xC4, Some(imm))
    }
    /// `pinsrd xmm, r/m32, imm8` — 66 0F 3A 22 /r ib
    pub fn pinsrd(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x22, Some(imm))
    }
    /// `pinsrq xmm, r/m64, imm8` — 66 REX.W 0F 3A 22 /r ib
    pub fn pinsrq(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        // Force REX.W by using 64-bit dst
        let dst64 = Reg::new(dst.get_idx(), crate::operand::Kind::Xmm, 64);
        self.buf.op_sse(&dst64, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x22, Some(imm))
    }
    /// `insertps xmm, xmm/m32, imm8` — 66 0F 3A 21 /r ib
    pub fn insertps(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x21, Some(imm))
    }

    // ── AVX Extract scalar (VEX) ──────────────────────────────
    /// `vpextrb r/m8, xmm, imm8` — VEX.128.66.0F3A 14 /r ib
    pub fn vpextrb(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(&src, None, &dst, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x14, Some(imm))
    }
    /// `vpextrw r32, xmm, imm8` — VEX.128.66.0F C5 /r ib
    pub fn vpextrw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F, 0xC5, Some(imm))
    }
    /// `vpextrd r/m32, xmm, imm8` — VEX.128.66.0F3A 16 /r ib
    pub fn vpextrd(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(&src, None, &dst, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16, Some(imm))
    }
    /// `vpextrq r/m64, xmm, imm8` — VEX.128.66.0F3A.W1 16 /r ib
    pub fn vpextrq(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(&src, None, &dst, TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x16, Some(imm))
    }
    /// `vextractps r/m32, xmm, imm8` — VEX.128.66.0F3A 17 /r ib
    pub fn vextractps(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(&src, None, &dst, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x17, Some(imm))
    }

    // ── AVX Insert scalar (VEX 3-op) ─────────────────────────
    /// `vpinsrb xmm, xmm, r/m8, imm8` — VEX.128.66.0F3A 20 /r ib
    pub fn vpinsrb(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &src2.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x20, Some(imm))
    }
    /// `vpinsrw xmm, xmm, r/m16, imm8` — VEX.128.66.0F C4 /r ib
    pub fn vpinsrw(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &src2.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0xC4, Some(imm))
    }
    /// `vpinsrd xmm, xmm, r/m32, imm8` — VEX.128.66.0F3A 22 /r ib
    pub fn vpinsrd(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &src2.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x22, Some(imm))
    }
    /// `vpinsrq xmm, xmm, r/m64, imm8` — VEX.128.66.0F3A.W1 22 /r ib
    pub fn vpinsrq(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &src2.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x22, Some(imm))
    }
    /// `vinsertps xmm, xmm, xmm/m32, imm8` — VEX.128.66.0F3A 21 /r ib
    pub fn vinsertps(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &src2.into(), TypeFlags::T_66 | TypeFlags::T_0F3A, 0x21, Some(imm))
    }

    // ── AVX Extract vector (VEX) ──────────────────────────────
    /// `vextractf128 xmm/m128, ymm, imm8` — VEX.256.66.0F3A 19 /r ib
    pub fn vextractf128(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM, 0x19, Some(imm))
    }
    /// `vextracti128 xmm/m128, ymm, imm8` — VEX.256.66.0F3A 39 /r ib
    pub fn vextracti128(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM, 0x39, Some(imm))
    }
    /// `vextractf32x4 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W0 19 /r ib
    pub fn vextractf32x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_N16, 0x19, Some(imm))
    }
    /// `vextracti32x4 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W0 39 /r ib
    pub fn vextracti32x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_N16, 0x39, Some(imm))
    }
    /// `vextractf64x2 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W1 19 /r ib
    pub fn vextractf64x2(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_EW1 | TypeFlags::T_YMM | TypeFlags::T_N16, 0x19, Some(imm))
    }
    /// `vextracti64x2 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W1 39 /r ib
    pub fn vextracti64x2(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_EW1 | TypeFlags::T_YMM | TypeFlags::T_N16, 0x39, Some(imm))
    }
    /// `vextractf32x8 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W0 1B /r ib
    pub fn vextractf32x8(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_N32, 0x1B, Some(imm))
    }
    /// `vextracti32x8 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W0 3B /r ib
    pub fn vextracti32x8(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_N32, 0x3B, Some(imm))
    }
    /// `vextractf64x4 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W1 1B /r ib
    pub fn vextractf64x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_EW1 | TypeFlags::T_YMM | TypeFlags::T_N32, 0x1B, Some(imm))
    }
    /// `vextracti64x4 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W1 3B /r ib
    pub fn vextracti64x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&src, None, &dst.into(), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_MUST_EVEX | TypeFlags::T_EW1 | TypeFlags::T_YMM | TypeFlags::T_N32, 0x3B, Some(imm))
    }

    // ── SSE4.1 Variable blend ─────────────────────────────────
    /// `blendvps xmm, xmm/m128, <XMM0>` — 66 0F 38 14 /r (implicit XMM0)
    pub fn blendvps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F38, 0x14, None)
    }
    /// `blendvpd xmm, xmm/m128, <XMM0>` — 66 0F 38 15 /r (implicit XMM0)
    pub fn blendvpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F38, 0x15, None)
    }
    /// `pblendvb xmm, xmm/m128, <XMM0>` — 66 0F 38 10 /r (implicit XMM0)
    pub fn pblendvb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F38, 0x10, None)
    }

    // ── vcvtps2ph — float32 to float16 ───────────────────────
    /// `vcvtps2ph xmm/m, xmm/ymm/zmm, imm8` — VEX.128/256.66.0F3A.W0 1D /r ib
    pub fn vcvtps2ph(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_EVEX | TypeFlags::T_N8 | TypeFlags::T_N_VL;
        self.buf.op_vex(&src, None, &dst.into(), t, 0x1D, Some(imm))
    }

    // ── SHLD / SHRD ──────────────────────────────────────────
    /// `shld r/m, r, imm8` — 0F A4 /r ib
    pub fn shld(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xA4)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                let mut m = *m;
                m.imm_size = 1;
                self.buf.op_mr(&m, &src, TypeFlags::T_0F, 0xA4)?;
                self.buf.db(imm)
            }
        }
    }
    /// `shld r/m, r, CL` — 0F A5 /r
    pub fn shld_cl(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xA5),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xA5),
        }
    }
    /// `shrd r/m, r, imm8` — 0F AC /r ib
    pub fn shrd(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xAC)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                let mut m = *m;
                m.imm_size = 1;
                self.buf.op_mr(&m, &src, TypeFlags::T_0F, 0xAC)?;
                self.buf.db(imm)
            }
        }
    }
    /// `shrd r/m, r, CL` — 0F AD /r
    pub fn shrd_cl(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xAD),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xAD),
        }
    }

    // ── BSWAP ─────────────────────────────────────────────────
    /// `bswap r32/r64` — 0F C8+rd
    pub fn bswap(&mut self, reg: Reg) -> Result<()> {
        if reg.is_bit(64) {
            self.buf.db(0x48 | if reg.get_idx() >= 8 { 1 } else { 0 })?;
        } else if reg.get_idx() >= 8 {
            self.buf.db(0x41)?;
        }
        self.buf.db(0x0F)?;
        self.buf.db(0xC8 + (reg.get_idx() & 7))
    }

    // ── MOVSS / MOVSD (special: reg,reg uses different pattern than reg,mem) ─
    /// `movss xmm, xmm/m32` — F3 0F 10 /r
    pub fn movss(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }
    /// `movsd xmm, xmm/m64` — F2 0F 10 /r
    pub fn movsd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf.op_sse(d, &src, TypeFlags::T_F2 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf.op_mr(m, s, TypeFlags::T_F2 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }
    /// `vmovss xmm, xmm, xmm` or `vmovss xmm, m32` or `vmovss m32, xmm`
    pub fn vmovss(&mut self, dst: impl Into<RegMem>, src1: impl Into<RegMem>, src2: Option<Reg>) -> Result<()> {
        let dst = dst.into();
        let src1 = src1.into();
        match (&dst, &src1, src2) {
            // vmovss xmm, xmm, xmm — 3-operand reg form
            (RegMem::Reg(d), RegMem::Reg(s1), Some(s2)) => {
                self.buf.op_vex(d, Some(s1), &RegMem::Reg(s2), TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4, 0x10, None)
            }
            // vmovss xmm, m32 — load
            (RegMem::Reg(d), RegMem::Mem(_), None) => {
                self.buf.op_vex(d, None, &src1, TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4, 0x10, None)
            }
            // vmovss m32, xmm — store
            (RegMem::Mem(m), RegMem::Reg(s), None) => {
                self.buf.op_vex(s, None, &RegMem::Mem(*m), TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4, 0x11, None)
            }
            _ => Err(Error::BadCombination),
        }
    }
    /// `vmovsd xmm, xmm, xmm` or `vmovsd xmm, m64` or `vmovsd m64, xmm`
    pub fn vmovsd(&mut self, dst: impl Into<RegMem>, src1: impl Into<RegMem>, src2: Option<Reg>) -> Result<()> {
        let dst = dst.into();
        let src1 = src1.into();
        match (&dst, &src1, src2) {
            (RegMem::Reg(d), RegMem::Reg(s1), Some(s2)) => {
                self.buf.op_vex(d, Some(s1), &RegMem::Reg(s2), TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_EW1 | TypeFlags::T_N8, 0x10, None)
            }
            (RegMem::Reg(d), RegMem::Mem(_), None) => {
                self.buf.op_vex(d, None, &src1, TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_EW1 | TypeFlags::T_N8, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s), None) => {
                self.buf.op_vex(s, None, &RegMem::Mem(*m), TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_EW1 | TypeFlags::T_N8, 0x11, None)
            }
            _ => Err(Error::BadCombination),
        }
    }

    // ── Cache prefetch ────────────────────────────────────────
    /// `prefetchnta [m]` — 0F 18 /0
    pub fn prefetchnta(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_0F, 0x18)
    }
    /// `prefetcht0 [m]` — 0F 18 /1
    pub fn prefetcht0(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(1, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_0F, 0x18)
    }
    /// `prefetcht1 [m]` — 0F 18 /2
    pub fn prefetcht1(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(2, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_0F, 0x18)
    }
    /// `prefetcht2 [m]` — 0F 18 /3
    pub fn prefetcht2(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(3, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_0F, 0x18)
    }
    /// `clflush [m]` — 0F AE /7
    pub fn clflush(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(7, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_0F, 0xAE)
    }
    /// `clflushopt [m]` — 66 0F AE /7
    pub fn clflushopt(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(7, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(&addr, &r, TypeFlags::T_66 | TypeFlags::T_0F, 0xAE)
    }

    // ── MOVMSKPS / MOVMSKPD ──────────────────────────────────
    /// `movmskps r32, xmm` — 0F 50 /r
    pub fn movmskps(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_rr(&dst, &src, TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0x50)
    }
    /// `movmskpd r32, xmm` — 66 0F 50 /r
    pub fn movmskpd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_rr(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE, 0x50)
    }
    /// `vmovmskps r32, xmm/ymm` — VEX 0F 50 /r
    pub fn vmovmskps(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_0F | TypeFlags::T_YMM, 0x50, None)
    }
    /// `vmovmskpd r32, xmm/ymm` — VEX.66 0F 50 /r
    pub fn vmovmskpd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM, 0x50, None)
    }
    /// `vpmovmskb r32, xmm/ymm` — VEX.66 0F D7 /r
    pub fn vpmovmskb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM, 0xD7, None)
    }

    // ── CVTSS2SI / CVTSD2SI / CVTTSS2SI / CVTTSD2SI ─────────
    /// `cvttss2si r32/r64, xmm/m32` — F3 0F 2C /r
    pub fn cvttss2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x2C, None)
    }
    /// `cvttsd2si r32/r64, xmm/m64` — F2 0F 2C /r
    pub fn cvttsd2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x2C, None)
    }
    /// `cvtss2si r32/r64, xmm/m32` — F3 0F 2D /r
    pub fn cvtss2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x2D, None)
    }
    /// `cvtsd2si r32/r64, xmm/m64` — F2 0F 2D /r
    pub fn cvtsd2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F2 | TypeFlags::T_0F, 0x2D, None)
    }

    // ── CVTDQ2PS / CVTPS2DQ / CVTTPS2DQ ─────────────────────
    /// `cvtdq2ps xmm, xmm/m128` — 0F 5B /r
    pub fn cvtdq2ps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5B, None)
    }
    /// `cvtps2dq xmm, xmm/m128` — 66 0F 5B /r
    pub fn cvtps2dq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x5B, None)
    }
    /// `cvttps2dq xmm, xmm/m128` — F3 0F 5B /r
    pub fn cvttps2dq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_F3 | TypeFlags::T_0F, 0x5B, None)
    }

    // ── PUNPCK / PACK / UNPACK ────────────────────────────────
    /// `punpcklbw xmm, xmm/m128` — 66 0F 60 /r
    pub fn punpcklbw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x60, None)
    }
    /// `punpcklwd xmm, xmm/m128` — 66 0F 61 /r
    pub fn punpcklwd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x61, None)
    }
    /// `punpckldq xmm, xmm/m128` — 66 0F 62 /r
    pub fn punpckldq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x62, None)
    }
    /// `punpcklqdq xmm, xmm/m128` — 66 0F 6C /r
    pub fn punpcklqdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x6C, None)
    }
    /// `punpckhbw xmm, xmm/m128` — 66 0F 68 /r
    pub fn punpckhbw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x68, None)
    }
    /// `punpckhwd xmm, xmm/m128` — 66 0F 69 /r
    pub fn punpckhwd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x69, None)
    }
    /// `punpckhdq xmm, xmm/m128` — 66 0F 6A /r
    pub fn punpckhdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x6A, None)
    }
    /// `punpckhqdq xmm, xmm/m128` — 66 0F 6D /r
    pub fn punpckhqdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x6D, None)
    }
    /// `packsswb xmm, xmm/m128` — 66 0F 63 /r
    pub fn packsswb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x63, None)
    }
    /// `packssdw xmm, xmm/m128` — 66 0F 6B /r
    pub fn packssdw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x6B, None)
    }
    /// `packuswb xmm, xmm/m128` — 66 0F 67 /r
    pub fn packuswb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x67, None)
    }
    /// `packusdw xmm, xmm/m128` — 66 0F 38 2B /r
    pub fn packusdw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F38, 0x2B, None)
    }
    /// `unpcklps xmm, xmm/m128` — 0F 14 /r
    pub fn unpcklps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x14, None)
    }
    /// `unpckhps xmm, xmm/m128` — 0F 15 /r
    pub fn unpckhps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x15, None)
    }
    /// `unpcklpd xmm, xmm/m128` — 66 0F 14 /r
    pub fn unpcklpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x14, None)
    }
    /// `unpckhpd xmm, xmm/m128` — 66 0F 15 /r
    pub fn unpckhpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(&dst, &src.into(), TypeFlags::T_66 | TypeFlags::T_0F, 0x15, None)
    }

    // ── PSLL / PSRL / PSRA immediate shifts ─────────────────
    /// `pslld xmm, imm8` — 66 0F 72 /6 ib
    pub fn pslld_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 6, TypeFlags::T_66 | TypeFlags::T_0F, 0x72, 1)?;
        self.buf.db(imm)
    }
    /// `psllq xmm, imm8` — 66 0F 73 /6 ib
    pub fn psllq_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 6, TypeFlags::T_66 | TypeFlags::T_0F, 0x73, 1)?;
        self.buf.db(imm)
    }
    /// `psrld xmm, imm8` — 66 0F 72 /2 ib
    pub fn psrld_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 2, TypeFlags::T_66 | TypeFlags::T_0F, 0x72, 1)?;
        self.buf.db(imm)
    }
    /// `psrlq xmm, imm8` — 66 0F 73 /2 ib
    pub fn psrlq_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 2, TypeFlags::T_66 | TypeFlags::T_0F, 0x73, 1)?;
        self.buf.db(imm)
    }
    /// `psrad xmm, imm8` — 66 0F 72 /4 ib
    pub fn psrad_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 4, TypeFlags::T_66 | TypeFlags::T_0F, 0x72, 1)?;
        self.buf.db(imm)
    }
    /// `psllw xmm, imm8` — 66 0F 71 /6 ib
    pub fn psllw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 6, TypeFlags::T_66 | TypeFlags::T_0F, 0x71, 1)?;
        self.buf.db(imm)
    }
    /// `psrlw xmm, imm8` — 66 0F 71 /2 ib
    pub fn psrlw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 2, TypeFlags::T_66 | TypeFlags::T_0F, 0x71, 1)?;
        self.buf.db(imm)
    }
    /// `psraw xmm, imm8` — 66 0F 71 /4 ib
    pub fn psraw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 4, TypeFlags::T_66 | TypeFlags::T_0F, 0x71, 1)?;
        self.buf.db(imm)
    }
    /// `pslldq xmm, imm8` — 66 0F 73 /7 ib (shift left by bytes)
    pub fn pslldq(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 7, TypeFlags::T_66 | TypeFlags::T_0F, 0x73, 1)?;
        self.buf.db(imm)
    }
    /// `psrldq xmm, imm8` — 66 0F 73 /3 ib (shift right by bytes)
    pub fn psrldq(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(&RegMem::Reg(dst), 3, TypeFlags::T_66 | TypeFlags::T_0F, 0x73, 1)?;
        self.buf.db(imm)
    }

    // ═══════════════════════════════════════════════════════════
    // AVX-512 Opmask (k-register) instructions
    // ═══════════════════════════════════════════════════════════

    // ── KMOV ──────────────────────────────────────────────────
    // kmovw k, k/m16:  VEX.L0.0F.W0 90 /r
    // kmovb k, k/m8:   VEX.L0.66.0F.W0 90 /r
    // kmovd k, k/m32:  VEX.L0.66.0F.W1 90 /r
    // kmovq k, k/m64:  VEX.L0.0F.W1 90 /r

    /// Helper for kmov instructions: auto-detects GPR operands and selects
    /// the correct opcode (0x90 for k↔k/m, 0x92 for k←gpr, 0x93 for gpr←k)
    /// and prefix (which may differ for the GPR form, e.g. kmovd).
    fn kmov_dispatch(
        &mut self, dst: Reg, src: impl Into<RegMem>,
        km_type: TypeFlags, gpr_type: TypeFlags,
    ) -> Result<()> {
        let src = src.into();
        if dst.is_opmask() {
            if let RegMem::Reg(r) = &src {
                if !r.is_opmask() {
                    // k ← GPR: opcode 0x92
                    return self.buf.op_vex(&dst, None, &src, gpr_type, 0x92, None);
                }
            }
            // k ← k/m: opcode 0x90
            self.buf.op_vex(&dst, None, &src, km_type, 0x90, None)
        } else {
            // GPR ← k: opcode 0x93
            self.buf.op_vex(&dst, None, &src, gpr_type, 0x93, None)
        }
    }

    /// `kmovw` — VEX.L0.0F.W0 {90,92,93} /r — auto-detects k↔k/m vs GPR forms.
    pub fn kmovw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let t = TypeFlags::T_0F | TypeFlags::T_W0;
        self.kmov_dispatch(dst, src, t, t)
    }
    /// `kmovb` — VEX.L0.66.0F.W0 {90,92,93} /r — auto-detects k↔k/m vs GPR forms.
    pub fn kmovb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W0;
        self.kmov_dispatch(dst, src, t, t)
    }
    /// `kmovd` — auto-detects k↔k/m (VEX.66.0F.W1 90) vs GPR (VEX.F2.0F.W0 92/93).
    pub fn kmovd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.kmov_dispatch(
            dst, src,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W1,
            TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_W0,
        )
    }
    /// `kmovq` — auto-detects k↔k/m (VEX.0F.W1 90) vs GPR (VEX.F2.0F.W1 92/93).
    pub fn kmovq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.kmov_dispatch(
            dst, src,
            TypeFlags::T_0F | TypeFlags::T_W1,
            TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_W1,
        )
    }

    // Store forms: kmov m, k
    /// `kmovw m16, k` — VEX.L0.0F.W0 91 /r
    pub fn kmovw_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F | TypeFlags::T_W0, 0x91, None)
    }
    /// `kmovb m8, k` — VEX.L0.66.0F.W0 91 /r
    pub fn kmovb_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W0, 0x91, None)
    }
    /// `kmovd m32, k` — VEX.L0.66.0F.W1 91 /r
    pub fn kmovd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W1, 0x91, None)
    }
    /// `kmovq m64, k` — VEX.L0.0F.W1 91 /r
    pub fn kmovq_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F | TypeFlags::T_W1, 0x91, None)
    }

    // Explicit GPR ↔ k-register aliases (kept for backward compatibility)
    pub fn kmovw_from_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovw(dst, src) }
    pub fn kmovw_to_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovw(dst, src) }
    pub fn kmovb_from_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovb(dst, src) }
    pub fn kmovb_to_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovb(dst, src) }
    pub fn kmovd_from_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovd(dst, src) }
    pub fn kmovd_to_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovd(dst, src) }
    pub fn kmovq_from_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovq(dst, src) }
    pub fn kmovq_to_gpr(&mut self, dst: Reg, src: Reg) -> Result<()> { self.kmovq(dst, src) }

    // ── KAND / KOR / KXOR / KANDN / KXNOR ────────────────────
    // All: VEX.L1.0F.Wx 41-47 /r  (3-op: k, k, k)
    // W0=word, W1(66)=byte, W1(plain)=dword/qword etc.

    /// Helper for 3-operand k-register operations
    fn k_op3(&mut self, dst: Reg, src1: Reg, src2: Reg, type_: TypeFlags, code: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src1), &RegMem::Reg(src2), type_ | TypeFlags::T_L1 | TypeFlags::T_0F, code, None)
    }

    // KAND
    pub fn kandw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x41) }
    pub fn kandb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x41) }
    pub fn kandd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x41) }
    pub fn kandq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x41) }

    // KANDN
    pub fn kandnw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x42) }
    pub fn kandnb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x42) }
    pub fn kandnd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x42) }
    pub fn kandnq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x42) }

    // KOR
    pub fn korw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x45) }
    pub fn korb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x45) }
    pub fn kord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x45) }
    pub fn korq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x45) }

    // KXOR
    pub fn kxorw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x47) }
    pub fn kxorb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x47) }
    pub fn kxord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x47) }
    pub fn kxorq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x47) }

    // KXNOR
    pub fn kxnorw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x46) }
    pub fn kxnorb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x46) }
    pub fn kxnord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x46) }
    pub fn kxnorq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x46) }

    // KADD
    pub fn kaddw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x4A) }
    pub fn kaddb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x4A) }
    pub fn kaddd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x4A) }
    pub fn kaddq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x4A) }

    // KUNPCK
    pub fn kunpckbw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x4B) }
    pub fn kunpckwd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x4B) }
    pub fn kunpckdq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x4B) }

    // ── KNOT / KORTEST / KTEST ────────────────────────────────
    // 2-operand k-register ops: VEX.L0.0F.Wx 44/98/99 /r (k, k)
    fn k_op2(&mut self, dst: Reg, src: Reg, type_: TypeFlags, code: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), type_ | TypeFlags::T_0F, code, None)
    }

    // KNOT
    pub fn knotw(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W0, 0x44) }
    pub fn knotb(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x44) }
    pub fn knotd(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x44) }
    pub fn knotq(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W1, 0x44) }

    // KORTEST
    pub fn kortestw(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W0, 0x98) }
    pub fn kortestb(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x98) }
    pub fn kortestd(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x98) }
    pub fn kortestq(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W1, 0x98) }

    // KTEST
    pub fn ktestw(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W0, 0x99) }
    pub fn ktestb(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x99) }
    pub fn ktestd(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x99) }
    pub fn ktestq(&mut self, dst: Reg, src: Reg) -> Result<()> { self.k_op2(dst, src, TypeFlags::T_W1, 0x99) }

    // ── KSHIFT ────────────────────────────────────────────────
    // kshiftl/r: VEX.L0.66.0F3A.W1 32-33 /r ib (k, k, imm8)
    /// `kshiftlw k, k, imm8` — VEX.L0.66.0F3A.W1 32 /r ib
    pub fn kshiftlw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x32, Some(imm))
    }
    /// `kshiftlb k, k, imm8`
    pub fn kshiftlb(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0, 0x32, Some(imm))
    }
    /// `kshiftld k, k, imm8`
    pub fn kshiftld(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0, 0x33, Some(imm))
    }
    /// `kshiftlq k, k, imm8`
    pub fn kshiftlq(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x33, Some(imm))
    }
    /// `kshiftrw k, k, imm8` — VEX.L0.66.0F3A.W1 30 /r ib
    pub fn kshiftrw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x30, Some(imm))
    }
    /// `kshiftrb k, k, imm8`
    pub fn kshiftrb(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0, 0x30, Some(imm))
    }
    /// `kshiftrd k, k, imm8`
    pub fn kshiftrd(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0, 0x31, Some(imm))
    }
    /// `kshiftrq k, k, imm8`
    pub fn kshiftrq(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Reg(src), TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1, 0x31, Some(imm))
    }

    // ═══════════════════════════════════════════════════════════
    // x87 FPU instructions
    // ═══════════════════════════════════════════════════════════

    // x87 uses escape opcodes D8-DF. Register forms: escape + (C0+i).
    // Memory forms: escape + ModRM with /digit extension.

    /// Helper: x87 register-register op: escape_byte + (modrm_base + st_idx)
    fn fpu_st(&mut self, escape: u8, modrm_base: u8, st: Reg) -> Result<()> {
        self.buf.db(escape)?;
        self.buf.db(modrm_base + st.get_idx())
    }

    /// Helper: x87 memory op with extension digit
    fn fpu_mem(&mut self, escape: u8, ext: u8, addr: &Address) -> Result<()> {
        let r = Reg::new(ext, crate::operand::Kind::Reg, 32);
        self.buf.emit_rex_for_reg_mem(&r, addr, TypeFlags::NONE)?;
        self.buf.db(escape)?;
        self.buf.emit_addr(addr, r.get_idx())
    }

    // ── FLD / FST / FSTP ──────────────────────────────────────
    /// `fld st(i)` — D9 C0+i
    pub fn fld_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD9, 0xC0, src) }
    /// `fld m32fp` — D9 /0
    pub fn fld_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD9, 0, &addr) }
    /// `fld m64fp` — DD /0
    pub fn fld_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDD, 0, &addr) }
    /// `fld m80fp` — DB /5
    pub fn fld_m80(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 5, &addr) }
    /// `fst st(i)` — DD D0+i
    pub fn fst_st(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDD, 0xD0, dst) }
    /// `fst m32fp` — D9 /2
    pub fn fst_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD9, 2, &addr) }
    /// `fst m64fp` — DD /2
    pub fn fst_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDD, 2, &addr) }
    /// `fstp st(i)` — DD D8+i
    pub fn fstp_st(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDD, 0xD8, dst) }
    /// `fstp m32fp` — D9 /3
    pub fn fstp_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD9, 3, &addr) }
    /// `fstp m64fp` — DD /3
    pub fn fstp_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDD, 3, &addr) }
    /// `fstp m80fp` — DB /7
    pub fn fstp_m80(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 7, &addr) }

    // ── FILD / FIST / FISTP / FISTTP ──────────────────────────
    /// `fild m16int` — DF /0
    pub fn fild_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 0, &addr) }
    /// `fild m32int` — DB /0
    pub fn fild_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 0, &addr) }
    /// `fild m64int` — DF /5
    pub fn fild_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 5, &addr) }
    /// `fist m16int` — DF /2
    pub fn fist_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 2, &addr) }
    /// `fist m32int` — DB /2
    pub fn fist_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 2, &addr) }
    /// `fistp m16int` — DF /3
    pub fn fistp_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 3, &addr) }
    /// `fistp m32int` — DB /3
    pub fn fistp_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 3, &addr) }
    /// `fistp m64int` — DF /7
    pub fn fistp_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 7, &addr) }
    /// `fisttp m16int` — DF /1
    pub fn fisttp_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDF, 1, &addr) }
    /// `fisttp m32int` — DB /1
    pub fn fisttp_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDB, 1, &addr) }
    /// `fisttp m64int` — DD /1
    pub fn fisttp_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDD, 1, &addr) }

    // ── FADD / FSUB / FMUL / FDIV (register forms) ───────────
    /// `fadd st(0), st(i)` — D8 C0+i
    pub fn fadd_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xC0, src) }
    /// `fadd st(i), st(0)` — DC C0+i
    pub fn fadd_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xC0, dst) }
    /// `faddp st(i), st(0)` — DE C0+i
    pub fn faddp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xC0, dst) }
    /// `fadd m32fp` — D8 /0
    pub fn fadd_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD8, 0, &addr) }
    /// `fadd m64fp` — DC /0
    pub fn fadd_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDC, 0, &addr) }

    /// `fsub st(0), st(i)` — D8 E0+i
    pub fn fsub_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xE0, src) }
    /// `fsub st(i), st(0)` — DC E8+i
    pub fn fsub_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xE8, dst) }
    /// `fsubp st(i), st(0)` — DE E8+i
    pub fn fsubp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xE8, dst) }
    /// `fsubr st(0), st(i)` — D8 E8+i
    pub fn fsubr_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xE8, src) }
    /// `fsubr st(i), st(0)` — DC E0+i
    pub fn fsubr_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xE0, dst) }
    /// `fsubrp st(i), st(0)` — DE E0+i
    pub fn fsubrp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xE0, dst) }
    /// `fsub m32fp` — D8 /4
    pub fn fsub_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD8, 4, &addr) }
    /// `fsub m64fp` — DC /4
    pub fn fsub_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDC, 4, &addr) }

    /// `fmul st(0), st(i)` — D8 C8+i
    pub fn fmul_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xC8, src) }
    /// `fmul st(i), st(0)` — DC C8+i
    pub fn fmul_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xC8, dst) }
    /// `fmulp st(i), st(0)` — DE C8+i
    pub fn fmulp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xC8, dst) }
    /// `fmul m32fp` — D8 /1
    pub fn fmul_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD8, 1, &addr) }
    /// `fmul m64fp` — DC /1
    pub fn fmul_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDC, 1, &addr) }

    /// `fdiv st(0), st(i)` — D8 F0+i
    pub fn fdiv_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xF0, src) }
    /// `fdiv st(i), st(0)` — DC F8+i
    pub fn fdiv_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xF8, dst) }
    /// `fdivp st(i), st(0)` — DE F8+i
    pub fn fdivp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xF8, dst) }
    /// `fdivr st(0), st(i)` — D8 F8+i
    pub fn fdivr_st0_st(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xF8, src) }
    /// `fdivr st(i), st(0)` — DC F0+i
    pub fn fdivr_st_st0(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDC, 0xF0, dst) }
    /// `fdivrp st(i), st(0)` — DE F0+i
    pub fn fdivrp(&mut self, dst: Reg) -> Result<()> { self.fpu_st(0xDE, 0xF0, dst) }
    /// `fdiv m32fp` — D8 /6
    pub fn fdiv_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD8, 6, &addr) }
    /// `fdiv m64fp` — DC /6
    pub fn fdiv_m64(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDC, 6, &addr) }

    // ── FCOM / FCOMP / FCOMPP ─────────────────────────────────
    /// `fcom st(i)` — D8 D0+i
    pub fn fcom(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xD0, src) }
    /// `fcomp st(i)` — D8 D8+i
    pub fn fcomp(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xD8, 0xD8, src) }
    /// `fcompp` — DE D9
    pub fn fcompp(&mut self) -> Result<()> { self.buf.db(0xDE)?; self.buf.db(0xD9) }
    /// `fucom st(i)` — DD E0+i
    pub fn fucom(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDD, 0xE0, src) }
    /// `fucomp st(i)` — DD E8+i
    pub fn fucomp(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDD, 0xE8, src) }
    /// `fucompp` — DA E9
    pub fn fucompp(&mut self) -> Result<()> { self.buf.db(0xDA)?; self.buf.db(0xE9) }
    /// `fucomi st(0), st(i)` — DB E8+i
    pub fn fucomi(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xE8, src) }
    /// `fucomip st(0), st(i)` — DF E8+i
    pub fn fucomip(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDF, 0xE8, src) }
    /// `fcomi st(0), st(i)` — DB F0+i
    pub fn fcomi(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xF0, src) }
    /// `fcomip st(0), st(i)` — DF F0+i
    pub fn fcomip(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDF, 0xF0, src) }

    // ── Unary / miscellaneous FPU ─────────────────────────────
    /// `fchs` — D9 E0
    pub fn fchs(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE0) }
    /// `fabs` — D9 E1
    pub fn fabs(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE1) }
    /// `fsqrt` — D9 FA
    pub fn fsqrt(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xFA) }
    /// `fsin` — D9 FE
    pub fn fsin(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xFE) }
    /// `fcos` — D9 FF
    pub fn fcos(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xFF) }
    /// `fptan` — D9 F2
    pub fn fptan(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF2) }
    /// `fpatan` — D9 F3
    pub fn fpatan(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF3) }
    /// `frndint` — D9 FC
    pub fn frndint(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xFC) }
    /// `fscale` — D9 FD
    pub fn fscale(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xFD) }
    /// `f2xm1` — D9 F0
    pub fn f2xm1(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF0) }
    /// `fyl2x` — D9 F1
    pub fn fyl2x(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF1) }
    /// `fyl2xp1` — D9 F9
    pub fn fyl2xp1(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF9) }
    /// `fprem` — D9 F8
    pub fn fprem(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF8) }
    /// `fprem1` — D9 F5
    pub fn fprem1(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF5) }
    /// `fxtract` — D9 F4
    pub fn fxtract(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF4) }
    /// `ftst` — D9 E4
    pub fn ftst(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE4) }
    /// `fxam` — D9 E5
    pub fn fxam(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE5) }

    // ── FPU Exchange ──────────────────────────────────────────
    /// `fxch st(i)` — D9 C8+i
    pub fn fxch(&mut self, st: Reg) -> Result<()> { self.fpu_st(0xD9, 0xC8, st) }

    // ── FPU Constants ─────────────────────────────────────────
    /// `fldz` — D9 EE (push +0.0)
    pub fn fldz(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xEE) }
    /// `fld1` — D9 E8 (push +1.0)
    pub fn fld1(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE8) }
    /// `fldpi` — D9 EB (push pi)
    pub fn fldpi(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xEB) }
    /// `fldl2t` — D9 E9 (push log2(10))
    pub fn fldl2t(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xE9) }
    /// `fldl2e` — D9 EA (push log2(e))
    pub fn fldl2e(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xEA) }
    /// `fldlg2` — D9 EC (push log10(2))
    pub fn fldlg2(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xEC) }
    /// `fldln2` — D9 ED (push ln(2))
    pub fn fldln2(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xED) }

    // ── FPU Control ───────────────────────────────────────────
    /// `fwait` / `wait` — 9B
    pub fn fwait(&mut self) -> Result<()> { self.buf.db(0x9B) }
    /// `finit` — 9B DB E3
    pub fn finit(&mut self) -> Result<()> { self.buf.db(0x9B)?; self.buf.db(0xDB)?; self.buf.db(0xE3) }
    /// `fninit` — DB E3
    pub fn fninit(&mut self) -> Result<()> { self.buf.db(0xDB)?; self.buf.db(0xE3) }
    /// `fldcw m16` — D9 /5
    pub fn fldcw(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD9, 5, &addr) }
    /// `fnstcw m16` — D9 /7
    pub fn fnstcw(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xD9, 7, &addr) }
    /// `fstcw m16` — 9B D9 /7
    pub fn fstcw(&mut self, addr: Address) -> Result<()> { self.buf.db(0x9B)?; self.fpu_mem(0xD9, 7, &addr) }
    /// `fnstsw m16` — DD /7
    pub fn fnstsw(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDD, 7, &addr) }
    /// `fnstsw ax` — DF E0
    pub fn fnstsw_ax(&mut self) -> Result<()> { self.buf.db(0xDF)?; self.buf.db(0xE0) }
    /// `fstsw ax` — 9B DF E0
    pub fn fstsw_ax(&mut self) -> Result<()> { self.buf.db(0x9B)?; self.buf.db(0xDF)?; self.buf.db(0xE0) }
    /// `fclex` — 9B DB E2
    pub fn fclex(&mut self) -> Result<()> { self.buf.db(0x9B)?; self.buf.db(0xDB)?; self.buf.db(0xE2) }
    /// `fnclex` — DB E2
    pub fn fnclex(&mut self) -> Result<()> { self.buf.db(0xDB)?; self.buf.db(0xE2) }
    /// `fnop` — D9 D0
    pub fn fnop(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xD0) }
    /// `fdecstp` — D9 F6
    pub fn fdecstp(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF6) }
    /// `fincstp` — D9 F7
    pub fn fincstp(&mut self) -> Result<()> { self.buf.db(0xD9)?; self.buf.db(0xF7) }
    /// `ffree st(i)` — DD C0+i
    pub fn ffree(&mut self, st: Reg) -> Result<()> { self.fpu_st(0xDD, 0xC0, st) }

    // ── FIADD / FISUB / FIMUL / FIDIV (integer memory ops) ───
    /// `fiadd m16int` — DE /0
    pub fn fiadd_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 0, &addr) }
    /// `fiadd m32int` — DA /0
    pub fn fiadd_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 0, &addr) }
    /// `fisub m16int` — DE /4
    pub fn fisub_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 4, &addr) }
    /// `fisub m32int` — DA /4
    pub fn fisub_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 4, &addr) }
    /// `fimul m16int` — DE /1
    pub fn fimul_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 1, &addr) }
    /// `fimul m32int` — DA /1
    pub fn fimul_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 1, &addr) }
    /// `fidiv m16int` — DE /6
    pub fn fidiv_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 6, &addr) }
    /// `fidiv m32int` — DA /6
    pub fn fidiv_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 6, &addr) }
    /// `ficom m16int` — DE /2
    pub fn ficom_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 2, &addr) }
    /// `ficom m32int` — DA /2
    pub fn ficom_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 2, &addr) }
    /// `ficomp m16int` — DE /3
    pub fn ficomp_m16(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDE, 3, &addr) }
    /// `ficomp m32int` — DA /3
    pub fn ficomp_m32(&mut self, addr: Address) -> Result<()> { self.fpu_mem(0xDA, 3, &addr) }

    // ── FCMOV (conditional move) ──────────────────────────────
    /// `fcmovb st(0), st(i)` — DA C0+i
    pub fn fcmovb(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDA, 0xC0, src) }
    /// `fcmove st(0), st(i)` — DA C8+i
    pub fn fcmove(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDA, 0xC8, src) }
    /// `fcmovbe st(0), st(i)` — DA D0+i
    pub fn fcmovbe(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDA, 0xD0, src) }
    /// `fcmovu st(0), st(i)` — DA D8+i
    pub fn fcmovu(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDA, 0xD8, src) }
    /// `fcmovnb st(0), st(i)` — DB C0+i
    pub fn fcmovnb(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xC0, src) }
    /// `fcmovne st(0), st(i)` — DB C8+i
    pub fn fcmovne(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xC8, src) }
    /// `fcmovnbe st(0), st(i)` — DB D0+i
    pub fn fcmovnbe(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xD0, src) }
    /// `fcmovnu st(0), st(i)` — DB D8+i
    pub fn fcmovnu(&mut self, src: Reg) -> Result<()> { self.fpu_st(0xDB, 0xD8, src) }

    // ═══════════════════════════════════════════════════════════
    // AMX (Advanced Matrix Extensions) tile instructions
    // ═══════════════════════════════════════════════════════════

    // AMX tile arithmetic uses VEX.128.0F38 with tmm register operands.
    // The VEX vvvv field encodes the third tmm operand.

    /// `tilerelease` — VEX.128.NP.0F38.W0 49 C0
    pub fn tilerelease(&mut self) -> Result<()> {
        self.buf.db(0xC4)?;
        self.buf.db(0xE2)?; // R=1 X=1 B=1 map=0F38
        self.buf.db(0x78)?; // W=0 vvvv=1111 L=0 pp=00
        self.buf.db(0x49)?;
        self.buf.db(0xC0)
    }

    /// `tilezero tmm` — VEX.128.F2.0F38.W0 49 /r
    pub fn tilezero(&mut self, dst: Reg) -> Result<()> {
        let tmm0 = Reg::tmm(0);
        self.buf.op_vex(&dst, Some(&tmm0), &RegMem::Reg(tmm0), TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_W0, 0x49, None)
    }

    /// Helper for AMX 3-operand tile dot-product instructions
    /// `tdp* dst, src1, src2` — VEX.128.pp.0F38.W0 opcode /r
    /// xbyak operand order: reg=dst, vvvv=src2, rm=src1
    fn amx_tdp(&mut self, dst: Reg, src1: Reg, src2: Reg, type_: TypeFlags, code: u8) -> Result<()> {
        self.buf.op_vex(&dst, Some(&src2), &RegMem::Reg(src1), type_ | TypeFlags::T_0F38 | TypeFlags::T_W0, code, None)
    }

    /// `tdpbssd tmm, tmm, tmm` — VEX.128.F2.0F38.W0 5E /r
    pub fn tdpbssd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::T_F2, 0x5E) }
    /// `tdpbsud tmm, tmm, tmm` — VEX.128.F3.0F38.W0 5E /r
    pub fn tdpbsud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::T_F3, 0x5E) }
    /// `tdpbusd tmm, tmm, tmm` — VEX.128.66.0F38.W0 5E /r
    pub fn tdpbusd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::T_66, 0x5E) }
    /// `tdpbuud tmm, tmm, tmm` — VEX.128.NP.0F38.W0 5E /r
    pub fn tdpbuud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::NONE, 0x5E) }
    /// `tdpbf16ps tmm, tmm, tmm` — VEX.128.F3.0F38.W0 5C /r
    pub fn tdpbf16ps(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::T_F3, 0x5C) }
    /// `tdpfp16ps tmm, tmm, tmm` — VEX.128.F2.0F38.W0 5C /r
    pub fn tdpfp16ps(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> { self.amx_tdp(dst, src1, src2, TypeFlags::T_F2, 0x5C) }

    /// `tileloadd tmm, [base + index*stride]` — VEX.128.F2.0F38.W0 4B /r
    /// Uses SIB-like addressing with base and index*stride.
    pub fn tileloadd(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Mem(addr), TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_W0, 0x4B, None)
    }
    /// `tileloaddt1 tmm, [base + index*stride]` — VEX.128.66.0F38.W0 4B /r
    pub fn tileloaddt1(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(&dst, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F38 | TypeFlags::T_W0, 0x4B, None)
    }
    /// `tilestored [base + index*stride], tmm` — VEX.128.F3.0F38.W0 4B /r
    pub fn tilestored(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_F3 | TypeFlags::T_0F38 | TypeFlags::T_W0, 0x4B, None)
    }

    /// `ldtilecfg [m512]` — VEX.128.NP.0F38.W0 49 /0
    pub fn ldtilecfg(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_vex(&r, None, &RegMem::Mem(addr), TypeFlags::T_0F38 | TypeFlags::T_W0, 0x49, None)
    }
    /// `sttilecfg [m512]` — VEX.128.66.0F38.W0 49 /0
    pub fn sttilecfg(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_vex(&r, None, &RegMem::Mem(addr), TypeFlags::T_66 | TypeFlags::T_0F38 | TypeFlags::T_W0, 0x49, None)
    }
}
