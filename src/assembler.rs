use crate::address::Address;
use crate::code_array::{AllocMode, CodeBuffer, LabelMode};
use crate::encoding_flags::TypeFlags;
use crate::error::{Error, Result};
use crate::label::{JmpLabel, JmpType, Label, LabelId, LabelManager};
use crate::operand::{Reg, RegMem, RegMemImm, Segment};

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
    pub fn size(&self) -> usize {
        self.buf.size()
    }

    /// Get the generated code as a byte slice.
    pub fn code(&self) -> &[u8] {
        self.buf.as_slice()
    }

    /// Reset the code size to zero (for re-generating code in the same buffer).
    pub fn reset_size(&mut self) {
        self.buf.reset_size();
    }

    /// Reset emitted code and all labels, preserving the allocated buffer.
    ///
    /// If `ready()` made an owned buffer read-execute, restore read-write
    /// protection after resetting the state, matching Xbyak 7.39 ordering.
    pub fn reset(&mut self) -> Result<()> {
        self.buf.reset_size();
        self.label_mgr.reset();
        self.buf.restore_writable_after_reset()
    }

    /// Set the code size to a specific value.
    ///
    /// Used to reset the code pointer back to after the prelude when clearing
    /// the block cache while preserving the dispatcher stubs.
    pub fn set_size(&mut self, size: usize) {
        self.buf.set_size(size);
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Get pointer to start of code buffer.
    pub fn top(&self) -> *const u8 {
        self.buf.top()
    }

    /// Finalize the code: resolve labels, set memory protection to RX.
    #[inline]
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
    #[inline]
    pub fn ready_re(&mut self) -> Result<()> {
        self.ready()
    }

    /// Set memory protection to Read+Execute.
    #[inline]
    pub fn set_protect_mode_re(&mut self) -> Result<()> {
        self.buf.protect_rx()
    }

    /// Set memory protection to Read+Write.
    #[inline]
    pub fn set_protect_mode_rw(&mut self) -> Result<()> {
        self.buf.protect_rw()
    }

    /// Set memory protection to Read+Write+Execute.
    #[inline]
    pub fn set_protect_mode_rwe(&mut self) -> Result<()> {
        self.buf.protect_rwe()
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
    #[inline]
    pub fn db(&mut self, v: u8) -> Result<()> {
        self.buf.db(v)
    }
    /// Emit a 16-bit word.
    #[inline]
    pub fn dw(&mut self, v: u16) -> Result<()> {
        self.buf.dw(v)
    }
    /// Emit a 32-bit dword.
    #[inline]
    pub fn dd(&mut self, v: u32) -> Result<()> {
        self.buf.dd(v)
    }
    /// Emit a 64-bit qword.
    #[inline]
    pub fn dq(&mut self, v: u64) -> Result<()> {
        self.buf.dq(v)
    }

    /// Emit a segment-override prefix.
    ///
    /// Mirrors Xbyak `CodeGenerator::putSeg`.
    #[inline]
    pub fn put_seg(&mut self, segment: Segment) -> Result<()> {
        self.buf.db(segment.prefix())
    }

    /// Align the code to a boundary using Xbyak's preferred multi-byte NOPs.
    #[inline]
    pub fn align(&mut self, n: usize) -> Result<()> {
        self.align_with_nop_mode(n, 2)
    }

    /// Align the code to a boundary using Xbyak's `useMultiByteNop` mode.
    ///
    /// `0` emits only `0x90`, `1` uses the recommended sequences up to nine
    /// bytes, and `2` also uses the AMD Zen 4 sequences up to fifteen bytes.
    #[inline]
    pub fn align_with_nop_mode(&mut self, n: usize, use_multi_byte_nop: u8) -> Result<()> {
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(Error::BadAlign);
        }
        if self.buf.alloc_mode() == AllocMode::AutoGrow
            && !crate::platform::page_size().is_multiple_of(n)
        {
            return Err(Error::BadAlign);
        }
        let remain = self.buf.cur() as usize % n;
        if remain != 0 {
            self.nop_bytes(n - remain, use_multi_byte_nop)?;
        }
        Ok(())
    }

    /// Embed an absolute label address (8 bytes) in the code stream.
    /// Used for building jump tables with absolute addresses.
    #[inline]
    pub fn put_l(&mut self, label: &Label) -> Result<()> {
        self.put_label(label, 8, false, 0)
    }

    // ─── Label management ──────────────────────────────────────

    /// Create a new anonymous label.
    pub fn create_label(&mut self) -> Label {
        self.label_mgr.create_label()
    }

    /// Bind a label to the current code position.
    #[inline]
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
                self.buf.rewrite(patch_offset, addr, size as usize)?;
            } else {
                self.buf.rewrite(patch_offset, disp, size as usize)?;
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
    #[inline]
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
                self.buf
                    .save(self.buf.size() - 8, offset as u64, 8, LabelMode::AddTop);
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
            self.label_mgr.add_undef(
                id,
                JmpLabel {
                    end_of_jmp: self.buf.size(),
                    jmp_size: if relative { jmp_size } else { 8 },
                    mode,
                    disp,
                },
            );
        }
        Ok(())
    }

    /// Emit an instruction whose only form is an 8-bit relative label branch.
    /// This is the short-only path of Xbyak `opJmp` used by LOOP/JECXZ.
    fn short_label_jump(&mut self, label: &Label, opcode: u8) -> Result<()> {
        self.buf.db(opcode)?;
        if let Some(offset) = self.label_mgr.get_offset(label) {
            let displacement = offset as i64 - self.buf.size() as i64 - 1;
            if !(-128..=127).contains(&displacement) {
                return Err(Error::LabelIsTooFar);
            }
            self.buf.db(displacement as u8)
        } else {
            self.buf.db(0)?;
            self.label_mgr.add_undef(
                label.id(),
                JmpLabel {
                    end_of_jmp: self.buf.size(),
                    jmp_size: 1,
                    mode: LabelMode::AsIs,
                    disp: 0,
                },
            );
            Ok(())
        }
    }

    /// Xbyak `opInOut(const Reg&, const Reg&, uint8_t)`.
    fn op_in_out_reg(&mut self, accumulator: Reg, port: Reg, code: u8) -> Result<()> {
        if accumulator.get_idx() == 0 && port.get_idx() == 2 && port.is_bit(16) {
            return match accumulator.get_bit() {
                8 => self.buf.db(code),
                16 => {
                    self.buf.db(0x66)?;
                    self.buf.db(code + 1)
                }
                32 => self.buf.db(code + 1),
                _ => Err(Error::BadCombination),
            };
        }
        Err(Error::BadCombination)
    }

    /// Xbyak `opInOut(const Reg&, uint8_t, uint8_t)`.
    fn op_in_out_imm(&mut self, accumulator: Reg, code: u8, port: u8) -> Result<()> {
        if accumulator.get_idx() == 0 {
            return match accumulator.get_bit() {
                8 => {
                    self.buf.db(code)?;
                    self.buf.db(port)
                }
                16 => {
                    self.buf.db(0x66)?;
                    self.buf.db(code + 1)?;
                    self.buf.db(port)
                }
                32 => {
                    self.buf.db(code + 1)?;
                    self.buf.db(port)
                }
                _ => Err(Error::BadCombination),
            };
        }
        Err(Error::BadCombination)
    }

    /// Xbyak `isMMX_XMMorMEM`.
    fn is_mmx_xmm_or_mem(dst: Reg, src: &RegMem) -> bool {
        dst.is_mmx()
            && (matches!(src, RegMem::Mem(_)) || matches!(src, RegMem::Reg(reg) if reg.is_xmm()))
    }

    /// Xbyak `isXMM_MMXorMEM`.
    fn is_xmm_mmx_or_mem(dst: Reg, src: &RegMem) -> bool {
        dst.is_xmm()
            && (matches!(src, RegMem::Mem(_)) || matches!(src, RegMem::Reg(reg) if reg.is_mmx()))
    }

    /// Xbyak `isXMMorMMX_MEM`.
    fn is_matching_mmx_or_xmm_mem(dst: Reg, src: &RegMem) -> bool {
        (dst.is_mmx()
            && (matches!(src, RegMem::Mem(_)) || matches!(src, RegMem::Reg(reg) if reg.is_mmx())))
            || (dst.is_xmm()
                && (matches!(src, RegMem::Mem(_))
                    || matches!(src, RegMem::Reg(reg) if reg.is_xmm())))
    }

    /// Get immediate bit size for arithmetic operations.
    fn get_imm_bit(reg_bit: u16, imm: i64) -> u8 {
        if reg_bit == 8 {
            return 8;
        }
        if (-128..=127).contains(&imm) {
            return 8;
        }
        if reg_bit == 16 {
            return 16;
        }
        32
    }

    // ─── x86 Instructions (manually implemented) ───────────────

    /// `nop` — Emit Xbyak's default one-byte no-op.
    #[inline]
    pub fn nop(&mut self) -> Result<()> {
        self.nop_bytes(1, 2)
    }

    /// Emit `size` bytes using Xbyak's multi-byte NOP table.
    ///
    /// This is the Rust counterpart of Xbyak's overloaded
    /// `nop(size, useMultiByteNop)` method.
    pub fn nop_bytes(&mut self, mut size: usize, use_multi_byte_nop: u8) -> Result<()> {
        const NOP_TABLE: [&[u8]; 15] = [
            &[0x90],
            &[0x66, 0x90],
            &[0x0F, 0x1F, 0x00],
            &[0x0F, 0x1F, 0x40, 0x00],
            &[0x0F, 0x1F, 0x44, 0x00, 0x00],
            &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
            &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
            &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            &[0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
            &[
                0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            &[
                0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            &[
                0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            &[
                0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            &[
                0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00,
                0x00,
            ],
        ];

        if use_multi_byte_nop == 0 {
            while size != 0 {
                self.buf.db(0x90)?;
                size -= 1;
            }
            return Ok(());
        }

        let sequence_count = if use_multi_byte_nop == 2 { 15 } else { 9 };
        while size != 0 {
            let len = sequence_count.min(size);
            for &byte in NOP_TABLE[len - 1] {
                self.buf.db(byte)?;
            }
            size -= len;
        }
        Ok(())
    }

    /// `ret` — Return from procedure.
    #[inline]
    pub fn ret(&mut self) -> Result<()> {
        self.buf.db(0xC3)
    }

    /// `ret imm16` — Return and pop imm16 bytes from stack.
    #[inline]
    pub fn ret_imm(&mut self, imm: u16) -> Result<()> {
        if imm == 0 {
            return self.ret();
        }
        self.buf.db(0xC2)?;
        self.buf.dw(imm)
    }

    /// `retf` — Far return.
    #[inline]
    pub fn retf(&mut self) -> Result<()> {
        self.buf.db(0xCB)
    }

    /// `retf imm16` — Far return and pop imm16 bytes from the stack.
    #[inline]
    pub fn retf_imm(&mut self, imm: u16) -> Result<()> {
        if imm == 0 {
            return self.retf();
        }
        self.buf.db(0xCA)?;
        self.buf.dw(imm)
    }

    /// `push reg` — Push register onto stack.
    #[inline]
    pub fn push(&mut self, reg: Reg) -> Result<()> {
        if reg.has_rex2() {
            let default = Reg::default();
            self.buf.emit_rex2(
                false,
                crate::encode::rex_rxb(3, false, &default, &reg, &default),
                &default,
                &reg,
                &default,
            )?;
            self.buf.db(0x50 | (reg.get_idx() & 7))
        } else {
            let bit = reg.get_bit();
            if bit == 16 {
                self.buf.db(0x66)?;
            }
            if bit == 16 || bit == 64 {
                if reg.get_idx() >= 8 {
                    self.buf.db(0x41)?;
                }
                self.buf.db(0x50 | (reg.get_idx() & 7))
            } else {
                Err(Error::BadCombination)
            }
        }
    }

    /// `push imm` — Push immediate onto stack.
    #[inline]
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
    #[inline]
    pub fn pop(&mut self, reg: Reg) -> Result<()> {
        if reg.has_rex2() {
            let default = Reg::default();
            self.buf.emit_rex2(
                false,
                crate::encode::rex_rxb(3, false, &default, &reg, &default),
                &default,
                &reg,
                &default,
            )?;
            self.buf.db(0x58 | (reg.get_idx() & 7))
        } else {
            let bit = reg.get_bit();
            if bit == 16 {
                self.buf.db(0x66)?;
            }
            if bit == 16 || bit == 64 {
                if reg.get_idx() >= 8 {
                    self.buf.db(0x41)?;
                }
                self.buf.db(0x58 | (reg.get_idx() & 7))
            } else {
                Err(Error::BadCombination)
            }
        }
    }

    /// `push2 r64, r64` — APX paired push.
    #[inline]
    pub fn push2(&mut self, first: Reg, second: Reg) -> Result<()> {
        self.push_pop2(first, second, TypeFlags::T_W0, 0xFF, 6)
    }

    /// `push2p r64, r64` — APX paired push with PPX hint.
    #[inline]
    pub fn push2p(&mut self, first: Reg, second: Reg) -> Result<()> {
        self.push_pop2(first, second, TypeFlags::T_W1, 0xFF, 6)
    }

    /// `pop2 r64, r64` — APX paired pop.
    #[inline]
    pub fn pop2(&mut self, first: Reg, second: Reg) -> Result<()> {
        self.push_pop2(first, second, TypeFlags::T_W0, 0x8F, 0)
    }

    /// `pop2p r64, r64` — APX paired pop with PPX hint.
    #[inline]
    pub fn pop2p(&mut self, first: Reg, second: Reg) -> Result<()> {
        self.push_pop2(first, second, TypeFlags::T_W1, 0x8F, 0)
    }

    fn push_pop2(
        &mut self,
        first: Reg,
        second: Reg,
        width: TypeFlags,
        opcode: u8,
        extension: u8,
    ) -> Result<()> {
        if !first.is_reg_bit(64) || !second.is_reg_bit(64) {
            return Err(Error::BadCombination);
        }
        self.buf.op_roo(
            &first,
            &RegMem::Reg(second),
            &RegMem::Reg(Reg::gpr64(extension)),
            TypeFlags::T_APX | TypeFlags::T_ND1 | width,
            opcode,
            0,
            None,
        )?;
        Ok(())
    }

    /// `pushp r64` — APX push with the PPX hint.
    #[inline]
    pub fn pushp(&mut self, reg: Reg) -> Result<()> {
        self.push_pop_p(reg, 0x50)
    }

    /// `popp r64` — APX pop with the PPX hint.
    #[inline]
    pub fn popp(&mut self, reg: Reg) -> Result<()> {
        self.push_pop_p(reg, 0x58)
    }

    /// Xbyak `opPushPopP`: REX2 is mandatory because it carries W=1 PPX.
    fn push_pop_p(&mut self, reg: Reg, opcode: u8) -> Result<()> {
        if !reg.is_reg_bit(64) {
            return Err(Error::BadCombination);
        }
        let none = Reg::default();
        self.buf.emit_rex2(
            false,
            crate::encode::rex_rxb(3, true, &none, &reg, &none),
            &none,
            &reg,
            &none,
        )?;
        self.buf.db(opcode | (reg.get_idx() & 7))
    }

    /// `mov dst, src` — Move data.
    #[inline]
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
            (RegMem::Reg(d), RegMemImm::Imm(imm)) => self.mov_reg_imm(&d, imm as u64),
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
                if bit == 0 {
                    return Err(Error::MemSizeIsNotSpecified);
                }
                let code = if bit == 8 { 0xC6u8 } else { 0xC7u8 };
                self.buf.op_rext(
                    &RegMem::Mem(m),
                    0,
                    TypeFlags::NONE,
                    code,
                    if bit == 8 { 1 } else { (bit.min(32) / 8) as u8 },
                )?;
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
            self.buf
                .emit_rex_for_reg_reg(&r32, &default, TypeFlags::NONE)?;
            self.buf.db(0xB8 | (idx & 7))?;
            self.buf.dd(imm as u32)?;
        } else if bit == 64 && crate::encode::is_in_int32(imm) {
            // Use sign-extending mov r/m64, imm32
            let default = Reg::default();
            self.buf
                .emit_rex_for_reg_reg(reg, &default, TypeFlags::NONE)?;
            self.buf.db(0xC7)?;
            self.buf.db(0xC0 | (idx & 7))?;
            self.buf.dd(imm as u32)?;
        } else {
            // Full-width immediate
            let default = Reg::default();
            self.buf
                .emit_rex_for_reg_reg(reg, &default, TypeFlags::NONE)?;
            let code = 0xB0u8 | (if bit == 8 { 0 } else { 8 }) | (idx & 7);
            self.buf.db(code)?;
            self.buf.db_n(imm, (bit / 8) as usize)?;
        }
        Ok(())
    }

    /// Generic arithmetic operation (add/or/adc/sbb/and/sub/xor/cmp).
    fn arith_op(
        &mut self,
        dst: impl Into<RegMem>,
        src: impl Into<RegMemImm>,
        ext: u8,
    ) -> Result<()> {
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
                if d.get_idx() == 0
                    && (d.get_bit() == imm_bit as u16 || (d.is_bit(64) && imm_bit == 32))
                {
                    let default = Reg::default();
                    self.buf
                        .emit_rex_for_reg_reg(&d, &default, TypeFlags::NONE)?;
                    self.buf
                        .db(base_code | 4 | if imm_bit == 8 { 0 } else { 1 })?;
                } else {
                    let tmp = if (imm_bit as u16) < d.get_bit().min(32) {
                        2u8
                    } else {
                        0
                    };
                    self.buf.op_rext(
                        &RegMem::Reg(d),
                        ext,
                        TypeFlags::NONE,
                        0x80 | tmp,
                        imm_bit / 8,
                    )?;
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
                if bit == 0 {
                    return Err(Error::MemSizeIsNotSpecified);
                }
                let imm_bit = Self::get_imm_bit(bit, imm);
                let tmp = if (imm_bit as u16) < bit.min(32) {
                    2u8
                } else {
                    0
                };
                self.buf.op_rext(
                    &RegMem::Mem(m),
                    ext,
                    TypeFlags::NONE,
                    0x80 | tmp,
                    imm_bit / 8,
                )?;
                self.buf.db_n(imm as u64, (imm_bit / 8) as usize)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `add dst, src`
    #[inline]
    pub fn add(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 0)
    }

    /// `or dst, src`
    #[inline]
    pub fn or_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 1)
    }

    /// `adc dst, src`
    #[inline]
    pub fn adc(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 2)
    }

    /// `sbb dst, src`
    #[inline]
    pub fn sbb(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 3)
    }

    /// `and dst, src`
    #[inline]
    pub fn and_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 4)
    }

    /// `sub dst, src`
    #[inline]
    pub fn sub(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 5)
    }

    /// `xor dst, src`
    #[inline]
    pub fn xor_(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 6)
    }

    /// `cmp dst, src`
    #[inline]
    pub fn cmp(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMemImm>) -> Result<()> {
        self.arith_op(dst, src, 7)
    }

    /// `lea dst, src` — Load effective address.
    #[inline]
    pub fn lea(&mut self, dst: Reg, src: Address) -> Result<()> {
        if dst.is_bit(8) {
            return Err(Error::BadCombination);
        }
        self.buf.op_mr(&src, &dst, TypeFlags::NONE, 0x8D)
    }

    /// `lea dst, [rip + label]` — Load label address via RIP-relative addressing.
    ///
    /// This is equivalent to xbyak's `mov(reg, label)` which generates
    /// `lea reg, [rip + disp32]` for 64-bit code.
    #[inline]
    pub fn lea_label(&mut self, dst: Reg, label: &Label) -> Result<()> {
        if !dst.is_bit(64) {
            return Err(Error::BadCombination);
        }
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
    #[inline]
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
                    self.buf
                        .emit_rex_for_reg_reg(&d, &default, TypeFlags::NONE)?;
                    let code = if d.is_bit(8) { 0xA8u8 } else { 0xA9u8 };
                    self.buf.db(code)?;
                } else {
                    self.buf.op_rext(
                        &RegMem::Reg(d),
                        0,
                        TypeFlags::NONE,
                        0xF6,
                        if d.is_bit(8) {
                            1
                        } else {
                            (d.get_bit().min(32) / 8) as u8
                        },
                    )?;
                }
                let n = if d.is_bit(8) {
                    1
                } else {
                    (d.get_bit().min(32) / 8) as usize
                };
                self.buf.db_n(imm as u64, n)
            }
            (RegMem::Mem(m), RegMemImm::Reg(s)) => {
                let code = if s.is_bit(8) { 0x84u8 } else { 0x85u8 };
                self.buf.op_mr(&m, &s, TypeFlags::NONE, code)
            }
            (RegMem::Mem(m), RegMemImm::Imm(imm)) => {
                let bit = m.get_bit();
                if bit == 0 {
                    return Err(Error::MemSizeIsNotSpecified);
                }
                let n = if bit == 8 {
                    1usize
                } else {
                    (bit.min(32) / 8) as usize
                };
                self.buf
                    .op_rext(&RegMem::Mem(m), 0, TypeFlags::NONE, 0xF6, n as u8)?;
                self.buf.db_n(imm as u64, n)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `inc r/m`
    #[inline]
    pub fn inc(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 0, TypeFlags::NONE, 0xFE, 0)
    }

    /// `dec r/m`
    #[inline]
    pub fn dec(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 1, TypeFlags::NONE, 0xFE, 0)
    }

    /// `neg r/m`
    #[inline]
    pub fn neg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 3, TypeFlags::NONE, 0xF6, 0)
    }

    /// `not r/m`
    #[inline]
    pub fn not_(&mut self, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        self.buf.op_rext(&op, 2, TypeFlags::NONE, 0xF6, 0)
    }

    // ─── Jump / Call ───────────────────────────────────────────

    /// `jmp label` — Jump to label.
    #[inline]
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
                    self.label_mgr.add_undef(
                        label.id(),
                        JmpLabel {
                            end_of_jmp: self.buf.size(),
                            jmp_size: 1,
                            mode: LabelMode::AsIs,
                            disp: 0,
                        },
                    );
                }
                Ok(())
            }
            JmpType::Near | JmpType::Auto => {
                self.buf.db(0xE9)?;
                self.put_label(label, 4, true, 0)
            }
        }
    }

    /// `jmpabs imm64` — APX absolute jump encoding.
    #[inline]
    pub fn jmpabs(&mut self, address: u64) -> Result<()> {
        self.buf.db(0xD5)?;
        self.buf.db(0x00)?;
        self.buf.db(0xA1)?;
        self.buf.dq(address)
    }

    /// `jecxz label` — Address-size override plus short-only branch.
    #[inline]
    pub fn jecxz(&mut self, label: &Label) -> Result<()> {
        self.buf.db(0x67)?;
        self.short_label_jump(label, 0xE3)
    }

    /// `jrcxz label` — Short-only branch.
    #[inline]
    pub fn jrcxz(&mut self, label: &Label) -> Result<()> {
        self.short_label_jump(label, 0xE3)
    }

    /// `loop label` — Short-only branch.
    #[inline]
    pub fn loop_(&mut self, label: &Label) -> Result<()> {
        self.short_label_jump(label, 0xE2)
    }

    /// `loope label` — Short-only branch.
    #[inline]
    pub fn loope(&mut self, label: &Label) -> Result<()> {
        self.short_label_jump(label, 0xE1)
    }

    /// `loopne label` — Short-only branch.
    #[inline]
    pub fn loopne(&mut self, label: &Label) -> Result<()> {
        self.short_label_jump(label, 0xE0)
    }

    /// `jmp reg` — Jump to address in register.
    #[inline]
    pub fn jmp_reg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf.op_rext(&op.into(), 4, TypeFlags::NONE, 0xFE, 0)
    }

    /// `call label` — Call subroutine.
    #[inline]
    pub fn call(&mut self, label: &Label) -> Result<()> {
        self.buf.db(0xE8)?;
        self.put_label(label, 4, true, 0)
    }

    /// `call reg` — Call address in register.
    #[inline]
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
                    self.label_mgr.add_undef(
                        label.id(),
                        JmpLabel {
                            end_of_jmp: self.buf.size(),
                            jmp_size: 1,
                            mode: LabelMode::AsIs,
                            disp: 0,
                        },
                    );
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
    #[inline]
    pub fn jo(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0, label, t)
    }
    #[inline]
    pub fn jno(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(1, label, t)
    }
    #[inline]
    pub fn jb(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(2, label, t)
    }
    #[inline]
    pub fn jnb(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(3, label, t)
    }
    #[inline]
    pub fn jz(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(4, label, t)
    }
    #[inline]
    pub fn jnz(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(5, label, t)
    }
    #[inline]
    pub fn jbe(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(6, label, t)
    }
    #[inline]
    pub fn jnbe(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(7, label, t)
    }
    #[inline]
    pub fn js(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(8, label, t)
    }
    #[inline]
    pub fn jns(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(9, label, t)
    }
    #[inline]
    pub fn jp(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xA, label, t)
    }
    #[inline]
    pub fn jnp(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xB, label, t)
    }
    #[inline]
    pub fn jl(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xC, label, t)
    }
    #[inline]
    pub fn jnl(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xD, label, t)
    }
    #[inline]
    pub fn jle(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xE, label, t)
    }
    #[inline]
    pub fn jnle(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jcc(0xF, label, t)
    }

    // Aliases
    #[inline]
    pub fn je(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jz(label, t)
    }
    #[inline]
    pub fn jne(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnz(label, t)
    }
    #[inline]
    pub fn jc(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jb(label, t)
    }
    #[inline]
    pub fn jnc(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnb(label, t)
    }
    #[inline]
    pub fn ja(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnbe(label, t)
    }
    #[inline]
    pub fn jae(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnb(label, t)
    }
    #[inline]
    pub fn jg(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnle(label, t)
    }
    #[inline]
    pub fn jge(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnl(label, t)
    }
    #[inline]
    pub fn jna(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jbe(label, t)
    }
    #[inline]
    pub fn jnae(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jb(label, t)
    }
    #[inline]
    pub fn jng(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jle(label, t)
    }
    #[inline]
    pub fn jnge(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jl(label, t)
    }
    #[inline]
    pub fn jpe(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jp(label, t)
    }
    #[inline]
    pub fn jpo(&mut self, label: &Label, t: JmpType) -> Result<()> {
        self.jnp(label, t)
    }

    /// `int3` — Software breakpoint.
    #[inline]
    pub fn int3(&mut self) -> Result<()> {
        self.buf.db(0xCC)
    }

    /// `int imm8` — Software interrupt.
    #[inline]
    pub fn int_(&mut self, vector: u8) -> Result<()> {
        self.buf.db(0xCD)?;
        self.buf.db(vector)
    }

    /// `in accumulator, dx`.
    #[inline]
    pub fn in_(&mut self, accumulator: Reg, port: Reg) -> Result<()> {
        self.op_in_out_reg(accumulator, port, 0xEC)
    }

    /// `in accumulator, imm8`.
    #[inline]
    pub fn in_imm(&mut self, accumulator: Reg, port: u8) -> Result<()> {
        self.op_in_out_imm(accumulator, 0xE4, port)
    }

    /// `out dx, accumulator`.
    #[inline]
    pub fn out_(&mut self, port: Reg, accumulator: Reg) -> Result<()> {
        self.op_in_out_reg(accumulator, port, 0xEE)
    }

    /// `out imm8, accumulator`.
    #[inline]
    pub fn out_imm(&mut self, port: u8, accumulator: Reg) -> Result<()> {
        self.op_in_out_imm(accumulator, 0xE6, port)
    }

    #[inline]
    pub fn outsb(&mut self) -> Result<()> {
        self.buf.db(0x6E)
    }

    #[inline]
    pub fn outsd(&mut self) -> Result<()> {
        self.buf.db(0x6F)
    }

    #[inline]
    pub fn outsw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0x6F)
    }

    /// `lfs reg, [mem]`.
    #[inline]
    pub fn lfs(&mut self, reg: Reg, addr: Address) -> Result<()> {
        self.buf.op_load_seg(&addr, &reg, TypeFlags::T_0F, 0xB4)
    }

    /// `lgs reg, [mem]`.
    #[inline]
    pub fn lgs(&mut self, reg: Reg, addr: Address) -> Result<()> {
        self.buf.op_load_seg(&addr, &reg, TypeFlags::T_0F, 0xB5)
    }

    /// `lss reg, [mem]`.
    #[inline]
    pub fn lss(&mut self, reg: Reg, addr: Address) -> Result<()> {
        self.buf.op_load_seg(&addr, &reg, TypeFlags::T_0F, 0xB2)
    }

    /// `xchg op1, op2` — Exchange values. At most one operand may be memory.
    #[inline]
    pub fn xchg(&mut self, op1: impl Into<RegMem>, op2: impl Into<RegMem>) -> Result<()> {
        let op1 = op1.into();
        let op2 = op2.into();
        // Normalize: ensure p1 is always the register.
        // Swap if p1 is memory, or if p2 is a non-8-bit register with idx=0 (eax/rax).
        let (p1, p2) = if op1.is_mem()
            || (op2.is_reg()
                && !op2.as_reg().unwrap().is_bit(8)
                && op2.as_reg().unwrap().get_idx() == 0)
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
                self.buf
                    .emit_rex_for_reg_reg(r2, &default, TypeFlags::NONE)?;
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
    #[inline]
    pub fn movzx(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        let src_bit = src.get_bit();
        if src_bit >= 32 {
            return Err(Error::BadCombination);
        }
        if dst.get_bit() <= src_bit {
            return Err(Error::BadCombination);
        }
        let w = if src_bit == 16 { 1u8 } else { 0 };
        match src {
            RegMem::Reg(s) => self.buf.op_rr(
                &dst,
                &s,
                TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0xB6 | w,
            ),
            RegMem::Mem(m) => self.buf.op_mr(
                &m,
                &dst,
                TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0xB6 | w,
            ),
        }
    }

    /// `movsx dst, src` — Move with sign-extend.
    #[inline]
    pub fn movsx(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        let src_bit = src.get_bit();
        if src_bit >= 32 {
            return Err(Error::BadCombination);
        }
        if dst.get_bit() <= src_bit {
            return Err(Error::BadCombination);
        }
        let w = if src_bit == 16 { 1u8 } else { 0 };
        match src {
            RegMem::Reg(s) => self.buf.op_rr(
                &dst,
                &s,
                TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0xBE | w,
            ),
            RegMem::Mem(m) => self.buf.op_mr(
                &m,
                &dst,
                TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0xBE | w,
            ),
        }
    }

    /// `movsxd dst, src` — Move with sign-extend (32→64).
    #[inline]
    pub fn movsxd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        if !dst.is_bit(64) {
            return Err(Error::BadCombination);
        }
        let src = src.into();
        match src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, &s, TypeFlags::T_ALLOW_DIFF_SIZE, 0x63),
            RegMem::Mem(m) => self.buf.op_mr(&m, &dst, TypeFlags::T_ALLOW_DIFF_SIZE, 0x63),
        }
    }

    /// `cdq` — Convert doubleword to quadword (sign-extend eax into edx:eax).
    #[inline]
    pub fn cdq(&mut self) -> Result<()> {
        self.buf.db(0x99)
    }
    /// `cqo` — Convert quadword to double-quadword (sign-extend rax into rdx:rax).
    #[inline]
    pub fn cqo(&mut self) -> Result<()> {
        self.buf.db(0x48)?; // REX.W
        self.buf.db(0x99)
    }

    /// `imul dst, src` — Signed multiply (2-operand form).
    #[inline]
    pub fn imul(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, &s, TypeFlags::T_0F, 0xAF),
            RegMem::Mem(m) => self.buf.op_mr(&m, &dst, TypeFlags::T_0F, 0xAF),
        }
    }

    // ─── Shift operations ──────────────────────────────────────

    /// `shl r/m, imm`
    #[inline]
    pub fn shl(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 4)
    }

    /// `sal r/m, imm` — Xbyak spelling alias for `shl`.
    #[inline]
    pub fn sal(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shl(op, imm)
    }

    /// `shr r/m, imm`
    #[inline]
    pub fn shr(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 5)
    }

    /// `sar r/m, imm`
    #[inline]
    pub fn sar(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.shift_op(op.into(), imm, 7)
    }

    fn shift_op(&mut self, op: RegMem, imm: u8, ext: u8) -> Result<()> {
        let code = if imm == 1 { 0xD0u8 } else { 0xC0u8 };
        let bit = match &op {
            RegMem::Reg(r) => r.get_bit(),
            RegMem::Mem(m) => {
                if m.get_bit() == 0 {
                    return Err(Error::MemSizeIsNotSpecified);
                }
                m.get_bit()
            }
        };
        let code = code | if bit == 8 { 0 } else { 1 };
        self.buf.op_rext(
            &op,
            ext,
            TypeFlags::NONE,
            code,
            if imm == 1 { 0 } else { 1 },
        )?;
        if imm != 1 {
            self.buf.db(imm)?;
        }
        Ok(())
    }

    fn shift_op_cl(&mut self, op: RegMem, ext: u8) -> Result<()> {
        self.buf.op_rext(&op, ext, TypeFlags::T_CODE1_IF1, 0xD2, 0)
    }

    /// `shl r/m, CL`
    #[inline]
    pub fn shl_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shift_op_cl(op.into(), 4)
    }

    /// `sal r/m, CL` — Xbyak spelling alias for `shl`.
    #[inline]
    pub fn sal_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shl_cl(op)
    }

    /// `shr r/m, CL`
    #[inline]
    pub fn shr_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.shift_op_cl(op.into(), 5)
    }

    /// `sar r/m, CL`
    #[inline]
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
            || (type_.contains(TypeFlags::T_YMM)
                && ((x1.is_ymm() && x2.is_ymm()) || (x1.is_zmm() && x2.is_zmm())));
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
    #[inline]
    pub fn addps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x58, None)
    }

    /// `addpd xmm, xmm/m128`
    #[inline]
    pub fn addpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x58,
            None,
        )
    }

    /// `addss xmm, xmm/m32`
    #[inline]
    pub fn addss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x58,
            None,
        )
    }

    /// `addsd xmm, xmm/m64`
    #[inline]
    pub fn addsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x58,
            None,
        )
    }

    /// `subps xmm, xmm/m128`
    #[inline]
    pub fn subps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5C, None)
    }

    /// `subpd xmm, xmm/m128`
    #[inline]
    pub fn subpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x5C,
            None,
        )
    }

    /// `subss xmm, xmm/m32`
    #[inline]
    pub fn subss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x5C,
            None,
        )
    }

    /// `subsd xmm, xmm/m64`
    #[inline]
    pub fn subsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x5C,
            None,
        )
    }

    /// `mulps xmm, xmm/m128`
    #[inline]
    pub fn mulps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x59, None)
    }

    /// `mulpd xmm, xmm/m128`
    #[inline]
    pub fn mulpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x59,
            None,
        )
    }

    /// `mulss xmm, xmm/m32`
    #[inline]
    pub fn mulss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x59,
            None,
        )
    }

    /// `mulsd xmm, xmm/m64`
    #[inline]
    pub fn mulsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x59,
            None,
        )
    }

    /// `divps xmm, xmm/m128`
    #[inline]
    pub fn divps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5E, None)
    }

    /// `divpd xmm, xmm/m128`
    #[inline]
    pub fn divpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x5E,
            None,
        )
    }

    /// `divss xmm, xmm/m32`
    #[inline]
    pub fn divss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x5E,
            None,
        )
    }

    /// `divsd xmm, xmm/m64`
    #[inline]
    pub fn divsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x5E,
            None,
        )
    }

    /// `xorps xmm, xmm/m128`
    #[inline]
    pub fn xorps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x57, None)
    }

    /// `xorpd xmm, xmm/m128`
    #[inline]
    pub fn xorpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x57,
            None,
        )
    }

    /// `andps xmm, xmm/m128`
    #[inline]
    pub fn andps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x54, None)
    }

    /// `andpd xmm, xmm/m128`
    #[inline]
    pub fn andpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x54,
            None,
        )
    }

    /// `orps xmm, xmm/m128`
    #[inline]
    pub fn orps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x56, None)
    }

    /// `orpd xmm, xmm/m128`
    #[inline]
    pub fn orpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x56,
            None,
        )
    }

    /// `sqrtps xmm, xmm/m128`
    #[inline]
    pub fn sqrtps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x51, None)
    }

    /// `sqrtpd xmm, xmm/m128`
    #[inline]
    pub fn sqrtpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x51,
            None,
        )
    }

    /// `sqrtss xmm, xmm/m32`
    #[inline]
    pub fn sqrtss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x51,
            None,
        )
    }

    /// `sqrtsd xmm, xmm/m64`
    #[inline]
    pub fn sqrtsd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x51,
            None,
        )
    }

    /// `movaps xmm, xmm/m128`
    #[inline]
    pub fn movaps(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_sse(d, &src, TypeFlags::T_0F, 0x28, None),
            (RegMem::Mem(m), RegMem::Reg(s)) => self.buf.op_mr(m, s, TypeFlags::T_0F, 0x29),
            _ => Err(Error::BadCombination),
        }
    }

    /// `movups xmm, xmm/m128`
    #[inline]
    pub fn movups(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_sse(d, &src, TypeFlags::T_0F, 0x10, None),
            (RegMem::Mem(m), RegMem::Reg(s)) => self.buf.op_mr(m, s, TypeFlags::T_0F, 0x11),
            _ => Err(Error::BadCombination),
        }
    }

    /// `movapd xmm, xmm/m128`
    #[inline]
    pub fn movapd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x28, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x29)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movupd xmm, xmm/m128`
    #[inline]
    pub fn movupd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movdqa xmm, xmm/m128`
    #[inline]
    pub fn movdqa(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x6F, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x7F)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `movdqu xmm, xmm/m128`
    #[inline]
    pub fn movdqu(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x6F, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0x7F)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `paddd xmm, xmm/m128`
    #[inline]
    pub fn paddd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xFE,
            None,
        )
    }

    /// `psubd xmm, xmm/m128`
    #[inline]
    pub fn psubd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xFA,
            None,
        )
    }

    /// `pxor xmm, xmm/m128`
    #[inline]
    pub fn pxor(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xEF,
            None,
        )
    }

    /// `pand xmm, xmm/m128`
    #[inline]
    pub fn pand(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xDB,
            None,
        )
    }

    /// `por xmm, xmm/m128`
    #[inline]
    pub fn por(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xEB,
            None,
        )
    }

    /// `movd xmm, r/m32` or `movd r/m32, xmm`
    #[inline]
    pub fn movd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            // movd xmm, r/m32
            (RegMem::Reg(d), _) if d.is_xmm() => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x6E, None)
            }
            // movd r/m32, xmm
            (_, RegMem::Reg(s)) if s.is_xmm() => match &dst {
                RegMem::Reg(d) => self
                    .buf
                    .op_rr(s, d, TypeFlags::T_66 | TypeFlags::T_0F, 0x7E),
                RegMem::Mem(m) => self
                    .buf
                    .op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0x7E),
            },
            _ => Err(Error::BadCombination),
        }
    }

    /// `movq xmm, xmm/m64` or `movq m64, xmm`
    #[inline]
    pub fn movq(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            // movq xmm, r64 — 66 REX.W 0F 6E /r
            (RegMem::Reg(d), RegMem::Reg(s)) if d.is_xmm() && s.is_reg_bit(64) => self.buf.op_sse(
                d,
                &src,
                TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0x6E,
                None,
            ),
            // movq xmm, xmm/m64 — F3 0F 7E /r
            (RegMem::Reg(d), _) if d.is_xmm() => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x7E, None)
            }
            // movq r64, xmm — 66 REX.W 0F 7E /r
            (RegMem::Reg(d), RegMem::Reg(s)) if d.is_reg_bit(64) && s.is_xmm() => self.buf.op_rr(
                s,
                d,
                TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
                0x7E,
            ),
            // movq m64, xmm — 66 0F D6 /r
            (RegMem::Mem(m), RegMem::Reg(s)) if s.is_xmm() => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_66 | TypeFlags::T_0F, 0xD6)
            }
            _ => Err(Error::BadCombination),
        }
    }

    /// `cvtpd2pi mm, xmm/m128`.
    #[inline]
    pub fn cvtpd2pi(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_mmx_xmm_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x2D, None)
    }

    /// `cvtpi2pd xmm, mm/m64`.
    #[inline]
    pub fn cvtpi2pd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_xmm_mmx_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x2A, None)
    }

    /// `cvtpi2ps xmm, mm/m64`.
    #[inline]
    pub fn cvtpi2ps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_xmm_mmx_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(&dst, &src, TypeFlags::T_0F, 0x2A, None)
    }

    /// `cvtps2pi mm, xmm/m64`.
    #[inline]
    pub fn cvtps2pi(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_mmx_xmm_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(&dst, &src, TypeFlags::T_0F, 0x2D, None)
    }

    /// `cvttpd2pi mm, xmm/m128`.
    #[inline]
    pub fn cvttpd2pi(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_mmx_xmm_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x2C, None)
    }

    /// `cvttps2pi mm, xmm/m64`.
    #[inline]
    pub fn cvttps2pi(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        if !Self::is_mmx_xmm_or_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(&dst, &src, TypeFlags::T_0F, 0x2C, None)
    }

    /// `maskmovdqu xmm, xmm`.
    #[inline]
    pub fn maskmovdqu(&mut self, src: Reg, mask: Reg) -> Result<()> {
        if !src.is_xmm() || !mask.is_xmm() {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(
            &src,
            &RegMem::Reg(mask),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xF7,
            None,
        )
    }

    /// `maskmovq mm, mm`.
    #[inline]
    pub fn maskmovq(&mut self, src: Reg, mask: Reg) -> Result<()> {
        if !src.is_mmx() || !mask.is_mmx() {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&src, &RegMem::Reg(mask), TypeFlags::T_0F, 0xF7, None)
    }

    /// `movdq2q mm, xmm`.
    #[inline]
    pub fn movdq2q(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_mmx() || !src.is_xmm() {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(
            &dst,
            &RegMem::Reg(src),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0xD6,
            None,
        )
    }

    /// `movq2dq xmm, mm`.
    #[inline]
    pub fn movq2dq(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_xmm() || !src.is_mmx() {
            return Err(Error::BadCombination);
        }
        self.buf.op_sse(
            &dst,
            &RegMem::Reg(src),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0xD6,
            None,
        )
    }

    /// `movntq [mem], mm`.
    #[inline]
    pub fn movntq(&mut self, dst: Address, src: Reg) -> Result<()> {
        if !src.is_mmx() {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&src, &RegMem::Mem(dst), TypeFlags::T_0F, 0xE7, None)
    }

    /// `pshufw mm, mm/m64, imm8` (Xbyak also accepts matching XMM operands).
    #[inline]
    pub fn pshufw(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        let src = src.into();
        if !Self::is_matching_mmx_or_xmm_mem(dst, &src) {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_sse(&dst, &src, TypeFlags::T_0F, 0x70, Some(imm))
    }

    /// `cvtsi2ss xmm, r/m32`
    #[inline]
    pub fn cvtsi2ss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x2A,
            None,
        )
    }

    /// `cvtsi2sd xmm, r/m32`
    #[inline]
    pub fn cvtsi2sd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x2A,
            None,
        )
    }

    /// `cvtss2sd xmm, xmm/m32`
    #[inline]
    pub fn cvtss2sd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x5A,
            None,
        )
    }

    /// `cvtsd2ss xmm, xmm/m64`
    #[inline]
    pub fn cvtsd2ss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x5A,
            None,
        )
    }

    /// `comiss xmm, xmm/m32`
    #[inline]
    pub fn comiss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x2F, None)
    }

    /// `comisd xmm, xmm/m64`
    #[inline]
    pub fn comisd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x2F,
            None,
        )
    }

    /// `ucomiss xmm, xmm/m32`
    #[inline]
    pub fn ucomiss(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x2E, None)
    }

    /// `ucomisd xmm, xmm/m64`
    #[inline]
    pub fn ucomisd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x2E,
            None,
        )
    }

    // ─── AVX Instructions (VEX-encoded) ─────────────────────────

    /// Rust counterpart of Xbyak's vector-result and opmask-result `vcmp*`
    /// overloads. The destination register kind selects the upstream overload.
    fn vcmp_dispatch(
        &mut self,
        dst: Reg,
        src1: Reg,
        src2: impl Into<RegMem>,
        imm: u8,
        vector_type: TypeFlags,
        opmask_type: TypeFlags,
    ) -> Result<()> {
        if dst.is_opmask() {
            self.op_avx_k_x_xm(dst, src1, src2, opmask_type, 0xC2, Some(imm))
        } else {
            self.op_avx_x_x_xm(dst, src1, src2, vector_type, 0xC2, Some(imm))
        }
    }

    /// `vcmppd` — VEX vector-result or EVEX opmask-result form.
    #[inline]
    pub fn vcmppd(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.vcmp_dispatch(
            dst,
            src1,
            src2,
            imm,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_SAE_Z
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B64,
        )
    }

    /// `vcmpps` — VEX vector-result or EVEX opmask-result form.
    #[inline]
    pub fn vcmpps(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.vcmp_dispatch(
            dst,
            src1,
            src2,
            imm,
            TypeFlags::T_0F | TypeFlags::T_YMM,
            TypeFlags::T_0F
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_SAE_Z
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
        )
    }

    /// `vcmpsd` — VEX vector-result or EVEX opmask-result form.
    #[inline]
    pub fn vcmpsd(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.vcmp_dispatch(
            dst,
            src1,
            src2,
            imm,
            TypeFlags::T_F2 | TypeFlags::T_0F,
            TypeFlags::T_N8
                | TypeFlags::T_F2
                | TypeFlags::T_0F
                | TypeFlags::T_EW1
                | TypeFlags::T_SAE_Z
                | TypeFlags::T_MUST_EVEX,
        )
    }

    /// `vcmpss` — VEX vector-result or EVEX opmask-result form.
    #[inline]
    pub fn vcmpss(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.vcmp_dispatch(
            dst,
            src1,
            src2,
            imm,
            TypeFlags::T_F3 | TypeFlags::T_0F,
            TypeFlags::T_N4
                | TypeFlags::T_F3
                | TypeFlags::T_0F
                | TypeFlags::T_W0
                | TypeFlags::T_SAE_Z
                | TypeFlags::T_MUST_EVEX,
        )
    }

    /// `vmovq xmm/m64/r64, xmm/m64/r64`
    ///
    /// Supports the five Xbyak overload families while retaining their
    /// distinct VEX/EVEX opcode selection.
    #[inline]
    pub fn vmovq(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let xmm0 = Reg::xmm(0);
        match (dst, src) {
            (RegMem::Reg(dst), RegMem::Mem(src)) if dst.is_xmm() => {
                let (type_, code) = if dst.get_idx() < 16 {
                    (TypeFlags::T_0F | TypeFlags::T_F3, 0x7E)
                } else {
                    (
                        TypeFlags::T_0F
                            | TypeFlags::T_66
                            | TypeFlags::T_EVEX
                            | TypeFlags::T_EW1
                            | TypeFlags::T_N8,
                        0x6E,
                    )
                };
                self.op_avx_x_x_xm(dst, xmm0, src, type_, code, None)
            }
            (RegMem::Mem(dst), RegMem::Reg(src)) if src.is_xmm() => {
                let code = if src.get_idx() < 16 { 0xD6 } else { 0x7E };
                self.op_avx_x_x_xm(
                    src,
                    xmm0,
                    dst,
                    TypeFlags::T_0F
                        | TypeFlags::T_66
                        | TypeFlags::T_EVEX
                        | TypeFlags::T_EW1
                        | TypeFlags::T_N8,
                    code,
                    None,
                )
            }
            (RegMem::Reg(dst), RegMem::Reg(src)) if dst.is_xmm() && src.is_xmm() => self
                .op_avx_x_x_xm(
                    dst,
                    xmm0,
                    src,
                    TypeFlags::T_0F
                        | TypeFlags::T_F3
                        | TypeFlags::T_EVEX
                        | TypeFlags::T_EW1
                        | TypeFlags::T_N8,
                    0x7E,
                    None,
                ),
            (RegMem::Reg(dst), RegMem::Reg(src)) if dst.is_xmm() && src.is_reg_bit(64) => self
                .op_avx_x_x_xm(
                    dst,
                    xmm0,
                    src,
                    TypeFlags::T_66
                        | TypeFlags::T_0F
                        | TypeFlags::T_W1
                        | TypeFlags::T_EVEX
                        | TypeFlags::T_EW1,
                    0x6E,
                    None,
                ),
            (RegMem::Reg(dst), RegMem::Reg(src)) if dst.is_reg_bit(64) && src.is_xmm() => self
                .op_avx_x_x_xm(
                    src,
                    xmm0,
                    dst,
                    TypeFlags::T_66
                        | TypeFlags::T_0F
                        | TypeFlags::T_W1
                        | TypeFlags::T_EVEX
                        | TypeFlags::T_EW1,
                    0x7E,
                    None,
                ),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vcvtsi2sd xmm, xmm, r/m32|r/m64`
    #[inline]
    pub fn vcvtsi2sd(&mut self, dst: Reg, merge: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt3(
            dst,
            merge,
            src.into(),
            TypeFlags::T_0F | TypeFlags::T_F2 | TypeFlags::T_EVEX,
            TypeFlags::T_W1 | TypeFlags::T_EW1 | TypeFlags::T_ER_R | TypeFlags::T_N8,
            TypeFlags::T_W0 | TypeFlags::T_N4,
            0x2A,
        )
    }

    /// `vcvtsi2ss xmm, xmm, r/m32|r/m64`
    #[inline]
    pub fn vcvtsi2ss(&mut self, dst: Reg, merge: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt3(
            dst,
            merge,
            src.into(),
            TypeFlags::T_0F | TypeFlags::T_F3 | TypeFlags::T_EVEX | TypeFlags::T_ER_R,
            TypeFlags::T_W1 | TypeFlags::T_EW1 | TypeFlags::T_N8,
            TypeFlags::T_W0 | TypeFlags::T_N4,
            0x2A,
        )
    }

    /// `vcvtusi2sd xmm, xmm, r/m32|r/m64`
    #[inline]
    pub fn vcvtusi2sd(&mut self, dst: Reg, merge: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt3(
            dst,
            merge,
            src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_MUST_EVEX,
            TypeFlags::T_W1 | TypeFlags::T_EW1 | TypeFlags::T_ER_R | TypeFlags::T_N8,
            TypeFlags::T_W0 | TypeFlags::T_N4,
            0x7B,
        )
    }

    /// `vcvtusi2ss xmm, xmm, r/m32|r/m64`
    #[inline]
    pub fn vcvtusi2ss(&mut self, dst: Reg, merge: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt3(
            dst,
            merge,
            src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_MUST_EVEX | TypeFlags::T_ER_R,
            TypeFlags::T_W1 | TypeFlags::T_EW1 | TypeFlags::T_N8,
            TypeFlags::T_W0 | TypeFlags::T_N4,
            0x7B,
        )
    }

    /// `vaddps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vaddps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x58,
            None,
        )
    }

    /// `vaddpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vaddpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x58,
            None,
        )
    }

    /// `vaddss xmm, xmm, xmm/m32`
    #[inline]
    pub fn vaddss(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_F3
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_EW0
                | TypeFlags::T_N4
                | TypeFlags::T_ER_X,
            0x58,
            None,
        )
    }

    /// `vaddsd xmm, xmm, xmm/m64`
    #[inline]
    pub fn vaddsd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_F2
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_EW1
                | TypeFlags::T_N8
                | TypeFlags::T_ER_X,
            0x58,
            None,
        )
    }

    /// `vsubps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vsubps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x5C,
            None,
        )
    }

    /// `vsubpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vsubpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x5C,
            None,
        )
    }

    /// `vmulps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vmulps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x59,
            None,
        )
    }

    /// `vmulpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vmulpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x59,
            None,
        )
    }

    /// `vdivps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vdivps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x5E,
            None,
        )
    }

    /// `vdivpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vdivpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x5E,
            None,
        )
    }

    /// `vxorps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vxorps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x57,
            None,
        )
    }

    /// `vxorpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vxorpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x57,
            None,
        )
    }

    /// `vandps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vandps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x54,
            None,
        )
    }

    /// `vandpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vandpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x54,
            None,
        )
    }

    /// `vorps xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vorps(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x56,
            None,
        )
    }

    /// `vorpd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vorpd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW1
                | TypeFlags::T_B64
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0x56,
            None,
        )
    }

    /// `vmovaps xmm/ymm, xmm/ymm/m` or `vmovaps m, xmm/ymm`
    #[inline]
    pub fn vmovaps(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_0F
            | TypeFlags::T_EVEX
            | TypeFlags::T_YMM
            | TypeFlags::T_EW0
            | TypeFlags::T_N16
            | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x28, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x29, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovups xmm/ymm, xmm/ymm/m` or `vmovups m, xmm/ymm`
    #[inline]
    pub fn vmovups(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_0F
            | TypeFlags::T_EVEX
            | TypeFlags::T_YMM
            | TypeFlags::T_EW0
            | TypeFlags::T_N16
            | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x10, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x11, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovapd xmm/ymm, xmm/ymm/m` or `vmovapd m, xmm/ymm`
    #[inline]
    pub fn vmovapd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66
            | TypeFlags::T_0F
            | TypeFlags::T_EVEX
            | TypeFlags::T_YMM
            | TypeFlags::T_EW1
            | TypeFlags::T_N16
            | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x28, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x29, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovupd xmm/ymm, xmm/ymm/m` or `vmovupd m, xmm/ymm`
    #[inline]
    pub fn vmovupd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66
            | TypeFlags::T_0F
            | TypeFlags::T_EVEX
            | TypeFlags::T_YMM
            | TypeFlags::T_EW1
            | TypeFlags::T_N16
            | TypeFlags::T_N_VL;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x10, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x11, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovdqa xmm/ymm, xmm/ymm/m` or `vmovdqa m, xmm/ymm`
    #[inline]
    pub fn vmovdqa(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x6F, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x7F, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vmovdqu xmm/ymm, xmm/ymm/m` or `vmovdqu m, xmm/ymm`
    #[inline]
    pub fn vmovdqu(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        let type_ = TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_YMM;
        match (&dst, &src) {
            (RegMem::Reg(d), _) => self.buf.op_vex(d, None, &src, type_, 0x6F, None),
            (RegMem::Mem(_), RegMem::Reg(s)) => self.buf.op_vex(s, None, &dst, type_, 0x7F, None),
            _ => Err(Error::BadCombination),
        }
    }

    /// `vpaddd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vpaddd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0xFE,
            None,
        )
    }

    /// `vpsubd xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vpsubd(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EVEX
                | TypeFlags::T_YMM
                | TypeFlags::T_EW0
                | TypeFlags::T_B32
                | TypeFlags::T_N16
                | TypeFlags::T_N_VL,
            0xFA,
            None,
        )
    }

    /// `vpxor xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vpxor(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            0xEF,
            None,
        )
    }

    /// `vpand xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vpand(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            0xDB,
            None,
        )
    }

    /// `vpor xmm/ymm, xmm/ymm, xmm/ymm/m`
    #[inline]
    pub fn vpor(&mut self, x1: Reg, x2: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.op_avx_x_x_xm(
            x1,
            x2,
            op,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            0xEB,
            None,
        )
    }

    // ── CMOVcc ────────────────────────────────────────────────
    fn cmovcc(&mut self, cc: u8, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0x40 | cc),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0x40 | cc),
        }
    }
    #[inline]
    pub fn cmovo(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(0, dst, src)
    }
    #[inline]
    pub fn cmovno(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(1, dst, src)
    }
    #[inline]
    pub fn cmovb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(2, dst, src)
    }
    #[inline]
    pub fn cmovc(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(2, dst, src)
    }
    #[inline]
    pub fn cmovnae(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(2, dst, src)
    }
    #[inline]
    pub fn cmovae(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(3, dst, src)
    }
    #[inline]
    pub fn cmovnb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(3, dst, src)
    }
    #[inline]
    pub fn cmovnc(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(3, dst, src)
    }
    #[inline]
    pub fn cmove(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(4, dst, src)
    }
    #[inline]
    pub fn cmovz(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(4, dst, src)
    }
    #[inline]
    pub fn cmovne(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(5, dst, src)
    }
    #[inline]
    pub fn cmovnz(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(5, dst, src)
    }
    #[inline]
    pub fn cmovbe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(6, dst, src)
    }
    #[inline]
    pub fn cmovna(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(6, dst, src)
    }
    #[inline]
    pub fn cmova(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(7, dst, src)
    }
    #[inline]
    pub fn cmovnbe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(7, dst, src)
    }
    #[inline]
    pub fn cmovs(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(8, dst, src)
    }
    #[inline]
    pub fn cmovns(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(9, dst, src)
    }
    #[inline]
    pub fn cmovp(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(10, dst, src)
    }
    #[inline]
    pub fn cmovpe(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(10, dst, src)
    }
    #[inline]
    pub fn cmovnp(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(11, dst, src)
    }
    #[inline]
    pub fn cmovpo(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(11, dst, src)
    }
    #[inline]
    pub fn cmovl(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(12, dst, src)
    }
    #[inline]
    pub fn cmovnge(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(12, dst, src)
    }
    #[inline]
    pub fn cmovge(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(13, dst, src)
    }
    #[inline]
    pub fn cmovnl(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(13, dst, src)
    }
    #[inline]
    pub fn cmovle(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(14, dst, src)
    }
    #[inline]
    pub fn cmovng(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(14, dst, src)
    }
    #[inline]
    pub fn cmovg(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(15, dst, src)
    }
    #[inline]
    pub fn cmovnle(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.cmovcc(15, dst, src)
    }

    // ── SETcc ─────────────────────────────────────────────────
    fn setcc(&mut self, cc: u8, op: impl Into<RegMem>) -> Result<()> {
        let op = op.into();
        // 0F 90+cc /0
        self.buf.op_rext(&op, 0, TypeFlags::T_0F, 0x90 | cc, 0)
    }
    #[inline]
    pub fn seto(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(0, op)
    }
    #[inline]
    pub fn setno(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(1, op)
    }
    #[inline]
    pub fn setb(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(2, op)
    }
    #[inline]
    pub fn setc(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(2, op)
    }
    #[inline]
    pub fn setnae(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(2, op)
    }
    #[inline]
    pub fn setae(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(3, op)
    }
    #[inline]
    pub fn setnb(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(3, op)
    }
    #[inline]
    pub fn setnc(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(3, op)
    }
    #[inline]
    pub fn sete(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(4, op)
    }
    #[inline]
    pub fn setz(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(4, op)
    }
    #[inline]
    pub fn setne(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(5, op)
    }
    #[inline]
    pub fn setnz(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(5, op)
    }
    #[inline]
    pub fn setbe(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(6, op)
    }
    #[inline]
    pub fn setna(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(6, op)
    }
    #[inline]
    pub fn seta(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(7, op)
    }
    #[inline]
    pub fn setnbe(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(7, op)
    }
    #[inline]
    pub fn sets(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(8, op)
    }
    #[inline]
    pub fn setns(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(9, op)
    }
    #[inline]
    pub fn setp(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(10, op)
    }
    #[inline]
    pub fn setpe(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(10, op)
    }
    #[inline]
    pub fn setnp(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(11, op)
    }
    #[inline]
    pub fn setpo(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(11, op)
    }
    #[inline]
    pub fn setl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(12, op)
    }
    #[inline]
    pub fn setnge(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(12, op)
    }
    #[inline]
    pub fn setge(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(13, op)
    }
    #[inline]
    pub fn setnl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(13, op)
    }
    #[inline]
    pub fn setle(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(14, op)
    }
    #[inline]
    pub fn setng(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(14, op)
    }
    #[inline]
    pub fn setg(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(15, op)
    }
    #[inline]
    pub fn setnle(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.setcc(15, op)
    }

    // ── Bit operations ────────────────────────────────────────
    /// BSF - Bit Scan Forward: 0F BC /r
    #[inline]
    pub fn bsf(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0xBC),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0xBC),
        }
    }
    /// BSR - Bit Scan Reverse: 0F BD /r
    #[inline]
    pub fn bsr(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self.buf.op_rr(&dst, s, TypeFlags::T_0F, 0xBD),
            RegMem::Mem(m) => self.buf.op_mr(m, &dst, TypeFlags::T_0F, 0xBD),
        }
    }
    /// POPCNT: F3 0F B8 /r
    #[inline]
    pub fn popcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self
                .buf
                .op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xB8),
            RegMem::Mem(m) => self
                .buf
                .op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xB8),
        }
    }
    /// LZCNT: F3 0F BD /r
    #[inline]
    pub fn lzcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self
                .buf
                .op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBD),
            RegMem::Mem(m) => self
                .buf
                .op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBD),
        }
    }
    /// TZCNT: F3 0F BC /r
    #[inline]
    pub fn tzcnt(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let src = src.into();
        match &src {
            RegMem::Reg(s) => self
                .buf
                .op_rr(&dst, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBC),
            RegMem::Mem(m) => self
                .buf
                .op_mr(m, &dst, TypeFlags::T_F3 | TypeFlags::T_0F, 0xBC),
        }
    }
    /// `crc32 r32/r64, r/m8/16/32/64` — F2 0F 38 F0/F1 /r
    #[inline]
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
    #[inline]
    pub fn bt(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xA3),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xA3),
        }
    }
    /// BTS - Bit Test and Set: 0F AB /r
    #[inline]
    pub fn bts(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xAB),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xAB),
        }
    }
    /// BTR - Bit Test and Reset: 0F B3 /r
    #[inline]
    pub fn btr(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, 0xB3),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xB3),
        }
    }
    /// BTC - Bit Test and Complement: 0F BB /r
    #[inline]
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
    #[inline]
    pub fn bt_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 4, imm)
    }
    /// `bts r/m, imm8` — 0F BA /5 ib
    #[inline]
    pub fn bts_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 5, imm)
    }
    /// `btr r/m, imm8` — 0F BA /6 ib
    #[inline]
    pub fn btr_imm(&mut self, op: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.bt_imm_op(op.into(), 6, imm)
    }
    /// `btc r/m, imm8` — 0F BA /7 ib
    #[inline]
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
    #[inline]
    pub fn rol(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 0, count)
    }
    #[inline]
    pub fn ror(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 1, count)
    }
    #[inline]
    pub fn rcl(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 2, count)
    }
    #[inline]
    pub fn rcr(&mut self, op: impl Into<RegMem>, count: u8) -> Result<()> {
        self.rotate_op(&op.into(), 3, count)
    }

    fn rotate_op_cl(&mut self, op: &RegMem, ext: u8) -> Result<()> {
        self.buf.op_rext(op, ext, TypeFlags::T_CODE1_IF1, 0xD2, 0)
    }
    /// `rol r/m, CL`
    #[inline]
    pub fn rol_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 0)
    }
    /// `ror r/m, CL`
    #[inline]
    pub fn ror_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 1)
    }
    /// `rcl r/m, CL`
    #[inline]
    pub fn rcl_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 2)
    }
    /// `rcr r/m, CL`
    #[inline]
    pub fn rcr_cl(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.rotate_op_cl(&op.into(), 3)
    }

    // ── Single-operand GPR ────────────────────────────────────
    /// MUL: F6 /4 (8-bit), F7 /4 (16/32/64)
    /// Single-operand unsigned multiply: RDX:RAX = RAX * op
    #[inline]
    pub fn mul(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_rext(&op.into(), 4, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// IMUL (1-operand form): F6 /5 (8-bit), F7 /5 (16/32/64)
    /// Single-operand signed multiply: RDX:RAX = RAX * op (signed)
    /// Upstream xbyak: imul(const Operand& op) { opRext(op, 0, 5, T_CODE1_IF1, 0xF6); }
    #[inline]
    pub fn imul_1op(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_rext(&op.into(), 5, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// DIV: F6 /6 (8-bit), F7 /6 (16/32/64)
    #[inline]
    pub fn div(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_rext(&op.into(), 6, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// IDIV: F6 /7 (8-bit), F7 /7 (16/32/64)
    #[inline]
    pub fn idiv(&mut self, op: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_rext(&op.into(), 7, TypeFlags::T_CODE1_IF1, 0xF6, 0)
    }
    /// LEAVE: C9
    #[inline]
    pub fn leave(&mut self) -> Result<()> {
        self.buf.db(0xC9)
    }
    /// ENTER: C8 iw ib
    #[inline]
    pub fn enter(&mut self, alloc_size: u16, nesting: u8) -> Result<()> {
        self.buf.db(0xC8)?;
        self.buf.dw(alloc_size)?;
        self.buf.db(nesting)
    }

    // ── Flag operations ───────────────────────────────────────
    #[inline]
    pub fn clc(&mut self) -> Result<()> {
        self.buf.db(0xF8)
    }
    #[inline]
    pub fn stc(&mut self) -> Result<()> {
        self.buf.db(0xF9)
    }
    #[inline]
    pub fn cld(&mut self) -> Result<()> {
        self.buf.db(0xFC)
    }
    #[inline]
    pub fn std_(&mut self) -> Result<()> {
        self.buf.db(0xFD)
    }
    #[inline]
    pub fn cmc(&mut self) -> Result<()> {
        self.buf.db(0xF5)
    }
    #[inline]
    pub fn cli(&mut self) -> Result<()> {
        self.buf.db(0xFA)
    }
    #[inline]
    pub fn sti(&mut self) -> Result<()> {
        self.buf.db(0xFB)
    }
    #[inline]
    pub fn sahf(&mut self) -> Result<()> {
        self.buf.db(0x9E)
    }
    #[inline]
    pub fn lahf(&mut self) -> Result<()> {
        self.buf.db(0x9F)
    }
    #[inline]
    pub fn hlt(&mut self) -> Result<()> {
        self.buf.db(0xF4)
    }
    #[inline]
    pub fn ud2(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x0B)
    }
    #[inline]
    pub fn cpuid(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0xA2)
    }
    #[inline]
    pub fn rdtsc(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x31)
    }
    #[inline]
    pub fn rdtscp(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xF9)
    }
    #[inline]
    pub fn clzero(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xFC)
    }
    #[inline]
    pub fn endbr32(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x1E)?;
        self.buf.db(0xFB)
    }
    #[inline]
    pub fn endbr64(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x1E)?;
        self.buf.db(0xFA)
    }
    #[inline]
    pub fn monitor(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xC8)
    }
    #[inline]
    pub fn monitorx(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xFA)
    }
    #[inline]
    pub fn mwait(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xC9)
    }
    #[inline]
    pub fn mwaitx(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xFB)
    }
    #[inline]
    pub fn rdmsr(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x32)
    }
    #[inline]
    pub fn rdpmc(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x33)
    }
    #[inline]
    pub fn rdrand(&mut self, reg: Reg) -> Result<()> {
        if reg.is_bit(8) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::new(6, crate::operand::Kind::Reg, reg.get_bit()),
            &reg,
            TypeFlags::T_0F,
            0xC7,
        )
    }
    #[inline]
    pub fn rdseed(&mut self, reg: Reg) -> Result<()> {
        if reg.is_bit(8) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::new(7, crate::operand::Kind::Reg, reg.get_bit()),
            &reg,
            TypeFlags::T_0F,
            0xC7,
        )
    }
    #[inline]
    pub fn rdfsbase(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg() || !(reg.is_bit(32) || reg.is_bit(64)) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(0),
            &reg,
            TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    #[inline]
    pub fn rdgsbase(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg() || !(reg.is_bit(32) || reg.is_bit(64)) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(1),
            &reg,
            TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    #[inline]
    pub fn wrfsbase(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg() || !(reg.is_bit(32) || reg.is_bit(64)) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(2),
            &reg,
            TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    #[inline]
    pub fn wrgsbase(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg() || !(reg.is_bit(32) || reg.is_bit(64)) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(3),
            &reg,
            TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    #[inline]
    pub fn senduipi(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg_bit(64) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(6),
            &reg.cvt32()?,
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0xC7,
        )
    }
    #[inline]
    pub fn serialize(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xE8)
    }
    #[inline]
    pub fn stac(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xCB)
    }
    #[inline]
    pub fn syscall(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x05)
    }
    #[inline]
    pub fn sysenter(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x34)
    }
    #[inline]
    pub fn sysexit(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x35)
    }
    #[inline]
    pub fn sysret(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x07)
    }
    #[inline]
    pub fn wbinvd(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x09)
    }
    #[inline]
    pub fn wrmsr(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x30)
    }
    #[inline]
    pub fn xabort(&mut self, imm: u8) -> Result<()> {
        self.buf.db(0xC6)?;
        self.buf.db(0xF8)?;
        self.buf.db(imm)
    }
    #[inline]
    pub fn xbegin(&mut self, rel: u32) -> Result<()> {
        self.buf.db(0xC7)?;
        self.buf.db(0xF8)?;
        self.buf.dd(rel)
    }
    #[inline]
    pub fn xend(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xD5)
    }
    #[inline]
    pub fn xgetbv(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xD0)
    }
    #[inline]
    pub fn xlatb(&mut self) -> Result<()> {
        self.buf.db(0xD7)
    }
    #[inline]
    pub fn xresldtrk(&mut self) -> Result<()> {
        self.buf.db(0xF2)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xE9)
    }
    #[inline]
    pub fn xsusldtrk(&mut self) -> Result<()> {
        self.buf.db(0xF2)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xE8)
    }
    #[inline]
    pub fn clui(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xEE)
    }
    #[inline]
    pub fn stui(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xEF)
    }
    #[inline]
    pub fn testui(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xED)
    }
    #[inline]
    pub fn uiret(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x0F)?;
        self.buf.db(0x01)?;
        self.buf.db(0xEC)
    }
    #[inline]
    pub fn pause(&mut self) -> Result<()> {
        self.buf.db(0xF3)?;
        self.buf.db(0x90)
    }
    /// `tpause r32` — 66 0F AE /6, with REX2 for r16d-r31d.
    #[inline]
    pub fn tpause(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg_bit(32) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(6),
            &reg,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xAE,
        )
    }
    /// `umonitor r16/r32/r64` — F3 0F AE /6.
    #[inline]
    pub fn umonitor(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg() || reg.is_bit(8) {
            return Err(Error::BadSizeOfRegister);
        }
        if reg.is_bit(32) {
            self.buf.db(0x67)?;
        }
        self.buf.op_rr(
            &Reg::gpr32(6),
            &reg.cvt32()?,
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0xAE,
        )
    }
    /// `umwait r32` — F2 0F AE /6, with REX2 for r16d-r31d.
    #[inline]
    pub fn umwait(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg_bit(32) {
            return Err(Error::BadSizeOfRegister);
        }
        self.buf.op_rr(
            &Reg::gpr32(6),
            &reg,
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0xAE,
        )
    }
    #[inline]
    pub fn lock(&mut self) -> Result<()> {
        self.buf.db(0xF0)
    }
    #[inline]
    pub fn lfence(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0xAE)?;
        self.buf.db(0xE8)
    }
    #[inline]
    pub fn mfence(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0xAE)?;
        self.buf.db(0xF0)
    }
    #[inline]
    pub fn sfence(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0xAE)?;
        self.buf.db(0xF8)
    }
    #[inline]
    pub fn emms(&mut self) -> Result<()> {
        self.buf.db(0x0F)?;
        self.buf.db(0x77)
    }
    #[inline]
    pub fn cbw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0x98)
    }
    #[inline]
    pub fn cwde(&mut self) -> Result<()> {
        self.buf.db(0x98)
    }
    #[inline]
    pub fn cwd(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0x99)
    }
    #[inline]
    pub fn cdqe(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0x98)
    }
    #[inline]
    pub fn popf(&mut self) -> Result<()> {
        self.buf.db(0x9D)
    }
    #[inline]
    pub fn popfq(&mut self) -> Result<()> {
        self.popf()
    }
    #[inline]
    pub fn pushf(&mut self) -> Result<()> {
        self.buf.db(0x9C)
    }
    #[inline]
    pub fn pushfq(&mut self) -> Result<()> {
        self.pushf()
    }
    #[inline]
    pub fn stmxcsr(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::new(3, crate::operand::Kind::Reg, 32),
            TypeFlags::T_0F,
            0xAE,
        )
    }
    #[inline]
    pub fn ldmxcsr(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::new(2, crate::operand::Kind::Reg, 32),
            TypeFlags::T_0F,
            0xAE,
        )
    }

    // ── String operations ─────────────────────────────────────
    #[inline]
    pub fn rep(&mut self) -> Result<()> {
        self.buf.db(0xF3)
    }
    #[inline]
    pub fn repe(&mut self) -> Result<()> {
        self.buf.db(0xF3)
    }
    #[inline]
    pub fn repz(&mut self) -> Result<()> {
        self.buf.db(0xF3)
    }
    #[inline]
    pub fn repne(&mut self) -> Result<()> {
        self.buf.db(0xF2)
    }
    #[inline]
    pub fn repnz(&mut self) -> Result<()> {
        self.buf.db(0xF2)
    }
    #[inline]
    pub fn lodsb(&mut self) -> Result<()> {
        self.buf.db(0xAC)
    }
    #[inline]
    pub fn lodsw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0xAD)
    }
    #[inline]
    pub fn lodsd(&mut self) -> Result<()> {
        self.buf.db(0xAD)
    }
    #[inline]
    pub fn lodsq(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0xAD)
    }
    #[inline]
    pub fn stosb(&mut self) -> Result<()> {
        self.buf.db(0xAA)
    }
    #[inline]
    pub fn stosw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0xAB)
    }
    #[inline]
    pub fn stosd(&mut self) -> Result<()> {
        self.buf.db(0xAB)
    }
    #[inline]
    pub fn stosq(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0xAB)
    }
    #[inline]
    pub fn movsb(&mut self) -> Result<()> {
        self.buf.db(0xA4)
    }
    #[inline]
    pub fn movsw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0xA5)
    }
    #[inline]
    pub fn movsd_string(&mut self) -> Result<()> {
        self.buf.db(0xA5)
    }
    #[inline]
    pub fn movsq(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0xA5)
    }
    #[inline]
    pub fn scasb(&mut self) -> Result<()> {
        self.buf.db(0xAE)
    }
    #[inline]
    pub fn scasw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0xAF)
    }
    #[inline]
    pub fn scasd(&mut self) -> Result<()> {
        self.buf.db(0xAF)
    }
    #[inline]
    pub fn scasq(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0xAF)
    }
    #[inline]
    pub fn cmpsb(&mut self) -> Result<()> {
        self.buf.db(0xA6)
    }
    #[inline]
    pub fn cmpsw(&mut self) -> Result<()> {
        self.buf.db(0x66)?;
        self.buf.db(0xA7)
    }
    #[inline]
    pub fn cmpsq(&mut self) -> Result<()> {
        self.buf.db(0x48)?;
        self.buf.db(0xA7)
    }

    // ── CMPXCHG ───────────────────────────────────────────────
    #[inline]
    pub fn cmpxchg(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let code = if src.get_bit() == 8 { 0xB0u8 } else { 0xB1u8 };
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, code),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, code),
        }
    }

    /// CMPXCHG8B: 0F C7 /1 [m64]  — atomic 8-byte compare-and-swap.
    /// Compares EDX:EAX against the 8-byte memory operand; on equal,
    /// stores ECX:EBX → memory; otherwise loads memory → EDX:EAX.
    /// Mirrors xbyak's `void cmpxchg8b(const Address& addr) { opMR(addr, Reg32(1), T_0F, 0xC7); }`.
    #[inline]
    pub fn cmpxchg8b(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::new(1, crate::operand::Kind::Reg, 32),
            TypeFlags::T_0F,
            0xC7,
        )
    }

    /// CMPXCHG16B: REX.W 0F C7 /1 [m128] — atomic 16-byte compare-and-swap.
    /// Compares RDX:RAX against the 16-byte memory operand; on equal,
    /// stores RCX:RBX → memory; otherwise loads memory → RDX:RAX.
    /// Required by upstream dynarmic's `EmitReadMemoryMov<128>` /
    /// `EmitWriteMemoryMov<128>` ordered (atomic) paths.
    /// Mirrors xbyak's `void cmpxchg16b(const Address& addr) { opMR(addr, Reg64(1), T_0F, 0xC7); }`.
    #[inline]
    pub fn cmpxchg16b(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::new(1, crate::operand::Kind::Reg, 64),
            TypeFlags::T_0F,
            0xC7,
        )
    }
    /// XADD: 0F C0/C1 /r
    #[inline]
    pub fn xadd(&mut self, op: impl Into<RegMem>, src: Reg) -> Result<()> {
        let code = if src.get_bit() == 8 { 0xC0u8 } else { 0xC1u8 };
        let op = op.into();
        match &op {
            RegMem::Reg(r) => self.buf.op_rr(&src, r, TypeFlags::T_0F, code),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, code),
        }
    }

    // ── VZEROALL / VZEROUPPER ─────────────────────────────────
    #[inline]
    pub fn vzeroall(&mut self) -> Result<()> {
        self.buf.db(0xC5)?;
        self.buf.db(0xFC)?;
        self.buf.db(0x77)
    }
    #[inline]
    pub fn vzeroupper(&mut self) -> Result<()> {
        self.buf.db(0xC5)?;
        self.buf.db(0xF8)?;
        self.buf.db(0x77)
    }

    // ── Non-temporal stores ───────────────────────────────────
    /// `movntps m128, xmm` — 0F 2B /r
    #[inline]
    pub fn movntps(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x2B)
    }
    /// `movntpd m128, xmm` — 66 0F 2B /r
    #[inline]
    pub fn movntpd(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x2B)
    }
    /// `movntdq m128, xmm` — 66 0F E7 /r
    #[inline]
    pub fn movntdq(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0xE7)
    }
    /// `movnti m32/m64, r32/r64` — 0F C3 /r
    #[inline]
    pub fn movnti(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0xC3)
    }
    /// `vmovntps m, xmm/ymm/zmm` — VEX.128/256.0F.WIG 2B /r
    #[inline]
    pub fn vmovntps(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf
            .op_vex(&src, None, &RegMem::Mem(addr), t, 0x2B, None)
    }
    /// `vmovntpd m, xmm/ymm/zmm` — VEX.128/256.66.0F.WIG 2B /r
    #[inline]
    pub fn vmovntpd(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf
            .op_vex(&src, None, &RegMem::Mem(addr), t, 0x2B, None)
    }
    /// `vmovntdq m, xmm/ymm/zmm` — VEX.128/256.66.0F.WIG E7 /r
    #[inline]
    pub fn vmovntdq(&mut self, addr: Address, src: Reg) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM | TypeFlags::T_EVEX;
        self.buf
            .op_vex(&src, None, &RegMem::Mem(addr), t, 0xE7, None)
    }

    // ── Partial register loads/stores ─────────────────────────
    /// `movhps xmm, m64` — 0F 16 /r (load)
    #[inline]
    pub fn movhps_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_0F, 0x16)
    }
    /// `movhps m64, xmm` — 0F 17 /r (store)
    #[inline]
    pub fn movhps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x17)
    }
    /// `movlps xmm, m64` — 0F 12 /r (load)
    #[inline]
    pub fn movlps_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(&addr, &dst, TypeFlags::T_0F, 0x12)
    }
    /// `movlps m64, xmm` — 0F 13 /r (store)
    #[inline]
    pub fn movlps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_mr(&addr, &src, TypeFlags::T_0F, 0x13)
    }
    /// `movhpd xmm, m64` — 66 0F 16 /r (load)
    #[inline]
    pub fn movhpd_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &dst, TypeFlags::T_66 | TypeFlags::T_0F, 0x16)
    }
    /// `movhpd m64, xmm` — 66 0F 17 /r (store)
    #[inline]
    pub fn movhpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x17)
    }
    /// `movlpd xmm, m64` — 66 0F 12 /r (load)
    #[inline]
    pub fn movlpd_load(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &dst, TypeFlags::T_66 | TypeFlags::T_0F, 0x12)
    }
    /// `movlpd m64, xmm` — 66 0F 13 /r (store)
    #[inline]
    pub fn movlpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_mr(&addr, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0x13)
    }
    /// `vmovhps xmm, xmm, m64` — VEX 0F 16 /r (load, 3-op)
    #[inline]
    pub fn vmovhps_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &RegMem::Mem(addr),
            TypeFlags::T_0F,
            0x16,
            None,
        )
    }
    /// `vmovhps m64, xmm` — VEX 0F 17 /r (store)
    #[inline]
    pub fn vmovhps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F, 0x17, None)
    }
    /// `vmovlps xmm, xmm, m64` — VEX 0F 12 /r (load, 3-op)
    #[inline]
    pub fn vmovlps_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &RegMem::Mem(addr),
            TypeFlags::T_0F,
            0x12,
            None,
        )
    }
    /// `vmovlps m64, xmm` — VEX 0F 13 /r (store)
    #[inline]
    pub fn vmovlps_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf
            .op_vex(&src, None, &RegMem::Mem(addr), TypeFlags::T_0F, 0x13, None)
    }
    /// `vmovhpd xmm, xmm, m64` — VEX.66 0F 16 /r (load, 3-op)
    #[inline]
    pub fn vmovhpd_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x16,
            None,
        )
    }
    /// `vmovhpd m64, xmm` — VEX.66 0F 17 /r (store)
    #[inline]
    pub fn vmovhpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x17,
            None,
        )
    }
    /// `vmovlpd xmm, xmm, m64` — VEX.66 0F 12 /r (load, 3-op)
    #[inline]
    pub fn vmovlpd_load(&mut self, dst: Reg, src1: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x12,
            None,
        )
    }
    /// `vmovlpd m64, xmm` — VEX.66 0F 13 /r (store)
    #[inline]
    pub fn vmovlpd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x13,
            None,
        )
    }

    // ── SSE4.1 Extract scalar ─────────────────────────────────
    /// `pextrb r/m8, xmm, imm8` — 66 0F 3A 14 /r ib
    #[inline]
    pub fn pextrb(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf
                    .op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x14)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf
                    .op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x14)?;
                self.buf.db(imm)
            }
        }
    }
    /// `pextrw r32, xmm, imm8` — 66 0F C5 /r ib (reg form)
    #[inline]
    pub fn pextrw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf
            .op_rr(&dst, &src, TypeFlags::T_66 | TypeFlags::T_0F, 0xC5)?;
        self.buf.db(imm)
    }
    /// `pextrd r/m32, xmm, imm8` — 66 0F 3A 16 /r ib
    #[inline]
    pub fn pextrd(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf
                    .op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf
                    .op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
        }
    }
    /// `pextrq r/m64, xmm, imm8` — 66 REX.W 0F 3A 16 /r ib
    #[inline]
    pub fn pextrq(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        // Need REX.W: use a 64-bit dummy reg to force W bit
        match &dst {
            RegMem::Reg(d) => {
                let d64 = Reg::new(d.get_idx(), crate::operand::Kind::Reg, 64);
                self.buf
                    .op_rr(&src, &d64, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                let src64 = Reg::new(src.get_idx(), crate::operand::Kind::Xmm, 64);
                self.buf
                    .op_mr(m, &src64, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x16)?;
                self.buf.db(imm)
            }
        }
    }
    /// `extractps r/m32, xmm, imm8` — 66 0F 3A 17 /r ib
    #[inline]
    pub fn extractps(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => {
                self.buf
                    .op_rr(&src, d, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x17)?;
                self.buf.db(imm)
            }
            RegMem::Mem(m) => {
                self.buf
                    .op_mr(m, &src, TypeFlags::T_66 | TypeFlags::T_0F3A, 0x17)?;
                self.buf.db(imm)
            }
        }
    }

    // ── SSE4.1 Insert scalar ──────────────────────────────────
    /// `pinsrb xmm, r/m8, imm8` — 66 0F 3A 20 /r ib
    #[inline]
    pub fn pinsrb(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x20,
            Some(imm),
        )
    }
    /// `pinsrw xmm, r/m16, imm8` — 66 0F C4 /r ib
    #[inline]
    pub fn pinsrw(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xC4,
            Some(imm),
        )
    }
    /// `pinsrd xmm, r/m32, imm8` — 66 0F 3A 22 /r ib
    #[inline]
    pub fn pinsrd(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x22,
            Some(imm),
        )
    }
    /// `pinsrq xmm, r/m64, imm8` — 66 REX.W 0F 3A 22 /r ib
    #[inline]
    pub fn pinsrq(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        // Force REX.W by using 64-bit dst
        let dst64 = Reg::new(dst.get_idx(), crate::operand::Kind::Xmm, 64);
        self.buf.op_sse(
            &dst64,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x22,
            Some(imm),
        )
    }
    /// `insertps xmm, xmm/m32, imm8` — 66 0F 3A 21 /r ib
    #[inline]
    pub fn insertps(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x21,
            Some(imm),
        )
    }

    // ── AVX Extract scalar (VEX) ──────────────────────────────
    /// `vpextrb r/m8, xmm, imm8` — VEX.128.66.0F3A 14 /r ib
    #[inline]
    pub fn vpextrb(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(
            &src,
            None,
            &dst,
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x14,
            Some(imm),
        )
    }
    /// `vpextrw r16/r32/r64/m16, xmm, imm8`
    #[inline]
    pub fn vpextrw(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        let valid_dst = match dst {
            RegMem::Reg(reg) => reg.is_reg() && matches!(reg.get_bit(), 16 | 32 | 64),
            RegMem::Mem(_) => true,
        };
        if !valid_dst || !src.is_xmm() {
            return Err(Error::BadCombination);
        }

        // Xbyak uses the compact VEX.66.0F C5 form only when both register
        // indices fit below 16. EGPR and memory destinations use 0F3A 15.
        if let RegMem::Reg(reg) = dst {
            if src.get_idx() < 16 && reg.get_idx() < 16 {
                return self.buf.op_vex(
                    &Reg::xmm(reg.get_idx()),
                    Some(&Reg::xmm(0)),
                    &RegMem::Reg(src),
                    TypeFlags::T_66 | TypeFlags::T_0F,
                    0xC5,
                    Some(imm),
                );
            }
        }

        self.buf.op_vex(
            &src,
            None,
            &dst,
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_EVEX | TypeFlags::T_N2,
            0x15,
            Some(imm),
        )
    }
    /// `vpextrd r/m32, xmm, imm8` — VEX.128.66.0F3A 16 /r ib
    #[inline]
    pub fn vpextrd(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(
            &src,
            None,
            &dst,
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x16,
            Some(imm),
        )
    }
    /// `vpextrq r/m64, xmm, imm8` — VEX.128.66.0F3A.W1 16 /r ib
    #[inline]
    pub fn vpextrq(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(
            &src,
            None,
            &dst,
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x16,
            Some(imm),
        )
    }
    /// `vextractps r/m32, xmm, imm8` — VEX.128.66.0F3A 17 /r ib
    #[inline]
    pub fn vextractps(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let dst = dst.into();
        self.buf.op_vex(
            &src,
            None,
            &dst,
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x17,
            Some(imm),
        )
    }

    // ── AVX Insert scalar (VEX 3-op) ─────────────────────────
    /// `vpinsrb xmm, xmm, r/m8, imm8` — VEX.128.66.0F3A 20 /r ib
    #[inline]
    pub fn vpinsrb(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &src2.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x20,
            Some(imm),
        )
    }
    /// `vpinsrw xmm, xmm, r/m16, imm8` — VEX.128.66.0F C4 /r ib
    #[inline]
    pub fn vpinsrw(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &src2.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0xC4,
            Some(imm),
        )
    }
    /// `vpinsrd xmm, xmm, r/m32, imm8` — VEX.128.66.0F3A 22 /r ib
    #[inline]
    pub fn vpinsrd(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &src2.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x22,
            Some(imm),
        )
    }
    /// `vpinsrq xmm, xmm, r/m64, imm8` — VEX.128.66.0F3A.W1 22 /r ib
    #[inline]
    pub fn vpinsrq(&mut self, dst: Reg, src1: Reg, src2: impl Into<RegMem>, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &src2.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x22,
            Some(imm),
        )
    }
    /// `vinsertps xmm, xmm, xmm/m32, imm8` — VEX.128.66.0F3A 21 /r ib
    #[inline]
    pub fn vinsertps(
        &mut self,
        dst: Reg,
        src1: Reg,
        src2: impl Into<RegMem>,
        imm: u8,
    ) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &src2.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A,
            0x21,
            Some(imm),
        )
    }

    // ── AVX Extract vector (VEX) ──────────────────────────────
    /// `vextractf128 xmm/m128, ymm, imm8` — VEX.256.66.0F3A 19 /r ib
    #[inline]
    pub fn vextractf128(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM,
            0x19,
            Some(imm),
        )
    }
    /// `vextracti128 xmm/m128, ymm, imm8` — VEX.256.66.0F3A 39 /r ib
    #[inline]
    pub fn vextracti128(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM,
            0x39,
            Some(imm),
        )
    }
    /// `vextractf32x4 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W0 19 /r ib
    #[inline]
    pub fn vextractf32x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_N16,
            0x19,
            Some(imm),
        )
    }
    /// `vextracti32x4 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W0 39 /r ib
    #[inline]
    pub fn vextracti32x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_N16,
            0x39,
            Some(imm),
        )
    }
    /// `vextractf64x2 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W1 19 /r ib
    #[inline]
    pub fn vextractf64x2(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_N16,
            0x19,
            Some(imm),
        )
    }
    /// `vextracti64x2 xmm/m128, ymm/zmm, imm8` — EVEX.256/512.66.0F3A.W1 39 /r ib
    #[inline]
    pub fn vextracti64x2(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_N16,
            0x39,
            Some(imm),
        )
    }
    /// `vextractf32x8 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W0 1B /r ib
    #[inline]
    pub fn vextractf32x8(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_N32,
            0x1B,
            Some(imm),
        )
    }
    /// `vextracti32x8 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W0 3B /r ib
    #[inline]
    pub fn vextracti32x8(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_N32,
            0x3B,
            Some(imm),
        )
    }
    /// `vextractf64x4 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W1 1B /r ib
    #[inline]
    pub fn vextractf64x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_N32,
            0x1B,
            Some(imm),
        )
    }
    /// `vextracti64x4 ymm/m256, zmm, imm8` — EVEX.512.66.0F3A.W1 3B /r ib
    #[inline]
    pub fn vextracti64x4(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &dst.into(),
            TypeFlags::T_66
                | TypeFlags::T_0F3A
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_N32,
            0x3B,
            Some(imm),
        )
    }

    // ── SSE4.1 Variable blend ─────────────────────────────────
    /// `blendvps xmm, xmm/m128, <XMM0>` — 66 0F 38 14 /r (implicit XMM0)
    #[inline]
    pub fn blendvps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F38,
            0x14,
            None,
        )
    }
    /// `blendvpd xmm, xmm/m128, <XMM0>` — 66 0F 38 15 /r (implicit XMM0)
    #[inline]
    pub fn blendvpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F38,
            0x15,
            None,
        )
    }
    /// `pblendvb xmm, xmm/m128, <XMM0>` — 66 0F 38 10 /r (implicit XMM0)
    #[inline]
    pub fn pblendvb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F38,
            0x10,
            None,
        )
    }

    // ── vcvtps2ph — float32 to float16 ───────────────────────
    /// `vcvtps2ph xmm/m, xmm/ymm/zmm, imm8` — VEX.128/256.66.0F3A.W0 1D /r ib
    #[inline]
    pub fn vcvtps2ph(&mut self, dst: impl Into<RegMem>, src: Reg, imm: u8) -> Result<()> {
        let t = TypeFlags::T_66
            | TypeFlags::T_0F3A
            | TypeFlags::T_W0
            | TypeFlags::T_YMM
            | TypeFlags::T_EVEX
            | TypeFlags::T_N8
            | TypeFlags::T_N_VL;
        self.buf.op_vex(&src, None, &dst.into(), t, 0x1D, Some(imm))
    }

    // ── SHLD / SHRD ──────────────────────────────────────────
    /// `shld r/m, r, imm8` — 0F A4 /r ib
    #[inline]
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
    #[inline]
    pub fn shld_cl(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xA5),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xA5),
        }
    }
    /// `shrd r/m, r, imm8` — 0F AC /r ib
    #[inline]
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
    #[inline]
    pub fn shrd_cl(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        let dst = dst.into();
        match &dst {
            RegMem::Reg(d) => self.buf.op_rr(&src, d, TypeFlags::T_0F, 0xAD),
            RegMem::Mem(m) => self.buf.op_mr(m, &src, TypeFlags::T_0F, 0xAD),
        }
    }

    // ── BSWAP ─────────────────────────────────────────────────
    /// `bswap r32/r64` — 0F C8+rd
    #[inline]
    pub fn bswap(&mut self, reg: Reg) -> Result<()> {
        if reg.is_bit(64) {
            self.buf.db(0x48 | if reg.get_idx() >= 8 { 1 } else { 0 })?;
        } else if reg.get_idx() >= 8 {
            self.buf.db(0x41)?;
        }
        self.buf.db(0x0F)?;
        self.buf.db(0xC8 + (reg.get_idx() & 7))
    }

    /// `rorx r32/r64, r/m32/r/m64, imm8` — VEX.LZ.F2.0F3A.F0 /r ib.
    ///
    /// RORX is kept as a hand-written size-dependent encoder: unlike the
    /// generated SIMD tables, VEX.W follows the scalar destination width.
    #[inline]
    pub fn rorx(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        if !dst.is_bit(32) && !dst.is_bit(64) {
            return Err(Error::BadCombination);
        }
        let src = src.into();
        if let RegMem::Reg(src_reg) = src {
            if src_reg.get_bit() != dst.get_bit() {
                return Err(Error::BadCombination);
            }
        }
        let mut flags = TypeFlags::T_F2 | TypeFlags::T_0F3A;
        flags = flags
            | if dst.is_bit(64) {
                TypeFlags::T_W1
            } else {
                TypeFlags::T_W0
            };
        self.buf.op_vex(&dst, None, &src, flags, 0xF0, Some(imm))
    }

    fn bmi_vex_flags(dst: Reg, mandatory_prefix: TypeFlags) -> Result<TypeFlags> {
        if !dst.is_bit(32) && !dst.is_bit(64) {
            return Err(Error::BadCombination);
        }
        Ok(TypeFlags::T_0F38
            | mandatory_prefix
            | if dst.is_bit(64) {
                TypeFlags::T_W1
            } else {
                TypeFlags::T_W0
            })
    }

    fn check_bmi_reg_width(dst: Reg, reg: Reg) -> Result<()> {
        if reg.get_bit() != dst.get_bit() {
            return Err(Error::BadCombination);
        }
        Ok(())
    }

    fn check_bmi_rm_width(dst: Reg, operand: &RegMem) -> Result<()> {
        if let RegMem::Reg(reg) = operand {
            Self::check_bmi_reg_width(dst, *reg)?;
        }
        Ok(())
    }

    /// `shlx r32/r64, r/m32/r/m64, r32/r64` — VEX.NDD.LZ.66.0F38.F7 /r.
    #[inline]
    pub fn shlx(&mut self, dst: Reg, src: impl Into<RegMem>, shift: Reg) -> Result<()> {
        let src = src.into();
        Self::check_bmi_reg_width(dst, shift)?;
        Self::check_bmi_rm_width(dst, &src)?;
        self.buf.op_vex(
            &dst,
            Some(&shift),
            &src,
            Self::bmi_vex_flags(dst, TypeFlags::T_66)?,
            0xF7,
            None,
        )
    }

    /// `shrx r32/r64, r/m32/r/m64, r32/r64` — VEX.NDD.LZ.F2.0F38.F7 /r.
    #[inline]
    pub fn shrx(&mut self, dst: Reg, src: impl Into<RegMem>, shift: Reg) -> Result<()> {
        let src = src.into();
        Self::check_bmi_reg_width(dst, shift)?;
        Self::check_bmi_rm_width(dst, &src)?;
        self.buf.op_vex(
            &dst,
            Some(&shift),
            &src,
            Self::bmi_vex_flags(dst, TypeFlags::T_F2)?,
            0xF7,
            None,
        )
    }

    /// `sarx r32/r64, r/m32/r/m64, r32/r64` — VEX.NDD.LZ.F3.0F38.F7 /r.
    #[inline]
    pub fn sarx(&mut self, dst: Reg, src: impl Into<RegMem>, shift: Reg) -> Result<()> {
        let src = src.into();
        Self::check_bmi_reg_width(dst, shift)?;
        Self::check_bmi_rm_width(dst, &src)?;
        self.buf.op_vex(
            &dst,
            Some(&shift),
            &src,
            Self::bmi_vex_flags(dst, TypeFlags::T_F3)?,
            0xF7,
            None,
        )
    }

    /// `andn r32/r64, r32/r64, r/m32/r/m64` — VEX.NDS.LZ.0F38.F2 /r.
    #[inline]
    pub fn andn(&mut self, dst: Reg, inverted: Reg, value: impl Into<RegMem>) -> Result<()> {
        let value = value.into();
        Self::check_bmi_reg_width(dst, inverted)?;
        Self::check_bmi_rm_width(dst, &value)?;
        self.buf.op_vex(
            &dst,
            Some(&inverted),
            &value,
            Self::bmi_vex_flags(dst, TypeFlags::NONE)?,
            0xF2,
            None,
        )
    }

    /// `bextr r32/r64, r/m32/r/m64, r32/r64` — VEX.NDS.LZ.0F38.F7 /r.
    #[inline]
    pub fn bextr(&mut self, dst: Reg, src: impl Into<RegMem>, control: Reg) -> Result<()> {
        let src = src.into();
        Self::check_bmi_reg_width(dst, control)?;
        Self::check_bmi_rm_width(dst, &src)?;
        self.buf.op_vex(
            &dst,
            Some(&control),
            &src,
            Self::bmi_vex_flags(dst, TypeFlags::NONE)?,
            0xF7,
            None,
        )
    }

    /// `bzhi r32/r64, r/m32/r/m64, r32/r64` — VEX.NDS.LZ.0F38.F5 /r.
    #[inline]
    pub fn bzhi(&mut self, dst: Reg, src: impl Into<RegMem>, index: Reg) -> Result<()> {
        let src = src.into();
        Self::check_bmi_reg_width(dst, index)?;
        Self::check_bmi_rm_width(dst, &src)?;
        self.buf.op_vex(
            &dst,
            Some(&index),
            &src,
            Self::bmi_vex_flags(dst, TypeFlags::NONE)?,
            0xF5,
            None,
        )
    }

    /// `pdep r32/r64, r32/r64, r/m32/r/m64` — VEX.NDS.LZ.F2.0F38.F5 /r.
    #[inline]
    pub fn pdep(&mut self, dst: Reg, src: Reg, mask: impl Into<RegMem>) -> Result<()> {
        let mask = mask.into();
        Self::check_bmi_reg_width(dst, src)?;
        Self::check_bmi_rm_width(dst, &mask)?;
        self.buf.op_vex(
            &dst,
            Some(&src),
            &mask,
            Self::bmi_vex_flags(dst, TypeFlags::T_F2)?,
            0xF5,
            None,
        )
    }

    /// `pext r32/r64, r32/r64, r/m32/r/m64` — VEX.NDS.LZ.F3.0F38.F5 /r.
    #[inline]
    pub fn pext(&mut self, dst: Reg, src: Reg, mask: impl Into<RegMem>) -> Result<()> {
        let mask = mask.into();
        Self::check_bmi_reg_width(dst, src)?;
        Self::check_bmi_rm_width(dst, &mask)?;
        self.buf.op_vex(
            &dst,
            Some(&src),
            &mask,
            Self::bmi_vex_flags(dst, TypeFlags::T_F3)?,
            0xF5,
            None,
        )
    }

    /// Emit an EVEX narrowing move whose architectural destination is encoded
    /// in ModRM.r/m and whose source is encoded in ModRM.reg.
    #[inline]
    fn vpmov_narrow_xmm(&mut self, dst: Reg, src: Reg, opcode: u8) -> Result<()> {
        if !dst.is_xmm() || !src.is_xmm() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Reg(dst),
            TypeFlags::T_F3 | TypeFlags::T_0F38 | TypeFlags::T_MUST_EVEX | TypeFlags::T_W0,
            opcode,
            None,
        )
    }

    /// `vpmovwb xmm, xmm` — EVEX.128.F3.0F38.W0 30 /r.
    #[inline]
    pub fn vpmovwb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.vpmov_narrow_xmm(dst, src, 0x30)
    }

    /// `vpmovdw xmm, xmm` — EVEX.128.F3.0F38.W0 33 /r.
    #[inline]
    pub fn vpmovdw(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.vpmov_narrow_xmm(dst, src, 0x33)
    }

    /// `vpmovqd xmm, xmm` — EVEX.128.F3.0F38.W0 35 /r.
    #[inline]
    pub fn vpmovqd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.vpmov_narrow_xmm(dst, src, 0x35)
    }

    /// Emit an EVEX vector sign-bit to opmask move.
    #[inline]
    fn vpmov_to_mask(&mut self, dst: Reg, src: Reg, flags: TypeFlags, opcode: u8) -> Result<()> {
        if !dst.is_opmask() || !src.is_simd() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_F3 | TypeFlags::T_0F38 | TypeFlags::T_MUST_EVEX | TypeFlags::T_YMM | flags,
            opcode,
            None,
        )
    }

    /// `vpmovb2m k, xmm/ymm/zmm` — EVEX.F3.0F38.W0 29 /r.
    #[inline]
    pub fn vpmovb2m(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.vpmov_to_mask(dst, src, TypeFlags::T_W0, 0x29)
    }

    /// `vpmovq2m k, xmm/ymm/zmm` — EVEX.F3.0F38.W1 39 /r.
    #[inline]
    pub fn vpmovq2m(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.vpmov_to_mask(dst, src, TypeFlags::T_EW1, 0x39)
    }

    // ── MOVSS / MOVSD (special: reg,reg uses different pattern than reg,mem) ─
    /// `movss xmm, xmm/m32` — F3 0F 10 /r
    #[inline]
    pub fn movss(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_F3 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_F3 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }
    /// `movsd xmm, xmm/m64` — F2 0F 10 /r
    #[inline]
    pub fn movsd(&mut self, dst: impl Into<RegMem>, src: impl Into<RegMem>) -> Result<()> {
        let dst = dst.into();
        let src = src.into();
        match (&dst, &src) {
            (RegMem::Reg(d), _) => {
                self.buf
                    .op_sse(d, &src, TypeFlags::T_F2 | TypeFlags::T_0F, 0x10, None)
            }
            (RegMem::Mem(m), RegMem::Reg(s)) => {
                self.buf
                    .op_mr(m, s, TypeFlags::T_F2 | TypeFlags::T_0F, 0x11)
            }
            _ => Err(Error::BadCombination),
        }
    }
    /// `vmovss xmm, xmm, xmm` or `vmovss xmm, m32` or `vmovss m32, xmm`
    #[inline]
    pub fn vmovss(
        &mut self,
        dst: impl Into<RegMem>,
        src1: impl Into<RegMem>,
        src2: Option<Reg>,
    ) -> Result<()> {
        let dst = dst.into();
        let src1 = src1.into();
        match (&dst, &src1, src2) {
            // vmovss xmm, xmm, xmm — 3-operand reg form
            (RegMem::Reg(d), RegMem::Reg(s1), Some(s2)) => self.buf.op_vex(
                d,
                Some(s1),
                &RegMem::Reg(s2),
                TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4,
                0x10,
                None,
            ),
            // vmovss xmm, m32 — load
            (RegMem::Reg(d), RegMem::Mem(_), None) => self.buf.op_vex(
                d,
                None,
                &src1,
                TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4,
                0x10,
                None,
            ),
            // vmovss m32, xmm — store
            (RegMem::Mem(m), RegMem::Reg(s), None) => self.buf.op_vex(
                s,
                None,
                &RegMem::Mem(*m),
                TypeFlags::T_F3 | TypeFlags::T_0F | TypeFlags::T_EVEX | TypeFlags::T_N4,
                0x11,
                None,
            ),
            _ => Err(Error::BadCombination),
        }
    }
    /// `vmovsd xmm, xmm, xmm` or `vmovsd xmm, m64` or `vmovsd m64, xmm`
    #[inline]
    pub fn vmovsd(
        &mut self,
        dst: impl Into<RegMem>,
        src1: impl Into<RegMem>,
        src2: Option<Reg>,
    ) -> Result<()> {
        let dst = dst.into();
        let src1 = src1.into();
        match (&dst, &src1, src2) {
            (RegMem::Reg(d), RegMem::Reg(s1), Some(s2)) => self.buf.op_vex(
                d,
                Some(s1),
                &RegMem::Reg(s2),
                TypeFlags::T_F2
                    | TypeFlags::T_0F
                    | TypeFlags::T_EVEX
                    | TypeFlags::T_EW1
                    | TypeFlags::T_N8,
                0x10,
                None,
            ),
            (RegMem::Reg(d), RegMem::Mem(_), None) => self.buf.op_vex(
                d,
                None,
                &src1,
                TypeFlags::T_F2
                    | TypeFlags::T_0F
                    | TypeFlags::T_EVEX
                    | TypeFlags::T_EW1
                    | TypeFlags::T_N8,
                0x10,
                None,
            ),
            (RegMem::Mem(m), RegMem::Reg(s), None) => self.buf.op_vex(
                s,
                None,
                &RegMem::Mem(*m),
                TypeFlags::T_F2
                    | TypeFlags::T_0F
                    | TypeFlags::T_EVEX
                    | TypeFlags::T_EW1
                    | TypeFlags::T_N8,
                0x11,
                None,
            ),
            _ => Err(Error::BadCombination),
        }
    }

    // ── Cache prefetch ────────────────────────────────────────
    /// `prefetchnta [m]` — 0F 18 /0
    #[inline]
    pub fn prefetchnta(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetcht0 [m]` — 0F 18 /1
    #[inline]
    pub fn prefetcht0(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(1, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetcht1 [m]` — 0F 18 /2
    #[inline]
    pub fn prefetcht1(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(2, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetcht2 [m]` — 0F 18 /3
    #[inline]
    pub fn prefetcht2(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(3, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetchrst2 [m]` — 0F 18 /4
    #[inline]
    pub fn prefetchrst2(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(4, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetchit1 [m]` — 0F 18 /6
    #[inline]
    pub fn prefetchit1(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(6, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetchit0 [m]` — 0F 18 /7
    #[inline]
    pub fn prefetchit0(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(7, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x18,
        )
    }
    /// `prefetchw [m]` — 0F 0D /1
    #[inline]
    pub fn prefetchw(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(1, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x0D,
        )
    }
    /// `clflush [m]` — 0F AE /7
    #[inline]
    pub fn clflush(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(7, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    /// `clflushopt [m]` — 66 0F AE /7
    #[inline]
    pub fn clflushopt(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(7, crate::operand::Kind::Reg, 32);
        self.buf.op_mr(
            &addr,
            &r,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    /// `cldemote [m]` — 0F 1C /0
    #[inline]
    pub fn cldemote(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::gpr32(0),
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x1C,
        )
    }
    /// `clwb [m]` — 66 0F AE /6
    #[inline]
    pub fn clwb(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::gpr32(6),
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    /// `movrs r8/r16/r32/r64, [m]` — 0F 38 8A/8B /r
    #[inline]
    pub fn movrs(&mut self, reg: Reg, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &reg,
            TypeFlags::T_0F38,
            if reg.is_bit(8) { 0x8A } else { 0x8B },
        )
    }

    // ── MOVMSKPS / MOVMSKPD ──────────────────────────────────
    /// `movmskps r32, xmm` — 0F 50 /r
    #[inline]
    pub fn movmskps(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_rr(
            &dst,
            &src,
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x50,
        )
    }
    /// `movmskpd r32, xmm` — 66 0F 50 /r
    #[inline]
    pub fn movmskpd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_rr(
            &dst,
            &src,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0x50,
        )
    }
    /// `vmovmskps r32, xmm/ymm` — VEX 0F 50 /r
    #[inline]
    pub fn vmovmskps(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_0F | TypeFlags::T_YMM,
            0x50,
            None,
        )
    }
    /// `vmovmskpd r32, xmm/ymm` — VEX.66 0F 50 /r
    #[inline]
    pub fn vmovmskpd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            0x50,
            None,
        )
    }
    /// `vpmovmskb r32, xmm/ymm` — VEX.66 0F D7 /r
    #[inline]
    pub fn vpmovmskb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_YMM,
            0xD7,
            None,
        )
    }

    // ── CVTSS2SI / CVTSD2SI / CVTTSS2SI / CVTTSD2SI ─────────
    /// `cvttss2si r32/r64, xmm/m32` — F3 0F 2C /r
    #[inline]
    pub fn cvttss2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x2C,
            None,
        )
    }
    /// `cvttsd2si r32/r64, xmm/m64` — F2 0F 2C /r
    #[inline]
    pub fn cvttsd2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x2C,
            None,
        )
    }
    /// `cvtss2si r32/r64, xmm/m32` — F3 0F 2D /r
    #[inline]
    pub fn cvtss2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x2D,
            None,
        )
    }
    /// `cvtsd2si r32/r64, xmm/m64` — F2 0F 2D /r
    #[inline]
    pub fn cvtsd2si(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F2 | TypeFlags::T_0F,
            0x2D,
            None,
        )
    }

    // ── CVTDQ2PS / CVTPS2DQ / CVTTPS2DQ ─────────────────────
    /// `cvtdq2ps xmm, xmm/m128` — 0F 5B /r
    #[inline]
    pub fn cvtdq2ps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x5B, None)
    }
    /// `cvtps2dq xmm, xmm/m128` — 66 0F 5B /r
    #[inline]
    pub fn cvtps2dq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x5B,
            None,
        )
    }
    /// `cvttps2dq xmm, xmm/m128` — F3 0F 5B /r
    #[inline]
    pub fn cvttps2dq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_F3 | TypeFlags::T_0F,
            0x5B,
            None,
        )
    }

    // ── PUNPCK / PACK / UNPACK ────────────────────────────────
    /// `punpcklbw xmm, xmm/m128` — 66 0F 60 /r
    #[inline]
    pub fn punpcklbw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x60,
            None,
        )
    }
    /// `punpcklwd xmm, xmm/m128` — 66 0F 61 /r
    #[inline]
    pub fn punpcklwd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x61,
            None,
        )
    }
    /// `punpckldq xmm, xmm/m128` — 66 0F 62 /r
    #[inline]
    pub fn punpckldq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x62,
            None,
        )
    }
    /// `punpcklqdq xmm, xmm/m128` — 66 0F 6C /r
    #[inline]
    pub fn punpcklqdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x6C,
            None,
        )
    }
    /// `punpckhbw xmm, xmm/m128` — 66 0F 68 /r
    #[inline]
    pub fn punpckhbw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x68,
            None,
        )
    }
    /// `punpckhwd xmm, xmm/m128` — 66 0F 69 /r
    #[inline]
    pub fn punpckhwd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x69,
            None,
        )
    }
    /// `punpckhdq xmm, xmm/m128` — 66 0F 6A /r
    #[inline]
    pub fn punpckhdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x6A,
            None,
        )
    }
    /// `punpckhqdq xmm, xmm/m128` — 66 0F 6D /r
    #[inline]
    pub fn punpckhqdq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x6D,
            None,
        )
    }
    /// `packsswb xmm, xmm/m128` — 66 0F 63 /r
    #[inline]
    pub fn packsswb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x63,
            None,
        )
    }
    /// `packssdw xmm, xmm/m128` — 66 0F 6B /r
    #[inline]
    pub fn packssdw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x6B,
            None,
        )
    }
    /// `packuswb xmm, xmm/m128` — 66 0F 67 /r
    #[inline]
    pub fn packuswb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x67,
            None,
        )
    }
    /// `packusdw xmm, xmm/m128` — 66 0F 38 2B /r
    #[inline]
    pub fn packusdw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F38,
            0x2B,
            None,
        )
    }
    /// `unpcklps xmm, xmm/m128` — 0F 14 /r
    #[inline]
    pub fn unpcklps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x14, None)
    }
    /// `unpckhps xmm, xmm/m128` — 0F 15 /r
    #[inline]
    pub fn unpckhps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf
            .op_sse(&dst, &src.into(), TypeFlags::T_0F, 0x15, None)
    }
    /// `unpcklpd xmm, xmm/m128` — 66 0F 14 /r
    #[inline]
    pub fn unpcklpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x14,
            None,
        )
    }
    /// `unpckhpd xmm, xmm/m128` — 66 0F 15 /r
    #[inline]
    pub fn unpckhpd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.buf.op_sse(
            &dst,
            &src.into(),
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x15,
            None,
        )
    }

    // ── PSLL / PSRL / PSRA immediate shifts ─────────────────
    /// Encode the VEX.NDD packed-shift immediate form. Unlike the register-count
    /// form, the ModRM.reg field is an opcode extension and the destination is
    /// encoded in VEX.vvvv.
    #[inline]
    fn vex_packed_shift_imm(
        &mut self,
        dst: Reg,
        src: Reg,
        opcode_extension: u8,
        type_flags: TypeFlags,
        opcode: u8,
        imm: u8,
    ) -> Result<()> {
        let same_width = (dst.is_xmm() && src.is_xmm())
            || (dst.is_ymm() && src.is_ymm())
            || (dst.is_zmm() && src.is_zmm());
        if !same_width {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &Reg::xmm(opcode_extension),
            Some(&dst),
            &RegMem::Reg(src),
            type_flags,
            opcode,
            Some(imm),
        )
    }

    /// `vpsllw xmm/ymm, xmm/ymm, imm8` — VEX.66.0F 71 /6 ib
    #[inline]
    pub fn vpsllw_imm(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.vex_packed_shift_imm(
            dst,
            src,
            6,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_YMM
                | TypeFlags::T_EVEX
                | TypeFlags::T_MEM_EVEX,
            0x71,
            imm,
        )
    }

    /// `vpsrlw xmm/ymm, xmm/ymm, imm8` — VEX.66.0F 71 /2 ib
    #[inline]
    pub fn vpsrlw_imm(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.vex_packed_shift_imm(
            dst,
            src,
            2,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_YMM
                | TypeFlags::T_EVEX
                | TypeFlags::T_MEM_EVEX,
            0x71,
            imm,
        )
    }

    /// `vpsrld xmm/ymm, xmm/ymm, imm8` — VEX.66.0F 72 /2 ib
    #[inline]
    pub fn vpsrld_imm(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.vex_packed_shift_imm(
            dst,
            src,
            2,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_EVEX
                | TypeFlags::T_B32
                | TypeFlags::T_MEM_EVEX,
            0x72,
            imm,
        )
    }

    /// `vpsllq xmm/ymm/zmm, xmm/ymm/zmm, imm8` — VEX/EVEX.66.0F.W1 73 /6 ib
    #[inline]
    pub fn vpsllq_imm(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.vex_packed_shift_imm(
            dst,
            src,
            6,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_EVEX
                | TypeFlags::T_B64
                | TypeFlags::T_MEM_EVEX,
            0x73,
            imm,
        )
    }

    /// `vpsraq xmm/ymm/zmm, xmm/ymm/zmm, imm8` — EVEX.66.0F.W1 72 /4 ib
    #[inline]
    pub fn vpsraq_imm(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.vex_packed_shift_imm(
            dst,
            src,
            4,
            TypeFlags::T_66
                | TypeFlags::T_0F
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B64,
            0x72,
            imm,
        )
    }

    /// `pslld xmm, imm8` — 66 0F 72 /6 ib
    #[inline]
    pub fn pslld_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            6,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x72,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psllq xmm, imm8` — 66 0F 73 /6 ib
    #[inline]
    pub fn psllq_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            6,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x73,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psrld xmm, imm8` — 66 0F 72 /2 ib
    #[inline]
    pub fn psrld_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            2,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x72,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psrlq xmm, imm8` — 66 0F 73 /2 ib
    #[inline]
    pub fn psrlq_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            2,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x73,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psrad xmm, imm8` — 66 0F 72 /4 ib
    #[inline]
    pub fn psrad_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            4,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x72,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psllw xmm, imm8` — 66 0F 71 /6 ib
    #[inline]
    pub fn psllw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            6,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x71,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psrlw xmm, imm8` — 66 0F 71 /2 ib
    #[inline]
    pub fn psrlw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            2,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x71,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psraw xmm, imm8` — 66 0F 71 /4 ib
    #[inline]
    pub fn psraw_imm(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            4,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x71,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `pslldq xmm, imm8` — 66 0F 73 /7 ib (shift left by bytes)
    #[inline]
    pub fn pslldq(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            7,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x73,
            1,
        )?;
        self.buf.db(imm)
    }
    /// `psrldq xmm, imm8` — 66 0F 73 /3 ib (shift right by bytes)
    #[inline]
    pub fn psrldq(&mut self, dst: Reg, imm: u8) -> Result<()> {
        self.buf.op_rext(
            &RegMem::Reg(dst),
            3,
            TypeFlags::T_66 | TypeFlags::T_0F,
            0x73,
            1,
        )?;
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
        &mut self,
        dst: Reg,
        src: impl Into<RegMem>,
        km_type: TypeFlags,
        gpr_type: TypeFlags,
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
    #[inline]
    pub fn kmovw(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let t = TypeFlags::T_0F | TypeFlags::T_W0;
        self.kmov_dispatch(dst, src, t, t)
    }
    /// `kmovb` — VEX.L0.66.0F.W0 {90,92,93} /r — auto-detects k↔k/m vs GPR forms.
    #[inline]
    pub fn kmovb(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W0;
        self.kmov_dispatch(dst, src, t, t)
    }
    /// `kmovd` — auto-detects k↔k/m (VEX.66.0F.W1 90) vs GPR (VEX.F2.0F.W0 92/93).
    #[inline]
    pub fn kmovd(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.kmov_dispatch(
            dst,
            src,
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W1,
            TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_W0,
        )
    }
    /// `kmovq` — auto-detects k↔k/m (VEX.0F.W1 90) vs GPR (VEX.F2.0F.W1 92/93).
    #[inline]
    pub fn kmovq(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.kmov_dispatch(
            dst,
            src,
            TypeFlags::T_0F | TypeFlags::T_W1,
            TypeFlags::T_F2 | TypeFlags::T_0F | TypeFlags::T_W1,
        )
    }

    // Store forms: kmov m, k
    /// `kmovw m16, k` — VEX.L0.0F.W0 91 /r
    #[inline]
    pub fn kmovw_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_0F | TypeFlags::T_W0,
            0x91,
            None,
        )
    }
    /// `kmovb m8, k` — VEX.L0.66.0F.W0 91 /r
    #[inline]
    pub fn kmovb_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W0,
            0x91,
            None,
        )
    }
    /// `kmovd m32, k` — VEX.L0.66.0F.W1 91 /r
    #[inline]
    pub fn kmovd_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_W1,
            0x91,
            None,
        )
    }
    /// `kmovq m64, k` — VEX.L0.0F.W1 91 /r
    #[inline]
    pub fn kmovq_store(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_0F | TypeFlags::T_W1,
            0x91,
            None,
        )
    }

    // ── KAND / KOR / KXOR / KANDN / KXNOR ────────────────────
    // All: VEX.L1.0F.Wx 41-47 /r  (3-op: k, k, k)
    // W0=word, W1(66)=byte, W1(plain)=dword/qword etc.

    /// Helper for 3-operand k-register operations
    fn k_op3(&mut self, dst: Reg, src1: Reg, src2: Reg, type_: TypeFlags, code: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src1),
            &RegMem::Reg(src2),
            type_ | TypeFlags::T_L1 | TypeFlags::T_0F,
            code,
            None,
        )
    }

    // KAND
    #[inline]
    pub fn kandw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x41)
    }
    #[inline]
    pub fn kandb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x41)
    }
    #[inline]
    pub fn kandd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x41)
    }
    #[inline]
    pub fn kandq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x41)
    }

    // KANDN
    #[inline]
    pub fn kandnw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x42)
    }
    #[inline]
    pub fn kandnb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x42)
    }
    #[inline]
    pub fn kandnd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x42)
    }
    #[inline]
    pub fn kandnq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x42)
    }

    // KOR
    #[inline]
    pub fn korw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x45)
    }
    #[inline]
    pub fn korb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x45)
    }
    #[inline]
    pub fn kord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x45)
    }
    #[inline]
    pub fn korq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x45)
    }

    // KXOR
    #[inline]
    pub fn kxorw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x47)
    }
    #[inline]
    pub fn kxorb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x47)
    }
    #[inline]
    pub fn kxord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x47)
    }
    #[inline]
    pub fn kxorq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x47)
    }

    // KXNOR
    #[inline]
    pub fn kxnorw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x46)
    }
    #[inline]
    pub fn kxnorb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x46)
    }
    #[inline]
    pub fn kxnord(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x46)
    }
    #[inline]
    pub fn kxnorq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x46)
    }

    // KADD
    #[inline]
    pub fn kaddw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x4A)
    }
    #[inline]
    pub fn kaddb(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x4A)
    }
    #[inline]
    pub fn kaddd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W1, 0x4A)
    }
    #[inline]
    pub fn kaddq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x4A)
    }

    // KUNPCK
    #[inline]
    pub fn kunpckbw(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_66 | TypeFlags::T_W0, 0x4B)
    }
    #[inline]
    pub fn kunpckwd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W0, 0x4B)
    }
    #[inline]
    pub fn kunpckdq(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.k_op3(dst, src1, src2, TypeFlags::T_W1, 0x4B)
    }

    // ── KNOT / KORTEST / KTEST ────────────────────────────────
    // 2-operand k-register ops: VEX.L0.0F.Wx 44/98/99 /r (k, k)
    fn k_op2(&mut self, dst: Reg, src: Reg, type_: TypeFlags, code: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            type_ | TypeFlags::T_0F,
            code,
            None,
        )
    }

    // KNOT
    #[inline]
    pub fn knotw(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W0, 0x44)
    }
    #[inline]
    pub fn knotb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x44)
    }
    #[inline]
    pub fn knotd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x44)
    }
    #[inline]
    pub fn knotq(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W1, 0x44)
    }

    // KORTEST
    #[inline]
    pub fn kortestw(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W0, 0x98)
    }
    #[inline]
    pub fn kortestb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x98)
    }
    #[inline]
    pub fn kortestd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x98)
    }
    #[inline]
    pub fn kortestq(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W1, 0x98)
    }

    // KTEST
    #[inline]
    pub fn ktestw(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W0, 0x99)
    }
    #[inline]
    pub fn ktestb(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W0, 0x99)
    }
    #[inline]
    pub fn ktestd(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_66 | TypeFlags::T_W1, 0x99)
    }
    #[inline]
    pub fn ktestq(&mut self, dst: Reg, src: Reg) -> Result<()> {
        self.k_op2(dst, src, TypeFlags::T_W1, 0x99)
    }

    // ── KSHIFT ────────────────────────────────────────────────
    // kshiftl/r: VEX.L0.66.0F3A.W1 32-33 /r ib (k, k, imm8)
    /// `kshiftlw k, k, imm8` — VEX.L0.66.0F3A.W1 32 /r ib
    #[inline]
    pub fn kshiftlw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x32,
            Some(imm),
        )
    }
    /// `kshiftlb k, k, imm8`
    #[inline]
    pub fn kshiftlb(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0,
            0x32,
            Some(imm),
        )
    }
    /// `kshiftld k, k, imm8`
    #[inline]
    pub fn kshiftld(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0,
            0x33,
            Some(imm),
        )
    }
    /// `kshiftlq k, k, imm8`
    #[inline]
    pub fn kshiftlq(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x33,
            Some(imm),
        )
    }
    /// `kshiftrw k, k, imm8` — VEX.L0.66.0F3A.W1 30 /r ib
    #[inline]
    pub fn kshiftrw(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x30,
            Some(imm),
        )
    }
    /// `kshiftrb k, k, imm8`
    #[inline]
    pub fn kshiftrb(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0,
            0x30,
            Some(imm),
        )
    }
    /// `kshiftrd k, k, imm8`
    #[inline]
    pub fn kshiftrd(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W0,
            0x31,
            Some(imm),
        )
    }
    /// `kshiftrq k, k, imm8`
    #[inline]
    pub fn kshiftrq(&mut self, dst: Reg, src: Reg, imm: u8) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_W1,
            0x31,
            Some(imm),
        )
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

    /// Xbyak `opFpuMem`: select the escape byte from the declared memory size.
    #[allow(clippy::too_many_arguments)]
    fn fpu_mem_by_size(
        &mut self,
        addr: &Address,
        m16: u8,
        m32: u8,
        m64: u8,
        mut ext: u8,
        m64_ext: u8,
    ) -> Result<()> {
        if addr.is_64bit_disp() {
            return Err(Error::CantUse64BitDisp);
        }
        let code = match addr.get_bit() {
            16 => m16,
            32 => m32,
            64 => m64,
            _ => 0,
        };
        if code == 0 {
            return Err(Error::BadMemSize);
        }
        if m64_ext != 0 && addr.get_bit() == 64 {
            ext = m64_ext;
        }
        self.buf
            .emit_rex_for_reg_mem(&Reg::fpu(0), addr, TypeFlags::NONE)?;
        self.buf.db(code)?;
        self.buf.emit_addr(addr, ext)
    }

    // ── FLD / FST / FSTP ──────────────────────────────────────
    /// `fld st(i)` — D9 C0+i
    #[inline]
    pub fn fld_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD9, 0xC0, src)
    }
    /// `fld m32fp` — D9 /0
    #[inline]
    pub fn fld_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD9, 0, &addr)
    }
    /// `fld m64fp` — DD /0
    #[inline]
    pub fn fld_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDD, 0, &addr)
    }
    /// `fld m80fp` — DB /5
    #[inline]
    pub fn fld_m80(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 5, &addr)
    }
    /// `fst st(i)` — DD D0+i
    #[inline]
    pub fn fst_st(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDD, 0xD0, dst)
    }
    /// `fst m32fp` — D9 /2
    #[inline]
    pub fn fst_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD9, 2, &addr)
    }
    /// `fst m64fp` — DD /2
    #[inline]
    pub fn fst_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDD, 2, &addr)
    }
    /// `fstp st(i)` — DD D8+i
    #[inline]
    pub fn fstp_st(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDD, 0xD8, dst)
    }
    /// `fstp m32fp` — D9 /3
    #[inline]
    pub fn fstp_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD9, 3, &addr)
    }
    /// `fstp m64fp` — DD /3
    #[inline]
    pub fn fstp_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDD, 3, &addr)
    }
    /// `fstp m80fp` — DB /7
    #[inline]
    pub fn fstp_m80(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 7, &addr)
    }

    // ── FILD / FIST / FISTP / FISTTP ──────────────────────────
    /// `fild m16int` — DF /0
    #[inline]
    pub fn fild_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 0, &addr)
    }
    /// `fild m32int` — DB /0
    #[inline]
    pub fn fild_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 0, &addr)
    }
    /// `fild m64int` — DF /5
    #[inline]
    pub fn fild_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 5, &addr)
    }
    /// `fist m16int` — DF /2
    #[inline]
    pub fn fist_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 2, &addr)
    }
    /// `fist m32int` — DB /2
    #[inline]
    pub fn fist_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 2, &addr)
    }
    /// `fistp m16int` — DF /3
    #[inline]
    pub fn fistp_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 3, &addr)
    }
    /// `fistp m32int` — DB /3
    #[inline]
    pub fn fistp_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 3, &addr)
    }
    /// `fistp m64int` — DF /7
    #[inline]
    pub fn fistp_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 7, &addr)
    }
    /// `fisttp m16int` — DF /1
    #[inline]
    pub fn fisttp_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDF, 1, &addr)
    }
    /// `fisttp m32int` — DB /1
    #[inline]
    pub fn fisttp_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDB, 1, &addr)
    }
    /// `fisttp m64int` — DD /1
    #[inline]
    pub fn fisttp_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDD, 1, &addr)
    }

    // ── FADD / FSUB / FMUL / FDIV (register forms) ───────────
    /// `fadd st(0), st(i)` — D8 C0+i
    #[inline]
    pub fn fadd_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xC0, src)
    }
    /// `fadd st(i), st(0)` — DC C0+i
    #[inline]
    pub fn fadd_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xC0, dst)
    }
    /// `faddp st(i), st(0)` — DE C0+i
    #[inline]
    pub fn faddp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xC0, dst)
    }
    /// `fadd m32fp` — D8 /0
    #[inline]
    pub fn fadd_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD8, 0, &addr)
    }
    /// `fadd m64fp` — DC /0
    #[inline]
    pub fn fadd_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDC, 0, &addr)
    }

    /// `fsub st(0), st(i)` — D8 E0+i
    #[inline]
    pub fn fsub_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xE0, src)
    }
    /// `fsub st(i), st(0)` — DC E8+i
    #[inline]
    pub fn fsub_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xE8, dst)
    }
    /// `fsubp st(i), st(0)` — DE E8+i
    #[inline]
    pub fn fsubp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xE8, dst)
    }
    /// `fsubr st(0), st(i)` — D8 E8+i
    #[inline]
    pub fn fsubr_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xE8, src)
    }
    /// `fsubr st(i), st(0)` — DC E0+i
    #[inline]
    pub fn fsubr_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xE0, dst)
    }
    /// `fsubrp st(i), st(0)` — DE E0+i
    #[inline]
    pub fn fsubrp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xE0, dst)
    }
    /// `fsub m32fp` — D8 /4
    #[inline]
    pub fn fsub_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD8, 4, &addr)
    }
    /// `fsub m64fp` — DC /4
    #[inline]
    pub fn fsub_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDC, 4, &addr)
    }

    /// `fmul st(0), st(i)` — D8 C8+i
    #[inline]
    pub fn fmul_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xC8, src)
    }
    /// `fmul st(i), st(0)` — DC C8+i
    #[inline]
    pub fn fmul_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xC8, dst)
    }
    /// `fmulp st(i), st(0)` — DE C8+i
    #[inline]
    pub fn fmulp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xC8, dst)
    }
    /// `fmul m32fp` — D8 /1
    #[inline]
    pub fn fmul_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD8, 1, &addr)
    }
    /// `fmul m64fp` — DC /1
    #[inline]
    pub fn fmul_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDC, 1, &addr)
    }

    /// `fdiv st(0), st(i)` — D8 F0+i
    #[inline]
    pub fn fdiv_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xF0, src)
    }
    /// `fdiv st(i), st(0)` — DC F8+i
    #[inline]
    pub fn fdiv_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xF8, dst)
    }
    /// `fdivp st(i), st(0)` — DE F8+i
    #[inline]
    pub fn fdivp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xF8, dst)
    }
    /// `fdivr st(0), st(i)` — D8 F8+i
    #[inline]
    pub fn fdivr_st0_st(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xF8, src)
    }
    /// `fdivr st(i), st(0)` — DC F0+i
    #[inline]
    pub fn fdivr_st_st0(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDC, 0xF0, dst)
    }
    /// `fdivrp st(i), st(0)` — DE F0+i
    #[inline]
    pub fn fdivrp(&mut self, dst: Reg) -> Result<()> {
        self.fpu_st(0xDE, 0xF0, dst)
    }
    /// `fdiv m32fp` — D8 /6
    #[inline]
    pub fn fdiv_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD8, 6, &addr)
    }
    /// `fdiv m64fp` — DC /6
    #[inline]
    pub fn fdiv_m64(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDC, 6, &addr)
    }

    // ── FCOM / FCOMP / FCOMPP ─────────────────────────────────
    /// `fcom st(i)` — D8 D0+i
    #[inline]
    pub fn fcom(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xD0, src)
    }
    /// `fcomp st(i)` — D8 D8+i
    #[inline]
    pub fn fcomp(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xD8, 0xD8, src)
    }
    /// `fcompp` — DE D9
    #[inline]
    pub fn fcompp(&mut self) -> Result<()> {
        self.buf.db(0xDE)?;
        self.buf.db(0xD9)
    }
    /// `fucom st(i)` — DD E0+i
    #[inline]
    pub fn fucom(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDD, 0xE0, src)
    }
    /// `fucomp st(i)` — DD E8+i
    #[inline]
    pub fn fucomp(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDD, 0xE8, src)
    }
    /// `fucompp` — DA E9
    #[inline]
    pub fn fucompp(&mut self) -> Result<()> {
        self.buf.db(0xDA)?;
        self.buf.db(0xE9)
    }
    /// `fucomi st(0), st(i)` — DB E8+i
    #[inline]
    pub fn fucomi(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xE8, src)
    }
    /// `fucomip st(0), st(i)` — DF E8+i
    #[inline]
    pub fn fucomip(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDF, 0xE8, src)
    }
    /// `fcomi st(0), st(i)` — DB F0+i
    #[inline]
    pub fn fcomi(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xF0, src)
    }
    /// `fcomip st(0), st(i)` — DF F0+i
    #[inline]
    pub fn fcomip(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDF, 0xF0, src)
    }

    // ── Unary / miscellaneous FPU ─────────────────────────────
    /// `fchs` — D9 E0
    #[inline]
    pub fn fchs(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE0)
    }
    /// `fabs` — D9 E1
    #[inline]
    pub fn fabs(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE1)
    }
    /// `fsqrt` — D9 FA
    #[inline]
    pub fn fsqrt(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFA)
    }
    /// `fsin` — D9 FE
    #[inline]
    pub fn fsin(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFE)
    }
    /// `fsincos` — D9 FB
    #[inline]
    pub fn fsincos(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFB)
    }
    /// `fcos` — D9 FF
    #[inline]
    pub fn fcos(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFF)
    }
    /// `fptan` — D9 F2
    #[inline]
    pub fn fptan(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF2)
    }
    /// `fpatan` — D9 F3
    #[inline]
    pub fn fpatan(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF3)
    }
    /// `frndint` — D9 FC
    #[inline]
    pub fn frndint(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFC)
    }
    /// `fscale` — D9 FD
    #[inline]
    pub fn fscale(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xFD)
    }
    /// `f2xm1` — D9 F0
    #[inline]
    pub fn f2xm1(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF0)
    }
    /// `fyl2x` — D9 F1
    #[inline]
    pub fn fyl2x(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF1)
    }
    /// `fyl2xp1` — D9 F9
    #[inline]
    pub fn fyl2xp1(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF9)
    }
    /// `fprem` — D9 F8
    #[inline]
    pub fn fprem(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF8)
    }
    /// `fprem1` — D9 F5
    #[inline]
    pub fn fprem1(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF5)
    }
    /// `fxtract` — D9 F4
    #[inline]
    pub fn fxtract(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF4)
    }
    /// `ftst` — D9 E4
    #[inline]
    pub fn ftst(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE4)
    }
    /// `fxam` — D9 E5
    #[inline]
    pub fn fxam(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE5)
    }

    // ── FPU Exchange ──────────────────────────────────────────
    /// `fxch st(i)` — D9 C8+i
    #[inline]
    pub fn fxch(&mut self, st: Reg) -> Result<()> {
        self.fpu_st(0xD9, 0xC8, st)
    }

    // ── FPU Constants ─────────────────────────────────────────
    /// `fldz` — D9 EE (push +0.0)
    #[inline]
    pub fn fldz(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xEE)
    }
    /// `fld1` — D9 E8 (push +1.0)
    #[inline]
    pub fn fld1(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE8)
    }
    /// `fldpi` — D9 EB (push pi)
    #[inline]
    pub fn fldpi(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xEB)
    }
    /// `fldl2t` — D9 E9 (push log2(10))
    #[inline]
    pub fn fldl2t(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xE9)
    }
    /// `fldl2e` — D9 EA (push log2(e))
    #[inline]
    pub fn fldl2e(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xEA)
    }
    /// `fldlg2` — D9 EC (push log10(2))
    #[inline]
    pub fn fldlg2(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xEC)
    }
    /// `fldln2` — D9 ED (push ln(2))
    #[inline]
    pub fn fldln2(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xED)
    }

    // ── FPU Control ───────────────────────────────────────────
    /// `fwait` / `wait` — 9B
    #[inline]
    pub fn fwait(&mut self) -> Result<()> {
        self.buf.db(0x9B)
    }
    #[inline]
    pub fn wait(&mut self) -> Result<()> {
        self.fwait()
    }
    /// `finit` — 9B DB E3
    #[inline]
    pub fn finit(&mut self) -> Result<()> {
        self.buf.db(0x9B)?;
        self.buf.db(0xDB)?;
        self.buf.db(0xE3)
    }
    /// `fninit` — DB E3
    #[inline]
    pub fn fninit(&mut self) -> Result<()> {
        self.buf.db(0xDB)?;
        self.buf.db(0xE3)
    }
    /// `fldcw m16` — D9 /5
    #[inline]
    pub fn fldcw(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD9, 5, &addr)
    }
    /// `fnstcw m16` — D9 /7
    #[inline]
    pub fn fnstcw(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xD9, 7, &addr)
    }
    /// `fstcw m16` — 9B D9 /7
    #[inline]
    pub fn fstcw(&mut self, addr: Address) -> Result<()> {
        self.buf.db(0x9B)?;
        self.fpu_mem(0xD9, 7, &addr)
    }
    /// `fnstsw m16` — DD /7
    #[inline]
    pub fn fnstsw(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDD, 7, &addr)
    }
    /// `fnstsw ax` — DF E0
    #[inline]
    pub fn fnstsw_ax(&mut self) -> Result<()> {
        self.buf.db(0xDF)?;
        self.buf.db(0xE0)
    }
    /// `fstsw ax` — 9B DF E0
    #[inline]
    pub fn fstsw_ax(&mut self) -> Result<()> {
        self.buf.db(0x9B)?;
        self.buf.db(0xDF)?;
        self.buf.db(0xE0)
    }
    /// `fstsw m16` — 9B DD /7
    #[inline]
    pub fn fstsw(&mut self, addr: Address) -> Result<()> {
        self.buf.db(0x9B)?;
        self.buf
            .op_mr(&addr, &Reg::gpr32(7), TypeFlags::T_ALLOW_DIFF_SIZE, 0xDD)
    }
    /// Xbyak's register overload accepts AX and rejects every other Reg16.
    #[inline]
    pub fn fstsw_reg(&mut self, reg: Reg) -> Result<()> {
        if !reg.is_reg_bit(16) || reg.get_idx() != 0 {
            return Err(Error::BadParameter);
        }
        self.fstsw_ax()
    }
    /// `fbld m80bcd` — DF /4
    #[inline]
    pub fn fbld(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(4), TypeFlags::T_ALLOW_DIFF_SIZE, 0xDF)
    }
    /// `fbstp m80bcd` — DF /6
    #[inline]
    pub fn fbstp(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(6), TypeFlags::T_ALLOW_DIFF_SIZE, 0xDF)
    }
    /// `fldenv [mem]` — D9 /4
    #[inline]
    pub fn fldenv(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(4), TypeFlags::T_ALLOW_DIFF_SIZE, 0xD9)
    }
    /// `fnsave [mem]` — DD /6
    #[inline]
    pub fn fnsave(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(6), TypeFlags::T_ALLOW_DIFF_SIZE, 0xDD)
    }
    /// `fnstenv [mem]` — D9 /6
    #[inline]
    pub fn fnstenv(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(6), TypeFlags::T_ALLOW_DIFF_SIZE, 0xD9)
    }
    /// `frstor [mem]` — DD /4
    #[inline]
    pub fn frstor(&mut self, addr: Address) -> Result<()> {
        self.buf
            .op_mr(&addr, &Reg::gpr32(4), TypeFlags::T_ALLOW_DIFF_SIZE, 0xDD)
    }
    /// `fsave [mem]` — WAIT + DD /6
    #[inline]
    pub fn fsave(&mut self, addr: Address) -> Result<()> {
        self.buf.db(0x9B)?;
        self.fnsave(addr)
    }
    /// `fstenv [mem]` — WAIT + D9 /6
    #[inline]
    pub fn fstenv(&mut self, addr: Address) -> Result<()> {
        self.buf.db(0x9B)?;
        self.fnstenv(addr)
    }
    /// `fxrstor [mem]` — 0F AE /1
    #[inline]
    pub fn fxrstor(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::gpr32(1),
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    /// `fxrstor64 [mem]` — REX.W + 0F AE /1
    #[inline]
    pub fn fxrstor64(&mut self, addr: Address) -> Result<()> {
        self.buf.op_mr(
            &addr,
            &Reg::gpr64(1),
            TypeFlags::T_0F | TypeFlags::T_ALLOW_DIFF_SIZE,
            0xAE,
        )
    }
    /// `fclex` — 9B DB E2
    #[inline]
    pub fn fclex(&mut self) -> Result<()> {
        self.buf.db(0x9B)?;
        self.buf.db(0xDB)?;
        self.buf.db(0xE2)
    }
    /// `fnclex` — DB E2
    #[inline]
    pub fn fnclex(&mut self) -> Result<()> {
        self.buf.db(0xDB)?;
        self.buf.db(0xE2)
    }
    /// `fnop` — D9 D0
    #[inline]
    pub fn fnop(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xD0)
    }
    /// `fdecstp` — D9 F6
    #[inline]
    pub fn fdecstp(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF6)
    }
    /// `fincstp` — D9 F7
    #[inline]
    pub fn fincstp(&mut self) -> Result<()> {
        self.buf.db(0xD9)?;
        self.buf.db(0xF7)
    }
    /// `ffree st(i)` — DD C0+i
    #[inline]
    pub fn ffree(&mut self, st: Reg) -> Result<()> {
        self.fpu_st(0xDD, 0xC0, st)
    }

    // ── FIADD / FISUB / FIMUL / FIDIV (integer memory ops) ───
    /// `fiadd m16int` — DE /0
    #[inline]
    pub fn fiadd_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 0, &addr)
    }
    /// `fiadd m32int` — DA /0
    #[inline]
    pub fn fiadd_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 0, &addr)
    }
    /// `fisub m16int` — DE /4
    #[inline]
    pub fn fisub_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 4, &addr)
    }
    /// `fisub m32int` — DA /4
    #[inline]
    pub fn fisub_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 4, &addr)
    }
    /// `fimul m16int` — DE /1
    #[inline]
    pub fn fimul_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 1, &addr)
    }
    /// `fimul m32int` — DA /1
    #[inline]
    pub fn fimul_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 1, &addr)
    }
    /// `fidiv m16int` — DE /6
    #[inline]
    pub fn fidiv_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 6, &addr)
    }
    /// `fidiv m32int` — DA /6
    #[inline]
    pub fn fidiv_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 6, &addr)
    }
    /// `fidivr m16int/m32int` — DE/DA /7
    #[inline]
    pub fn fidivr(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem_by_size(&addr, 0xDE, 0xDA, 0, 7, 0)
    }
    /// `fisubr m16int/m32int` — DE/DA /5
    #[inline]
    pub fn fisubr(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem_by_size(&addr, 0xDE, 0xDA, 0, 5, 0)
    }
    /// `ficom m16int` — DE /2
    #[inline]
    pub fn ficom_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 2, &addr)
    }
    /// `ficom m32int` — DA /2
    #[inline]
    pub fn ficom_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 2, &addr)
    }
    /// `ficomp m16int` — DE /3
    #[inline]
    pub fn ficomp_m16(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDE, 3, &addr)
    }
    /// `ficomp m32int` — DA /3
    #[inline]
    pub fn ficomp_m32(&mut self, addr: Address) -> Result<()> {
        self.fpu_mem(0xDA, 3, &addr)
    }

    // ── FCMOV (conditional move) ──────────────────────────────
    /// `fcmovb st(0), st(i)` — DA C0+i
    #[inline]
    pub fn fcmovb(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDA, 0xC0, src)
    }
    /// `fcmove st(0), st(i)` — DA C8+i
    #[inline]
    pub fn fcmove(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDA, 0xC8, src)
    }
    /// `fcmovbe st(0), st(i)` — DA D0+i
    #[inline]
    pub fn fcmovbe(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDA, 0xD0, src)
    }
    /// `fcmovu st(0), st(i)` — DA D8+i
    #[inline]
    pub fn fcmovu(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDA, 0xD8, src)
    }
    /// `fcmovnb st(0), st(i)` — DB C0+i
    #[inline]
    pub fn fcmovnb(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xC0, src)
    }
    /// `fcmovne st(0), st(i)` — DB C8+i
    #[inline]
    pub fn fcmovne(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xC8, src)
    }
    /// `fcmovnbe st(0), st(i)` — DB D0+i
    #[inline]
    pub fn fcmovnbe(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xD0, src)
    }
    /// `fcmovnu st(0), st(i)` — DB D8+i
    #[inline]
    pub fn fcmovnu(&mut self, src: Reg) -> Result<()> {
        self.fpu_st(0xDB, 0xD8, src)
    }

    // ── ACE 1.15 block-scale register ─────────────────────────

    /// `bsrinit bsr0`
    pub fn bsrinit(&mut self, bsr: Reg) -> Result<()> {
        if !bsr.is_bsr() {
            return Err(Error::BadCombination);
        }
        self.buf.emit_vex(
            &bsr,
            &bsr,
            None,
            TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_W1,
            0x49,
            false,
        )?;
        self.buf.set_modrm(3, bsr.get_idx(), 0)
    }

    /// `bsrmovf bsr0, zmm, zmm/m512`
    pub fn bsrmovf(&mut self, bsr: Reg, zmm: Reg, op: impl Into<RegMem>) -> Result<()> {
        if !bsr.is_bsr() || !zmm.is_zmm() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &bsr,
            Some(&zmm),
            &op.into(),
            TypeFlags::T_MUST_EVEX | TypeFlags::T_MAP6 | TypeFlags::T_EW1 | TypeFlags::T_N1,
            0x95,
            None,
        )
    }

    /// `bsrmovh bsr0, zmm/m512`
    pub fn bsrmovh_load(&mut self, bsr: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.bsr_move_load(bsr, op.into(), TypeFlags::T_F2)
    }

    /// `bsrmovh zmm/m512, bsr0`
    pub fn bsrmovh_store(&mut self, op: impl Into<RegMem>, bsr: Reg) -> Result<()> {
        self.bsr_move_store(op.into(), bsr, TypeFlags::T_F2)
    }

    /// `bsrmovl bsr0, zmm/m512`
    pub fn bsrmovl_load(&mut self, bsr: Reg, op: impl Into<RegMem>) -> Result<()> {
        self.bsr_move_load(bsr, op.into(), TypeFlags::T_F3)
    }

    /// `bsrmovl zmm/m512, bsr0`
    pub fn bsrmovl_store(&mut self, op: impl Into<RegMem>, bsr: Reg) -> Result<()> {
        self.bsr_move_store(op.into(), bsr, TypeFlags::T_F3)
    }

    fn bsr_move_load(&mut self, bsr: Reg, op: RegMem, prefix: TypeFlags) -> Result<()> {
        if !bsr.is_bsr() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &bsr,
            None,
            &op,
            TypeFlags::T_N1
                | prefix
                | TypeFlags::T_MAP6
                | TypeFlags::T_EW1
                | TypeFlags::T_MUST_EVEX,
            0x95,
            None,
        )
    }

    fn bsr_move_store(&mut self, op: RegMem, bsr: Reg, prefix: TypeFlags) -> Result<()> {
        if !bsr.is_bsr() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &bsr,
            None,
            &op,
            TypeFlags::T_N1 | prefix | TypeFlags::T_MAP6 | TypeFlags::T_W0 | TypeFlags::T_MUST_EVEX,
            0x95,
            None,
        )
    }

    // ═══════════════════════════════════════════════════════════
    // AMX (Advanced Matrix Extensions) tile instructions
    // ═══════════════════════════════════════════════════════════

    // ACE 1.15 AMX instructions use EVEX with TMM destinations and ZMM
    // sources. The older AMX instructions below continue to use VEX.128.

    /// `tilemovcol tmm, zmm, r32`
    pub fn tilemovcol_reg(&mut self, dst: Reg, src: Reg, column: Reg) -> Result<()> {
        if !dst.is_tmm() || !src.is_zmm() || !column.is_reg_bit(32) {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &dst,
            Some(&column),
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F38 | TypeFlags::T_EW1 | TypeFlags::T_MUST_EVEX,
            0x4B,
            None,
        )
    }

    /// `tilemovcol tmm, zmm, imm8`
    pub fn tilemovcol_imm(&mut self, dst: Reg, src: Reg, column: u8) -> Result<()> {
        if !dst.is_tmm() || !src.is_zmm() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66 | TypeFlags::T_0F3A | TypeFlags::T_EW1 | TypeFlags::T_MUST_EVEX,
            0x2F,
            Some(column),
        )
    }

    /// `top2bf16ps tmm, zmm, zmm`
    pub fn top2bf16ps(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F3, 0x5C, None)
    }

    /// `top4bssd tmm, zmm, zmm`
    pub fn top4bssd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F2, 0x5E, None)
    }

    /// `top4bsud tmm, zmm, zmm`
    pub fn top4bsud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F3, 0x5E, None)
    }

    /// `top4busd tmm, zmm, zmm`
    pub fn top4busd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_66, 0x5E, None)
    }

    /// `top4buud tmm, zmm, zmm`
    pub fn top4buud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::NONE, 0x5E, None)
    }

    /// `top4mxbf8ps tmm, zmm, zmm, imm8`
    pub fn top4mxbf8ps(&mut self, dst: Reg, src1: Reg, src2: Reg, imm: u8) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::NONE, 0x8D, Some(imm))
    }

    /// `top4mxbhf8ps tmm, zmm, zmm, imm8`
    pub fn top4mxbhf8ps(&mut self, dst: Reg, src1: Reg, src2: Reg, imm: u8) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F2, 0x8D, Some(imm))
    }

    /// `top4mxbssps tmm, zmm, zmm, imm8`
    pub fn top4mxbssps(&mut self, dst: Reg, src1: Reg, src2: Reg, imm: u8) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F2, 0x8F, Some(imm))
    }

    /// `top4mxhbf8ps tmm, zmm, zmm, imm8`
    pub fn top4mxhbf8ps(&mut self, dst: Reg, src1: Reg, src2: Reg, imm: u8) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_F3, 0x8D, Some(imm))
    }

    /// `top4mxhf8ps tmm, zmm, zmm, imm8`
    pub fn top4mxhf8ps(&mut self, dst: Reg, src1: Reg, src2: Reg, imm: u8) -> Result<()> {
        self.amx_zmm_op(dst, src1, src2, TypeFlags::T_66, 0x8D, Some(imm))
    }

    fn amx_zmm_op(
        &mut self,
        dst: Reg,
        src1: Reg,
        src2: Reg,
        prefix: TypeFlags,
        opcode: u8,
        imm: Option<u8>,
    ) -> Result<()> {
        if !dst.is_tmm() || !src1.is_zmm() || !src2.is_zmm() {
            return Err(Error::BadCombination);
        }
        let map = if imm.is_some() {
            TypeFlags::T_0F3A
        } else {
            TypeFlags::T_0F38
        };
        self.buf.op_vex(
            &dst,
            Some(&src2),
            &RegMem::Reg(src1),
            prefix | map | TypeFlags::T_W0 | TypeFlags::T_MUST_EVEX,
            opcode,
            imm,
        )
    }

    /// `tilerelease` — VEX.128.NP.0F38.W0 49 C0
    #[inline]
    pub fn tilerelease(&mut self) -> Result<()> {
        self.buf.db(0xC4)?;
        self.buf.db(0xE2)?; // R=1 X=1 B=1 map=0F38
        self.buf.db(0x78)?; // W=0 vvvv=1111 L=0 pp=00
        self.buf.db(0x49)?;
        self.buf.db(0xC0)
    }

    /// `tilezero tmm` — VEX.128.F2.0F38.W0 49 /r
    #[inline]
    pub fn tilezero(&mut self, dst: Reg) -> Result<()> {
        let tmm0 = Reg::tmm(0);
        self.buf.op_vex(
            &dst,
            Some(&tmm0),
            &RegMem::Reg(tmm0),
            TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x49,
            None,
        )
    }

    /// Helper for AMX 3-operand tile dot-product instructions
    /// `tdp* dst, src1, src2` — VEX.128.pp.0F38.W0 opcode /r
    /// xbyak operand order: reg=dst, vvvv=src2, rm=src1
    fn amx_tdp(
        &mut self,
        dst: Reg,
        src1: Reg,
        src2: Reg,
        type_: TypeFlags,
        code: u8,
    ) -> Result<()> {
        self.buf.op_vex(
            &dst,
            Some(&src2),
            &RegMem::Reg(src1),
            type_ | TypeFlags::T_0F38 | TypeFlags::T_W0,
            code,
            None,
        )
    }

    /// `tdpbssd tmm, tmm, tmm` — VEX.128.F2.0F38.W0 5E /r
    #[inline]
    pub fn tdpbssd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::T_F2, 0x5E)
    }
    /// `tdpbsud tmm, tmm, tmm` — VEX.128.F3.0F38.W0 5E /r
    #[inline]
    pub fn tdpbsud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::T_F3, 0x5E)
    }
    /// `tdpbusd tmm, tmm, tmm` — VEX.128.66.0F38.W0 5E /r
    #[inline]
    pub fn tdpbusd(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::T_66, 0x5E)
    }
    /// `tdpbuud tmm, tmm, tmm` — VEX.128.NP.0F38.W0 5E /r
    #[inline]
    pub fn tdpbuud(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::NONE, 0x5E)
    }
    /// `tdpbf16ps tmm, tmm, tmm` — VEX.128.F3.0F38.W0 5C /r
    #[inline]
    pub fn tdpbf16ps(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::T_F3, 0x5C)
    }
    /// `tdpfp16ps tmm, tmm, tmm` — VEX.128.F2.0F38.W0 5C /r
    #[inline]
    pub fn tdpfp16ps(&mut self, dst: Reg, src1: Reg, src2: Reg) -> Result<()> {
        self.amx_tdp(dst, src1, src2, TypeFlags::T_F2, 0x5C)
    }

    /// `tileloadd tmm, [base + index*stride]` — VEX.128.F2.0F38.W0 4B /r
    /// Uses SIB-like addressing with base and index*stride.
    #[inline]
    pub fn tileloadd(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_F2 | TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x4B,
            None,
        )
    }
    /// `tileloaddt1 tmm, [base + index*stride]` — VEX.128.66.0F38.W0 4B /r
    #[inline]
    pub fn tileloaddt1(&mut self, dst: Reg, addr: Address) -> Result<()> {
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x4B,
            None,
        )
    }
    /// `tilestored [base + index*stride], tmm` — VEX.128.F3.0F38.W0 4B /r
    #[inline]
    pub fn tilestored(&mut self, addr: Address, src: Reg) -> Result<()> {
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_F3 | TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x4B,
            None,
        )
    }

    /// `ldtilecfg [m512]` — VEX.128.NP.0F38.W0 49 /0
    #[inline]
    pub fn ldtilecfg(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_vex(
            &r,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x49,
            None,
        )
    }
    /// `sttilecfg [m512]` — VEX.128.66.0F38.W0 49 /0
    #[inline]
    pub fn sttilecfg(&mut self, addr: Address) -> Result<()> {
        let r = Reg::new(0, crate::operand::Kind::Reg, 32);
        self.buf.op_vex(
            &r,
            None,
            &RegMem::Mem(addr),
            TypeFlags::T_66 | TypeFlags::T_0F38 | TypeFlags::T_W0,
            0x49,
            None,
        )
    }

    // ── ACE 1.15 vector conversions ───────────────────────────

    /// `vcvtbf42hf8 xmm/ymm/zmm, xmm/ymm/m`
    pub fn vcvtbf42hf8(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt1(
            dst,
            src.into(),
            TypeFlags::T_N8
                | TypeFlags::T_N_VL
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x37,
        )
    }

    /// `vcvtbf62hf8 xmm/ymm/zmm, xmm/ymm/zmm`
    pub fn vcvtbf62hf8(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_simd() || !src.is_simd() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66
                | TypeFlags::T_MAP5
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x37,
            None,
        )
    }

    /// `vcvtbf82bf4s xmm/ymm/m, xmm/ymm/zmm`
    pub fn vcvtbf82bf4s(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        self.op_vmov(
            dst.into(),
            src,
            TypeFlags::T_N8
                | TypeFlags::T_N_VL
                | TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x3D,
            true,
        )
    }

    /// `vcvtbf82bf6s xmm/ymm/zmm, xmm/ymm/zmm`
    pub fn vcvtbf82bf6s(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_simd() || !src.is_simd() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Reg(dst),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x3E,
            None,
        )
    }

    /// `vcvtbf82ps xmm/ymm/zmm, xmm/m`
    pub fn vcvtbf82ps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_vmov(
            src.into(),
            dst,
            TypeFlags::T_N4
                | TypeFlags::T_N_VL
                | TypeFlags::T_MAP5
                | TypeFlags::T_EW1
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x36,
            false,
        )
    }

    /// `vcvtbiasps2bf8 xmm, xmm/ymm/zmm, xmm/ymm/zmm/m`
    pub fn vcvtbiasps2bf8(&mut self, dst: Reg, bias: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt7(
            dst,
            bias,
            src.into(),
            TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x39,
        )
    }

    /// `vcvtbiasps2bf8s xmm, xmm/ymm/zmm, xmm/ymm/zmm/m`
    pub fn vcvtbiasps2bf8s(&mut self, dst: Reg, bias: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt7(
            dst,
            bias,
            src.into(),
            TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x3B,
        )
    }

    /// `vcvtbiasps2hf8 xmm, xmm/ymm/zmm, xmm/ymm/zmm/m`
    pub fn vcvtbiasps2hf8(&mut self, dst: Reg, bias: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt7(
            dst,
            bias,
            src.into(),
            TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x38,
        )
    }

    /// `vcvtbiasps2hf8s xmm, xmm/ymm/zmm, xmm/ymm/zmm/m`
    pub fn vcvtbiasps2hf8s(&mut self, dst: Reg, bias: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt7(
            dst,
            bias,
            src.into(),
            TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x3A,
        )
    }

    /// `vcvthf62hf8 xmm/ymm/zmm, xmm/ymm/zmm`
    pub fn vcvthf62hf8(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_simd() || !src.is_simd() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &dst,
            None,
            &RegMem::Reg(src),
            TypeFlags::T_66
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x37,
            None,
        )
    }

    /// `vcvthf82bf4s xmm/ymm/m, xmm/ymm/zmm`
    pub fn vcvthf82bf4s(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        self.op_vmov(
            dst.into(),
            src,
            TypeFlags::T_N8
                | TypeFlags::T_N_VL
                | TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x3D,
            true,
        )
    }

    /// `vcvthf82hf6s xmm/ymm/zmm, xmm/ymm/zmm`
    pub fn vcvthf82hf6s(&mut self, dst: Reg, src: Reg) -> Result<()> {
        if !dst.is_simd() || !src.is_simd() {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(
            &src,
            None,
            &RegMem::Reg(dst),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x3C,
            None,
        )
    }

    /// `vcvthf82ps xmm/ymm/zmm, xmm/m`
    pub fn vcvthf82ps(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_vmov(
            src.into(),
            dst,
            TypeFlags::T_N4
                | TypeFlags::T_N_VL
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX,
            0x36,
            false,
        )
    }

    pub fn vcvtps2bf8(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x39,
        )
    }

    pub fn vcvtps2bf8s(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x3B,
        )
    }

    pub fn vcvtps2hf8(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x38,
        )
    }

    pub fn vcvtps2hf8s(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_F3
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x3A,
        )
    }

    pub fn vcvtrops2hf8(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_66
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x38,
        )
    }

    pub fn vcvtrops2hf8s(&mut self, dst: Reg, src: impl Into<RegMem>) -> Result<()> {
        self.op_cvt5(
            dst,
            src.into(),
            TypeFlags::T_66
                | TypeFlags::T_MAP5
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_B32,
            0x3A,
        )
    }

    /// `vpmovssdb xmm/m, xmm/ymm/zmm`
    pub fn vpmovssdb(&mut self, dst: impl Into<RegMem>, src: Reg) -> Result<()> {
        self.op_vmov(
            dst.into(),
            src,
            TypeFlags::T_N4
                | TypeFlags::T_N_VL
                | TypeFlags::T_F3
                | TypeFlags::T_0F38
                | TypeFlags::T_W0
                | TypeFlags::T_YMM
                | TypeFlags::T_MUST_EVEX
                | TypeFlags::T_M_K,
            0x41,
            false,
        )
    }

    /// `vunpackb xmm/ymm/zmm, xmm/ymm/zmm/m, imm8`
    pub fn vunpackb(&mut self, dst: Reg, src: impl Into<RegMem>, imm: u8) -> Result<()> {
        let zero = if dst.is_zmm() {
            Reg::zmm(0)
        } else if dst.is_ymm() {
            Reg::ymm(0)
        } else {
            Reg::xmm(0)
        };
        self.op_avx_x_x_xm(
            dst,
            zero,
            src,
            TypeFlags::T_0F3A | TypeFlags::T_W0 | TypeFlags::T_YMM | TypeFlags::T_MUST_EVEX,
            0x3D,
            Some(imm),
        )
    }

    // Xbyak opCvt3: scalar signed/unsigned integer to scalar float.
    fn op_cvt3(
        &mut self,
        dst: Reg,
        merge: Reg,
        src: RegMem,
        type_: TypeFlags,
        type64: TypeFlags,
        type32: TypeFlags,
        opcode: u8,
    ) -> Result<()> {
        let valid_src = match src {
            RegMem::Reg(reg) => reg.is_reg() && matches!(reg.get_bit(), 32 | 64),
            RegMem::Mem(_) => true,
        };
        if !dst.is_xmm() || !merge.is_xmm() || !valid_src {
            return Err(Error::BadSizeOfRegister);
        }
        let width_type = if src.get_bit() == 64 { type64 } else { type32 };
        self.buf
            .op_vex(&dst, Some(&merge), &src, type_ | width_type, opcode, None)
    }

    // Xbyak opCvt1: (x, x/m), (y, x/m256), (z, y/m).
    fn op_cvt1(&mut self, dst: Reg, src: RegMem, type_: TypeFlags, opcode: u8) -> Result<()> {
        let valid = match src {
            RegMem::Mem(_) => true,
            RegMem::Reg(src) => {
                ((dst.is_xmm() || dst.is_ymm()) && src.is_xmm()) || (dst.is_zmm() && src.is_ymm())
            }
        };
        if !valid {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(&dst, None, &src, type_, opcode, None)
    }

    // Xbyak opCvt5: (x, x/y/z/xword/yword/zword).
    fn op_cvt5(&mut self, dst: Reg, src: RegMem, type_: TypeFlags, opcode: u8) -> Result<()> {
        let src_bit = src.get_bit();
        if !dst.is_xmm() || !matches!(src_bit, 128 | 256 | 512) {
            return Err(Error::BadCombination);
        }
        let kind = match src_bit {
            128 => crate::operand::Kind::Xmm,
            256 => crate::operand::Kind::Ymm,
            _ => crate::operand::Kind::Zmm,
        };
        let encoded_dst = dst.copy_and_set_kind(kind);
        let xmm0 = Reg::xmm(0);
        self.buf
            .op_vex(&encoded_dst, Some(&xmm0), &src, type_, opcode, None)
    }

    // Xbyak opCvt7: destination remains XMM regardless of source VL.
    fn op_cvt7(
        &mut self,
        dst: Reg,
        bias: Reg,
        src: RegMem,
        type_: TypeFlags,
        opcode: u8,
    ) -> Result<()> {
        if !dst.is_xmm() || !bias.is_simd() || (!src.is_mem() && src.get_bit() != bias.get_bit()) {
            return Err(Error::BadCombination);
        }
        self.buf
            .op_vex(&dst, Some(&bias), &src, type_, opcode, None)
    }

    // Xbyak opVmov for vector-width narrowing conversions.
    fn op_vmov(
        &mut self,
        operand: RegMem,
        src: Reg,
        type_: TypeFlags,
        opcode: u8,
        mode: bool,
    ) -> Result<()> {
        if !src.is_simd() {
            return Err(Error::BadCombination);
        }
        let valid = match operand {
            RegMem::Mem(_) => true,
            RegMem::Reg(dst) if mode => {
                (dst.is_xmm() && (src.is_xmm() || src.is_ymm())) || (dst.is_ymm() && src.is_zmm())
            }
            RegMem::Reg(dst) => dst.is_xmm(),
        };
        if !valid {
            return Err(Error::BadCombination);
        }
        self.buf.op_vex(&src, None, &operand, type_, opcode, None)
    }
}
