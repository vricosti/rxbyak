use crate::address::{Address, AddressMode};
use crate::code_array::CodeBuffer;
use crate::encoding_flags::TypeFlags;
use crate::error::{Error, Result};
use crate::operand::{Reg, RegMem};

/// Check if a value fits in a signed 8-bit displacement.
#[inline]
pub(crate) fn is_in_disp8(v: u32) -> bool {
    let s = v as i32;
    (-128..=127).contains(&s)
}

/// Check if a value fits in a signed 32-bit integer.
#[inline]
pub(crate) fn is_in_int32(v: u64) -> bool {
    let s = v as i64;
    (i32::MIN as i64..=i32::MAX as i64).contains(&s)
}

/// Compute REX R, X, B bits.
///
/// `bit` = which index bit to test (3 for REX, 4 for REX2).
/// `bit3` = whether the W bit should be set.
#[inline]
pub(crate) fn rex_rxb(bit: u32, bit3: bool, r: &Reg, b: &Reg, x: &Reg) -> u8 {
    let mut v: u8 = if bit3 { 8 } else { 0 };
    if r.has_idx_bit(bit) { v |= 4; }
    if x.has_idx_bit(bit) { v |= 2; }
    if b.has_idx_bit(bit) { v |= 1; }
    v
}

/// Encoding helper methods on CodeBuffer.
impl CodeBuffer {
    /// Emit ModR/M byte: `(mod << 6) | ((r1 & 7) << 3) | (r2 & 7)`.
    pub(crate) fn set_modrm(&mut self, mod_: u8, r1: u8, r2: u8) -> Result<()> {
        self.db((mod_ << 6) | ((r1 & 7) << 3) | (r2 & 7))
    }

    /// Emit REX2 prefix (0xD5 + payload).
    pub(crate) fn emit_rex2(&mut self, bit3: bool, rex4bit: u8, r: &Reg, b: &Reg, x: &Reg) -> Result<()> {
        self.db(0xD5)?;
        self.db((rex_rxb(4, bit3, r, b, x) << 4) | rex4bit)
    }

    /// Emit REX prefix. Returns true if REX2 was emitted instead.
    ///
    /// Handles legacy prefixes (0x66, 0xF2, 0xF3) and the REX/REX2 prefix.
    pub(crate) fn emit_rex_for_reg_reg(
        &mut self,
        r1: &Reg,
        r2: &Reg,
        type_: TypeFlags,
    ) -> Result<bool> {
        if r1.get_nf() || r2.get_nf() { return Err(Error::InvalidNf); }
        if r1.get_zu() || r2.get_zu() { return Err(Error::InvalidZu); }

        // 16-bit prefix
        let p66 = (r1.is_bit(16) && !(r2.is_bit(32) || r2.is_bit(64)))
            || (r2.is_bit(16) && !(r1.is_bit(32) || r1.is_bit(64)));
        if type_.contains(TypeFlags::T_66) || p66 { self.db(0x66)?; }
        if type_.contains(TypeFlags::T_F2) { self.db(0xF2)?; }
        if type_.contains(TypeFlags::T_F3) { self.db(0xF3)?; }

        let is0f = type_.intersects(TypeFlags::T_0F);
        let default_reg = Reg::default();

        // reg, reg encoding: ModRM(r2, r1)
        let rex = rex_rxb(3, r1.is_reg_bit(64) || r2.is_reg_bit(64), r2, r1, &default_reg);
        if r1.has_rex2() || r2.has_rex2() {
            if type_.intersects(TypeFlags::T_0F38 | TypeFlags::T_0F3A) {
                return Err(Error::CantUseRex2);
            }
            self.emit_rex2(is0f, rex, r2, r1, &default_reg)?;
            return Ok(true);
        }
        let final_rex = if rex != 0 || r1.is_ext8bit() || r2.is_ext8bit() {
            rex | 0x40
        } else {
            0
        };
        if final_rex != 0 { self.db(final_rex)?; }
        Ok(false)
    }

    /// Emit REX prefix for reg+address. Returns true if REX2 was emitted.
    pub(crate) fn emit_rex_for_reg_mem(
        &mut self,
        r: &Reg,
        addr: &Address,
        type_: TypeFlags,
    ) -> Result<bool> {
        if r.get_nf() { return Err(Error::InvalidNf); }
        if r.get_zu() { return Err(Error::InvalidZu); }

        let p66 = (r.is_bit(16) && addr.get_bit() > 0 && !(addr.get_bit() == 32 || addr.get_bit() == 64))
            || (addr.get_bit() == 16 && !(r.is_bit(32) || r.is_bit(64)));
        if type_.contains(TypeFlags::T_66) || p66 { self.db(0x66)?; }
        if type_.contains(TypeFlags::T_F2) { self.db(0xF2)?; }
        if type_.contains(TypeFlags::T_F3) { self.db(0xF3)?; }

        let is0f = type_.intersects(TypeFlags::T_0F);
        let exp = addr.get_reg_exp();
        let base = exp.get_base();
        let idx = exp.get_index();

        // 32-bit address in 64-bit mode
        if addr.is_32bit() { self.db(0x67)?; }

        let rex = rex_rxb(3, r.is_reg_bit(64), r, base, idx);
        if r.has_rex2() || addr.has_rex2() {
            if type_.intersects(TypeFlags::T_0F38 | TypeFlags::T_0F3A) {
                return Err(Error::CantUseRex2);
            }
            self.emit_rex2(is0f, rex, r, base, idx)?;
            return Ok(true);
        }
        let final_rex = if rex != 0 || r.is_ext8bit() {
            rex | 0x40
        } else {
            0
        };
        if final_rex != 0 { self.db(final_rex)?; }
        Ok(false)
    }

    /// Emit opcode bytes after REX prefix.
    ///
    /// Handles 0x0F, 0x0F38, 0x0F3A prefix bytes.
    /// Also handles T_CODE1_IF1 (OR code with 1 if register is not 8-bit).
    pub(crate) fn write_code(
        &mut self,
        type_: TypeFlags,
        r: &Reg,
        code: u8,
        rex2: bool,
    ) -> Result<()> {
        if !(type_.contains(TypeFlags::T_APX) || rex2) {
            if type_.contains(TypeFlags::T_0F) {
                self.db(0x0F)?;
            } else if type_.contains(TypeFlags::T_0F38) {
                self.db(0x0F)?;
                self.db(0x38)?;
            } else if type_.contains(TypeFlags::T_0F3A) {
                self.db(0x0F)?;
                self.db(0x3A)?;
            }
        }
        // T_CODE1_IF1: if type has no attribute flags (is raw opcode) or has T_CODE1_IF1,
        // and register is not 8-bit, OR the code with 1
        let sentry_mask = type_.and(TypeFlags::T_SENTRY);
        let code1_if1 = sentry_mask.0 == 0 || type_.contains(TypeFlags::T_CODE1_IF1);
        let final_code = if code1_if1 && !r.is_bit(8) {
            code | 1
        } else {
            code
        };
        self.db(final_code)
    }

    /// Emit SIB addressing (ModRM + optional SIB + displacement).
    pub(crate) fn emit_sib(&mut self, addr: &Address, reg: u8) -> Result<()> {
        let exp = addr.get_reg_exp();
        let disp64 = exp.get_disp();

        // Verify displacement fits
        let high = (disp64 as u64) >> 31;
        if high != 0 && high != 0x1FFFFFFFF {
            return Err(Error::OffsetIsTooBig);
        }

        let disp = disp64 as u32;
        let base = exp.get_base();
        let index = exp.get_index();
        let base_idx = base.get_idx();
        let base_bit = base.get_bit();
        let index_bit = index.get_bit();
        let disp8n = addr.disp8n;

        // Determine mod field
        let mod_ = if base_bit == 0 || ((base_idx & 7) != 5 && disp == 0 && addr.get_label_id().is_none()) {
            0u8 // mod00
        } else if addr.get_label_id().is_some() {
            2u8 // mod10 (always disp32 for labels)
        } else if disp8n == 0 {
            if is_in_disp8(disp) { 1 } else { 2 } // mod01 or mod10
        } else {
            let t = (disp as i32 / disp8n as i32) as u32;
            if (disp as i32 % disp8n as i32) == 0 && is_in_disp8(t) {
                1 // mod01 with scaled disp
            } else {
                2 // mod10
            }
        };

        let new_base_idx = if base_bit > 0 { base_idx & 7 } else { 5 }; // EBP=5 for disp32-only

        // Determine if SIB byte is needed
        let has_sib = index_bit > 0 || (base_idx & 7) == 4 // ESP
            || (base_bit == 0 && index_bit == 0); // 64-bit: disp-only needs SIB

        if has_sib {
            self.set_modrm(mod_, reg, 4)?; // ESP=4 signals SIB follows
            // SIB: [SS:index:base]
            let idx = if index_bit > 0 { index.get_idx() & 7 } else { 4 }; // ESP=4 = no index
            let scale = exp.get_scale();
            let ss = match scale {
                8 => 3,
                4 => 2,
                2 => 1,
                _ => 0,
            };
            self.set_modrm(ss, idx, new_base_idx)?;
        } else {
            self.set_modrm(mod_, reg, new_base_idx)?;
        }

        // Emit displacement
        let actual_disp = if disp8n > 0 && mod_ == 1 {
            (disp as i32 / disp8n as i32) as u32
        } else {
            disp
        };

        if mod_ == 1 {
            self.db(actual_disp as u8)?;
        } else if mod_ == 2 || (mod_ == 0 && base_bit == 0) {
            self.dd(actual_disp)?;
        }

        Ok(())
    }

    /// Emit address operand (ModRM + SIB + displacement), handling
    /// ModRM, RIP-relative, and 64-bit displacement modes.
    pub(crate) fn emit_addr(&mut self, addr: &Address, reg: u8) -> Result<()> {
        if !addr.permit_vsib && addr.is_vsib() {
            return Err(Error::BadVsibAddressing);
        }
        match addr.get_mode() {
            AddressMode::ModRM => {
                self.emit_sib(addr, reg)
            }
            AddressMode::Rip | AddressMode::RipAddr => {
                self.set_modrm(0, reg, 5)?;
                if addr.get_label_id().is_some() {
                    // Label-relative RIP: will be patched later
                    self.dd(0)?; // placeholder
                } else {
                    let mut disp = addr.get_disp();
                    if addr.get_mode() == AddressMode::RipAddr {
                        // Adjust for current position + 4 bytes displacement + immSize
                        let cur = self.cur() as usize;
                        disp -= (cur + 4 + addr.imm_size as usize) as i64;
                    }
                    self.dd(disp as u32)?;
                }
                Ok(())
            }
            _ => Ok(())
        }
    }

    /// Emit VEX prefix (2 or 3-byte form).
    pub(crate) fn emit_vex(
        &mut self,
        reg: &Reg,
        base: &Reg,
        v: Option<&Reg>,
        type_: TypeFlags,
        code: u8,
        x: bool,
    ) -> Result<()> {
        let w = if type_.contains(TypeFlags::T_W1) { 1u8 } else { 0 };
        let is256 = type_.contains(TypeFlags::T_L1) || reg.is_ymm() || base.is_ymm();
        let r = reg.is_ext_idx();
        let b = base.is_ext_idx();
        let idx = v.map_or(0, |v| v.get_idx());

        // VEX doesn't support registers >= 16
        if (idx | reg.get_idx() | base.get_idx()) >= 16 {
            return Err(Error::BadCombination);
        }

        let pp = type_.get_pp();
        let vvvv = (((!idx) & 15) << 3) | (if is256 { 4 } else { 0 }) | pp;

        if !b && !x && w == 0 && type_.intersects(TypeFlags::T_0F) {
            // 2-byte VEX
            self.db(0xC5)?;
            self.db((if r { 0 } else { 0x80 }) | vvvv)?;
        } else {
            // 3-byte VEX
            let mmmm = type_.get_map();
            self.db(0xC4)?;
            self.db((if r { 0 } else { 0x80 }) | (if x { 0 } else { 0x40 }) | (if b { 0 } else { 0x20 }) | mmmm)?;
            self.db((w << 7) | vvvv)?;
        }
        self.db(code)
    }

    /// Emit EVEX prefix (4-byte).
    ///
    /// Returns the disp8*N scaling factor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_evex(
        &mut self,
        reg: &Reg,
        base: &Reg,
        v: Option<&Reg>,
        type_: TypeFlags,
        code: u8,
        x_reg: Option<&Reg>,
        b_flag: bool,
        aaa: u8,
        vl_hint: u32,
        hi16_vidx: bool,
    ) -> Result<u8> {
        if !type_.intersects(TypeFlags::T_EVEX | TypeFlags::T_MUST_EVEX) {
            return Err(Error::EvexIsInvalid);
        }
        let w = if type_.contains(TypeFlags::T_EW1) { 1u8 } else { 0 };
        let mmm = type_.get_map();
        let pp = type_.get_pp();
        let idx = v.map_or(0, |v| v.get_idx());
        let vvvv = !idx;

        let r_flag = reg.is_ext_idx();
        let x3 = x_reg.map_or(false, |x| x.is_ext_idx())
            || (base.is_simd() && base.is_ext_idx2());
        let b4 = if base.is_reg() && base.is_ext_idx2() { 8u8 } else { 0 };
        let u = if x_reg.map_or(false, |x| x.is_reg() && x.is_ext_idx2()) { 0u8 } else { 4 };
        let b_ext = base.is_ext_idx();
        let rp = reg.is_ext_idx2();

        // Rounding
        let r_round = reg.get_rounding() as u8;
        let b_round = base.get_rounding() as u8;
        let v_round = v.map_or(0, |v| v.get_rounding() as u8);
        // Verify at most one rounding source
        let rounding = {
            let vals = [r_round, b_round, v_round];
            let non_zero: Vec<u8> = vals.iter().copied().filter(|&x| x > 0).collect();
            if non_zero.len() > 1 {
                // Check they're all the same
                if !non_zero.iter().all(|&x| x == non_zero[0]) {
                    return Err(Error::RoundingIsAlreadySet);
                }
            }
            non_zero.first().copied().unwrap_or(0)
        };

        let mut disp8n = 1u8;
        let ll;
        let mut b_bit = b_flag;
        let mut type_mut = type_;

        if rounding > 0 {
            if rounding == 5 {
                // T_SAE
                ll = 0;
            } else {
                ll = rounding - 1;
            }
            b_bit = true;
        } else {
            let mut vl = vl_hint;
            if let Some(v_reg) = v {
                vl = vl.max(v_reg.get_bit() as u32);
            }
            vl = vl.max(reg.get_bit() as u32).max(base.get_bit() as u32);
            ll = if vl >= 512 { 2 } else if vl == 256 { 1 } else { 0 };

            if b_bit {
                let b16 = TypeFlags::T_B16;
                let b32 = TypeFlags::T_B32;
                disp8n = if (type_.0 & b16.0) == b16.0 { 2 }
                    else if type_.contains(b32) { 4 }
                    else { 8 };
            } else if type_.get_n() == TypeFlags::T_DUP.0 as u8 {
                disp8n = match vl { 128 => 8, 256 => 32, _ => 64 };
            } else {
                if (type_.0 & (TypeFlags::T_NX_MASK.0 | TypeFlags::T_N_VL.0)) == 0 {
                    type_mut = type_mut | TypeFlags::T_N16 | TypeFlags::T_N_VL;
                }
                let low = (type_mut.0 & TypeFlags::T_NX_MASK.0) as u8;
                if low > 0 {
                    disp8n = 1 << (low - 1);
                    if type_mut.contains(TypeFlags::T_N_VL) {
                        disp8n *= match vl { 512 => 4, 256 => 2, _ => 1 };
                    }
                }
            }
        }

        let v4 = v.map_or(false, |v| v.is_ext_idx2()) || hi16_vidx;
        let z_flag = reg.has_zero() || base.has_zero() || v.map_or(false, |v| v.has_zero());

        // Opmask
        let final_aaa = if aaa > 0 {
            aaa
        } else {
            let vals = [base.get_opmask_idx(), reg.get_opmask_idx(),
                       v.map_or(0, |v| v.get_opmask_idx())];
            let non_zero: Vec<u8> = vals.iter().copied().filter(|&x| x > 0).collect();
            if non_zero.len() > 1 && !non_zero.iter().all(|&x| x == non_zero[0]) {
                return Err(Error::OpmaskIsAlreadySet);
            }
            non_zero.first().copied().unwrap_or(0)
        };
        let z_bit = if final_aaa == 0 { false } else { z_flag };

        // Emit EVEX 4-byte prefix
        self.db(0x62)?;
        self.db(
            (if r_flag { 0 } else { 0x80 })
            | (if x3 { 0 } else { 0x40 })
            | (if b_ext { 0 } else { 0x20 })
            | (if rp { 0 } else { 0x10 })
            | b4
            | mmm
        )?;
        self.db(
            (if w == 1 { 0x80 } else { 0 })
            | (((vvvv & 15) as u8) << 3)
            | u
            | (pp & 3)
        )?;
        self.db(
            (if z_bit { 0x80 } else { 0 })
            | ((ll & 3) << 5)
            | (if b_bit { 0x10 } else { 0 })
            | (if v4 { 0 } else { 8 })
            | (final_aaa & 7)
        )?;
        self.db(code)?;

        Ok(disp8n)
    }

    /// Emit EVEX prefix for legacy encoding (APX).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_evex_leg(
        &mut self,
        r: &Reg,
        b: &Reg,
        x: &Reg,
        v: &Reg,
        type_: TypeFlags,
        sc: Option<u8>,
    ) -> Result<()> {
        let mut m = type_.get_map();
        if m == 0 { m = 4; } // legacy

        let r3 = if !r.is_ext_idx() { 0x80u8 } else { 0 };
        let x3 = if !x.is_ext_idx() { 0x40u8 } else { 0 };
        let b3 = if !b.is_ext_idx() { 0x20u8 } else { 0 };
        let r4 = if !r.is_ext_idx2() { 0x10u8 } else { 0 };
        let b4 = if b.is_ext_idx2() { 0x08u8 } else { 0 };

        let w = if type_.contains(TypeFlags::T_W0) { 0u8 }
            else if r.is_bit(64) || v.is_bit(64) || type_.contains(TypeFlags::T_W1) { 1 }
            else { 0 };

        let v_bits = (((!v.get_idx()) & 15) as u8) << 3;
        let x4 = if !x.is_ext_idx2() { 0x04u8 } else { 0 };

        let pp = if type_.intersects(TypeFlags::T_F2 | TypeFlags::T_F3 | TypeFlags::T_66) {
            type_.get_pp()
        } else if r.is_bit(16) || v.is_bit(16) {
            1
        } else {
            0
        };

        let v4_bit = if !v.is_ext_idx2() { 1u8 } else { 0 };
        let nd = if type_.contains(TypeFlags::T_ZU) {
            if r.get_zu() || b.get_zu() { 1u8 } else { 0 }
        } else if type_.contains(TypeFlags::T_ND1) {
            1
        } else if type_.contains(TypeFlags::T_APX) {
            0
        } else if v.is_reg() {
            1
        } else {
            0
        };

        let nf = (r.get_nf() || b.get_nf() || x.get_nf() || v.get_nf()) as u8;
        if !type_.contains(TypeFlags::T_NF) && nf != 0 { return Err(Error::InvalidNf); }
        if !type_.contains(TypeFlags::T_ZU) && r.get_zu() { return Err(Error::InvalidZu); }

        let l = 0u8;

        self.db(0x62)?;
        self.db(r3 | x3 | b3 | r4 | b4 | m)?;
        self.db((w << 7) | v_bits | x4 | pp)?;

        if let Some(sc_val) = sc {
            self.db((l << 5) | (nd << 4) | sc_val)?;
        } else {
            self.db((l << 5) | (nd << 4) | (v4_bit << 3) | (nf << 2))?;
        }

        Ok(())
    }

    /// Encode reg-reg instruction: REX + opcode + ModR/M(mod=3).
    pub(crate) fn op_rr(
        &mut self,
        r1: &Reg,
        r2: &Reg,
        type_: TypeFlags,
        code: u8,
    ) -> Result<()> {
        if !type_.contains(TypeFlags::T_ALLOW_DIFF_SIZE)
            && r1.is_reg() && r2.is_reg()
            && r1.get_bit() != r2.get_bit()
        {
            return Err(Error::BadSizeOfRegister);
        }
        let rex2 = self.emit_rex_for_reg_reg(r2, r1, type_)?;
        self.write_code(type_, r1, code, rex2)?;
        self.set_modrm(3, r1.get_idx(), r2.get_idx())
    }

    /// Encode mem-reg instruction: REX + opcode + ModR/M+SIB+disp.
    pub(crate) fn op_mr(
        &mut self,
        addr: &Address,
        r: &Reg,
        type_: TypeFlags,
        code: u8,
    ) -> Result<()> {
        if addr.is_64bit_disp() { return Err(Error::CantUse64BitDisp); }
        let rex2 = self.emit_rex_for_reg_mem(r, addr, type_)?;
        self.write_code(type_, r, code, rex2)?;
        self.emit_addr(addr, r.get_idx())
    }

    /// Encode reg + (reg or mem) instruction with extension field.
    pub(crate) fn op_rext(
        &mut self,
        rm: &RegMem,
        ext: u8,
        type_: TypeFlags,
        code: u8,
        imm_size: u8,
    ) -> Result<()> {
        match rm {
            RegMem::Mem(addr) => {
                let r = Reg::new(ext, crate::operand::Kind::Reg, if addr.get_bit() > 0 { addr.get_bit() } else { 32 });
                let mut addr_with_imm = *addr;
                addr_with_imm.imm_size = imm_size;
                self.op_mr(&addr_with_imm, &r, type_, code)
            }
            RegMem::Reg(reg) => {
                let r = Reg::new(ext, crate::operand::Kind::Reg, reg.get_bit());
                self.op_rr(&r, reg, type_ | TypeFlags::T_ALLOW_ABCDH, code)
            }
        }
    }

    /// Dispatch reg to reg-or-mem operand (opRO in C++).
    pub(crate) fn op_ro(
        &mut self,
        r: &Reg,
        op: &RegMem,
        type_: TypeFlags,
        code: u8,
        cond_r: bool,
        imm_size: u8,
    ) -> Result<()> {
        match op {
            RegMem::Mem(addr) => {
                let mut addr = *addr;
                addr.imm_size = imm_size;
                self.op_mr(&addr, r, type_, code)
            }
            RegMem::Reg(reg) => {
                if cond_r {
                    self.op_rr(r, reg, type_, code)
                } else {
                    Err(Error::BadCombination)
                }
            }
        }
    }

    /// SSE instruction encoding helper.
    pub(crate) fn op_sse(
        &mut self,
        r: &Reg,
        op: &RegMem,
        type_: TypeFlags,
        code: u8,
        imm8: Option<u8>,
    ) -> Result<()> {
        self.op_ro(r, op, type_, code, true, if imm8.is_some() { 1 } else { 0 })?;
        if let Some(imm) = imm8 {
            self.db(imm)?;
        }
        Ok(())
    }

    /// Main VEX/EVEX dispatch for SIMD instructions (opVex in C++).
    ///
    /// `r` is the register field of ModRM.
    /// `p1` is the optional VEX vvvv register (None for 2-operand SSE).
    /// `op2` is the r/m operand (register or memory).
    /// `type_` contains encoding flags.
    /// `code` is the opcode byte.
    /// `imm8` is an optional immediate byte.
    pub(crate) fn op_vex(
        &mut self,
        r: &Reg,
        p1: Option<&Reg>,
        op2: &RegMem,
        type_: TypeFlags,
        code: u8,
        imm8: Option<u8>,
    ) -> Result<()> {
        match op2 {
            RegMem::Mem(addr) => {
                let mut addr = *addr;
                let exp = addr.get_reg_exp();
                let base = *exp.get_base();
                let index = *exp.get_index();

                // 32-bit address override in 64-bit mode
                if addr.is_32bit() { self.db(0x67)?; }

                let need_evex = type_.intersects(TypeFlags::T_MUST_EVEX | TypeFlags::T_MEM_EVEX)
                    || r.has_evex()
                    || p1.map_or(false, |p| p.has_evex())
                    || addr.is_broadcast()
                    || addr.has_rex2();

                if need_evex {
                    let b = addr.is_broadcast();
                    if b && !type_.intersects(TypeFlags::T_B32 | TypeFlags::T_B64) {
                        return Err(Error::InvalidBroadcast);
                    }
                    let vl = if exp.is_vsib() { index.get_bit() as u32 } else { 0 };
                    let hi16_vidx = index.is_simd() && index.is_ext_idx2();
                    let disp8n = self.emit_evex(r, &base, p1, type_, code, Some(&index), b, 0, vl, hi16_vidx)?;
                    if disp8n > 0 { addr.disp8n = disp8n; }
                } else {
                    self.emit_vex(r, &base, p1, type_, code, index.is_ext_idx())?;
                }

                if type_.contains(TypeFlags::T_VSIB) { addr.permit_vsib = true; }
                if imm8.is_some() { addr.imm_size = 1; }
                self.emit_addr(&addr, r.get_idx())?;
            }
            RegMem::Reg(base) => {
                let need_evex = type_.contains(TypeFlags::T_MUST_EVEX)
                    || r.has_evex()
                    || p1.map_or(false, |p| p.has_evex())
                    || base.has_evex();

                if need_evex {
                    self.emit_evex(r, base, p1, type_, code, None, false, 0, 0, false)?;
                } else {
                    self.emit_vex(r, base, p1, type_, code, false)?;
                }
                self.set_modrm(3, r.get_idx(), base.get_idx())?;
            }
        }
        if let Some(imm) = imm8 {
            self.db(imm)?;
        }
        Ok(())
    }
}
