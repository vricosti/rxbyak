use crate::error::{Error, Result};

/// Bit flag for ext8bit registers (spl, bpl, sil, dil).
const EXT8BIT: u8 = 0x20;

/// Kind of operand, matching xbyak's Operand::Kind bit flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Kind {
    None   = 0,
    Mem    = 1 << 0,
    Reg    = 1 << 1,
    Mmx    = 1 << 2,
    Fpu    = 1 << 3,
    Xmm    = 1 << 4,
    Ymm    = 1 << 5,
    Zmm    = 1 << 6,
    Opmask = 1 << 7,
    BndReg = 1 << 8,
    Tmm    = 1 << 9,
}

/// Rounding modes for EVEX encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Rounding {
    None  = 0,
    RnSae = 1,
    RdSae = 2,
    RuSae = 3,
    RzSae = 4,
    Sae   = 5,
}

/// Segment register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Segment {
    Es = 0,
    Cs = 1,
    Ss = 2,
    Ds = 3,
    Fs = 4,
    Gs = 5,
}

impl Segment {
    /// Segment override prefix byte.
    pub fn prefix(self) -> u8 {
        match self {
            Segment::Es => 0x26,
            Segment::Cs => 0x2E,
            Segment::Ss => 0x36,
            Segment::Ds => 0x3E,
            Segment::Fs => 0x64,
            Segment::Gs => 0x65,
        }
    }
}

/// A register operand. Flat struct replacing the C++ class hierarchy.
///
/// Encodes: GPR (8/16/32/64-bit), MMX, XMM, YMM, ZMM, Opmask (k0-k7),
/// FPU (st0-st7), BndReg (bnd0-bnd3), TMM (tmm0-tmm7).
#[derive(Clone, Copy, Debug)]
pub struct Reg {
    /// Register index (0..31). Bit 5 set for ext8bit (spl/bpl/sil/dil).
    idx: u8,
    /// Register kind.
    kind: Kind,
    /// Bit width (8, 16, 32, 64, 128, 256, 512, 8192).
    bit: u16,
    /// EVEX writemask index (k0-k7), 0 = no mask.
    mask: u8,
    /// EVEX rounding mode.
    rounding: Rounding,
    /// EVEX zeroing masking {z}.
    zero: bool,
    /// APX no-flags (NF).
    nf: bool,
    /// APX ND=ZU.
    zu: bool,
}

impl PartialEq for Reg {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx
            && self.kind as u16 == other.kind as u16
            && self.bit == other.bit
    }
}

impl Eq for Reg {}

impl Reg {
    /// Create a new register.
    pub const fn new(idx: u8, kind: Kind, bit: u16) -> Self {
        Self {
            idx,
            kind,
            bit,
            mask: 0,
            rounding: Rounding::None,
            zero: false,
            nf: false,
            zu: false,
        }
    }

    /// Create a new ext8bit register (spl, bpl, sil, dil).
    pub const fn new_ext8(idx: u8) -> Self {
        Self {
            idx: idx | EXT8BIT,
            kind: Kind::Reg,
            bit: 8,
            mask: 0,
            rounding: Rounding::None,
            zero: false,
            nf: false,
            zu: false,
        }
    }

    // Convenience constructors for each register kind
    pub const fn gpr8(idx: u8) -> Self { Self::new(idx, Kind::Reg, 8) }
    pub const fn gpr16(idx: u8) -> Self { Self::new(idx, Kind::Reg, 16) }
    pub const fn gpr32(idx: u8) -> Self { Self::new(idx, Kind::Reg, 32) }
    pub const fn gpr64(idx: u8) -> Self { Self::new(idx, Kind::Reg, 64) }
    pub const fn mmx(idx: u8) -> Self { Self::new(idx, Kind::Mmx, 64) }
    pub const fn xmm(idx: u8) -> Self { Self::new(idx, Kind::Xmm, 128) }
    pub const fn ymm(idx: u8) -> Self { Self::new(idx, Kind::Ymm, 256) }
    pub const fn zmm(idx: u8) -> Self { Self::new(idx, Kind::Zmm, 512) }
    pub const fn opmask(idx: u8) -> Self { Self::new(idx, Kind::Opmask, 64) }
    pub const fn fpu(idx: u8) -> Self { Self::new(idx, Kind::Fpu, 32) }
    pub const fn bndreg(idx: u8) -> Self { Self::new(idx, Kind::BndReg, 128) }
    pub const fn tmm(idx: u8) -> Self { Self::new(idx, Kind::Tmm, 8192) }

    // Accessors

    /// Get the raw index value (includes ext8bit flag).
    pub const fn raw_idx(&self) -> u8 { self.idx }
    /// Get the register index (0..31), stripping ext8bit flag.
    pub const fn get_idx(&self) -> u8 { self.idx & (EXT8BIT - 1) }
    /// Get the register kind.
    pub const fn get_kind(&self) -> Kind { self.kind }
    /// Get the bit width.
    pub const fn get_bit(&self) -> u16 { self.bit }
    /// Get the opmask index (0 = no mask).
    pub const fn get_opmask_idx(&self) -> u8 { self.mask }
    /// Get the rounding mode.
    pub const fn get_rounding(&self) -> Rounding { self.rounding }
    /// Whether zeroing masking is enabled.
    pub const fn has_zero(&self) -> bool { self.zero }
    /// Whether NF (no-flags) is set.
    pub const fn get_nf(&self) -> bool { self.nf }
    /// Whether ZU is set.
    pub const fn get_zu(&self) -> bool { self.zu }

    // Kind checks

    pub fn is_none(&self) -> bool { self.kind as u16 == 0 }
    pub fn is_reg(&self) -> bool { self.kind as u16 & Kind::Reg as u16 != 0 }
    pub fn is_reg_bit(&self, bit: u16) -> bool { self.is_reg() && (bit == 0 || self.bit == bit) }
    pub fn is_mmx(&self) -> bool { self.kind as u16 & Kind::Mmx as u16 != 0 }
    pub fn is_xmm(&self) -> bool { self.kind as u16 & Kind::Xmm as u16 != 0 }
    pub fn is_ymm(&self) -> bool { self.kind as u16 & Kind::Ymm as u16 != 0 }
    pub fn is_zmm(&self) -> bool { self.kind as u16 & Kind::Zmm as u16 != 0 }
    pub fn is_simd(&self) -> bool {
        self.kind as u16 & (Kind::Xmm as u16 | Kind::Ymm as u16 | Kind::Zmm as u16) != 0
    }
    pub fn is_tmm(&self) -> bool { self.kind as u16 & Kind::Tmm as u16 != 0 }
    pub fn is_opmask(&self) -> bool { self.kind as u16 & Kind::Opmask as u16 != 0 }
    pub fn is_bndreg(&self) -> bool { self.kind as u16 & Kind::BndReg as u16 != 0 }
    pub fn is_fpu(&self) -> bool { self.kind as u16 & Kind::Fpu as u16 != 0 }

    /// Check if the bit width matches (0 means any).
    pub fn is_bit(&self, bit: u16) -> bool { bit == 0 || self.bit == bit }

    /// Is this an ext8bit register (spl, bpl, sil, dil)?
    pub fn is_ext8bit(&self) -> bool { (self.idx & EXT8BIT) != 0 }

    /// Is this a high 8-bit register (ah, ch, dh, bh)?
    pub fn is_high8bit(&self) -> bool {
        self.bit == 8 && !self.is_ext8bit() && (4..8).contains(&self.get_idx())
    }

    /// Does the index have bit N set (for REX)?
    pub fn has_idx_bit(&self, bit: u32) -> bool { self.get_idx() & (1 << bit) != 0 }

    /// Does this register need an extended index (idx >= 8)?
    pub fn is_ext_idx(&self) -> bool { (self.get_idx() & 8) != 0 }

    /// Does this register need an extended index 2 (idx >= 16)?
    pub fn is_ext_idx2(&self) -> bool { (self.get_idx() & 16) != 0 }

    /// Does this register require EVEX encoding?
    pub fn has_evex(&self) -> bool {
        self.is_zmm() || self.is_ext_idx2() || self.mask != 0
            || self.rounding as u8 != 0
    }

    /// Does this register require a REX prefix?
    pub fn has_rex(&self) -> bool {
        self.is_ext8bit() || self.is_reg_bit(64) || self.is_ext_idx()
    }

    /// Does this register require a REX2 prefix?
    pub fn has_rex2(&self) -> bool {
        self.is_ext_idx2() && !self.is_simd() && !self.is_opmask()
    }

    /// Does this register require REX2 or NF?
    pub fn has_rex2_nf(&self) -> bool { self.has_rex2() || self.nf }

    /// Does this register require REX2 or NF or ZU?
    pub fn has_rex2_nf_zu(&self) -> bool { self.has_rex2() || self.nf || self.zu }

    // Modifiers

    /// Set opmask index (k1-k7) for EVEX writemask. Returns modified copy.
    pub fn k(mut self, mask_idx: u8) -> Self {
        assert!(mask_idx <= 7);
        self.mask = mask_idx;
        self
    }

    /// Enable zeroing masking {z}. Returns modified copy.
    pub fn z(mut self) -> Self {
        self.zero = true;
        self
    }

    /// Set rounding mode. Returns modified copy.
    pub fn rounding(mut self, r: Rounding) -> Self {
        self.rounding = r;
        self
    }

    /// Set NF (no-flags). Returns modified copy.
    pub fn nf(mut self) -> Self {
        self.nf = true;
        self
    }

    /// Set ZU. Returns modified copy.
    pub fn zu(mut self) -> Self {
        self.zu = true;
        self
    }

    /// Set opmask index (mutable in place).
    pub fn set_opmask_idx(&mut self, idx: u8) -> Result<()> {
        if self.mask != 0 && self.mask != idx {
            return Err(Error::OpmaskIsAlreadySet);
        }
        self.mask = idx;
        Ok(())
    }

    /// Set rounding (mutable in place).
    pub fn set_rounding(&mut self, r: Rounding) -> Result<()> {
        if self.rounding as u8 != 0 && self.rounding as u8 != r as u8 {
            return Err(Error::RoundingIsAlreadySet);
        }
        self.rounding = r;
        Ok(())
    }

    /// Convert this register to a different bit width.
    pub fn change_bit(&self, bit: u16) -> Result<Reg> {
        let mut r = *self;
        let idx = r.get_idx();
        match bit {
            8 => {
                if idx >= 32 { return Err(Error::CantConvert); }
                if (4..8).contains(&idx) {
                    r.idx = idx | EXT8BIT;
                }
                r.kind = Kind::Reg;
            }
            16 | 32 | 64 => {
                if idx >= 32 { return Err(Error::CantConvert); }
                r.kind = Kind::Reg;
            }
            128 => { r.kind = Kind::Xmm; }
            256 => { r.kind = Kind::Ymm; }
            512 => { r.kind = Kind::Zmm; }
            8192 => { r.kind = Kind::Tmm; }
            _ => { return Err(Error::CantConvert); }
        }
        r.bit = bit;
        if bit < 128 {
            r.mask = 0;
            r.rounding = Rounding::None;
        }
        Ok(r)
    }

    /// Convert to 8-bit register.
    pub fn cvt8(&self) -> Result<Reg> { self.change_bit(8) }
    /// Convert to 16-bit register.
    pub fn cvt16(&self) -> Result<Reg> { self.change_bit(16) }
    /// Convert to 32-bit register.
    pub fn cvt32(&self) -> Result<Reg> { self.change_bit(32) }
    /// Convert to 64-bit register.
    pub fn cvt64(&self) -> Result<Reg> { self.change_bit(64) }
    /// Convert to XMM register.
    pub fn cvt128(&self) -> Result<Reg> { self.change_bit(128) }
    /// Convert to YMM register.
    pub fn cvt256(&self) -> Result<Reg> { self.change_bit(256) }
    /// Convert to ZMM register.
    pub fn cvt512(&self) -> Result<Reg> { self.change_bit(512) }

    /// Copy this register and set its kind (for VEX size adjustment).
    pub fn copy_and_set_kind(&self, kind: Kind) -> Self {
        let bit = match kind {
            Kind::Xmm => 128,
            Kind::Ymm => 256,
            Kind::Zmm => 512,
            _ => self.bit,
        };
        Self {
            idx: self.idx,
            kind,
            bit,
            mask: self.mask,
            rounding: self.rounding,
            zero: self.zero,
            nf: self.nf,
            zu: self.zu,
        }
    }
}

impl Default for Reg {
    fn default() -> Self {
        Self::new(0, Kind::None, 0)
    }
}

/// Operand that can be either a register or a memory address.
#[derive(Clone, Copy, Debug)]
pub enum RegMem {
    Reg(Reg),
    Mem(crate::address::Address),
}

impl RegMem {
    pub fn is_reg(&self) -> bool { matches!(self, RegMem::Reg(_)) }
    pub fn is_mem(&self) -> bool { matches!(self, RegMem::Mem(_)) }

    pub fn as_reg(&self) -> Option<&Reg> {
        match self {
            RegMem::Reg(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_mem(&self) -> Option<&crate::address::Address> {
        match self {
            RegMem::Mem(m) => Some(m),
            _ => None,
        }
    }

    pub fn get_bit(&self) -> u16 {
        match self {
            RegMem::Reg(r) => r.get_bit(),
            RegMem::Mem(m) => m.get_bit(),
        }
    }
}

impl From<Reg> for RegMem {
    fn from(r: Reg) -> Self { RegMem::Reg(r) }
}

impl From<crate::address::Address> for RegMem {
    fn from(m: crate::address::Address) -> Self { RegMem::Mem(m) }
}

/// Operand that can be a register, memory address, or immediate value.
#[derive(Clone, Copy, Debug)]
pub enum RegMemImm {
    Reg(Reg),
    Mem(crate::address::Address),
    Imm(i64),
}

impl From<Reg> for RegMemImm {
    fn from(r: Reg) -> Self { RegMemImm::Reg(r) }
}

impl From<crate::address::Address> for RegMemImm {
    fn from(m: crate::address::Address) -> Self { RegMemImm::Mem(m) }
}

impl From<i64> for RegMemImm {
    fn from(v: i64) -> Self { RegMemImm::Imm(v) }
}

impl From<i32> for RegMemImm {
    fn from(v: i32) -> Self { RegMemImm::Imm(v as i64) }
}

impl From<u32> for RegMemImm {
    fn from(v: u32) -> Self { RegMemImm::Imm(v as i64) }
}

impl From<RegMem> for RegMemImm {
    fn from(rm: RegMem) -> Self {
        match rm {
            RegMem::Reg(r) => RegMemImm::Reg(r),
            RegMem::Mem(m) => RegMemImm::Mem(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpr_construction() {
        let eax = Reg::gpr32(0);
        assert_eq!(eax.get_idx(), 0);
        assert_eq!(eax.get_bit(), 32);
        assert!(eax.is_reg());
        assert!(!eax.is_xmm());
    }

    #[test]
    fn test_ext8bit() {
        let spl = Reg::new_ext8(4);
        assert!(spl.is_ext8bit());
        assert_eq!(spl.get_idx(), 4);
        assert_eq!(spl.get_bit(), 8);
    }

    #[test]
    fn test_high8bit() {
        let ah = Reg::gpr8(4);
        assert!(ah.is_high8bit());
        let al = Reg::gpr8(0);
        assert!(!al.is_high8bit());
    }

    #[test]
    fn test_simd_regs() {
        let xmm0 = Reg::xmm(0);
        assert!(xmm0.is_xmm());
        assert!(xmm0.is_simd());
        assert_eq!(xmm0.get_bit(), 128);

        let zmm31 = Reg::zmm(31);
        assert!(zmm31.is_zmm());
        assert!(zmm31.is_ext_idx2());
        assert!(zmm31.has_evex());
    }

    #[test]
    fn test_opmask_modifier() {
        let xmm0 = Reg::xmm(0).k(1).z();
        assert_eq!(xmm0.get_opmask_idx(), 1);
        assert!(xmm0.has_zero());
    }

    #[test]
    fn test_change_bit() {
        let eax = Reg::gpr32(0);
        let rax = eax.cvt64().unwrap();
        assert_eq!(rax.get_bit(), 64);
        assert!(rax.is_reg());

        let xmm = eax.cvt128().unwrap();
        assert!(xmm.is_xmm());
        assert_eq!(xmm.get_bit(), 128);
    }
}
