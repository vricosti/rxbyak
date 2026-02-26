use core::ops::{Add, Mul, Sub};

use crate::error::{Error, Result};
use crate::label::LabelId;
use crate::operand::Reg;

/// Addressing mode for an Address operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    None,
    ModRM,
    Disp64,   // 64-bit absolute displacement (moffset)
    Rip,      // [rip + disp]
    RipAddr,  // [rip + label]
}

/// A register expression representing `[base + index * scale + disp]`.
#[derive(Clone, Copy, Debug)]
pub struct RegExp {
    pub(crate) base: Reg,
    pub(crate) index: Reg,
    pub(crate) scale: u8,
    pub(crate) disp: i64,
    pub(crate) label_id: Option<LabelId>,
    pub(crate) rip: bool,
}

impl Default for RegExp {
    fn default() -> Self {
        Self {
            base: Reg::default(),
            index: Reg::default(),
            scale: 0,
            disp: 0,
            label_id: None,
            rip: false,
        }
    }
}

impl RegExp {
    /// Create a RegExp from a single displacement.
    pub fn from_disp(disp: i64) -> Self {
        Self { disp, ..Default::default() }
    }

    /// Create a RegExp from a single register (as base).
    pub fn from_reg(r: Reg) -> Result<Self> {
        Self::from_reg_scale(r, 1)
    }

    /// Create a RegExp from a register with scale.
    pub fn from_reg_scale(r: Reg, scale: u8) -> Result<Self> {
        let is_gpr = r.is_reg() && (r.get_bit() == 32 || r.get_bit() == 64);
        let is_simd_idx = r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_tmm();
        if !is_gpr && !is_simd_idx {
            return Err(Error::BadSizeOfRegister);
        }
        if scale == 0 {
            return Ok(Self::default());
        }
        if scale != 1 && scale != 2 && scale != 4 && scale != 8 {
            return Err(Error::BadScale);
        }

        let mut exp = Self::default();
        if r.get_bit() >= 128 || scale != 1 {
            // SIMD registers are always index, also scaled registers
            exp.index = r;
            exp.scale = scale;
        } else {
            exp.base = r;
            exp.scale = 1;
        }
        Ok(exp)
    }

    /// Create a RIP-relative RegExp.
    pub fn rip() -> Self {
        Self { rip: true, ..Default::default() }
    }

    /// Whether this expression uses VSIB addressing.
    pub fn is_vsib(&self) -> bool {
        let b = self.index.get_bit();
        b == 128 || b == 256 || b == 512
    }

    /// Whether this expression is only a displacement (no base/index).
    pub fn is_only_disp(&self) -> bool {
        self.base.get_bit() == 0 && self.index.get_bit() == 0
    }

    /// Optimize: `[reg*2]` → `[reg + reg]`
    pub fn optimize(&self) -> Self {
        let mut exp = *self;
        let is_gpr32e = exp.index.is_reg()
            && (exp.index.get_bit() == 32 || exp.index.get_bit() == 64);
        if is_gpr32e && exp.base.get_bit() == 0 && exp.scale == 2 {
            exp.base = exp.index;
            exp.scale = 1;
        }
        exp
    }

    /// Validate the expression.
    pub fn verify(&self) -> Result<()> {
        if self.base.get_bit() >= 128 {
            return Err(Error::BadSizeOfRegister);
        }
        if self.index.get_bit() > 0 && self.index.get_bit() <= 64 {
            // ESP can't be index
            if self.index.get_idx() == 4 && self.index.is_reg() {
                return Err(Error::EspCantBeIndex);
            }
            if self.base.get_bit() > 0 && self.base.get_bit() != self.index.get_bit() {
                return Err(Error::BadSizeOfRegister);
            }
        }
        Ok(())
    }

    /// Combine two RegExp expressions (a + b).
    pub fn add(a: &RegExp, b: &RegExp) -> Result<RegExp> {
        if a.index.get_bit() > 0 && b.index.get_bit() > 0 {
            return Err(Error::BadAddressing);
        }
        if a.label_id.is_some() && b.label_id.is_some() {
            return Err(Error::BadAddressing);
        }
        if b.rip {
            return Err(Error::BadAddressing);
        }
        if a.rip && !b.is_only_disp() {
            return Err(Error::BadAddressing);
        }

        let mut ret = *a;
        if ret.label_id.is_none() {
            ret.label_id = b.label_id;
        }
        if ret.index.get_bit() == 0 {
            ret.index = b.index;
            ret.scale = b.scale;
        }
        if b.base.get_bit() > 0 {
            if ret.base.get_bit() > 0 {
                if ret.index.get_bit() > 0 {
                    return Err(Error::BadAddressing);
                }
                // base + base → base + index*1
                ret.index = b.base;
                // [reg + esp] → [esp + reg]
                if ret.index.get_idx() == 4 && ret.index.is_reg() {
                    core::mem::swap(&mut ret.base, &mut ret.index);
                }
                ret.scale = 1;
            } else {
                ret.base = b.base;
            }
        }
        ret.disp = ret.disp.wrapping_add(b.disp);
        Ok(ret)
    }

    pub fn get_base(&self) -> &Reg { &self.base }
    pub fn get_index(&self) -> &Reg { &self.index }
    pub fn get_scale(&self) -> u8 { self.scale }
    pub fn get_disp(&self) -> i64 { self.disp }
    pub fn is_rip(&self) -> bool { self.rip }
    pub fn get_label_id(&self) -> Option<LabelId> { self.label_id }
}

impl PartialEq for RegExp {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
            && self.index == other.index
            && self.disp == other.disp
            && self.scale == other.scale
    }
}

// Reg + Reg → RegExp
impl Add<Reg> for Reg {
    type Output = RegExp;
    fn add(self, rhs: Reg) -> RegExp {
        let a = RegExp::from_reg(self).expect("bad register for address");
        let b = RegExp::from_reg(rhs).expect("bad register for address");
        RegExp::add(&a, &b).expect("bad addressing")
    }
}

// Reg * scale → RegExp
impl Mul<u8> for Reg {
    type Output = RegExp;
    fn mul(self, scale: u8) -> RegExp {
        RegExp::from_reg_scale(self, scale).expect("bad scale")
    }
}

// RegExp + Reg → RegExp
impl Add<Reg> for RegExp {
    type Output = RegExp;
    fn add(self, rhs: Reg) -> RegExp {
        let b = RegExp::from_reg(rhs).expect("bad register for address");
        RegExp::add(&self, &b).expect("bad addressing")
    }
}

// Reg + RegExp → RegExp
impl Add<RegExp> for Reg {
    type Output = RegExp;
    fn add(self, rhs: RegExp) -> RegExp {
        let a = RegExp::from_reg(self).expect("bad register for address");
        RegExp::add(&a, &rhs).expect("bad addressing")
    }
}

// RegExp + RegExp → RegExp
impl Add for RegExp {
    type Output = RegExp;
    fn add(self, rhs: RegExp) -> RegExp {
        RegExp::add(&self, &rhs).expect("bad addressing")
    }
}

// RegExp + i32 → RegExp
impl Add<i32> for RegExp {
    type Output = RegExp;
    fn add(self, disp: i32) -> RegExp {
        let b = RegExp::from_disp(disp as i64);
        RegExp::add(&self, &b).expect("bad addressing")
    }
}

// Reg + i32 → RegExp
impl Add<i32> for Reg {
    type Output = RegExp;
    fn add(self, disp: i32) -> RegExp {
        let a = RegExp::from_reg(self).expect("bad register for address");
        a + disp
    }
}

// RegExp - i32 → RegExp
impl Sub<i32> for RegExp {
    type Output = RegExp;
    fn sub(self, disp: i32) -> RegExp {
        let mut ret = self;
        ret.disp = ret.disp.wrapping_sub(disp as i64);
        ret
    }
}

// Reg - i32 → RegExp
impl Sub<i32> for Reg {
    type Output = RegExp;
    fn sub(self, disp: i32) -> RegExp {
        let a = RegExp::from_reg(self).expect("bad register for address");
        a - disp
    }
}

/// A memory address operand `[base + index * scale + disp]` with size hint.
#[derive(Clone, Copy, Debug)]
pub struct Address {
    /// The address expression.
    pub(crate) exp: RegExp,
    /// Size hint in bits (0, 8, 16, 32, 64, 128, 256, 512).
    pub(crate) bit: u16,
    /// Addressing mode.
    pub(crate) mode: AddressMode,
    /// Immediate size for the mnemonic (0, 1, 2, 4).
    pub(crate) imm_size: u8,
    /// disp8*N scaling (0=normal, 1=force disp32, 2/4/8 for EVEX scaling).
    pub(crate) disp8n: u8,
    /// Whether VSIB is permitted.
    pub(crate) permit_vsib: bool,
    /// Whether broadcast is enabled.
    pub(crate) broadcast: bool,
    /// Whether optimization is enabled.
    pub(crate) optimize: bool,
    /// Optional label reference.
    pub(crate) label_id: Option<LabelId>,
}

impl Address {
    /// Create a new Address from a RegExp with size hint and broadcast flag.
    pub fn new(bit: u16, broadcast: bool, exp: RegExp) -> Result<Self> {
        let mode = if exp.rip {
            if exp.label_id.is_some() {
                AddressMode::RipAddr
            } else {
                AddressMode::Rip
            }
        } else if exp.is_only_disp() {
            let disp = exp.disp as u64;
            if (0x80000000..=0xFFFFFFFF80000000).contains(&disp) || exp.label_id.is_some() {
                AddressMode::Disp64
            } else {
                AddressMode::ModRM
            }
        } else {
            AddressMode::ModRM
        };

        exp.verify()?;

        Ok(Self {
            label_id: exp.label_id,
            exp,
            bit,
            mode,
            imm_size: 0,
            disp8n: 0,
            permit_vsib: false,
            broadcast,
            optimize: true,
        })
    }

    /// Get the (potentially optimized) RegExp.
    pub fn get_reg_exp(&self) -> RegExp {
        if self.optimize { self.exp.optimize() } else { self.exp }
    }

    /// Clone without optimization.
    pub fn clone_no_optimize(&self) -> Self {
        let mut addr = *self;
        addr.optimize = false;
        addr
    }

    pub fn get_mode(&self) -> AddressMode { self.mode }
    pub fn get_bit(&self) -> u16 { self.bit }
    pub fn is_broadcast(&self) -> bool { self.broadcast }
    pub fn is_vsib(&self) -> bool { self.exp.is_vsib() }
    pub fn is_only_disp(&self) -> bool { self.exp.is_only_disp() }
    pub fn get_disp(&self) -> i64 { self.exp.disp }
    pub fn is_64bit_disp(&self) -> bool { self.mode == AddressMode::Disp64 }
    pub fn get_label_id(&self) -> Option<LabelId> { self.label_id }

    pub fn is_32bit(&self) -> bool {
        self.exp.base.get_bit() == 32 || self.exp.index.get_bit() == 32
    }

    pub fn has_rex2(&self) -> bool {
        self.exp.base.has_rex2() || self.exp.index.has_rex2()
    }

    /// Set immediate size for this address context.
    pub fn with_imm_size(mut self, imm_size: u8) -> Self {
        self.imm_size = imm_size;
        self
    }
}

// Address frame functions — equivalent to xbyak's AddressFrame

/// Create an unsized memory reference.
pub fn ptr(exp: RegExp) -> Address {
    Address::new(0, false, exp).expect("bad address")
}

/// Create an 8-bit (byte) memory reference.
pub fn byte_ptr(exp: RegExp) -> Address {
    Address::new(8, false, exp).expect("bad address")
}

/// Create a 16-bit (word) memory reference.
pub fn word_ptr(exp: RegExp) -> Address {
    Address::new(16, false, exp).expect("bad address")
}

/// Create a 32-bit (dword) memory reference.
pub fn dword_ptr(exp: RegExp) -> Address {
    Address::new(32, false, exp).expect("bad address")
}

/// Create a 64-bit (qword) memory reference.
pub fn qword_ptr(exp: RegExp) -> Address {
    Address::new(64, false, exp).expect("bad address")
}

/// Create a 128-bit (xmmword) memory reference.
pub fn xmmword_ptr(exp: RegExp) -> Address {
    Address::new(128, false, exp).expect("bad address")
}

/// Create a 256-bit (ymmword) memory reference.
pub fn ymmword_ptr(exp: RegExp) -> Address {
    Address::new(256, false, exp).expect("bad address")
}

/// Create a 512-bit (zmmword) memory reference.
pub fn zmmword_ptr(exp: RegExp) -> Address {
    Address::new(512, false, exp).expect("bad address")
}

/// Create a broadcast memory reference with the given element size.
pub fn broadcast_ptr(bit: u16, exp: RegExp) -> Address {
    Address::new(bit, true, exp).expect("bad address")
}

/// Helpers to create RegExp from a single register (for use in ptr functions).
impl From<Reg> for RegExp {
    fn from(r: Reg) -> Self {
        if r.get_bit() == 0 {
            return Self::default();
        }
        RegExp::from_reg(r).expect("bad register for address")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reg::*;

    #[test]
    fn test_simple_reg_addr() {
        let addr = ptr(RAX.into());
        let exp = addr.get_reg_exp();
        assert_eq!(exp.get_base().get_idx(), 0);
        assert_eq!(exp.get_base().get_bit(), 64);
        assert_eq!(exp.get_disp(), 0);
    }

    #[test]
    fn test_reg_plus_disp() {
        let exp = RAX + 0x10;
        let addr = dword_ptr(exp);
        assert_eq!(addr.get_bit(), 32);
        assert_eq!(addr.get_reg_exp().get_disp(), 0x10);
    }

    #[test]
    fn test_reg_plus_reg() {
        let exp = RAX + RCX;
        assert_eq!(exp.get_base().get_idx(), 0); // rax
        assert_eq!(exp.get_index().get_idx(), 1); // rcx
        assert_eq!(exp.get_scale(), 1);
    }

    #[test]
    fn test_base_plus_index_scaled() {
        let exp = RBX + RSI * 4;
        assert_eq!(exp.get_base().get_idx(), 3); // rbx
        assert_eq!(exp.get_index().get_idx(), 6); // rsi
        assert_eq!(exp.get_scale(), 4);
    }

    #[test]
    fn test_full_sib() {
        let exp = RBP + RDI * 8 + 0x100;
        assert_eq!(exp.get_base().get_idx(), 5); // rbp
        assert_eq!(exp.get_index().get_idx(), 7); // rdi
        assert_eq!(exp.get_scale(), 8);
        assert_eq!(exp.get_disp(), 0x100);
    }

    #[test]
    fn test_optimize_scale2() {
        // [rcx*2] → [rcx + rcx*1]
        let exp = RCX * 2;
        let opt = exp.optimize();
        assert_eq!(opt.get_base().get_idx(), 1);
        assert_eq!(opt.get_index().get_idx(), 1);
        assert_eq!(opt.get_scale(), 1);
    }
}
