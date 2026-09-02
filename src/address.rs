use core::ops::{Add, Mul, Sub};

use crate::error::{Error, Result};
use crate::label::LabelId;
use crate::operand::Reg;

/// Addressing mode for an Address operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressMode {
    None,
    ModRM,
    Disp64,  // 64-bit absolute displacement (moffset)
    Rip,     // [rip + disp]
    RipAddr, // [rip + label]
}

/// A register expression representing `[base + index * scale + disp]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegExp {
    pub(crate) base: Reg,
    pub(crate) index: Reg,
    pub(crate) scale: u8,
    pub(crate) disp: i64,
    pub(crate) label_id: Option<LabelId>,
    pub(crate) rip: bool,
    /// When true with rip=true, disp holds an absolute target address.
    /// The encoder computes the RIP-relative displacement at emit time:
    ///   disp32 = target - (current_emit_pos + 4 + imm_size)
    /// Matches Xbyak's `RegRip::isAddr_` used by `code.rip + void_ptr`.
    pub(crate) is_addr: bool,
}

impl RegExp {
    /// Create a RegExp from a single displacement.
    pub fn from_disp(disp: i64) -> Self {
        Self {
            disp,
            ..Default::default()
        }
    }

    /// Create a RegExp from a single register (as base).
    pub fn from_reg(r: Reg) -> Result<Self> {
        Self::from_reg_scale(r, 1)
    }

    /// Create a RegExp from a register with scale.
    pub fn from_reg_scale(r: Reg, scale: u8) -> Result<Self> {
        let is_gpr = r.is_reg() && (r.bit_width() == 32 || r.bit_width() == 64);
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
        if r.bit_width() >= 128 || scale != 1 {
            // SIMD registers are always index, also scaled registers
            exp.index = r;
            exp.scale = scale;
        } else {
            exp.base = r;
            exp.scale = 1;
        }
        Ok(exp)
    }

    /// Create a RIP-relative RegExp with raw displacement.
    pub fn rip() -> Self {
        Self {
            rip: true,
            ..Default::default()
        }
    }

    /// Create a RIP-relative RegExp with an absolute target address.
    /// The encoder computes `disp32 = addr - (emit_pos + 4 + imm_size)`
    /// at emit time, matching Xbyak's `code.rip + void_ptr` (isAddr_=true).
    pub fn rip_addr(addr: i64) -> Self {
        Self {
            rip: true,
            // Xbyak 7.35 treats a null pointer as the integer displacement
            // zero instead of an absolute target address.
            is_addr: addr != 0,
            disp: addr,
            ..Default::default()
        }
    }

    /// Whether this expression uses VSIB addressing.
    pub fn is_vsib(&self) -> bool {
        let b = self.index.bit_width();
        b == 128 || b == 256 || b == 512
    }

    /// Whether this expression is only a displacement (no base/index).
    pub fn is_only_disp(&self) -> bool {
        self.base.bit_width() == 0 && self.index.bit_width() == 0
    }

    /// Optimize: `[reg*2]` → `[reg + reg]`
    pub fn optimize(&self) -> Self {
        let mut exp = *self;
        let is_gpr32e =
            exp.index.is_reg() && (exp.index.bit_width() == 32 || exp.index.bit_width() == 64);
        if is_gpr32e && exp.base.bit_width() == 0 && exp.scale == 2 {
            exp.base = exp.index;
            exp.scale = 1;
        }
        exp
    }

    /// Validate the expression.
    pub fn verify(&self) -> Result<()> {
        if self.base.bit_width() >= 128 {
            return Err(Error::BadSizeOfRegister);
        }
        if self.index.bit_width() > 0 && self.index.bit_width() <= 64 {
            // ESP can't be index
            if self.index.index() == 4 && self.index.is_reg() {
                return Err(Error::EspCantBeIndex);
            }
            if self.base.bit_width() > 0 && self.base.bit_width() != self.index.bit_width() {
                return Err(Error::BadSizeOfRegister);
            }
        }
        Ok(())
    }

    /// Combine two RegExp expressions (a + b).
    pub fn add(a: &RegExp, b: &RegExp) -> Result<RegExp> {
        if a.index.bit_width() > 0 && b.index.bit_width() > 0 {
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
        if ret.index.bit_width() == 0 {
            ret.index = b.index;
            ret.scale = b.scale;
        }
        if b.base.bit_width() > 0 {
            if ret.base.bit_width() > 0 {
                if ret.index.bit_width() > 0 {
                    return Err(Error::BadAddressing);
                }
                // base + base → base + index*1
                ret.index = b.base;
                // [reg + esp] → [esp + reg]
                if ret.index.index() == 4 && ret.index.is_reg() {
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

    pub fn base(&self) -> &Reg {
        &self.base
    }
    pub fn index(&self) -> &Reg {
        &self.index
    }
    pub fn scale(&self) -> u8 {
        self.scale
    }
    pub fn displacement(&self) -> i64 {
        self.disp
    }
    pub fn is_rip(&self) -> bool {
        self.rip
    }
    pub fn label_id(&self) -> Option<LabelId> {
        self.label_id
    }
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
    /// EVEX opmask index carried by a memory destination (`mem{k}`).
    pub(crate) mask: u8,
    /// Zeroing modifier preserved for Xbyak-compatible validation.
    pub(crate) zero: bool,
    /// Whether optimization is enabled.
    pub(crate) optimize: bool,
    /// Optional label reference.
    pub(crate) label_id: Option<LabelId>,
}

impl Address {
    /// Create a new Address from a RegExp with size hint and broadcast flag.
    pub fn new(bit: u16, broadcast: bool, exp: RegExp) -> Result<Self> {
        let mode = if exp.rip {
            if exp.label_id.is_some() || exp.is_addr {
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
            mask: 0,
            zero: false,
            optimize: true,
        })
    }

    /// Get the (potentially optimized) RegExp.
    pub fn register_expression(&self) -> RegExp {
        if self.optimize {
            self.exp.optimize()
        } else {
            self.exp
        }
    }

    /// Clone without optimization.
    pub fn clone_no_optimize(&self) -> Self {
        let mut addr = *self;
        addr.optimize = false;
        addr
    }

    pub fn mode(&self) -> AddressMode {
        self.mode
    }
    pub fn bit_width(&self) -> u16 {
        self.bit
    }
    pub fn is_broadcast(&self) -> bool {
        self.broadcast
    }
    pub fn opmask_index(&self) -> u8 {
        self.mask
    }
    pub fn has_zero(&self) -> bool {
        self.zero
    }
    pub fn is_vsib(&self) -> bool {
        self.exp.is_vsib()
    }
    pub fn is_only_disp(&self) -> bool {
        self.exp.is_only_disp()
    }
    pub fn displacement(&self) -> i64 {
        self.exp.disp
    }
    pub fn is_64bit_disp(&self) -> bool {
        self.mode == AddressMode::Disp64
    }
    pub fn label_id(&self) -> Option<LabelId> {
        self.label_id
    }

    pub fn is_32bit(&self) -> bool {
        self.exp.base.bit_width() == 32 || self.exp.index.bit_width() == 32
    }

    pub fn has_rex2(&self) -> bool {
        self.exp.base.has_rex2() || self.exp.index.has_rex2()
    }

    /// Set the EVEX writemask on a memory operand, matching Xbyak's
    /// `address | kN` operand modifier.
    pub fn k(mut self, mask_idx: u8) -> Self {
        assert!(mask_idx <= 7);
        self.mask = mask_idx;
        self
    }

    /// Preserve an explicitly requested zeroing modifier. Encoders reject it
    /// where Xbyak rejects zeroing on a memory destination.
    pub fn z(mut self) -> Self {
        self.zero = true;
        self
    }

    /// Return a copy with a different memory-size hint.
    ///
    /// Mirrors Xbyak 7.37 `Address::changeBit`.
    pub fn change_bit(&self, bit: u16) -> Self {
        let mut addr = *self;
        addr.bit = bit;
        addr
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
        if r.bit_width() == 0 {
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
        let exp = addr.register_expression();
        assert_eq!(exp.base().index(), 0);
        assert_eq!(exp.base().bit_width(), 64);
        assert_eq!(exp.displacement(), 0);
    }

    #[test]
    fn test_reg_plus_disp() {
        let exp = RAX + 0x10;
        let addr = dword_ptr(exp);
        assert_eq!(addr.bit_width(), 32);
        assert_eq!(addr.register_expression().displacement(), 0x10);
    }

    #[test]
    fn test_reg_plus_reg() {
        let exp = RAX + RCX;
        assert_eq!(exp.base().index(), 0); // rax
        assert_eq!(exp.index().index(), 1); // rcx
        assert_eq!(exp.scale(), 1);
    }

    #[test]
    fn test_base_plus_index_scaled() {
        let exp = RBX + RSI * 4;
        assert_eq!(exp.base().index(), 3); // rbx
        assert_eq!(exp.index().index(), 6); // rsi
        assert_eq!(exp.scale(), 4);
    }

    #[test]
    fn test_full_sib() {
        let exp = RBP + RDI * 8 + 0x100;
        assert_eq!(exp.base().index(), 5); // rbp
        assert_eq!(exp.index().index(), 7); // rdi
        assert_eq!(exp.scale(), 8);
        assert_eq!(exp.displacement(), 0x100);
    }

    #[test]
    fn test_optimize_scale2() {
        // [rcx*2] → [rcx + rcx*1]
        let exp = RCX * 2;
        let opt = exp.optimize();
        assert_eq!(opt.base().index(), 1);
        assert_eq!(opt.index().index(), 1);
        assert_eq!(opt.scale(), 1);
    }

    #[test]
    fn test_null_rip_address_is_plain_displacement() {
        let addr = ptr(RegExp::rip_addr(0));
        assert_eq!(addr.mode(), AddressMode::Rip);
        assert_eq!(addr.displacement(), 0);
    }

    #[test]
    fn test_address_change_bit_preserves_expression() {
        let byte = byte_ptr(RAX + 16);
        let dword = byte.change_bit(32);
        assert_eq!(byte.bit_width(), 8);
        assert_eq!(dword.bit_width(), 32);
        assert_eq!(dword.register_expression().base(), &RAX);
        assert_eq!(dword.displacement(), 16);
    }
}
