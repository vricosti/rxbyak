/// Encoding type flags for instruction encoding.
///
/// These flags control how instructions are encoded (prefix selection,
/// VEX/EVEX parameters, broadcast, masking, etc.). Stored as a u64
/// bit field, matching xbyak's T_* constants exactly.
///
/// We don't use bitflags! because many values are multi-bit fields
/// or combined from other values, and we need direct arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TypeFlags(pub u64);

impl TypeFlags {
    pub const NONE: Self = Self(0);

    // Low 3 bits: disp8*N encoding
    pub const T_N1: Self = Self(1);
    pub const T_N2: Self = Self(2);
    pub const T_N4: Self = Self(3);
    pub const T_N8: Self = Self(4);
    pub const T_N16: Self = Self(5);
    pub const T_N32: Self = Self(6);
    pub const T_NX_MASK: Self = Self(7);
    /// N = (8, 32, 64)
    pub const T_DUP: Self = Self(7); // == T_NX_MASK

    /// N * (1, 2, 4) for VL
    pub const T_N_VL: Self = Self(1 << 3);
    /// APX instruction
    pub const T_APX: Self = Self(1 << 4);
    /// 0x66 prefix (pp = 1)
    pub const T_66: Self = Self(1 << 5);
    /// 0xF3 prefix (pp = 2)
    pub const T_F3: Self = Self(1 << 6);
    /// reg{er}
    pub const T_ER_R: Self = Self(1 << 7);
    /// 0F map
    pub const T_0F: Self = Self(1 << 8);
    /// 0F 38 map
    pub const T_0F38: Self = Self(1 << 9);
    /// 0F 3A map
    pub const T_0F3A: Self = Self(1 << 10);
    /// MAP5
    pub const T_MAP5: Self = Self(1 << 11);
    /// 256-bit vector length (L=1)
    pub const T_L1: Self = Self(1 << 12);
    /// W=0 (VEX/EVEX)
    pub const T_W0: Self = Self(1 << 13);
    /// W=1 (VEX)
    pub const T_W1: Self = Self(1 << 14);
    /// W=1 (EVEX)
    pub const T_EW1: Self = Self(1 << 16);
    /// Support YMM/ZMM
    pub const T_YMM: Self = Self(1 << 17);
    /// EVEX encoding
    pub const T_EVEX: Self = Self(1 << 18);
    /// xmm{er}
    pub const T_ER_X: Self = Self(1 << 19);
    /// ymm{er}
    pub const T_ER_Y: Self = Self(1 << 20);
    /// zmm{er}
    pub const T_ER_Z: Self = Self(1 << 21);
    /// xmm{sae}
    pub const T_SAE_X: Self = Self(1 << 22);
    /// ymm{sae}
    pub const T_SAE_Y: Self = Self(1 << 23);
    /// zmm{sae}
    pub const T_SAE_Z: Self = Self(1 << 24);
    /// Must use EVEX (contains T_EVEX)
    pub const T_MUST_EVEX: Self = Self(1 << 25);
    /// m32bcst
    pub const T_B32: Self = Self(1 << 26);
    /// m64bcst
    pub const T_B64: Self = Self(1 << 27);
    /// m16bcst (T_B32 | T_B64)
    pub const T_B16: Self = Self((1 << 26) | (1 << 27));
    /// mem{k}
    pub const T_M_K: Self = Self(1 << 28);
    /// VSIB addressing
    pub const T_VSIB: Self = Self(1 << 29);
    /// Use EVEX if memory operand
    pub const T_MEM_EVEX: Self = Self(1 << 30);
    /// MAP6
    pub const T_MAP6: Self = Self(1 << 31);
    /// NF (no-flags, APX)
    pub const T_NF: Self = Self(1 << 32);
    /// code|=1 if !r.isBit(8)
    pub const T_CODE1_IF1: Self = Self(1 << 33);
    /// ND=1
    pub const T_ND1: Self = Self(1 << 35);
    /// ND=ZU
    pub const T_ZU: Self = Self(1 << 36);
    /// 0xF2 prefix (pp = 3)
    pub const T_F2: Self = Self(1 << 37);
    /// Sentinel: attributes >= T_SENTRY are for error checks, not encoding
    pub const T_SENTRY: Self = Self((1 << 38) - 1);
    /// Allow different register sizes
    pub const T_ALLOW_DIFF_SIZE: Self = Self(1 << 38);
    /// Allow [abcd]h registers
    pub const T_ALLOW_ABCDH: Self = Self(1 << 39);

    // Also alias T_EW0 = T_W0
    pub const T_EW0: Self = Self::T_W0;

    /// Test if flag(s) are set.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Test if any of the flag(s) are set.
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Get PP value from type flags: T_66→1, T_F3→2, T_F2→3, else→0
    #[inline]
    pub const fn get_pp(self) -> u8 {
        if self.0 & Self::T_66.0 != 0 {
            1
        } else if self.0 & Self::T_F3.0 != 0 {
            2
        } else if self.0 & Self::T_F2.0 != 0 {
            3
        } else {
            0
        }
    }

    /// Get MAP value from type flags.
    #[inline]
    pub const fn get_map(self) -> u8 {
        if self.0 & Self::T_MAP6.0 != 0 {
            6
        } else if self.0 & Self::T_MAP5.0 != 0 {
            5
        } else if self.0 & Self::T_0F.0 != 0 {
            1
        } else if self.0 & Self::T_0F38.0 != 0 {
            2
        } else if self.0 & Self::T_0F3A.0 != 0 {
            3
        } else {
            0
        }
    }

    /// Get the N value for disp8*N (low 3 bits).
    #[inline]
    pub const fn get_n(self) -> u8 {
        (self.0 & Self::T_NX_MASK.0) as u8
    }

    /// Combine two flag sets.
    #[inline]
    pub const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Mask flag set.
    #[inline]
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl core::ops::BitOr for TypeFlags {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

impl core::ops::BitAnd for TypeFlags {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
}

impl core::ops::BitOrAssign for TypeFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
}

impl core::ops::Not for TypeFlags {
    type Output = Self;
    #[inline]
    fn not(self) -> Self { Self(!self.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pp() {
        assert_eq!(TypeFlags::T_66.get_pp(), 1);
        assert_eq!(TypeFlags::T_F3.get_pp(), 2);
        assert_eq!(TypeFlags::T_F2.get_pp(), 3);
        assert_eq!(TypeFlags::NONE.get_pp(), 0);
    }

    #[test]
    fn test_map() {
        assert_eq!(TypeFlags::T_0F.get_map(), 1);
        assert_eq!(TypeFlags::T_0F38.get_map(), 2);
        assert_eq!(TypeFlags::T_0F3A.get_map(), 3);
        assert_eq!(TypeFlags::T_MAP5.get_map(), 5);
        assert_eq!(TypeFlags::T_MAP6.get_map(), 6);
    }

    #[test]
    fn test_combine() {
        let t = TypeFlags::T_66 | TypeFlags::T_0F | TypeFlags::T_EVEX;
        assert!(t.contains(TypeFlags::T_66));
        assert!(t.contains(TypeFlags::T_0F));
        assert!(t.contains(TypeFlags::T_EVEX));
        assert!(!t.contains(TypeFlags::T_F3));
    }
}
