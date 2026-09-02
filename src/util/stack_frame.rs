//! StackFrame — automatic function prologue/epilogue generator.
//!
//! Port of xbyak's `Xbyak::util::StackFrame` (xbyak_util.h).
//!
//! Generates proper calling-convention-aware prologue and epilogue code:
//! - Preserves callee-saved registers
//! - Maintains 16-byte stack alignment
//! - Maps function parameters to registers
//! - Provides temporary register allocation
//!
//! # Platform differences
//!
//! | | Windows x64 | System V AMD64 (Linux/macOS) |
//! |---|---|---|
//! | Param regs | RCX, RDX, R8, R9 | RDI, RSI, RDX, RCX, R8, R9 |
//! | Scratch regs | RAX, RCX, RDX, R8-R11 | RAX, RCX, RDX, RSI, RDI, R8-R11 |
//! | Callee-saved | RBX, RBP, RDI, RSI, R12-R15 | RBX, RBP, R12-R15 |
//!
//! # Example
//!
//! ```no_run
//! use rxbyak::{CodeAssembler, Result};
//! use rxbyak::util::stack_frame::StackFrame;
//!
//! let mut asm = CodeAssembler::new(4096)?;
//! let sf = StackFrame::new(&mut asm, 2, 1, 0)?;
//! // sf.p[0] and sf.p[1] hold the two parameters
//! // sf.t[0] is a temporary register
//! asm.mov(sf.p[0], sf.t[0])?;
//! sf.close(&mut asm)?;
//! # Ok::<(), rxbyak::Error>(())
//! ```

use crate::address::ptr;
use crate::assembler::CodeAssembler;
use crate::error::{Error, Result};
use crate::operand::Reg;
use crate::reg::*;

pub const USE_RBP: usize = 1 << 5;
pub const USE_RCX: usize = 1 << 6;
pub const USE_RDX: usize = 1 << 7;
pub const USE_RSI: usize = 1 << 8;
pub const USE_RDI: usize = 1 << 9;
pub const USE_R30_R31: usize = 1 << 10;
pub const USE_RBP_AS_FRAME_POINTER: usize = USE_RBP | (1 << 30);
pub const USE_PUSH2: usize = 1 << 28;
pub const USE_PPX: usize = 1 << 29;

const USE_VEC_NUM_SHIFT: usize = 16;
const USE_VEC_SSE: usize = 1 << 22;
const USE_VEC_AVX: usize = 1 << 23;
pub const NO_VZEROUPPER: usize = 1 << 24;

pub const fn use_sse(count: usize) -> usize {
    USE_VEC_SSE | (count << USE_VEC_NUM_SHIFT)
}

pub const fn use_avx(count: usize) -> usize {
    USE_VEC_AVX | (count << USE_VEC_NUM_SHIFT)
}

const USE_MASK: usize =
    USE_RCX | USE_RDX | USE_RSI | USE_RDI | USE_RBP | USE_R30_R31 | USE_PUSH2 | USE_PPX;
const USE_VEC_MASK: usize = (63 << USE_VEC_NUM_SHIFT) | USE_VEC_SSE | USE_VEC_AVX | NO_VZEROUPPER;

const MAX_PARAMS: usize = 4;
const MAX_TEMPS: usize = 14;

/// Total usable GPRs: 16 minus RSP and RAX = 14.
const MAX_REG_NUM: usize = 14;

/// Register allocation order: parameter regs first, then scratch, then callee-saved.
/// Matches xbyak's `getOrderTbl()` (16 - RSP - RAX = 14 entries).
#[cfg(target_os = "windows")]
const REG_ORDER: [Reg; MAX_REG_NUM] = [
    RCX, RDX, R8, R9, // param regs (4)
    R10, R11, // scratch (2) — total scratch (noSaveNum) = 6
    RDI, RSI, RBX, RBP, // callee-saved
    R12, R13, R14, R15,
];

#[cfg(not(target_os = "windows"))]
const REG_ORDER: [Reg; MAX_REG_NUM] = [
    RDI, RSI, RDX, RCX, // param regs (first 4 of 6)
    R8, R9, // param regs 5-6 (still scratch)
    R10, R11, // scratch — total scratch (noSaveNum) = 8
    RBX, RBP, // callee-saved
    R12, R13, R14, R15,
];

/// Number of registers at the start of REG_ORDER that do NOT need saving
/// (i.e., caller-saved / scratch registers).
#[cfg(target_os = "windows")]
const NO_SAVE_NUM: usize = 6;

#[cfg(not(target_os = "windows"))]
const NO_SAVE_NUM: usize = 8;

const MAX_SAVE_REG_NUM: usize = 10;

/// Automatic function prologue/epilogue generator for x86-64 calling conventions.
///
/// Creates a proper stack frame with callee-saved register preservation,
/// stack alignment, and parameter/temporary register mapping.
///
#[derive(Debug)]
pub struct StackFrame {
    /// Parameter registers. `p[0..pNum]` are mapped to the platform's
    /// parameter passing registers, with reserved registers replaced by their
    /// upstream-defined backup registers where available.
    pub p: [Reg; MAX_PARAMS],
    /// Temporary registers. `t[0..tNum]` are allocated from the register
    /// order table after the parameter registers.
    pub t: [Reg; MAX_TEMPS],
    save_num: usize,
    save_regs: [Reg; MAX_SAVE_REG_NUM],
    p_: usize,
    vec_save_num: usize,
    vec_pos: usize,
    vzeroupper: bool,
    use_vmovaps: bool,
    use_regs: usize,
}

impl StackFrame {
    /// Generate a function prologue and return the register mapping.
    ///
    /// # Parameters
    ///
    /// - `asm`: The assembler to emit prologue code into.
    /// - `p_num`: Number of function parameters (0..=4).
    /// - `t_num`: Number of temporary registers needed, optionally OR'd with
    ///   the `USE_*` register, vector, and prologue flags.
    /// - `stack_size`: Extra local stack space in bytes (rounded up to 8-byte boundary).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadPnum`] if `p_num > 4`, or [`Error::BadTnum`] if
    /// the total register demand exceeds 14 available registers.
    pub fn new(
        asm: &mut CodeAssembler,
        p_num: usize,
        t_num: usize,
        stack_size: usize,
    ) -> Result<Self> {
        if p_num > MAX_PARAMS {
            return Err(Error::BadPnum);
        }

        let t_num_actual = t_num & !(USE_MASK | USE_RBP_AS_FRAME_POINTER | USE_VEC_MASK);
        let use_regs = t_num & USE_MASK;
        let vec_kind = t_num & (USE_VEC_SSE | USE_VEC_AVX);
        let vec_num = (t_num >> USE_VEC_NUM_SHIFT) & 63;
        if vec_kind == (USE_VEC_SSE | USE_VEC_AVX) {
            return Err(Error::BadTnum);
        }
        if t_num & NO_VZEROUPPER != 0 && vec_kind != USE_VEC_AVX {
            return Err(Error::BadTnum);
        }
        if vec_kind == 0 {
            if vec_num > 0 {
                return Err(Error::BadTnum);
            }
        } else if vec_num > if vec_kind == USE_VEC_AVX { 32 } else { 16 } {
            return Err(Error::BadTnum);
        }
        let use_vmovaps = vec_kind == USE_VEC_AVX && t_num & NO_VZEROUPPER != 0;
        let vzeroupper = vec_kind == USE_VEC_AVX && !use_vmovaps;
        #[cfg(target_os = "windows")]
        let vec_save_num = vec_num.saturating_sub(6).min(10);
        #[cfg(not(target_os = "windows"))]
        let vec_save_num = 0;

        let caller_use_num = REG_ORDER[..NO_SAVE_NUM]
            .iter()
            .filter(|reg| use_regs & use_flag_of(**reg) != 0)
            .count();
        let callee_use_num = REG_ORDER[NO_SAVE_NUM..]
            .iter()
            .filter(|reg| use_regs & use_flag_of(**reg) != 0)
            .count();
        let use_num = caller_use_num + callee_use_num;
        if p_num + t_num_actual + use_num > MAX_REG_NUM {
            return Err(Error::BadTnum);
        }
        let base_save_num = (p_num + t_num_actual + use_num).saturating_sub(NO_SAVE_NUM);

        let mut save_regs = [Reg::default(); MAX_SAVE_REG_NUM];
        let mut save_num = 0;
        let mut pushed_rbp = false;
        if use_regs & USE_RBP != 0 {
            if use_regs & USE_PPX != 0 {
                asm.pushp(RBP)?;
            } else {
                asm.push(RBP)?;
            }
            save_regs[save_num] = RBP;
            save_num += 1;
            pushed_rbp = true;
            if t_num & USE_RBP_AS_FRAME_POINTER == USE_RBP_AS_FRAME_POINTER {
                asm.mov(RBP, RSP)?;
            }
        }
        if use_regs & USE_R30_R31 != 0 {
            save_regs[save_num] = R30;
            save_num += 1;
            save_regs[save_num] = R31;
            save_num += 1;
        }
        for (index, reg) in REG_ORDER[NO_SAVE_NUM..].iter().copied().enumerate() {
            if index < base_save_num || use_regs & use_flag_of(reg) != 0 {
                if pushed_rbp && reg == RBP {
                    continue;
                }
                save_regs[save_num] = reg;
                save_num += 1;
            }
        }

        let mut index = usize::from(pushed_rbp);
        while index < save_num {
            if use_regs & USE_PUSH2 != 0 && index & 1 != 0 && index + 1 < save_num {
                if use_regs & USE_PPX != 0 {
                    asm.push2p(save_regs[index], save_regs[index + 1])?;
                } else {
                    asm.push2(save_regs[index], save_regs[index + 1])?;
                }
                index += 2;
            } else {
                if use_regs & USE_PPX != 0 {
                    asm.pushp(save_regs[index])?;
                } else {
                    asm.push(save_regs[index])?;
                }
                index += 1;
            }
        }

        let (p_, vec_pos) = if vec_save_num > 0 {
            let vec_pos = (stack_size + 15) & !15;
            let mut size = vec_pos + vec_save_num * 16;
            if save_num & 1 == 0 {
                size += 8;
            }
            asm.sub(RSP, size as i64)?;
            for i in 0..vec_save_num {
                let addr = ptr(RSP + (vec_pos + i * 16) as i32);
                if use_vmovaps {
                    asm.vmovaps(addr, Reg::xmm((6 + i) as u8))?;
                } else {
                    asm.movaps(addr, Reg::xmm((6 + i) as u8))?;
                }
            }
            (size, vec_pos)
        } else {
            let mut slots = stack_size.div_ceil(8);
            if slots > 0 && (slots & 1) == (save_num & 1) {
                slots += 1;
            }
            let size = slots * 8;
            if size > 0 {
                asm.sub(RSP, size as i64)?;
            }
            (size, 0)
        };

        let mut pos = 0usize;
        let mut p = [Reg::gpr64(0); MAX_PARAMS];
        for slot in p.iter_mut().take(p_num) {
            *slot = next_register(&mut pos, use_regs);
        }
        let mut t = [Reg::gpr64(0); MAX_TEMPS];
        for slot in t.iter_mut().take(t_num_actual) {
            *slot = next_register(&mut pos, use_regs);
        }
        for slot in reg_slot_table() {
            if use_regs & use_flag_of(slot.target) != 0 && slot.pos < p_num {
                if let Some(alt) = slot.alt {
                    asm.mov(alt, slot.target)?;
                }
            }
        }

        Ok(StackFrame {
            p,
            t,
            save_num,
            save_regs,
            p_,
            vec_save_num,
            vec_pos,
            vzeroupper,
            use_vmovaps,
            use_regs,
        })
    }

    /// Generate the function epilogue.
    ///
    /// Emits: restore local stack → pop callee-saved registers (reverse order)
    /// → `ret`.
    pub fn close(self, asm: &mut CodeAssembler) -> Result<()> {
        self.close_with_ret(asm, true)
    }

    pub fn close_with_ret(self, asm: &mut CodeAssembler, call_ret: bool) -> Result<()> {
        if self.vzeroupper {
            asm.vzeroupper()?;
        }
        for i in 0..self.vec_save_num {
            let addr = ptr(RSP + (self.vec_pos + i * 16) as i32);
            if self.use_vmovaps {
                asm.vmovaps(Reg::xmm((6 + i) as u8), addr)?;
            } else {
                asm.movaps(Reg::xmm((6 + i) as u8), addr)?;
            }
        }
        if self.p_ > 0 {
            asm.add(RSP, self.p_ as i64)?;
        }

        let start = usize::from(self.use_regs & USE_RBP != 0);
        let mut index = self.save_num;
        while index > 0 {
            let i = index - 1;
            if self.use_regs & USE_PUSH2 != 0 && i & 1 == 0 && i > start {
                if self.use_regs & USE_PPX != 0 {
                    asm.pop2p(self.save_regs[i], self.save_regs[i - 1])?;
                } else {
                    asm.pop2(self.save_regs[i], self.save_regs[i - 1])?;
                }
                index -= 2;
            } else {
                if self.use_regs & USE_PPX != 0 {
                    asm.popp(self.save_regs[i])?;
                } else {
                    asm.pop(self.save_regs[i])?;
                }
                index -= 1;
            }
        }
        if call_ret {
            asm.ret()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RegSlot {
    target: Reg,
    pos: usize,
    alt: Option<Reg>,
}

#[cfg(target_os = "windows")]
const REG_SLOTS: [RegSlot; MAX_PARAMS] = [
    RegSlot {
        target: RCX,
        pos: 0,
        alt: Some(R10),
    },
    RegSlot {
        target: RDX,
        pos: 1,
        alt: Some(R11),
    },
    RegSlot {
        target: RDI,
        pos: 6,
        alt: None,
    },
    RegSlot {
        target: RSI,
        pos: 7,
        alt: None,
    },
];

#[cfg(not(target_os = "windows"))]
const REG_SLOTS: [RegSlot; MAX_PARAMS] = [
    RegSlot {
        target: RCX,
        pos: 3,
        alt: Some(R10),
    },
    RegSlot {
        target: RDX,
        pos: 2,
        alt: Some(R11),
    },
    RegSlot {
        target: RDI,
        pos: 0,
        alt: Some(R8),
    },
    RegSlot {
        target: RSI,
        pos: 1,
        alt: Some(R9),
    },
];

fn reg_slot_table() -> &'static [RegSlot; MAX_PARAMS] {
    &REG_SLOTS
}

fn use_flag_of(reg: Reg) -> usize {
    match reg.index() {
        1 => USE_RCX,
        2 => USE_RDX,
        6 => USE_RSI,
        7 => USE_RDI,
        5 => USE_RBP,
        _ => 0,
    }
}

fn next_register(pos: &mut usize, use_regs: usize) -> Reg {
    loop {
        let r = REG_ORDER[*pos];
        *pos += 1;
        let mut retry = false;
        for slot in reg_slot_table() {
            if use_regs & use_flag_of(slot.target) == 0 {
                continue;
            }
            if slot.alt == Some(r) {
                retry = true;
                break;
            }
            if r == slot.target {
                if let Some(alt) = slot.alt {
                    return alt;
                }
                retry = true;
                break;
            }
        }
        if retry || use_regs & use_flag_of(r) != 0 {
            continue;
        }
        return r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg_order_length() {
        assert_eq!(REG_ORDER.len(), MAX_REG_NUM);
    }

    #[test]
    fn test_no_save_num_within_bounds() {
        const { assert!(NO_SAVE_NUM <= MAX_REG_NUM) };
    }

    #[test]
    fn test_reg_slots_within_bounds() {
        for slot in reg_slot_table() {
            assert!(slot.pos < MAX_REG_NUM);
        }
    }

    #[test]
    fn test_bad_pnum() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        assert_eq!(
            StackFrame::new(&mut asm, 5, 0, 0).unwrap_err(),
            Error::BadPnum
        );
    }

    #[test]
    fn test_bad_tnum() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        // 4 params + 11 temps = 15 > 14
        assert_eq!(
            StackFrame::new(&mut asm, 4, 11, 0).unwrap_err(),
            Error::BadTnum
        );
    }

    #[test]
    fn test_get_reg_idx_no_skip() {
        let mut pos = 0;
        let r0 = next_register(&mut pos, 0);
        let r1 = next_register(&mut pos, 0);
        assert_eq!(r0, REG_ORDER[0]);
        assert_eq!(r1, REG_ORDER[1]);
        assert_eq!(pos, 2);
    }
}
