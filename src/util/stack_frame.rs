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

use crate::assembler::CodeAssembler;
use crate::error::{Error, Result};
use crate::operand::Reg;
use crate::reg::*;

/// Bit flag to request RCX as a temporary register.
///
/// OR this with `t_num` in [`StackFrame::new`]. When RCX holds a parameter
/// value (determined by the calling convention), its value is preserved
/// into R10 via `mov r10, rcx` before the function body.
pub const USE_RCX: usize = 1 << 6;

/// Bit flag to request RDX as a temporary register.
///
/// OR this with `t_num` in [`StackFrame::new`]. When RDX holds a parameter
/// value, its value is preserved into R11 via `mov r11, rdx`.
pub const USE_RDX: usize = 1 << 7;

const MAX_PARAMS: usize = 4;
const MAX_TEMPS: usize = 10;

/// Total usable GPRs: 16 minus RSP and RAX = 14.
const MAX_REG_NUM: usize = 14;

/// Register allocation order: parameter regs first, then scratch, then callee-saved.
/// Matches xbyak's `getOrderTbl()` (16 - RSP - RAX = 14 entries).
#[cfg(target_os = "windows")]
const REG_ORDER: [Reg; MAX_REG_NUM] = [
    RCX, RDX, R8, R9,      // param regs (4)
    R10, R11,               // scratch (2) — total scratch (noSaveNum) = 6
    RDI, RSI, RBX, RBP,    // callee-saved
    R12, R13, R14, R15,
];

#[cfg(not(target_os = "windows"))]
const REG_ORDER: [Reg; MAX_REG_NUM] = [
    RDI, RSI, RDX, RCX,    // param regs (first 4 of 6)
    R8, R9,                 // param regs 5-6 (still scratch)
    R10, R11,               // scratch — total scratch (noSaveNum) = 8
    RBX, RBP,               // callee-saved
    R12, R13, R14, R15,
];

/// Number of registers at the start of REG_ORDER that do NOT need saving
/// (i.e., caller-saved / scratch registers).
#[cfg(target_os = "windows")]
const NO_SAVE_NUM: usize = 6;

#[cfg(not(target_os = "windows"))]
const NO_SAVE_NUM: usize = 8;

/// Position of RCX in REG_ORDER.
#[cfg(target_os = "windows")]
const RCX_POS: usize = 0;

#[cfg(not(target_os = "windows"))]
const RCX_POS: usize = 3;

/// Position of RDX in REG_ORDER.
#[cfg(target_os = "windows")]
const RDX_POS: usize = 1;

#[cfg(not(target_os = "windows"))]
const RDX_POS: usize = 2;

/// Automatic function prologue/epilogue generator for x86-64 calling conventions.
///
/// Creates a proper stack frame with callee-saved register preservation,
/// stack alignment, and parameter/temporary register mapping.
///
/// When `USE_RCX` or `USE_RDX` flags are set, those registers are skipped
/// in the allocation order (so they remain free for special uses like shifts).
/// If the skipped register was holding a parameter value, the value is
/// preserved: RCX → R10, RDX → R11.
#[derive(Debug)]
pub struct StackFrame {
    /// Parameter registers. `p[0..pNum]` are mapped to the platform's
    /// parameter passing registers (skipping RCX/RDX if USE_RCX/USE_RDX set).
    pub p: [Reg; MAX_PARAMS],
    /// Temporary registers. `t[0..tNum]` are allocated from the register
    /// order table after the parameter registers.
    pub t: [Reg; MAX_TEMPS],
    /// Number of callee-saved registers that were pushed.
    save_num: usize,
    /// Stack space allocated via `sub rsp` (includes alignment padding), in bytes.
    p_: usize,
}

impl StackFrame {
    /// Generate a function prologue and return the register mapping.
    ///
    /// # Parameters
    ///
    /// - `asm`: The assembler to emit prologue code into.
    /// - `p_num`: Number of function parameters (0..=4).
    /// - `t_num`: Number of temporary registers needed, optionally OR'd with
    ///   [`USE_RCX`] and/or [`USE_RDX`].
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

        let use_rcx = (t_num & USE_RCX) != 0;
        let use_rdx = (t_num & USE_RDX) != 0;
        let t_num_actual = t_num & !(USE_RCX | USE_RDX);

        // USE_RCX/USE_RDX each consume an extra register slot (the skipped
        // register is replaced by the next one in order).
        let all_reg_num = p_num
            + t_num_actual
            + use_rcx as usize
            + use_rdx as usize;
        if all_reg_num > MAX_REG_NUM {
            return Err(Error::BadTnum);
        }

        // How many callee-saved registers we need to push.
        let save_num = all_reg_num.saturating_sub(NO_SAVE_NUM);

        // Push callee-saved registers.
        for i in 0..save_num {
            asm.push(REG_ORDER[NO_SAVE_NUM + i])?;
        }

        // Stack alignment for local space.
        // P = number of 8-byte slots for local stack (rounded up).
        // We need (save_num + P + 1) to be even for 16-byte alignment
        // (the +1 is the return address). So if P > 0 and P and save_num
        // have the same parity, add one more slot.
        let mut p_slots = stack_size.div_ceil(8);
        if p_slots > 0 && (p_slots & 1) == (save_num & 1) {
            p_slots += 1;
        }
        let p_ = p_slots * 8;
        if p_ > 0 {
            asm.sub(RSP, p_ as i64)?;
        }

        // Map parameter and temporary registers using getRegIdx logic:
        // walk through REG_ORDER, skipping RCX if use_rcx, RDX if use_rdx.
        let mut pos = 0usize;

        let mut p = [Reg::gpr64(0); MAX_PARAMS];
        for slot in p.iter_mut().take(p_num) {
            *slot = get_reg_idx(&mut pos, use_rcx, use_rdx);
        }

        let mut t = [Reg::gpr64(0); MAX_TEMPS];
        for slot in t.iter_mut().take(t_num_actual) {
            *slot = get_reg_idx(&mut pos, use_rcx, use_rdx);
        }

        // Preserve parameter values displaced by USE_RCX/USE_RDX.
        // If RCX held a parameter (rcxPos < pNum), save it to R10.
        if use_rcx && RCX_POS < p_num {
            asm.mov(R10, RCX)?;
        }
        // If RDX held a parameter (rdxPos < pNum), save it to R11.
        if use_rdx && RDX_POS < p_num {
            asm.mov(R11, RDX)?;
        }

        Ok(StackFrame { p, t, save_num, p_ })
    }

    /// Generate the function epilogue.
    ///
    /// Emits: restore local stack → pop callee-saved registers (reverse order)
    /// → `ret`.
    pub fn close(self, asm: &mut CodeAssembler) -> Result<()> {
        // Restore local stack
        if self.p_ > 0 {
            asm.add(RSP, self.p_ as i64)?;
        }

        // Pop callee-saved registers in reverse order
        for i in (0..self.save_num).rev() {
            asm.pop(REG_ORDER[NO_SAVE_NUM + i])?;
        }

        asm.ret()
    }
}

/// Walk through REG_ORDER, advancing `pos`, skipping RCX/RDX as requested.
fn get_reg_idx(pos: &mut usize, skip_rcx: bool, skip_rdx: bool) -> Reg {
    loop {
        let r = REG_ORDER[*pos];
        *pos += 1;
        if skip_rcx && r == RCX {
            continue;
        }
        if skip_rdx && r == RDX {
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
        assert!(NO_SAVE_NUM <= MAX_REG_NUM);
    }

    #[test]
    fn test_rcx_rdx_pos_within_bounds() {
        assert!(RCX_POS < MAX_REG_NUM);
        assert!(RDX_POS < MAX_REG_NUM);
    }

    #[test]
    fn test_bad_pnum() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        assert_eq!(StackFrame::new(&mut asm, 5, 0, 0).unwrap_err(), Error::BadPnum);
    }

    #[test]
    fn test_bad_tnum() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        // 4 params + 11 temps = 15 > 14
        assert_eq!(StackFrame::new(&mut asm, 4, 11, 0).unwrap_err(), Error::BadTnum);
    }

    #[test]
    fn test_get_reg_idx_no_skip() {
        let mut pos = 0;
        let r0 = get_reg_idx(&mut pos, false, false);
        let r1 = get_reg_idx(&mut pos, false, false);
        assert_eq!(r0, REG_ORDER[0]);
        assert_eq!(r1, REG_ORDER[1]);
        assert_eq!(pos, 2);
    }
}
