use rxbyak::*;
use rxbyak::util::stack_frame::StackFrame;

fn assemble_sf(
    p_num: usize,
    t_num: usize,
    stack_size: usize,
    body: impl FnOnce(&mut CodeAssembler, &StackFrame) -> Result<()>,
) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, p_num, t_num, stack_size).unwrap();
    body(&mut asm, &sf).unwrap();
    sf.close(&mut asm).unwrap();
    asm.code().to_vec()
}

fn assemble_sf_no_body(p_num: usize, t_num: usize, stack_size: usize) -> Vec<u8> {
    assemble_sf(p_num, t_num, stack_size, |_, _| Ok(()))
}

// ─── Register mapping tests ──────────────────────────────────────────

// These tests verify the register allocation matches the platform's ABI.
// System V AMD64 order: RDI, RSI, RDX, RCX, R8, R9, R10, R11, RBX, RBP, R12-R15
// Windows x64 order:    RCX, RDX, R8, R9, R10, R11, RDI, RSI, RBX, RBP, R12-R15

#[test]
fn test_reg_mapping_p1_t0() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 1, 0, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    assert_eq!(sf.p[0], RDI);
    #[cfg(target_os = "windows")]
    assert_eq!(sf.p[0], RCX);
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_reg_mapping_p2_t0() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 2, 0, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
    }
    #[cfg(target_os = "windows")]
    {
        assert_eq!(sf.p[0], RCX);
        assert_eq!(sf.p[1], RDX);
    }
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_reg_mapping_p4_t0() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 4, 0, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
        assert_eq!(sf.p[2], RDX);
        assert_eq!(sf.p[3], RCX);
    }
    #[cfg(target_os = "windows")]
    {
        assert_eq!(sf.p[0], RCX);
        assert_eq!(sf.p[1], RDX);
        assert_eq!(sf.p[2], R8);
        assert_eq!(sf.p[3], R9);
    }
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_reg_mapping_p0_t3() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 0, 3, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(sf.t[0], RDI);
        assert_eq!(sf.t[1], RSI);
        assert_eq!(sf.t[2], RDX);
    }
    #[cfg(target_os = "windows")]
    {
        assert_eq!(sf.t[0], RCX);
        assert_eq!(sf.t[1], RDX);
        assert_eq!(sf.t[2], R8);
    }
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_reg_mapping_p2_t2() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 2, 2, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
        assert_eq!(sf.t[0], RDX);
        assert_eq!(sf.t[1], RCX);
    }
    #[cfg(target_os = "windows")]
    {
        assert_eq!(sf.p[0], RCX);
        assert_eq!(sf.p[1], RDX);
        assert_eq!(sf.t[0], R8);
        assert_eq!(sf.t[1], R9);
    }
    sf.close(&mut asm).unwrap();
}

// ─── USE_RCX / USE_RDX register mapping tests ───────────────────────

#[test]
fn test_use_rcx_skips_rcx_in_mapping() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 4, 0 | USE_RCX, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        // Linux order: RDI, RSI, RDX, [RCX skipped], R8, R9, R10, R11, ...
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
        assert_eq!(sf.p[2], RDX);
        assert_eq!(sf.p[3], R8); // RCX skipped
    }
    #[cfg(target_os = "windows")]
    {
        // Windows order: [RCX skipped], RDX, R8, R9, R10, R11, ...
        assert_eq!(sf.p[0], RDX);
        assert_eq!(sf.p[1], R8);
        assert_eq!(sf.p[2], R9);
        assert_eq!(sf.p[3], R10);
    }
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_use_rdx_skips_rdx_in_mapping() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 4, 0 | USE_RDX, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        // Linux order: RDI, RSI, [RDX skipped], RCX, R8, R9, ...
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
        assert_eq!(sf.p[2], RCX); // RDX skipped
        assert_eq!(sf.p[3], R8);
    }
    #[cfg(target_os = "windows")]
    {
        // Windows order: RCX, [RDX skipped], R8, R9, R10, ...
        assert_eq!(sf.p[0], RCX);
        assert_eq!(sf.p[1], R8);
        assert_eq!(sf.p[2], R9);
        assert_eq!(sf.p[3], R10);
    }
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_use_rcx_rdx_together() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let sf = StackFrame::new(&mut asm, 4, 0 | USE_RCX | USE_RDX, 0).unwrap();
    #[cfg(not(target_os = "windows"))]
    {
        // Linux: RDI, RSI, [RDX skip], [RCX skip], R8, R9, R10, R11, ...
        assert_eq!(sf.p[0], RDI);
        assert_eq!(sf.p[1], RSI);
        assert_eq!(sf.p[2], R8);
        assert_eq!(sf.p[3], R9);
    }
    #[cfg(target_os = "windows")]
    {
        // Windows: [RCX skip], [RDX skip], R8, R9, R10, R11, ...
        assert_eq!(sf.p[0], R8);
        assert_eq!(sf.p[1], R9);
        assert_eq!(sf.p[2], R10);
        assert_eq!(sf.p[3], R11);
    }
    sf.close(&mut asm).unwrap();
}

// ─── Error handling tests ────────────────────────────────────────────

#[test]
fn test_bad_pnum_5() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(StackFrame::new(&mut asm, 5, 0, 0).unwrap_err(), Error::BadPnum);
}

#[test]
fn test_bad_tnum_overflow() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // 4 + 11 = 15 > 14
    assert_eq!(StackFrame::new(&mut asm, 4, 11, 0).unwrap_err(), Error::BadTnum);
}

#[test]
fn test_max_regs_ok() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // 4 + 10 = 14 = MAX_REG_NUM — should work
    let sf = StackFrame::new(&mut asm, 4, 10, 0).unwrap();
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_max_regs_with_use_rcx() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // 4 + 9 + 1(USE_RCX) = 14 — should work
    let sf = StackFrame::new(&mut asm, 4, 9 | USE_RCX, 0).unwrap();
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_max_regs_with_use_rcx_rdx() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // 4 + 8 + 2(USE_RCX+USE_RDX) = 14 — should work
    let sf = StackFrame::new(&mut asm, 4, 8 | USE_RCX | USE_RDX, 0).unwrap();
    sf.close(&mut asm).unwrap();
}

#[test]
fn test_use_rcx_rdx_causes_overflow() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // 4 + 9 + 2 = 15 > 14
    assert_eq!(
        StackFrame::new(&mut asm, 4, 9 | USE_RCX | USE_RDX, 0).unwrap_err(),
        Error::BadTnum
    );
}

// ─── Byte-level encoding tests (System V ABI) ───────────────────────

#[cfg(not(target_os = "windows"))]
mod sysv_encoding {
    use super::*;

    #[test]
    fn test_empty_frame() {
        // pNum=0, tNum=0, stack=0 → just ret
        let code = assemble_sf_no_body(0, 0, 0);
        assert_eq!(code, [0xC3]); // ret
    }

    #[test]
    fn test_p1_t0_no_stack() {
        // No callee-saved regs needed (1 < 8), no stack → just ret
        let code = assemble_sf_no_body(1, 0, 0);
        assert_eq!(code, [0xC3]);
    }

    #[test]
    fn test_p4_t5_no_stack() {
        // allRegNum=9, saveNum=1, push RBX
        // P_ = 0
        // Epilogue: pop RBX; ret
        let code = assemble_sf_no_body(4, 5, 0);
        assert_eq!(
            code,
            [
                0x53,       // push rbx
                0x5B,       // pop rbx
                0xC3,       // ret
            ]
        );
    }

    #[test]
    fn test_p4_t6_no_stack() {
        // allRegNum=10, saveNum=2, push RBX, push RBP
        let code = assemble_sf_no_body(4, 6, 0);
        assert_eq!(
            code,
            [
                0x53,       // push rbx
                0x55,       // push rbp
                0x5D,       // pop rbp
                0x5B,       // pop rbx
                0xC3,       // ret
            ]
        );
    }

    #[test]
    fn test_p4_t7_no_stack() {
        // allRegNum=11, saveNum=3, push RBX, RBP, R12
        let code = assemble_sf_no_body(4, 7, 0);
        assert_eq!(
            code,
            [
                0x53,             // push rbx
                0x55,             // push rbp
                0x41, 0x54,       // push r12
                0x41, 0x5C,       // pop r12
                0x5D,             // pop rbp
                0x5B,             // pop rbx
                0xC3,             // ret
            ]
        );
    }

    #[test]
    fn test_stack_8_no_save() {
        // allRegNum=0, saveNum=0
        // P_ = (8+7)/8 = 1 slot, (1 & 1) == (0 & 1)? no → P_ = 8
        let code = assemble_sf_no_body(0, 0, 8);
        assert_eq!(
            code,
            [
                0x48, 0x83, 0xEC, 0x08, // sub rsp, 8
                0x48, 0x83, 0xC4, 0x08, // add rsp, 8
                0xC3,                    // ret
            ]
        );
    }

    #[test]
    fn test_stack_16_no_save() {
        // P_ = (16+7)/8 = 2 slots, (2 & 1) == (0 & 1)? yes → 3 slots → P_ = 24
        let code = assemble_sf_no_body(0, 0, 16);
        assert_eq!(
            code,
            [
                0x48, 0x83, 0xEC, 0x18, // sub rsp, 24
                0x48, 0x83, 0xC4, 0x18, // add rsp, 24
                0xC3,                    // ret
            ]
        );
    }

    #[test]
    fn test_stack_24_no_save() {
        // P_ = (24+7)/8 = 3 slots, (3 & 1) == (0 & 1)? no → P_ = 24
        let code = assemble_sf_no_body(0, 0, 24);
        assert_eq!(
            code,
            [
                0x48, 0x83, 0xEC, 0x18, // sub rsp, 24
                0x48, 0x83, 0xC4, 0x18, // add rsp, 24
                0xC3,                    // ret
            ]
        );
    }

    #[test]
    fn test_stack_8_save1() {
        // saveNum=1 (e.g., p4+t5), P_ = 1 slot
        // (1 & 1) == (1 & 1)? yes → 2 slots → P_ = 16
        let code = assemble_sf_no_body(4, 5, 8);
        assert_eq!(
            code,
            [
                0x53,                    // push rbx
                0x48, 0x83, 0xEC, 0x10, // sub rsp, 16
                0x48, 0x83, 0xC4, 0x10, // add rsp, 16
                0x5B,                    // pop rbx
                0xC3,                    // ret
            ]
        );
    }

    #[test]
    fn test_stack_8_save2() {
        // saveNum=2 (e.g., p4+t6), P_ = 1 slot
        // (1 & 1) == (2 & 1)? no → P_ = 8
        let code = assemble_sf_no_body(4, 6, 8);
        assert_eq!(
            code,
            [
                0x53,                    // push rbx
                0x55,                    // push rbp
                0x48, 0x83, 0xEC, 0x08, // sub rsp, 8
                0x48, 0x83, 0xC4, 0x08, // add rsp, 8
                0x5D,                    // pop rbp
                0x5B,                    // pop rbx
                0xC3,                    // ret
            ]
        );
    }

    #[test]
    fn test_use_rcx_emits_mov_r10_rcx() {
        // pNum=4, USE_RCX → RCX_POS(3) < pNum(4), so emit mov r10, rcx
        let code = assemble_sf_no_body(4, USE_RCX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xCA, // mov r10, rcx
                0xC3,             // ret
            ]
        );
    }

    #[test]
    fn test_use_rdx_emits_mov_r11_rdx() {
        // pNum=4, USE_RDX → RDX_POS(2) < pNum(4), so emit mov r11, rdx
        let code = assemble_sf_no_body(4, USE_RDX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xD3, // mov r11, rdx
                0xC3,             // ret
            ]
        );
    }

    #[test]
    fn test_use_rcx_rdx_emits_both_movs() {
        let code = assemble_sf_no_body(4, USE_RCX | USE_RDX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xCA, // mov r10, rcx
                0x49, 0x89, 0xD3, // mov r11, rdx
                0xC3,             // ret
            ]
        );
    }

    #[test]
    fn test_use_rcx_no_mov_when_rcx_not_param() {
        // pNum=2, USE_RCX → RCX_POS(3) >= pNum(2), no mov needed
        let code = assemble_sf_no_body(2, USE_RCX, 0);
        assert_eq!(code, [0xC3]); // just ret
    }

    #[test]
    fn test_use_rdx_no_mov_when_rdx_not_param() {
        // pNum=1, USE_RDX → RDX_POS(2) >= pNum(1), no mov needed
        let code = assemble_sf_no_body(1, USE_RDX, 0);
        assert_eq!(code, [0xC3]); // just ret
    }

    #[test]
    fn test_large_stack() {
        // Stack = 200 bytes → P_ = (200+7)/8 = 25 slots
        // (25 & 1) == (0 & 1)? no → P_ = 200
        // 200 > 127, so sub rsp uses imm32
        let code = assemble_sf_no_body(0, 0, 200);
        assert_eq!(
            code,
            [
                0x48, 0x81, 0xEC, 0xC8, 0x00, 0x00, 0x00, // sub rsp, 200
                0x48, 0x81, 0xC4, 0xC8, 0x00, 0x00, 0x00, // add rsp, 200
                0xC3,                                       // ret
            ]
        );
    }

    #[test]
    fn test_all_callee_saved_pushed() {
        // pNum=4, tNum=10, allRegNum=14, saveNum=6
        // Push: RBX, RBP, R12, R13, R14, R15
        let code = assemble_sf_no_body(4, 10, 0);
        assert_eq!(
            code,
            [
                0x53,             // push rbx
                0x55,             // push rbp
                0x41, 0x54,       // push r12
                0x41, 0x55,       // push r13
                0x41, 0x56,       // push r14
                0x41, 0x57,       // push r15
                0x41, 0x5F,       // pop r15
                0x41, 0x5E,       // pop r14
                0x41, 0x5D,       // pop r13
                0x41, 0x5C,       // pop r12
                0x5D,             // pop rbp
                0x5B,             // pop rbx
                0xC3,             // ret
            ]
        );
    }

    #[test]
    fn test_callee_saved_with_stack() {
        // pNum=4, tNum=10, saveNum=6, stack=32
        // P_ = (32+7)/8 = 4 slots, (4 & 1) == (6 & 1)? 0==0 yes → 5 slots → P_ = 40
        let code = assemble_sf_no_body(4, 10, 32);
        assert_eq!(
            code,
            [
                0x53,                    // push rbx
                0x55,                    // push rbp
                0x41, 0x54,              // push r12
                0x41, 0x55,              // push r13
                0x41, 0x56,              // push r14
                0x41, 0x57,              // push r15
                0x48, 0x83, 0xEC, 0x28, // sub rsp, 40
                0x48, 0x83, 0xC4, 0x28, // add rsp, 40
                0x41, 0x5F,              // pop r15
                0x41, 0x5E,              // pop r14
                0x41, 0x5D,              // pop r13
                0x41, 0x5C,              // pop r12
                0x5D,                    // pop rbp
                0x5B,                    // pop rbx
                0xC3,                    // ret
            ]
        );
    }
}

// ─── Byte-level encoding tests (Windows x64 ABI) ────────────────────

#[cfg(target_os = "windows")]
mod win64_encoding {
    use super::*;

    #[test]
    fn test_empty_frame() {
        let code = assemble_sf_no_body(0, 0, 0);
        assert_eq!(code, [0xC3]); // ret
    }

    #[test]
    fn test_p4_t3_no_stack() {
        // allRegNum=7, saveNum=max(0, 7-6)=1
        // Windows callee-saved start at index 6: RDI
        // push RDI = 57
        let code = assemble_sf_no_body(4, 3, 0);
        assert_eq!(
            code,
            [
                0x57,       // push rdi
                0x5F,       // pop rdi
                0xC3,       // ret
            ]
        );
    }

    #[test]
    fn test_p4_t4_no_stack() {
        // allRegNum=8, saveNum=2: push RDI(57), RSI(56)
        let code = assemble_sf_no_body(4, 4, 0);
        assert_eq!(
            code,
            [
                0x57,       // push rdi
                0x56,       // push rsi
                0x5E,       // pop rsi
                0x5F,       // pop rdi
                0xC3,       // ret
            ]
        );
    }

    #[test]
    fn test_p4_t5_no_stack() {
        // allRegNum=9, saveNum=3: push RDI(57), RSI(56), RBX(53)
        let code = assemble_sf_no_body(4, 5, 0);
        assert_eq!(
            code,
            [
                0x57,       // push rdi
                0x56,       // push rsi
                0x53,       // push rbx
                0x5B,       // pop rbx
                0x5E,       // pop rsi
                0x5F,       // pop rdi
                0xC3,       // ret
            ]
        );
    }

    #[test]
    fn test_stack_8_no_save() {
        let code = assemble_sf_no_body(0, 0, 8);
        assert_eq!(
            code,
            [
                0x48, 0x83, 0xEC, 0x08, // sub rsp, 8
                0x48, 0x83, 0xC4, 0x08, // add rsp, 8
                0xC3,
            ]
        );
    }

    #[test]
    fn test_stack_16_no_save() {
        // P_ = 2 slots, (2 & 1)==(0 & 1)? yes → 3 slots → 24
        let code = assemble_sf_no_body(0, 0, 16);
        assert_eq!(
            code,
            [
                0x48, 0x83, 0xEC, 0x18, // sub rsp, 24
                0x48, 0x83, 0xC4, 0x18, // add rsp, 24
                0xC3,
            ]
        );
    }

    #[test]
    fn test_use_rcx_emits_mov_r10_rcx() {
        // Windows: RCX_POS=0, pNum=4 → 0 < 4 → emit mov r10, rcx
        let code = assemble_sf_no_body(4, USE_RCX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xCA, // mov r10, rcx
                0xC3,
            ]
        );
    }

    #[test]
    fn test_use_rdx_emits_mov_r11_rdx() {
        // Windows: RDX_POS=1, pNum=4 → 1 < 4 → emit mov r11, rdx
        let code = assemble_sf_no_body(4, USE_RDX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xD3, // mov r11, rdx
                0xC3,
            ]
        );
    }

    #[test]
    fn test_use_rcx_no_mov_when_not_param() {
        // Windows: RCX_POS=0, pNum=0 → 0 >= 0 → no mov
        let code = assemble_sf_no_body(0, USE_RCX, 0);
        assert_eq!(code, [0xC3]);
    }

    #[test]
    fn test_all_callee_saved_pushed() {
        // pNum=4, tNum=10, allRegNum=14, saveNum=8
        // Windows callee-saved: RDI, RSI, RBX, RBP, R12, R13, R14, R15
        let code = assemble_sf_no_body(4, 10, 0);
        assert_eq!(
            code,
            [
                0x57,             // push rdi
                0x56,             // push rsi
                0x53,             // push rbx
                0x55,             // push rbp
                0x41, 0x54,       // push r12
                0x41, 0x55,       // push r13
                0x41, 0x56,       // push r14
                0x41, 0x57,       // push r15
                0x41, 0x5F,       // pop r15
                0x41, 0x5E,       // pop r14
                0x41, 0x5D,       // pop r13
                0x41, 0x5C,       // pop r12
                0x5D,             // pop rbp
                0x5B,             // pop rbx
                0x5E,             // pop rsi
                0x5F,             // pop rdi
                0xC3,
            ]
        );
    }
}

// ─── Windows push/pop balance ────────────────────────────────────────

#[test]
#[cfg(target_os = "windows")]
fn test_push_pop_balance_windows() {
    for total in 0..=14usize {
        let p_num = total.min(4);
        let t_num = total - p_num;
        let code = assemble_sf_no_body(p_num, t_num, 0);

        let save_num = total.saturating_sub(6); // NO_SAVE_NUM = 6 on Windows
        let mut push_count = 0;
        let mut pop_count = 0;
        let mut i = 0;
        while i < code.len() {
            let b = code[i];
            if b == 0x41 && i + 1 < code.len() {
                let next = code[i + 1];
                if (0x50..=0x57).contains(&next) {
                    push_count += 1;
                    i += 2;
                    continue;
                } else if (0x58..=0x5F).contains(&next) {
                    pop_count += 1;
                    i += 2;
                    continue;
                }
            }
            if (0x50..=0x57).contains(&b) {
                push_count += 1;
            } else if (0x58..=0x5F).contains(&b) {
                pop_count += 1;
            }
            i += 1;
        }
        assert_eq!(push_count, save_num, "push count for total={total}");
        assert_eq!(pop_count, save_num, "pop count for total={total}");
    }
}

// ─── Parametric sweep: all valid combinations ────────────────────────

#[test]
fn test_parametric_sweep() {
    // Sweep over all valid combinations of pNum, tNum, stackSize, and mode.
    // Verify that:
    // 1. Code generates without error
    // 2. Code ends with 0xC3 (ret)
    // 3. Code is well-formed (non-empty)
    for p_num in 0..=4 {
        for t_base in 0..=10 {
            for &mode in &[0, USE_RCX, USE_RDX, USE_RCX | USE_RDX] {
                let t_num = t_base | mode;
                let t_actual = t_base;
                let extra = (mode & USE_RCX != 0) as usize
                    + (mode & USE_RDX != 0) as usize;
                let all_reg = p_num + t_actual + extra;
                if all_reg > 14 {
                    continue; // would error
                }

                for &stack in &[0usize, 8, 16, 24, 32, 64, 128] {
                    let mut asm = CodeAssembler::new(4096).unwrap();
                    let result = StackFrame::new(&mut asm, p_num, t_num, stack);
                    match result {
                        Ok(sf) => {
                            sf.close(&mut asm).unwrap();
                            let code = asm.code();
                            assert!(
                                !code.is_empty(),
                                "empty code for p={p_num} t={t_num} stack={stack}"
                            );
                            assert_eq!(
                                *code.last().unwrap(), 0xC3,
                                "last byte not ret for p={p_num} t={t_num} stack={stack}"
                            );
                        }
                        Err(e) => {
                            panic!(
                                "unexpected error for p={p_num} t={t_num} stack={stack}: {e}"
                            );
                        }
                    }
                }
            }
        }
    }
}

// ─── Callee-saved register preservation check ────────────────────────

/// For a given allRegNum, verify that the correct number of push/pop pairs
/// are emitted.
#[test]
#[cfg(not(target_os = "windows"))]
fn test_push_pop_balance() {
    // Count push (0x50-0x57, 41 5x) and pop (0x58-0x5F, 41 5x) instructions
    for total in 0..=14usize {
        let p_num = total.min(4);
        let t_num = total - p_num;
        let code = assemble_sf_no_body(p_num, t_num, 0);

        let save_num = total.saturating_sub(8); // NO_SAVE_NUM = 8 on Linux
        let mut push_count = 0;
        let mut pop_count = 0;
        let mut i = 0;
        while i < code.len() {
            let b = code[i];
            if b == 0x41 && i + 1 < code.len() {
                let next = code[i + 1];
                if (0x50..=0x57).contains(&next) {
                    push_count += 1;
                    i += 2;
                    continue;
                } else if (0x58..=0x5F).contains(&next) {
                    pop_count += 1;
                    i += 2;
                    continue;
                }
            }
            if (0x50..=0x57).contains(&b) {
                push_count += 1;
            } else if (0x58..=0x5F).contains(&b) {
                pop_count += 1;
            }
            i += 1;
        }
        assert_eq!(
            push_count, save_num,
            "push count mismatch for total={total}: expected {save_num}, got {push_count}"
        );
        assert_eq!(
            pop_count, save_num,
            "pop count mismatch for total={total}: expected {save_num}, got {pop_count}"
        );
    }
}

// ─── Execution tests (cross-platform via StackFrame ABI abstraction) ──

#[cfg(target_arch = "x86_64")]
mod execution {
    use super::*;

    /// JIT a function that returns a constant.
    #[test]
    fn test_exec_return_constant() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 0, 0, 0).unwrap();
        asm.mov(RAX, 42i64).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn() -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(), 42);
    }

    /// JIT a function that adds two parameters.
    #[test]
    fn test_exec_add_params() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 2, 0, 0).unwrap();
        asm.mov(RAX, sf.p[0]).unwrap();
        asm.add(RAX, sf.p[1]).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(10, 32), 42);
        assert_eq!(f(100, 200), 300);
        assert_eq!(f(-5, 5), 0);
    }

    /// JIT a function using a temp register.
    #[test]
    fn test_exec_with_temp() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 2, 1, 0).unwrap();
        asm.mov(sf.t[0], sf.p[0]).unwrap();
        asm.add(sf.t[0], sf.p[0]).unwrap(); // t[0] = p[0] * 2
        asm.mov(RAX, sf.t[0]).unwrap();
        asm.add(RAX, sf.p[1]).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(10, 5), 25); // 10*2 + 5 = 25
    }

    /// Verify callee-saved registers are properly preserved.
    #[test]
    fn test_exec_callee_saved_preserved() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 1, 9, 0).unwrap();
        for i in 0..9 {
            asm.mov(sf.t[i], i as i64 + 100).unwrap();
        }
        asm.mov(RAX, sf.p[0]).unwrap();
        asm.add(RAX, sf.t[8]).unwrap(); // t[8] = 108
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(1), 109); // 1 + 108
    }

    /// Test with local stack space.
    #[test]
    fn test_exec_with_stack() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 1, 0, 16).unwrap();
        asm.mov(qword_ptr(RSP.into()), sf.p[0]).unwrap();
        asm.mov(RAX, qword_ptr(RSP.into())).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(99), 99);
    }

    /// Test with 4 parameters.
    #[test]
    fn test_exec_4_params() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 4, 0, 0).unwrap();
        asm.mov(RAX, sf.p[0]).unwrap();
        asm.add(RAX, sf.p[1]).unwrap();
        asm.add(RAX, sf.p[2]).unwrap();
        asm.add(RAX, sf.p[3]).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64, i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(1, 2, 3, 4), 10);
    }

    /// Test with callee-saved regs and stack space together.
    #[test]
    fn test_exec_full_frame() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        let sf = StackFrame::new(&mut asm, 2, 8, 32).unwrap();
        asm.mov(sf.t[0], sf.p[0]).unwrap();
        asm.add(sf.t[0], sf.p[1]).unwrap();
        asm.mov(qword_ptr(RSP.into()), sf.t[0]).unwrap();
        asm.mov(RAX, qword_ptr(RSP.into())).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(100, 200), 300);
    }

    /// Test USE_RCX preserves parameter value (Linux-specific ABI test).
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_exec_use_rcx_linux() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        // Linux: RCX is param[3] (RCX_POS=3), saved to R10
        let sf = StackFrame::new(&mut asm, 4, USE_RCX, 0).unwrap();
        asm.mov(RAX, sf.p[0]).unwrap();
        asm.add(RAX, R10).unwrap(); // R10 has original 4th param
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64, i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(10, 20, 30, 40), 50); // 10 + 40
    }

    /// Test USE_RCX preserves parameter value (Windows-specific ABI test).
    #[cfg(target_os = "windows")]
    #[test]
    fn test_exec_use_rcx_windows() {
        let mut asm = CodeAssembler::new(4096).unwrap();
        // Windows: RCX is param[0] (RCX_POS=0), saved to R10
        // With USE_RCX: p[0]=RDX(param1), p[1]=R8(param2), p[2]=R9(param3), p[3]=R10
        // After mov r10, rcx: R10 has original param[0]
        let sf = StackFrame::new(&mut asm, 4, USE_RCX, 0).unwrap();
        // Return R10 (original param0) + p[0] (which is RDX = param1)
        asm.mov(RAX, R10).unwrap();
        asm.add(RAX, sf.p[0]).unwrap();
        sf.close(&mut asm).unwrap();
        asm.ready().unwrap();
        let f: fn(i64, i64, i64, i64) -> i64 = unsafe { asm.get_code() };
        assert_eq!(f(10, 20, 30, 40), 30); // 10 (from R10) + 20 (from RDX)
    }
}

// ─── Specific gen patterns from xbyak's sf_test.cpp ──────────────────

#[cfg(not(target_os = "windows"))]
mod gen_patterns {
    use super::*;

    /// gen1: pNum=0, tNum=3, stack=0 — no saves needed on Linux
    #[test]
    fn test_gen1() {
        let code = assemble_sf(0, 3, 0, |asm, sf| {
            asm.mov(sf.t[0], sf.t[1])?;
            Ok(())
        });
        // No pushes needed (allRegNum=3 < 8)
        // mov rdi, rsi → 48 89 F7
        // ret → C3
        assert_eq!(
            code,
            [0x48, 0x89, 0xF7, 0xC3]
        );
    }

    /// gen2: pNum=3, tNum=2, stack=0 — no saves needed
    #[test]
    fn test_gen2() {
        let code = assemble_sf(3, 2, 0, |asm, sf| {
            // Move p[0] to t[0]
            asm.mov(sf.t[0], sf.p[0])?;
            Ok(())
        });
        // p[0]=RDI, t[0]=RCX, mov rcx, rdi → 48 89 F9
        assert_eq!(
            code,
            [0x48, 0x89, 0xF9, 0xC3]
        );
    }

    /// gen3: pNum=4, tNum=8, stack=0 — saveNum=4 (RBX, RBP, R12, R13)
    #[test]
    fn test_gen3_structure() {
        let code = assemble_sf_no_body(4, 8, 0);
        // allRegNum=12, saveNum=4
        // Push: RBX(53), RBP(55), R12(41 54), R13(41 55)
        // Pop: R13(41 5D), R12(41 5C), RBP(5D), RBX(5B)
        // ret(C3)
        assert_eq!(
            code,
            [
                0x53, 0x55, 0x41, 0x54, 0x41, 0x55, // push rbx, rbp, r12, r13
                0x41, 0x5D, 0x41, 0x5C, 0x5D, 0x5B, // pop r13, r12, rbp, rbx
                0xC3,
            ]
        );
    }

    /// gen4: with stack space and callee-saved
    #[test]
    fn test_gen4_stack_and_save() {
        // pNum=2, tNum=7, stack=24 → allRegNum=9, saveNum=1 (RBX)
        // P_ = (24+7)/8 = 3 slots, (3 & 1) == (1 & 1)? yes → 4 slots → P_ = 32
        let code = assemble_sf_no_body(2, 7, 24);
        assert_eq!(
            code,
            [
                0x53,                    // push rbx
                0x48, 0x83, 0xEC, 0x20, // sub rsp, 32
                0x48, 0x83, 0xC4, 0x20, // add rsp, 32
                0x5B,                    // pop rbx
                0xC3,
            ]
        );
    }

    /// gen5: USE_RCX with temps and callee-saved
    #[test]
    fn test_gen5_use_rcx_with_temps() {
        // pNum=4, tNum=2|USE_RCX, stack=0
        // allRegNum = 4 + 2 + 1 = 7, saveNum = 0
        // RCX_POS=3 < pNum=4 → mov r10, rcx
        let code = assemble_sf_no_body(4, 2 | USE_RCX, 0);
        assert_eq!(
            code,
            [
                0x49, 0x89, 0xCA, // mov r10, rcx
                0xC3,
            ]
        );
    }

    /// Test with USE_RCX and callee-saved regs
    #[test]
    fn test_use_rcx_with_saves() {
        // pNum=4, tNum=5|USE_RCX → allRegNum=4+5+1=10, saveNum=2
        let code = assemble_sf_no_body(4, 5 | USE_RCX, 0);
        // Push RBX, RBP; mov r10, rcx; ...; pop RBP, RBX; ret
        assert_eq!(code[0], 0x53);       // push rbx
        assert_eq!(code[1], 0x55);       // push rbp
        assert_eq!(&code[2..5], [0x49, 0x89, 0xCA]); // mov r10, rcx
        // Epilogue: pop rbp, pop rbx, ret
        let tail = &code[5..];
        assert_eq!(tail, [0x5D, 0x5B, 0xC3]);
    }
}
