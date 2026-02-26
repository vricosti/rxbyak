/// Error validation tests — ensures rxbyak correctly rejects invalid inputs.
/// Ported from xbyak's bad_address.cpp.

use rxbyak::*;

// ─── ESP/RSP as index register ──────────────────────────────────

#[test]
fn test_esp_as_index_rejected() {
    // ESP cannot be used as an index register in SIB encoding.
    // dword_ptr() panics via .expect(), so we catch_unwind.
    let result = std::panic::catch_unwind(|| {
        dword_ptr(EAX + ESP * 2)
    });
    assert!(result.is_err(), "ESP as index should be rejected");
}

#[test]
fn test_rsp_as_index_rejected() {
    // RSP cannot be used as an index register.
    let result = std::panic::catch_unwind(|| {
        qword_ptr(RAX + RSP * 2)
    });
    assert!(result.is_err(), "RSP as index should be rejected");
}

// ─── Invalid scale values ───────────────────────────────────────

#[test]
fn test_invalid_scale_3() {
    // Scale 3 is not valid (only 1, 2, 4, 8)
    let result = std::panic::catch_unwind(|| {
        let _ = RAX * 3u8;
    });
    assert!(result.is_err(), "Scale 3 should be rejected");
}

#[test]
fn test_invalid_scale_5() {
    let result = std::panic::catch_unwind(|| {
        let _ = RAX * 5u8;
    });
    assert!(result.is_err(), "Scale 5 should be rejected");
}

#[test]
fn test_invalid_scale_16() {
    let result = std::panic::catch_unwind(|| {
        let _ = RAX * 16u8;
    });
    assert!(result.is_err(), "Scale 16 should be rejected");
}

// ─── Register size mismatches ───────────────────────────────────

#[test]
fn test_mov_size_mismatch_32_64() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // Cannot mov between different-sized GPRs without explicit extension
    let result = asm.add(EAX, RAX);
    assert!(result.is_err(), "add eax, rax should fail (size mismatch)");
}

#[test]
fn test_xchg_size_mismatch() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let result = asm.xchg(EAX, RAX);
    assert!(result.is_err(), "xchg with different sizes should fail");
}

// ─── movzx/movsx invalid combinations ───────────────────────────

#[test]
fn test_movzx_same_size() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // movzx dst, src where dst.size <= src.size should fail
    let result = asm.movzx(EAX, EAX);
    assert!(result.is_err(), "movzx eax, eax should fail (same size)");
}

#[test]
fn test_movsx_32bit_src() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // movsx with 32-bit source should fail (use movsxd instead)
    let result = asm.movsx(RAX, EAX);
    assert!(result.is_err(), "movsx rax, eax should fail (use movsxd)");
}

#[test]
fn test_movsxd_non64_dst() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // movsxd requires 64-bit destination
    let result = asm.movsxd(EAX, ECX);
    assert!(result.is_err(), "movsxd eax, ecx should fail (need 64-bit dst)");
}

// ─── EVEX invalid combinations ──────────────────────────────────

#[test]
fn test_k0_as_writemask() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // k0 cannot be used as a writemask (it means "no mask")
    // Using .k(0) should result in no masking, which is allowed
    // but some instructions may reject it explicitly
    let result = asm.vaddps(ZMM0.k(0), ZMM1, ZMM2);
    // k0 means "no mask", so this should actually succeed
    assert!(result.is_ok(), "k0 should be treated as 'no mask'");
}

#[test]
fn test_zeroing_without_mask() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // In rxbyak, zeroing without a mask is silently ignored (z_bit forced false
    // when aaa==0), matching xbyak behavior. Verify it doesn't crash.
    let result = asm.vaddps(ZMM0.z(), ZMM1, ZMM2);
    assert!(result.is_ok(), "zeroing without mask is silently ignored");
}

// ─── Label errors ───────────────────────────────────────────────

#[test]
fn test_undefined_label() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let label = asm.create_label();
    asm.jmp(&label, JmpType::Near).unwrap();
    // Don't bind the label — ready() should fail
    let result = asm.ready();
    assert!(result.is_err(), "undefined label should cause ready() to fail");
}

#[test]
fn test_label_redefined() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let label = asm.create_label();
    asm.bind(&label).unwrap();
    asm.nop().unwrap();
    let result = asm.bind(&label);
    assert!(result.is_err(), "binding same label twice should fail");
}

// ─── Push/pop invalid register sizes ────────────────────────────

#[test]
fn test_push_32bit_in_64bit_mode() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // In 64-bit mode, push of 32-bit register is not encodable
    let result = asm.push(EAX);
    assert!(result.is_err(), "push eax in 64-bit mode should fail");
}

// ─── Code buffer overflow ───────────────────────────────────────

#[test]
fn test_code_too_big() {
    let mut asm = CodeAssembler::new(4).unwrap();
    // Fill up the tiny buffer
    asm.nop().unwrap();
    asm.nop().unwrap();
    asm.nop().unwrap();
    asm.nop().unwrap();
    // This should fail — buffer is full
    let result = asm.nop();
    assert!(result.is_err(), "exceeding code buffer should fail");
}

// ─── Memory size not specified ──────────────────────────────────

#[test]
fn test_shift_mem_no_size() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // shl [mem], imm requires memory size to be specified
    let result = asm.shl(ptr(RAX.into()), 4);
    assert!(result.is_err(), "shift on unsized memory should fail");
}

// ─── Valid operations that should succeed ────────────────────────

#[test]
fn test_valid_scales() {
    // All valid scales should work
    let _ = RAX * 1u8;
    let _ = RAX * 2u8;
    let _ = RAX * 4u8;
    let _ = RAX * 8u8;
}

#[test]
fn test_valid_evex_masks() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    // k1-k7 should all work as writemasks
    for k in 1u8..=7 {
        asm.vaddps(ZMM0.k(k), ZMM1, ZMM2).unwrap();
    }
}

#[test]
fn test_valid_rounding_modes() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    asm.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RnSae)).unwrap();
    asm.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RdSae)).unwrap();
    asm.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RuSae)).unwrap();
    asm.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RzSae)).unwrap();
}
