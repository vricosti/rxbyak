/// Tests for auto-generated instruction methods (Phase 6 validation).
/// Verifies byte-level encoding against known-good NASM output.

use rxbyak::*;

fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

// ─── Generated SSE 2-operand instructions ─────────────────────

#[test]
fn test_gen_minps() {
    // minps xmm0, xmm1 → 0F 5D C1
    let code = assemble(|a| a.minps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x5D, 0xC1]);
}

#[test]
fn test_gen_maxps() {
    // maxps xmm0, xmm1 → 0F 5F C1
    let code = assemble(|a| a.maxps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x5F, 0xC1]);
}

#[test]
fn test_gen_minpd() {
    // minpd xmm2, xmm3 → 66 0F 5D D3
    let code = assemble(|a| a.minpd(XMM2, XMM3));
    assert_eq!(code, [0x66, 0x0F, 0x5D, 0xD3]);
}

#[test]
fn test_gen_maxpd() {
    // maxpd xmm2, xmm3 → 66 0F 5F D3
    let code = assemble(|a| a.maxpd(XMM2, XMM3));
    assert_eq!(code, [0x66, 0x0F, 0x5F, 0xD3]);
}

#[test]
fn test_gen_minss() {
    // minss xmm0, xmm1 → F3 0F 5D C1
    let code = assemble(|a| a.minss(XMM0, XMM1));
    assert_eq!(code, [0xF3, 0x0F, 0x5D, 0xC1]);
}

#[test]
fn test_gen_maxss() {
    // maxss xmm0, xmm1 → F3 0F 5F C1
    let code = assemble(|a| a.maxss(XMM0, XMM1));
    assert_eq!(code, [0xF3, 0x0F, 0x5F, 0xC1]);
}

#[test]
fn test_gen_minsd() {
    // minsd xmm0, xmm1 → F2 0F 5D C1
    let code = assemble(|a| a.minsd(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0x5D, 0xC1]);
}

#[test]
fn test_gen_maxsd() {
    // maxsd xmm0, xmm1 → F2 0F 5F C1
    let code = assemble(|a| a.maxsd(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0x5F, 0xC1]);
}

#[test]
fn test_gen_rcpps() {
    // rcpps xmm0, xmm1 → 0F 53 C1
    let code = assemble(|a| a.rcpps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x53, 0xC1]);
}

#[test]
fn test_gen_rsqrtps() {
    // rsqrtps xmm0, xmm1 → 0F 52 C1
    let code = assemble(|a| a.rsqrtps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x52, 0xC1]);
}

#[test]
fn test_gen_unpcklps() {
    // unpcklps xmm0, xmm1 → 0F 14 C1
    let code = assemble(|a| a.unpcklps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x14, 0xC1]);
}

#[test]
fn test_gen_unpckhps() {
    // unpckhps xmm0, xmm1 → 0F 15 C1
    let code = assemble(|a| a.unpckhps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x15, 0xC1]);
}

#[test]
fn test_gen_andnps() {
    // andnps xmm0, xmm1 → 0F 55 C1
    let code = assemble(|a| a.andnps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x55, 0xC1]);
}

#[test]
fn test_gen_andnpd() {
    // andnpd xmm0, xmm1 → 66 0F 55 C1
    let code = assemble(|a| a.andnpd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x55, 0xC1]);
}

// ─── SSE with extended registers ──────────────────────────────

#[test]
fn test_gen_minps_xmm8_xmm9() {
    // minps xmm8, xmm9 → 45 0F 5D C1
    let code = assemble(|a| a.minps(XMM8, XMM9));
    assert_eq!(code, [0x45, 0x0F, 0x5D, 0xC1]);
}

// ─── SSE convert ops ─────────────────────────────────────────

#[test]
fn test_gen_cvtdq2ps() {
    // cvtdq2ps xmm0, xmm1 → 0F 5B C1
    let code = assemble(|a| a.cvtdq2ps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x5B, 0xC1]);
}

#[test]
fn test_gen_cvtps2dq() {
    // cvtps2dq xmm0, xmm1 → 66 0F 5B C1
    let code = assemble(|a| a.cvtps2dq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x5B, 0xC1]);
}

#[test]
fn test_gen_cvtdq2pd() {
    // cvtdq2pd xmm0, xmm1 → F3 0F E6 C1
    let code = assemble(|a| a.cvtdq2pd(XMM0, XMM1));
    assert_eq!(code, [0xF3, 0x0F, 0xE6, 0xC1]);
}

#[test]
fn test_gen_cvtpd2dq() {
    // cvtpd2dq xmm0, xmm1 → F2 0F E6 C1
    let code = assemble(|a| a.cvtpd2dq(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0xE6, 0xC1]);
}

// ─── SSE with imm8 ────────────────────────────────────────────

#[test]
fn test_gen_cmpps() {
    // cmpps xmm0, xmm1, 0 → 0F C2 C1 00
    let code = assemble(|a| a.cmpps(XMM0, XMM1, 0));
    assert_eq!(code, [0x0F, 0xC2, 0xC1, 0x00]);
}

#[test]
fn test_gen_cmppd() {
    // cmppd xmm0, xmm1, 1 → 66 0F C2 C1 01
    let code = assemble(|a| a.cmppd(XMM0, XMM1, 1));
    assert_eq!(code, [0x66, 0x0F, 0xC2, 0xC1, 0x01]);
}

#[test]
fn test_gen_shufps() {
    // shufps xmm0, xmm1, 0 → 0F C6 C1 00
    let code = assemble(|a| a.shufps(XMM0, XMM1, 0));
    assert_eq!(code, [0x0F, 0xC6, 0xC1, 0x00]);
}

#[test]
fn test_gen_shufpd() {
    // shufpd xmm0, xmm1, 1 → 66 0F C6 C1 01
    let code = assemble(|a| a.shufpd(XMM0, XMM1, 1));
    assert_eq!(code, [0x66, 0x0F, 0xC6, 0xC1, 0x01]);
}

// ─── Generated AVX 3-operand instructions ─────────────────────

#[test]
fn test_gen_vminps() {
    // vminps xmm0, xmm1, xmm2 → C5 F0 5D C2
    let code = assemble(|a| a.vminps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF0, 0x5D, 0xC2]);
}

#[test]
fn test_gen_vmaxps() {
    // vmaxps xmm0, xmm1, xmm2 → C5 F0 5F C2
    let code = assemble(|a| a.vmaxps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF0, 0x5F, 0xC2]);
}

#[test]
fn test_gen_vminpd_ymm() {
    // vminpd ymm0, ymm1, ymm2 → C5 F5 5D C2
    let code = assemble(|a| a.vminpd(YMM0, YMM1, YMM2));
    assert_eq!(code, [0xC5, 0xF5, 0x5D, 0xC2]);
}

#[test]
fn test_gen_vmaxpd_ymm() {
    // vmaxpd ymm0, ymm1, ymm2 → C5 F5 5F C2
    let code = assemble(|a| a.vmaxpd(YMM0, YMM1, YMM2));
    assert_eq!(code, [0xC5, 0xF5, 0x5F, 0xC2]);
}

#[test]
fn test_gen_vandnps() {
    // vandnps xmm0, xmm1, xmm2 → C5 F0 55 C2
    let code = assemble(|a| a.vandnps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF0, 0x55, 0xC2]);
}

#[test]
fn test_gen_vunpcklps() {
    // vunpcklps xmm0, xmm1, xmm2 → C5 F0 14 C2
    let code = assemble(|a| a.vunpcklps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF0, 0x14, 0xC2]);
}

#[test]
fn test_gen_vunpckhps() {
    // vunpckhps xmm0, xmm1, xmm2 → C5 F0 15 C2
    let code = assemble(|a| a.vunpckhps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF0, 0x15, 0xC2]);
}

// ─── AVX 3-operand with imm8 ─────────────────────────────────

#[test]
fn test_gen_vcmpps() {
    // vcmpps xmm0, xmm1, xmm2, 0 → C5 F0 C2 C2 00
    let code = assemble(|a| a.vcmpps(XMM0, XMM1, XMM2, 0));
    assert_eq!(code, [0xC5, 0xF0, 0xC2, 0xC2, 0x00]);
}

#[test]
fn test_gen_vcmppd() {
    // vcmppd xmm0, xmm1, xmm2, 1 → C5 F1 C2 C2 01
    let code = assemble(|a| a.vcmppd(XMM0, XMM1, XMM2, 1));
    assert_eq!(code, [0xC5, 0xF1, 0xC2, 0xC2, 0x01]);
}

#[test]
fn test_gen_vshufps() {
    // vshufps xmm0, xmm1, xmm2, 0x44 → C5 F0 C6 C2 44
    let code = assemble(|a| a.vshufps(XMM0, XMM1, XMM2, 0x44));
    assert_eq!(code, [0xC5, 0xF0, 0xC6, 0xC2, 0x44]);
}

#[test]
fn test_gen_vshufpd() {
    // vshufpd xmm0, xmm1, xmm2, 1 → C5 F1 C6 C2 01
    let code = assemble(|a| a.vshufpd(XMM0, XMM1, XMM2, 1));
    assert_eq!(code, [0xC5, 0xF1, 0xC6, 0xC2, 0x01]);
}

// ─── FMA instructions ─────────────────────────────────────────

#[test]
fn test_gen_vfmadd132ps() {
    // vfmadd132ps xmm0, xmm1, xmm2 → C4 E2 71 98 C2
    let code = assemble(|a| a.vfmadd132ps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x98, 0xC2]);
}

#[test]
fn test_gen_vfmadd213ps() {
    // vfmadd213ps xmm0, xmm1, xmm2 → C4 E2 71 A8 C2
    let code = assemble(|a| a.vfmadd213ps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xA8, 0xC2]);
}

#[test]
fn test_gen_vfmadd231ps() {
    // vfmadd231ps xmm0, xmm1, xmm2 → C4 E2 71 B8 C2
    let code = assemble(|a| a.vfmadd231ps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xB8, 0xC2]);
}

#[test]
fn test_gen_vfmadd132pd() {
    // vfmadd132pd xmm0, xmm1, xmm2 → C4 E2 F1 98 C2
    let code = assemble(|a| a.vfmadd132pd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0xF1, 0x98, 0xC2]);
}

#[test]
fn test_gen_vfmsub132ps() {
    // vfmsub132ps xmm0, xmm1, xmm2 → C4 E2 71 9A C2
    let code = assemble(|a| a.vfmsub132ps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x9A, 0xC2]);
}

#[test]
fn test_gen_vfnmadd132ps() {
    // vfnmadd132ps xmm0, xmm1, xmm2 → C4 E2 71 9C C2
    let code = assemble(|a| a.vfnmadd132ps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x9C, 0xC2]);
}

#[test]
fn test_gen_vfmadd132ss() {
    // vfmadd132ss xmm0, xmm1, xmm2 → C4 E2 71 99 C2
    let code = assemble(|a| a.vfmadd132ss(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x99, 0xC2]);
}

#[test]
fn test_gen_vfmadd132sd() {
    // vfmadd132sd xmm0, xmm1, xmm2 → C4 E2 F1 99 C2
    let code = assemble(|a| a.vfmadd132sd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0xF1, 0x99, 0xC2]);
}

#[test]
fn test_gen_vfmadd132ps_ymm() {
    // vfmadd132ps ymm0, ymm1, ymm2 → C4 E2 75 98 C2
    let code = assemble(|a| a.vfmadd132ps(YMM0, YMM1, YMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x75, 0x98, 0xC2]);
}

// ─── VEX 2-operand (no vvvv) ──────────────────────────────────

#[test]
fn test_gen_vsqrtps() {
    // vsqrtps xmm0, xmm1 → C5 F8 51 C1
    let code = assemble(|a| a.vsqrtps(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x51, 0xC1]);
}

#[test]
fn test_gen_vsqrtpd() {
    // vsqrtpd xmm0, xmm1 → C5 F9 51 C1
    let code = assemble(|a| a.vsqrtpd(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF9, 0x51, 0xC1]);
}

#[test]
fn test_gen_vcvtdq2ps() {
    // vcvtdq2ps xmm0, xmm1 → C5 F8 5B C1
    let code = assemble(|a| a.vcvtdq2ps(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x5B, 0xC1]);
}

#[test]
fn test_gen_vcvtps2dq() {
    // vcvtps2dq xmm0, xmm1 → C5 F9 5B C1
    let code = assemble(|a| a.vcvtps2dq(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF9, 0x5B, 0xC1]);
}

#[test]
fn test_gen_vrcpps() {
    // vrcpps xmm0, xmm1 → C5 F8 53 C1
    let code = assemble(|a| a.vrcpps(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x53, 0xC1]);
}

#[test]
fn test_gen_vrsqrtps() {
    // vrsqrtps xmm0, xmm1 → C5 F8 52 C1
    let code = assemble(|a| a.vrsqrtps(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x52, 0xC1]);
}

// ─── VEX 2-operand with imm8 ─────────────────────────────────

#[test]
fn test_gen_vroundps() {
    // vroundps xmm0, xmm1, 0 → C4 E3 79 08 C1 00
    let code = assemble(|a| a.vroundps(XMM0, XMM1, 0));
    assert_eq!(code, [0xC4, 0xE3, 0x79, 0x08, 0xC1, 0x00]);
}

#[test]
fn test_gen_vroundpd() {
    // vroundpd xmm0, xmm1, 0 → C4 E3 79 09 C1 00
    let code = assemble(|a| a.vroundpd(XMM0, XMM1, 0));
    assert_eq!(code, [0xC4, 0xE3, 0x79, 0x09, 0xC1, 0x00]);
}

// ─── VEX mov-style instructions ───────────────────────────────

#[test]
fn test_gen_vmovdqa32_reg() {
    // vmovdqa32 zmm0, zmm1 → 62 F1 7D 48 6F C1
    let code = assemble(|a| a.vmovdqa32(ZMM0, ZMM1));
    assert_eq!(code, [0x62, 0xF1, 0x7D, 0x48, 0x6F, 0xC1]);
}

#[test]
fn test_gen_vmovdqa64_reg() {
    // vmovdqa64 zmm0, zmm1 → 62 F1 FD 48 6F C1
    let code = assemble(|a| a.vmovdqa64(ZMM0, ZMM1));
    assert_eq!(code, [0x62, 0xF1, 0xFD, 0x48, 0x6F, 0xC1]);
}

#[test]
fn test_gen_vmovdqu32_reg() {
    // vmovdqu32 zmm0, zmm1 → 62 F1 7E 48 6F C1
    let code = assemble(|a| a.vmovdqu32(ZMM0, ZMM1));
    assert_eq!(code, [0x62, 0xF1, 0x7E, 0x48, 0x6F, 0xC1]);
}

#[test]
fn test_gen_vmovdqu64_reg() {
    // vmovdqu64 zmm0, zmm1 → 62 F1 FE 48 6F C1
    let code = assemble(|a| a.vmovdqu64(ZMM0, ZMM1));
    assert_eq!(code, [0x62, 0xF1, 0xFE, 0x48, 0x6F, 0xC1]);
}

// ─── AVX-512 3-operand (EVEX) ─────────────────────────────────

#[test]
fn test_gen_vpandd() {
    // vpandd zmm0, zmm1, zmm2 → 62 F1 75 48 DB C2
    let code = assemble(|a| a.vpandd(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0x75, 0x48, 0xDB, 0xC2]);
}

#[test]
fn test_gen_vpandq() {
    // vpandq zmm0, zmm1, zmm2 → 62 F1 F5 48 DB C2
    let code = assemble(|a| a.vpandq(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0xF5, 0x48, 0xDB, 0xC2]);
}

#[test]
fn test_gen_vpord() {
    // vpord zmm0, zmm1, zmm2 → 62 F1 75 48 EB C2
    let code = assemble(|a| a.vpord(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0x75, 0x48, 0xEB, 0xC2]);
}

#[test]
fn test_gen_vpxord() {
    // vpxord zmm0, zmm1, zmm2 → 62 F1 75 48 EF C2
    let code = assemble(|a| a.vpxord(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0x75, 0x48, 0xEF, 0xC2]);
}

#[test]
fn test_gen_vpaddq() {
    // vpaddq zmm0, zmm1, zmm2 → 62 F1 F5 48 D4 C2
    let code = assemble(|a| a.vpaddq(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0xF5, 0x48, 0xD4, 0xC2]);
}

#[test]
fn test_gen_vpsubq() {
    // vpsubq zmm0, zmm1, zmm2 → 62 F1 F5 48 FB C2
    let code = assemble(|a| a.vpsubq(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF1, 0xF5, 0x48, 0xFB, 0xC2]);
}

#[test]
fn test_gen_vpmulld() {
    // vpmulld zmm0, zmm1, zmm2 → 62 F2 75 48 40 C2
    let code = assemble(|a| a.vpmulld(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF2, 0x75, 0x48, 0x40, 0xC2]);
}

#[test]
fn test_gen_vpmullq() {
    // vpmullq zmm0, zmm1, zmm2 → 62 F2 F5 48 40 C2
    let code = assemble(|a| a.vpmullq(ZMM0, ZMM1, ZMM2));
    assert_eq!(code, [0x62, 0xF2, 0xF5, 0x48, 0x40, 0xC2]);
}

// ─── AVX-512 with extended registers ──────────────────────────

#[test]
fn test_gen_vpandd_zmm16() {
    // vpandd zmm16, zmm17, zmm18 → 62 A1 75 40 DB C2
    let code = assemble(|a| a.vpandd(ZMM16, ZMM17, ZMM18));
    assert_eq!(code, [0x62, 0xA1, 0x75, 0x40, 0xDB, 0xC2]);
}

// ─── AVX float 3-op instructions ──────────────────────────────

#[test]
fn test_gen_vsubss() {
    // vsubss xmm0, xmm1, xmm2 → C5 F2 5C C2
    let code = assemble(|a| a.vsubss(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF2, 0x5C, 0xC2]);
}

#[test]
fn test_gen_vsubsd() {
    // vsubsd xmm0, xmm1, xmm2 → C5 F3 5C C2
    let code = assemble(|a| a.vsubsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x5C, 0xC2]);
}

#[test]
fn test_gen_vmulss() {
    // vmulss xmm0, xmm1, xmm2 → C5 F2 59 C2
    let code = assemble(|a| a.vmulss(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF2, 0x59, 0xC2]);
}

#[test]
fn test_gen_vmulsd() {
    // vmulsd xmm0, xmm1, xmm2 → C5 F3 59 C2
    let code = assemble(|a| a.vmulsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x59, 0xC2]);
}

#[test]
fn test_gen_vdivss() {
    // vdivss xmm0, xmm1, xmm2 → C5 F2 5E C2
    let code = assemble(|a| a.vdivss(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF2, 0x5E, 0xC2]);
}

#[test]
fn test_gen_vdivsd() {
    // vdivsd xmm0, xmm1, xmm2 → C5 F3 5E C2
    let code = assemble(|a| a.vdivsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x5E, 0xC2]);
}

#[test]
fn test_gen_vminss() {
    // vminss xmm0, xmm1, xmm2 → C5 F2 5D C2
    let code = assemble(|a| a.vminss(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF2, 0x5D, 0xC2]);
}

#[test]
fn test_gen_vmaxsd() {
    // vmaxsd xmm0, xmm1, xmm2 → C5 F3 5F C2
    let code = assemble(|a| a.vmaxsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x5F, 0xC2]);
}

// ─── AVX integer packed ops ───────────────────────────────────

#[test]
fn test_gen_vpaddw() {
    // vpaddw xmm0, xmm1, xmm2 → C5 F1 FD C2
    let code = assemble(|a| a.vpaddw(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0xFD, 0xC2]);
}

#[test]
fn test_gen_vpsubw() {
    // vpsubw xmm0, xmm1, xmm2 → C5 F1 F9 C2
    let code = assemble(|a| a.vpsubw(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0xF9, 0xC2]);
}

#[test]
fn test_gen_vpmullw() {
    // vpmullw xmm0, xmm1, xmm2 → C5 F1 D5 C2
    let code = assemble(|a| a.vpmullw(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0xD5, 0xC2]);
}

#[test]
fn test_gen_vpcmpeqb() {
    // vpcmpeqb xmm0, xmm1, xmm2 → C5 F1 74 C2
    let code = assemble(|a| a.vpcmpeqb(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0x74, 0xC2]);
}

#[test]
fn test_gen_vpminsd() {
    // vpminsd xmm0, xmm1, xmm2 → C4 E2 71 39 C2
    let code = assemble(|a| a.vpminsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x39, 0xC2]);
}

#[test]
fn test_gen_vpmaxsd() {
    // vpmaxsd xmm0, xmm1, xmm2 → C4 E2 71 3D C2
    let code = assemble(|a| a.vpmaxsd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x3D, 0xC2]);
}

// ─── AES instructions (generated) ────────────────────────────

#[test]
fn test_gen_aesenc() {
    // aesenc xmm0, xmm1 → 66 0F 38 DC C1
    let code = assemble(|a| a.aesenc(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0xDC, 0xC1]);
}

#[test]
fn test_gen_aesenclast() {
    // aesenclast xmm0, xmm1 → 66 0F 38 DD C1
    let code = assemble(|a| a.aesenclast(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0xDD, 0xC1]);
}

#[test]
fn test_gen_aesdec() {
    // aesdec xmm0, xmm1 → 66 0F 38 DE C1
    let code = assemble(|a| a.aesdec(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0xDE, 0xC1]);
}

#[test]
fn test_gen_aesdeclast() {
    // aesdeclast xmm0, xmm1 → 66 0F 38 DF C1
    let code = assemble(|a| a.aesdeclast(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0xDF, 0xC1]);
}

// ─── PCLMULQDQ instruction (generated) ───────────────────────

#[test]
fn test_gen_pclmulqdq() {
    // pclmulqdq xmm0, xmm1, 0 → 66 0F 3A 44 C1 00
    let code = assemble(|a| a.pclmulqdq(XMM0, XMM1, 0));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x44, 0xC1, 0x00]);
}

// ─── AVX AES 3-operand ───────────────────────────────────────

#[test]
fn test_gen_vaesenc() {
    // vaesenc xmm0, xmm1, xmm2 → C4 E2 71 DC C2
    let code = assemble(|a| a.vaesenc(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xDC, 0xC2]);
}

#[test]
fn test_gen_vaesenclast() {
    // vaesenclast xmm0, xmm1, xmm2 → C4 E2 71 DD C2
    let code = assemble(|a| a.vaesenclast(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xDD, 0xC2]);
}

#[test]
fn test_gen_vaesdec() {
    // vaesdec xmm0, xmm1, xmm2 → C4 E2 71 DE C2
    let code = assemble(|a| a.vaesdec(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xDE, 0xC2]);
}

#[test]
fn test_gen_vaesdeclast() {
    // vaesdeclast xmm0, xmm1, xmm2 → C4 E2 71 DF C2
    let code = assemble(|a| a.vaesdeclast(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0xDF, 0xC2]);
}

// ─── SSE4 instructions (generated) ────────────────────────────

#[test]
fn test_gen_ptest() {
    // ptest xmm0, xmm1 → 66 0F 38 17 C1
    let code = assemble(|a| a.ptest(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x17, 0xC1]);
}

#[test]
fn test_gen_pmovzxbw() {
    // pmovzxbw xmm0, xmm1 → 66 0F 38 30 C1
    let code = assemble(|a| a.pmovzxbw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x30, 0xC1]);
}

#[test]
fn test_gen_pmovsxbw() {
    // pmovsxbw xmm0, xmm1 → 66 0F 38 20 C1
    let code = assemble(|a| a.pmovsxbw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x20, 0xC1]);
}

// ─── SSE4 shuffle with imm8 ──────────────────────────────────

#[test]
fn test_gen_roundps() {
    // roundps xmm0, xmm1, 0 → 66 0F 3A 08 C1 00
    let code = assemble(|a| a.roundps(XMM0, XMM1, 0));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x08, 0xC1, 0x00]);
}

#[test]
fn test_gen_roundpd() {
    // roundpd xmm0, xmm1, 0 → 66 0F 3A 09 C1 00
    let code = assemble(|a| a.roundpd(XMM0, XMM1, 0));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x09, 0xC1, 0x00]);
}

#[test]
fn test_gen_pshufd() {
    // pshufd xmm0, xmm1, 0x1B → 66 0F 70 C1 1B
    let code = assemble(|a| a.pshufd(XMM0, XMM1, 0x1B));
    assert_eq!(code, [0x66, 0x0F, 0x70, 0xC1, 0x1B]);
}

#[test]
fn test_gen_pshufhw() {
    // pshufhw xmm0, xmm1, 0 → F3 0F 70 C1 00
    let code = assemble(|a| a.pshufhw(XMM0, XMM1, 0));
    assert_eq!(code, [0xF3, 0x0F, 0x70, 0xC1, 0x00]);
}

#[test]
fn test_gen_pshuflw() {
    // pshuflw xmm0, xmm1, 0 → F2 0F 70 C1 00
    let code = assemble(|a| a.pshuflw(XMM0, XMM1, 0));
    assert_eq!(code, [0xF2, 0x0F, 0x70, 0xC1, 0x00]);
}

// ─── AVX-512 specific: vpternlogd ─────────────────────────────

#[test]
fn test_gen_vpternlogd() {
    // vpternlogd zmm0, zmm1, zmm2, 0xFF → 62 F3 75 48 25 C2 FF
    let code = assemble(|a| a.vpternlogd(ZMM0, ZMM1, ZMM2, 0xFF));
    assert_eq!(code, [0x62, 0xF3, 0x75, 0x48, 0x25, 0xC2, 0xFF]);
}

#[test]
fn test_gen_vpternlogq() {
    // vpternlogq zmm0, zmm1, zmm2, 0xDB → 62 F3 F5 48 25 C2 DB
    let code = assemble(|a| a.vpternlogq(ZMM0, ZMM1, ZMM2, 0xDB));
    assert_eq!(code, [0x62, 0xF3, 0xF5, 0x48, 0x25, 0xC2, 0xDB]);
}

// ─── Newly added SSE2 packed integer instructions ────────────

#[test]
fn test_gen_paddb() {
    // paddb xmm0, xmm1 → 66 0F FC C1
    let code = assemble(|a| a.paddb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xFC, 0xC1]);
}

#[test]
fn test_gen_paddw() {
    // paddw xmm0, xmm1 → 66 0F FD C1
    let code = assemble(|a| a.paddw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xFD, 0xC1]);
}

#[test]
fn test_gen_paddq() {
    // paddq xmm0, xmm1 → 66 0F D4 C1
    let code = assemble(|a| a.paddq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD4, 0xC1]);
}

#[test]
fn test_gen_paddsb() {
    // paddsb xmm0, xmm1 → 66 0F EC C1
    let code = assemble(|a| a.paddsb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xEC, 0xC1]);
}

#[test]
fn test_gen_paddsw() {
    // paddsw xmm0, xmm1 → 66 0F ED C1
    let code = assemble(|a| a.paddsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xED, 0xC1]);
}

#[test]
fn test_gen_paddusb() {
    // paddusb xmm0, xmm1 → 66 0F DC C1
    let code = assemble(|a| a.paddusb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xDC, 0xC1]);
}

#[test]
fn test_gen_paddusw() {
    // paddusw xmm0, xmm1 → 66 0F DD C1
    let code = assemble(|a| a.paddusw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xDD, 0xC1]);
}

#[test]
fn test_gen_psubb() {
    // psubb xmm0, xmm1 → 66 0F F8 C1
    let code = assemble(|a| a.psubb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF8, 0xC1]);
}

#[test]
fn test_gen_psubw() {
    // psubw xmm0, xmm1 → 66 0F F9 C1
    let code = assemble(|a| a.psubw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF9, 0xC1]);
}

#[test]
fn test_gen_psubq() {
    // psubq xmm0, xmm1 → 66 0F FB C1
    let code = assemble(|a| a.psubq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xFB, 0xC1]);
}

#[test]
fn test_gen_psubsb() {
    // psubsb xmm0, xmm1 → 66 0F E8 C1
    let code = assemble(|a| a.psubsb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE8, 0xC1]);
}

#[test]
fn test_gen_psubsw() {
    // psubsw xmm0, xmm1 → 66 0F E9 C1
    let code = assemble(|a| a.psubsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE9, 0xC1]);
}

#[test]
fn test_gen_psubusb() {
    // psubusb xmm0, xmm1 → 66 0F D8 C1
    let code = assemble(|a| a.psubusb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD8, 0xC1]);
}

#[test]
fn test_gen_psubusw() {
    // psubusw xmm0, xmm1 → 66 0F D9 C1
    let code = assemble(|a| a.psubusw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD9, 0xC1]);
}

#[test]
fn test_gen_pmullw() {
    // pmullw xmm0, xmm1 → 66 0F D5 C1
    let code = assemble(|a| a.pmullw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD5, 0xC1]);
}

#[test]
fn test_gen_pmulhw() {
    // pmulhw xmm0, xmm1 → 66 0F E5 C1
    let code = assemble(|a| a.pmulhw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE5, 0xC1]);
}

#[test]
fn test_gen_pmulhuw() {
    // pmulhuw xmm0, xmm1 → 66 0F E4 C1
    let code = assemble(|a| a.pmulhuw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE4, 0xC1]);
}

#[test]
fn test_gen_pmuludq() {
    // pmuludq xmm0, xmm1 → 66 0F F4 C1
    let code = assemble(|a| a.pmuludq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF4, 0xC1]);
}

#[test]
fn test_gen_pmaddwd() {
    // pmaddwd xmm0, xmm1 → 66 0F F5 C1
    let code = assemble(|a| a.pmaddwd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF5, 0xC1]);
}

// ─── SSE2 packed compare ─────────────────────────────────────

#[test]
fn test_gen_pcmpeqb() {
    // pcmpeqb xmm0, xmm1 → 66 0F 74 C1
    let code = assemble(|a| a.pcmpeqb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x74, 0xC1]);
}

#[test]
fn test_gen_pcmpeqw() {
    // pcmpeqw xmm0, xmm1 → 66 0F 75 C1
    let code = assemble(|a| a.pcmpeqw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x75, 0xC1]);
}

#[test]
fn test_gen_pcmpeqd() {
    // pcmpeqd xmm0, xmm1 → 66 0F 76 C1
    let code = assemble(|a| a.pcmpeqd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x76, 0xC1]);
}

#[test]
fn test_gen_pcmpgtb() {
    // pcmpgtb xmm0, xmm1 → 66 0F 64 C1
    let code = assemble(|a| a.pcmpgtb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x64, 0xC1]);
}

#[test]
fn test_gen_pcmpgtw() {
    // pcmpgtw xmm0, xmm1 → 66 0F 65 C1
    let code = assemble(|a| a.pcmpgtw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x65, 0xC1]);
}

#[test]
fn test_gen_pcmpgtd() {
    // pcmpgtd xmm0, xmm1 → 66 0F 66 C1
    let code = assemble(|a| a.pcmpgtd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x66, 0xC1]);
}

// ─── SSE2 packed misc ────────────────────────────────────────

#[test]
fn test_gen_pandn() {
    // pandn xmm0, xmm1 → 66 0F DF C1
    let code = assemble(|a| a.pandn(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xDF, 0xC1]);
}

#[test]
fn test_gen_pavgb() {
    // pavgb xmm0, xmm1 → 66 0F E0 C1
    let code = assemble(|a| a.pavgb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE0, 0xC1]);
}

#[test]
fn test_gen_pavgw() {
    // pavgw xmm0, xmm1 → 66 0F E3 C1
    let code = assemble(|a| a.pavgw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE3, 0xC1]);
}

#[test]
fn test_gen_psadbw() {
    // psadbw xmm0, xmm1 → 66 0F F6 C1
    let code = assemble(|a| a.psadbw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF6, 0xC1]);
}

#[test]
fn test_gen_packsswb() {
    // packsswb xmm0, xmm1 → 66 0F 63 C1
    let code = assemble(|a| a.packsswb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x63, 0xC1]);
}

#[test]
fn test_gen_packssdw() {
    // packssdw xmm0, xmm1 → 66 0F 6B C1
    let code = assemble(|a| a.packssdw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x6B, 0xC1]);
}

#[test]
fn test_gen_packuswb() {
    // packuswb xmm0, xmm1 → 66 0F 67 C1
    let code = assemble(|a| a.packuswb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x67, 0xC1]);
}

// ─── SSE2 packed unpack ──────────────────────────────────────

#[test]
fn test_gen_punpckhbw() {
    // punpckhbw xmm0, xmm1 → 66 0F 68 C1
    let code = assemble(|a| a.punpckhbw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x68, 0xC1]);
}

#[test]
fn test_gen_punpckhwd() {
    // punpckhwd xmm0, xmm1 → 66 0F 69 C1
    let code = assemble(|a| a.punpckhwd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x69, 0xC1]);
}

#[test]
fn test_gen_punpckhdq() {
    // punpckhdq xmm0, xmm1 → 66 0F 6A C1
    let code = assemble(|a| a.punpckhdq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x6A, 0xC1]);
}

#[test]
fn test_gen_punpcklbw() {
    // punpcklbw xmm0, xmm1 → 66 0F 60 C1
    let code = assemble(|a| a.punpcklbw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x60, 0xC1]);
}

#[test]
fn test_gen_punpcklwd() {
    // punpcklwd xmm0, xmm1 → 66 0F 61 C1
    let code = assemble(|a| a.punpcklwd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x61, 0xC1]);
}

#[test]
fn test_gen_punpckldq() {
    // punpckldq xmm0, xmm1 → 66 0F 62 C1
    let code = assemble(|a| a.punpckldq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x62, 0xC1]);
}

// ─── SSE1 movhlps/movlhps ───────────────────────────────────

#[test]
fn test_gen_movhlps() {
    // movhlps xmm0, xmm1 → 0F 12 C1
    let code = assemble(|a| a.movhlps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x12, 0xC1]);
}

#[test]
fn test_gen_movlhps() {
    // movlhps xmm0, xmm1 → 0F 16 C1
    let code = assemble(|a| a.movlhps(XMM0, XMM1));
    assert_eq!(code, [0x0F, 0x16, 0xC1]);
}

// ─── SSSE3 packed instructions ───────────────────────────────

#[test]
fn test_gen_pshufb() {
    // pshufb xmm0, xmm1 → 66 0F 38 00 C1
    let code = assemble(|a| a.pshufb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x00, 0xC1]);
}

#[test]
fn test_gen_phaddw() {
    // phaddw xmm0, xmm1 → 66 0F 38 01 C1
    let code = assemble(|a| a.phaddw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x01, 0xC1]);
}

#[test]
fn test_gen_phaddd() {
    // phaddd xmm0, xmm1 → 66 0F 38 02 C1
    let code = assemble(|a| a.phaddd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x02, 0xC1]);
}

#[test]
fn test_gen_phaddsw() {
    // phaddsw xmm0, xmm1 → 66 0F 38 03 C1
    let code = assemble(|a| a.phaddsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x03, 0xC1]);
}

#[test]
fn test_gen_phsubw() {
    // phsubw xmm0, xmm1 → 66 0F 38 05 C1
    let code = assemble(|a| a.phsubw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x05, 0xC1]);
}

#[test]
fn test_gen_phsubd() {
    // phsubd xmm0, xmm1 → 66 0F 38 06 C1
    let code = assemble(|a| a.phsubd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x06, 0xC1]);
}

#[test]
fn test_gen_phsubsw() {
    // phsubsw xmm0, xmm1 → 66 0F 38 07 C1
    let code = assemble(|a| a.phsubsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x07, 0xC1]);
}

#[test]
fn test_gen_psignb() {
    // psignb xmm0, xmm1 → 66 0F 38 08 C1
    let code = assemble(|a| a.psignb(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x08, 0xC1]);
}

#[test]
fn test_gen_psignw() {
    // psignw xmm0, xmm1 → 66 0F 38 09 C1
    let code = assemble(|a| a.psignw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x09, 0xC1]);
}

#[test]
fn test_gen_psignd() {
    // psignd xmm0, xmm1 → 66 0F 38 0A C1
    let code = assemble(|a| a.psignd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x0A, 0xC1]);
}

#[test]
fn test_gen_pmaddubsw() {
    // pmaddubsw xmm0, xmm1 → 66 0F 38 04 C1
    let code = assemble(|a| a.pmaddubsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x04, 0xC1]);
}

#[test]
fn test_gen_pmulhrsw() {
    // pmulhrsw xmm0, xmm1 → 66 0F 38 0B C1
    let code = assemble(|a| a.pmulhrsw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x38, 0x0B, 0xC1]);
}

#[test]
fn test_gen_palignr() {
    // palignr xmm0, xmm1, 4 → 66 0F 3A 0F C1 04
    let code = assemble(|a| a.palignr(XMM0, XMM1, 4));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x04]);
}

// ─── SSE2 shift by xmm ──────────────────────────────────────

#[test]
fn test_gen_psllw() {
    // psllw xmm0, xmm1 → 66 0F F1 C1
    let code = assemble(|a| a.psllw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF1, 0xC1]);
}

#[test]
fn test_gen_pslld() {
    // pslld xmm0, xmm1 → 66 0F F2 C1
    let code = assemble(|a| a.pslld(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF2, 0xC1]);
}

#[test]
fn test_gen_psllq() {
    // psllq xmm0, xmm1 → 66 0F F3 C1
    let code = assemble(|a| a.psllq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xF3, 0xC1]);
}

#[test]
fn test_gen_psrlw() {
    // psrlw xmm0, xmm1 → 66 0F D1 C1
    let code = assemble(|a| a.psrlw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD1, 0xC1]);
}

#[test]
fn test_gen_psrld() {
    // psrld xmm0, xmm1 → 66 0F D2 C1
    let code = assemble(|a| a.psrld(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD2, 0xC1]);
}

#[test]
fn test_gen_psrlq() {
    // psrlq xmm0, xmm1 → 66 0F D3 C1
    let code = assemble(|a| a.psrlq(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD3, 0xC1]);
}

#[test]
fn test_gen_psraw() {
    // psraw xmm0, xmm1 → 66 0F E1 C1
    let code = assemble(|a| a.psraw(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE1, 0xC1]);
}

#[test]
fn test_gen_psrad() {
    // psrad xmm0, xmm1 → 66 0F E2 C1
    let code = assemble(|a| a.psrad(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xE2, 0xC1]);
}

// ─── REX prefix tests with high registers ────────────────────

#[test]
fn test_gen_paddb_rex() {
    // paddb xmm8, xmm9 → 66 45 0F FC C1
    let code = assemble(|a| a.paddb(XMM8, XMM9));
    assert_eq!(code, [0x66, 0x45, 0x0F, 0xFC, 0xC1]);
}

#[test]
fn test_gen_pcmpeqb_rex() {
    // pcmpeqb xmm10, xmm11 → 66 45 0F 74 D3
    let code = assemble(|a| a.pcmpeqb(XMM10, XMM11));
    assert_eq!(code, [0x66, 0x45, 0x0F, 0x74, 0xD3]);
}

#[test]
fn test_gen_pshufb_rex() {
    // pshufb xmm8, xmm0 → 66 44 0F 38 00 C0
    let code = assemble(|a| a.pshufb(XMM8, XMM0));
    assert_eq!(code, [0x66, 0x44, 0x0F, 0x38, 0x00, 0xC0]);
}

// ─── SSE3 float instructions ─────────────────────────────────

#[test]
fn test_gen_addsubps() {
    // addsubps xmm0, xmm1 → F2 0F D0 C1
    let code = assemble(|a| a.addsubps(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0xD0, 0xC1]);
}

#[test]
fn test_gen_addsubpd() {
    // addsubpd xmm0, xmm1 → 66 0F D0 C1
    let code = assemble(|a| a.addsubpd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0xD0, 0xC1]);
}

#[test]
fn test_gen_haddps() {
    // haddps xmm0, xmm1 → F2 0F 7C C1
    let code = assemble(|a| a.haddps(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0x7C, 0xC1]);
}

#[test]
fn test_gen_haddpd() {
    // haddpd xmm0, xmm1 → 66 0F 7C C1
    let code = assemble(|a| a.haddpd(XMM0, XMM1));
    assert_eq!(code, [0x66, 0x0F, 0x7C, 0xC1]);
}

#[test]
fn test_gen_hsubps() {
    // hsubps xmm0, xmm1 → F2 0F 7D C1
    let code = assemble(|a| a.hsubps(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0x7D, 0xC1]);
}

#[test]
fn test_gen_movshdup() {
    // movshdup xmm0, xmm1 → F3 0F 16 C1
    let code = assemble(|a| a.movshdup(XMM0, XMM1));
    assert_eq!(code, [0xF3, 0x0F, 0x16, 0xC1]);
}

#[test]
fn test_gen_movsldup() {
    // movsldup xmm0, xmm1 → F3 0F 12 C1
    let code = assemble(|a| a.movsldup(XMM0, XMM1));
    assert_eq!(code, [0xF3, 0x0F, 0x12, 0xC1]);
}

#[test]
fn test_gen_movddup() {
    // movddup xmm0, xmm1 → F2 0F 12 C1
    let code = assemble(|a| a.movddup(XMM0, XMM1));
    assert_eq!(code, [0xF2, 0x0F, 0x12, 0xC1]);
}

// ─── More SSE4 tests ─────────────────────────────────────────

#[test]
fn test_gen_blendpd() {
    // blendpd xmm0, xmm1, 3 → 66 0F 3A 0D C1 03
    let code = assemble(|a| a.blendpd(XMM0, XMM1, 3));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x0D, 0xC1, 0x03]);
}

#[test]
fn test_gen_blendps() {
    // blendps xmm0, xmm1, 5 → 66 0F 3A 0C C1 05
    let code = assemble(|a| a.blendps(XMM0, XMM1, 5));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x0C, 0xC1, 0x05]);
}

#[test]
fn test_gen_insertps() {
    // insertps xmm0, xmm1, 0x10 → 66 0F 3A 21 C1 10
    let code = assemble(|a| a.insertps(XMM0, XMM1, 0x10));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x21, 0xC1, 0x10]);
}

#[test]
fn test_gen_roundsd() {
    // roundsd xmm0, xmm1, 1 → 66 0F 3A 0B C1 01
    let code = assemble(|a| a.roundsd(XMM0, XMM1, 1));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x0B, 0xC1, 0x01]);
}

#[test]
fn test_gen_roundss() {
    // roundss xmm0, xmm1, 2 → 66 0F 3A 0A C1 02
    let code = assemble(|a| a.roundss(XMM0, XMM1, 2));
    assert_eq!(code, [0x66, 0x0F, 0x3A, 0x0A, 0xC1, 0x02]);
}

// ─── AVX broadcast and convert tests ─────────────────────────

#[test]
fn test_gen_vbroadcastss() {
    // vbroadcastss ymm0, xmm1 → C4 E2 7D 18 C1
    let code = assemble(|a| a.vbroadcastss(YMM0, XMM1));
    assert_eq!(code, [0xC4, 0xE2, 0x7D, 0x18, 0xC1]);
}

#[test]
fn test_gen_vpbroadcastd() {
    // vpbroadcastd ymm0, xmm1 → C4 E2 7D 58 C1
    let code = assemble(|a| a.vpbroadcastd(YMM0, XMM1));
    assert_eq!(code, [0xC4, 0xE2, 0x7D, 0x58, 0xC1]);
}

#[test]
fn test_gen_vcvttps2dq() {
    // vcvttps2dq xmm0, xmm1 → C5 FA 5B C1
    let code = assemble(|a| a.vcvttps2dq(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xFA, 0x5B, 0xC1]);
}

#[test]
fn test_gen_vcvtps2dq_xmm() {
    // vcvtps2dq xmm0, xmm1 → C5 F9 5B C1
    let code = assemble(|a| a.vcvtps2dq(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF9, 0x5B, 0xC1]);
}

// ─── AVX compare 2-op ────────────────────────────────────────

#[test]
fn test_gen_vcomiss() {
    // vcomiss xmm0, xmm1 → C5 F8 2F C1
    let code = assemble(|a| a.vcomiss(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x2F, 0xC1]);
}

#[test]
fn test_gen_vcomisd() {
    // vcomisd xmm0, xmm1 → C5 F9 2F C1
    let code = assemble(|a| a.vcomisd(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF9, 0x2F, 0xC1]);
}

#[test]
fn test_gen_vucomiss() {
    // vucomiss xmm0, xmm1 → C5 F8 2E C1
    let code = assemble(|a| a.vucomiss(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF8, 0x2E, 0xC1]);
}

#[test]
fn test_gen_vucomisd() {
    // vucomisd xmm0, xmm1 → C5 F9 2E C1
    let code = assemble(|a| a.vucomisd(XMM0, XMM1));
    assert_eq!(code, [0xC5, 0xF9, 0x2E, 0xC1]);
}

// ─── AVX3 horizontal ops ─────────────────────────────────────

#[test]
fn test_gen_vhaddps() {
    // vhaddps xmm0, xmm1, xmm2 → C5 F3 7C C2
    let code = assemble(|a| a.vhaddps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x7C, 0xC2]);
}

#[test]
fn test_gen_vhaddpd() {
    // vhaddpd xmm0, xmm1, xmm2 → C5 F1 7C C2
    let code = assemble(|a| a.vhaddpd(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0x7C, 0xC2]);
}

#[test]
fn test_gen_vhsubps() {
    // vhsubps xmm0, xmm1, xmm2 → C5 F3 7D C2
    let code = assemble(|a| a.vhsubps(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF3, 0x7D, 0xC2]);
}

// ─── AVX pack/unpack ─────────────────────────────────────────

#[test]
fn test_gen_vpacksswb() {
    // vpacksswb xmm0, xmm1, xmm2 → C5 F1 63 C2
    let code = assemble(|a| a.vpacksswb(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0x63, 0xC2]);
}

#[test]
fn test_gen_vpunpckhbw() {
    // vpunpckhbw xmm0, xmm1, xmm2 → C5 F1 68 C2
    let code = assemble(|a| a.vpunpckhbw(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0x68, 0xC2]);
}

#[test]
fn test_gen_vpunpcklbw() {
    // vpunpcklbw xmm0, xmm1, xmm2 → C5 F1 60 C2
    let code = assemble(|a| a.vpunpcklbw(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC5, 0xF1, 0x60, 0xC2]);
}

// ─── AVX shuffle/sign ────────────────────────────────────────

#[test]
fn test_gen_vpshufb() {
    // vpshufb xmm0, xmm1, xmm2 → C4 E2 71 00 C2
    let code = assemble(|a| a.vpshufb(XMM0, XMM1, XMM2));
    assert_eq!(code, [0xC4, 0xE2, 0x71, 0x00, 0xC2]);
}

#[test]
fn test_gen_vpshufd() {
    // vpshufd xmm0, xmm1, 0x1B → C5 F9 70 C1 1B
    let code = assemble(|a| a.vpshufd(XMM0, XMM1, 0x1B));
    assert_eq!(code, [0xC5, 0xF9, 0x70, 0xC1, 0x1B]);
}

#[test]
fn test_gen_vpalignr() {
    // vpalignr xmm0, xmm1, xmm2, 4 → C4 E3 71 0F C2 04
    let code = assemble(|a| a.vpalignr(XMM0, XMM1, XMM2, 4));
    assert_eq!(code, [0xC4, 0xE3, 0x71, 0x0F, 0xC2, 0x04]);
}
