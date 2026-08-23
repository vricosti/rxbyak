use rxbyak::*;

fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

#[test]
fn test_bsr_register_matches_xbyak_7_40() {
    assert!(BSR0.is_bsr());
    assert_eq!(BSR0.get_idx(), 0);
    assert_eq!(BSR0.get_bit(), 1024);
}

#[test]
fn test_bsr_instructions_match_xbyak_7_40() {
    let code = assemble(|a| {
        a.bsrinit(BSR0)?;
        a.bsrmovf(BSR0, ZMM1, ZMM2)?;
        a.bsrmovh_load(BSR0, ZMM3)?;
        a.bsrmovh_store(ZMM4, BSR0)?;
        a.bsrmovh_store(ptr(RAX + 64), BSR0)?;
        a.bsrmovl_load(BSR0, ZMM5)?;
        a.bsrmovl_store(ZMM6, BSR0)?;
        a.bsrmovl_store(ptr(R13 + 128), BSR0)
    });

    assert_eq!(
        code,
        [
            0xC4, 0xE2, 0xFB, 0x49, 0xC0, 0x62, 0xF6, 0xF4, 0x48, 0x95, 0xC2, 0x62, 0xF6, 0xFF,
            0x48, 0x95, 0xC3, 0x62, 0xF6, 0x7F, 0x48, 0x95, 0xC4, 0x62, 0xF6, 0x7F, 0x48, 0x95,
            0x40, 0x40, 0x62, 0xF6, 0xFE, 0x48, 0x95, 0xC5, 0x62, 0xF6, 0x7E, 0x48, 0x95, 0xC6,
            0x62, 0xD6, 0x7E, 0x48, 0x95, 0x85, 0x80, 0x00, 0x00, 0x00,
        ]
    );
}

#[test]
fn test_bsr_instructions_reject_other_register_kinds() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.bsrinit(ZMM0), Err(Error::BadCombination));
    assert_eq!(asm.bsrmovf(BSR0, XMM0, ZMM1), Err(Error::BadCombination));
    assert_eq!(asm.bsrmovh_load(ZMM0, ZMM1), Err(Error::BadCombination));
}

#[test]
fn test_ace_amx_instructions_match_xbyak_7_40() {
    let code = assemble(|a| {
        a.tilemovcol_reg(TMM1, ZMM2, EAX)?;
        a.tilemovcol_imm(TMM3, ZMM4, 0x5A)?;
        a.top2bf16ps(TMM1, ZMM2, ZMM3)?;
        a.top4bssd(TMM1, ZMM2, ZMM3)?;
        a.top4bsud(TMM1, ZMM2, ZMM3)?;
        a.top4busd(TMM1, ZMM2, ZMM3)?;
        a.top4buud(TMM1, ZMM2, ZMM3)?;
        a.top4mxbf8ps(TMM1, ZMM2, ZMM3, 1)?;
        a.top4mxbhf8ps(TMM1, ZMM2, ZMM3, 2)?;
        a.top4mxbssps(TMM1, ZMM2, ZMM3, 3)?;
        a.top4mxhbf8ps(TMM1, ZMM2, ZMM3, 4)?;
        a.top4mxhf8ps(TMM1, ZMM2, ZMM3, 5)
    });

    assert_eq!(
        code,
        [
            0x62, 0xF2, 0xFD, 0x48, 0x4B, 0xCA, 0x62, 0xF3, 0xFD, 0x48, 0x2F, 0xDC, 0x5A, 0x62,
            0xF2, 0x66, 0x48, 0x5C, 0xCA, 0x62, 0xF2, 0x67, 0x48, 0x5E, 0xCA, 0x62, 0xF2, 0x66,
            0x48, 0x5E, 0xCA, 0x62, 0xF2, 0x65, 0x48, 0x5E, 0xCA, 0x62, 0xF2, 0x64, 0x48, 0x5E,
            0xCA, 0x62, 0xF3, 0x64, 0x48, 0x8D, 0xCA, 0x01, 0x62, 0xF3, 0x67, 0x48, 0x8D, 0xCA,
            0x02, 0x62, 0xF3, 0x67, 0x48, 0x8F, 0xCA, 0x03, 0x62, 0xF3, 0x66, 0x48, 0x8D, 0xCA,
            0x04, 0x62, 0xF3, 0x65, 0x48, 0x8D, 0xCA, 0x05,
        ]
    );
}

#[test]
fn test_ace_amx_instructions_validate_register_classes() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.tilemovcol_reg(ZMM0, ZMM1, EAX),
        Err(Error::BadCombination)
    );
    assert_eq!(
        asm.tilemovcol_reg(TMM0, ZMM1, RAX),
        Err(Error::BadCombination)
    );
    assert_eq!(asm.top2bf16ps(TMM0, XMM1, ZMM2), Err(Error::BadCombination));
}

#[test]
fn test_ace_vector_conversions_match_xbyak_7_40() {
    let code = assemble(|a| {
        a.vcvtbf42hf8(XMM1, XMM2)?;
        a.vcvtbf62hf8(YMM1, YMM2)?;
        a.vcvtbf82bf4s(XMM1, XMM2)?;
        a.vcvtbf82bf6s(YMM1, YMM2)?;
        a.vcvtbf82ps(ZMM1, XMM2)?;
        a.vcvtbiasps2bf8(XMM1, ZMM2, ZMM3)?;
        a.vcvtbiasps2bf8s(XMM1, YMM2, YMM3)?;
        a.vcvtbiasps2hf8(XMM1, XMM2, XMM3)?;
        a.vcvtbiasps2hf8s(XMM1, ZMM2, ZMM3)?;
        a.vcvthf62hf8(ZMM1, ZMM2)?;
        a.vcvthf82bf4s(YMM1, ZMM2)?;
        a.vcvthf82hf6s(XMM1, XMM2)?;
        a.vcvthf82ps(YMM1, XMM2)?;
        a.vcvtps2bf8(XMM1, XMM2)?;
        a.vcvtps2bf8s(XMM1, YMM2)?;
        a.vcvtps2hf8(XMM1, ZMM2)?;
        a.vcvtps2hf8s(XMM1, XMM2)?;
        a.vcvtrops2hf8(XMM1, YMM2)?;
        a.vcvtrops2hf8s(XMM1, ZMM2)?;
        a.vcvtbf42hf8(ZMM4, YMM5)?;
        a.vcvtps2bf8(XMM6, zmmword_ptr(RAX + 64))
    });

    assert_eq!(
        code,
        [
            0x62, 0xF5, 0x7C, 0x08, 0x37, 0xCA, 0x62, 0xF5, 0xFD, 0x28, 0x37, 0xCA, 0x62, 0xF5,
            0xFE, 0x08, 0x3D, 0xD1, 0x62, 0xF5, 0xFE, 0x28, 0x3E, 0xD1, 0x62, 0xF5, 0xFC, 0x48,
            0x36, 0xCA, 0x62, 0xF5, 0x6C, 0x48, 0x39, 0xCB, 0x62, 0xF5, 0x6C, 0x28, 0x3B, 0xCB,
            0x62, 0xF5, 0x6C, 0x08, 0x38, 0xCB, 0x62, 0xF5, 0x6C, 0x48, 0x3A, 0xCB, 0x62, 0xF5,
            0x7D, 0x48, 0x37, 0xCA, 0x62, 0xF5, 0x7E, 0x48, 0x3D, 0xD1, 0x62, 0xF5, 0x7E, 0x08,
            0x3C, 0xD1, 0x62, 0xF5, 0x7C, 0x28, 0x36, 0xCA, 0x62, 0xF5, 0x7E, 0x08, 0x39, 0xCA,
            0x62, 0xF5, 0x7E, 0x28, 0x3B, 0xCA, 0x62, 0xF5, 0x7E, 0x48, 0x38, 0xCA, 0x62, 0xF5,
            0x7E, 0x08, 0x3A, 0xCA, 0x62, 0xF5, 0x7D, 0x28, 0x38, 0xCA, 0x62, 0xF5, 0x7D, 0x48,
            0x3A, 0xCA, 0x62, 0xF5, 0x7C, 0x48, 0x37, 0xE5, 0x62, 0xF5, 0x7E, 0x48, 0x39, 0x70,
            0x01,
        ]
    );
}

#[test]
fn test_ace_vector_conversions_validate_width_contracts() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.vcvtbf42hf8(XMM0, YMM1), Err(Error::BadCombination));
    assert_eq!(
        asm.vcvtbiasps2bf8(YMM0, YMM1, YMM2),
        Err(Error::BadCombination)
    );
    assert_eq!(
        asm.vcvtbiasps2bf8(XMM0, YMM1, ZMM2),
        Err(Error::BadCombination)
    );
    assert_eq!(asm.vcvtps2bf8(YMM0, ZMM1), Err(Error::BadCombination));
}

#[test]
fn test_remaining_ace_instructions_match_xbyak_7_40() {
    let code = assemble(|a| {
        a.vpmovssdb(XMM1, ZMM2)?;
        a.vpmovssdb(dword_ptr(RAX + 64), ZMM3)?;
        a.vunpackb(XMM1, XMM2, 0x12)?;
        a.vunpackb(YMM3, YMM4, 0x34)?;
        a.vunpackb(ZMM5, ZMM6, 0x56)
    });
    assert_eq!(
        code,
        [
            0x62, 0xF2, 0x7E, 0x48, 0x41, 0xD1, 0x62, 0xF2, 0x7E, 0x48, 0x41, 0x58, 0x04, 0x62,
            0xF3, 0x7C, 0x08, 0x3D, 0xCA, 0x12, 0x62, 0xF3, 0x7C, 0x28, 0x3D, 0xDC, 0x34, 0x62,
            0xF3, 0x7C, 0x48, 0x3D, 0xEE, 0x56,
        ]
    );
}

#[test]
fn test_xbyak_7_35_4_fp8_disp8_scaling() {
    let code = assemble(|a| {
        a.vcvt2ph2bf8(XMM1, XMM2, xmmword_ptr(RAX + 64))?;
        a.vcvt2ph2bf8s(YMM3, YMM4, ymmword_ptr(RAX + 64))?;
        a.vcvt2ph2hf8(ZMM5, ZMM6, zmmword_ptr(RAX + 64))?;
        a.vcvt2ph2hf8s(XMM7, XMM8, xmmword_ptr(RAX + 64))?;
        a.vcvthf82ph(XMM9, xmmword_ptr(RAX + 64))?;
        a.vcvthf82ph(YMM10, xmmword_ptr(RAX + 64))?;
        a.vcvthf82ph(ZMM11, ymmword_ptr(RAX + 64))
    });
    assert_eq!(
        code,
        [
            0x62, 0xF2, 0x6F, 0x08, 0x74, 0x48, 0x04, 0x62, 0xF5, 0x5F, 0x28, 0x74, 0x58, 0x02,
            0x62, 0xF5, 0x4F, 0x48, 0x18, 0x68, 0x01, 0x62, 0xF5, 0x3F, 0x08, 0x1B, 0x78, 0x04,
            0x62, 0x75, 0x7F, 0x08, 0x1E, 0x48, 0x08, 0x62, 0x75, 0x7F, 0x28, 0x1E, 0x50, 0x04,
            0x62, 0x75, 0x7F, 0x48, 0x1E, 0x58, 0x02,
        ]
    );
}
