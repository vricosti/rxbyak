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
