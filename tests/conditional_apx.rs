use rxbyak::*;

fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

fn decode_hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn ccmp_register_memory_and_immediate_forms_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.ccmpb(RAX, RBX, 0)), "62f4840239d8"),
        (assemble(|a| a.ccmpb(R30B, R31B, 1)), "624c0c0238fe"),
        (
            assemble(|a| a.ccmpb(ptr(R30.into()), R31, 8)),
            "624cc402393e",
        ),
        (assemble(|a| a.ccmpb_imm(R20B, 0x12, 9)), "627c4c0280fc12"),
        (
            assemble(|a| a.ccmpb_imm(R20W, 0x1234, 9)),
            "627c4d0281fc3412",
        ),
        (
            assemble(|a| a.ccmpb_imm(R20D, 0x1234_5678, 9)),
            "627c4c0281fc78563412",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.ccmpb(RAX, RBX, -1), Err(Error::InvalidDfv));
    assert_eq!(asm.ccmpb(RAX, RBX, 16), Err(Error::InvalidDfv));
    assert_eq!(
        asm.ccmpb_imm(ptr(RAX.into()), 1, 0),
        Err(Error::MemSizeIsNotSpecified)
    );
}

#[test]
fn ctest_register_memory_and_immediate_forms_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.ctestb(R30B, R31B, 0)), "624c040284fe"),
        (
            assemble(|a| a.ctestb(ptr(R30.into()), R31, 7)),
            "624cbc02853e",
        ),
        (assemble(|a| a.ctestb_imm(R30B, 0x12, 8)), "62dc4402f6c612"),
        (
            assemble(|a| a.ctestb_imm(R30, 0x1234_5678, 11)),
            "62dcdc02f7c678563412",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.ctestb_imm(ptr(RAX.into()), 1, 0),
        Err(Error::MemSizeIsNotSpecified)
    );
}

#[test]
fn every_ccmp_and_ctest_spelling_uses_the_xbyak_selector() {
    macro_rules! assert_ccmp {
        ($name:ident, $imm_name:ident, $sc:expr) => {{
            let reg = assemble(|a| a.$name(RAX, RCX, 0));
            let imm = assemble(|a| a.$imm_name(EAX, 1, 0));
            assert_eq!(reg[3] & 0x0F, $sc);
            assert_eq!(imm[3] & 0x0F, $sc);
        }};
    }
    macro_rules! assert_ctest {
        ($name:ident, $imm_name:ident, $sc:expr) => {{
            let reg = assemble(|a| a.$name(RAX, RCX, 0));
            let imm = assemble(|a| a.$imm_name(EAX, 1, 0));
            assert_eq!(reg[3] & 0x0F, $sc);
            assert_eq!(imm[3] & 0x0F, $sc);
        }};
    }

    assert_ccmp!(ccmpa, ccmpa_imm, 7);
    assert_ccmp!(ccmpae, ccmpae_imm, 3);
    assert_ccmp!(ccmpb, ccmpb_imm, 2);
    assert_ccmp!(ccmpbe, ccmpbe_imm, 6);
    assert_ccmp!(ccmpc, ccmpc_imm, 2);
    assert_ccmp!(ccmpe, ccmpe_imm, 4);
    assert_ccmp!(ccmpf, ccmpf_imm, 11);
    assert_ccmp!(ccmpg, ccmpg_imm, 15);
    assert_ccmp!(ccmpge, ccmpge_imm, 13);
    assert_ccmp!(ccmpl, ccmpl_imm, 12);
    assert_ccmp!(ccmple, ccmple_imm, 14);
    assert_ccmp!(ccmpna, ccmpna_imm, 6);
    assert_ccmp!(ccmpnae, ccmpnae_imm, 2);
    assert_ccmp!(ccmpnb, ccmpnb_imm, 3);
    assert_ccmp!(ccmpnbe, ccmpnbe_imm, 7);
    assert_ccmp!(ccmpnc, ccmpnc_imm, 3);
    assert_ccmp!(ccmpne, ccmpne_imm, 5);
    assert_ccmp!(ccmpng, ccmpng_imm, 14);
    assert_ccmp!(ccmpnge, ccmpnge_imm, 12);
    assert_ccmp!(ccmpnl, ccmpnl_imm, 13);
    assert_ccmp!(ccmpnle, ccmpnle_imm, 15);
    assert_ccmp!(ccmpno, ccmpno_imm, 1);
    assert_ccmp!(ccmpns, ccmpns_imm, 9);
    assert_ccmp!(ccmpnz, ccmpnz_imm, 5);
    assert_ccmp!(ccmpo, ccmpo_imm, 0);
    assert_ccmp!(ccmps, ccmps_imm, 8);
    assert_ccmp!(ccmpt, ccmpt_imm, 10);
    assert_ccmp!(ccmpz, ccmpz_imm, 4);

    assert_ctest!(ctesta, ctesta_imm, 7);
    assert_ctest!(ctestae, ctestae_imm, 3);
    assert_ctest!(ctestb, ctestb_imm, 2);
    assert_ctest!(ctestbe, ctestbe_imm, 6);
    assert_ctest!(ctestc, ctestc_imm, 2);
    assert_ctest!(cteste, cteste_imm, 4);
    assert_ctest!(ctestf, ctestf_imm, 11);
    assert_ctest!(ctestg, ctestg_imm, 15);
    assert_ctest!(ctestge, ctestge_imm, 13);
    assert_ctest!(ctestl, ctestl_imm, 12);
    assert_ctest!(ctestle, ctestle_imm, 14);
    assert_ctest!(ctestna, ctestna_imm, 6);
    assert_ctest!(ctestnae, ctestnae_imm, 2);
    assert_ctest!(ctestnb, ctestnb_imm, 3);
    assert_ctest!(ctestnbe, ctestnbe_imm, 7);
    assert_ctest!(ctestnc, ctestnc_imm, 3);
    assert_ctest!(ctestne, ctestne_imm, 5);
    assert_ctest!(ctestng, ctestng_imm, 14);
    assert_ctest!(ctestnge, ctestnge_imm, 12);
    assert_ctest!(ctestnl, ctestnl_imm, 13);
    assert_ctest!(ctestnle, ctestnle_imm, 15);
    assert_ctest!(ctestno, ctestno_imm, 1);
    assert_ctest!(ctestns, ctestns_imm, 9);
    assert_ctest!(ctestnz, ctestnz_imm, 5);
    assert_ctest!(ctesto, ctesto_imm, 0);
    assert_ctest!(ctests, ctests_imm, 8);
    assert_ctest!(ctestt, ctestt_imm, 10);
    assert_ctest!(ctestz, ctestz_imm, 4);
}

#[test]
fn cfcmov_forms_and_all_condition_opcodes_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.cfcmovb(R30W, R31W)), "624c7d0c42fe"),
        (
            assemble(|a| a.cfcmovb(ptr(R8 + R20 * 4 + 3), R19D)),
            "62c4780c425ca003",
        ),
        (assemble(|a| a.cfcmovb3(R20, R30, R31)), "624cdc1442f7"),
        (
            assemble(|a| a.cfcmovb3(R20, R30, ptr(R9.into()))),
            "6244dc144231",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    macro_rules! assert_opcode {
        ($name:ident, $name3:ident, $opcode:expr) => {{
            assert_eq!(assemble(|a| a.$name(R20, R21))[4], $opcode);
            assert_eq!(assemble(|a| a.$name3(R20, R21, R22))[4], $opcode);
        }};
    }
    assert_opcode!(cfcmovo, cfcmovo3, 0x40);
    assert_opcode!(cfcmovno, cfcmovno3, 0x41);
    assert_opcode!(cfcmovb, cfcmovb3, 0x42);
    assert_opcode!(cfcmovnb, cfcmovnb3, 0x43);
    assert_opcode!(cfcmovz, cfcmovz3, 0x44);
    assert_opcode!(cfcmovnz, cfcmovnz3, 0x45);
    assert_opcode!(cfcmovbe, cfcmovbe3, 0x46);
    assert_opcode!(cfcmovnbe, cfcmovnbe3, 0x47);
    assert_opcode!(cfcmovs, cfcmovs3, 0x48);
    assert_opcode!(cfcmovns, cfcmovns3, 0x49);
    assert_opcode!(cfcmovp, cfcmovp3, 0x4A);
    assert_opcode!(cfcmovnp, cfcmovnp3, 0x4B);
    assert_opcode!(cfcmovl, cfcmovl3, 0x4C);
    assert_opcode!(cfcmovnl, cfcmovnl3, 0x4D);
    assert_opcode!(cfcmovle, cfcmovle3, 0x4E);
    assert_opcode!(cfcmovnle, cfcmovnle3, 0x4F);

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.cfcmovb(R20B, R21B), Err(Error::BadSizeOfRegister));
    assert_eq!(asm.cfcmovb3(R20D, R21, R22), Err(Error::BadSizeOfRegister));
}

#[test]
fn every_cmpccxadd_opcode_matches_xbyak_7_40() {
    let addr = ptr(R20 + R30 * 8);
    assert_eq!(
        assemble(|a| a.cmpbexadd(addr, R21, R22)),
        decode_hex("62aac900e62cf4")
    );

    macro_rules! assert_opcode {
        ($name:ident, $opcode:expr) => {
            assert_eq!(assemble(|a| a.$name(addr, R21, R22))[4], $opcode)
        };
    }
    assert_opcode!(cmpoxadd, 0xE0);
    assert_opcode!(cmpnoxadd, 0xE1);
    assert_opcode!(cmpbxadd, 0xE2);
    assert_opcode!(cmpnbxadd, 0xE3);
    assert_opcode!(cmpzxadd, 0xE4);
    assert_opcode!(cmpnzxadd, 0xE5);
    assert_opcode!(cmpbexadd, 0xE6);
    assert_opcode!(cmpnbexadd, 0xE7);
    assert_opcode!(cmpsxadd, 0xE8);
    assert_opcode!(cmpnsxadd, 0xE9);
    assert_opcode!(cmppxadd, 0xEA);
    assert_opcode!(cmpnpxadd, 0xEB);
    assert_opcode!(cmplxadd, 0xEC);
    assert_opcode!(cmpnlxadd, 0xED);
    assert_opcode!(cmplexadd, 0xEE);
    assert_opcode!(cmpnlexadd, 0xEF);
}
