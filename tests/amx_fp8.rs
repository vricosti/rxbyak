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
fn amx_load_rs_uses_xbyak_vex_and_apx_paths() {
    let cases = [
        (
            assemble(|a| a.tileloaddrs(TMM3, ptr(RDI + RDX * 2 + 8))),
            "c4e27b4a5c5708",
        ),
        (
            assemble(|a| a.tileloaddrs(TMM7, ptr(R31 + RDX * 2 + 8))),
            "62da7f084a7c5708",
        ),
        (
            assemble(|a| a.tileloaddrst1(TMM4, ptr(R8 + R9 + 32))),
            "c482794a640820",
        ),
        (
            assemble(|a| a.tileloaddrst1(TMM4, ptr(R25 + R9 + 32))),
            "629a7d084a640920",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.tileloadd(TMM1, ptr(RAX.into())),
        Err(Error::NotSupported)
    );
}

#[test]
fn amx_fp8_and_complex_dot_products_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.tdpbf8ps(TMM1, TMM2, TMM3)), "c4e560fdca"),
        (assemble(|a| a.tdpbhf8ps(TMM1, TMM2, TMM3)), "c4e563fdca"),
        (assemble(|a| a.tdphbf8ps(TMM1, TMM2, TMM3)), "c4e562fdca"),
        (assemble(|a| a.tdphf8ps(TMM1, TMM2, TMM3)), "c4e561fdca"),
        (assemble(|a| a.tcmmimfp16ps(TMM1, TMM2, TMM3)), "c4e2616cca"),
        (assemble(|a| a.tcmmrlfp16ps(TMM1, TMM2, TMM3)), "c4e2606cca"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.tdpbf8ps(ZMM1, TMM2, TMM3), Err(Error::BadCombination));
}

#[test]
fn amx_row_conversions_match_every_xbyak_overload() {
    let cases = [
        (
            assemble(|a| a.tcvtrowd2ps(ZMM20, TMM1, R30D)),
            "62e20e404ae1",
        ),
        (
            assemble(|a| a.tcvtrowd2ps_imm(ZMM20, TMM1, 0x12)),
            "62e37e4807e112",
        ),
        (
            assemble(|a| a.tcvtrowps2bf16h(ZMM1, TMM2, R30D)),
            "62f20f406dca",
        ),
        (
            assemble(|a| a.tcvtrowps2bf16h_imm(ZMM29, TMM2, 0x12)),
            "62637f4807ea12",
        ),
        (
            assemble(|a| a.tcvtrowps2bf16l(ZMM1, TMM2, R30D)),
            "62f20e406dca",
        ),
        (
            assemble(|a| a.tcvtrowps2bf16l_imm(ZMM29, TMM2, 0x12)),
            "62637e4877ea12",
        ),
        (
            assemble(|a| a.tcvtrowps2phh(ZMM1, TMM2, R30D)),
            "62f20c406dca",
        ),
        (
            assemble(|a| a.tcvtrowps2phh_imm(ZMM29, TMM2, 0x12)),
            "62637c4807ea12",
        ),
        (
            assemble(|a| a.tcvtrowps2phl(ZMM1, TMM2, R30D)),
            "62f20d406dca",
        ),
        (
            assemble(|a| a.tcvtrowps2phl_imm(ZMM29, TMM2, 0x12)),
            "62637f4877ea12",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn tilemovrow_directions_and_index_forms_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.tilemovrow(ZMM1, TMM2, R30D)), "62f20d404aca"),
        (
            assemble(|a| a.tilemovrow_imm(ZMM29, TMM2, 0x12)),
            "62637d4807ea12",
        ),
        (
            assemble(|a| a.tilemovrow(TMM3, ZMM16, R16D)),
            "62b2fd404ad8",
        ),
        (
            assemble(|a| a.tilemovrow_imm(TMM4, ZMM24, 0x3F)),
            "6293fd4807e03f",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.tilemovrow(TMM1, TMM2, EAX), Err(Error::BadCombination));
}
