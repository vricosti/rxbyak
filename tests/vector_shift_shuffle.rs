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
fn byte_lane_shifts_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vpslldq(XMM20, XMM21, 0x12)),
            "62b15d0073fd12",
        ),
        (
            assemble(|a| a.vpslldq(ZMM20, ZMM21, 0x12)),
            "62b15d4073fd12",
        ),
        (
            assemble(|a| a.vpsrldq(YMM20, YMM21, 0x12)),
            "62b15d2073dd12",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn packed_rotates_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vprold(ZMM20, ZMM21, 0x12)), "62b15d4072cd12"),
        (assemble(|a| a.vprolq(ZMM20, ZMM21, 0x12)), "62b1dd4072cd12"),
        (assemble(|a| a.vprord(ZMM20, ZMM21, 0x12)), "62b15d4072c512"),
        (assemble(|a| a.vprorq(ZMM20, ZMM21, 0x12)), "62b1dd4072c512"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn cross_lane_shuffles_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vshuff32x4(YMM20, YMM21, YMM22, 0x12)),
            "62a3552023e612",
        ),
        (
            assemble(|a| a.vshuff64x2(ZMM20, ZMM21, ZMM22, 0x12)),
            "62a3d54023e612",
        ),
        (
            assemble(|a| a.vshufi32x4(YMM20, YMM21, YMM22, 0x12)),
            "62a3552043e612",
        ),
        (
            assemble(|a| a.vshufi64x2(ZMM20, ZMM21, ZMM22, 0x12)),
            "62a3d54043e612",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vshuff32x4(XMM1, XMM2, XMM3, 0),
        Err(Error::BadCombination)
    );
}
