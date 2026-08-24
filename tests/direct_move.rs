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
fn movdiri_and_movdir64b_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.movdiri(ptr(RAX.into()), R8D)), "440f38f900"),
        (
            assemble(|a| a.movdiri(ptr(RAX.into()), R16D)),
            "62e47c08f900",
        ),
        (
            assemble(|a| a.movdir64b(R8, ptr(RAX.into()))),
            "66440f38f800",
        ),
        (
            assemble(|a| a.movdir64b(R16, ptr(RAX.into()))),
            "62e47d08f800",
        ),
        (
            assemble(|a| a.movdir64b(XMM1, ptr(RAX.into()))),
            "660f38f808",
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.movdiri(ptr(RAX.into()), AX), Err(Error::BadCombination));
}

#[test]
fn movbe_legacy_and_apx_directions_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.movbe(R8D, ptr(RAX.into()))), "440f38f000"),
        (assemble(|a| a.movbe(R16D, ptr(RAX.into()))), "62e47c086000"),
        (assemble(|a| a.movbe(ptr(RAX.into()), R8D)), "440f38f100"),
        (assemble(|a| a.movbe(ptr(RAX.into()), R16D)), "62e47c086100"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.movbe(ptr(RAX.into()), ptr(RBX.into())),
        Err(Error::BadCombination)
    );
}

#[test]
fn movntdqa_matches_xbyak_legacy_sse_contract() {
    assert_eq!(
        assemble(|a| a.movntdqa(XMM8, ptr(R13 + R14 * 8 + 0x20))),
        decode_hex("66470f382a44f520")
    );
    assert_eq!(
        assemble(|a| a.movntdqa(YMM1, ptr(RAX.into()))),
        decode_hex("660f382a08")
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.movntdqa(XMM16, ptr(RAX.into())),
        Err(Error::NotSupported)
    );
}
