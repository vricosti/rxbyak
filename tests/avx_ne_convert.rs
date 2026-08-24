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
fn avx_ne_convert_instructions_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vbcstnebf162ps(XMM8, ptr(R13 + 0x20))),
            "c4427ab14520",
        ),
        (
            assemble(|a| a.vbcstnesh2ps(YMM8, ptr(R13 + 0x20))),
            "c4427db14520",
        ),
        (
            assemble(|a| a.vcvtneebf162ps(XMM8, ptr(R13 + 0x20))),
            "c4427ab04520",
        ),
        (
            assemble(|a| a.vcvtneeph2ps(YMM8, ptr(R13 + 0x20))),
            "c4427db04520",
        ),
        (
            assemble(|a| a.vcvtneobf162ps(XMM8, ptr(R13 + 0x20))),
            "c4427bb04520",
        ),
        (
            assemble(|a| a.vcvtneoph2ps(YMM8, ptr(R13 + 0x20))),
            "c4427cb04520",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn avx_ne_convert_retains_xbyak_non_evex_limit() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vcvtneobf162ps(XMM20, ptr(R13.into())),
        Err(Error::EvexIsInvalid)
    );
}
