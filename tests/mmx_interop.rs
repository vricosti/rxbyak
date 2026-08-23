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
fn mmx_conversion_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.cvtpd2pi(MM1, XMM2)), "660f2dca"),
        (
            assemble(|a| a.cvtpd2pi(MM3, ptr(R13 + 0x20))),
            "66410f2d5d20",
        ),
        (assemble(|a| a.cvtpi2pd(XMM4, MM5)), "660f2ae5"),
        (assemble(|a| a.cvtpi2ps(XMM6, ptr(R8 + 0x10))), "410f2a7010"),
        (assemble(|a| a.cvtps2pi(MM7, XMM8)), "410f2df8"),
        (assemble(|a| a.cvttpd2pi(MM0, ptr(RAX.into()))), "660f2c00"),
        (assemble(|a| a.cvttps2pi(MM2, XMM3)), "0f2cd3"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.cvtpd2pi(XMM1, XMM2), Err(Error::BadCombination));
}

#[test]
fn mmx_move_and_mask_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.maskmovdqu(XMM8, XMM9)), "66450ff7c1"),
        (assemble(|a| a.maskmovq(MM1, MM2)), "0ff7ca"),
        (assemble(|a| a.movdq2q(MM3, XMM4)), "f20fd6dc"),
        (assemble(|a| a.movq2dq(XMM5, MM6)), "f30fd6ee"),
        (
            assemble(|a| a.movntq(ptr(R13 + R14 * 8 + 0x20), MM7)),
            "430fe77cf520",
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.maskmovdqu(XMM16, XMM1), Err(Error::NotSupported));
    assert_eq!(
        asm.movntq(ptr(RAX.into()), XMM1),
        Err(Error::BadCombination)
    );
}

#[test]
fn pshufw_encodings_and_operand_validation_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.pshufw(MM1, MM2, 0x7f)), "0f70ca7f"),
        (
            assemble(|a| a.pshufw(MM3, ptr(R8 + 0x10), 0x55)),
            "410f70581055",
        ),
        (assemble(|a| a.pshufw(XMM1, XMM2, 0x33)), "0f70ca33"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.pshufw(XMM1, MM2, 0x33), Err(Error::BadCombination));
}
