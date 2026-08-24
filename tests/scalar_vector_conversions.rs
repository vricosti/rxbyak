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
fn signed_scalar_float_to_integer_matches_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vcvtsd2si(R20D, XMM21)), "62a17f082de5"),
        (assemble(|a| a.vcvtsd2si(R20, XMM21)), "62a1ff082de5"),
        (assemble(|a| a.vcvtss2si(R20D, XMM21)), "62a17e082de5"),
        (assemble(|a| a.vcvtss2si(R20, XMM21)), "62a1fe082de5"),
        (assemble(|a| a.vcvttsd2si(R20D, XMM21)), "62a17f082ce5"),
        (assemble(|a| a.vcvttsd2si(R20, XMM21)), "62a1ff082ce5"),
        (assemble(|a| a.vcvttss2si(R20D, XMM21)), "62a17e082ce5"),
        (assemble(|a| a.vcvttss2si(R20, XMM21)), "62a1fe082ce5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn unsigned_scalar_float_to_integer_matches_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vcvtsd2usi(R20D, XMM21)), "62a17f0879e5"),
        (assemble(|a| a.vcvtsd2usi(R20, XMM21)), "62a1ff0879e5"),
        (assemble(|a| a.vcvtss2usi(R20D, XMM21)), "62a17e0879e5"),
        (assemble(|a| a.vcvtss2usi(R20, XMM21)), "62a1fe0879e5"),
        (assemble(|a| a.vcvttsd2usi(R20D, XMM21)), "62a17f0878e5"),
        (assemble(|a| a.vcvttsd2usi(R20, XMM21)), "62a1ff0878e5"),
        (assemble(|a| a.vcvttss2usi(R20D, XMM21)), "62a17e0878e5"),
        (assemble(|a| a.vcvttss2usi(R20, XMM21)), "62a1fe0878e5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn half_scalar_to_integer_matches_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vcvtsh2si(R20D, XMM21)), "62a57e082de5"),
        (assemble(|a| a.vcvtsh2si(R20, XMM21)), "62a5fe082de5"),
        (assemble(|a| a.vcvtsh2usi(R20D, XMM21)), "62a57e0879e5"),
        (assemble(|a| a.vcvttsh2si(R20, XMM21)), "62a5fe082ce5"),
        (assemble(|a| a.vcvttsh2usi(R20D, XMM21)), "62a57e0878e5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn integer_to_scalar_half_matches_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vcvtsi2sh(XMM20, XMM21, R22D)),
            "62ed56002ae6",
        ),
        (assemble(|a| a.vcvtsi2sh(XMM20, XMM21, R22)), "62edd6002ae6"),
        (
            assemble(|a| a.vcvtusi2sh(XMM20, XMM21, R22D)),
            "62ed56007be6",
        ),
        (
            assemble(|a| a.vcvtusi2sh(XMM20, XMM21, R22)),
            "62edd6007be6",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vcvtsi2sh(XMM1, XMM2, ptr(RAX.into())),
        Err(Error::BadCombination)
    );
}
