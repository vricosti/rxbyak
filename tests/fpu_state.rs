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
fn bcd_and_integer_reverse_arithmetic_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.fbld(ptr(RAX.into()))), "df20"),
        (
            assemble(|a| a.fbstp(ptr(R13 + R14 * 8 + 0x20))),
            "43df74f520",
        ),
        (assemble(|a| a.fidivr(word_ptr(RAX.into()))), "de38"),
        (assemble(|a| a.fidivr(dword_ptr(R8 + 0x10))), "41da7810"),
        (assemble(|a| a.fisubr(word_ptr(RAX.into()))), "de28"),
        (assemble(|a| a.fisubr(dword_ptr(R8 + 0x10))), "41da6810"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.fidivr(ptr(RAX.into())), Err(Error::BadMemSize));
    assert_eq!(asm.fisubr(ptr(RAX.into())), Err(Error::BadMemSize));
}

#[test]
fn x87_environment_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.fldenv(ptr(RAX.into()))), "d920"),
        (assemble(|a| a.fnsave(ptr(R8 + 0x10))), "41dd7010"),
        (
            assemble(|a| a.fnstenv(ptr(R13 + R14 * 8 + 0x20))),
            "43d974f520",
        ),
        (assemble(|a| a.frstor(ptr(RAX.into()))), "dd20"),
        (assemble(|a| a.fsave(ptr(R8 + 0x10))), "9b41dd7010"),
        (
            assemble(|a| a.fstenv(ptr(R13 + R14 * 8 + 0x20))),
            "9b43d974f520",
        ),
        (assemble(|a| a.fstsw(ptr(RAX.into()))), "9bdd38"),
        (assemble(|a| a.fstsw_reg(AX)), "9bdfe0"),
        (assemble(|a| a.fsincos()), "d9fb"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.fstsw_reg(CX), Err(Error::BadParameter));
}

#[test]
fn fxrstor_encodings_match_xbyak_7_40() {
    let address = ptr(R13 + R14 * 8 + 0x20);
    assert_eq!(assemble(|a| a.fxrstor(address)), decode_hex("430fae4cf520"));
    assert_eq!(
        assemble(|a| a.fxrstor64(address)),
        decode_hex("4b0fae4cf520")
    );
}
