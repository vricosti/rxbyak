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
fn rao_int_legacy_and_apx_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.aadd(ptr(RAX.into()), ECX)), "0f38fc08"),
        (assemble(|a| a.aand(ptr(RAX.into()), RCX)), "66480f38fc08"),
        (assemble(|a| a.aor(ptr(R16 + R31), R17D)), "62ac7b08fc0c38"),
        (assemble(|a| a.axor(ptr(R16 + R31), R17)), "62acfa08fc0c38"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.aadd(ptr(RAX.into()), AX), Err(Error::BadCombination));
}

#[test]
fn adcx_and_adox_overloads_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.adcx(EAX, ECX)), "660f38f6c1"),
        (assemble(|a| a.adcx(R8, R9)), "664d0f38f6c1"),
        (assemble(|a| a.adcx(R20D, ptr(RAX.into()))), "62e47d086620"),
        (assemble(|a| a.adcx3(RAX, RCX, RDX)), "62f4fd1866ca"),
        (assemble(|a| a.adox(EAX, ECX)), "f30f38f6c1"),
        (assemble(|a| a.adox(R20D, ptr(RAX.into()))), "62e47e086620"),
        (assemble(|a| a.adox3(RAX, RCX, RDX)), "62f4fe1866ca"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.adcx(EAX, RCX), Err(Error::BadSizeOfRegister));
}

#[test]
fn bmi_rro_vex_and_apx_selection_matches_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.blsi(EAX, ECX)), "c4e278f3d9"),
        (assemble(|a| a.blsi(R8, R9)), "c4c2b8f3d9"),
        (assemble(|a| a.blsi(R30.nf(), R31)), "62da8c04f3df"),
        (assemble(|a| a.blsmsk(EAX, ptr(R13 + 16))), "c4c278f35510"),
        (
            assemble(|a| a.blsr(R30, ptr(R31 + R20 * 4))),
            "62da8800f30ca7",
        ),
        (assemble(|a| a.mulx(EAX, ECX, EDX)), "c4e273f6c2"),
        (assemble(|a| a.mulx(R8, R9, ptr(R13 + 16))), "c442b3f64510"),
        (assemble(|a| a.mulx(R29, R30, R31)), "624a8f00f6ef"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.blsi(EAX, RCX), Err(Error::BadCombination));
    assert_eq!(asm.mulx(EAX, RCX, EDX), Err(Error::BadCombination));
}

#[test]
fn existing_bmi_methods_now_share_xbyak_op_rro_semantics() {
    let cases = [
        (assemble(|a| a.andn(R29, R30, R31)), "624a8c00f2ef"),
        (assemble(|a| a.andn(R29.nf(), R30, R31)), "624a8c04f2ef"),
        (assemble(|a| a.bextr(R29, R30, R31)), "624a8400f7ee"),
        (assemble(|a| a.bzhi(R29.nf(), R30, R31)), "624a8404f5ee"),
        (
            assemble(|a| a.sarx(R29, ptr(R31 + R20 * 4), R30)),
            "624a8a00f72ca7",
        ),
        (assemble(|a| a.shlx(R29, R30, R31)), "624a8500f7ee"),
        (assemble(|a| a.shrx(EAX, ECX, R17D)), "62f27700f7c1"),
        (assemble(|a| a.pdep(R29, R30, R31)), "624a8f00f5ef"),
        (
            assemble(|a| a.pext(R29, R30, ptr(R31 + R20 * 4))),
            "624a8a00f52ca7",
        ),
        (
            assemble(|a| a.rorx(R30, ptr(R31 + R20 * 4), 4)),
            "624bfb08f034a704",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}
