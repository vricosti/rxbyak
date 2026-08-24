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
fn mpx_check_and_move_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.bnd()), "f2"),
        (assemble(|a| a.bndcl(BND1, EAX)), "f30f1ac8"),
        (assemble(|a| a.bndcl(BND1, RAX)), "f30f1ac8"),
        (
            assemble(|a| a.bndcl(BND1, qword_ptr(RAX.into()))),
            "f30f1a08",
        ),
        (assemble(|a| a.bndcl(BND1, ptr(R13 + 16))), "f3410f1a4d10"),
        (assemble(|a| a.bndcn(BND2, R9)), "f2410f1bd1"),
        (assemble(|a| a.bndcu(BND3, ptr(RAX.into()))), "f20f1a18"),
        (assemble(|a| a.bndmov(BND3, BND1)), "660f1ad9"),
        (assemble(|a| a.bndmov(BND2, ptr(R13 + 16))), "66410f1a5510"),
        (
            assemble(|a| a.bndmov_store(ptr(R13 + 16), BND2)),
            "66410f1b5510",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn mpx_mib_preserves_non_optimized_addressing_like_xbyak_7_40() {
    assert_eq!(
        assemble(|a| a.bndldx(BND1, ptr(RAX * 2))),
        decode_hex("0f1a0c4500000000")
    );
    assert_eq!(
        assemble(|a| a.bndmk(BND2, ptr(RAX * 2))),
        decode_hex("f30f1b1400")
    );
    assert_eq!(
        assemble(|a| a.bndstx(ptr(RAX * 2), BND3)),
        decode_hex("0f1b1c4500000000")
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.bndldx(BND0, ptr(RegExp::rip())),
        Err(Error::InvalidMibAddress)
    );
}
