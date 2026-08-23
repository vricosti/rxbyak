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
fn aes_key_locker_legacy_and_apx_encodings_match_xbyak_7_40() {
    let legacy = ptr(RAX + RCX * 4 + 0x12);
    let apx = ptr(R30 + R29 * 8 + 0x34);
    let cases = [
        (
            assemble(|a| a.aesdec128kl(XMM15, legacy)),
            "f3440f38dd7c8812",
        ),
        (assemble(|a| a.aesdec128kl(XMM15, apx)), "621c7a08dd7cee34"),
        (
            assemble(|a| a.aesdec256kl(XMM15, legacy)),
            "f3440f38df7c8812",
        ),
        (assemble(|a| a.aesdec256kl(XMM15, apx)), "621c7a08df7cee34"),
        (assemble(|a| a.aesdecwide128kl(legacy)), "f30f38d84c8812"),
        (assemble(|a| a.aesdecwide128kl(apx)), "629c7a08d84cee34"),
        (assemble(|a| a.aesdecwide256kl(legacy)), "f30f38d85c8812"),
        (assemble(|a| a.aesdecwide256kl(apx)), "629c7a08d85cee34"),
        (
            assemble(|a| a.aesenc128kl(XMM15, legacy)),
            "f3440f38dc7c8812",
        ),
        (assemble(|a| a.aesenc128kl(XMM15, apx)), "621c7a08dc7cee34"),
        (
            assemble(|a| a.aesenc256kl(XMM15, legacy)),
            "f3440f38de7c8812",
        ),
        (assemble(|a| a.aesenc256kl(XMM15, apx)), "621c7a08de7cee34"),
        (assemble(|a| a.aesencwide128kl(legacy)), "f30f38d8448812"),
        (assemble(|a| a.aesencwide128kl(apx)), "629c7a08d844ee34"),
        (assemble(|a| a.aesencwide256kl(legacy)), "f30f38d8548812"),
        (assemble(|a| a.aesencwide256kl(apx)), "629c7a08d854ee34"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.aesenc128kl(XMM16, legacy), Err(Error::NotSupported));

    assert_eq!(
        assemble(|a| a.aesenc128kl(YMM1, ptr(R16.into()))),
        decode_hex("62fc7e08dc08")
    );
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.aesenc128kl(YMM1, ptr(RAX.into())),
        Err(Error::BadCombination)
    );
}

#[test]
fn encodekey_register_threshold_matches_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.encodekey128(EAX, EBX)), "f30f38fac3"),
        (assemble(|a| a.encodekey128(EAX, R8D)), "62d47e08dac0"),
        (assemble(|a| a.encodekey128(R8D, EBX)), "62747e08dac3"),
        (assemble(|a| a.encodekey128(R30D, R29D)), "624c7e08daf5"),
        (assemble(|a| a.encodekey256(EAX, EBX)), "f30f38fbc3"),
        (assemble(|a| a.encodekey256(EAX, R8D)), "62d47e08dbc0"),
        (assemble(|a| a.encodekey256(R8D, EBX)), "62747e08dbc3"),
        (assemble(|a| a.encodekey256(R30D, R29D)), "624c7e08dbf5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.encodekey128(RAX, EBX), Err(Error::BadCombination));
}
