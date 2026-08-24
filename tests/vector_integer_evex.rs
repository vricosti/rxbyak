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
fn packed_masked_moves_match_every_xbyak_7_40_direction() {
    let cases = [
        (
            assemble(|a| a.vpmaskmovd_store(ptr(R13 + 0x20), YMM8, YMM9)),
            "c4423d8e4d20",
        ),
        (
            assemble(|a| a.vpmaskmovd(YMM8, YMM9, ptr(R13 + 0x20))),
            "c442358c4520",
        ),
        (
            assemble(|a| a.vpmaskmovq_store(ptr(R13 + 0x20), YMM8, YMM9)),
            "c442bd8e4d20",
        ),
        (
            assemble(|a| a.vpmaskmovq(YMM8, YMM9, ptr(R13 + 0x20))),
            "c442b58c4520",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn packed_compressions_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vpcompressb(XMM20.k(3), ZMM21)),
            "62a27d4b63ec",
        ),
        (
            assemble(|a| a.vpcompressd(ZMM20.k(3), ZMM21)),
            "62a27d4b8bec",
        ),
        (
            assemble(|a| a.vpcompressq(ZMM20.k(3), ZMM21)),
            "62a2fd4b8bec",
        ),
        (
            assemble(|a| a.vpcompressw(YMM20.k(3), ZMM21)),
            "62a2fd4b63ec",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn mask_vector_conversions_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vpmovd2m(K3, ZMM20)), "62b27e4839dc"),
        (assemble(|a| a.vpmovm2b(ZMM20, K3)), "62e27e4828e3"),
        (assemble(|a| a.vpmovm2d(ZMM20, K3)), "62e27e4838e3"),
        (assemble(|a| a.vpmovm2q(ZMM20, K3)), "62e2fe4838e3"),
        (assemble(|a| a.vpmovm2w(ZMM20, K3)), "62e2fe4828e3"),
        (assemble(|a| a.vpmovw2m(K3, ZMM20)), "62b2fe4829dc"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.vpmovd2m(XMM1, ZMM20), Err(Error::BadCombination));
    assert_eq!(asm.vpmovm2d(ZMM20, XMM1), Err(Error::BadCombination));
}

#[test]
fn truncating_and_saturating_moves_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vpmovdb(XMM21, ZMM20)), "62a27e4831e5"),
        (assemble(|a| a.vpmovqb(XMM21, ZMM20)), "62a27e4832e5"),
        (assemble(|a| a.vpmovqw(XMM21, ZMM20)), "62a27e4834e5"),
        (assemble(|a| a.vpmovsdb(XMM21, ZMM20)), "62a27e4821e5"),
        (assemble(|a| a.vpmovsdw(YMM21, ZMM20)), "62a27e4823e5"),
        (assemble(|a| a.vpmovsqb(XMM21, ZMM20)), "62a27e4822e5"),
        (assemble(|a| a.vpmovsqd(YMM21, ZMM20)), "62a27e4825e5"),
        (assemble(|a| a.vpmovsqw(XMM21, ZMM20)), "62a27e4824e5"),
        (assemble(|a| a.vpmovswb(YMM21, ZMM20)), "62a27e4820e5"),
        (assemble(|a| a.vpmovusdb(XMM21, ZMM20)), "62a27e4811e5"),
        (assemble(|a| a.vpmovusdw(YMM21, ZMM20)), "62a27e4813e5"),
        (assemble(|a| a.vpmovusqb(XMM21, ZMM20)), "62a27e4812e5"),
        (assemble(|a| a.vpmovusqd(YMM21, ZMM20)), "62a27e4815e5"),
        (assemble(|a| a.vpmovusqw(XMM21, ZMM20)), "62a27e4814e5"),
        (assemble(|a| a.vpmovuswb(YMM21, ZMM20)), "62a27e4810e5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn narrowing_memory_masks_and_width_modes_match_xbyak_7_40() {
    assert_eq!(
        assemble(|a| a.vpmovdb(ptr(R13 + 0x40).k(3), ZMM20)),
        decode_hex("62c27e4b316504")
    );
    assert_eq!(
        assemble(|a| a.vpmovsdw(ptr(R13 + 0x40).k(3), ZMM20)),
        decode_hex("62c27e4b236502")
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.vpmovsdw(XMM21, ZMM20), Err(Error::BadCombination));
    assert_eq!(asm.vpmovdb(YMM21, ZMM20), Err(Error::BadCombination));
}
