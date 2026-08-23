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
fn variable_blends_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vblendvpd(XMM1, XMM2, XMM3, XMM4)),
            "c4e3694bcb40",
        ),
        (
            assemble(|a| a.vblendvpd(YMM8, YMM9, ptr(R13 + R14 * 8 + 0x20), XMM15)),
            "c403354b44f520f0",
        ),
        (
            assemble(|a| a.vblendvps(XMM1, XMM2, XMM3, XMM4)),
            "c4e3694acb40",
        ),
        (
            assemble(|a| a.vpblendvb(YMM8, YMM9, YMM10, XMM15)),
            "c443354cc2f0",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vblendvpd(XMM1, XMM2, XMM3, EAX),
        Err(Error::BadCombination)
    );
}

#[test]
fn mxcsr_and_masked_moves_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vldmxcsr(ptr(R13 + 0x20))), "c4c178ae5520"),
        (assemble(|a| a.vstmxcsr(ptr(R13 + 0x20))), "c4c178ae5d20"),
        (assemble(|a| a.vmaskmovdqu(XMM8, XMM9)), "c44179f7c1"),
        (
            assemble(|a| a.vmaskmovpd_store(ptr(R13 + 0x20), YMM8, YMM9)),
            "c4423d2f4d20",
        ),
        (
            assemble(|a| a.vmaskmovpd(YMM8, YMM9, ptr(R13 + 0x20))),
            "c442352d4520",
        ),
        (
            assemble(|a| a.vmaskmovps_store(ptr(R13 + 0x20), YMM8, YMM9)),
            "c4423d2e4d20",
        ),
        (
            assemble(|a| a.vmaskmovps(YMM8, YMM9, ptr(R13 + 0x20))),
            "c442352c4520",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn half_moves_and_non_temporal_loads_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vmovhlps(XMM8, XMM9, XMM10)), "c4413012c2"),
        (assemble(|a| a.vmovhlps_2(XMM8, XMM9)), "c4413812c1"),
        (assemble(|a| a.vmovlhps(XMM8, XMM9, XMM10)), "c4413016c2"),
        (assemble(|a| a.vmovlhps_2(XMM8, XMM9)), "c4413816c1"),
        (
            assemble(|a| a.vmovntdqa(XMM8, ptr(R13 + 0x20))),
            "c442792a4520",
        ),
        (
            assemble(|a| a.vmovntdqa(YMM8, ptr(R13 + 0x20))),
            "c4427d2a4520",
        ),
        (
            assemble(|a| a.vmovntdqa(ZMM8, ptr(R13 + 0x20))),
            "62527d482a8520000000",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn evex_compress_half_and_absolute_moves_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vcompresspd(zmmword_ptr(R13 + 0x40).k(3), ZMM20)),
            "62c2fd4b8a6508",
        ),
        (
            assemble(|a| a.vcompresspd(ZMM20.k(3), ZMM21)),
            "62a2fd4b8aec",
        ),
        (
            assemble(|a| a.vcompressps(ymmword_ptr(R13 + 0x40).k(3), YMM20)),
            "62c27d2b8a6510",
        ),
        (
            assemble(|a| a.vmovsh_store(word_ptr(R13 + 0x20).k(3), XMM20)),
            "62c57e0b116510",
        ),
        (
            assemble(|a| a.vmovsh_load(XMM20, word_ptr(R13 + 0x20))),
            "62c57e08106510",
        ),
        (assemble(|a| a.vmovsh(XMM20, XMM21, XMM22)), "62a5560010e6"),
        (assemble(|a| a.vpabsq(ZMM20.k(3), ZMM21)), "62a2fd4b1fe5"),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn direct_store_vector_loads_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vmovrsb(XMM20, ptr(R13 + 0x20))),
            "62c57f086f6502",
        ),
        (
            assemble(|a| a.vmovrsd(XMM20, ptr(R13 + 0x20))),
            "62c57e086f6502",
        ),
        (
            assemble(|a| a.vmovrsq(XMM20, ptr(R13 + 0x20))),
            "62c5fe086f6502",
        ),
        (
            assemble(|a| a.vmovrsw(XMM20, ptr(R13 + 0x20))),
            "62c5ff086f6502",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}
