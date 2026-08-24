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
fn avx2_gathers_match_every_xbyak_7_40_family() {
    let cases = [
        (
            assemble(|a| a.vgatherdpd_avx2(YMM8, ptr(R13 + XMM14 * 4 + 0x20), YMM9)),
            "c402b59244b520",
        ),
        (
            assemble(|a| a.vgatherdps_avx2(YMM8, ptr(R13 + YMM14 * 4 + 0x20), YMM9)),
            "c402359244b520",
        ),
        (
            assemble(|a| a.vgatherqpd_avx2(YMM8, ptr(R13 + YMM14 * 4 + 0x20), YMM9)),
            "c402b59344b520",
        ),
        (
            assemble(|a| a.vgatherqps_avx2(XMM8, ptr(R13 + YMM14 * 4 + 0x20), XMM9)),
            "c402359344b520",
        ),
        (
            assemble(|a| a.vpgatherdd_avx2(YMM8, ptr(R13 + YMM14 * 4 + 0x20), YMM9)),
            "c402359044b520",
        ),
        (
            assemble(|a| a.vpgatherdq_avx2(YMM8, ptr(R13 + XMM14 * 4 + 0x20), YMM9)),
            "c402b59044b520",
        ),
        (
            assemble(|a| a.vpgatherqd_avx2(XMM8, ptr(R13 + YMM14 * 4 + 0x20), XMM9)),
            "c402359144b520",
        ),
        (
            assemble(|a| a.vpgatherqq_avx2(YMM8, ptr(R13 + YMM14 * 4 + 0x20), YMM9)),
            "c402b59144b520",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn evex_gathers_match_every_xbyak_7_40_family() {
    let cases = [
        (
            assemble(|a| a.vgatherdpd(ZMM20.k(3), ptr(R13 + YMM22 * 4 + 0x20))),
            "62c2fd439264b504",
        ),
        (
            assemble(|a| a.vgatherdps(ZMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c27d439264b508",
        ),
        (
            assemble(|a| a.vgatherqpd(ZMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c2fd439364b504",
        ),
        (
            assemble(|a| a.vgatherqps(YMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c27d439364b508",
        ),
        (
            assemble(|a| a.vpgatherdd(ZMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c27d439064b508",
        ),
        (
            assemble(|a| a.vpgatherdq(ZMM20.k(3), ptr(R13 + YMM22 * 4 + 0x20))),
            "62c2fd439064b504",
        ),
        (
            assemble(|a| a.vpgatherqd(YMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c27d439164b508",
        ),
        (
            assemble(|a| a.vpgatherqq(ZMM20.k(3), ptr(R13 + ZMM22 * 4 + 0x20))),
            "62c2fd439164b504",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn evex_scatters_match_every_xbyak_7_40_family() {
    let cases = [
        (
            assemble(|a| a.vpscatterdd(ptr(R13 + ZMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c27d43a064b508",
        ),
        (
            assemble(|a| a.vpscatterdq(ptr(R13 + YMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c2fd43a064b504",
        ),
        (
            assemble(|a| a.vpscatterqd(ptr(R13 + ZMM22 * 4 + 0x20).k(3), YMM20)),
            "62c27d43a164b508",
        ),
        (
            assemble(|a| a.vpscatterqq(ptr(R13 + ZMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c2fd43a164b504",
        ),
        (
            assemble(|a| a.vscatterdpd(ptr(R13 + YMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c2fd43a264b504",
        ),
        (
            assemble(|a| a.vscatterdps(ptr(R13 + ZMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c27d43a264b508",
        ),
        (
            assemble(|a| a.vscatterqpd(ptr(R13 + ZMM22 * 4 + 0x20).k(3), ZMM20)),
            "62c2fd43a364b504",
        ),
        (
            assemble(|a| a.vscatterqps(ptr(R13 + ZMM22 * 4 + 0x20).k(3), YMM20)),
            "62c27d43a364b508",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn gather_prefetches_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vgatherpf0dpd(ptr(R13 + YMM22 * 4 + 0x20).k(3))),
            "62d2fd43c64cb504",
        ),
        (
            assemble(|a| a.vgatherpf0dps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c64cb508",
        ),
        (
            assemble(|a| a.vgatherpf0qpd(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d2fd43c74cb504",
        ),
        (
            assemble(|a| a.vgatherpf0qps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c74cb508",
        ),
        (
            assemble(|a| a.vgatherpf1dpd(ptr(R13 + YMM22 * 4 + 0x20).k(3))),
            "62d2fd43c654b504",
        ),
        (
            assemble(|a| a.vgatherpf1dps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c654b508",
        ),
        (
            assemble(|a| a.vgatherpf1qpd(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d2fd43c754b504",
        ),
        (
            assemble(|a| a.vgatherpf1qps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c754b508",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn scatter_prefetches_match_xbyak_7_40() {
    let cases = [
        (
            assemble(|a| a.vscatterpf0dpd(ptr(R13 + YMM22 * 4 + 0x20).k(3))),
            "62d2fd43c66cb504",
        ),
        (
            assemble(|a| a.vscatterpf0dps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c66cb508",
        ),
        (
            assemble(|a| a.vscatterpf0qpd(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d2fd43c76cb504",
        ),
        (
            assemble(|a| a.vscatterpf0qps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c76cb508",
        ),
        (
            assemble(|a| a.vscatterpf1dpd(ptr(R13 + YMM22 * 4 + 0x20).k(3))),
            "62d2fd43c674b504",
        ),
        (
            assemble(|a| a.vscatterpf1dps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c674b508",
        ),
        (
            assemble(|a| a.vscatterpf1qpd(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d2fd43c774b504",
        ),
        (
            assemble(|a| a.vscatterpf1qps(ptr(R13 + ZMM22 * 4 + 0x20).k(3))),
            "62d27d43c774b508",
        ),
    ];
    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn gather_validation_matches_xbyak_contracts() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vgatherdps(ZMM20, ptr(R13 + ZMM22 * 4)),
        Err(Error::K0IsInvalid)
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vgatherdps(ZMM20.k(3).z(), ptr(R13 + ZMM22 * 4)),
        Err(Error::InvalidZero)
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vgatherdps(ZMM20.k(3), ptr(R13 + ZMM20 * 4)),
        Err(Error::SameRegsAreInvalid)
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vgatherdps_avx2(YMM8, ptr(R13 + YMM8 * 4), YMM9),
        Err(Error::SameRegsAreInvalid)
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vgatherpf0dpd(ptr(R13 + ZMM22 * 4).k(3)),
        Err(Error::BadVsibAddressing)
    );
}
