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
        .map(|pair| {
            let text = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(text, 16).unwrap()
        })
        .collect()
}

#[test]
fn test_vcmp_base_overloads_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.vcmppd(XMM1, XMM2, XMM3, 0x1d)), "c5e9c2cb1d"),
        (assemble(|a| a.vcmppd(YMM1, YMM2, YMM3, 0x1d)), "c5edc2cb1d"),
        (assemble(|a| a.vcmpps(XMM4, XMM5, XMM6, 0x1e)), "c5d0c2e61e"),
        (assemble(|a| a.vcmpps(YMM4, YMM5, YMM6, 0x1e)), "c5d4c2e61e"),
        (
            assemble(|a| a.vcmpsd(XMM7, XMM8, XMM9, 0x1f)),
            "c4c13bc2f91f",
        ),
        (
            assemble(|a| a.vcmpss(XMM10, XMM11, XMM12, 0x10)),
            "c44122c2d410",
        ),
        (
            assemble(|a| a.vcmppd(K1, XMM2, XMM3, 0x1d)),
            "62f1ed08c2cb1d",
        ),
        (
            assemble(|a| a.vcmppd(K2, YMM3, YMM4, 0x1d)),
            "62f1e528c2d41d",
        ),
        (
            assemble(|a| a.vcmppd(K3, ZMM4, ZMM5, 0x1d)),
            "62f1dd48c2dd1d",
        ),
        (
            assemble(|a| a.vcmpps(K4, ZMM5, ZMM6, 0x1e)),
            "62f15448c2e61e",
        ),
        (
            assemble(|a| a.vcmpsd(K5, XMM6, XMM7, 0x1f)),
            "62f1cf08c2ef1f",
        ),
        (
            assemble(|a| a.vcmpss(K6, XMM7, XMM8, 0x10)),
            "62d14608c2f010",
        ),
    ];

    for (actual, expected_hex) in cases {
        assert_eq!(actual, decode_hex(expected_hex));
    }
}

macro_rules! assert_vcmp_predicate {
    ($imm:expr, $pd:ident, $ps:ident, $sd:ident, $ss:ident) => {{
        assert_eq!(
            assemble(|a| a.$pd(XMM1, XMM2, XMM3)),
            assemble(|a| a.vcmppd(XMM1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ps(XMM1, XMM2, XMM3)),
            assemble(|a| a.vcmpps(XMM1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$sd(XMM1, XMM2, XMM3)),
            assemble(|a| a.vcmpsd(XMM1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ss(XMM1, XMM2, XMM3)),
            assemble(|a| a.vcmpss(XMM1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$pd(K1, XMM2, XMM3)),
            assemble(|a| a.vcmppd(K1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ps(K1, XMM2, XMM3)),
            assemble(|a| a.vcmpps(K1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$sd(K1, XMM2, XMM3)),
            assemble(|a| a.vcmpsd(K1, XMM2, XMM3, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ss(K1, XMM2, XMM3)),
            assemble(|a| a.vcmpss(K1, XMM2, XMM3, $imm))
        );
    }};
}

#[test]
fn test_all_vcmp_predicate_aliases_use_xbyak_immediates() {
    assert_vcmp_predicate!(0, vcmpeqpd, vcmpeqps, vcmpeqsd, vcmpeqss);
    assert_vcmp_predicate!(1, vcmpltpd, vcmpltps, vcmpltsd, vcmpltss);
    assert_vcmp_predicate!(2, vcmplepd, vcmpleps, vcmplesd, vcmpless);
    assert_vcmp_predicate!(3, vcmpunordpd, vcmpunordps, vcmpunordsd, vcmpunordss);
    assert_vcmp_predicate!(4, vcmpneqpd, vcmpneqps, vcmpneqsd, vcmpneqss);
    assert_vcmp_predicate!(5, vcmpnltpd, vcmpnltps, vcmpnltsd, vcmpnltss);
    assert_vcmp_predicate!(6, vcmpnlepd, vcmpnleps, vcmpnlesd, vcmpnless);
    assert_vcmp_predicate!(7, vcmpordpd, vcmpordps, vcmpordsd, vcmpordss);
    assert_vcmp_predicate!(8, vcmpeq_uqpd, vcmpeq_uqps, vcmpeq_uqsd, vcmpeq_uqss);
    assert_vcmp_predicate!(9, vcmpngepd, vcmpngeps, vcmpngesd, vcmpngess);
    assert_vcmp_predicate!(10, vcmpngtpd, vcmpngtps, vcmpngtsd, vcmpngtss);
    assert_vcmp_predicate!(11, vcmpfalsepd, vcmpfalseps, vcmpfalsesd, vcmpfalsess);
    assert_vcmp_predicate!(12, vcmpneq_oqpd, vcmpneq_oqps, vcmpneq_oqsd, vcmpneq_oqss);
    assert_vcmp_predicate!(13, vcmpgepd, vcmpgeps, vcmpgesd, vcmpgess);
    assert_vcmp_predicate!(14, vcmpgtpd, vcmpgtps, vcmpgtsd, vcmpgtss);
    assert_vcmp_predicate!(15, vcmptruepd, vcmptrueps, vcmptruesd, vcmptruess);
    assert_vcmp_predicate!(16, vcmpeq_ospd, vcmpeq_osps, vcmpeq_ossd, vcmpeq_osss);
    assert_vcmp_predicate!(17, vcmplt_oqpd, vcmplt_oqps, vcmplt_oqsd, vcmplt_oqss);
    assert_vcmp_predicate!(18, vcmple_oqpd, vcmple_oqps, vcmple_oqsd, vcmple_oqss);
    assert_vcmp_predicate!(
        19,
        vcmpunord_spd,
        vcmpunord_sps,
        vcmpunord_ssd,
        vcmpunord_sss
    );
    assert_vcmp_predicate!(20, vcmpneq_uspd, vcmpneq_usps, vcmpneq_ussd, vcmpneq_usss);
    assert_vcmp_predicate!(21, vcmpnlt_uqpd, vcmpnlt_uqps, vcmpnlt_uqsd, vcmpnlt_uqss);
    assert_vcmp_predicate!(22, vcmpnle_uqpd, vcmpnle_uqps, vcmpnle_uqsd, vcmpnle_uqss);
    assert_vcmp_predicate!(23, vcmpord_spd, vcmpord_sps, vcmpord_ssd, vcmpord_sss);
    assert_vcmp_predicate!(24, vcmpeq_uspd, vcmpeq_usps, vcmpeq_ussd, vcmpeq_usss);
    assert_vcmp_predicate!(25, vcmpnge_uqpd, vcmpnge_uqps, vcmpnge_uqsd, vcmpnge_uqss);
    assert_vcmp_predicate!(26, vcmpngt_uqpd, vcmpngt_uqps, vcmpngt_uqsd, vcmpngt_uqss);
    assert_vcmp_predicate!(
        27,
        vcmpfalse_ospd,
        vcmpfalse_osps,
        vcmpfalse_ossd,
        vcmpfalse_osss
    );
    assert_vcmp_predicate!(28, vcmpneq_ospd, vcmpneq_osps, vcmpneq_ossd, vcmpneq_osss);
    assert_vcmp_predicate!(29, vcmpge_oqpd, vcmpge_oqps, vcmpge_oqsd, vcmpge_oqss);
    assert_vcmp_predicate!(30, vcmpgt_oqpd, vcmpgt_oqps, vcmpgt_oqsd, vcmpgt_oqss);
    assert_vcmp_predicate!(
        31,
        vcmptrue_uspd,
        vcmptrue_usps,
        vcmptrue_ussd,
        vcmptrue_usss
    );
}

macro_rules! assert_sse_predicate {
    ($imm:expr, $pd:ident, $ps:ident, $sd:ident, $ss:ident) => {{
        assert_eq!(
            assemble(|a| a.$pd(XMM1, XMM2)),
            assemble(|a| a.cmppd(XMM1, XMM2, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ps(XMM1, XMM2)),
            assemble(|a| a.cmpps(XMM1, XMM2, $imm))
        );
        assert_eq!(
            assemble(|a| a.$sd(XMM1, XMM2)),
            assemble(|a| a.cmpsd_xmm(XMM1, XMM2, $imm))
        );
        assert_eq!(
            assemble(|a| a.$ss(XMM1, XMM2)),
            assemble(|a| a.cmpss(XMM1, XMM2, $imm))
        );
    }};
}

#[test]
fn test_all_sse_compare_aliases_use_xbyak_immediates() {
    assert_sse_predicate!(0, cmpeqpd, cmpeqps, cmpeqsd, cmpeqss);
    assert_sse_predicate!(1, cmpltpd, cmpltps, cmpltsd, cmpltss);
    assert_sse_predicate!(2, cmplepd, cmpleps, cmplesd, cmpless);
    assert_sse_predicate!(3, cmpunordpd, cmpunordps, cmpunordsd, cmpunordss);
    assert_sse_predicate!(4, cmpneqpd, cmpneqps, cmpneqsd, cmpneqss);
    assert_sse_predicate!(5, cmpnltpd, cmpnltps, cmpnltsd, cmpnltss);
    assert_sse_predicate!(6, cmpnlepd, cmpnleps, cmpnlesd, cmpnless);
    assert_sse_predicate!(7, cmpordpd, cmpordps, cmpordsd, cmpordss);
}

macro_rules! assert_pclmul_alias {
    ($imm:expr, $legacy:ident, $vector:ident) => {{
        assert_eq!(
            assemble(|a| a.$legacy(XMM1, XMM2)),
            assemble(|a| a.pclmulqdq(XMM1, XMM2, $imm))
        );
        assert_eq!(
            assemble(|a| a.$vector(YMM1, YMM2, YMM3)),
            assemble(|a| a.vpclmulqdq(YMM1, YMM2, YMM3, $imm))
        );
    }};
}

#[test]
fn test_pclmul_aliases_use_xbyak_immediates() {
    assert_pclmul_alias!(0x00, pclmullqlqdq, vpclmullqlqdq);
    assert_pclmul_alias!(0x01, pclmulhqlqdq, vpclmulhqlqdq);
    assert_pclmul_alias!(0x10, pclmullqhqdq, vpclmullqhqdq);
    assert_pclmul_alias!(0x11, pclmulhqhqdq, vpclmulhqhqdq);
    assert_eq!(
        assemble(|a| a.pclmulhqhqdq(XMM1, XMM2)),
        [0x66, 0x0f, 0x3a, 0x44, 0xca, 0x11]
    );
    assert_eq!(
        assemble(|a| a.vpclmulhqhqdq(YMM1, YMM2, YMM3)),
        [0xc4, 0xe3, 0x6d, 0x44, 0xcb, 0x11]
    );
}

#[test]
fn test_scalar_spelling_aliases_match_xbyak() {
    assert_eq!(assemble(|a| a.pushfq()), [0x9c]);
    assert_eq!(assemble(|a| a.popfq()), [0x9d]);
    assert_eq!(assemble(|a| a.wait()), [0x9b]);
    assert_eq!(assemble(|a| a.sal(EAX, 3)), assemble(|a| a.shl(EAX, 3)));
    assert_eq!(assemble(|a| a.sal_cl(RAX)), assemble(|a| a.shl_cl(RAX)));

    macro_rules! assert_jump_alias {
        ($alias:ident, $canonical:ident) => {{
            let alias = assemble(|a| {
                let label = a.create_label();
                a.$alias(&label, JmpType::Short)?;
                a.bind(&label)
            });
            let canonical = assemble(|a| {
                let label = a.create_label();
                a.$canonical(&label, JmpType::Short)?;
                a.bind(&label)
            });
            assert_eq!(alias, canonical);
        }};
    }

    assert_jump_alias!(jna, jbe);
    assert_jump_alias!(jnae, jb);
    assert_jump_alias!(jng, jle);
    assert_jump_alias!(jnge, jl);
    assert_jump_alias!(jpe, jp);
    assert_jump_alias!(jpo, jnp);
}
