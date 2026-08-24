use rxbyak::*;

fn assemble(emit: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut assembler = CodeAssembler::new(4096).unwrap();
    emit(&mut assembler).unwrap();
    assembler.code().to_vec()
}

#[test]
fn vpermil_immediate_overloads_match_xbyak_v740() {
    assert_eq!(
        assemble(|a| a.vpermilps_imm(XMM1, XMM2, 0x1b)),
        [0xc4, 0xe3, 0x79, 0x04, 0xca, 0x1b]
    );
    assert_eq!(
        assemble(|a| a.vpermilpd_imm(XMM1, XMM2, 0x03)),
        [0xc4, 0xe3, 0x79, 0x05, 0xca, 0x03]
    );
    assert_eq!(
        assemble(|a| a.vpermilps_imm(YMM8, YMM9, 0xe4)),
        [0xc4, 0x43, 0x7d, 0x04, 0xc1, 0xe4]
    );
    assert_eq!(
        assemble(|a| a.vpermilpd_imm(ZMM20.k(3).z(), ZMM21, 0x01)),
        [0x62, 0xa3, 0xfd, 0xcb, 0x05, 0xe5, 0x01]
    );
}

#[test]
fn vpermil_variable_control_overloads_remain_distinct() {
    assert_eq!(
        assemble(|a| a.vpermilps(XMM1, XMM2, XMM3)),
        [0xc4, 0xe2, 0x69, 0x0c, 0xcb]
    );
    assert_eq!(
        assemble(|a| a.vpermilpd(XMM1, XMM2, XMM3)),
        [0xc4, 0xe2, 0x69, 0x0d, 0xcb]
    );
}
