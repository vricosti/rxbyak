use rxbyak::*;

fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

#[test]
fn memory_destination_opmask_matches_xbyak_7_40() {
    assert_eq!(
        assemble(|a| a.vpmovssdb(ptr(R13 + 0x40).k(3), ZMM20)),
        [0x62, 0xC2, 0x7E, 0x4B, 0x41, 0x65, 0x04]
    );
}

#[test]
fn memory_opmask_and_zero_validation_matches_xbyak_7_40() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vpmovssdb(ptr(R13.into()).k(3).z(), ZMM20),
        Err(Error::InvalidZero)
    );

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(
        asm.vaddps(ZMM1, ZMM2, ptr(R13.into()).k(3)),
        Err(Error::InvalidOpmaskWithMemory)
    );
}
