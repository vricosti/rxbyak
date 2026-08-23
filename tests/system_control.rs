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
fn zero_operand_and_immediate_encodings_match_xbyak_7_40() {
    let bytes = assemble(|asm| {
        asm.clzero()?;
        asm.endbr32()?;
        asm.endbr64()?;
        asm.monitor()?;
        asm.monitorx()?;
        asm.mwait()?;
        asm.mwaitx()?;
        asm.rdmsr()?;
        asm.rdpmc()?;
        asm.serialize()?;
        asm.stac()?;
        asm.syscall()?;
        asm.sysenter()?;
        asm.sysexit()?;
        asm.sysret()?;
        asm.wbinvd()?;
        asm.wrmsr()?;
        asm.xabort(0x7f)?;
        asm.xbegin(0x1234_5678)?;
        asm.xend()?;
        asm.xgetbv()?;
        asm.xlatb()?;
        asm.xresldtrk()?;
        asm.xsusldtrk()?;
        asm.clui()?;
        asm.stui()?;
        asm.testui()?;
        asm.uiret()
    });

    assert_eq!(
        bytes,
        decode_hex(concat!(
            "0f01fc",
            "f30f1efb",
            "f30f1efa",
            "0f01c8",
            "0f01fa",
            "0f01c9",
            "0f01fb",
            "0f32",
            "0f33",
            "0f01e8",
            "0f01cb",
            "0f05",
            "0f34",
            "0f35",
            "0f07",
            "0f09",
            "0f30",
            "c6f87f",
            "c7f878563412",
            "0f01d5",
            "0f01d0",
            "d7",
            "f20f01e9",
            "f20f01e8",
            "f30f01ee",
            "f30f01ef",
            "f30f01ed",
            "f30f01ec"
        ))
    );
}

#[test]
fn register_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.rdrand(AX)), "660fc7f0"),
        (assemble(|a| a.rdrand(EAX)), "0fc7f0"),
        (assemble(|a| a.rdrand(RAX)), "480fc7f0"),
        (assemble(|a| a.rdrand(R8)), "490fc7f0"),
        (assemble(|a| a.rdrand(R16)), "d598c7f0"),
        (assemble(|a| a.rdrand(XMM1)), "0fc7f1"),
        (assemble(|a| a.rdseed(R9W)), "66410fc7f9"),
        (assemble(|a| a.rdseed(R17D)), "d590c7f9"),
        (assemble(|a| a.rdfsbase(EAX)), "f30faec0"),
        (assemble(|a| a.rdfsbase(RAX)), "f3480faec0"),
        (assemble(|a| a.rdgsbase(R8D)), "f3410faec8"),
        (assemble(|a| a.wrfsbase(R8)), "f3490faed0"),
        (assemble(|a| a.wrgsbase(R16)), "f3d598aed8"),
        (assemble(|a| a.senduipi(RAX)), "f30fc7f0"),
        (assemble(|a| a.senduipi(R8)), "f3410fc7f0"),
        (assemble(|a| a.senduipi(R16)), "f3d590c7f0"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn memory_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.cldemote(ptr(RAX.into()))), "0f1c00"),
        (
            assemble(|a| a.cldemote(ptr(R13 + R14 * 8 + 0x20))),
            "430f1c44f520",
        ),
        (assemble(|a| a.clwb(ptr(RAX.into()))), "660fae30"),
        (
            assemble(|a| a.clwb(ptr(R13 + R14 * 8 + 0x20))),
            "66430fae74f520",
        ),
        (assemble(|a| a.movrs(AL, ptr(RAX.into()))), "0f388a00"),
        (assemble(|a| a.movrs(AX, ptr(RBX + 0x10))), "660f388b4310"),
        (assemble(|a| a.movrs(EAX, ptr(R8.into()))), "410f388b00"),
        (
            assemble(|a| a.movrs(RAX, ptr(R13 + R14 * 8 + 0x20))),
            "4b0f388b44f520",
        ),
        (assemble(|a| a.movrs(R8, ptr(RAX.into()))), "4c0f388b00"),
        (assemble(|a| a.movrs(XMM1, ptr(RAX.into()))), "0f388b08"),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, decode_hex(expected));
    }
}

#[test]
fn invalid_operand_classes_match_xbyak_contracts() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.rdrand(AL), Err(Error::BadSizeOfRegister));
    assert_eq!(asm.rdfsbase(AX), Err(Error::BadSizeOfRegister));
    assert_eq!(asm.senduipi(EAX), Err(Error::BadSizeOfRegister));
    assert_eq!(asm.movrs(R16, ptr(RAX.into())), Err(Error::CantUseRex2));
}
