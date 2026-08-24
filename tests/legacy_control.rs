use rxbyak::*;

fn assemble(f: impl FnOnce(&mut CodeAssembler) -> Result<()>) -> Vec<u8> {
    let mut asm = CodeAssembler::new(4096).unwrap();
    f(&mut asm).unwrap();
    asm.code().to_vec()
}

#[test]
fn port_io_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.in_(AL, DX)), vec![0xec]),
        (assemble(|a| a.in_(AX, DX)), vec![0x66, 0xed]),
        (assemble(|a| a.in_(EAX, DX)), vec![0xed]),
        (assemble(|a| a.in_imm(AL, 0x7f)), vec![0xe4, 0x7f]),
        (assemble(|a| a.in_imm(AX, 0x7f)), vec![0x66, 0xe5, 0x7f]),
        (assemble(|a| a.in_imm(EAX, 0x7f)), vec![0xe5, 0x7f]),
        (assemble(|a| a.out_(DX, AL)), vec![0xee]),
        (assemble(|a| a.out_(DX, AX)), vec![0x66, 0xef]),
        (assemble(|a| a.out_(DX, EAX)), vec![0xef]),
        (assemble(|a| a.out_imm(0x7f, AL)), vec![0xe6, 0x7f]),
        (assemble(|a| a.out_imm(0x7f, AX)), vec![0x66, 0xe7, 0x7f]),
        (assemble(|a| a.out_imm(0x7f, EAX)), vec![0xe7, 0x7f]),
    ];

    for (actual, expected) in cases {
        assert_eq!(actual, expected);
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.in_(BL, DX), Err(Error::BadCombination));
    assert_eq!(asm.in_(AL, EDX), Err(Error::BadCombination));
}

#[test]
fn scalar_control_encodings_match_xbyak_7_40() {
    assert_eq!(
        assemble(|a| {
            a.int_(0x80)?;
            a.outsb()?;
            a.outsd()?;
            a.outsw()?;
            a.retf()?;
            a.retf_imm(0x1234)?;
            a.jmpabs(0x0123_4567_89ab_cdef)
        }),
        [
            0xcd, 0x80, 0x6e, 0x6f, 0x66, 0x6f, 0xcb, 0xca, 0x34, 0x12, 0xd5, 0x00, 0xa1, 0xef,
            0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
        ]
    );
    assert_eq!(assemble(|a| a.ret_imm(0)), [0xc3]);
    assert_eq!(assemble(|a| a.retf_imm(0)), [0xcb]);
}

#[test]
fn load_segment_encodings_match_xbyak_7_40() {
    let cases = [
        (assemble(|a| a.lfs(AX, ptr(RAX.into()))), "660fb400"),
        (assemble(|a| a.lgs(EAX, ptr(R8 + 0x10))), "410fb54010"),
        (
            assemble(|a| a.lss(RAX, ptr(R13 + R14 * 8 + 0x20))),
            "4b0fb244f520",
        ),
        (assemble(|a| a.lfs(R16, ptr(RAX.into()))), "d5c80fb400"),
    ];

    for (actual, expected) in cases {
        let expected = expected
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    let mut asm = CodeAssembler::new(4096).unwrap();
    assert_eq!(asm.lfs(AL, ptr(RAX.into())), Err(Error::BadSizeOfRegister));
}

#[test]
fn short_only_label_branches_match_xbyak_7_40() {
    assert_eq!(
        assemble(|a| {
            let label = a.create_label();
            a.loop_(&label)?;
            a.nop()?;
            a.bind(&label)
        }),
        [0xe2, 0x01, 0x90]
    );
    assert_eq!(
        assemble(|a| {
            let label = a.create_label();
            a.bind(&label)?;
            a.nop()?;
            a.loope(&label)
        }),
        [0x90, 0xe1, 0xfd]
    );
    assert_eq!(
        assemble(|a| {
            let label = a.create_label();
            a.loopne(&label)?;
            a.nop()?;
            a.bind(&label)
        }),
        [0xe0, 0x01, 0x90]
    );
    assert_eq!(
        assemble(|a| {
            let label = a.create_label();
            a.jecxz(&label)?;
            a.nop()?;
            a.bind(&label)
        }),
        [0x67, 0xe3, 0x01, 0x90]
    );
    assert_eq!(
        assemble(|a| {
            let label = a.create_label();
            a.bind(&label)?;
            a.nop()?;
            a.jrcxz(&label)
        }),
        [0x90, 0xe3, 0xfd]
    );
}

#[test]
fn short_only_forward_branch_rejects_a_far_label() {
    let mut asm = CodeAssembler::new(4096).unwrap();
    let label = asm.create_label();
    asm.loop_(&label).unwrap();
    for _ in 0..128 {
        asm.nop().unwrap();
    }
    assert_eq!(asm.bind(&label), Err(Error::LabelIsTooFar));
}
