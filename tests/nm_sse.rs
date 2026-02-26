/// SSE/SSE2/SSE3/SSSE3/SSE4 instruction NASM conformance tests.

mod common;

use common::*;
use rxbyak::*;

macro_rules! skip_if_no_nasm {
    () => {
        match find_nasm() {
            Some(p) => p,
            None => {
                eprintln!("NASM not found, skipping test");
                return;
            }
        }
    };
}

type NmPair = (String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>);

// ═══════════════════════════════════════════════════════════════════
// SSE arithmetic: addps/pd/ss/sd, subps/pd/ss/sd, mulps/pd/ss/sd,
//                 divps/pd/ss/sd, sqrtps/pd/ss/sd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_arith_rr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("addps", |a, d, s| a.addps(d, s)),
        ("addpd", |a, d, s| a.addpd(d, s)),
        ("addss", |a, d, s| a.addss(d, s)),
        ("addsd", |a, d, s| a.addsd(d, s)),
        ("subps", |a, d, s| a.subps(d, s)),
        ("subpd", |a, d, s| a.subpd(d, s)),
        ("subss", |a, d, s| a.subss(d, s)),
        ("subsd", |a, d, s| a.subsd(d, s)),
        ("mulps", |a, d, s| a.mulps(d, s)),
        ("mulpd", |a, d, s| a.mulpd(d, s)),
        ("mulss", |a, d, s| a.mulss(d, s)),
        ("mulsd", |a, d, s| a.mulsd(d, s)),
        ("divps", |a, d, s| a.divps(d, s)),
        ("divpd", |a, d, s| a.divpd(d, s)),
        ("divss", |a, d, s| a.divss(d, s)),
        ("divsd", |a, d, s| a.divsd(d, s)),
        ("sqrtps", |a, d, s| a.sqrtps(d, s)),
        ("sqrtpd", |a, d, s| a.sqrtpd(d, s)),
        ("sqrtss", |a, d, s| a.sqrtss(d, s)),
        ("sqrtsd", |a, d, s| a.sqrtsd(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM7, "xmm7", XMM3, "xmm3"),
        (XMM8, "xmm8", XMM15, "xmm15"),
        (XMM0, "xmm0", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_sse_arith_rm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Address) -> Result<()>)] = &[
        ("addps", |a, d, m| a.addps(d, m)),
        ("addpd", |a, d, m| a.addpd(d, m)),
        ("subps", |a, d, m| a.subps(d, m)),
        ("mulps", |a, d, m| a.mulps(d, m)),
        ("divps", |a, d, m| a.divps(d, m)),
    ];

    for &(name, op_fn) in ops {
        for (addr, nasm_mem) in mems128() {
            let asm = format!("{} xmm0, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, XMM0, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE logical: andps/pd, orps/pd, xorps/pd, andnps/pd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_logic() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("andps",  |a, d, s| a.andps(d, s)),
        ("andpd",  |a, d, s| a.andpd(d, s)),
        ("orps",   |a, d, s| a.orps(d, s)),
        ("orpd",   |a, d, s| a.orpd(d, s)),
        ("xorps",  |a, d, s| a.xorps(d, s)),
        ("xorpd",  |a, d, s| a.xorpd(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM7, "xmm7", XMM3, "xmm3"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE mov: movaps/ups/apd/upd/dqa/dqu
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_mov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // reg, reg
    let mov_ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("movaps", |a, d, s| a.movaps(d, s)),
        ("movups", |a, d, s| a.movups(d, s)),
        ("movapd", |a, d, s| a.movapd(d, s)),
        ("movupd", |a, d, s| a.movupd(d, s)),
        ("movdqa", |a, d, s| a.movdqa(d, s)),
        ("movdqu", |a, d, s| a.movdqu(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
        (XMM5, "xmm5", XMM12, "xmm12"),
    ];

    for &(name, op_fn) in mov_ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    // reg, mem (load)
    let load_ops: &[(&str, fn(&mut CodeAssembler, Reg, Address) -> Result<()>)] = &[
        ("movaps", |a, d, m| a.movaps(d, m)),
        ("movups", |a, d, m| a.movups(d, m)),
        ("movapd", |a, d, m| a.movapd(d, m)),
        ("movdqa", |a, d, m| a.movdqa(d, m)),
        ("movdqu", |a, d, m| a.movdqu(d, m)),
    ];

    for &(name, op_fn) in load_ops {
        for (addr, nasm_mem) in mems128() {
            let asm = format!("{} xmm0, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, XMM0, addr))));
        }
    }

    // mem, reg (store)
    let store_ops: &[(&str, fn(&mut CodeAssembler, Address, Reg) -> Result<()>)] = &[
        ("movaps", |a, m, s| a.movaps(m, s)),
        ("movups", |a, m, s| a.movups(m, s)),
        ("movapd", |a, m, s| a.movapd(m, s)),
        ("movdqa", |a, m, s| a.movdqa(m, s)),
        ("movdqu", |a, m, s| a.movdqu(m, s)),
    ];

    for &(name, op_fn) in store_ops {
        for (addr, nasm_mem) in mems128() {
            let asm = format!("{} {}, xmm0", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, XMM0))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// movd / movq (XMM <-> GPR/mem)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_movd_movq() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // movd xmm, r32
    for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(gpr, gn) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
            let asm = format!("movd {}, {}", xn, gn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movd(xmm, gpr))));
        }
    }
    // movd r32, xmm
    for &(gpr, gn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
            let asm = format!("movd {}, {}", gn, xn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movd(gpr, xmm))));
        }
    }
    // movq xmm, r64
    for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(gpr, gn) in &[(RAX, "rax"), (R8, "r8")] {
            let asm = format!("movq {}, {}", xn, gn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movq(xmm, gpr))));
        }
    }
    // movq r64, xmm
    for &(gpr, gn) in &[(RAX, "rax"), (R8, "r8")] {
        for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
            let asm = format!("movq {}, {}", gn, xn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movq(gpr, xmm))));
        }
    }
    // movq xmm, xmm
    insns.push(("movq xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.movq(XMM0, XMM1))));
    insns.push(("movq xmm8, xmm15".into(), Box::new(|a: &mut CodeAssembler| a.movq(XMM8, XMM15))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// movss / movsd (special: 2-op reg,mem and 3-op reg,reg)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_movss_movsd() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // movss xmm, xmm
    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];
    for &(dst, dn, src, sn) in pairs {
        let asm = format!("movss {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movss(dst, src))));
        let asm = format!("movsd {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsd(dst, src))));
    }

    // movss xmm, [mem] (load)
    for (addr, nasm_mem) in mems32() {
        let asm = format!("movss xmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movss(XMM0, addr))));
    }
    // movss [mem], xmm (store)
    for (addr, nasm_mem) in mems32() {
        let asm = format!("movss {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movss(addr, XMM0))));
    }
    // movsd xmm, [mem] (load)
    for (addr, nasm_mem) in mems64() {
        let asm = format!("movsd xmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsd(XMM0, addr))));
    }
    // movsd [mem], xmm (store)
    for (addr, nasm_mem) in mems64() {
        let asm = format!("movsd {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsd(addr, XMM0))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// movhps/lps/hpd/lpd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_movhlps() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    for (addr, nasm_mem) in mems64() {
        // movhps xmm, [mem]
        let asm = format!("movhps xmm0, {}", nasm_mem);
        let a1 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movhps_load(XMM0, a1))));
        // movhps [mem], xmm
        let asm = format!("movhps {}, xmm0", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movhps_store(a2, XMM0))));
        // movlps xmm, [mem]
        let asm = format!("movlps xmm0, {}", nasm_mem);
        let a3 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movlps_load(XMM0, a3))));
        // movlps [mem], xmm
        let asm = format!("movlps {}, xmm0", nasm_mem);
        let a4 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movlps_store(a4, XMM0))));
        // movhpd / movlpd
        let asm = format!("movhpd xmm0, {}", nasm_mem);
        let a5 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movhpd_load(XMM0, a5))));
        let asm = format!("movhpd {}, xmm0", nasm_mem);
        let a6 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movhpd_store(a6, XMM0))));
        let asm = format!("movlpd xmm0, {}", nasm_mem);
        let a7 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movlpd_load(XMM0, a7))));
        let asm = format!("movlpd {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movlpd_store(addr, XMM0))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE integer: paddb..paddq, psubb..psubq, pmullw, pmaddwd, pcmpeq/gt, pand/or/xor
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_int() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("paddb",   |a, d, s| a.paddb(d, s)),
        ("paddw",   |a, d, s| a.paddw(d, s)),
        ("paddd",   |a, d, s| a.paddd(d, s)),
        ("paddq",   |a, d, s| a.paddq(d, s)),
        ("psubb",   |a, d, s| a.psubb(d, s)),
        ("psubw",   |a, d, s| a.psubw(d, s)),
        ("psubd",   |a, d, s| a.psubd(d, s)),
        ("psubq",   |a, d, s| a.psubq(d, s)),
        ("pmullw",  |a, d, s| a.pmullw(d, s)),
        ("pmaddwd", |a, d, s| a.pmaddwd(d, s)),
        ("pcmpeqb", |a, d, s| a.pcmpeqb(d, s)),
        ("pcmpeqw", |a, d, s| a.pcmpeqw(d, s)),
        ("pcmpeqd", |a, d, s| a.pcmpeqd(d, s)),
        ("pcmpgtb", |a, d, s| a.pcmpgtb(d, s)),
        ("pcmpgtw", |a, d, s| a.pcmpgtw(d, s)),
        ("pcmpgtd", |a, d, s| a.pcmpgtd(d, s)),
        ("pand",    |a, d, s| a.pand(d, s)),
        ("por",     |a, d, s| a.por(d, s)),
        ("pxor",    |a, d, s| a.pxor(d, s)),
        ("pandn",   |a, d, s| a.pandn(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM7, "xmm7", XMM3, "xmm3"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE pack/unpack
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_pack() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("punpcklbw",  |a, d, s| a.punpcklbw(d, s)),
        ("punpcklwd",  |a, d, s| a.punpcklwd(d, s)),
        ("punpckldq",  |a, d, s| a.punpckldq(d, s)),
        ("punpcklqdq", |a, d, s| a.punpcklqdq(d, s)),
        ("punpckhbw",  |a, d, s| a.punpckhbw(d, s)),
        ("punpckhwd",  |a, d, s| a.punpckhwd(d, s)),
        ("punpckhdq",  |a, d, s| a.punpckhdq(d, s)),
        ("punpckhqdq", |a, d, s| a.punpckhqdq(d, s)),
        ("packsswb",   |a, d, s| a.packsswb(d, s)),
        ("packssdw",   |a, d, s| a.packssdw(d, s)),
        ("packuswb",   |a, d, s| a.packuswb(d, s)),
        ("packusdw",   |a, d, s| a.packusdw(d, s)),
        ("unpcklps",   |a, d, s| a.unpcklps(d, s)),
        ("unpckhps",   |a, d, s| a.unpckhps(d, s)),
        ("unpcklpd",   |a, d, s| a.unpcklpd(d, s)),
        ("unpckhpd",   |a, d, s| a.unpckhpd(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE shuffle: pshufd, shufps/pd, pshufb, palignr
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_shuffle() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // pshufd xmm, xmm, imm8
    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];
    for &(dst, dn, src, sn) in pairs {
        for imm in [0x00u8, 0x1B, 0xFF] {
            let asm = format!("pshufd {}, {}, 0x{:02x}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pshufd(dst, src, imm))));
        }
    }
    // shufps xmm, xmm, imm8
    for &(dst, dn, src, sn) in pairs {
        for imm in [0x00u8, 0xE4] {
            let asm = format!("shufps {}, {}, 0x{:02x}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shufps(dst, src, imm))));
        }
    }
    // shufpd xmm, xmm, imm8
    for &(dst, dn, src, sn) in pairs {
        let asm = format!("shufpd {}, {}, 0x01", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shufpd(dst, src, 0x01))));
    }
    // pshufb xmm, xmm
    for &(dst, dn, src, sn) in pairs {
        let asm = format!("pshufb {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pshufb(dst, src))));
    }
    // palignr xmm, xmm, imm8
    for &(dst, dn, src, sn) in pairs {
        let asm = format!("palignr {}, {}, 4", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.palignr(dst, src, 4))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE conversion
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_cvt() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // cvtsi2ss xmm, r32/r64
    for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(gpr, gn) in &[(EAX, "eax"), (R8D, "r8d")] {
            let asm = format!("cvtsi2ss {}, {}", xn, gn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.cvtsi2ss(xmm, gpr))));
        }
        for &(gpr, gn) in &[(RAX, "rax"), (R8, "r8")] {
            let asm = format!("cvtsi2ss {}, {}", xn, gn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.cvtsi2ss(xmm, gpr))));
        }
    }
    // cvtsi2sd xmm, r32/r64
    for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(gpr, gn) in &[(EAX, "eax"), (RAX, "rax")] {
            let asm = format!("cvtsi2sd {}, {}", xn, gn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.cvtsi2sd(xmm, gpr))));
        }
    }
    // cvtss2sd / cvtsd2ss
    insns.push(("cvtss2sd xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.cvtss2sd(XMM0, XMM1))));
    insns.push(("cvtsd2ss xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.cvtsd2ss(XMM0, XMM1))));
    insns.push(("cvtss2sd xmm8, xmm15".into(), Box::new(|a: &mut CodeAssembler| a.cvtss2sd(XMM8, XMM15))));

    // cvtps2dq / cvtdq2ps / cvttps2dq
    insns.push(("cvtps2dq xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.cvtps2dq(XMM0, XMM1))));
    insns.push(("cvtdq2ps xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.cvtdq2ps(XMM0, XMM1))));
    insns.push(("cvttps2dq xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.cvttps2dq(XMM0, XMM1))));

    // cvtss2si / cvtsd2si / cvttss2si / cvttsd2si
    for &(gpr, gn) in &[(EAX, "eax"), (RAX, "rax")] {
        insns.push((format!("cvtss2si {}, xmm0", gn), Box::new(move |a: &mut CodeAssembler| a.cvtss2si(gpr, XMM0))));
        insns.push((format!("cvtsd2si {}, xmm0", gn), Box::new(move |a: &mut CodeAssembler| a.cvtsd2si(gpr, XMM0))));
        insns.push((format!("cvttss2si {}, xmm0", gn), Box::new(move |a: &mut CodeAssembler| a.cvttss2si(gpr, XMM0))));
        insns.push((format!("cvttsd2si {}, xmm0", gn), Box::new(move |a: &mut CodeAssembler| a.cvttsd2si(gpr, XMM0))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE compare: comiss/sd, ucomiss/sd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_cmp() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("comiss",  |a, d, s| a.comiss(d, s)),
        ("comisd",  |a, d, s| a.comisd(d, s)),
        ("ucomiss", |a, d, s| a.ucomiss(d, s)),
        ("ucomisd", |a, d, s| a.ucomisd(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE extract/insert: pextrb/w/d/q, pinsrb/w/d/q, extractps, insertps
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_extract_insert() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // pextrb r32, xmm, imm8
    for &(gpr, gn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for &(xmm, xn) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
            let asm = format!("pextrb {}, {}, 3", gn, xn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pextrb(gpr, xmm, 3))));
        }
    }
    // pextrw r32, xmm, imm8
    for &(gpr, gn) in &[(EAX, "eax"), (R8D, "r8d")] {
        let asm = format!("pextrw {}, xmm0, 3", gn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pextrw(gpr, XMM0, 3))));
    }
    // pextrd r32, xmm, imm8
    insns.push(("pextrd eax, xmm0, 2".into(), Box::new(|a: &mut CodeAssembler| a.pextrd(EAX, XMM0, 2))));
    insns.push(("pextrd r8d, xmm8, 1".into(), Box::new(|a: &mut CodeAssembler| a.pextrd(R8D, XMM8, 1))));
    // pextrq r64, xmm, imm8
    insns.push(("pextrq rax, xmm0, 1".into(), Box::new(|a: &mut CodeAssembler| a.pextrq(RAX, XMM0, 1))));
    insns.push(("pextrq r8, xmm8, 0".into(), Box::new(|a: &mut CodeAssembler| a.pextrq(R8, XMM8, 0))));
    // extractps r32, xmm, imm8
    insns.push(("extractps eax, xmm0, 2".into(), Box::new(|a: &mut CodeAssembler| a.extractps(EAX, XMM0, 2))));

    // pinsrb xmm, r32, imm8
    insns.push(("pinsrb xmm0, eax, 3".into(), Box::new(|a: &mut CodeAssembler| a.pinsrb(XMM0, EAX, 3))));
    insns.push(("pinsrb xmm8, r8d, 5".into(), Box::new(|a: &mut CodeAssembler| a.pinsrb(XMM8, R8D, 5))));
    // pinsrw xmm, r32, imm8
    insns.push(("pinsrw xmm0, eax, 3".into(), Box::new(|a: &mut CodeAssembler| a.pinsrw(XMM0, EAX, 3))));
    // pinsrd xmm, r32, imm8
    insns.push(("pinsrd xmm0, eax, 2".into(), Box::new(|a: &mut CodeAssembler| a.pinsrd(XMM0, EAX, 2))));
    insns.push(("pinsrd xmm8, r8d, 1".into(), Box::new(|a: &mut CodeAssembler| a.pinsrd(XMM8, R8D, 1))));
    // pinsrq xmm, r64, imm8
    insns.push(("pinsrq xmm0, rax, 1".into(), Box::new(|a: &mut CodeAssembler| a.pinsrq(XMM0, RAX, 1))));
    // insertps xmm, xmm, imm8
    insns.push(("insertps xmm0, xmm1, 0x10".into(), Box::new(|a: &mut CodeAssembler| a.insertps(XMM0, XMM1, 0x10))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SSE misc: movmskps/pd, pmovmskb, non-temporal
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_sse_misc() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // movmskps r32, xmm
    insns.push(("movmskps eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.movmskps(EAX, XMM0))));
    insns.push(("movmskps eax, xmm8".into(), Box::new(|a: &mut CodeAssembler| a.movmskps(EAX, XMM8))));
    insns.push(("movmskps r8d, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.movmskps(R8D, XMM0))));
    // movmskpd r32, xmm
    insns.push(("movmskpd eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.movmskpd(EAX, XMM0))));
    insns.push(("movmskpd eax, xmm8".into(), Box::new(|a: &mut CodeAssembler| a.movmskpd(EAX, XMM8))));
    // pmovmskb r32, xmm
    insns.push(("pmovmskb eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.pmovmskb(EAX, XMM0))));
    insns.push(("pmovmskb eax, xmm8".into(), Box::new(|a: &mut CodeAssembler| a.pmovmskb(EAX, XMM8))));

    // non-temporal SSE stores
    for (addr, nasm_mem) in mems128() {
        let asm = format!("movntps {}, xmm0", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movntps(a2, XMM0))));
        let asm = format!("movntpd {}, xmm0", nasm_mem);
        let a3 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movntpd(a3, XMM0))));
        let asm = format!("movntdq {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movntdq(addr, XMM0))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
