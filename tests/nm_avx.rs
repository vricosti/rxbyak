/// AVX/AVX2 instruction NASM conformance tests (VEX 3-operand forms).

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
// AVX arithmetic (3-operand): vaddps/pd/ss/sd, vsubps/pd/ss/sd, etc.
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_arith_xmm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps",  |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd",  |a, d, s1, s2| a.vaddpd(d, s1, s2)),
        ("vaddss",  |a, d, s1, s2| a.vaddss(d, s1, s2)),
        ("vaddsd",  |a, d, s1, s2| a.vaddsd(d, s1, s2)),
        ("vsubps",  |a, d, s1, s2| a.vsubps(d, s1, s2)),
        ("vsubpd",  |a, d, s1, s2| a.vsubpd(d, s1, s2)),
        ("vsubss",  |a, d, s1, s2| a.vsubss(d, s1, s2)),
        ("vsubsd",  |a, d, s1, s2| a.vsubsd(d, s1, s2)),
        ("vmulps",  |a, d, s1, s2| a.vmulps(d, s1, s2)),
        ("vmulpd",  |a, d, s1, s2| a.vmulpd(d, s1, s2)),
        ("vmulss",  |a, d, s1, s2| a.vmulss(d, s1, s2)),
        ("vmulsd",  |a, d, s1, s2| a.vmulsd(d, s1, s2)),
        ("vdivps",  |a, d, s1, s2| a.vdivps(d, s1, s2)),
        ("vdivpd",  |a, d, s1, s2| a.vdivpd(d, s1, s2)),
        ("vdivss",  |a, d, s1, s2| a.vdivss(d, s1, s2)),
        ("vdivsd",  |a, d, s1, s2| a.vdivsd(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM15, "xmm15"),
        (XMM0, "xmm0", XMM7, "xmm7", XMM8, "xmm8"),
    ];

    for &(name, op_fn) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_avx_arith_ymm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps",  |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd",  |a, d, s1, s2| a.vaddpd(d, s1, s2)),
        ("vsubps",  |a, d, s1, s2| a.vsubps(d, s1, s2)),
        ("vmulps",  |a, d, s1, s2| a.vmulps(d, s1, s2)),
        ("vdivps",  |a, d, s1, s2| a.vdivps(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
        (YMM8, "ymm8", YMM9, "ymm9", YMM15, "ymm15"),
    ];

    for &(name, op_fn) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_avx_arith_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vaddps xmm, xmm, [mem]
    for (addr, nasm_mem) in mems128() {
        let asm = format!("vaddps xmm0, xmm1, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(XMM0, XMM1, addr))));
    }
    // vaddps ymm, ymm, [mem]
    for (addr, nasm_mem) in mems256() {
        let asm = format!("vaddps ymm0, ymm1, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(YMM0, YMM1, addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX logical: vandps/pd, vorps/pd, vxorps/pd, vandnps/pd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_logic() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vandps",  |a, d, s1, s2| a.vandps(d, s1, s2)),
        ("vandpd",  |a, d, s1, s2| a.vandpd(d, s1, s2)),
        ("vorps",   |a, d, s1, s2| a.vorps(d, s1, s2)),
        ("vorpd",   |a, d, s1, s2| a.vorpd(d, s1, s2)),
        ("vxorps",  |a, d, s1, s2| a.vxorps(d, s1, s2)),
        ("vxorpd",  |a, d, s1, s2| a.vxorpd(d, s1, s2)),
        ("vandnps", |a, d, s1, s2| a.vandnps(d, s1, s2)),
        ("vandnpd", |a, d, s1, s2| a.vandnpd(d, s1, s2)),
    ];

    let xmm_triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM15, "xmm15"),
    ];
    let ymm_triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
        (YMM8, "ymm8", YMM9, "ymm9", YMM15, "ymm15"),
    ];

    for &(name, op_fn) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in xmm_triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
        for &(d, dn, s1, s1n, s2, s2n) in ymm_triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX mov: vmovaps/ups/apd/upd/dqa/dqu
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_mov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // reg, reg (xmm)
    let mov_ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("vmovaps", |a, d, s| a.vmovaps(d, s)),
        ("vmovups", |a, d, s| a.vmovups(d, s)),
        ("vmovapd", |a, d, s| a.vmovapd(d, s)),
        ("vmovupd", |a, d, s| a.vmovupd(d, s)),
        ("vmovdqa", |a, d, s| a.vmovdqa(d, s)),
        ("vmovdqu", |a, d, s| a.vmovdqu(d, s)),
    ];

    let xpairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM8, "xmm8", XMM15, "xmm15"),
    ];
    let ypairs: &[(Reg, &str, Reg, &str)] = &[
        (YMM0, "ymm0", YMM1, "ymm1"),
        (YMM8, "ymm8", YMM15, "ymm15"),
    ];

    for &(name, op_fn) in mov_ops {
        for &(dst, dn, src, sn) in xpairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
        for &(dst, dn, src, sn) in ypairs {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    // load/store with memory
    let load_ops: &[(&str, fn(&mut CodeAssembler, Reg, Address) -> Result<()>)] = &[
        ("vmovaps", |a, d, m| a.vmovaps(d, m)),
        ("vmovups", |a, d, m| a.vmovups(d, m)),
        ("vmovdqa", |a, d, m| a.vmovdqa(d, m)),
    ];
    for &(name, op_fn) in load_ops {
        for (addr, nasm_mem) in mems128() {
            let asm = format!("{} xmm0, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, XMM0, addr))));
        }
        for (addr, nasm_mem) in mems256() {
            let asm = format!("{} ymm0, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, YMM0, addr))));
        }
    }

    let store_ops: &[(&str, fn(&mut CodeAssembler, Address, Reg) -> Result<()>)] = &[
        ("vmovaps", |a, m, s| a.vmovaps(m, s)),
        ("vmovups", |a, m, s| a.vmovups(m, s)),
        ("vmovdqa", |a, m, s| a.vmovdqa(m, s)),
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
// AVX integer: vpaddd, vpsubd, vpand, vpor, vpxor
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_int() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vpaddd", |a, d, s1, s2| a.vpaddd(d, s1, s2)),
        ("vpsubd", |a, d, s1, s2| a.vpsubd(d, s1, s2)),
        ("vpand",  |a, d, s1, s2| a.vpand(d, s1, s2)),
        ("vpor",   |a, d, s1, s2| a.vpor(d, s1, s2)),
        ("vpxor",  |a, d, s1, s2| a.vpxor(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM15, "xmm15"),
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
        (YMM8, "ymm8", YMM9, "ymm9", YMM15, "ymm15"),
    ];

    for &(name, op_fn) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    // with memory
    for (addr, nasm_mem) in mems128() {
        let asm = format!("vpaddd xmm0, xmm1, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vpaddd(XMM0, XMM1, addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX shuffle: vpshufd, vshufps/pd, vpshufb, vpalignr
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_shuffle() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vpshufd xmm, xmm, imm8 (2-op VEX)
    insns.push(("vpshufd xmm0, xmm1, 0x1b".into(), Box::new(|a: &mut CodeAssembler| a.vpshufd(XMM0, XMM1, 0x1B))));
    insns.push(("vpshufd xmm8, xmm15, 0xff".into(), Box::new(|a: &mut CodeAssembler| a.vpshufd(XMM8, XMM15, 0xFF))));
    insns.push(("vpshufd ymm0, ymm1, 0xe4".into(), Box::new(|a: &mut CodeAssembler| a.vpshufd(YMM0, YMM1, 0xE4))));

    // vshufps xmm, xmm, xmm, imm8 (3-op)
    insns.push(("vshufps xmm0, xmm1, xmm2, 0xe4".into(), Box::new(|a: &mut CodeAssembler| a.vshufps(XMM0, XMM1, XMM2, 0xE4))));
    insns.push(("vshufps ymm0, ymm1, ymm2, 0x1b".into(), Box::new(|a: &mut CodeAssembler| a.vshufps(YMM0, YMM1, YMM2, 0x1B))));
    insns.push(("vshufps xmm8, xmm9, xmm15, 0x00".into(), Box::new(|a: &mut CodeAssembler| a.vshufps(XMM8, XMM9, XMM15, 0x00))));

    // vshufpd
    insns.push(("vshufpd xmm0, xmm1, xmm2, 0x01".into(), Box::new(|a: &mut CodeAssembler| a.vshufpd(XMM0, XMM1, XMM2, 0x01))));

    // vpshufb xmm, xmm, xmm (3-op)
    insns.push(("vpshufb xmm0, xmm1, xmm2".into(), Box::new(|a: &mut CodeAssembler| a.vpshufb(XMM0, XMM1, XMM2))));
    insns.push(("vpshufb xmm8, xmm9, xmm15".into(), Box::new(|a: &mut CodeAssembler| a.vpshufb(XMM8, XMM9, XMM15))));

    // vpalignr xmm, xmm, xmm, imm8 (3-op)
    insns.push(("vpalignr xmm0, xmm1, xmm2, 4".into(), Box::new(|a: &mut CodeAssembler| a.vpalignr(XMM0, XMM1, XMM2, 4))));
    insns.push(("vpalignr xmm8, xmm9, xmm15, 8".into(), Box::new(|a: &mut CodeAssembler| a.vpalignr(XMM8, XMM9, XMM15, 8))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX FMA: vfmadd/vfmsub/vfnmadd/vfnmsub (132/213/231 x ps/pd/ss/sd)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_fma() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let fma_ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vfmadd132ps",  |a, d, s1, s2| a.vfmadd132ps(d, s1, s2)),
        ("vfmadd213ps",  |a, d, s1, s2| a.vfmadd213ps(d, s1, s2)),
        ("vfmadd231ps",  |a, d, s1, s2| a.vfmadd231ps(d, s1, s2)),
        ("vfmadd132pd",  |a, d, s1, s2| a.vfmadd132pd(d, s1, s2)),
        ("vfmadd213pd",  |a, d, s1, s2| a.vfmadd213pd(d, s1, s2)),
        ("vfmadd231pd",  |a, d, s1, s2| a.vfmadd231pd(d, s1, s2)),
        ("vfmadd132ss",  |a, d, s1, s2| a.vfmadd132ss(d, s1, s2)),
        ("vfmadd213ss",  |a, d, s1, s2| a.vfmadd213ss(d, s1, s2)),
        ("vfmadd231ss",  |a, d, s1, s2| a.vfmadd231ss(d, s1, s2)),
        ("vfmadd132sd",  |a, d, s1, s2| a.vfmadd132sd(d, s1, s2)),
        ("vfmadd213sd",  |a, d, s1, s2| a.vfmadd213sd(d, s1, s2)),
        ("vfmadd231sd",  |a, d, s1, s2| a.vfmadd231sd(d, s1, s2)),
        ("vfmsub132ps",  |a, d, s1, s2| a.vfmsub132ps(d, s1, s2)),
        ("vfmsub213ps",  |a, d, s1, s2| a.vfmsub213ps(d, s1, s2)),
        ("vfmsub231ps",  |a, d, s1, s2| a.vfmsub231ps(d, s1, s2)),
        ("vfnmadd132ps", |a, d, s1, s2| a.vfnmadd132ps(d, s1, s2)),
        ("vfnmadd213ps", |a, d, s1, s2| a.vfnmadd213ps(d, s1, s2)),
        ("vfnmadd231ps", |a, d, s1, s2| a.vfnmadd231ps(d, s1, s2)),
        ("vfnmsub132ps", |a, d, s1, s2| a.vfnmsub132ps(d, s1, s2)),
        ("vfnmsub213ps", |a, d, s1, s2| a.vfnmsub213ps(d, s1, s2)),
        ("vfnmsub231ps", |a, d, s1, s2| a.vfnmsub231ps(d, s1, s2)),
    ];

    let xmm_triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM15, "xmm15"),
    ];

    for &(name, op_fn) in fma_ops {
        for &(d, dn, s1, s1n, s2, s2n) in xmm_triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    // YMM variants (packed only)
    let ymm_triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
    ];
    let fma_ymm: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vfmadd132ps", |a, d, s1, s2| a.vfmadd132ps(d, s1, s2)),
        ("vfmadd213ps", |a, d, s1, s2| a.vfmadd213ps(d, s1, s2)),
        ("vfmadd231ps", |a, d, s1, s2| a.vfmadd231ps(d, s1, s2)),
        ("vfmadd132pd", |a, d, s1, s2| a.vfmadd132pd(d, s1, s2)),
    ];
    for &(name, op_fn) in fma_ymm {
        for &(d, dn, s1, s1n, s2, s2n) in ymm_triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX extract/insert: vextractf128/i128, vpextrb/w/d/q, vextractps
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_extract() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vextractf128 xmm, ymm, imm8
    insns.push(("vextractf128 xmm0, ymm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.vextractf128(XMM0, YMM1, 1))));
    insns.push(("vextractf128 xmm8, ymm15, 0".into(), Box::new(|a: &mut CodeAssembler| a.vextractf128(XMM8, YMM15, 0))));
    // vextracti128 xmm, ymm, imm8
    insns.push(("vextracti128 xmm0, ymm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.vextracti128(XMM0, YMM1, 1))));

    // vpextrb r32, xmm, imm8
    insns.push(("vpextrb eax, xmm0, 3".into(), Box::new(|a: &mut CodeAssembler| a.vpextrb(EAX, XMM0, 3))));
    insns.push(("vpextrb r8d, xmm8, 5".into(), Box::new(|a: &mut CodeAssembler| a.vpextrb(R8D, XMM8, 5))));
    // vpextrw r32, xmm, imm8
    insns.push(("vpextrw eax, xmm0, 3".into(), Box::new(|a: &mut CodeAssembler| a.vpextrw(EAX, XMM0, 3))));
    // vpextrd r32, xmm, imm8
    insns.push(("vpextrd eax, xmm0, 2".into(), Box::new(|a: &mut CodeAssembler| a.vpextrd(EAX, XMM0, 2))));
    insns.push(("vpextrd r8d, xmm8, 1".into(), Box::new(|a: &mut CodeAssembler| a.vpextrd(R8D, XMM8, 1))));
    // vpextrq r64, xmm, imm8
    insns.push(("vpextrq rax, xmm0, 1".into(), Box::new(|a: &mut CodeAssembler| a.vpextrq(RAX, XMM0, 1))));
    // vextractps r32, xmm, imm8
    insns.push(("vextractps eax, xmm0, 2".into(), Box::new(|a: &mut CodeAssembler| a.vextractps(EAX, XMM0, 2))));

    // vpinsrb/w/d/q
    insns.push(("vpinsrb xmm0, xmm1, eax, 3".into(), Box::new(|a: &mut CodeAssembler| a.vpinsrb(XMM0, XMM1, EAX, 3))));
    insns.push(("vpinsrw xmm0, xmm1, eax, 3".into(), Box::new(|a: &mut CodeAssembler| a.vpinsrw(XMM0, XMM1, EAX, 3))));
    insns.push(("vpinsrd xmm0, xmm1, eax, 2".into(), Box::new(|a: &mut CodeAssembler| a.vpinsrd(XMM0, XMM1, EAX, 2))));
    insns.push(("vpinsrq xmm0, xmm1, rax, 1".into(), Box::new(|a: &mut CodeAssembler| a.vpinsrq(XMM0, XMM1, RAX, 1))));
    // vinsertps
    insns.push(("vinsertps xmm0, xmm1, xmm2, 0x10".into(), Box::new(|a: &mut CodeAssembler| a.vinsertps(XMM0, XMM1, XMM2, 0x10))));
    // vinsertf128/i128
    insns.push(("vinsertf128 ymm0, ymm1, xmm2, 1".into(), Box::new(|a: &mut CodeAssembler| a.vinsertf128(YMM0, YMM1, XMM2, 1))));
    insns.push(("vinserti128 ymm0, ymm1, xmm2, 1".into(), Box::new(|a: &mut CodeAssembler| a.vinserti128(YMM0, YMM1, XMM2, 1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX broadcast: vbroadcastss/sd/f128
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_broadcast() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vbroadcastss xmm, xmm (AVX2)
    insns.push(("vbroadcastss xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.vbroadcastss(XMM0, XMM1))));
    insns.push(("vbroadcastss ymm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.vbroadcastss(YMM0, XMM1))));
    insns.push(("vbroadcastss xmm8, xmm15".into(), Box::new(|a: &mut CodeAssembler| a.vbroadcastss(XMM8, XMM15))));

    // vbroadcastss xmm, [mem]
    for (addr, nasm_mem) in mems32() {
        let asm = format!("vbroadcastss xmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vbroadcastss(XMM0, addr))));
    }
    // vbroadcastsd ymm, xmm (AVX2)
    insns.push(("vbroadcastsd ymm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.vbroadcastsd(YMM0, XMM1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX misc: vzeroall, vzeroupper, vmovmskps/pd, vpmovmskb
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_misc() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    insns.push(("vzeroall".into(), Box::new(|a: &mut CodeAssembler| a.vzeroall())));
    insns.push(("vzeroupper".into(), Box::new(|a: &mut CodeAssembler| a.vzeroupper())));

    // vmovmskps r32, xmm/ymm
    insns.push(("vmovmskps eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.vmovmskps(EAX, XMM0))));
    insns.push(("vmovmskps eax, ymm0".into(), Box::new(|a: &mut CodeAssembler| a.vmovmskps(EAX, YMM0))));
    insns.push(("vmovmskps r8d, xmm8".into(), Box::new(|a: &mut CodeAssembler| a.vmovmskps(R8D, XMM8))));
    // vmovmskpd
    insns.push(("vmovmskpd eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.vmovmskpd(EAX, XMM0))));
    insns.push(("vmovmskpd eax, ymm0".into(), Box::new(|a: &mut CodeAssembler| a.vmovmskpd(EAX, YMM0))));
    // vpmovmskb
    insns.push(("vpmovmskb eax, xmm0".into(), Box::new(|a: &mut CodeAssembler| a.vpmovmskb(EAX, XMM0))));
    insns.push(("vpmovmskb eax, ymm0".into(), Box::new(|a: &mut CodeAssembler| a.vpmovmskb(EAX, YMM0))));

    // vmovntps/pd/dq
    for (addr, nasm_mem) in mems128() {
        let asm = format!("vmovntps {}, xmm0", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovntps(a2, XMM0))));
        let asm = format!("vmovntpd {}, xmm0", nasm_mem);
        let a3 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovntpd(a3, XMM0))));
        let asm = format!("vmovntdq {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovntdq(addr, XMM0))));
    }

    // vsqrtps xmm, xmm
    insns.push(("vsqrtps xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.vsqrtps(XMM0, XMM1))));
    insns.push(("vsqrtps ymm0, ymm1".into(), Box::new(|a: &mut CodeAssembler| a.vsqrtps(YMM0, YMM1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// AVX special: vmovss/vmovsd, vmovhps/lps/hpd/lpd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_avx_special() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vmovss xmm, xmm, xmm (3-operand)
    insns.push(("vmovss xmm0, xmm1, xmm2".into(), Box::new(|a: &mut CodeAssembler| a.vmovss(XMM0, XMM1, Some(XMM2)))));
    insns.push(("vmovss xmm8, xmm9, xmm15".into(), Box::new(|a: &mut CodeAssembler| a.vmovss(XMM8, XMM9, Some(XMM15)))));
    // vmovsd xmm, xmm, xmm (3-operand)
    insns.push(("vmovsd xmm0, xmm1, xmm2".into(), Box::new(|a: &mut CodeAssembler| a.vmovsd(XMM0, XMM1, Some(XMM2)))));
    // vmovss xmm, [mem] (load, 2-operand): dst=xmm, src1=mem, src2=None
    for (addr, nasm_mem) in mems32() {
        let asm = format!("vmovss xmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovss(XMM0, addr, None))));
    }
    // vmovss [mem], xmm (store): dst=mem, src1=xmm, src2=None
    for (addr, nasm_mem) in mems32() {
        let asm = format!("vmovss {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovss(addr, XMM0, None))));
    }
    // vmovsd xmm, [mem] (load)
    for (addr, nasm_mem) in mems64() {
        let asm = format!("vmovsd xmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovsd(XMM0, addr, None))));
    }

    // vmovhps xmm, xmm, [mem] (3-op load)
    for (addr, nasm_mem) in mems64() {
        let asm = format!("vmovhps xmm0, xmm1, {}", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovhps_load(XMM0, XMM1, a2))));
        // vmovhps [mem], xmm (store)
        let asm = format!("vmovhps {}, xmm0", nasm_mem);
        let a3 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovhps_store(a3, XMM0))));
        // vmovlps
        let asm = format!("vmovlps xmm0, xmm1, {}", nasm_mem);
        let a4 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovlps_load(XMM0, XMM1, a4))));
        let asm = format!("vmovlps {}, xmm0", nasm_mem);
        let a5 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovlps_store(a5, XMM0))));
        // vmovhpd
        let asm = format!("vmovhpd xmm0, xmm1, {}", nasm_mem);
        let a6 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovhpd_load(XMM0, XMM1, a6))));
        let asm = format!("vmovhpd {}, xmm0", nasm_mem);
        let a7 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovhpd_store(a7, XMM0))));
        // vmovlpd
        let asm = format!("vmovlpd xmm0, xmm1, {}", nasm_mem);
        let a8 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovlpd_load(XMM0, XMM1, a8))));
        let asm = format!("vmovlpd {}, xmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovlpd_store(addr, XMM0))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
