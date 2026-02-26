/// SSE/AVX instruction tests validated against NASM reference assembler.

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

// ─── SSE 2-operand float (reg, reg) ─────────────────────────────

#[test]
fn test_nasm_sse_float_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

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

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM2, "xmm2", XMM3, "xmm3"),
        (XMM7, "xmm7", XMM0, "xmm0"),
        (XMM8, "xmm8", XMM9, "xmm9"),
        (XMM0, "xmm0", XMM15, "xmm15"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE logic ops ──────────────────────────────────────────────

#[test]
fn test_nasm_sse_logic_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("andps", |a, d, s| a.andps(d, s)),
        ("andpd", |a, d, s| a.andpd(d, s)),
        ("orps", |a, d, s| a.orps(d, s)),
        ("orpd", |a, d, s| a.orpd(d, s)),
        ("xorps", |a, d, s| a.xorps(d, s)),
        ("xorpd", |a, d, s| a.xorpd(d, s)),
    ];

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM0, "xmm0"),
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM5, "xmm5", XMM7, "xmm7"),
        (XMM8, "xmm8", XMM9, "xmm9"),
        (XMM15, "xmm15", XMM0, "xmm0"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE data movement ─────────────────────────────────────────

#[test]
fn test_nasm_sse_mov_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // movaps, movups, movapd, movupd, movdqa, movdqu (reg, reg)
    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("movaps", |a, d, s| a.movaps(d, s)),
        ("movups", |a, d, s| a.movups(d, s)),
        ("movapd", |a, d, s| a.movapd(d, s)),
        ("movupd", |a, d, s| a.movupd(d, s)),
        ("movdqa", |a, d, s| a.movdqa(d, s)),
        ("movdqu", |a, d, s| a.movdqu(d, s)),
    ];

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM7, "xmm7", XMM0, "xmm0"),
        (XMM8, "xmm8", XMM9, "xmm9"),
        (XMM15, "xmm15", XMM0, "xmm0"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE integer ops ────────────────────────────────────────────

#[test]
fn test_nasm_sse_int_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("paddd", |a, d, s| a.paddd(d, s)),
        ("psubd", |a, d, s| a.psubd(d, s)),
        ("pxor", |a, d, s| a.pxor(d, s)),
        ("pand", |a, d, s| a.pand(d, s)),
        ("por", |a, d, s| a.por(d, s)),
    ];

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM0, "xmm0", XMM0, "xmm0"),
        (XMM7, "xmm7", XMM5, "xmm5"),
        (XMM8, "xmm8", XMM9, "xmm9"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE compare ops ────────────────────────────────────────────

#[test]
fn test_nasm_sse_compare() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("comiss", |a, d, s| a.comiss(d, s)),
        ("comisd", |a, d, s| a.comisd(d, s)),
        ("ucomiss", |a, d, s| a.ucomiss(d, s)),
        ("ucomisd", |a, d, s| a.ucomisd(d, s)),
    ];

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM3, "xmm3", XMM7, "xmm7"),
        (XMM8, "xmm8", XMM9, "xmm9"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE convert ops ────────────────────────────────────────────

#[test]
fn test_nasm_sse_convert() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("cvtss2sd", |a, d, s| a.cvtss2sd(d, s)),
        ("cvtsd2ss", |a, d, s| a.cvtsd2ss(d, s)),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
            for &(src, src_name) in &[(XMM1, "xmm1"), (XMM9, "xmm9")] {
                let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
                insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE with memory operands ───────────────────────────────────

#[test]
fn test_nasm_sse_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // addps xmm, [mem]
    for &(xmm, xmm_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(base, base_name) in &[(RAX, "rax"), (RBX, "rbx"), (R8, "r8")] {
            let asm_text = format!("addps {}, oword [{}]", xmm_name, base_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.addps(xmm, xmmword_ptr(base.into()))
            })));
        }
    }

    // movaps [mem], xmm
    for &(xmm, xmm_name) in &[(XMM0, "xmm0"), (XMM15, "xmm15")] {
        for &(base, base_name) in &[(RAX, "rax"), (R8, "r8")] {
            let asm_text = format!("movaps oword [{}], {}", base_name, xmm_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.movaps(xmmword_ptr(base.into()), xmm)
            })));
        }
    }

    // movaps xmm, [mem + disp]
    for &(xmm, xmm_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        let asm_text = format!("movaps {}, oword [rax+0x10]", xmm_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.movaps(xmm, xmmword_ptr(RAX + 0x10))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── movd / movq ────────────────────────────────────────────────

#[test]
fn test_nasm_movd_movq() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // movd xmm, gpr32
    for &(xmm, xmm_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(gpr, gpr_name) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
            let asm_text = format!("movd {}, {}", xmm_name, gpr_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movd(xmm, gpr))));
        }
    }

    // movd gpr32, xmm
    for &(gpr, gpr_name) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
        for &(xmm, xmm_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
            let asm_text = format!("movd {}, {}", gpr_name, xmm_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movd(gpr, xmm))));
        }
    }

    // movq xmm, xmm
    for &(dst, dst_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8")] {
        for &(src, src_name) in &[(XMM1, "xmm1"), (XMM9, "xmm9")] {
            let asm_text = format!("movq {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movq(dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AVX 3-operand float (VEX-encoded) ─────────────────────────

#[test]
fn test_nasm_avx_float_ops_xmm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps", |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd", |a, d, s1, s2| a.vaddpd(d, s1, s2)),
        ("vaddss", |a, d, s1, s2| a.vaddss(d, s1, s2)),
        ("vaddsd", |a, d, s1, s2| a.vaddsd(d, s1, s2)),
        ("vsubps", |a, d, s1, s2| a.vsubps(d, s1, s2)),
        ("vsubpd", |a, d, s1, s2| a.vsubpd(d, s1, s2)),
        ("vmulps", |a, d, s1, s2| a.vmulps(d, s1, s2)),
        ("vmulpd", |a, d, s1, s2| a.vmulpd(d, s1, s2)),
        ("vdivps", |a, d, s1, s2| a.vdivps(d, s1, s2)),
        ("vdivpd", |a, d, s1, s2| a.vdivpd(d, s1, s2)),
        ("vxorps", |a, d, s1, s2| a.vxorps(d, s1, s2)),
        ("vxorpd", |a, d, s1, s2| a.vxorpd(d, s1, s2)),
        ("vandps", |a, d, s1, s2| a.vandps(d, s1, s2)),
        ("vandpd", |a, d, s1, s2| a.vandpd(d, s1, s2)),
        ("vorps", |a, d, s1, s2| a.vorps(d, s1, s2)),
        ("vorpd", |a, d, s1, s2| a.vorpd(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM0, "xmm0", XMM0, "xmm0", XMM0, "xmm0"),
        (XMM7, "xmm7", XMM5, "xmm5", XMM3, "xmm3"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM10, "xmm10"),
        (XMM0, "xmm0", XMM15, "xmm15", XMM8, "xmm8"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AVX 3-operand YMM ─────────────────────────────────────────

#[test]
fn test_nasm_avx_float_ops_ymm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps", |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd", |a, d, s1, s2| a.vaddpd(d, s1, s2)),
        ("vsubps", |a, d, s1, s2| a.vsubps(d, s1, s2)),
        ("vmulps", |a, d, s1, s2| a.vmulps(d, s1, s2)),
        ("vdivps", |a, d, s1, s2| a.vdivps(d, s1, s2)),
        ("vxorps", |a, d, s1, s2| a.vxorps(d, s1, s2)),
        ("vandps", |a, d, s1, s2| a.vandps(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
        (YMM7, "ymm7", YMM5, "ymm5", YMM3, "ymm3"),
        (YMM8, "ymm8", YMM9, "ymm9", YMM10, "ymm10"),
        (YMM0, "ymm0", YMM15, "ymm15", YMM8, "ymm8"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AVX data movement ─────────────────────────────────────────

#[test]
fn test_nasm_avx_mov_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("vmovaps", |a, d, s| a.vmovaps(d, s)),
        ("vmovups", |a, d, s| a.vmovups(d, s)),
        ("vmovapd", |a, d, s| a.vmovapd(d, s)),
        ("vmovupd", |a, d, s| a.vmovupd(d, s)),
        ("vmovdqa", |a, d, s| a.vmovdqa(d, s)),
        ("vmovdqu", |a, d, s| a.vmovdqu(d, s)),
    ];

    // XMM pairs
    for &(mnemonic, method) in ops {
        for &(dst, dst_name) in &[(XMM0, "xmm0"), (XMM8, "xmm8"), (XMM15, "xmm15")] {
            for &(src, src_name) in &[(XMM1, "xmm1"), (XMM9, "xmm9")] {
                let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
                insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
            }
        }
    }

    // YMM pairs
    for &(mnemonic, method) in ops {
        for &(dst, dst_name) in &[(YMM0, "ymm0"), (YMM8, "ymm8")] {
            for &(src, src_name) in &[(YMM1, "ymm1"), (YMM9, "ymm9")] {
                let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
                insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AVX integer ops ────────────────────────────────────────────

#[test]
fn test_nasm_avx_int_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vpaddd", |a, d, s1, s2| a.vpaddd(d, s1, s2)),
        ("vpsubd", |a, d, s1, s2| a.vpsubd(d, s1, s2)),
        ("vpxor", |a, d, s1, s2| a.vpxor(d, s1, s2)),
        ("vpand", |a, d, s1, s2| a.vpand(d, s1, s2)),
        ("vpor", |a, d, s1, s2| a.vpor(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM10, "xmm10"),
        (YMM0, "ymm0", YMM1, "ymm1", YMM2, "ymm2"),
        (YMM8, "ymm8", YMM9, "ymm9", YMM10, "ymm10"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AVX with memory operands ───────────────────────────────────

#[test]
fn test_nasm_avx_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vaddps xmm, xmm, [mem]
    for &(base, base_name) in &[(RAX, "rax"), (RBX, "rbx"), (R8, "r8")] {
        let asm_text = format!("vaddps xmm0, xmm1, oword [{}]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vaddps(XMM0, XMM1, xmmword_ptr(base.into()))
        })));
    }

    // vaddps xmm, xmm, [mem + disp]
    {
        let asm_text = "vaddps xmm0, xmm1, oword [rax+0x10]".to_string();
        insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
            a.vaddps(XMM0, XMM1, xmmword_ptr(RAX + 0x10))
        })));
    }

    // vmovaps [mem], xmm
    for &(base, base_name) in &[(RAX, "rax"), (R8, "r8")] {
        let asm_text = format!("vmovaps oword [{}], xmm0", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vmovaps(xmmword_ptr(base.into()), XMM0)
        })));
    }

    // vaddps ymm, ymm, [mem]
    {
        let asm_text = "vaddps ymm0, ymm1, yword [rax]".to_string();
        insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
            a.vaddps(YMM0, YMM1, ymmword_ptr(RAX.into()))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── FMA instructions ───────────────────────────────────────────

#[test]
fn test_nasm_fma_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vfmadd132ps", |a, d, s1, s2| a.vfmadd132ps(d, s1, s2)),
        ("vfmadd213ps", |a, d, s1, s2| a.vfmadd213ps(d, s1, s2)),
        ("vfmadd231ps", |a, d, s1, s2| a.vfmadd231ps(d, s1, s2)),
        ("vfmadd132pd", |a, d, s1, s2| a.vfmadd132pd(d, s1, s2)),
        ("vfmadd213pd", |a, d, s1, s2| a.vfmadd213pd(d, s1, s2)),
        ("vfmadd231pd", |a, d, s1, s2| a.vfmadd231pd(d, s1, s2)),
        ("vfmsub132ps", |a, d, s1, s2| a.vfmsub132ps(d, s1, s2)),
        ("vfnmadd132ps", |a, d, s1, s2| a.vfnmadd132ps(d, s1, s2)),
        ("vfmadd132ss", |a, d, s1, s2| a.vfmadd132ss(d, s1, s2)),
        ("vfmadd132sd", |a, d, s1, s2| a.vfmadd132sd(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1", XMM2, "xmm2"),
        (XMM8, "xmm8", XMM9, "xmm9", XMM10, "xmm10"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    // YMM variants
    let ymm_ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vfmadd132ps", |a, d, s1, s2| a.vfmadd132ps(d, s1, s2)),
        ("vfmadd213ps", |a, d, s1, s2| a.vfmadd213ps(d, s1, s2)),
        ("vfmadd231ps", |a, d, s1, s2| a.vfmadd231ps(d, s1, s2)),
    ];

    for &(mnemonic, method) in ymm_ops {
        let asm_text = format!("{} ymm0, ymm1, ymm2", mnemonic);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, YMM0, YMM1, YMM2))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE with memory (full all-register sweep) ─────────────────

#[test]
fn test_nasm_sse_all_xmm_pairs() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // addps with all xmm0-xmm15 pairs (subset: each with xmm0 and xmm8)
    for &(dst, dst_name) in XMMS.iter().chain(XMMS_EXT.iter()) {
        for &(src, src_name) in &[(XMM0, "xmm0"), (XMM1, "xmm1"), (XMM8, "xmm8"), (XMM15, "xmm15")] {
            let asm_text = format!("addps {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.addps(dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Generated SSE instructions (min/max, shuffle, etc.) ────────

#[test]
fn test_nasm_generated_sse_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("minps", |a, d, s| a.minps(d, s)),
        ("maxps", |a, d, s| a.maxps(d, s)),
        ("minpd", |a, d, s| a.minpd(d, s)),
        ("maxpd", |a, d, s| a.maxpd(d, s)),
        ("minss", |a, d, s| a.minss(d, s)),
        ("maxss", |a, d, s| a.maxss(d, s)),
        ("minsd", |a, d, s| a.minsd(d, s)),
        ("maxsd", |a, d, s| a.maxsd(d, s)),
        ("rcpps", |a, d, s| a.rcpps(d, s)),
        ("rsqrtps", |a, d, s| a.rsqrtps(d, s)),
        ("unpcklps", |a, d, s| a.unpcklps(d, s)),
        ("unpckhps", |a, d, s| a.unpckhps(d, s)),
        ("andnps", |a, d, s| a.andnps(d, s)),
        ("andnpd", |a, d, s| a.andnpd(d, s)),
        ("movhlps", |a, d, s| a.movhlps(d, s)),
        ("movlhps", |a, d, s| a.movlhps(d, s)),
    ];

    let xmm_pairs: &[(Reg, &str, Reg, &str)] = &[
        (XMM0, "xmm0", XMM1, "xmm1"),
        (XMM2, "xmm2", XMM3, "xmm3"),
        (XMM8, "xmm8", XMM9, "xmm9"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in xmm_pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Generated SSE3/SSSE3 instructions ──────────────────────────

#[test]
fn test_nasm_sse3_ssse3_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("addsubps", |a, d, s| a.addsubps(d, s)),
        ("addsubpd", |a, d, s| a.addsubpd(d, s)),
        ("haddps", |a, d, s| a.haddps(d, s)),
        ("haddpd", |a, d, s| a.haddpd(d, s)),
        ("hsubps", |a, d, s| a.hsubps(d, s)),
        ("movshdup", |a, d, s| a.movshdup(d, s)),
        ("movsldup", |a, d, s| a.movsldup(d, s)),
        ("movddup", |a, d, s| a.movddup(d, s)),
        ("pshufb", |a, d, s| a.pshufb(d, s)),
        ("phaddw", |a, d, s| a.phaddw(d, s)),
        ("phaddd", |a, d, s| a.phaddd(d, s)),
        ("phaddsw", |a, d, s| a.phaddsw(d, s)),
        ("phsubw", |a, d, s| a.phsubw(d, s)),
        ("phsubd", |a, d, s| a.phsubd(d, s)),
        ("phsubsw", |a, d, s| a.phsubsw(d, s)),
        ("psignb", |a, d, s| a.psignb(d, s)),
        ("psignw", |a, d, s| a.psignw(d, s)),
        ("psignd", |a, d, s| a.psignd(d, s)),
        ("pmaddubsw", |a, d, s| a.pmaddubsw(d, s)),
        ("pmulhrsw", |a, d, s| a.pmulhrsw(d, s)),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in &[
            (XMM0, "xmm0", XMM1, "xmm1"),
            (XMM8, "xmm8", XMM9, "xmm9"),
        ] {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE4 instructions ──────────────────────────────────────────

#[test]
fn test_nasm_sse4_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("ptest", |a, d, s| a.ptest(d, s)),
        ("pmovzxbw", |a, d, s| a.pmovzxbw(d, s)),
        ("pmovsxbw", |a, d, s| a.pmovsxbw(d, s)),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in &[
            (XMM0, "xmm0", XMM1, "xmm1"),
            (XMM8, "xmm8", XMM9, "xmm9"),
        ] {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── AES instructions ───────────────────────────────────────────

#[test]
fn test_nasm_aes_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("aesenc", |a, d, s| a.aesenc(d, s)),
        ("aesenclast", |a, d, s| a.aesenclast(d, s)),
        ("aesdec", |a, d, s| a.aesdec(d, s)),
        ("aesdeclast", |a, d, s| a.aesdeclast(d, s)),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in &[
            (XMM0, "xmm0", XMM1, "xmm1"),
            (XMM8, "xmm8", XMM9, "xmm9"),
        ] {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── SSE/AVX with imm8 ─────────────────────────────────────────

#[test]
fn test_nasm_sse_imm8_ops() {
    let nasm = skip_if_no_nasm!();
    let insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = vec![
        ("cmpps xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.cmpps(XMM0, XMM1, 0))),
        ("cmpps xmm0, xmm1, 5".into(), Box::new(|a: &mut CodeAssembler| a.cmpps(XMM0, XMM1, 5))),
        ("cmppd xmm0, xmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.cmppd(XMM0, XMM1, 1))),
        ("shufps xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.shufps(XMM0, XMM1, 0))),
        ("shufps xmm0, xmm1, 0x44".into(), Box::new(|a: &mut CodeAssembler| a.shufps(XMM0, XMM1, 0x44))),
        ("shufpd xmm0, xmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.shufpd(XMM0, XMM1, 1))),
        ("pshufd xmm0, xmm1, 0x1b".into(), Box::new(|a: &mut CodeAssembler| a.pshufd(XMM0, XMM1, 0x1B))),
        ("pshufhw xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.pshufhw(XMM0, XMM1, 0))),
        ("pshuflw xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.pshuflw(XMM0, XMM1, 0))),
        ("roundps xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.roundps(XMM0, XMM1, 0))),
        ("roundpd xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.roundpd(XMM0, XMM1, 0))),
        ("roundss xmm0, xmm1, 2".into(), Box::new(|a: &mut CodeAssembler| a.roundss(XMM0, XMM1, 2))),
        ("roundsd xmm0, xmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.roundsd(XMM0, XMM1, 1))),
        ("blendps xmm0, xmm1, 5".into(), Box::new(|a: &mut CodeAssembler| a.blendps(XMM0, XMM1, 5))),
        ("blendpd xmm0, xmm1, 3".into(), Box::new(|a: &mut CodeAssembler| a.blendpd(XMM0, XMM1, 3))),
        ("insertps xmm0, xmm1, 0x10".into(), Box::new(|a: &mut CodeAssembler| a.insertps(XMM0, XMM1, 0x10))),
        ("palignr xmm0, xmm1, 4".into(), Box::new(|a: &mut CodeAssembler| a.palignr(XMM0, XMM1, 4))),
        ("pclmulqdq xmm0, xmm1, 0".into(), Box::new(|a: &mut CodeAssembler| a.pclmulqdq(XMM0, XMM1, 0))),
    ];
    compare_nasm_batch(&nasm, 64, insns);
}
