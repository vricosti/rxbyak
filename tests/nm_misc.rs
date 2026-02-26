// NASM conformance tests — miscellaneous instructions
// (AMX, variable blend, vcvtps2ph, clflushopt, FPU integer-memory ops)

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

// ── AMX tile instructions ──────────────────────────────────────

#[test]
fn test_nm_amx_basic() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // tilerelease (0-operand)
    cases.push((
        "tilerelease".to_string(),
        Box::new(|a: &mut CodeAssembler| a.tilerelease()),
    ));

    // tilezero tmm0..tmm7
    let tmms: &[(Reg, &str)] = &[
        (TMM0, "tmm0"), (TMM1, "tmm1"), (TMM2, "tmm2"), (TMM3, "tmm3"),
        (TMM4, "tmm4"), (TMM5, "tmm5"), (TMM6, "tmm6"), (TMM7, "tmm7"),
    ];
    for &(reg, name) in tmms {
        let nasm_str = format!("tilezero {}", name);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.tilezero(reg))));
    }

    compare_nasm_batch(&nasm, 64, cases);
}

#[test]
fn test_nm_amx_tdp() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // tdpbssd/bsud/busd/buud/bf16ps/fp16ps — tmm, tmm, tmm
    let tdp_ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("tdpbssd", CodeAssembler::tdpbssd),
        ("tdpbsud", CodeAssembler::tdpbsud),
        ("tdpbusd", CodeAssembler::tdpbusd),
        ("tdpbuud", CodeAssembler::tdpbuud),
        ("tdpbf16ps", CodeAssembler::tdpbf16ps),
        ("tdpfp16ps", CodeAssembler::tdpfp16ps),
    ];

    // Test a few representative tmm combinations
    let tmm_combos: &[(Reg, Reg, Reg, &str, &str, &str)] = &[
        (TMM0, TMM1, TMM2, "tmm0", "tmm1", "tmm2"),
        (TMM3, TMM4, TMM5, "tmm3", "tmm4", "tmm5"),
        (TMM6, TMM0, TMM7, "tmm6", "tmm0", "tmm7"),
    ];

    for &(nasm_name, method) in tdp_ops {
        for &(dst, src1, src2, dn, s1n, s2n) in tmm_combos {
            let nasm_str = format!("{} {}, {}, {}", nasm_name, dn, s1n, s2n);
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| method(a, dst, src1, src2))));
        }
    }

    compare_nasm_batch(&nasm, 64, cases);
}

#[test]
fn test_nm_amx_load_store() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let tmms: &[(Reg, &str)] = &[
        (TMM0, "tmm0"), (TMM1, "tmm1"), (TMM2, "tmm2"),
    ];

    // AMX load/store uses sibmem addressing: [base + index*stride]
    let addrs: &[(RegExp, &str)] = &[
        (RAX + RCX * 1, "[rax+rcx]"),
        (RBX + RDX * 1, "[rbx+rdx]"),
        (R13 + R14 * 1, "[r13+r14]"),
    ];

    for &(tmm, tn) in tmms {
        for (expr, nasm_addr) in addrs {
            let addr = ptr(expr.clone());
            let nasm_str = format!("tileloadd {}, {}", tn, nasm_addr);
            let addr2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.tileloadd(tmm, addr2))));
        }
    }

    for &(tmm, tn) in tmms {
        for (expr, nasm_addr) in addrs {
            let addr = ptr(expr.clone());
            let nasm_str = format!("tileloaddt1 {}, {}", tn, nasm_addr);
            let addr2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.tileloaddt1(tmm, addr2))));
        }
    }

    for &(tmm, tn) in tmms {
        for (expr, nasm_addr) in addrs {
            let addr = ptr(expr.clone());
            let nasm_str = format!("tilestored {}, {}", nasm_addr, tn);
            let addr2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.tilestored(addr2, tmm))));
        }
    }

    compare_nasm_batch(&nasm, 64, cases);
}

#[test]
fn test_nm_amx_tilecfg() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let addrs: &[(RegExp, &str)] = &[
        (RAX.into(), "[rax]"),
        (RBX.into(), "[rbx]"),
        (R13.into(), "[r13]"),
    ];

    for (expr, nasm_addr) in addrs {
        let addr = ptr(expr.clone());
        let nasm_str = format!("ldtilecfg {}", nasm_addr);
        let addr2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.ldtilecfg(addr2))));
    }

    for (expr, nasm_addr) in addrs {
        let addr = ptr(expr.clone());
        let nasm_str = format!("sttilecfg {}", nasm_addr);
        let addr2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.sttilecfg(addr2))));
    }

    compare_nasm_batch(&nasm, 64, cases);
}

// ── Variable blend (SSE4.1, implicit XMM0) ──────────────────────

#[test]
fn test_nm_blendv() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let xmm_pairs: &[(Reg, Reg, &str, &str)] = &[
        (XMM1, XMM2, "xmm1", "xmm2"),
        (XMM5, XMM7, "xmm5", "xmm7"),
        (XMM0, XMM15, "xmm0", "xmm15"),
    ];

    for &(dst, src, dn, sn) in xmm_pairs {
        let nasm_str = format!("blendvps {}, {}", dn, sn);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.blendvps(dst, src))));
    }
    for &(dst, src, dn, sn) in xmm_pairs {
        let nasm_str = format!("blendvpd {}, {}", dn, sn);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.blendvpd(dst, src))));
    }
    for &(dst, src, dn, sn) in xmm_pairs {
        let nasm_str = format!("pblendvb {}, {}", dn, sn);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.pblendvb(dst, src))));
    }

    // Also test with memory operand
    for (addr, nasm_mem) in mems128() {
        let nasm_str = format!("blendvps xmm1, {}", nasm_mem);
        let a2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.blendvps(XMM1, a2))));
    }
    for (addr, nasm_mem) in mems128() {
        let nasm_str = format!("blendvpd xmm1, {}", nasm_mem);
        let a2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.blendvpd(XMM1, a2))));
    }
    for (addr, nasm_mem) in mems128() {
        let nasm_str = format!("pblendvb xmm1, {}", nasm_mem);
        let a2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.pblendvb(XMM1, a2))));
    }

    compare_nasm_batch(&nasm, 64, cases);
}

// ── vcvtps2ph — float32 to float16 ─────────────────────────────

#[test]
fn test_nm_vcvtps2ph() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vcvtps2ph xmm, xmm, imm8
    let pairs: &[(Reg, Reg, &str, &str)] = &[
        (XMM0, XMM1, "xmm0", "xmm1"),
        (XMM5, XMM10, "xmm5", "xmm10"),
        (XMM15, XMM0, "xmm15", "xmm0"),
    ];
    for &(dst, src, dn, sn) in pairs {
        let nasm_str = format!("vcvtps2ph {}, {}, 0x04", dn, sn);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.vcvtps2ph(dst, src, 0x04))));
    }

    // vcvtps2ph xmm, ymm, imm8
    let ymm_pairs: &[(Reg, Reg, &str, &str)] = &[
        (XMM0, YMM1, "xmm0", "ymm1"),
        (XMM5, YMM10, "xmm5", "ymm10"),
    ];
    for &(dst, src, dn, sn) in ymm_pairs {
        let nasm_str = format!("vcvtps2ph {}, {}, 0x04", dn, sn);
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.vcvtps2ph(dst, src, 0x04))));
    }

    // vcvtps2ph [mem], xmm, imm8
    for (addr, nasm_mem) in mems128() {
        let nasm_str = format!("vcvtps2ph {}, xmm3, 0x04", nasm_mem);
        let a2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.vcvtps2ph(a2, XMM3, 0x04))));
    }

    compare_nasm_batch(&nasm, 64, cases);
}

// ── clflushopt ──────────────────────────────────────────────────

#[test]
fn test_nm_clflushopt() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let addrs: &[(RegExp, &str)] = &[
        (RAX.into(), "[rax]"),
        (RBX.into(), "[rbx]"),
        (R13.into(), "[r13]"),
        ((RAX + RCX * 4 + 0x100), "[rax+rcx*4+0x100]"),
    ];

    for (expr, nasm_addr) in addrs {
        let addr = byte_ptr(expr.clone());
        let nasm_str = format!("clflushopt {}", nasm_addr);
        let a2 = addr.clone();
        cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| a.clflushopt(a2))));
    }

    compare_nasm_batch_normalized(&nasm, 64, cases);
}

// ── FPU integer-memory operations ───────────────────────────────

#[test]
fn test_nm_fpu_int_mem() {
    let nasm = skip_if_no_nasm!();

    let mut cases: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // fiadd/fisub/fimul/fidiv with word and dword memory
    let int_arith: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>, &str)] = &[
        ("fiadd", CodeAssembler::fiadd_m16, "word"),
        ("fiadd", CodeAssembler::fiadd_m32, "dword"),
        ("fisub", CodeAssembler::fisub_m16, "word"),
        ("fisub", CodeAssembler::fisub_m32, "dword"),
        ("fimul", CodeAssembler::fimul_m16, "word"),
        ("fimul", CodeAssembler::fimul_m32, "dword"),
        ("fidiv", CodeAssembler::fidiv_m16, "word"),
        ("fidiv", CodeAssembler::fidiv_m32, "dword"),
    ];

    let addrs: &[(RegExp, &str)] = &[
        (RAX.into(), "[rax]"),
        (RBX.into(), "[rbx]"),
        ((RBP + 0x10), "[rbp+0x10]"),
    ];

    for &(nasm_name, method, size) in int_arith {
        for (expr, nasm_addr) in addrs {
            let addr = if size == "word" {
                word_ptr(expr.clone())
            } else {
                dword_ptr(expr.clone())
            };
            let nasm_str = format!("{} {} {}", nasm_name, size, nasm_addr);
            let a2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| method(a, a2))));
        }
    }

    // ficom/ficomp with word and dword memory
    let int_cmp: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>, &str)] = &[
        ("ficom", CodeAssembler::ficom_m16, "word"),
        ("ficom", CodeAssembler::ficom_m32, "dword"),
        ("ficomp", CodeAssembler::ficomp_m16, "word"),
        ("ficomp", CodeAssembler::ficomp_m32, "dword"),
    ];

    for &(nasm_name, method, size) in int_cmp {
        for (expr, nasm_addr) in addrs {
            let addr = if size == "word" {
                word_ptr(expr.clone())
            } else {
                dword_ptr(expr.clone())
            };
            let nasm_str = format!("{} {} {}", nasm_name, size, nasm_addr);
            let a2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| method(a, a2))));
        }
    }

    // fisttp with word, dword, qword memory
    let fisttp_ops: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>, &str)] = &[
        ("fisttp", CodeAssembler::fisttp_m16, "word"),
        ("fisttp", CodeAssembler::fisttp_m32, "dword"),
        ("fisttp", CodeAssembler::fisttp_m64, "qword"),
    ];

    for &(nasm_name, method, size) in fisttp_ops {
        for (expr, nasm_addr) in addrs {
            let addr = if size == "word" {
                word_ptr(expr.clone())
            } else if size == "dword" {
                dword_ptr(expr.clone())
            } else {
                qword_ptr(expr.clone())
            };
            let nasm_str = format!("{} {} {}", nasm_name, size, nasm_addr);
            let a2 = addr.clone();
            cases.push((nasm_str, Box::new(move |a: &mut CodeAssembler| method(a, a2))));
        }
    }

    compare_nasm_batch(&nasm, 64, cases);
}
