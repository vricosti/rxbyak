/// AVX-512 (EVEX) instruction tests validated against NASM reference assembler.

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

// ─── Basic EVEX (zmm, zmm, zmm) ────────────────────────────────

#[test]
fn test_nasm_evex_basic_float() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps", |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd", |a, d, s1, s2| a.vaddpd(d, s1, s2)),
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
        (ZMM0, "zmm0", ZMM1, "zmm1", ZMM2, "zmm2"),
        (ZMM0, "zmm0", ZMM0, "zmm0", ZMM0, "zmm0"),
        (ZMM7, "zmm7", ZMM5, "zmm5", ZMM3, "zmm3"),
        (ZMM8, "zmm8", ZMM9, "zmm9", ZMM10, "zmm10"),
        (ZMM15, "zmm15", ZMM14, "zmm14", ZMM13, "zmm13"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX integer ops ───────────────────────────────────────────

#[test]
fn test_nasm_evex_int_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vpaddd", |a, d, s1, s2| a.vpaddd(d, s1, s2)),
        ("vpsubd", |a, d, s1, s2| a.vpsubd(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (ZMM0, "zmm0", ZMM1, "zmm1", ZMM2, "zmm2"),
        (ZMM8, "zmm8", ZMM9, "zmm9", ZMM10, "zmm10"),
        (ZMM15, "zmm15", ZMM0, "zmm0", ZMM7, "zmm7"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX-only instructions (vpandd, vpord, vpxord, etc.) ───────

#[test]
fn test_nasm_evex_only_int() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vpandd", |a, d, s1, s2| a.vpandd(d, s1, s2)),
        ("vpandq", |a, d, s1, s2| a.vpandq(d, s1, s2)),
        ("vpord", |a, d, s1, s2| a.vpord(d, s1, s2)),
        ("vpxord", |a, d, s1, s2| a.vpxord(d, s1, s2)),
        ("vpaddq", |a, d, s1, s2| a.vpaddq(d, s1, s2)),
        ("vpsubq", |a, d, s1, s2| a.vpsubq(d, s1, s2)),
        ("vpmulld", |a, d, s1, s2| a.vpmulld(d, s1, s2)),
        ("vpmullq", |a, d, s1, s2| a.vpmullq(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (ZMM0, "zmm0", ZMM1, "zmm1", ZMM2, "zmm2"),
        (ZMM8, "zmm8", ZMM9, "zmm9", ZMM10, "zmm10"),
    ];

    for &(mnemonic, method) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm_text = format!("{} {}, {}, {}", mnemonic, dn, s1n, s2n);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, d, s1, s2))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX with opmask ───────────────────────────────────────────

#[test]
fn test_nasm_evex_opmask() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vpaddd zmm{k1}, zmm, zmm
    for k in 1u8..=7 {
        let asm_text = format!("vpaddd zmm0{{k{}}}, zmm1, zmm2", k);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vpaddd(ZMM0.k(k), ZMM1, ZMM2)
        })));
    }

    // vaddps zmm{k1}, zmm, zmm
    let asm_text = "vaddps zmm0{k1}, zmm1, zmm2".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0.k(1), ZMM1, ZMM2)
    })));

    // With extended registers and mask
    let asm_text = "vaddps zmm8{k2}, zmm9, zmm10".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM8.k(2), ZMM9, ZMM10)
    })));

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX with zeroing mask ─────────────────────────────────────

#[test]
fn test_nasm_evex_zeroing() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vpaddd zmm{k1}{z}, zmm, zmm
    let asm_text = "vpaddd zmm0{k1}{z}, zmm1, zmm2".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vpaddd(ZMM0.k(1).z(), ZMM1, ZMM2)
    })));

    // vaddps zmm{k1}{z}, zmm, zmm
    let asm_text = "vaddps zmm0{k1}{z}, zmm1, zmm2".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0.k(1).z(), ZMM1, ZMM2)
    })));

    // Extended regs with zeroing
    let asm_text = "vaddps zmm8{k2}{z}, zmm9, zmm10".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM8.k(2).z(), ZMM9, ZMM10)
    })));

    let asm_text = "vsubps zmm15{k3}{z}, zmm14, zmm13".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vsubps(ZMM15.k(3).z(), ZMM14, ZMM13)
    })));

    // Multiple mask values with zeroing
    for k in 1u8..=7 {
        let asm_text = format!("vaddpd zmm0{{k{}}}, zmm1, zmm2", k);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vaddpd(ZMM0.k(k), ZMM1, ZMM2)
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX extended registers (zmm16-zmm31) ─────────────────────

#[test]
fn test_nasm_evex_extended_regs() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (ZMM16, "zmm16", ZMM17, "zmm17", ZMM18, "zmm18"),
        (ZMM20, "zmm20", ZMM21, "zmm21", ZMM22, "zmm22"),
        (ZMM24, "zmm24", ZMM25, "zmm25", ZMM26, "zmm26"),
        (ZMM28, "zmm28", ZMM29, "zmm29", ZMM30, "zmm30"),
        (ZMM31, "zmm31", ZMM0, "zmm0", ZMM1, "zmm1"),
        (ZMM0, "zmm0", ZMM16, "zmm16", ZMM31, "zmm31"),
    ];

    for &(d, dn, s1, s1n, s2, s2n) in triples {
        let asm_text = format!("vaddps {}, {}, {}", dn, s1n, s2n);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.vaddps(d, s1, s2))));
    }

    for &(d, dn, s1, s1n, s2, s2n) in triples {
        let asm_text = format!("vpaddd {}, {}, {}", dn, s1n, s2n);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.vpaddd(d, s1, s2))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX mov instructions ─────────────────────────────────────

#[test]
fn test_nasm_evex_mov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("vmovdqa32", |a, d, s| a.vmovdqa32(d, s)),
        ("vmovdqa64", |a, d, s| a.vmovdqa64(d, s)),
        ("vmovdqu32", |a, d, s| a.vmovdqu32(d, s)),
        ("vmovdqu64", |a, d, s| a.vmovdqu64(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (ZMM0, "zmm0", ZMM1, "zmm1"),
        (ZMM8, "zmm8", ZMM9, "zmm9"),
        (ZMM16, "zmm16", ZMM17, "zmm17"),
    ];

    for &(mnemonic, method) in ops {
        for &(dst, dst_name, src, src_name) in pairs {
            let asm_text = format!("{} {}, {}", mnemonic, dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX vpternlogd/q ─────────────────────────────────────────

#[test]
fn test_nasm_evex_vpternlog() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let imm_values: &[u8] = &[0x00, 0xFF, 0xDB, 0x96];
    for &imm in imm_values {
        let asm_text = format!("vpternlogd zmm0, zmm1, zmm2, 0x{:x}", imm);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vpternlogd(ZMM0, ZMM1, ZMM2, imm)
        })));
    }

    for &imm in imm_values {
        let asm_text = format!("vpternlogq zmm0, zmm1, zmm2, 0x{:x}", imm);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vpternlogq(ZMM0, ZMM1, ZMM2, imm)
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX with broadcast ────────────────────────────────────────

#[test]
fn test_nasm_evex_broadcast() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vpaddd zmm0, zmm1, dword [rax]{1to16}
    let asm_text = "vpaddd zmm0, zmm1, dword [rax]{1to16}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vpaddd(ZMM0, ZMM1, broadcast_ptr(32, RAX.into()))
    })));

    // vaddps zmm0, zmm1, dword [rax]{1to16}
    let asm_text = "vaddps zmm0, zmm1, dword [rax]{1to16}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, broadcast_ptr(32, RAX.into()))
    })));

    // vaddpd zmm0, zmm1, qword [rax]{1to8}
    let asm_text = "vaddpd zmm0, zmm1, qword [rax]{1to8}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddpd(ZMM0, ZMM1, broadcast_ptr(64, RAX.into()))
    })));

    // With extended base register
    let asm_text = "vpaddd zmm0, zmm1, dword [r8]{1to16}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vpaddd(ZMM0, ZMM1, broadcast_ptr(32, R8.into()))
    })));

    // With displacement
    let asm_text = "vpaddd zmm0, zmm1, dword [rax+0x10]{1to16}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vpaddd(ZMM0, ZMM1, broadcast_ptr(32, RAX + 0x10))
    })));

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX with rounding ─────────────────────────────────────────

#[test]
fn test_nasm_evex_rounding() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vaddps zmm0, zmm1, zmm2, {rn-sae}
    let asm_text = "vaddps zmm0, zmm1, zmm2, {rn-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RnSae))
    })));

    // vaddps zmm0, zmm1, zmm2, {rd-sae}
    let asm_text = "vaddps zmm0, zmm1, zmm2, {rd-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RdSae))
    })));

    // vaddps zmm0, zmm1, zmm2, {ru-sae}
    let asm_text = "vaddps zmm0, zmm1, zmm2, {ru-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RuSae))
    })));

    // vaddps zmm0, zmm1, zmm2, {rz-sae}
    let asm_text = "vaddps zmm0, zmm1, zmm2, {rz-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, ZMM2.rounding(Rounding::RzSae))
    })));

    // vaddpd with rounding
    let asm_text = "vaddpd zmm0, zmm1, zmm2, {rn-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddpd(ZMM0, ZMM1, ZMM2.rounding(Rounding::RnSae))
    })));

    // Combined: mask + rounding
    let asm_text = "vaddps zmm0{k1}, zmm1, zmm2, {rn-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0.k(1), ZMM1, ZMM2.rounding(Rounding::RnSae))
    })));

    // Combined: mask + zeroing + rounding
    let asm_text = "vaddps zmm0{k1}{z}, zmm1, zmm2, {rz-sae}".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0.k(1).z(), ZMM1, ZMM2.rounding(Rounding::RzSae))
    })));

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX with memory ───────────────────────────────────────────

#[test]
fn test_nasm_evex_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // vaddps zmm, zmm, [mem]
    let asm_text = "vaddps zmm0, zmm1, zword [rax]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, zmmword_ptr(RAX.into()))
    })));

    let asm_text = "vaddps zmm0, zmm1, zword [rax+0x40]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM0, ZMM1, zmmword_ptr(RAX + 0x40))
    })));

    let asm_text = "vaddps zmm8, zmm9, zword [r8]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vaddps(ZMM8, ZMM9, zmmword_ptr(R8.into()))
    })));

    // vmovdqa32 zmm, [mem]
    let asm_text = "vmovdqa32 zmm0, zword [rax]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vmovdqa32(ZMM0, zmmword_ptr(RAX.into()))
    })));

    // vmovdqa32 [mem], zmm
    let asm_text = "vmovdqa32 zword [rax], zmm0".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.vmovdqa32(zmmword_ptr(RAX.into()), ZMM0)
    })));

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── EVEX FMA with ZMM ─────────────────────────────────────────

#[test]
fn test_nasm_evex_fma() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vfmadd132ps", |a, d, s1, s2| a.vfmadd132ps(d, s1, s2)),
        ("vfmadd213ps", |a, d, s1, s2| a.vfmadd213ps(d, s1, s2)),
        ("vfmadd231ps", |a, d, s1, s2| a.vfmadd231ps(d, s1, s2)),
        ("vfmadd132pd", |a, d, s1, s2| a.vfmadd132pd(d, s1, s2)),
        ("vfmadd213pd", |a, d, s1, s2| a.vfmadd213pd(d, s1, s2)),
        ("vfmadd231pd", |a, d, s1, s2| a.vfmadd231pd(d, s1, s2)),
    ];

    for &(mnemonic, method) in ops {
        let asm_text = format!("{} zmm0, zmm1, zmm2", mnemonic);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, ZMM0, ZMM1, ZMM2))));
    }

    for &(mnemonic, method) in ops {
        let asm_text = format!("{} zmm16, zmm17, zmm18", mnemonic);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| method(a, ZMM16, ZMM17, ZMM18))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
