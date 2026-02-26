/// AVX-512/EVEX instruction NASM conformance tests.

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
// EVEX arithmetic with ZMM
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_arith() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("vaddps", |a, d, s1, s2| a.vaddps(d, s1, s2)),
        ("vaddpd", |a, d, s1, s2| a.vaddpd(d, s1, s2)),
        ("vsubps", |a, d, s1, s2| a.vsubps(d, s1, s2)),
        ("vmulps", |a, d, s1, s2| a.vmulps(d, s1, s2)),
        ("vdivps", |a, d, s1, s2| a.vdivps(d, s1, s2)),
    ];

    let triples: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (ZMM0, "zmm0", ZMM1, "zmm1", ZMM2, "zmm2"),
        (ZMM16, "zmm16", ZMM17, "zmm17", ZMM31, "zmm31"),
        (ZMM0, "zmm0", ZMM8, "zmm8", ZMM31, "zmm31"),
    ];

    for &(name, op_fn) in ops {
        for &(d, dn, s1, s1n, s2, s2n) in triples {
            let asm = format!("{} {}, {}, {}", name, dn, s1n, s2n);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, d, s1, s2))));
        }
    }

    // ZMM with memory
    for (addr, nasm_mem) in mems512() {
        let asm = format!("vaddps zmm0, zmm1, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(ZMM0, ZMM1, addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX with extended registers (zmm16-zmm31)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_extended_regs() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // Test various zmm16-31 combinations
    let combos: &[(Reg, &str, Reg, &str, Reg, &str)] = &[
        (ZMM16, "zmm16", ZMM17, "zmm17", ZMM18, "zmm18"),
        (ZMM24, "zmm24", ZMM25, "zmm25", ZMM31, "zmm31"),
        (ZMM0, "zmm0", ZMM16, "zmm16", ZMM31, "zmm31"),
        (ZMM31, "zmm31", ZMM0, "zmm0", ZMM16, "zmm16"),
    ];

    for &(d, dn, s1, s1n, s2, s2n) in combos {
        let asm = format!("vaddps {}, {}, {}", dn, s1n, s2n);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(d, s1, s2))));
        let asm = format!("vmulpd {}, {}, {}", dn, s1n, s2n);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmulpd(d, s1, s2))));
    }

    // Extended XMM registers (xmm16-31)
    insns.push(("vaddps xmm16, xmm17, xmm18".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(XMM16, XMM17, XMM18))));
    insns.push(("vaddps xmm31, xmm0, xmm16".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(XMM31, XMM0, XMM16))));
    // Extended YMM registers
    insns.push(("vaddps ymm16, ymm17, ymm18".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(YMM16, YMM17, YMM18))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX opmask: {k1}-{k7} writemask
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_opmask() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vaddps zmm{k1}, zmm, zmm
    for &(k, ki, kn) in &[(K1, 1u8, "k1"), (K3, 3, "k3"), (K7, 7, "k7")] {
        let _ = k; // use ki for the .k() modifier
        let asm = format!("vaddps zmm0{{{}}}, zmm1, zmm2", kn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(ZMM0.k(ki), ZMM1, ZMM2))));
    }

    // vaddpd xmm{k2}, xmm, xmm
    insns.push(("vaddpd xmm0{k2}, xmm1, xmm2".into(), Box::new(|a: &mut CodeAssembler| a.vaddpd(XMM0.k(2), XMM1, XMM2))));
    // vaddps ymm{k5}, ymm, ymm
    insns.push(("vaddps ymm0{k5}, ymm1, ymm2".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(YMM0.k(5), YMM1, YMM2))));

    // vmovdqa32 with mask
    insns.push(("vmovdqa32 zmm0{k1}, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(ZMM0.k(1), ZMM1))));
    insns.push(("vmovdqa32 zmm16{k7}, zmm31".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(ZMM16.k(7), ZMM31))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX zeroing: {z} modifier
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_zeroing() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vaddps zmm{k1}{z}, zmm, zmm
    insns.push(("vaddps zmm0{k1}{z}, zmm1, zmm2".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(ZMM0.k(1).z(), ZMM1, ZMM2))));
    insns.push(("vaddps zmm16{k7}{z}, zmm17, zmm31".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(ZMM16.k(7).z(), ZMM17, ZMM31))));
    // vaddpd xmm{k2}{z}, xmm, xmm
    insns.push(("vaddpd xmm0{k2}{z}, xmm1, xmm2".into(), Box::new(|a: &mut CodeAssembler| a.vaddpd(XMM0.k(2).z(), XMM1, XMM2))));
    // vmovdqa32 with mask + zeroing
    insns.push(("vmovdqa32 zmm0{k1}{z}, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(ZMM0.k(1).z(), ZMM1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX rounding: {rn-sae}/{rd-sae}/{ru-sae}/{rz-sae}
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_rounding() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let modes: &[(Rounding, &str)] = &[
        (Rounding::RnSae, "{rn-sae}"),
        (Rounding::RdSae, "{rd-sae}"),
        (Rounding::RuSae, "{ru-sae}"),
        (Rounding::RzSae, "{rz-sae}"),
    ];

    for &(rnd, nasm_rnd) in modes {
        let asm = format!("vaddps zmm0, zmm1, zmm2, {}", nasm_rnd);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vaddps(ZMM0, ZMM1, ZMM2.rounding(rnd)))));
    }

    // vaddpd with rounding
    insns.push(("vaddpd zmm0, zmm1, zmm2, {rn-sae}".into(), Box::new(|a: &mut CodeAssembler| a.vaddpd(ZMM0, ZMM1, ZMM2.rounding(Rounding::RnSae)))));

    // Combined: mask + zeroing + rounding
    insns.push(("vaddps zmm0{k1}{z}, zmm1, zmm2, {rn-sae}".into(), Box::new(|a: &mut CodeAssembler| a.vaddps(ZMM0.k(1).z(), ZMM1, ZMM2.rounding(Rounding::RnSae)))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX mov instructions: vmovdqa32/64, vmovdqu8/16/32/64
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_mov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vmovdqa32 zmm, zmm
    insns.push(("vmovdqa32 zmm0, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(ZMM0, ZMM1))));
    insns.push(("vmovdqa32 zmm16, zmm31".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(ZMM16, ZMM31))));
    // vmovdqa64 zmm, zmm
    insns.push(("vmovdqa64 zmm0, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa64(ZMM0, ZMM1))));
    // vmovdqu32 zmm, zmm
    insns.push(("vmovdqu32 zmm0, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqu32(ZMM0, ZMM1))));
    // vmovdqu64 zmm, zmm
    insns.push(("vmovdqu64 zmm0, zmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqu64(ZMM0, ZMM1))));

    // vmovdqa32 zmm, [mem]
    for (addr, nasm_mem) in mems512() {
        let asm = format!("vmovdqa32 zmm0, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovdqa32(ZMM0, addr))));
    }
    // vmovdqa32 [mem], zmm
    for (addr, nasm_mem) in mems512() {
        let asm = format!("vmovdqa32 {}, zmm0", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.vmovdqa32(addr, ZMM0))));
    }

    // xmm/ymm EVEX mov
    insns.push(("vmovdqa32 xmm0, xmm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(XMM0, XMM1))));
    insns.push(("vmovdqa32 ymm0, ymm1".into(), Box::new(|a: &mut CodeAssembler| a.vmovdqa32(YMM0, YMM1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// Opmask (K-register) instructions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_opmask_mov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // kmovw k, k
    insns.push(("kmovw k1, k2".into(), Box::new(|a: &mut CodeAssembler| a.kmovw(K1, K2))));
    insns.push(("kmovw k7, k0".into(), Box::new(|a: &mut CodeAssembler| a.kmovw(K7, K0))));
    // kmovw k, r32
    insns.push(("kmovw k1, eax".into(), Box::new(|a: &mut CodeAssembler| a.kmovw(K1, EAX))));
    insns.push(("kmovw k7, r8d".into(), Box::new(|a: &mut CodeAssembler| a.kmovw(K7, R8D))));
    // kmovw r32, k
    insns.push(("kmovw eax, k1".into(), Box::new(|a: &mut CodeAssembler| a.kmovw(EAX, K1))));
    // kmovb
    insns.push(("kmovb k1, k2".into(), Box::new(|a: &mut CodeAssembler| a.kmovb(K1, K2))));
    // kmovd
    insns.push(("kmovd k1, k2".into(), Box::new(|a: &mut CodeAssembler| a.kmovd(K1, K2))));
    insns.push(("kmovd k1, eax".into(), Box::new(|a: &mut CodeAssembler| a.kmovd(K1, EAX))));
    // kmovq
    insns.push(("kmovq k1, k2".into(), Box::new(|a: &mut CodeAssembler| a.kmovq(K1, K2))));

    // kmovw k, [mem]
    for (addr, nasm_mem) in mems_nosizeptr() {
        let asm = format!("kmovw k1, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.kmovw(K1, addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_opmask_logic() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // kandw k, k, k
    let ops3: &[(&str, fn(&mut CodeAssembler, Reg, Reg, Reg) -> Result<()>)] = &[
        ("kandw",  |a, d, s1, s2| a.kandw(d, s1, s2)),
        ("kandb",  |a, d, s1, s2| a.kandb(d, s1, s2)),
        ("kandd",  |a, d, s1, s2| a.kandd(d, s1, s2)),
        ("kandq",  |a, d, s1, s2| a.kandq(d, s1, s2)),
        ("kandnw", |a, d, s1, s2| a.kandnw(d, s1, s2)),
        ("korw",   |a, d, s1, s2| a.korw(d, s1, s2)),
        ("korb",   |a, d, s1, s2| a.korb(d, s1, s2)),
        ("kxorw",  |a, d, s1, s2| a.kxorw(d, s1, s2)),
        ("kxnorw", |a, d, s1, s2| a.kxnorw(d, s1, s2)),
        ("kaddw",  |a, d, s1, s2| a.kaddw(d, s1, s2)),
    ];

    for &(name, op_fn) in ops3 {
        let asm = format!("{} k1, k2, k3", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, K1, K2, K3))));
        let asm = format!("{} k7, k0, k6", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, K7, K0, K6))));
    }

    // 2-operand: knotw, kortestw
    let ops2: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("knotw",    |a, d, s| a.knotw(d, s)),
        ("knotb",    |a, d, s| a.knotb(d, s)),
        ("knotd",    |a, d, s| a.knotd(d, s)),
        ("knotq",    |a, d, s| a.knotq(d, s)),
        ("kortestw", |a, d, s| a.kortestw(d, s)),
        ("kortestb", |a, d, s| a.kortestb(d, s)),
    ];

    for &(name, op_fn) in ops2 {
        let asm = format!("{} k1, k2", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, K1, K2))));
    }

    // kshiftlw k, k, imm8
    insns.push(("kshiftlw k1, k2, 4".into(), Box::new(|a: &mut CodeAssembler| a.kshiftlw(K1, K2, 4))));
    insns.push(("kshiftrw k1, k2, 8".into(), Box::new(|a: &mut CodeAssembler| a.kshiftrw(K1, K2, 8))));

    // kunpckbw k, k, k
    insns.push(("kunpckbw k1, k2, k3".into(), Box::new(|a: &mut CodeAssembler| a.kunpckbw(K1, K2, K3))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// EVEX extract: vextractf32x4/i32x4, vextractf64x4/i64x4
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_evex_extract() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // vextractf32x4 xmm, zmm, imm8
    insns.push(("vextractf32x4 xmm0, zmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.vextractf32x4(XMM0, ZMM1, 1))));
    insns.push(("vextractf32x4 xmm16, zmm31, 3".into(), Box::new(|a: &mut CodeAssembler| a.vextractf32x4(XMM16, ZMM31, 3))));
    // vextracti32x4
    insns.push(("vextracti32x4 xmm0, zmm1, 2".into(), Box::new(|a: &mut CodeAssembler| a.vextracti32x4(XMM0, ZMM1, 2))));
    // vextractf64x4 ymm, zmm, imm8
    insns.push(("vextractf64x4 ymm0, zmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.vextractf64x4(YMM0, ZMM1, 1))));
    // vextracti64x4
    insns.push(("vextracti64x4 ymm0, zmm1, 1".into(), Box::new(|a: &mut CodeAssembler| a.vextracti64x4(YMM0, ZMM1, 1))));

    compare_nasm_batch(&nasm, 64, insns);
}
