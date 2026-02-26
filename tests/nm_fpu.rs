/// x87 FPU instruction NASM conformance tests.

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
// FPU load/store
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_load_store() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // fld st(i)
    for &(st, sn) in &[(ST0, "st0"), (ST1, "st1"), (ST3, "st3"), (ST7, "st7")] {
        let asm = format!("fld {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fld_st(st))));
    }
    // fld m32fp
    for (addr, nasm_mem) in mems32() {
        let asm = format!("fld {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fld_m32(addr))));
    }
    // fld m64fp
    for (addr, nasm_mem) in mems64() {
        let asm = format!("fld {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fld_m64(addr))));
    }
    // fst st(i)
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fst {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fst_st(st))));
    }
    // fstp st(i)
    for &(st, sn) in &[(ST0, "st0"), (ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fstp {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fstp_st(st))));
    }
    // fst m32, fstp m32
    for (addr, nasm_mem) in mems32() {
        let asm = format!("fst {}", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fst_m32(a2))));
        let asm = format!("fstp {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fstp_m32(addr))));
    }
    // fst m64, fstp m64
    for (addr, nasm_mem) in mems64() {
        let asm = format!("fst {}", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fst_m64(a2))));
        let asm = format!("fstp {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fstp_m64(addr))));
    }
    // fild m32, fild m16
    for (addr, nasm_mem) in mems32() {
        let asm = format!("fild {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fild_m32(addr))));
    }
    for (addr, nasm_mem) in mems16() {
        let asm = format!("fild {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fild_m16(addr))));
    }
    // fist m32, fistp m32
    for (addr, nasm_mem) in mems32() {
        let asm = format!("fist {}", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fist_m32(a2))));
        let asm = format!("fistp {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fistp_m32(addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// FPU arithmetic
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_arith() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // fadd st(0), st(i)
    for &(st, sn) in &[(ST1, "st1"), (ST3, "st3"), (ST7, "st7")] {
        let asm = format!("fadd st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fadd_st0_st(st))));
    }
    // fadd st(i), st(0)
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fadd {}, st0", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fadd_st_st0(st))));
    }
    // faddp st(i), st(0)
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("faddp {}, st0", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.faddp(st))));
    }
    // fadd m32
    for (addr, nasm_mem) in mems32() {
        let asm = format!("fadd {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fadd_m32(addr))));
    }
    // fadd m64
    for (addr, nasm_mem) in mems64() {
        let asm = format!("fadd {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fadd_m64(addr))));
    }

    // fsub st(0), st(i)
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fsub st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fsub_st0_st(st))));
        let asm = format!("fsubr st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fsubr_st0_st(st))));
    }
    // fsubp / fsubrp
    insns.push(("fsubp st1, st0".into(), Box::new(|a: &mut CodeAssembler| a.fsubp(ST1))));
    insns.push(("fsubrp st1, st0".into(), Box::new(|a: &mut CodeAssembler| a.fsubrp(ST1))));

    // fmul st(0), st(i) / fmulp
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fmul st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fmul_st0_st(st))));
    }
    insns.push(("fmulp st1, st0".into(), Box::new(|a: &mut CodeAssembler| a.fmulp(ST1))));

    // fdiv st(0), st(i) / fdivp / fdivr / fdivrp
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fdiv st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fdiv_st0_st(st))));
        let asm = format!("fdivr st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fdivr_st0_st(st))));
    }
    insns.push(("fdivp st1, st0".into(), Box::new(|a: &mut CodeAssembler| a.fdivp(ST1))));
    insns.push(("fdivrp st1, st0".into(), Box::new(|a: &mut CodeAssembler| a.fdivrp(ST1))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// FPU compare
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_compare() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // fcom st(i)
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fcom {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fcom(st))));
        let asm = format!("fcomp {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fcomp(st))));
    }
    insns.push(("fcompp".into(), Box::new(|a: &mut CodeAssembler| a.fcompp())));
    // fucom
    for &(st, sn) in &[(ST1, "st1"), (ST7, "st7")] {
        let asm = format!("fucom {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fucom(st))));
        let asm = format!("fucomp {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fucomp(st))));
    }
    insns.push(("fucompp".into(), Box::new(|a: &mut CodeAssembler| a.fucompp())));
    // fcomi / fcomip / fucomi / fucomip
    for &(st, sn) in &[(ST1, "st1"), (ST3, "st3")] {
        let asm = format!("fcomi st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fcomi(st))));
        let asm = format!("fcomip st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fcomip(st))));
        let asm = format!("fucomi st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fucomi(st))));
        let asm = format!("fucomip st0, {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fucomip(st))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// FPU misc: fabs, fchs, fsqrt, fsin, fcos, fxch, constants
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_misc() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // Zero-operand
    insns.push(("fchs".into(), Box::new(|a: &mut CodeAssembler| a.fchs())));
    insns.push(("fabs".into(), Box::new(|a: &mut CodeAssembler| a.fabs())));
    insns.push(("fsqrt".into(), Box::new(|a: &mut CodeAssembler| a.fsqrt())));
    insns.push(("fsin".into(), Box::new(|a: &mut CodeAssembler| a.fsin())));
    insns.push(("fcos".into(), Box::new(|a: &mut CodeAssembler| a.fcos())));
    insns.push(("fptan".into(), Box::new(|a: &mut CodeAssembler| a.fptan())));
    insns.push(("fpatan".into(), Box::new(|a: &mut CodeAssembler| a.fpatan())));
    insns.push(("frndint".into(), Box::new(|a: &mut CodeAssembler| a.frndint())));
    insns.push(("fscale".into(), Box::new(|a: &mut CodeAssembler| a.fscale())));
    insns.push(("f2xm1".into(), Box::new(|a: &mut CodeAssembler| a.f2xm1())));
    insns.push(("fyl2x".into(), Box::new(|a: &mut CodeAssembler| a.fyl2x())));
    insns.push(("fyl2xp1".into(), Box::new(|a: &mut CodeAssembler| a.fyl2xp1())));
    insns.push(("fprem".into(), Box::new(|a: &mut CodeAssembler| a.fprem())));
    insns.push(("fprem1".into(), Box::new(|a: &mut CodeAssembler| a.fprem1())));
    insns.push(("fxtract".into(), Box::new(|a: &mut CodeAssembler| a.fxtract())));
    insns.push(("ftst".into(), Box::new(|a: &mut CodeAssembler| a.ftst())));
    insns.push(("fxam".into(), Box::new(|a: &mut CodeAssembler| a.fxam())));

    // Constants
    insns.push(("fldz".into(), Box::new(|a: &mut CodeAssembler| a.fldz())));
    insns.push(("fld1".into(), Box::new(|a: &mut CodeAssembler| a.fld1())));
    insns.push(("fldpi".into(), Box::new(|a: &mut CodeAssembler| a.fldpi())));
    insns.push(("fldl2t".into(), Box::new(|a: &mut CodeAssembler| a.fldl2t())));
    insns.push(("fldl2e".into(), Box::new(|a: &mut CodeAssembler| a.fldl2e())));
    insns.push(("fldlg2".into(), Box::new(|a: &mut CodeAssembler| a.fldlg2())));
    insns.push(("fldln2".into(), Box::new(|a: &mut CodeAssembler| a.fldln2())));

    // fxch
    for &(st, sn) in &[(ST1, "st1"), (ST3, "st3"), (ST7, "st7")] {
        let asm = format!("fxch {}", sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fxch(st))));
    }

    // Stack manipulation
    insns.push(("fdecstp".into(), Box::new(|a: &mut CodeAssembler| a.fdecstp())));
    insns.push(("fincstp".into(), Box::new(|a: &mut CodeAssembler| a.fincstp())));
    insns.push(("fnop".into(), Box::new(|a: &mut CodeAssembler| a.fnop())));
    insns.push(("ffree st1".into(), Box::new(|a: &mut CodeAssembler| a.ffree(ST1))));
    insns.push(("ffree st7".into(), Box::new(|a: &mut CodeAssembler| a.ffree(ST7))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// FPU control
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_control() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    insns.push(("fninit".into(), Box::new(|a: &mut CodeAssembler| a.fninit())));
    insns.push(("fwait".into(), Box::new(|a: &mut CodeAssembler| a.fwait())));
    insns.push(("fnclex".into(), Box::new(|a: &mut CodeAssembler| a.fnclex())));

    // fldcw / fnstcw [mem]
    for (addr, nasm_mem) in mems16() {
        let asm = format!("fldcw {}", nasm_mem);
        let a2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fldcw(a2))));
        let asm = format!("fnstcw {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.fnstcw(addr))));
    }

    // fnstsw ax
    insns.push(("fnstsw ax".into(), Box::new(|a: &mut CodeAssembler| a.fnstsw_ax())));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// FPU conditional move
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_fpu_cmov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg) -> Result<()>)] = &[
        ("fcmovb",   |a, s| a.fcmovb(s)),
        ("fcmove",   |a, s| a.fcmove(s)),
        ("fcmovbe",  |a, s| a.fcmovbe(s)),
        ("fcmovu",   |a, s| a.fcmovu(s)),
        ("fcmovnb",  |a, s| a.fcmovnb(s)),
        ("fcmovne",  |a, s| a.fcmovne(s)),
        ("fcmovnbe", |a, s| a.fcmovnbe(s)),
        ("fcmovnu",  |a, s| a.fcmovnu(s)),
    ];

    for &(name, op_fn) in ops {
        for &(st, sn) in &[(ST1, "st1"), (ST3, "st3"), (ST7, "st7")] {
            let asm = format!("{} st0, {}", name, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, st))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}
