/// GPR instruction NASM conformance tests (make_nm port).
/// Tests all valid operand combinations against NASM reference output.

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

/// Type alias for instruction test pairs.
type NmPair = (String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>);

// ═══════════════════════════════════════════════════════════════════
// 0-operand instructions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_simple() {
    let nasm = skip_if_no_nasm!();
    let insns: Vec<NmPair> = vec![
        ("nop".into(), Box::new(|a: &mut CodeAssembler| a.nop())),
        ("ret".into(), Box::new(|a: &mut CodeAssembler| a.ret())),
        ("clc".into(), Box::new(|a: &mut CodeAssembler| a.clc())),
        ("stc".into(), Box::new(|a: &mut CodeAssembler| a.stc())),
        ("cld".into(), Box::new(|a: &mut CodeAssembler| a.cld())),
        ("std".into(), Box::new(|a: &mut CodeAssembler| a.std_())),
        ("cmc".into(), Box::new(|a: &mut CodeAssembler| a.cmc())),
        ("hlt".into(), Box::new(|a: &mut CodeAssembler| a.hlt())),
        ("ud2".into(), Box::new(|a: &mut CodeAssembler| a.ud2())),
        ("cpuid".into(), Box::new(|a: &mut CodeAssembler| a.cpuid())),
        ("rdtsc".into(), Box::new(|a: &mut CodeAssembler| a.rdtsc())),
        ("rdtscp".into(), Box::new(|a: &mut CodeAssembler| a.rdtscp())),
        ("pause".into(), Box::new(|a: &mut CodeAssembler| a.pause())),
        ("lfence".into(), Box::new(|a: &mut CodeAssembler| a.lfence())),
        ("mfence".into(), Box::new(|a: &mut CodeAssembler| a.mfence())),
        ("sfence".into(), Box::new(|a: &mut CodeAssembler| a.sfence())),
        ("emms".into(), Box::new(|a: &mut CodeAssembler| a.emms())),
        ("cbw".into(), Box::new(|a: &mut CodeAssembler| a.cbw())),
        ("cwde".into(), Box::new(|a: &mut CodeAssembler| a.cwde())),
        ("cdqe".into(), Box::new(|a: &mut CodeAssembler| a.cdqe())),
        ("cdq".into(), Box::new(|a: &mut CodeAssembler| a.cdq())),
        ("cqo".into(), Box::new(|a: &mut CodeAssembler| a.cqo())),
        ("cwd".into(), Box::new(|a: &mut CodeAssembler| a.cwd())),
        ("sahf".into(), Box::new(|a: &mut CodeAssembler| a.sahf())),
        ("lahf".into(), Box::new(|a: &mut CodeAssembler| a.lahf())),
        ("pushfq".into(), Box::new(|a: &mut CodeAssembler| a.pushf())),
        ("popfq".into(), Box::new(|a: &mut CodeAssembler| a.popf())),
        ("leave".into(), Box::new(|a: &mut CodeAssembler| a.leave())),
    ];
    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// push / pop
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_push_pop() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // push reg64
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm = format!("push {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.push(reg))));
    }
    // pop reg64
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm = format!("pop {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pop(reg))));
    }
    // push reg16
    for &(reg, name) in REGS16.iter() {
        let asm = format!("push {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.push(reg))));
    }
    // pop reg16
    for &(reg, name) in REGS16.iter() {
        let asm = format!("pop {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.pop(reg))));
    }
    // push imm
    for imm in [0i32, 1, 0x7F, -1, -128, 0x100, 0x12345678] {
        let asm = format!("push 0x{:x}", imm as u32);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.push_imm(imm))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// inc / dec
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_inc_dec() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // inc/dec reg32
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm = format!("inc {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.inc(reg))));
    }
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm = format!("dec {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.dec(reg))));
    }
    // inc/dec reg64
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm = format!("inc {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.inc(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm = format!("dec {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.dec(reg))));
    }
    // inc/dec reg8
    for &(reg, name) in REGS8.iter().chain(REGS8_EXT.iter()) {
        let asm = format!("inc {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.inc(reg))));
    }
    for &(reg, name) in REGS8.iter().chain(REGS8_EXT.iter()) {
        let asm = format!("dec {}", name);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.dec(reg))));
    }
    // inc/dec mem
    for (addr, nasm_str) in mems32() {
        let asm = format!("inc {}", nasm_str);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.inc(addr))));
    }
    for (addr, nasm_str) in mems64() {
        let asm = format!("dec {}", nasm_str);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.dec(addr))));
    }
    for (addr, nasm_str) in mems8() {
        let asm = format!("inc {}", nasm_str);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.inc(addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// ALU: add/sub/and/or/xor/cmp/adc/sbb — reg,reg
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_alu_rr32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("add", |a, d, s| a.add(d, s)),
        ("sub", |a, d, s| a.sub(d, s)),
        ("and", |a, d, s| a.and_(d, s)),
        ("or",  |a, d, s| a.or_(d, s)),
        ("xor", |a, d, s| a.xor_(d, s)),
        ("cmp", |a, d, s| a.cmp(d, s)),
        ("adc", |a, d, s| a.adc(d, s)),
        ("sbb", |a, d, s| a.sbb(d, s)),
    ];

    // Representative 32-bit pairs (not full Cartesian to keep manageable)
    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"),
        (EBX, "ebx", EDX, "edx"),
        (ESI, "esi", EDI, "edi"),
        (R8D, "r8d", R9D, "r9d"),
        (EAX, "eax", R15D, "r15d"),
        (R12D, "r12d", EBP, "ebp"),
        (ESP, "esp", EAX, "eax"),
    ];

    for &(op_name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", op_name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_alu_rr64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("add", |a, d, s| a.add(d, s)),
        ("sub", |a, d, s| a.sub(d, s)),
        ("and", |a, d, s| a.and_(d, s)),
        ("or",  |a, d, s| a.or_(d, s)),
        ("xor", |a, d, s| a.xor_(d, s)),
        ("cmp", |a, d, s| a.cmp(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"),
        (RBX, "rbx", RDX, "rdx"),
        (RSI, "rsi", RDI, "rdi"),
        (R8, "r8", R9, "r9"),
        (RAX, "rax", R15, "r15"),
        (R12, "r12", RBP, "rbp"),
    ];

    for &(op_name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", op_name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_alu_rr8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("add", |a, d, s| a.add(d, s)),
        ("sub", |a, d, s| a.sub(d, s)),
        ("cmp", |a, d, s| a.cmp(d, s)),
        ("xor", |a, d, s| a.xor_(d, s)),
    ];

    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (AL, "al", CL, "cl"),
        (BL, "bl", DL, "dl"),
        (R8B, "r8b", R9B, "r9b"),
        (AL, "al", R15B, "r15b"),
    ];

    for &(op_name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs {
            let asm = format!("{} {}, {}", op_name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// ALU: reg, imm
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_alu_ri32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, i32) -> Result<()>)] = &[
        ("add", |a, d, i| a.add(d, i)),
        ("sub", |a, d, i| a.sub(d, i)),
        ("and", |a, d, i| a.and_(d, i)),
        ("or",  |a, d, i| a.or_(d, i)),
        ("xor", |a, d, i| a.xor_(d, i)),
        ("cmp", |a, d, i| a.cmp(d, i)),
        ("adc", |a, d, i| a.adc(d, i)),
        ("sbb", |a, d, i| a.sbb(d, i)),
    ];

    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (R8D, "r8d"), (R15D, "r15d"),
    ];
    let imms: &[i32] = &[0, 1, 0x7F, 0x80, 0x1234];

    for &(op_name, op_fn) in ops {
        for &(reg, rn) in regs {
            for &imm in imms {
                let asm = format!("{} {}, 0x{:x}", op_name, rn, imm as u32);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg, imm))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_alu_ri64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, i32) -> Result<()>)] = &[
        ("add", |a, d, i| a.add(d, i)),
        ("sub", |a, d, i| a.sub(d, i)),
        ("and", |a, d, i| a.and_(d, i)),
        ("or",  |a, d, i| a.or_(d, i)),
        ("xor", |a, d, i| a.xor_(d, i)),
        ("cmp", |a, d, i| a.cmp(d, i)),
    ];

    let regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (R8, "r8"), (R15, "r15"),
    ];
    let imms: &[i32] = &[0, 1, 0x7F, 0x80, 0x1234];

    for &(op_name, op_fn) in ops {
        for &(reg, rn) in regs {
            for &imm in imms {
                let asm = format!("{} {}, 0x{:x}", op_name, rn, imm as u32);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg, imm))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// ALU: reg, mem / mem, reg
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_alu_rm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Address) -> Result<()>)] = &[
        ("add", |a, d, m| a.add(d, m)),
        ("sub", |a, d, m| a.sub(d, m)),
        ("and", |a, d, m| a.and_(d, m)),
        ("or",  |a, d, m| a.or_(d, m)),
        ("xor", |a, d, m| a.xor_(d, m)),
        ("cmp", |a, d, m| a.cmp(d, m)),
    ];

    let regs32: &[(Reg, &str)] = &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")];

    for &(op_name, op_fn) in ops {
        for &(reg, rn) in regs32 {
            for (addr, nasm_mem) in mems32() {
                let asm = format!("{} {}, {}", op_name, rn, nasm_mem);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg, addr))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_alu_mr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Address, Reg) -> Result<()>)] = &[
        ("add", |a, m, s| a.add(m, s)),
        ("sub", |a, m, s| a.sub(m, s)),
        ("and", |a, m, s| a.and_(m, s)),
        ("or",  |a, m, s| a.or_(m, s)),
        ("xor", |a, m, s| a.xor_(m, s)),
        ("cmp", |a, m, s| a.cmp(m, s)),
    ];

    let regs32: &[(Reg, &str)] = &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")];

    for &(op_name, op_fn) in ops {
        for (addr, nasm_mem) in mems32() {
            for &(reg, rn) in regs32 {
                let asm = format!("{} {}, {}", op_name, nasm_mem, rn);
                let addr = addr.clone();
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, reg))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_alu_mi() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Address, i32) -> Result<()>)] = &[
        ("add", |a, m, i| a.add(m, i)),
        ("sub", |a, m, i| a.sub(m, i)),
        ("and", |a, m, i| a.and_(m, i)),
        ("or",  |a, m, i| a.or_(m, i)),
        ("xor", |a, m, i| a.xor_(m, i)),
        ("cmp", |a, m, i| a.cmp(m, i)),
    ];

    let imms: &[i32] = &[0, 1, 0x7F, 0x1234];

    for &(op_name, op_fn) in ops {
        for (addr, nasm_mem) in mems32() {
            for &imm in imms {
                let asm = format!("{} {}, 0x{:x}", op_name, nasm_mem, imm as u32);
                let addr = addr.clone();
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, imm))));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// mov
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_mov_rr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // 32-bit
    for &(dst, dn) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &(src, sn) in REGS32.iter().chain(REGS32_EXT.iter()) {
            let asm = format!("mov {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(dst, src))));
        }
    }
    // 64-bit (representative)
    let r64_pairs: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
        (RBP, "rbp", RSP, "rsp"), (RDI, "rdi", R12, "r12"),
    ];
    for &(dst, dn, src, sn) in r64_pairs {
        let asm = format!("mov {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(dst, src))));
    }
    // 8-bit
    let r8_pairs: &[(Reg, &str, Reg, &str)] = &[
        (AL, "al", CL, "cl"), (BL, "bl", DL, "dl"),
        (R8B, "r8b", R9B, "r9b"),
    ];
    for &(dst, dn, src, sn) in r8_pairs {
        let asm = format!("mov {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(dst, src))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_mov_ri() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // mov reg32, imm32
    for &(reg, rn) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for imm in [0i32, 1, 0x7F, 0xFF, 0x12345678] {
            let asm = format!("mov {}, 0x{:x}", rn, imm as u32);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(reg, imm))));
        }
    }
    // mov reg8, imm8
    for &(reg, rn) in REGS8.iter() {
        for imm in [0i32, 1, 0x7F, 0xFF] {
            let asm = format!("mov {}, 0x{:x}", rn, imm as u8);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(reg, imm))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_mov_rm_mr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let regs32: &[(Reg, &str)] = &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d"), (R15D, "r15d")];
    let regs64: &[(Reg, &str)] = &[(RAX, "rax"), (RCX, "rcx"), (R8, "r8")];

    // mov reg32, mem32
    for &(reg, rn) in regs32 {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("mov {}, {}", rn, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(reg, addr))));
        }
    }
    // mov mem32, reg32
    for (addr, nasm_mem) in mems32() {
        for &(reg, rn) in regs32 {
            let asm = format!("mov {}, {}", nasm_mem, rn);
            let addr = addr.clone();
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(addr, reg))));
        }
    }
    // mov reg64, mem64
    for &(reg, rn) in regs64 {
        for (addr, nasm_mem) in mems64() {
            let asm = format!("mov {}, {}", rn, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(reg, addr))));
        }
    }
    // mov mem64, reg64
    for (addr, nasm_mem) in mems64() {
        for &(reg, rn) in regs64 {
            let asm = format!("mov {}, {}", nasm_mem, rn);
            let addr = addr.clone();
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(addr, reg))));
        }
    }
    // mov mem32, imm32
    for (addr, nasm_mem) in mems32() {
        for imm in [0i32, 1, 0x42, 0x12345678] {
            let asm = format!("mov {}, 0x{:x}", nasm_mem, imm as u32);
            let addr = addr.clone();
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.mov(addr, imm))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// Shift: shl/shr/sar/rol/ror/rcl/rcr
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_shift() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, u8) -> Result<()>)] = &[
        ("shl", |a, r, i| a.shl(r, i)),
        ("shr", |a, r, i| a.shr(r, i)),
        ("sar", |a, r, i| a.sar(r, i)),
        ("rol", |a, r, i| a.rol(r, i)),
        ("ror", |a, r, i| a.ror(r, i)),
        ("rcl", |a, r, i| a.rcl(r, i)),
        ("rcr", |a, r, i| a.rcr(r, i)),
    ];

    let regs32: &[(Reg, &str)] = &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")];
    let regs64: &[(Reg, &str)] = &[(RAX, "rax"), (R8, "r8")];
    let counts: &[u8] = &[1, 4, 31];

    for &(op_name, op_fn) in ops {
        for &(reg, rn) in regs32 {
            for &cnt in counts {
                let asm = format!("{} {}, {}", op_name, rn, cnt);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg, cnt))));
            }
        }
        for &(reg, rn) in regs64 {
            for &cnt in counts {
                let asm = format!("{} {}, {}", op_name, rn, cnt);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg, cnt))));
            }
        }
    }

    // Shift mem
    let shift_mem_ops: &[(&str, fn(&mut CodeAssembler, Address, u8) -> Result<()>)] = &[
        ("shl", |a, m, i| a.shl(m, i)),
        ("shr", |a, m, i| a.shr(m, i)),
        ("sar", |a, m, i| a.sar(m, i)),
    ];
    for &(op_name, op_fn) in shift_mem_ops {
        for (addr, nasm_mem) in mems32() {
            for &cnt in &[1u8, 5] {
                let asm = format!("{} {}, {}", op_name, nasm_mem, cnt);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, cnt))));
            }
        }
    }

    // Shift/rotate by CL register
    let cl_ops: &[(&str, fn(&mut CodeAssembler, Reg) -> Result<()>)] = &[
        ("shl", |a, r| a.shl_cl(r)),
        ("shr", |a, r| a.shr_cl(r)),
        ("sar", |a, r| a.sar_cl(r)),
        ("rol", |a, r| a.rol_cl(r)),
        ("ror", |a, r| a.ror_cl(r)),
        ("rcl", |a, r| a.rcl_cl(r)),
        ("rcr", |a, r| a.rcr_cl(r)),
    ];

    for &(op_name, op_fn) in cl_ops {
        for &(reg, rn) in regs32 {
            let asm = format!("{} {}, cl", op_name, rn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg))));
        }
        for &(reg, rn) in regs64 {
            let asm = format!("{} {}, cl", op_name, rn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg))));
        }
    }

    // Shift/rotate mem by CL
    let cl_mem_ops: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>)] = &[
        ("shl", |a, m| a.shl_cl(m)),
        ("shr", |a, m| a.shr_cl(m)),
        ("sar", |a, m| a.sar_cl(m)),
        ("rol", |a, m| a.rol_cl(m)),
        ("ror", |a, m| a.ror_cl(m)),
        ("rcl", |a, m| a.rcl_cl(m)),
        ("rcr", |a, m| a.rcr_cl(m)),
    ];
    for &(op_name, op_fn) in cl_mem_ops {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} {}, cl", op_name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// test
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_test() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // test reg32, reg32
    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (EBX, "ebx", EDX, "edx"),
        (R8D, "r8d", R9D, "r9d"), (EAX, "eax", R15D, "r15d"),
    ];
    for &(dst, dn, src, sn) in pairs32 {
        let asm = format!("test {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.test(dst, src))));
    }
    // test reg64, reg64
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];
    for &(dst, dn, src, sn) in pairs64 {
        let asm = format!("test {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.test(dst, src))));
    }
    // test reg32, imm32
    for &(reg, rn) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
        for imm in [1i32, 0xFF, 0x12345678] {
            let asm = format!("test {}, 0x{:x}", rn, imm as u32);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.test(reg, imm))));
        }
    }
    // test mem, reg
    for (addr, nasm_mem) in mems32() {
        let asm = format!("test {}, ecx", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.test(addr, ECX))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// xchg
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_xchg() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // xchg reg32, reg32 (including EAX short forms)
    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (EAX, "eax", EBX, "ebx"),
        (ECX, "ecx", EDX, "edx"), (R8D, "r8d", R9D, "r9d"),
        (EAX, "eax", R8D, "r8d"),
    ];
    for &(dst, dn, src, sn) in pairs32 {
        let asm = format!("xchg {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.xchg(dst, src))));
    }
    // xchg reg64, reg64
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
        (RAX, "rax", R8, "r8"),
    ];
    for &(dst, dn, src, sn) in pairs64 {
        let asm = format!("xchg {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.xchg(dst, src))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// lea
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_lea() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (R8D, "r8d"),
        (RAX, "rax"), (RCX, "rcx"), (R8, "r8"),
    ];
    let addrs: &[(Address, &str)] = &[
        (ptr(RAX.into()), "[rax]"),
        (ptr(RCX + 0x10), "[rcx+0x10]"),
        (ptr(RBP + RCX * 4 + 0x100), "[rbp+rcx*4+0x100]"),
        (ptr(R13 + R14 * 8 + 0x20), "[r13+r14*8+0x20]"),
        (ptr(RAX + RAX * 1), "[rax+rax]"),
    ];

    for &(reg, rn) in regs {
        for (addr, nasm_mem) in addrs {
            let asm = format!("lea {}, {}", rn, nasm_mem);
            let addr = addr.clone();
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.lea(reg, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// movzx / movsx / movsxd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_movzx_movsx() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // movzx reg32, reg8
    for &(dst, dn) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
        for &(src, sn) in &[(AL, "al"), (CL, "cl"), (R8B, "r8b")] {
            let asm = format!("movzx {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movzx(dst, src))));
        }
    }
    // movzx reg32, reg16
    for &(dst, dn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for &(src, sn) in &[(AX, "ax"), (CX, "cx")] {
            let asm = format!("movzx {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movzx(dst, src))));
        }
    }
    // movzx reg32, mem8
    for &(reg, rn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for (addr, nasm_mem) in mems8() {
            let asm = format!("movzx {}, {}", rn, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movzx(reg, addr))));
        }
    }
    // movsx reg32, reg8
    for &(dst, dn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for &(src, sn) in &[(AL, "al"), (CL, "cl"), (R8B, "r8b")] {
            let asm = format!("movsx {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsx(dst, src))));
        }
    }
    // movsx reg32, reg16
    for &(dst, dn) in &[(EAX, "eax"), (R8D, "r8d")] {
        for &(src, sn) in &[(AX, "ax"), (CX, "cx")] {
            let asm = format!("movsx {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsx(dst, src))));
        }
    }
    // movsxd reg64, reg32
    for &(dst, dn) in &[(RAX, "rax"), (R8, "r8")] {
        for &(src, sn) in &[(EAX, "eax"), (ECX, "ecx"), (R8D, "r8d")] {
            let asm = format!("movsxd {}, {}", dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movsxd(dst, src))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// CMOVcc
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_cmov() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let cmovs: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("cmovo",  |a, d, s| a.cmovo(d, s)),
        ("cmovno", |a, d, s| a.cmovno(d, s)),
        ("cmovb",  |a, d, s| a.cmovb(d, s)),
        ("cmovae", |a, d, s| a.cmovae(d, s)),
        ("cmove",  |a, d, s| a.cmove(d, s)),
        ("cmovne", |a, d, s| a.cmovne(d, s)),
        ("cmovbe", |a, d, s| a.cmovbe(d, s)),
        ("cmova",  |a, d, s| a.cmova(d, s)),
        ("cmovs",  |a, d, s| a.cmovs(d, s)),
        ("cmovns", |a, d, s| a.cmovns(d, s)),
        ("cmovp",  |a, d, s| a.cmovp(d, s)),
        ("cmovnp", |a, d, s| a.cmovnp(d, s)),
        ("cmovl",  |a, d, s| a.cmovl(d, s)),
        ("cmovge", |a, d, s| a.cmovge(d, s)),
        ("cmovle", |a, d, s| a.cmovle(d, s)),
        ("cmovg",  |a, d, s| a.cmovg(d, s)),
    ];

    // reg32, reg32
    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (R8D, "r8d", R15D, "r15d"),
        (EBX, "ebx", R12D, "r12d"),
    ];
    for &(name, op_fn) in cmovs {
        for &(dst, dn, src, sn) in pairs32 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }
    // reg64, reg64
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];
    for &(name, op_fn) in cmovs {
        for &(dst, dn, src, sn) in pairs64 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }
    // reg32, mem32
    for &(name, _) in cmovs.iter().take(4) {
        // Test a subset with mem to keep test size reasonable
        let cmov_mem: fn(&mut CodeAssembler, Reg, Address) -> Result<()> = match name {
            "cmovo"  => |a, d, m| a.cmovo(d, m),
            "cmovno" => |a, d, m| a.cmovno(d, m),
            "cmovb"  => |a, d, m| a.cmovb(d, m),
            "cmovae" => |a, d, m| a.cmovae(d, m),
            _ => unreachable!(),
        };
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} eax, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| cmov_mem(a, EAX, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// SETcc
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_setcc() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let sets: &[(&str, fn(&mut CodeAssembler, Reg) -> Result<()>)] = &[
        ("seto",  |a, r| a.seto(r)),
        ("setno", |a, r| a.setno(r)),
        ("setb",  |a, r| a.setb(r)),
        ("setae", |a, r| a.setae(r)),
        ("sete",  |a, r| a.sete(r)),
        ("setne", |a, r| a.setne(r)),
        ("setbe", |a, r| a.setbe(r)),
        ("seta",  |a, r| a.seta(r)),
        ("sets",  |a, r| a.sets(r)),
        ("setns", |a, r| a.setns(r)),
        ("setp",  |a, r| a.setp(r)),
        ("setnp", |a, r| a.setnp(r)),
        ("setl",  |a, r| a.setl(r)),
        ("setge", |a, r| a.setge(r)),
        ("setle", |a, r| a.setle(r)),
        ("setg",  |a, r| a.setg(r)),
    ];

    // set* reg8
    for &(name, op_fn) in sets {
        for &(reg, rn) in REGS8.iter().chain(REGS8_EXT.iter()) {
            let asm = format!("{} {}", name, rn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg))));
        }
    }
    // set* mem8 (subset)
    let sets_mem: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>)] = &[
        ("seto",  |a, m| a.seto(m)),
        ("setb",  |a, m| a.setb(m)),
        ("sete",  |a, m| a.sete(m)),
        ("setg",  |a, m| a.setg(m)),
    ];
    for &(name, op_fn) in sets_mem {
        for (addr, nasm_mem) in mems8() {
            let asm = format!("{} {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// bt / bts / btr / btc — reg, reg form
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_bt() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("bt",  |a, d, s| a.bt(d, s)),
        ("bts", |a, d, s| a.bts(d, s)),
        ("btr", |a, d, s| a.btr(d, s)),
        ("btc", |a, d, s| a.btc(d, s)),
    ];

    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (R8D, "r8d", R9D, "r9d"),
        (EBX, "ebx", R15D, "r15d"),
    ];
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs32 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
        for &(dst, dn, src, sn) in pairs64 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }

    // bt mem, reg
    let ops_mem: &[(&str, fn(&mut CodeAssembler, Address, Reg) -> Result<()>)] = &[
        ("bt",  |a, m, s| a.bt(m, s)),
        ("bts", |a, m, s| a.bts(m, s)),
    ];
    for &(name, op_fn) in ops_mem {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} {}, ecx", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, ECX))));
        }
    }

    // bt/bts/btr/btc reg, imm8
    let imm_ops: &[(&str, fn(&mut CodeAssembler, Reg, u8) -> Result<()>)] = &[
        ("bt",  |a, r, i| a.bt_imm(r, i)),
        ("bts", |a, r, i| a.bts_imm(r, i)),
        ("btr", |a, r, i| a.btr_imm(r, i)),
        ("btc", |a, r, i| a.btc_imm(r, i)),
    ];
    let imms: &[u8] = &[0, 5, 31];
    for &(name, op_fn) in imm_ops {
        for &(dst, dn, _, _) in pairs32 {
            for &imm in imms {
                let asm = format!("{} {}, {}", name, dn, imm);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, imm))));
            }
        }
        for &(dst, dn, _, _) in pairs64 {
            for &imm in imms {
                let asm = format!("{} {}, {}", name, dn, imm);
                insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, imm))));
            }
        }
    }

    // bt/bts/btr/btc mem, imm8
    let imm_mem_ops: &[(&str, fn(&mut CodeAssembler, Address, u8) -> Result<()>)] = &[
        ("bt",  |a, m, i| a.bt_imm(m, i)),
        ("bts", |a, m, i| a.bts_imm(m, i)),
        ("btr", |a, m, i| a.btr_imm(m, i)),
        ("btc", |a, m, i| a.btc_imm(m, i)),
    ];
    for &(name, op_fn) in imm_mem_ops {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} {}, 5", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr, 5))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// bsf / bsr / popcnt / lzcnt / tzcnt
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_bsf_bsr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Reg, Reg) -> Result<()>)] = &[
        ("bsf",    |a, d, s| a.bsf(d, s)),
        ("bsr",    |a, d, s| a.bsr(d, s)),
        ("popcnt", |a, d, s| a.popcnt(d, s)),
        ("lzcnt",  |a, d, s| a.lzcnt(d, s)),
        ("tzcnt",  |a, d, s| a.tzcnt(d, s)),
    ];

    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (R8D, "r8d", R9D, "r9d"),
    ];
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];

    for &(name, op_fn) in ops {
        for &(dst, dn, src, sn) in pairs32 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
        for &(dst, dn, src, sn) in pairs64 {
            let asm = format!("{} {}, {}", name, dn, sn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, dst, src))));
        }
    }
    // reg, mem
    let ops_mem: &[(&str, fn(&mut CodeAssembler, Reg, Address) -> Result<()>)] = &[
        ("bsf",    |a, d, m| a.bsf(d, m)),
        ("bsr",    |a, d, m| a.bsr(d, m)),
        ("popcnt", |a, d, m| a.popcnt(d, m)),
    ];
    for &(name, op_fn) in ops_mem {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} eax, {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, EAX, addr))));
        }
    }

    compare_nasm_batch_normalized(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// shld / shrd
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_shld_shrd() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // shld/shrd reg, reg, imm8
    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (R8D, "r8d", R9D, "r9d"),
    ];
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];

    for &(dst, dn, src, sn) in pairs32 {
        for &imm in &[1u8, 4, 16] {
            let asm = format!("shld {}, {}, {}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shld(dst, src, imm))));
            let asm = format!("shrd {}, {}, {}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shrd(dst, src, imm))));
        }
    }
    for &(dst, dn, src, sn) in pairs64 {
        for &imm in &[1u8, 4, 16] {
            let asm = format!("shld {}, {}, {}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shld(dst, src, imm))));
            let asm = format!("shrd {}, {}, {}", dn, sn, imm);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shrd(dst, src, imm))));
        }
    }
    // shld/shrd reg, reg, CL
    for &(dst, dn, src, sn) in pairs32 {
        let asm = format!("shld {}, {}, cl", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shld_cl(dst, src))));
        let asm = format!("shrd {}, {}, cl", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.shrd_cl(dst, src))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// mul / div / idiv / neg / not
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_mul_div() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops_reg: &[(&str, fn(&mut CodeAssembler, Reg) -> Result<()>)] = &[
        ("mul",  |a, r| a.mul(r)),
        ("div",  |a, r| a.div(r)),
        ("idiv", |a, r| a.idiv(r)),
        ("neg",  |a, r| a.neg(r)),
        ("not",  |a, r| a.not_(r)),
    ];

    let regs32: &[(Reg, &str)] = &[(ECX, "ecx"), (EBX, "ebx"), (R8D, "r8d"), (R15D, "r15d")];
    let regs64: &[(Reg, &str)] = &[(RCX, "rcx"), (R8, "r8")];

    for &(name, op_fn) in ops_reg {
        for &(reg, rn) in regs32 {
            let asm = format!("{} {}", name, rn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg))));
        }
        for &(reg, rn) in regs64 {
            let asm = format!("{} {}", name, rn);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, reg))));
        }
    }
    // mul/div/neg/not mem
    let ops_mem: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>)] = &[
        ("mul",  |a, m| a.mul(m)),
        ("div",  |a, m| a.div(m)),
        ("neg",  |a, m| a.neg(m)),
        ("not",  |a, m| a.not_(m)),
    ];
    for &(name, op_fn) in ops_mem {
        for (addr, nasm_mem) in mems32() {
            let asm = format!("{} {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// imul (2-operand)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_imul() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // imul reg32, reg32
    let pairs32: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", ECX, "ecx"), (R8D, "r8d", R9D, "r9d"),
        (EBX, "ebx", R15D, "r15d"), (ESP, "esp", EBP, "ebp"),
    ];
    for &(dst, dn, src, sn) in pairs32 {
        let asm = format!("imul {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.imul(dst, src))));
    }
    // imul reg64, reg64
    let pairs64: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"), (R8, "r8", R15, "r15"),
    ];
    for &(dst, dn, src, sn) in pairs64 {
        let asm = format!("imul {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.imul(dst, src))));
    }
    // imul reg32, mem32
    for (addr, nasm_mem) in mems32() {
        let asm = format!("imul eax, {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.imul(EAX, addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// Misc: cmpxchg / xadd / bswap / enter
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_misc() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // cmpxchg reg, reg
    let pairs: &[(Reg, &str, Reg, &str)] = &[
        (ECX, "ecx", EAX, "eax"), (R8D, "r8d", R9D, "r9d"),
        (EBX, "ebx", EDX, "edx"),
    ];
    for &(dst, dn, src, sn) in pairs {
        let asm = format!("cmpxchg {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.cmpxchg(dst, src))));
    }
    // cmpxchg mem, reg
    for (addr, nasm_mem) in mems32() {
        let asm = format!("cmpxchg {}, ecx", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.cmpxchg(addr, ECX))));
    }

    // xadd reg, reg
    for &(dst, dn, src, sn) in &[(ECX, "ecx", EAX, "eax"), (R8D, "r8d", R9D, "r9d")] {
        let asm = format!("xadd {}, {}", dn, sn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.xadd(dst, src))));
    }
    // xadd mem, reg
    for (addr, nasm_mem) in mems32() {
        let asm = format!("xadd {}, ecx", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.xadd(addr, ECX))));
    }

    // bswap
    for &(reg, rn) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm = format!("bswap {}", rn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.bswap(reg))));
    }
    for &(reg, rn) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm = format!("bswap {}", rn);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.bswap(reg))));
    }

    // enter
    insns.push(("enter 0, 0".into(), Box::new(|a: &mut CodeAssembler| a.enter(0, 0))));
    insns.push(("enter 16, 0".into(), Box::new(|a: &mut CodeAssembler| a.enter(16, 0))));
    insns.push(("enter 256, 1".into(), Box::new(|a: &mut CodeAssembler| a.enter(256, 1))));

    // ret imm
    insns.push(("ret 8".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(8))));
    insns.push(("ret 16".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(16))));

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// String operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_string() {
    let nasm = skip_if_no_nasm!();
    let insns: Vec<NmPair> = vec![
        ("lodsb".into(), Box::new(|a: &mut CodeAssembler| a.lodsb())),
        ("lodsw".into(), Box::new(|a: &mut CodeAssembler| a.lodsw())),
        ("lodsd".into(), Box::new(|a: &mut CodeAssembler| a.lodsd())),
        ("lodsq".into(), Box::new(|a: &mut CodeAssembler| a.lodsq())),
        ("stosb".into(), Box::new(|a: &mut CodeAssembler| a.stosb())),
        ("stosw".into(), Box::new(|a: &mut CodeAssembler| a.stosw())),
        ("stosd".into(), Box::new(|a: &mut CodeAssembler| a.stosd())),
        ("stosq".into(), Box::new(|a: &mut CodeAssembler| a.stosq())),
        ("movsb".into(), Box::new(|a: &mut CodeAssembler| a.movsb())),
        ("movsw".into(), Box::new(|a: &mut CodeAssembler| a.movsw())),
        ("movsd".into(), Box::new(|a: &mut CodeAssembler| a.movsd_string())),
        ("movsq".into(), Box::new(|a: &mut CodeAssembler| a.movsq())),
        ("scasb".into(), Box::new(|a: &mut CodeAssembler| a.scasb())),
        ("scasw".into(), Box::new(|a: &mut CodeAssembler| a.scasw())),
        ("scasd".into(), Box::new(|a: &mut CodeAssembler| a.scasd())),
        ("scasq".into(), Box::new(|a: &mut CodeAssembler| a.scasq())),
        ("cmpsb".into(), Box::new(|a: &mut CodeAssembler| a.cmpsb())),
        ("cmpsw".into(), Box::new(|a: &mut CodeAssembler| a.cmpsw())),
        ("cmpsq".into(), Box::new(|a: &mut CodeAssembler| a.cmpsq())),
        // rep prefix combinations
        ("rep stosb".into(), Box::new(|a: &mut CodeAssembler| { a.rep()?; a.stosb() })),
        ("rep stosd".into(), Box::new(|a: &mut CodeAssembler| { a.rep()?; a.stosd() })),
        ("rep movsb".into(), Box::new(|a: &mut CodeAssembler| { a.rep()?; a.movsb() })),
        ("rep movsq".into(), Box::new(|a: &mut CodeAssembler| { a.rep()?; a.movsq() })),
        ("repe cmpsb".into(), Box::new(|a: &mut CodeAssembler| { a.repe()?; a.cmpsb() })),
        ("repne scasb".into(), Box::new(|a: &mut CodeAssembler| { a.repne()?; a.scasb() })),
    ];
    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// Non-temporal stores and prefetch
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_nontemporal() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    // movnti [mem], reg32/64
    for (addr, nasm_mem) in mems_nosizeptr() {
        let asm = format!("movnti {}, eax", nasm_mem);
        let addr2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movnti(addr2, EAX))));
        let asm = format!("movnti {}, rax", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.movnti(addr, RAX))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nm_prefetch() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    let ops: &[(&str, fn(&mut CodeAssembler, Address) -> Result<()>)] = &[
        ("prefetchnta", |a, m| a.prefetchnta(m)),
        ("prefetcht0",  |a, m| a.prefetcht0(m)),
        ("prefetcht1",  |a, m| a.prefetcht1(m)),
        ("prefetcht2",  |a, m| a.prefetcht2(m)),
        ("clflush",     |a, m| a.clflush(m)),
        ("clflushopt",  |a, m| a.clflushopt(m)),
    ];

    for &(name, op_fn) in ops {
        for (addr, nasm_mem) in mems_nosizeptr() {
            let asm = format!("{} {}", name, nasm_mem);
            insns.push((asm, Box::new(move |a: &mut CodeAssembler| op_fn(a, addr))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ═══════════════════════════════════════════════════════════════════
// stmxcsr / ldmxcsr
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_nm_mxcsr() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<NmPair> = Vec::new();

    for (addr, nasm_mem) in mems_nosizeptr() {
        let asm = format!("stmxcsr {}", nasm_mem);
        let addr2 = addr.clone();
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.stmxcsr(addr2))));
        let asm = format!("ldmxcsr {}", nasm_mem);
        insns.push((asm, Box::new(move |a: &mut CodeAssembler| a.ldmxcsr(addr))));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
