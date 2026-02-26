/// GPR instruction tests validated against NASM reference assembler.
/// Tests are skipped if NASM is not found.

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

// ─── mov reg, reg (32-bit) ──────────────────────────────────────

#[test]
fn test_nasm_mov_reg32_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(dst, dst_name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &(src, src_name) in REGS32.iter().chain(REGS32_EXT.iter()) {
            let asm_text = format!("mov {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.mov(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── mov reg, reg (64-bit) ──────────────────────────────────────

#[test]
fn test_nasm_mov_reg64_reg64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(dst, dst_name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        for &(src, src_name) in REGS64.iter().chain(REGS64_EXT.iter()) {
            let asm_text = format!("mov {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.mov(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── mov reg, imm ───────────────────────────────────────────────

#[test]
fn test_nasm_mov_reg32_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[0, 1, 42, 0x7F, 0x80, 0xFF, 0x100, 0x12345678];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("mov {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.mov(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_mov_reg8_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[0, 1, 0x7F, 0xFF];
    for &(reg, name) in REGS8.iter() {
        for &imm in imm_values {
            let asm_text = format!("mov {}, 0x{:x}", name, imm as u8);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.mov(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── ALU ops: add, sub, and, or, xor, cmp (reg, reg) ───────────

macro_rules! test_alu_reg_reg {
    ($test_name:ident, $method:ident, $mnemonic:expr, $regs:expr) => {
        #[test]
        fn $test_name() {
            let nasm = skip_if_no_nasm!();
            let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
            for &(dst, dst_name) in $regs {
                for &(src, src_name) in $regs {
                    let asm_text = format!("{} {}, {}", $mnemonic, dst_name, src_name);
                    insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.$method(dst, src))));
                }
            }
            compare_nasm_batch(&nasm, 64, insns);
        }
    };
}

test_alu_reg_reg!(test_nasm_add_reg32, add, "add", &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (ESP, "esp"), (EBP, "ebp"), (ESI, "esi"), (EDI, "edi"),
    (R8D, "r8d"), (R9D, "r9d"), (R10D, "r10d"), (R11D, "r11d"),
    (R12D, "r12d"), (R13D, "r13d"), (R14D, "r14d"), (R15D, "r15d"),
]);

test_alu_reg_reg!(test_nasm_sub_reg64, sub, "sub", &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (RSP, "rsp"), (RBP, "rbp"), (RSI, "rsi"), (RDI, "rdi"),
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R11, "r11"),
    (R12, "r12"), (R13, "r13"), (R14, "r14"), (R15, "r15"),
]);

test_alu_reg_reg!(test_nasm_and_reg32, and_, "and", &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (R8D, "r8d"), (R9D, "r9d"),
]);

test_alu_reg_reg!(test_nasm_or_reg32, or_, "or", &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (R8D, "r8d"), (R9D, "r9d"),
]);

test_alu_reg_reg!(test_nasm_xor_reg32, xor_, "xor", &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (R8D, "r8d"), (R9D, "r9d"),
]);

test_alu_reg_reg!(test_nasm_cmp_reg32, cmp, "cmp", &[
    (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
    (R8D, "r8d"), (R9D, "r9d"),
]);

test_alu_reg_reg!(test_nasm_add_reg64, add, "add", &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R15, "r15"),
]);

test_alu_reg_reg!(test_nasm_xor_reg64, xor_, "xor", &[
    (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"), (RBX, "rbx"),
    (R8, "r8"), (R9, "r9"), (R10, "r10"), (R15, "r15"),
]);

// ─── ALU ops with immediate ─────────────────────────────────────

#[test]
fn test_nasm_add_reg32_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[1, 0x7F, 0x80, 0x12345678];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("add {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.add(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_sub_reg64_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[1, 0x7F, 0x80, 0x12345678];
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("sub {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.sub(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_cmp_reg32_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[0, 1, 0x7F, 0x80, 0xFF, 0x12345678];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("cmp {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.cmp(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_and_reg64_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[1, 0x7F, 0xFF, 0x12345678];
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("and {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.and_(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Shift/rotate operations ────────────────────────────────────

#[test]
fn test_nasm_shl_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let shift_values: &[u8] = &[1, 2, 4, 7, 16, 31];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in shift_values {
            let asm_text = format!("shl {}, {}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.shl(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_shr_reg64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let shift_values: &[u8] = &[1, 2, 4, 8, 32, 63];
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        for &imm in shift_values {
            let asm_text = format!("shr {}, {}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.shr(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_sar_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let shift_values: &[u8] = &[1, 4, 7, 31];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in shift_values {
            let asm_text = format!("sar {}, {}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.sar(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Unary operations: inc, dec, neg, not ───────────────────────

#[test]
fn test_nasm_inc_reg() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm_text = format!("inc {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.inc(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("inc {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.inc(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_dec_reg() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm_text = format!("dec {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.dec(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("dec {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.dec(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_neg_reg() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm_text = format!("neg {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.neg(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("neg {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.neg(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_not_reg() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        let asm_text = format!("not {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.not_(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("not {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.not_(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── push/pop ───────────────────────────────────────────────────

#[test]
fn test_nasm_push_pop_reg64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("push {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.push(reg))));
    }
    for &(reg, name) in REGS64.iter().chain(REGS64_EXT.iter()) {
        let asm_text = format!("pop {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.pop(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_push_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[0, 1, 42, 0x7F, 0x80, 0x12345678];
    for &imm in imm_values {
        let asm_text = format!("push 0x{:x}", imm);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.push_imm(imm))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── test reg, reg / test reg, imm ──────────────────────────────

#[test]
fn test_nasm_test_reg32_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            let asm_text = format!("test {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.test(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_test_reg32_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let imm_values: &[i32] = &[0xFF, 0x12345678];
    for &(reg, name) in REGS32.iter().chain(REGS32_EXT.iter()) {
        for &imm in imm_values {
            let asm_text = format!("test {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.test(reg, imm))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── xchg ───────────────────────────────────────────────────────

#[test]
fn test_nasm_xchg_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            if dst.get_idx() == src.get_idx() {
                continue; // xchg reg, same_reg has special encoding
            }
            let asm_text = format!("xchg {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.xchg(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_xchg_reg64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"),
        (R8, "r8"), (R9, "r9"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            if dst.get_idx() == src.get_idx() {
                continue;
            }
            let asm_text = format!("xchg {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.xchg(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── movzx / movsx ──────────────────────────────────────────────

#[test]
fn test_nasm_movzx_reg32_reg8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let dst_regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    let src_regs: &[(Reg, &str)] = &[
        (AL, "al"), (CL, "cl"), (DL, "dl"), (BL, "bl"),
        (R8B, "r8b"), (R9B, "r9b"),
    ];
    for &(dst, dst_name) in dst_regs {
        for &(src, src_name) in src_regs {
            let asm_text = format!("movzx {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movzx(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_movzx_reg32_reg16() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let dst_regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"),
        (R8D, "r8d"),
    ];
    let src_regs: &[(Reg, &str)] = &[
        (AX, "ax"), (CX, "cx"), (DX, "dx"),
        (R8W, "r8w"),
    ];
    for &(dst, dst_name) in dst_regs {
        for &(src, src_name) in src_regs {
            let asm_text = format!("movzx {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movzx(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_movzx_reg64_reg8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let dst_regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (R8, "r8"),
    ];
    let src_regs: &[(Reg, &str)] = &[
        (AL, "al"), (CL, "cl"), (R8B, "r8b"),
    ];
    for &(dst, dst_name) in dst_regs {
        for &(src, src_name) in src_regs {
            let asm_text = format!("movzx {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movzx(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_movsx_reg32_reg8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let dst_regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    let src_regs: &[(Reg, &str)] = &[
        (AL, "al"), (CL, "cl"), (DL, "dl"),
        (R8B, "r8b"), (R9B, "r9b"),
    ];
    for &(dst, dst_name) in dst_regs {
        for &(src, src_name) in src_regs {
            let asm_text = format!("movsx {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movsx(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_movsxd_reg64_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let dst_regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (R8, "r8"), (R9, "r9"),
    ];
    let src_regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (R8D, "r8d"), (R9D, "r9d"),
    ];
    for &(dst, dst_name) in dst_regs {
        for &(src, src_name) in src_regs {
            let asm_text = format!("movsxd {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.movsxd(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── imul (2-operand form) ──────────────────────────────────────

#[test]
fn test_nasm_imul_reg32_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (EBX, "ebx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            let asm_text = format!("imul {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.imul(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_imul_reg64_reg64() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (RDX, "rdx"),
        (R8, "r8"), (R9, "r9"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            let asm_text = format!("imul {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.imul(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── lea ────────────────────────────────────────────────────────

#[test]
fn test_nasm_lea_variants() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // lea reg, [base + disp]
    let bases: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (RBX, "rbx"),
        (RBP, "rbp"), (RSI, "rsi"), (R8, "r8"), (R13, "r13"),
    ];
    let disps: &[i32] = &[0x10, 0x100];
    for &(base, base_name) in bases {
        for &disp in disps {
            let asm_text = format!("lea rax, [{}+0x{:x}]", base_name, disp);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.lea(RAX, ptr(base + disp))
            })));
        }
    }

    // lea reg, [base + index*scale + disp]
    let indices: &[(Reg, &str)] = &[
        (RCX, "rcx"), (RDX, "rdx"), (RSI, "rsi"), (R8, "r8"),
    ];
    for &(base, base_name) in &[(RAX, "rax"), (RBX, "rbx"), (R8, "r8")] {
        for &(idx, idx_name) in indices {
            for &scale in &[1u8, 2, 4, 8] {
                let asm_text = format!(
                    "lea rax, [{}+{}*{}+0x10]",
                    base_name, idx_name, scale
                );
                insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                    a.lea(RAX, ptr(base + idx * scale + 0x10))
                })));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── mov reg, [mem] and mov [mem], reg ──────────────────────────

#[test]
fn test_nasm_mov_reg_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"), (R8D, "r8d"),
    ];
    let bases: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (RBX, "rbx"),
        (RSP, "rsp"), (RBP, "rbp"), (R8, "r8"), (R12, "r12"), (R13, "r13"),
    ];

    // mov reg, [base]
    for &(reg, reg_name) in regs {
        for &(base, base_name) in bases {
            let asm_text = format!("mov {}, dword [{}]", reg_name, base_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(reg, dword_ptr(base.into()))
            })));
        }
    }

    // mov [base], reg
    for &(reg, reg_name) in regs {
        for &(base, base_name) in bases {
            let asm_text = format!("mov dword [{}], {}", base_name, reg_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(dword_ptr(base.into()), reg)
            })));
        }
    }

    // mov reg, [base + disp8]
    for &(reg, reg_name) in regs {
        for &(base, base_name) in bases {
            let asm_text = format!("mov {}, dword [{}+0x10]", reg_name, base_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(reg, dword_ptr(base + 0x10))
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

#[test]
fn test_nasm_mov_reg64_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let regs: &[(Reg, &str)] = &[
        (RAX, "rax"), (RCX, "rcx"), (R8, "r8"),
    ];
    let bases: &[(Reg, &str)] = &[
        (RAX, "rax"), (RBX, "rbx"), (RSP, "rsp"), (RBP, "rbp"),
        (R8, "r8"), (R12, "r12"), (R13, "r13"),
    ];

    for &(reg, reg_name) in regs {
        for &(base, base_name) in bases {
            let asm_text = format!("mov {}, qword [{}]", reg_name, base_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(reg, qword_ptr(base.into()))
            })));
        }
    }

    for &(reg, reg_name) in regs {
        for &(base, base_name) in bases {
            let asm_text = format!("mov qword [{}], {}", base_name, reg_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(qword_ptr(base.into()), reg)
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Zero-operand instructions ──────────────────────────────────

#[test]
fn test_nasm_zero_operand() {
    let nasm = skip_if_no_nasm!();
    let insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = vec![
        ("nop".into(), Box::new(|a: &mut CodeAssembler| a.nop())),
        ("ret".into(), Box::new(|a: &mut CodeAssembler| a.ret())),
        ("int3".into(), Box::new(|a: &mut CodeAssembler| a.int3())),
        ("cdq".into(), Box::new(|a: &mut CodeAssembler| a.cdq())),
        ("cqo".into(), Box::new(|a: &mut CodeAssembler| a.cqo())),
    ];
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── adc / sbb ──────────────────────────────────────────────────

#[test]
fn test_nasm_adc_sbb_reg32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (EDX, "edx"),
        (R8D, "r8d"), (R9D, "r9d"),
    ];
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            let asm_text = format!("adc {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.adc(dst, src))));
        }
    }
    for &(dst, dst_name) in regs {
        for &(src, src_name) in regs {
            let asm_text = format!("sbb {}, {}", dst_name, src_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.sbb(dst, src))));
        }
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── push/pop 16-bit ────────────────────────────────────────────

#[test]
fn test_nasm_push_pop_reg16() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();
    for &(reg, name) in REGS16.iter() {
        let asm_text = format!("push {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.push(reg))));
    }
    for &(reg, name) in REGS16.iter() {
        let asm_text = format!("pop {}", name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.pop(reg))));
    }
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── mov with indexed memory ────────────────────────────────────

#[test]
fn test_nasm_mov_indexed_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // mov eax, [base + index*scale + disp]
    let combos: &[(Reg, &str, Reg, &str, u8, i32)] = &[
        (RBX, "rbx", RSI, "rsi", 1, 0x10),
        (RBX, "rbx", RSI, "rsi", 2, 0x10),
        (RBX, "rbx", RSI, "rsi", 4, 0x10),
        (RBX, "rbx", RSI, "rsi", 8, 0x10),
        (RAX, "rax", RCX, "rcx", 4, 0x100),
        (R8, "r8", R9, "r9", 8, 0x10),
        (RBX, "rbx", R12, "r12", 4, 0x20),
        (RSP, "rsp", RAX, "rax", 4, 0x10),
        (RBP, "rbp", RCX, "rcx", 2, 0x10),
    ];

    for &(base, base_name, idx, idx_name, scale, disp) in combos {
        let asm_text = format!(
            "mov eax, dword [{}+{}*{}+0x{:x}]",
            base_name, idx_name, scale, disp
        );
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(EAX, dword_ptr(base + idx * scale + disp))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── ret imm ────────────────────────────────────────────────────

#[test]
fn test_nasm_ret_imm() {
    let nasm = skip_if_no_nasm!();
    let insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = vec![
        ("ret 0".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(0))),
        ("ret 8".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(8))),
        ("ret 16".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(16))),
        ("ret 256".into(), Box::new(|a: &mut CodeAssembler| a.ret_imm(256))),
    ];
    compare_nasm_batch(&nasm, 64, insns);
}

// ─── or/and/xor with immediate ──────────────────────────────────

#[test]
fn test_nasm_or_and_xor_imm() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let regs: &[(Reg, &str)] = &[
        (EAX, "eax"), (ECX, "ecx"), (R8D, "r8d"),
    ];
    let imm_values: &[i32] = &[1, 0x7F, 0xFF, 0x12345678];

    for &(reg, name) in regs {
        for &imm in imm_values {
            let asm_text = format!("or {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.or_(reg, imm))));
        }
    }
    for &(reg, name) in regs {
        for &imm in imm_values {
            let asm_text = format!("xor {}, 0x{:x}", name, imm);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| a.xor_(reg, imm))));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── add/sub with memory operands ───────────────────────────────

#[test]
fn test_nasm_add_sub_mem() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // add reg, [mem]
    let combos: &[(Reg, &str, Reg, &str)] = &[
        (EAX, "eax", RAX, "rax"),
        (ECX, "ecx", RBX, "rbx"),
        (R8D, "r8d", R9, "r9"),
    ];

    for &(reg, reg_name, base, base_name) in combos {
        let asm_text = format!("add {}, dword [{}]", reg_name, base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.add(reg, dword_ptr(base.into()))
        })));
    }

    for &(reg, reg_name, base, base_name) in combos {
        let asm_text = format!("sub {}, dword [{}]", reg_name, base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.sub(reg, dword_ptr(base.into()))
        })));
    }

    // add [mem], reg
    for &(reg, reg_name, base, base_name) in combos {
        let asm_text = format!("add dword [{}], {}", base_name, reg_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.add(dword_ptr(base.into()), reg)
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
