/// Address encoding tests validated against NASM reference assembler.
/// Combinatorial: mov ecx, [base + index*scale + disp]

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

// ─── [base] — all 16 GPR64 as base ─────────────────────────────

#[test]
fn test_nasm_addr_base_only() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    for &(base, base_name) in BASES64.iter() {
        let asm_text = format!("mov ecx, dword [{}]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(base.into()))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + disp8] ─────────────────────────────────────────────

#[test]
fn test_nasm_addr_base_disp8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    for &(base, base_name) in BASES64.iter() {
        let asm_text = format!("mov ecx, dword [{}+0x1]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(base + 1))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + disp32] ────────────────────────────────────────────

#[test]
fn test_nasm_addr_base_disp32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    for &(base, base_name) in BASES64.iter() {
        let asm_text = format!("mov ecx, dword [{}+0x12345678]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(base + 0x12345678))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + disp at boundaries] ────────────────────────────────

#[test]
fn test_nasm_addr_base_disp_boundaries() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let disps: &[(i32, &str)] = &[
        (0x7F, "0x7f"),    // max disp8
        (0x80, "0x80"),    // min disp32 (overflows disp8)
        (-1, "-0x1"),      // negative disp8
    ];

    for &(base, base_name) in &[
        (RAX, "rax"), (RBP, "rbp"), (RSP, "rsp"), (R13, "r13"), (R12, "r12"),
    ] {
        for &(disp, disp_str) in disps {
            let sep = if disp >= 0 { "+" } else { "" };
            let asm_text = format!("mov ecx, dword [{}{}{}]", base_name, sep, disp_str);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(ECX, dword_ptr(base + disp))
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + index*scale] ───────────────────────────────────────

#[test]
fn test_nasm_addr_base_index_scale() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // Test representative base+index combos with all scales
    let combos: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"),
        (RBX, "rbx", RSI, "rsi"),
        (RBX, "rbx", RDI, "rdi"),
        (RSP, "rsp", RAX, "rax"),
        (RBP, "rbp", RCX, "rcx"),
        (R8, "r8", R9, "r9"),
        (R12, "r12", R13, "r13"),
        (RAX, "rax", R8, "r8"),
        (R8, "r8", RAX, "rax"),
    ];

    for &(base, base_name, idx, idx_name) in combos {
        for &scale in &[1u8, 2, 4, 8] {
            let asm_text = format!(
                "mov ecx, dword [{}+{}*{}]",
                base_name, idx_name, scale
            );
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(ECX, dword_ptr(base + idx * scale))
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + index*scale + disp8] ───────────────────────────────

#[test]
fn test_nasm_addr_base_index_scale_disp8() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let combos: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"),
        (RBX, "rbx", RSI, "rsi"),
        (RSP, "rsp", RAX, "rax"),
        (RBP, "rbp", RDX, "rdx"),
        (R8, "r8", R9, "r9"),
        (R13, "r13", R14, "r14"),
    ];

    for &(base, base_name, idx, idx_name) in combos {
        for &scale in &[1u8, 2, 4, 8] {
            let asm_text = format!(
                "mov ecx, dword [{}+{}*{}+0x10]",
                base_name, idx_name, scale
            );
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(ECX, dword_ptr(base + idx * scale + 0x10))
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── [base + index*scale + disp32] ──────────────────────────────

#[test]
fn test_nasm_addr_base_index_scale_disp32() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    let combos: &[(Reg, &str, Reg, &str)] = &[
        (RAX, "rax", RCX, "rcx"),
        (RBX, "rbx", RSI, "rsi"),
        (RSP, "rsp", RAX, "rax"),
        (RBP, "rbp", RDX, "rdx"),
        (R8, "r8", R9, "r9"),
    ];

    for &(base, base_name, idx, idx_name) in combos {
        for &scale in &[1u8, 4, 8] {
            let asm_text = format!(
                "mov ecx, dword [{}+{}*{}+0x12345678]",
                base_name, idx_name, scale
            );
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.mov(ECX, dword_ptr(base + idx * scale + 0x12345678))
            })));
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── RBP/R13 special cases (force disp) ─────────────────────────

#[test]
fn test_nasm_addr_rbp_r13_special() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // RBP as base: [rbp] requires disp8=0, [rbp+disp8], [rbp+disp32]
    for &(disp, disp_str) in &[(0, ""), (1, "+0x1"), (0x80, "+0x80")] {
        let asm_text = if disp == 0 {
            "mov ecx, dword [rbp]".to_string()
        } else {
            format!("mov ecx, dword [rbp{}]", disp_str)
        };
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(RBP + disp))
        })));
    }

    // R13 as base: same special behavior as RBP
    for &(disp, disp_str) in &[(0, ""), (1, "+0x1"), (0x80, "+0x80")] {
        let asm_text = if disp == 0 {
            "mov ecx, dword [r13]".to_string()
        } else {
            format!("mov ecx, dword [r13{}]", disp_str)
        };
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(R13 + disp))
        })));
    }

    // RBP with index
    for &scale in &[1u8, 4, 8] {
        let asm_text = format!("mov ecx, dword [rbp+rcx*{}]", scale);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(RBP + RCX * scale))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── RSP/R12 special cases (force SIB) ──────────────────────────

#[test]
fn test_nasm_addr_rsp_r12_special() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // RSP as base: always needs SIB byte
    for &(disp, disp_str) in &[(0, ""), (1, "+0x1"), (0x80, "+0x80")] {
        let asm_text = if disp == 0 {
            "mov ecx, dword [rsp]".to_string()
        } else {
            format!("mov ecx, dword [rsp{}]", disp_str)
        };
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(RSP + disp))
        })));
    }

    // R12 as base: same SIB behavior as RSP
    for &(disp, disp_str) in &[(0, ""), (1, "+0x1"), (0x80, "+0x80")] {
        let asm_text = if disp == 0 {
            "mov ecx, dword [r12]".to_string()
        } else {
            format!("mov ecx, dword [r12{}]", disp_str)
        };
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(R12 + disp))
        })));
    }

    // RSP with index
    for &(idx, idx_name) in &[(RAX, "rax"), (RCX, "rcx"), (R8, "r8")] {
        let asm_text = format!("mov ecx, dword [rsp+{}*4+0x10]", idx_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(RSP + idx * 4 + 0x10))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── 64-bit operations with various address modes ───────────────

#[test]
fn test_nasm_addr_64bit_ops() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // mov rax, qword [base + disp]
    for &(base, base_name) in &[
        (RAX, "rax"), (RBX, "rbx"), (RSP, "rsp"), (RBP, "rbp"),
        (R8, "r8"), (R12, "r12"), (R13, "r13"),
    ] {
        let asm_text = format!("mov rax, qword [{}+0x10]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(RAX, qword_ptr(base + 0x10))
        })));
    }

    // mov qword [base + index*scale], rax
    for &(base, base_name) in &[(RAX, "rax"), (R8, "r8")] {
        for &(idx, idx_name) in &[(RCX, "rcx"), (R9, "r9")] {
            for &scale in &[1u8, 4, 8] {
                let asm_text = format!(
                    "mov qword [{}+{}*{}], rax",
                    base_name, idx_name, scale
                );
                insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                    a.mov(qword_ptr(base + idx * scale), RAX)
                })));
            }
        }
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Different data sizes through same address ──────────────────

#[test]
fn test_nasm_addr_data_sizes() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // byte, word, dword, qword from [rax+0x10]
    let asm_text = "mov al, byte [rax+0x10]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(AL, byte_ptr(RAX + 0x10))
    })));

    let asm_text = "mov ax, word [rax+0x10]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(AX, word_ptr(RAX + 0x10))
    })));

    let asm_text = "mov eax, dword [rax+0x10]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(EAX, dword_ptr(RAX + 0x10))
    })));

    let asm_text = "mov rax, qword [rbx+0x10]".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(RAX, qword_ptr(RBX + 0x10))
    })));

    // Store different sizes
    let asm_text = "mov byte [rax+0x10], cl".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(byte_ptr(RAX + 0x10), CL)
    })));

    let asm_text = "mov word [rax+0x10], cx".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(word_ptr(RAX + 0x10), CX)
    })));

    let asm_text = "mov dword [rax+0x10], ecx".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(dword_ptr(RAX + 0x10), ECX)
    })));

    let asm_text = "mov qword [rax+0x10], rcx".to_string();
    insns.push((asm_text, Box::new(|a: &mut CodeAssembler| {
        a.mov(qword_ptr(RAX + 0x10), RCX)
    })));

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Comprehensive index register sweep ─────────────────────────

#[test]
fn test_nasm_addr_all_index_regs() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // Use all valid index registers (all except RSP)
    for &(idx, idx_name) in INDICES64.iter() {
        let asm_text = format!("mov ecx, dword [rax+{}*4+0x10]", idx_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(RAX + idx * 4 + 0x10))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── XMM/SSE with various address modes ─────────────────────────

#[test]
fn test_nasm_addr_simd() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // movaps xmm, [base]
    for &(base, base_name) in &[
        (RAX, "rax"), (RSP, "rsp"), (RBP, "rbp"), (R8, "r8"),
    ] {
        let asm_text = format!("movaps xmm0, oword [{}]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.movaps(XMM0, xmmword_ptr(base.into()))
        })));
    }

    // movaps xmm, [base + index*scale]
    for &(base, base_name) in &[(RAX, "rax"), (R8, "r8")] {
        for &(idx, idx_name) in &[(RCX, "rcx"), (R9, "r9")] {
            let asm_text = format!("movaps xmm0, oword [{}+{}*4]", base_name, idx_name);
            insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
                a.movaps(XMM0, xmmword_ptr(base + idx * 4))
            })));
        }
    }

    // vmovaps ymm, [base + disp]
    for &(base, base_name) in &[(RAX, "rax"), (RSP, "rsp")] {
        let asm_text = format!("vmovaps ymm0, yword [{}+0x20]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.vmovaps(YMM0, ymmword_ptr(base + 0x20))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── ALU ops with various address modes ─────────────────────────

#[test]
fn test_nasm_addr_alu_combos() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    // add eax, [base + index*scale + disp]
    let combos: &[(Reg, &str, Reg, &str, u8, i32)] = &[
        (RAX, "rax", RCX, "rcx", 1, 0x10),
        (RAX, "rax", RCX, "rcx", 4, 0x100),
        (RBX, "rbx", RSI, "rsi", 8, 0x10),
        (RSP, "rsp", RAX, "rax", 4, 0x10),
        (RBP, "rbp", RCX, "rcx", 2, 0x10),
        (R8, "r8", R9, "r9", 4, 0x10),
    ];

    for &(base, base_name, idx, idx_name, scale, disp) in combos {
        let asm_text = format!(
            "add eax, dword [{}+{}*{}+0x{:x}]",
            base_name, idx_name, scale, disp
        );
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.add(EAX, dword_ptr(base + idx * scale + disp))
        })));
    }

    // cmp [base], reg
    for &(base, base_name) in &[(RAX, "rax"), (RSP, "rsp"), (R8, "r8")] {
        let asm_text = format!("cmp dword [{}], ecx", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.cmp(dword_ptr(base.into()), ECX)
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}

// ─── Negative displacements ─────────────────────────────────────

#[test]
fn test_nasm_addr_negative_disp() {
    let nasm = skip_if_no_nasm!();
    let mut insns: Vec<(String, Box<dyn FnOnce(&mut CodeAssembler) -> Result<()>>)> = Vec::new();

    for &(base, base_name) in &[
        (RAX, "rax"), (RBP, "rbp"), (RSP, "rsp"), (R8, "r8"),
    ] {
        let asm_text = format!("mov ecx, dword [{}-0x8]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(base - 8))
        })));
    }

    for &(base, base_name) in &[
        (RAX, "rax"), (RBP, "rbp"), (RSP, "rsp"),
    ] {
        let asm_text = format!("mov ecx, dword [{}-0x80]", base_name);
        insns.push((asm_text, Box::new(move |a: &mut CodeAssembler| {
            a.mov(ECX, dword_ptr(base - 0x80))
        })));
    }

    compare_nasm_batch(&nasm, 64, insns);
}
