//! Port of xbyak sample/toyvm.cpp (rewritten from 32-bit to 64-bit)
//! A toy VM with 2 registers (A, B) + 64K memory, with interpreter and JIT recompiler.

use rxbyak::*;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum VmReg { A = 0, B = 1 }

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum Op {
    Ld = 0, Ldi, St, Add, Addi, Sub, Subi, Put, Jnz,
}

struct ToyVm {
    code: Vec<u32>,
    mem: Vec<u32>,
    mark: i32,
}

impl ToyVm {
    fn new() -> Self {
        ToyVm {
            code: Vec::new(),
            mem: vec![0u32; 65536],
            mark: 0,
        }
    }

    fn encode(&mut self, op: Op, r: VmReg, imm: u16) {
        let x = ((op as u32) << 24) | ((r as u32) << 16) | (imm as u32);
        self.code.push(x);
    }

    fn decode(x: u32) -> (u32, u32, u16) {
        let op = x >> 24;
        let r = (x >> 16) & 0xff;
        let imm = (x & 0xffff) as u16;
        (op, r, imm)
    }

    fn vldi(&mut self, r: VmReg, imm: u16) { self.encode(Op::Ldi, r, imm); }
    fn vld(&mut self, r: VmReg, idx: u16) { self.encode(Op::Ld, r, idx); }
    fn vst(&mut self, r: VmReg, idx: u16) { self.encode(Op::St, r, idx); }
    fn vadd(&mut self, r: VmReg, idx: u16) { self.encode(Op::Add, r, idx); }
    #[allow(dead_code)]
    fn vaddi(&mut self, r: VmReg, imm: u16) { self.encode(Op::Addi, r, imm); }
    #[allow(dead_code)]
    fn vsub(&mut self, r: VmReg, idx: u16) { self.encode(Op::Sub, r, idx); }
    fn vsubi(&mut self, r: VmReg, imm: u16) { self.encode(Op::Subi, r, imm); }
    fn vput(&mut self, r: VmReg) { self.encode(Op::Put, r, 0); }
    fn vjnz(&mut self, r: VmReg, offset: i16) { self.encode(Op::Jnz, r, offset as u16); }
    fn set_mark(&mut self) { self.mark = self.code.len() as i32; }
    fn get_mark_offset(&self) -> i16 {
        (self.mark - self.code.len() as i32 - 1) as i16
    }

    fn run(&mut self) {
        let mut reg = [0u32; 2];
        let end = self.code.len();
        let mut pc: usize = 0;

        loop {
            let x = self.code[pc];
            let (op, r, imm) = Self::decode(x);
            let r = r as usize;

            match op {
                x if x == Op::Ldi as u32 => reg[r] = imm as u32,
                x if x == Op::Ld as u32 => reg[r] = self.mem[imm as usize],
                x if x == Op::St as u32 => self.mem[imm as usize] = reg[r],
                x if x == Op::Add as u32 => reg[r] = reg[r].wrapping_add(self.mem[imm as usize]),
                x if x == Op::Addi as u32 => reg[r] = reg[r].wrapping_add(imm as u32),
                x if x == Op::Sub as u32 => reg[r] = reg[r].wrapping_sub(self.mem[imm as usize]),
                x if x == Op::Subi as u32 => reg[r] = reg[r].wrapping_sub(imm as u32),
                x if x == Op::Put as u32 => {
                    println!("{} {:>8}(0x{:08x})", (b'A' + r as u8) as char, reg[r], reg[r]);
                }
                x if x == Op::Jnz as u32 => {
                    if reg[r] != 0 {
                        pc = (pc as i32 + imm as i16 as i32) as usize;
                    }
                }
                _ => panic!("unknown opcode {}", op),
            }
            pc += 1;
            if pc >= end { break; }
        }
    }

    fn recompile(&self) -> Result<(CodeAssembler, fn())> {
        let mut asm = CodeAssembler::new(65536)?;

        // 64-bit port using callee-saved regs:
        // R12D/R13D = register A/B (32-bit values)
        // R14 = mem ptr (64-bit)
        // R15D/EBX/EBP = mem[0]/mem[1]/mem[2] cache (32-bit values)
        asm.push(RBX)?;
        asm.push(RBP)?;
        asm.push(R12)?;
        asm.push(R13)?;
        asm.push(R14)?;
        asm.push(R15)?;

        // Use 32-bit sub-registers for value operations
        let vm_reg = [R12D, R13D]; // A, B (32-bit)
        let mem_cache = [R15D, EBX, EBP]; // mem[0..2] cache (32-bit)
        let mem_cache_num = mem_cache.len();

        // Zero all cached regs
        for &r in &mem_cache {
            asm.xor_(r, r)?;
        }
        asm.xor_(R12D, R12D)?;
        asm.xor_(R13D, R13D)?;

        // R14 = mem ptr (64-bit address)
        asm.mov(R14, self.mem.as_ptr() as i64)?;

        // Create labels for each instruction
        let mut labels: Vec<Label> = Vec::new();
        for _ in 0..self.code.len() {
            labels.push(asm.create_label());
        }

        for (pc, &x) in self.code.iter().enumerate() {
            let (op, r, imm) = Self::decode(x);
            let r = r as usize;
            let imm_usize = imm as usize;

            asm.bind(&labels[pc])?;

            match op {
                x if x == Op::Ldi as u32 => {
                    asm.mov(vm_reg[r], imm as i64)?;
                }
                x if x == Op::Ld as u32 => {
                    if imm_usize < mem_cache_num {
                        asm.mov(vm_reg[r], mem_cache[imm_usize])?;
                    } else {
                        asm.mov(vm_reg[r], dword_ptr(R14 + (imm as i32) * 4))?;
                    }
                }
                x if x == Op::St as u32 => {
                    if imm_usize < mem_cache_num {
                        asm.mov(mem_cache[imm_usize], vm_reg[r])?;
                    } else {
                        asm.mov(dword_ptr(R14 + (imm as i32) * 4), vm_reg[r])?;
                    }
                }
                x if x == Op::Add as u32 => {
                    if imm_usize < mem_cache_num {
                        asm.add(vm_reg[r], mem_cache[imm_usize])?;
                    } else {
                        asm.add(vm_reg[r], dword_ptr(R14 + (imm as i32) * 4))?;
                    }
                }
                x if x == Op::Addi as u32 => {
                    asm.add(vm_reg[r], imm as i64)?;
                }
                x if x == Op::Sub as u32 => {
                    if imm_usize < mem_cache_num {
                        asm.sub(vm_reg[r], mem_cache[imm_usize])?;
                    } else {
                        asm.sub(vm_reg[r], dword_ptr(R14 + (imm as i32) * 4))?;
                    }
                }
                x if x == Op::Subi as u32 => {
                    asm.sub(vm_reg[r], imm as i64)?;
                }
                x if x == Op::Put as u32 => {
                    // Save caller-saved regs, maintain 16-byte stack alignment
                    asm.push(RAX)?;
                    asm.push(RCX)?;
                    asm.push(RDX)?;
                    asm.push(RSI)?;
                    asm.push(RDI)?;
                    asm.push(R8)?;
                    asm.push(R9)?;
                    asm.push(R10)?;
                    asm.push(R11)?;
                    // 6 callee-saved pushes + return addr = 7 pushes (RSP ≡ 8 mod 16)
                    // 9 caller-saved pushes → RSP ≡ 0 mod 16 (aligned for call)

                    // Call: print_val(reg_name: u64, val: u64)
                    // System V: RDI = reg_name char, RSI = value
                    asm.mov(EDI, (b'A' + r as u8) as i64)?;
                    // Zero-extend 32-bit vm_reg to 64-bit RSI
                    asm.mov(ESI, vm_reg[r])?;
                    asm.mov(RAX, print_val as usize as i64)?;
                    asm.call_reg(RAX)?;

                    asm.pop(R11)?;
                    asm.pop(R10)?;
                    asm.pop(R9)?;
                    asm.pop(R8)?;
                    asm.pop(RDI)?;
                    asm.pop(RSI)?;
                    asm.pop(RDX)?;
                    asm.pop(RCX)?;
                    asm.pop(RAX)?;
                }
                x if x == Op::Jnz as u32 => {
                    asm.test(vm_reg[r], vm_reg[r])?;
                    let target = (pc as i32 + imm as i16 as i32 + 1) as usize;
                    asm.jnz(&labels[target], JmpType::Near)?;
                }
                _ => panic!("unknown opcode {}", op),
            }
        }

        asm.pop(R15)?;
        asm.pop(R14)?;
        asm.pop(R13)?;
        asm.pop(R12)?;
        asm.pop(RBP)?;
        asm.pop(RBX)?;
        asm.ret()?;
        asm.ready()?;

        let f: fn() = unsafe { asm.get_code() };
        Ok((asm, f))
    }
}

extern "C" fn print_val(reg_name: u64, val: u64) {
    let val32 = val as u32;
    println!("{} {:>8}(0x{:08x})", reg_name as u8 as char, val32, val32);
}

fn fib_native(n: u32) -> u32 {
    let mut p: u32 = 1;
    let mut c: u32 = 1;
    let mut n = n;
    loop {
        let t = c;
        c = c.wrapping_add(p);
        p = t;
        n -= 1;
        if n == 0 { break; }
    }
    c
}

fn main() -> Result<()> {
    let n: u16 = 10000;

    let mut fib = ToyVm::new();
    // Fibonacci program:
    // A = c, B = temporary
    // mem[0] = p, mem[1] = t, mem[2] = n
    fib.vldi(VmReg::A, 1);      // c = 1
    fib.vst(VmReg::A, 0);       // p = 1
    fib.vldi(VmReg::B, n);
    fib.vst(VmReg::B, 2);       // n
    fib.set_mark();
    fib.vst(VmReg::A, 1);       // t = c
    fib.vadd(VmReg::A, 0);      // c += p
    fib.vld(VmReg::B, 1);
    fib.vst(VmReg::B, 0);       // p = t
    fib.vld(VmReg::B, 2);
    fib.vsubi(VmReg::B, 1);
    fib.vst(VmReg::B, 2);       // n--
    fib.vjnz(VmReg::B, fib.get_mark_offset());
    fib.vput(VmReg::A);

    // Interpreter
    {
        let start = Instant::now();
        fib.run();
        println!("vm       {:.2?}", start.elapsed());
    }

    // JIT
    {
        let (_asm, jit_fn) = fib.recompile()?;
        let start = Instant::now();
        jit_fn();
        println!("jit      {:.2?}", start.elapsed());
    }

    // Native Rust
    {
        let start = Instant::now();
        let c = fib_native(n as u32);
        println!("c={} (0x{:08x})", c, c);
        println!("native   {:.2?}", start.elapsed());
    }

    Ok(())
}
