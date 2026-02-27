//! Port of xbyak sample/calc2.cpp
//! Single-variable polynomial with VM+JIT comparison.
//! Replaces Boost.Spirit with a Rust recursive descent parser.

use rxbyak::*;
use std::time::Instant;

// ─── VM (stack-based) ──────────────────────────────────

#[derive(Clone)]
enum VmOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Imm(f64),
    VarX,
}

struct Vm {
    code: Vec<VmOp>,
}

impl Vm {
    fn new() -> Self {
        Vm { code: Vec::new() }
    }

    fn eval(&self, x: f64) -> f64 {
        let mut stack = Vec::with_capacity(16);
        for op in &self.code {
            match op {
                VmOp::VarX => stack.push(x),
                VmOp::Imm(v) => stack.push(*v),
                VmOp::Neg => {
                    let v = stack.last_mut().unwrap();
                    *v = -*v;
                }
                VmOp::Add => {
                    let b = stack.pop().unwrap();
                    *stack.last_mut().unwrap() += b;
                }
                VmOp::Sub => {
                    let b = stack.pop().unwrap();
                    *stack.last_mut().unwrap() -= b;
                }
                VmOp::Mul => {
                    let b = stack.pop().unwrap();
                    *stack.last_mut().unwrap() *= b;
                }
                VmOp::Div => {
                    let b = stack.pop().unwrap();
                    *stack.last_mut().unwrap() /= b;
                }
            }
        }
        stack[0]
    }
}

// ─── JIT compiler ──────────────────────────────────────

const MAX_CONST_NUM: usize = 32;

struct Jit {
    asm: CodeAssembler,
    // Heap-allocated so the data pointer survives struct moves
    const_tbl: Vec<f64>,
    const_tbl_pos: usize,
    reg_idx: i32,
}

impl Jit {
    fn new() -> Result<Self> {
        // Extra slot at the end for the sign-bit mask used by gen_neg
        let mut const_tbl = vec![0.0f64; MAX_CONST_NUM + 1];
        // Store sign-bit mask (1<<63) as f64 bits for XOR negation
        const_tbl[MAX_CONST_NUM] = f64::from_bits(1u64 << 63);

        let mut asm = CodeAssembler::new(4096)?;

        // 64-bit System V: XMM0 = x
        // Save XMM0 (the x parameter) into XMM7
        asm.movaps(XMM7, XMM0)?;
        // Load const table address into RDI (patched in complete())
        asm.mov(RDI, const_tbl.as_ptr() as i64)?;

        Ok(Jit {
            asm,
            const_tbl,
            const_tbl_pos: 0,
            reg_idx: -1,
        })
    }

    fn gen_push(&mut self, n: f64) -> Result<()> {
        if self.const_tbl_pos >= MAX_CONST_NUM {
            return Err(Error::BadParameter);
        }
        self.const_tbl[self.const_tbl_pos] = n;
        self.reg_idx += 1;
        let xmm = Reg::xmm(self.reg_idx as u8);
        let offset = (self.const_tbl_pos * 8) as i32;
        self.asm.movsd(xmm, qword_ptr(RDI + offset))?;
        self.const_tbl_pos += 1;
        Ok(())
    }

    fn gen_var_x(&mut self) -> Result<()> {
        self.reg_idx += 1;
        let xmm = Reg::xmm(self.reg_idx as u8);
        self.asm.movsd(xmm, XMM7)?;
        Ok(())
    }

    fn gen_add(&mut self) -> Result<()> {
        let dst = Reg::xmm((self.reg_idx - 1) as u8);
        let src = Reg::xmm(self.reg_idx as u8);
        self.asm.addsd(dst, src)?;
        self.reg_idx -= 1;
        Ok(())
    }

    fn gen_sub(&mut self) -> Result<()> {
        let dst = Reg::xmm((self.reg_idx - 1) as u8);
        let src = Reg::xmm(self.reg_idx as u8);
        self.asm.subsd(dst, src)?;
        self.reg_idx -= 1;
        Ok(())
    }

    fn gen_mul(&mut self) -> Result<()> {
        let dst = Reg::xmm((self.reg_idx - 1) as u8);
        let src = Reg::xmm(self.reg_idx as u8);
        self.asm.mulsd(dst, src)?;
        self.reg_idx -= 1;
        Ok(())
    }

    fn gen_div(&mut self) -> Result<()> {
        let dst = Reg::xmm((self.reg_idx - 1) as u8);
        let src = Reg::xmm(self.reg_idx as u8);
        self.asm.divsd(dst, src)?;
        self.reg_idx -= 1;
        Ok(())
    }

    fn gen_neg(&mut self) -> Result<()> {
        // XOR with sign bit to negate
        let xmm = Reg::xmm(self.reg_idx as u8);
        let offset = (MAX_CONST_NUM * 8) as i32;
        self.asm.xorpd(xmm, qword_ptr(RDI + offset))?;
        Ok(())
    }

    fn complete(&mut self) -> Result<()> {
        // Result is in XMM0
        self.asm.ret()?;
        self.asm.ready()
    }

    fn get_func(&self) -> fn(f64) -> f64 {
        unsafe { self.asm.get_code() }
    }
}

// ─── Recursive descent parser ──────────────────────────

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input: input.as_bytes(), pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        self.skip_ws();
        if self.pos < self.input.len() && self.input[self.pos] == expected {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_number(&mut self) -> Option<f64> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.input.len()
            && (self.input[self.pos].is_ascii_digit() || self.input[self.pos] == b'.')
        {
            self.pos += 1;
        }
        if self.pos == start { return None; }
        let s = std::str::from_utf8(&self.input[start..self.pos]).ok()?;
        s.parse().ok()
    }

    fn parse_factor_vm(&mut self, vm: &mut Vm) -> Result<()> {
        if self.consume(b'(') {
            self.parse_expr_vm(vm)?;
            if !self.consume(b')') { return Err(Error::BadParameter); }
            return Ok(());
        }
        if self.consume(b'-') {
            self.parse_factor_vm(vm)?;
            vm.code.push(VmOp::Neg);
            return Ok(());
        }
        if self.consume(b'+') {
            return self.parse_factor_vm(vm);
        }

        let saved = self.pos;
        if let Some(n) = self.parse_number() {
            vm.code.push(VmOp::Imm(n));
            return Ok(());
        }
        self.pos = saved;

        if self.consume(b'x') {
            vm.code.push(VmOp::VarX);
            return Ok(());
        }

        Err(Error::BadParameter)
    }

    fn parse_term_vm(&mut self, vm: &mut Vm) -> Result<()> {
        self.parse_factor_vm(vm)?;
        loop {
            if self.consume(b'*') {
                self.parse_factor_vm(vm)?;
                vm.code.push(VmOp::Mul);
            } else if self.consume(b'/') {
                self.parse_factor_vm(vm)?;
                vm.code.push(VmOp::Div);
            } else {
                break;
            }
        }
        Ok(())
    }

    fn parse_expr_vm(&mut self, vm: &mut Vm) -> Result<()> {
        self.parse_term_vm(vm)?;
        loop {
            if self.consume(b'+') {
                self.parse_term_vm(vm)?;
                vm.code.push(VmOp::Add);
            } else if self.consume(b'-') {
                self.parse_term_vm(vm)?;
                vm.code.push(VmOp::Sub);
            } else {
                break;
            }
        }
        Ok(())
    }

    fn parse_factor_jit(&mut self, jit: &mut Jit) -> Result<()> {
        if self.consume(b'(') {
            self.parse_expr_jit(jit)?;
            if !self.consume(b')') { return Err(Error::BadParameter); }
            return Ok(());
        }
        if self.consume(b'-') {
            self.parse_factor_jit(jit)?;
            jit.gen_neg()?;
            return Ok(());
        }
        if self.consume(b'+') {
            return self.parse_factor_jit(jit);
        }

        let saved = self.pos;
        if let Some(n) = self.parse_number() {
            jit.gen_push(n)?;
            return Ok(());
        }
        self.pos = saved;

        if self.consume(b'x') {
            jit.gen_var_x()?;
            return Ok(());
        }

        Err(Error::BadParameter)
    }

    fn parse_term_jit(&mut self, jit: &mut Jit) -> Result<()> {
        self.parse_factor_jit(jit)?;
        loop {
            if self.consume(b'*') {
                self.parse_factor_jit(jit)?;
                jit.gen_mul()?;
            } else if self.consume(b'/') {
                self.parse_factor_jit(jit)?;
                jit.gen_div()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn parse_expr_jit(&mut self, jit: &mut Jit) -> Result<()> {
        self.parse_term_jit(jit)?;
        loop {
            if self.consume(b'+') {
                self.parse_term_jit(jit)?;
                jit.gen_add()?;
            } else if self.consume(b'-') {
                self.parse_term_jit(jit)?;
                jit.gen_sub()?;
            } else {
                break;
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let formula = if args.len() >= 2 {
        args[1].clone()
    } else {
        "x*x + 2*x - 1".to_string()
    };

    println!("formula: {}", formula);

    // Parse for VM
    let mut vm = Vm::new();
    {
        let mut parser = Parser::new(&formula);
        parser.parse_expr_vm(&mut vm)?;
    }
    println!("VM eval(2.3) = {:.6}", vm.eval(2.3));

    // Parse for JIT
    let mut jit = Jit::new()?;
    {
        let mut parser = Parser::new(&formula);
        parser.parse_expr_jit(&mut jit)?;
    }
    jit.complete()?;
    let jit_func = jit.get_func();
    println!("JIT eval(2.3) = {:.6}", jit_func(2.3));

    // Benchmark
    let mut sum_vm = 0.0f64;
    let mut sum_jit = 0.0f64;

    let start = Instant::now();
    let mut x = 0.0f64;
    while x < 1000.0 {
        sum_vm += vm.eval(x);
        x += 0.0001;
    }
    println!("VM:  sum={:.6}, {:.3?}", sum_vm, start.elapsed());

    let start = Instant::now();
    x = 0.0;
    while x < 1000.0 {
        sum_jit += jit_func(x);
        x += 0.0001;
    }
    println!("JIT: sum={:.6}, {:.3?}", sum_jit, start.elapsed());

    Ok(())
}
