//! Port of xbyak sample/calc.cpp
//! Multi-variable polynomial calculator using JIT.
//! Replaces Boost.Spirit with a Rust recursive descent parser.

use rxbyak::*;
use std::collections::HashMap;

const MAX_CONST_NUM: usize = 32;

struct FuncGen {
    asm: CodeAssembler,
    const_tbl: Vec<f64>,
    reg_idx: i32,
    var_map: HashMap<String, usize>,
    const_tbl_ptr: *const f64,
}

impl FuncGen {
    fn new(var_tbl: &[String]) -> Result<Self> {
        let const_tbl = vec![0.0f64; MAX_CONST_NUM];
        let const_tbl_ptr = const_tbl.as_ptr();
        let mut asm = CodeAssembler::new(4096)?;
        let mut var_map = HashMap::new();

        for (i, name) in var_tbl.iter().enumerate() {
            var_map.insert(name.clone(), i);
        }

        // 64-bit System V: RDI = valTbl (pointer to doubles)
        // Load const table address into RSI
        asm.mov(RSI, const_tbl_ptr as i64)?;

        Ok(FuncGen {
            asm,
            const_tbl,
            reg_idx: -1,
            var_map,
            const_tbl_ptr,
        })
    }

    fn gen_push(&mut self, n: f64) -> Result<()> {
        if self.const_tbl.len() >= MAX_CONST_NUM {
            return Err(Error::BadParameter);
        }
        let pos = (self.reg_idx + 1) as usize;
        if pos < self.const_tbl.len() {
            self.const_tbl[pos] = n;
        }
        // Update the actual memory
        unsafe {
            let p = self.const_tbl_ptr as *mut f64;
            *p.add(pos) = n;
        }
        self.reg_idx += 1;
        let xmm = Reg::xmm(self.reg_idx as u8);
        let offset = (self.reg_idx as usize) * 8;
        self.asm.movsd(xmm, qword_ptr(RSI + offset as i32))?;
        Ok(())
    }

    fn gen_val(&mut self, var_name: &str) -> Result<()> {
        let idx = *self.var_map.get(var_name)
            .ok_or(Error::BadParameter)?;
        self.reg_idx += 1;
        let xmm = Reg::xmm(self.reg_idx as u8);
        self.asm.movsd(xmm, qword_ptr(RDI + (idx * 8) as i32))?;
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

    fn complete(&mut self) -> Result<()> {
        // Result is in XMM0
        self.asm.ret()?;
        self.asm.ready()
    }

    fn get_func(&self) -> fn(*const f64) -> f64 {
        unsafe { self.asm.get_code() }
    }
}

// ─── Recursive descent parser ──────────────────────────────

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
        // Optional minus handled by factor
        while self.pos < self.input.len()
            && (self.input[self.pos].is_ascii_digit() || self.input[self.pos] == b'.')
        {
            self.pos += 1;
        }
        if self.pos == start { return None; }
        let s = std::str::from_utf8(&self.input[start..self.pos]).ok()?;
        s.parse().ok()
    }

    fn parse_var(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_alphabetic() {
            self.pos += 1;
        }
        if self.pos == start { return None; }
        Some(String::from_utf8_lossy(&self.input[start..self.pos]).to_string())
    }

    // factor = number | var | '(' expr ')'
    fn parse_factor(&mut self, gen: &mut FuncGen) -> Result<()> {
        if self.consume(b'(') {
            self.parse_expr(gen)?;
            if !self.consume(b')') {
                return Err(Error::BadParameter);
            }
            return Ok(());
        }

        // Try number
        let saved = self.pos;
        if let Some(n) = self.parse_number() {
            return gen.gen_push(n);
        }
        self.pos = saved;

        // Try variable
        if let Some(var) = self.parse_var() {
            return gen.gen_val(&var);
        }

        Err(Error::BadParameter)
    }

    // term = factor (('*' | '/') factor)*
    fn parse_term(&mut self, gen: &mut FuncGen) -> Result<()> {
        self.parse_factor(gen)?;
        loop {
            if self.consume(b'*') {
                self.parse_factor(gen)?;
                gen.gen_mul()?;
            } else if self.consume(b'/') {
                self.parse_factor(gen)?;
                gen.gen_div()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    // expr = term (('+' | '-') term)*
    fn parse_expr(&mut self, gen: &mut FuncGen) -> Result<()> {
        self.parse_term(gen)?;
        loop {
            if self.consume(b'+') {
                self.parse_term(gen)?;
                gen.gen_add()?;
            } else if self.consume(b'-') {
                self.parse_term(gen)?;
                gen.gen_sub()?;
            } else {
                break;
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 2 {
        eprintln!("calc \"var1 var2 ...\" \"function of var\"");
        eprintln!("eg. calc x \"x*x\"");
        eprintln!("eg. calc \"x y z\" \"x*x + y - z\"");
        return Ok(());
    }

    let var_str = &args[1];
    let poly = &args[2];

    // Parse variable names
    let var_tbl: Vec<String> = var_str.split_whitespace()
        .map(|s| s.to_string())
        .collect();

    print!("varTbl = {{ ");
    for (i, v) in var_tbl.iter().enumerate() {
        print!("{}:{}, ", v, i);
    }
    println!("}}");

    let mut gen = FuncGen::new(&var_tbl)?;
    let mut parser = Parser::new(poly);
    parser.parse_expr(&mut gen)?;
    gen.complete()?;

    let func = gen.get_func();

    println!("64bit mode");

    let mut rng: u32 = 42;
    let mut val_tbl: Vec<f64> = vec![0.0; var_tbl.len()];

    for _ in 0..10 {
        for v in val_tbl.iter_mut() {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((rng >> 16) % 7) as f64;
        }
        let y = func(val_tbl.as_ptr());
        print!("f(");
        for (i, v) in val_tbl.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{:.6}", v);
        }
        println!(")={:.6}", y);
    }

    Ok(())
}
