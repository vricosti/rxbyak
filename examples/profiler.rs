//! Port of xbyak sample/profiler.cpp
//! JIT loop benchmark — skip VTune/perf integration.

use rxbyak::*;
use std::time::Instant;

const N: i32 = 3_000_000;

/// Generate: fn() -> i32 with sub-loop (10 decrements per iteration)
fn make_sub_loop() -> Result<(CodeAssembler, fn() -> i32)> {
    let mut asm = CodeAssembler::new(4096)?;
    asm.mov(EAX, N)?;
    let lp = asm.create_label();
    asm.bind(&lp)?;
    for _ in 0..10 {
        asm.sub(EAX, 1)?;
    }
    asm.jg(&lp, JmpType::Near)?;
    asm.mov(EAX, 1)?;
    asm.ret()?;
    asm.ready()?;
    let f: fn() -> i32 = unsafe { asm.get_code() };
    Ok((asm, f))
}

/// Generate: fn() -> i32 with xorps loop
fn make_xorps_loop() -> Result<(CodeAssembler, fn() -> i32)> {
    let mut asm = CodeAssembler::new(4096)?;
    asm.mov(EAX, N)?;
    let lp = asm.create_label();
    asm.bind(&lp)?;
    for _ in 0..10 {
        asm.xorps(XMM0, XMM0)?;
    }
    asm.sub(EAX, 1)?;
    asm.jg(&lp, JmpType::Near)?;
    asm.mov(EAX, 1)?;
    asm.ret()?;
    asm.ready()?;
    let f: fn() -> i32 = unsafe { asm.get_code() };
    Ok((asm, f))
}

fn s1(n: i32) -> f64 {
    let mut r = 0.0f64;
    for i in 0..n {
        r += 1.0 / (i as f64 + 1.0);
    }
    r
}

fn s2(n: i32) -> f64 {
    let mut r = 0.0f64;
    for i in 0..n {
        r += 1.0 / (i as f64 * i as f64 + 1.0) + 2.0 / (i as f64 + 3.0);
    }
    r
}

fn main() -> Result<()> {
    let (asm_f, f) = make_sub_loop()?;
    let (asm_g, g) = make_xorps_loop()?;

    println!("f:{:p}, {}", asm_f.top(), asm_f.size());
    println!("g:{:p}, {}", asm_g.top(), asm_g.size());

    let mut sum = 0.0f64;
    let start = Instant::now();
    for i in 0..20000 {
        sum += s1(i);
        sum += s2(i);
    }
    println!("native sum={:.6} ({:.3?})", sum, start.elapsed());

    let start = Instant::now();
    for _ in 0..2000 {
        sum += f() as f64;
    }
    println!("f sum={:.6} ({:.3?})", sum, start.elapsed());

    let start = Instant::now();
    for _ in 0..2000 {
        sum += g() as f64;
    }
    println!("g sum={:.6} ({:.3?})", sum, start.elapsed());

    println!("end");
    Ok(())
}
