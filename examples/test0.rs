//! Port of xbyak sample/test0.cpp (64-bit path)
//! Demonstrates sum loop, AddFunc with labels, and multiple instances.

use rxbyak::*;

/// Generate: fn(n: i32) -> i32 that returns 0 + 1 + ... + n
fn make_sum(asm: &mut CodeAssembler) -> Result<()> {
    asm.enter_local();
    // System V: edi = n
    asm.mov(ECX, EDI)?;
    asm.xor_(EAX, EAX)?; // sum = 0
    asm.test(ECX, ECX)?;
    let exit = asm.create_label();
    asm.jz(&exit, JmpType::Near)?;
    asm.xor_(EDX, EDX)?; // i = 0
    let lp = asm.create_label();
    asm.bind(&lp)?;
    asm.add(EAX, EDX)?;
    asm.inc(EDX)?;
    asm.cmp(EDX, ECX)?;
    asm.jbe(&lp, JmpType::Near)?;
    asm.bind(&exit)?;
    asm.ret()?;
    asm.leave_local()?;
    Ok(())
}

/// Generate: fn(x: i32) -> i32 that returns x + y (where y is baked in)
fn make_add_func(y: i32) -> Result<(CodeAssembler, fn(i32) -> i32)> {
    let mut asm = CodeAssembler::new(4096)?;
    // System V: edi = x
    asm.lea(EAX, ptr(RDI + y))?;
    asm.ret()?;
    asm.ready()?;
    let f: fn(i32) -> i32 = unsafe { asm.get_code() };
    Ok((asm, f))
}

fn main() -> Result<()> {
    println!("64bit mode");

    // Part 1: Sum function
    let mut asm = CodeAssembler::new(4096)?;
    make_sum(&mut asm)?;
    asm.ready()?;
    let func: fn(i32) -> i32 = unsafe { asm.get_code() };
    for i in 0..=10 {
        println!("0 + ... + {} = {}", i, func(i));
    }

    // Part 2: AddFunc — generates fn(x) -> x + y for each y
    for i in 0..10 {
        let (_asm, add) = make_add_func(i)?;
        println!("{} + {} = {}", i, i, add(i));
    }

    puts_ok();
    Ok(())
}

fn puts_ok() {
    println!("OK");
}
