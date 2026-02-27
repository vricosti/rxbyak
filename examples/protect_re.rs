//! Port of xbyak sample/protect-re.cpp
//! Demonstrates protect mode toggling (RE/RW) on JIT code.

use rxbyak::*;

/// Test 1: Fixed allocation with manual protect mode toggle
fn test1() -> Result<()> {
    let mut asm = CodeAssembler::new(4096)?;
    asm.mov(EAX, 123)?;
    asm.ret()?;

    asm.set_protect_mode_re()?;
    let f: fn() -> i32 = unsafe { asm.get_code() };
    println!("f={}", f());

    asm.set_protect_mode_rw()?;
    asm.db(0)?; // can write after switching to RW
    println!("ok");
    Ok(())
}

/// Test 2: Auto-grow with ready_re()
fn test2() -> Result<()> {
    let mut asm = CodeAssembler::new_auto_grow(64)?;
    asm.mov(EAX, 123)?;
    asm.ret()?;

    asm.ready_re()?;
    let f: fn() -> i32 = unsafe { asm.get_code() };
    println!("f={}", f());

    asm.set_protect_mode_rw()?;
    asm.db(0)?; // can write after switching to RW
    println!("ok");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse::<i32>().unwrap_or(0)
    } else {
        0
    };

    match n {
        1 => test1()?,
        2 => test2()?,
        _ => {
            // Run both tests
            println!("--- test1 ---");
            test1()?;
            println!("--- test2 ---");
            test2()?;
        }
    }
    Ok(())
}
