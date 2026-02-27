//! Port of xbyak sample/stackframe.cpp
//! Demonstrates StackFrame for automatic prologue/epilogue generation.

use rxbyak::*;
use rxbyak::util::stack_frame::StackFrame;

fn main() -> Result<()> {
    // Test 1: 3 params, 0 temps
    {
        let mut asm = CodeAssembler::new(4096)?;
        let sf = StackFrame::new(&mut asm, 3, 0, 0)?;
        asm.mov(RAX, sf.p[0])?;
        asm.add(RAX, sf.p[1])?;
        asm.add(RAX, sf.p[2])?;
        sf.close(&mut asm)?;
        asm.ready()?;

        let f: fn(i64, i64, i64) -> i64 = unsafe { asm.get_code() };
        let ret = f(3, 5, 2);
        if ret == 3 + 5 + 2 {
            println!("3 + 5 + 2 = {} ok", ret);
        } else {
            println!("ng: expected 10, got {}", ret);
        }
    }

    // Test 2: 2 params, 1 temp
    {
        let mut asm = CodeAssembler::new(4096)?;
        let sf = StackFrame::new(&mut asm, 2, 1, 0)?;
        asm.mov(sf.t[0], sf.p[0])?;
        asm.add(sf.t[0], sf.p[1])?;
        asm.mov(RAX, sf.t[0])?;
        sf.close(&mut asm)?;
        asm.ready()?;

        let f: fn(i64, i64) -> i64 = unsafe { asm.get_code() };
        let ret = f(100, 200);
        if ret == 300 {
            println!("100 + 200 = {} ok", ret);
        } else {
            println!("ng: expected 300, got {}", ret);
        }
    }

    Ok(())
}
