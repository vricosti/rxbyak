//! Port of xbyak sample/static_buf.cpp
//! Demonstrates using a user-provided buffer for code generation.

use rxbyak::*;

fn main() -> Result<()> {
    const BUF_SIZE: usize = 4096;

    // Allocate a page-aligned buffer
    let layout = std::alloc::Layout::from_size_align(BUF_SIZE, BUF_SIZE).unwrap();
    let buf_ptr = unsafe { std::alloc::alloc(layout) };
    if buf_ptr.is_null() {
        eprintln!("failed to allocate aligned buffer");
        return Ok(());
    }

    // Generate code into user buffer
    let mut asm = unsafe { CodeAssembler::from_user_buf(buf_ptr, BUF_SIZE) };

    println!("generate");
    println!("ptr={:p}, {:p}", asm.top(), buf_ptr);

    // System V: lea rax, [rdi + rsi]
    asm.lea(RAX, ptr(RDI + RSI))?;
    asm.ret()?;

    // Manually set protection to RE
    unsafe {
        rxbyak::platform::protect(buf_ptr, BUF_SIZE, rxbyak::platform::ProtectMode::ReadExec)?;
    }

    let add_fn: fn(i64, i64) -> i64 = unsafe {
        std::mem::transmute(buf_ptr)
    };

    let mut sum: i64 = 0;
    for i in 0..10 {
        sum += add_fn(i, 5);
    }
    println!("sum={}", sum);

    // Restore RW protection before deallocation
    unsafe {
        rxbyak::platform::protect(buf_ptr, BUF_SIZE, rxbyak::platform::ProtectMode::ReadWrite)?;
        std::alloc::dealloc(buf_ptr, layout);
    }

    Ok(())
}
