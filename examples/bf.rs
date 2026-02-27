//! Port of xbyak sample/bf.cpp (64-bit System V path)
//! Brainfuck JIT compiler.

use rxbyak::*;
use std::io::Read;

extern "C" fn bf_putchar(c: i32) -> i32 {
    print!("{}", c as u8 as char);
    c
}

extern "C" fn bf_getchar() -> i32 {
    let mut buf = [0u8; 1];
    match std::io::stdin().read(&mut buf) {
        Ok(1) => buf[0] as i32,
        _ => -1,
    }
}

fn get_continuous_char(src: &[u8], pos: &mut usize, c: u8) -> i32 {
    let mut count = 1;
    while *pos + 1 < src.len() && src[*pos + 1] == c {
        *pos += 1;
        count += 1;
    }
    count
}

fn compile_bf(src: &[u8]) -> Result<CodeAssembler> {
    let mut asm = CodeAssembler::new(100000)?;

    // System V ABI:
    //   RDI = putchar fn ptr
    //   RSI = getchar fn ptr
    //   RDX = stack ptr (u8 array)
    // Use callee-saved registers:
    let p_putchar = RBX;
    let p_getchar = RBP;
    let stack = R12;

    asm.push(RBX)?;
    asm.push(RBP)?;
    asm.push(R12)?;
    asm.mov(p_putchar, RDI)?;    // putchar
    asm.mov(p_getchar, RSI)?;    // getchar
    asm.mov(stack, RDX)?;        // stack

    let mut label_f: Vec<Label> = Vec::new();
    let mut label_b: Vec<Label> = Vec::new();

    let mut pos = 0;
    while pos < src.len() {
        let c = src[pos];
        match c {
            b'+' | b'-' => {
                let count = get_continuous_char(src, &mut pos, c);
                if count == 1 {
                    if c == b'+' {
                        asm.inc(byte_ptr(stack.into()))?;
                    } else {
                        asm.dec(byte_ptr(stack.into()))?;
                    }
                } else {
                    let val = if c == b'+' { count } else { -count };
                    asm.add(byte_ptr(stack.into()), val)?;
                }
            }
            b'>' | b'<' => {
                let count = get_continuous_char(src, &mut pos, c) as i64;
                let val = if c == b'>' { count } else { -count };
                asm.add(stack, val)?;
            }
            b'.' => {
                // System V: EDI = first arg
                asm.movzx(EDI, byte_ptr(stack.into()))?;
                asm.call_reg(p_putchar)?;
            }
            b',' => {
                asm.call_reg(p_getchar)?;
                asm.mov(byte_ptr(stack.into()), AL)?;
            }
            b'[' => {
                let b = asm.create_label();
                asm.bind(&b)?;
                label_b.push(b);
                asm.movzx(EAX, byte_ptr(stack.into()))?;
                asm.test(EAX, EAX)?;
                let f = asm.create_label();
                asm.jz(&f, JmpType::Near)?;
                label_f.push(f);
            }
            b']' => {
                let b = label_b.pop().expect("unmatched ']'");
                asm.jmp(&b, JmpType::Near)?;
                let f = label_f.pop().expect("unmatched ']'");
                asm.bind(&f)?;
            }
            _ => {}
        }
        pos += 1;
    }

    asm.pop(R12)?;
    asm.pop(RBP)?;
    asm.pop(RBX)?;
    asm.ret()?;
    asm.ready()?;
    Ok(asm)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let src = if args.len() > 1 {
        std::fs::read(&args[1]).expect("cannot read BF file")
    } else {
        // Embedded Hello World
        include_bytes!("hello.bf").to_vec()
    };

    let asm = compile_bf(&src)?;

    type BfFunc = fn(usize, usize, *mut u8);
    let f: BfFunc = unsafe { asm.get_code() };

    let mut stack = vec![0u8; 128 * 1024];
    f(
        bf_putchar as usize,
        bf_getchar as usize,
        stack.as_mut_ptr(),
    );
    println!();

    Ok(())
}
