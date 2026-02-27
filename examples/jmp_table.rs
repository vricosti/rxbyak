//! Port of xbyak sample/jmp_table.cpp
//! Demonstrates three jump table patterns with both fixed and auto-grow allocation.

use rxbyak::*;

const EXPECT_TBL: [i64; 3] = [5, 9, 12];

/// Mode 0: Inline jump table with align(8), computed jump via lea+jmp
fn gen_mode0(asm: &mut CodeAssembler) -> Result<()> {
    asm.enter_local();

    // System V: RDI = index
    asm.mov(RAX, RDI)?;

    // lea rcx, [rip + jmp_table]
    let jmp_table = asm.create_label();
    asm.lea_label(RCX, &jmp_table)?;
    asm.lea(RCX, ptr(RCX + RAX * 8))?;
    asm.jmp_reg(RCX)?;

    asm.align(8)?;
    asm.bind(&jmp_table)?;
    asm.mov(RAX, EXPECT_TBL[0])?;
    asm.ret()?;

    asm.align(8)?;
    asm.mov(RAX, EXPECT_TBL[1])?;
    asm.ret()?;

    asm.align(8)?;
    asm.mov(RAX, EXPECT_TBL[2])?;
    asm.ret()?;

    asm.leave_local()?;
    Ok(())
}

/// Mode 1: putL labels after code, jump via indirect memory
fn gen_mode1(asm: &mut CodeAssembler) -> Result<()> {
    asm.enter_local();

    // System V: RDI = index
    asm.mov(RAX, RDI)?;

    let jmp_table = asm.create_label();
    let label1 = asm.create_label();
    let label2 = asm.create_label();
    let label3 = asm.create_label();
    let end = asm.create_label();

    asm.lea_label(RCX, &jmp_table)?;
    asm.jmp_reg(qword_ptr(RCX + RAX * 8))?;

    asm.bind(&label1)?;
    asm.mov(RAX, EXPECT_TBL[0])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&label2)?;
    asm.mov(RAX, EXPECT_TBL[1])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&label3)?;
    asm.mov(RAX, EXPECT_TBL[2])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&end)?;
    asm.ret()?;
    asm.ud2()?;

    asm.align(8)?;
    asm.bind(&jmp_table)?;
    asm.put_l(&label1)?;
    asm.put_l(&label2)?;
    asm.put_l(&label3)?;

    asm.leave_local()?;
    Ok(())
}

/// Mode 2: putL labels before code (forward refs), jump via indirect memory
fn gen_mode2(asm: &mut CodeAssembler) -> Result<()> {
    asm.enter_local();

    // System V: RDI = index
    asm.mov(RAX, RDI)?;

    let jmp_table = asm.create_label();
    let label1 = asm.create_label();
    let label2 = asm.create_label();
    let label3 = asm.create_label();
    let end = asm.create_label();
    let in_label = asm.create_label();

    asm.jmp(&in_label, JmpType::Near)?;
    asm.ud2()?;

    asm.align(8)?;
    asm.bind(&jmp_table)?;
    asm.put_l(&label1)?;
    asm.put_l(&label2)?;
    asm.put_l(&label3)?;

    asm.bind(&in_label)?;
    asm.lea_label(RCX, &jmp_table)?;
    asm.jmp_reg(qword_ptr(RCX + RAX * 8))?;

    asm.bind(&label1)?;
    asm.mov(RAX, EXPECT_TBL[0])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&label2)?;
    asm.mov(RAX, EXPECT_TBL[1])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&label3)?;
    asm.mov(RAX, EXPECT_TBL[2])?;
    asm.jmp(&end, JmpType::Near)?;

    asm.bind(&end)?;
    asm.ret()?;

    asm.leave_local()?;
    Ok(())
}

fn main() -> Result<()> {
    for mode in 0..3 {
        println!("mode={}", mode);
        for grow in 0..2 {
            println!("auto grow={}", if grow == 1 { "on" } else { "off" });

            let mut asm = if grow == 1 {
                CodeAssembler::new_auto_grow(30)?
            } else {
                CodeAssembler::new(4096)?
            };

            match mode {
                0 => gen_mode0(&mut asm)?,
                1 => gen_mode1(&mut asm)?,
                2 => gen_mode2(&mut asm)?,
                _ => unreachable!(),
            }

            asm.ready()?;
            let f: fn(i64) -> i64 = unsafe { asm.get_code() };

            for i in 0..3 {
                let a = EXPECT_TBL[i as usize];
                let b = f(i);
                if a != b {
                    println!("ERR i={}, a={}, b={}", i, a, b);
                    std::process::exit(1);
                }
            }
        }
    }
    println!("ok");
    Ok(())
}
