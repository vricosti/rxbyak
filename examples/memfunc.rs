//! Port of xbyak sample/memfunc.cpp (adapted for Rust)
//! JIT function that reads fields from a struct pointer.

use rxbyak::*;
#[repr(C)]
struct A {
    x: i32,
    y: i32,
}

impl A {
    fn new(x: i32, y: i32) -> Self {
        A { x, y }
    }
    fn func(&self, a: i32, b: i32, c: i32, d: i32, e: i32) -> i32 {
        self.x + self.y + a + b + c + d + e
    }
}

fn main() -> Result<()> {
    println!("64bit linux");

    // System V ABI: fn(self_ptr, a, b, c, d, e) -> i32
    // RDI=self, ESI=a, EDX=b, ECX=c, R8D=d, R9D=e
    let mut asm = CodeAssembler::new(4096)?;

    // Use raw registers matching System V calling convention
    // 6 params total: self_ptr + 5 ints
    // RDI = self_ptr, RSI = a, RDX = b, RCX = c, R8 = d, R9 = e
    asm.mov(EAX, dword_ptr(RDI.into()))?;     // self.x
    asm.add(EAX, dword_ptr(RDI + 4))?;       // self.y
    asm.add(EAX, ESI)?;                        // a
    asm.add(EAX, EDX)?;                        // b
    asm.add(EAX, ECX)?;                        // c
    asm.add(EAX, R8D)?;                        // d
    asm.add(EAX, R9D)?;                        // e
    asm.ret()?;
    asm.ready()?;

    let jit_func: fn(*const A, i32, i32, i32, i32, i32) -> i32 =
        unsafe { asm.get_code() };

    let mut rng_state: u32 = 42;
    let mut next_rand = || -> i32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state >> 16) & 0x7fff) as i32
    };

    for _ in 0..10 {
        let a = A::new(next_rand(), next_rand());
        let t1 = next_rand();
        let t2 = next_rand();
        let t3 = next_rand();
        let t4 = next_rand();
        let t5 = next_rand();
        let x = a.func(t1, t2, t3, t4, t5);
        let y = jit_func(&a, t1, t2, t3, t4, t5);
        print!("{} {}, {}\n", if x == y { 'o' } else { 'x' }, x, y);
    }

    Ok(())
}
