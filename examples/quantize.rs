//! Port of xbyak sample/quantize.cpp (rewritten from 32-bit to 64-bit)
//! JPEG quantization via multiplicative inverse (fast division).

use rxbyak::*;
use std::time::Instant;

const N: usize = 64;

fn ilog2(x: u32) -> u32 {
    let mut shift = 0;
    while (1u64 << shift) <= x as u64 {
        shift += 1;
    }
    shift - 1
}

/// Generate JIT quantization function for given quantization table.
/// fn(dest: *mut u32, src: *const u32)
fn make_quantize(q_tbl: &[u32; N]) -> Result<(CodeAssembler, fn(*mut u32, *const u32))> {
    let mut asm = CodeAssembler::new(65536)?;

    // System V ABI: RDI = dest, RSI = src
    // We use StackFrame-like approach but manually to match original closely

    for i in 0..N {
        let dividend = q_tbl[i];
        let offset = (i * 4) as i32;

        // Load src[i] into EAX
        asm.mov(EAX, dword_ptr(RSI + offset))?;

        // Fast division: dividend = odd * 2^exponent
        let mut exponent = 0u32;
        let mut odd = dividend;
        while (odd & 1) == 0 {
            odd >>= 1;
            exponent += 1;
        }

        if odd == 1 {
            // Trivial case: just shift
            if exponent > 0 {
                asm.shr(EAX, exponent as u8)?;
            }
        } else {
            let len0 = ilog2(odd) + 1;
            let mut m_high;
            let mut len = len0;

            {
                let round_up: u64 = 1u64 << (32 + len);
                let k = round_up / (0xFFFFFFFFu64 - (0xFFFFFFFFu64 % odd as u64));
                let m_low = round_up / odd as u64;
                m_high = (round_up + k) / odd as u64;

                let mut ml = m_low;
                let mut mh = m_high;
                while (ml >> 1) < (mh >> 1) && len > 0 {
                    ml >>= 1;
                    mh >>= 1;
                    len -= 1;
                }
                m_high = mh;
            }

            let (m, a): (u64, bool);
            if (m_high >> 32) == 0 {
                m = m_high;
                a = false;
            } else {
                len = ilog2(odd);
                let round_down: u64 = 1u64 << (32 + len);
                let m_low = round_down / odd as u64;
                let r = (round_down % odd as u64) as u32;
                m = if r <= (odd >> 1) { m_low } else { m_low + 1 };
                a = true;
            }

            let mut m_final = m;
            let mut len_final = len;
            while (m_final & 1) == 0 {
                m_final >>= 1;
                len_final -= 1;
            }
            len_final += exponent;

            asm.mov(EDX, m_final as i32)?;
            asm.mul(EDX)?;
            if a {
                asm.add(EAX, m_final as i64)?;
                asm.adc(EDX, 0)?;
            }
            if len_final > 0 {
                asm.shr(EDX, len_final as u8)?;
            }
            asm.mov(EAX, EDX)?;
        }

        // Store result to dest[i]
        asm.mov(dword_ptr(RDI + offset), EAX)?;
    }

    asm.ret()?;
    asm.ready()?;
    let f: fn(*mut u32, *const u32) = unsafe { asm.get_code() };
    Ok((asm, f))
}

fn quantize_native(dest: &mut [u32; N], src: &[u32; N], q_tbl: &[u32; N]) {
    for i in 0..N {
        dest[i] = src[i] / q_tbl[i];
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let q: u32 = if args.len() > 1 {
        args[1].parse().unwrap_or(1)
    } else {
        1
    };

    println!("q={}", q);

    #[rustfmt::skip]
    let mut q_tbl: [u32; N] = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ];

    for v in q_tbl.iter_mut() {
        *v /= q;
        if *v == 0 { *v = 1; }
    }

    // Generate random source data
    let mut rng: u32 = 42;
    let mut src = [0u32; N];
    for v in src.iter_mut() {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        *v = (rng >> 16) % 2048;
    }

    let (_asm, jit_quantize) = make_quantize(&q_tbl)?;

    // Validate correctness
    let mut dest_native = [0u32; N];
    let mut dest_jit = [0u32; N];

    quantize_native(&mut dest_native, &src, &q_tbl);
    jit_quantize(dest_jit.as_mut_ptr(), src.as_ptr());

    let mut errors = 0;
    for i in 0..N {
        if dest_native[i] != dest_jit[i] {
            println!("err[{}] {} {}", i, dest_native[i], dest_jit[i]);
            errors += 1;
        }
    }
    if errors == 0 {
        println!("all {} entries match", N);
    }

    // Benchmark
    let count = 10_000_000;

    let start = Instant::now();
    for _ in 0..count {
        quantize_native(&mut dest_native, &src, &q_tbl);
    }
    println!("native: {:.3?}", start.elapsed());

    let start = Instant::now();
    for _ in 0..count {
        jit_quantize(dest_jit.as_mut_ptr(), src.as_ptr());
    }
    println!("jit:    {:.3?}", start.elapsed());

    Ok(())
}
