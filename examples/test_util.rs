//! Port of xbyak sample/test_util.cpp
//! CPU feature detection and popcnt JIT validation.

use rxbyak::*;

#[cfg(target_arch = "x86_64")]
fn cpuid(eax_in: u32, ecx_in: u32) -> (u32, u32, u32, u32) {
    let (eax, ebx, ecx, edx): (u32, u32, u32, u32);
    unsafe {
        std::arch::asm!(
            "push rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "pop rbx",
            inout("eax") eax_in => eax,
            ebx_out = out(reg) ebx,
            inout("ecx") ecx_in => ecx,
            out("edx") edx,
        );
    }
    (eax, ebx, ecx, edx)
}

/// Check if POPCNT is supported via CPUID
#[cfg(target_arch = "x86_64")]
fn has_popcnt() -> bool {
    let (_, _, ecx1, _) = cpuid(1, 0);
    (ecx1 & (1 << 23)) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn has_popcnt() -> bool { false }

fn print_cpu_features() {
    #[cfg(target_arch = "x86_64")]
    {
        // Get vendor
        let (_, ebx, ecx, edx) = cpuid(0, 0);
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&ecx.to_le_bytes());
        let vendor_str = String::from_utf8_lossy(&vendor);
        let is_intel = vendor_str.contains("Intel");
        println!("vendor {}", if is_intel { "intel" } else { "amd" });

        let (_, _, ecx1, edx1) = cpuid(1, 0);

        let features = [
            (edx1, 23, "mmx"),
            (edx1, 25, "sse"),
            (edx1, 26, "sse2"),
            (ecx1, 0, "sse3"),
            (ecx1, 9, "ssse3"),
            (ecx1, 19, "sse41"),
            (ecx1, 20, "sse42"),
            (ecx1, 23, "popcnt"),
            (ecx1, 25, "aesni"),
            (ecx1, 28, "avx"),
            (ecx1, 12, "fma"),
        ];

        let mut line = String::new();
        for (reg, bit, name) in &features {
            if (reg & (1 << bit)) != 0 {
                line.push(' ');
                line.push_str(name);
            }
        }

        // Check extended features (CPUID EAX=7, ECX=0)
        let (_, ebx7, _, _) = cpuid(7, 0);
        let ext_features = [
            (ebx7, 5, "avx2"),
            (ebx7, 16, "avx512f"),
            (ebx7, 17, "avx512dq"),
            (ebx7, 30, "avx512bw"),
            (ebx7, 31, "avx512vl"),
        ];
        for (reg, bit, name) in &ext_features {
            if (reg & (1 << bit)) != 0 {
                line.push(' ');
                line.push_str(name);
            }
        }
        println!("{}", line);
    }
}

fn main() -> Result<()> {
    println!("64bit");
    print_cpu_features();

    if has_popcnt() {
        let n: i32 = 0x12345678; // popcount = 13
        let expected = 13;

        let mut asm = CodeAssembler::new(4096)?;
        asm.mov(EAX, n)?;
        asm.popcnt(EAX, EAX)?;
        asm.ret()?;
        asm.set_protect_mode_re()?;

        let f: fn() -> i32 = unsafe { asm.get_code() };
        let r = f();
        if r == expected {
            println!("popcnt ok");
        } else {
            println!("popcnt ng {} {}", r, expected);
        }

        asm.set_protect_mode_rw()?;
    } else {
        println!("popcnt not supported");
    }

    Ok(())
}
