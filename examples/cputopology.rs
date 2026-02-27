//! Port of xbyak sample/cputopology.cpp
//! CPU topology analysis using CPUID.
//! Note: rxbyak doesn't have a full CpuTopology utility, so we use raw CPUID.

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

fn main() {
    println!();
    println!("rxbyak CPU Topology Info");
    println!("========================");
    println!();

    #[cfg(target_arch = "x86_64")]
    {
        // Get vendor
        let (max_func, ebx, ecx, edx) = cpuid(0, 0);
        let mut vendor = [0u8; 12];
        vendor[0..4].copy_from_slice(&ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&edx.to_le_bytes());
        vendor[8..12].copy_from_slice(&ecx.to_le_bytes());
        println!("Vendor: {}", String::from_utf8_lossy(&vendor));
        println!("Max CPUID function: {}", max_func);

        // Get family/model/stepping
        let (eax1, _, _, _) = cpuid(1, 0);
        let stepping = eax1 & 0xF;
        let model = (eax1 >> 4) & 0xF;
        let family = (eax1 >> 8) & 0xF;
        let ext_model = (eax1 >> 16) & 0xF;
        let ext_family = (eax1 >> 20) & 0xFF;

        let display_family = if family == 0xF { family + ext_family } else { family };
        let display_model = if family == 0x6 || family == 0xF {
            (ext_model << 4) + model
        } else {
            model
        };

        println!("Family: 0x{:X}, Model: 0x{:X}, Stepping: {}", display_family, display_model, stepping);

        // Get cache info via CPUID function 4 (Intel)
        println!("\nCache hierarchy:");
        for idx in 0..8u32 {
            let (eax4, ebx4, ecx4, _) = cpuid(4, idx);
            let cache_type = eax4 & 0x1F;
            if cache_type == 0 { break; }

            let level = (eax4 >> 5) & 0x7;
            let line_size = (ebx4 & 0xFFF) + 1;
            let partitions = ((ebx4 >> 12) & 0x3FF) + 1;
            let ways = ((ebx4 >> 22) & 0x3FF) + 1;
            let sets = ecx4 + 1;
            let size = ways * partitions * line_size * sets;

            let type_str = match cache_type {
                1 => "Data",
                2 => "Instruction",
                3 => "Unified",
                _ => "Unknown",
            };

            if size >= 1024 * 1024 {
                println!("  L{} {} Cache: {:.1} MB, {}-way, {} byte line",
                    level, type_str, size as f64 / (1024.0 * 1024.0), ways, line_size);
            } else {
                println!("  L{} {} Cache: {} KB, {}-way, {} byte line",
                    level, type_str, size / 1024, ways, line_size);
            }
        }

        // Logical processor count
        println!("\nLogical processors: {}", num_cpus());
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("CPU topology info only available on x86_64");
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
