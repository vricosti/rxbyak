/// CPU feature detection via CPUID.
///
/// Port of xbyak_util.h `Cpu` class. Detects x86/x64 CPU features
/// at runtime using the CPUID instruction.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};

/// 128-bit CPU feature type (low 64 + high 64 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuType {
    lo: u64,
    hi: u64,
}

impl CpuType {
    pub const fn new(lo: u64, hi: u64) -> Self {
        Self { lo, hi }
    }
    pub const fn from_id(id: u32) -> Self {
        if id < 64 {
            Self { lo: 1u64 << id, hi: 0 }
        } else {
            Self { lo: 0, hi: 1u64 << (id - 64) }
        }
    }
    pub fn contains(self, other: Self) -> bool {
        (self.lo & other.lo) == other.lo && (self.hi & other.hi) == other.hi
    }
    pub fn is_empty(self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

impl std::ops::BitOr for CpuType {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self { lo: self.lo | rhs.lo, hi: self.hi | rhs.hi }
    }
}

impl std::ops::BitOrAssign for CpuType {
    fn bitor_assign(&mut self, rhs: Self) {
        self.lo |= rhs.lo;
        self.hi |= rhs.hi;
    }
}

impl std::ops::BitAnd for CpuType {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self { lo: self.lo & rhs.lo, hi: self.hi & rhs.hi }
    }
}

// Feature constants — matches xbyak_util.h IDs exactly.
macro_rules! cpu_feature {
    ($name:ident, $id:expr) => {
        pub const $name: CpuType = CpuType::from_id($id);
    };
}

cpu_feature!(MMX,              0);
cpu_feature!(MMX2,             1);
cpu_feature!(CMOV,             2);
cpu_feature!(SSE,              3);
cpu_feature!(SSE2,             4);
cpu_feature!(SSE3,             5);
cpu_feature!(SSSE3,            6);
cpu_feature!(SSE41,            7);
cpu_feature!(SSE42,            8);
cpu_feature!(POPCNT,           9);
cpu_feature!(AESNI,           10);
cpu_feature!(AVX512_FP16,     11);
cpu_feature!(OSXSAVE,         12);
cpu_feature!(PCLMULQDQ,       13);
cpu_feature!(AVX,             14);
cpu_feature!(FMA,             15);
cpu_feature!(F3DN,            16);
cpu_feature!(E3DN,            17);
cpu_feature!(WAITPKG,         18);
cpu_feature!(RDTSCP,          19);
cpu_feature!(AVX2,            20);
cpu_feature!(BMI1,            21);
cpu_feature!(BMI2,            22);
cpu_feature!(LZCNT,           23);
cpu_feature!(INTEL,           24);
cpu_feature!(AMD,             25);
cpu_feature!(ENHANCED_REP,    26);
cpu_feature!(RDRAND,          27);
cpu_feature!(ADX,             28);
cpu_feature!(RDSEED,          29);
cpu_feature!(SMAP,            30);
cpu_feature!(HLE,             31);
cpu_feature!(RTM,             32);
cpu_feature!(F16C,            33);
cpu_feature!(MOVBE,           34);
cpu_feature!(AVX512F,         35);
cpu_feature!(AVX512DQ,        36);
cpu_feature!(AVX512_IFMA,     37);
cpu_feature!(AVX512PF,        38);
cpu_feature!(AVX512ER,        39);
cpu_feature!(AVX512CD,        40);
cpu_feature!(AVX512BW,        41);
cpu_feature!(AVX512VL,        42);
cpu_feature!(AVX512_VBMI,     43);
cpu_feature!(AVX512_4VNNIW,   44);
cpu_feature!(AVX512_4FMAPS,   45);
cpu_feature!(PREFETCHWT1,     46);
cpu_feature!(PREFETCHW,       47);
cpu_feature!(SHA,             48);
cpu_feature!(MPX,             49);
cpu_feature!(AVX512_VBMI2,    50);
cpu_feature!(GFNI,            51);
cpu_feature!(VAES,            52);
cpu_feature!(VPCLMULQDQ,      53);
cpu_feature!(AVX512_VNNI,     54);
cpu_feature!(AVX512_BITALG,   55);
cpu_feature!(AVX512_VPOPCNTDQ, 56);
cpu_feature!(AVX512_BF16,     57);
cpu_feature!(AVX512_VP2INTERSECT, 58);
cpu_feature!(AMX_TILE,        59);
cpu_feature!(AMX_INT8,        60);
cpu_feature!(AMX_BF16,        61);
cpu_feature!(AVX_VNNI,        62);
cpu_feature!(CLFLUSHOPT,      63);
cpu_feature!(CLDEMOTE,        64);
cpu_feature!(MOVDIRI,         65);
cpu_feature!(MOVDIR64B,       66);
cpu_feature!(CLZERO,          67);
cpu_feature!(AMX_FP16,        68);
cpu_feature!(AVX_VNNI_INT8,   69);
cpu_feature!(AVX_NE_CONVERT,  70);
cpu_feature!(AVX_IFMA,        71);
cpu_feature!(RAO_INT,         72);
cpu_feature!(CMPCCXADD,       73);
cpu_feature!(PREFETCHITI,     74);
cpu_feature!(SERIALIZE,       75);
cpu_feature!(UINTR,           76);
cpu_feature!(XSAVE,           77);
cpu_feature!(SHA512,          78);
cpu_feature!(SM3,             79);
cpu_feature!(SM4,             80);
cpu_feature!(AVX_VNNI_INT16,  81);
cpu_feature!(APX_F,           82);
cpu_feature!(AVX10,           83);
cpu_feature!(AESKLE,          84);
cpu_feature!(WIDE_KL,         85);
cpu_feature!(KEYLOCKER,       86);
cpu_feature!(KEYLOCKER_WIDE,  87);
cpu_feature!(SSE4A,           88);
cpu_feature!(CLWB,            89);
cpu_feature!(TSXLDTRK,        90);
cpu_feature!(AMX_TRANSPOSE,   91);
cpu_feature!(AMX_TF32,        92);
cpu_feature!(AMX_AVX512,      93);
cpu_feature!(AMX_MOVRS,       94);
cpu_feature!(AMX_FP8,         95);
cpu_feature!(MOVRS,           96);
cpu_feature!(HYBRID,          97);

/// CPU feature detection.
pub struct Cpu {
    type_: CpuType,
    pub model: u32,
    pub family: u32,
    pub stepping: u32,
    pub ext_model: u32,
    pub ext_family: u32,
    pub display_family: u32,
    pub display_model: u32,
    num_cores: [u32; 2],
    data_cache_size: [u32; 10],
    cores_sharing_data_cache: [u32; 10],
    data_cache_levels: u32,
    avx10_version: u32,
}

fn extract_bit(val: u32, base: u32, end: u32) -> u32 {
    (val >> base) & ((1u32 << (end + 1 - base)) - 1)
}

fn get32_as_le(s: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*s)
}

/// Check if [ebx:ecx:edx] matches a 12-byte vendor string.
fn is_vendor(ebx: u32, ecx: u32, edx: u32, vendor: &[u8; 12]) -> bool {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&vendor[0..4]);
    let v0 = get32_as_le(&buf);
    buf.copy_from_slice(&vendor[4..8]);
    let v1 = get32_as_le(&buf);
    buf.copy_from_slice(&vendor[8..12]);
    let v2 = get32_as_le(&buf);
    ebx == v0 && edx == v1 && ecx == v2
}

impl Cpu {
    /// Detect CPU features. On non-x86 platforms, returns empty feature set.
    pub fn new() -> Self {
        let mut cpu = Cpu {
            type_: CpuType::default(),
            model: 0,
            family: 0,
            stepping: 0,
            ext_model: 0,
            ext_family: 0,
            display_family: 0,
            display_model: 0,
            num_cores: [0; 2],
            data_cache_size: [0; 10],
            cores_sharing_data_cache: [0; 10],
            data_cache_levels: 0,
            avx10_version: 0,
        };
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        cpu.detect();
        cpu
    }

    /// Check if the CPU has a specific feature.
    pub fn has(&self, feature: CpuType) -> bool {
        self.type_.contains(feature)
    }

    /// Get number of logical processors per topology level.
    /// `SmtLevel` (1) returns threads per core, `CoreLevel` (2) returns cores per package.
    pub fn get_num_cores(&self, level: u32) -> u32 {
        match level {
            1 => self.num_cores[0], // SmtLevel
            2 => {
                if self.num_cores[0] == 0 { 0 }
                else { self.num_cores[1] / self.num_cores[0] }
            } // CoreLevel
            _ => 0,
        }
    }

    /// Number of data cache levels detected.
    pub fn data_cache_levels(&self) -> u32 {
        self.data_cache_levels
    }

    /// Data cache size in bytes for level `i` (0-indexed).
    pub fn data_cache_size(&self, i: u32) -> Option<u32> {
        if i < self.data_cache_levels {
            Some(self.data_cache_size[i as usize])
        } else {
            None
        }
    }

    /// Number of cores sharing data cache at level `i`.
    pub fn cores_sharing_data_cache(&self, i: u32) -> Option<u32> {
        if i < self.data_cache_levels {
            Some(self.cores_sharing_data_cache[i as usize])
        } else {
            None
        }
    }

    /// AVX10 version (0 if not supported).
    pub fn avx10_version(&self) -> u32 {
        self.avx10_version
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect(&mut self) {
        unsafe { self.detect_impl() }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn detect_impl(&mut self) {
        let r = __cpuid(0);
        let max_num = r.eax;

        if is_vendor(r.ebx, r.ecx, r.edx, b"AuthenticAMD") {
            self.type_ |= AMD;
            let r2 = __cpuid(0x80000001);
            if r2.edx & (1 << 31) != 0 {
                self.type_ |= F3DN | PREFETCHW;
            }
            if r2.edx & (1 << 29) != 0 {
                self.type_ |= PREFETCHW;
            }
        } else if is_vendor(r.ebx, r.ecx, r.edx, b"GenuineIntel") {
            self.type_ |= INTEL;
        }

        // Extended flags
        let rext = __cpuid(0x80000000);
        let max_extended = rext.eax;
        if max_extended >= 0x80000001 {
            let r2 = __cpuid(0x80000001);
            if r2.ecx & (1 << 5) != 0 { self.type_ |= LZCNT; }
            if r2.ecx & (1 << 6) != 0 { self.type_ |= SSE4A; }
            if r2.ecx & (1 << 8) != 0 { self.type_ |= PREFETCHW; }
            if r2.edx & (1 << 15) != 0 { self.type_ |= CMOV; }
            if r2.edx & (1 << 22) != 0 { self.type_ |= MMX2; }
            if r2.edx & (1 << 27) != 0 { self.type_ |= RDTSCP; }
            if r2.edx & (1 << 30) != 0 { self.type_ |= E3DN; }
            if r2.edx & (1 << 31) != 0 { self.type_ |= F3DN; }
        }
        if max_extended >= 0x80000008 {
            let r2 = __cpuid(0x80000008);
            if r2.ebx & 1 != 0 { self.type_ |= CLZERO; }
        }

        // Leaf 1 — basic features
        let r1 = __cpuid(1);
        if r1.ecx & (1 << 0) != 0 { self.type_ |= SSE3; }
        if r1.ecx & (1 << 1) != 0 { self.type_ |= PCLMULQDQ; }
        if r1.ecx & (1 << 9) != 0 { self.type_ |= SSSE3; }
        if r1.ecx & (1 << 19) != 0 { self.type_ |= SSE41; }
        if r1.ecx & (1 << 20) != 0 { self.type_ |= SSE42; }
        if r1.ecx & (1 << 22) != 0 { self.type_ |= MOVBE; }
        if r1.ecx & (1 << 23) != 0 { self.type_ |= POPCNT; }
        if r1.ecx & (1 << 25) != 0 { self.type_ |= AESNI; }
        if r1.ecx & (1 << 26) != 0 { self.type_ |= XSAVE; }
        if r1.ecx & (1 << 27) != 0 { self.type_ |= OSXSAVE; }
        if r1.ecx & (1 << 29) != 0 { self.type_ |= F16C; }
        if r1.ecx & (1 << 30) != 0 { self.type_ |= RDRAND; }
        if r1.edx & (1 << 15) != 0 { self.type_ |= CMOV; }
        if r1.edx & (1 << 23) != 0 { self.type_ |= MMX; }
        if r1.edx & (1 << 25) != 0 { self.type_ |= MMX2 | SSE; }
        if r1.edx & (1 << 26) != 0 { self.type_ |= SSE2; }

        // AVX/AVX-512 require OS XSAVE support
        if self.has(OSXSAVE) {
            let bv = _xgetbv(0);
            if (bv & 6) == 6 {
                if r1.ecx & (1 << 12) != 0 { self.type_ |= FMA; }
                if r1.ecx & (1 << 28) != 0 { self.type_ |= AVX; }
                // AVX-512 state check
                if ((bv >> 5) & 7) == 7 {
                    let r7 = __cpuid_count(7, 0);
                    if r7.ebx & (1 << 16) != 0 { self.type_ |= AVX512F; }
                    if self.has(AVX512F) {
                        if r7.ebx & (1 << 17) != 0 { self.type_ |= AVX512DQ; }
                        if r7.ebx & (1 << 21) != 0 { self.type_ |= AVX512_IFMA; }
                        if r7.ebx & (1 << 26) != 0 { self.type_ |= AVX512PF; }
                        if r7.ebx & (1 << 27) != 0 { self.type_ |= AVX512ER; }
                        if r7.ebx & (1 << 28) != 0 { self.type_ |= AVX512CD; }
                        if r7.ebx & (1 << 30) != 0 { self.type_ |= AVX512BW; }
                        if r7.ebx & (1 << 31) != 0 { self.type_ |= AVX512VL; }
                        if r7.ecx & (1 << 1) != 0 { self.type_ |= AVX512_VBMI; }
                        if r7.ecx & (1 << 6) != 0 { self.type_ |= AVX512_VBMI2; }
                        if r7.ecx & (1 << 11) != 0 { self.type_ |= AVX512_VNNI; }
                        if r7.ecx & (1 << 12) != 0 { self.type_ |= AVX512_BITALG; }
                        if r7.ecx & (1 << 14) != 0 { self.type_ |= AVX512_VPOPCNTDQ; }
                        if r7.edx & (1 << 2) != 0 { self.type_ |= AVX512_4VNNIW; }
                        if r7.edx & (1 << 3) != 0 { self.type_ |= AVX512_4FMAPS; }
                        if r7.edx & (1 << 8) != 0 { self.type_ |= AVX512_VP2INTERSECT; }
                        if self.has(AVX512BW) && (r7.edx & (1 << 23) != 0) {
                            self.type_ |= AVX512_FP16;
                        }
                    }
                }
            }
        }

        // Leaf 7 — extended features
        if max_num >= 7 {
            let r7 = __cpuid_count(7, 0);
            let max_sub = r7.eax;
            if self.has(AVX) && (r7.ebx & (1 << 5) != 0) { self.type_ |= AVX2; }
            if r7.ebx & (1 << 3) != 0 { self.type_ |= BMI1; }
            if r7.ebx & (1 << 4) != 0 { self.type_ |= HLE; }
            if r7.ebx & (1 << 8) != 0 { self.type_ |= BMI2; }
            if r7.ebx & (1 << 9) != 0 { self.type_ |= ENHANCED_REP; }
            if r7.ebx & (1 << 11) != 0 { self.type_ |= RTM; }
            if r7.ebx & (1 << 14) != 0 { self.type_ |= MPX; }
            if r7.ebx & (1 << 18) != 0 { self.type_ |= RDSEED; }
            if r7.ebx & (1 << 19) != 0 { self.type_ |= ADX; }
            if r7.ebx & (1 << 20) != 0 { self.type_ |= SMAP; }
            if r7.ebx & (1 << 23) != 0 { self.type_ |= CLFLUSHOPT; }
            if r7.ebx & (1 << 24) != 0 { self.type_ |= CLWB; }
            if r7.ebx & (1 << 29) != 0 { self.type_ |= SHA; }
            if r7.ecx & (1 << 0) != 0 { self.type_ |= PREFETCHWT1; }
            if r7.ecx & (1 << 5) != 0 { self.type_ |= WAITPKG; }
            if r7.ecx & (1 << 8) != 0 { self.type_ |= GFNI; }
            if r7.ecx & (1 << 9) != 0 { self.type_ |= VAES; }
            if r7.ecx & (1 << 10) != 0 { self.type_ |= VPCLMULQDQ; }
            if r7.ecx & (1 << 23) != 0 { self.type_ |= KEYLOCKER; }
            if r7.ecx & (1 << 25) != 0 { self.type_ |= CLDEMOTE; }
            if r7.ecx & (1 << 27) != 0 { self.type_ |= MOVDIRI; }
            if r7.ecx & (1 << 28) != 0 { self.type_ |= MOVDIR64B; }
            if r7.edx & (1 << 5) != 0 { self.type_ |= UINTR; }
            if r7.edx & (1 << 14) != 0 { self.type_ |= SERIALIZE; }
            if r7.edx & (1 << 15) != 0 { self.type_ |= HYBRID; }
            if r7.edx & (1 << 16) != 0 { self.type_ |= TSXLDTRK; }
            if r7.edx & (1 << 22) != 0 { self.type_ |= AMX_BF16; }
            if r7.edx & (1 << 24) != 0 { self.type_ |= AMX_TILE; }
            if r7.edx & (1 << 25) != 0 { self.type_ |= AMX_INT8; }

            if max_sub >= 1 {
                let r71 = __cpuid_count(7, 1);
                if r71.eax & (1 << 0) != 0 { self.type_ |= SHA512; }
                if r71.eax & (1 << 1) != 0 { self.type_ |= SM3; }
                if r71.eax & (1 << 2) != 0 { self.type_ |= SM4; }
                if r71.eax & (1 << 3) != 0 { self.type_ |= RAO_INT; }
                if r71.eax & (1 << 4) != 0 { self.type_ |= AVX_VNNI; }
                if self.has(AVX512F) && (r71.eax & (1 << 5) != 0) {
                    self.type_ |= AVX512_BF16;
                }
                if r71.eax & (1 << 7) != 0 { self.type_ |= CMPCCXADD; }
                if r71.eax & (1 << 21) != 0 { self.type_ |= AMX_FP16; }
                if r71.eax & (1 << 23) != 0 { self.type_ |= AVX_IFMA; }
                if r71.eax & (1 << 31) != 0 { self.type_ |= MOVRS; }
                if r71.edx & (1 << 4) != 0 { self.type_ |= AVX_VNNI_INT8; }
                if r71.edx & (1 << 5) != 0 { self.type_ |= AVX_NE_CONVERT; }
                if r71.edx & (1 << 10) != 0 { self.type_ |= AVX_VNNI_INT16; }
                if r71.edx & (1 << 14) != 0 { self.type_ |= PREFETCHITI; }
                if r71.edx & (1 << 19) != 0 { self.type_ |= AVX10; }
                if r71.edx & (1 << 21) != 0 { self.type_ |= APX_F; }

                let r1e = __cpuid_count(0x1e, 1);
                if r1e.eax & (1 << 4) != 0 { self.type_ |= AMX_FP8; }
                if r1e.eax & (1 << 5) != 0 { self.type_ |= AMX_TRANSPOSE; }
                if r1e.eax & (1 << 6) != 0 { self.type_ |= AMX_TF32; }
                if r1e.eax & (1 << 7) != 0 { self.type_ |= AMX_AVX512; }
                if r1e.eax & (1 << 8) != 0 { self.type_ |= AMX_MOVRS; }
            }
        }

        if max_num >= 0x19 {
            let r19 = __cpuid_count(0x19, 0);
            if r19.ebx & 1 != 0 { self.type_ |= AESKLE; }
            if r19.ebx & (1 << 2) != 0 { self.type_ |= WIDE_KL; }
            if self.has(KEYLOCKER) || self.has(AESKLE) || self.has(WIDE_KL) {
                self.type_ |= KEYLOCKER_WIDE;
            }
        }

        if self.has(AVX10) && max_num >= 0x24 {
            let r24 = __cpuid_count(0x24, 0);
            self.avx10_version = r24.ebx & 0x7F;
        }

        self.set_family();
        self.set_num_cores();
        self.set_cache_hierarchy();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn set_family(&mut self) {
        let r = unsafe { __cpuid(1) };
        self.stepping = extract_bit(r.eax, 0, 3);
        self.model = extract_bit(r.eax, 4, 7);
        self.family = extract_bit(r.eax, 8, 11);
        self.ext_model = extract_bit(r.eax, 16, 19);
        self.ext_family = extract_bit(r.eax, 20, 27);
        self.display_family = if self.family == 0x0f {
            self.family + self.ext_family
        } else {
            self.family
        };
        self.display_model = if (self.has(INTEL) && self.family == 6) || self.family == 0x0f {
            (self.ext_model << 4) + self.model
        } else {
            self.model
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn set_num_cores(&mut self) {
        if !self.has(INTEL) && !self.has(AMD) { return; }

        let r0 = unsafe { __cpuid(0) };
        if r0.eax >= 0xB {
            let rb = unsafe { __cpuid_count(0xB, 0) };
            if rb.eax != 0 || rb.ebx != 0 {
                for i in 0..2u32 {
                    let r = unsafe { __cpuid_count(0xB, i) };
                    let level = extract_bit(r.ecx, 8, 15);
                    if level == 1 || level == 2 {
                        self.num_cores[(level - 1) as usize] = extract_bit(r.ebx, 0, 15);
                    }
                }
                self.num_cores[0] = self.num_cores[0].max(1);
                self.num_cores[1] = self.num_cores[1].max(self.num_cores[0]);
                return;
            }
        }

        // Legacy method
        let r1 = unsafe { __cpuid(1) };
        let logical = extract_bit(r1.ebx, 16, 23);
        let htt = extract_bit(r1.edx, 28, 28);

        if self.has(AMD) {
            let mut physical = 0u32;
            let rext = unsafe { __cpuid(0x80000000) };
            let max_ext = rext.eax;
            if max_ext >= 0x80000008 {
                let r8 = unsafe { __cpuid(0x80000008) };
                physical = extract_bit(r8.ecx, 0, 7) + 1;
            }
            if htt == 0 {
                self.num_cores = [1, 1];
            } else if physical > 1 {
                if self.display_family >= 0x17 && max_ext >= 0x8000001E {
                    let re = unsafe { __cpuid(0x8000001E) };
                    let threads_per_cu = extract_bit(re.ebx, 8, 15) + 1;
                    physical /= threads_per_cu;
                }
                self.num_cores[0] = if physical > 0 { logical / physical } else { 1 };
                self.num_cores[1] = logical;
            } else {
                self.num_cores[0] = 1;
                self.num_cores[1] = if logical > 1 { logical } else { 2 };
            }
        } else {
            // Intel legacy
            let mut physical = 0u32;
            let r0b = unsafe { __cpuid(0) };
            if r0b.eax >= 4 {
                let r4 = unsafe { __cpuid(4) };
                physical = extract_bit(r4.eax, 26, 31) + 1;
            }
            if htt == 0 {
                self.num_cores = [1, 1];
            } else if physical > 1 {
                self.num_cores[0] = if physical > 0 { logical / physical } else { 1 };
                self.num_cores[1] = logical;
            } else {
                self.num_cores[0] = 1;
                self.num_cores[1] = if logical > 0 { logical } else { 1 };
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn set_cache_hierarchy(&mut self) {
        if self.has(AMD) {
            let rext = unsafe { __cpuid(0x80000000) };
            if rext.eax >= 0x8000001D {
                self.data_cache_levels = 0;
                for sub in 0..10u32 {
                    let r = unsafe { __cpuid_count(0x8000001D, sub) };
                    let cache_type = extract_bit(r.eax, 0, 4);
                    if cache_type == 0 { break; }
                    if cache_type == 2 { continue; } // instruction cache
                    let fully_assoc = extract_bit(r.eax, 9, 9);
                    let mut sharing = extract_bit(r.eax, 14, 25) + 1;
                    let ways = extract_bit(r.ebx, 22, 31) + 1;
                    let partitions = extract_bit(r.ebx, 12, 21) + 1;
                    let line_size = extract_bit(r.ebx, 0, 11) + 1;
                    let sets = r.ecx + 1;
                    let mut size = line_size * partitions * ways;
                    if fully_assoc == 0 { size *= sets; }
                    if sub > 0 {
                        sharing = sharing.min(self.num_cores[1]);
                        sharing /= self.cores_sharing_data_cache[0].max(1);
                    }
                    let lvl = self.data_cache_levels as usize;
                    self.data_cache_size[lvl] = size;
                    self.cores_sharing_data_cache[lvl] = sharing;
                    self.data_cache_levels += 1;
                }
                self.cores_sharing_data_cache[0] = self.cores_sharing_data_cache[0].min(1);
            } else if rext.eax >= 0x80000006 {
                self.data_cache_levels = 1;
                let r5 = unsafe { __cpuid(0x80000005) };
                self.data_cache_size[0] = extract_bit(r5.ecx, 24, 31) * 1024;
                self.cores_sharing_data_cache[0] = 1;
                let r6 = unsafe { __cpuid(0x80000006) };
                if extract_bit(r6.ecx, 12, 15) > 0 {
                    self.data_cache_levels = 2;
                    self.data_cache_size[1] = extract_bit(r6.ecx, 16, 31) * 1024;
                    self.cores_sharing_data_cache[1] = 1;
                }
                if extract_bit(r6.edx, 12, 15) > 0 {
                    self.data_cache_levels = 3;
                    self.data_cache_size[2] = extract_bit(r6.edx, 18, 31) * 512 * 1024;
                    self.cores_sharing_data_cache[2] = self.num_cores[1];
                }
            }
        } else if self.has(INTEL) {
            let smt_width = self.num_cores[0];
            let logical_cores = self.num_cores[1];
            for i in 0..10u32 {
                let r = unsafe { __cpuid_count(4, i) };
                let cache_type = extract_bit(r.eax, 0, 4);
                if cache_type == 0 { break; } // NO_CACHE
                if cache_type == 1 || cache_type == 3 { // DATA or UNIFIED
                    let mut actual = extract_bit(r.eax, 14, 25) + 1;
                    if logical_cores != 0 { actual = actual.min(logical_cores); }
                    let size = (extract_bit(r.ebx, 22, 31) + 1)
                        * (extract_bit(r.ebx, 12, 21) + 1)
                        * (extract_bit(r.ebx, 0, 11) + 1)
                        * (r.ecx + 1);
                    let lvl = self.data_cache_levels as usize;
                    self.data_cache_size[lvl] = size;
                    let sw = if smt_width == 0 && cache_type == 1 { actual } else { smt_width };
                    self.cores_sharing_data_cache[lvl] = if sw > 0 { (actual / sw).max(1) } else { 1 };
                    self.data_cache_levels += 1;
                }
            }
        }
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_type_ops() {
        let a = CpuType::from_id(0);
        let b = CpuType::from_id(1);
        let ab = a | b;
        assert!(ab.contains(a));
        assert!(ab.contains(b));
        assert!(!a.contains(b));
    }

    #[test]
    fn test_cpu_detect() {
        let cpu = Cpu::new();
        // On any modern x86, SSE2 should be present
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(cpu.has(SSE2));
        let _ = cpu.display_family;
        let _ = cpu.display_model;
    }

    #[test]
    fn test_cpu_feature_ids() {
        assert_eq!(MMX.lo, 1 << 0);
        assert_eq!(AVX.lo, 1 << 14);
        assert_eq!(AVX512F.lo, 1 << 35);
        assert_eq!(CLFLUSHOPT.lo, 1 << 63);
        assert_eq!(CLDEMOTE.hi, 1 << 0);
        assert_eq!(HYBRID.hi, 1 << (97 - 64));
    }
}
