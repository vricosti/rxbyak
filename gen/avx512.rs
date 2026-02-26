/// AVX-512 instruction tables from gen_avx512.cpp.
/// These cover EVEX-only instructions (T_MUST_EVEX), FP16, BF16, etc.

use super::Insn;

// ─── TypeFlags constants (matching encoding_flags.rs) ────────────────────
const T_N_VL: u64 = 1 << 3;
const T_66: u64 = 1 << 5;
const T_F3: u64 = 1 << 6;
const T_0F: u64 = 1 << 8;
const T_0F38: u64 = 1 << 9;
const T_0F3A: u64 = 1 << 10;
const T_MAP5: u64 = 1 << 11;
#[allow(dead_code)]
const T_L1: u64 = 1 << 12;
const T_W0: u64 = 1 << 13;
#[allow(dead_code)]
const T_W1: u64 = 1 << 14;
const T_EW1: u64 = 1 << 16;
const T_YMM: u64 = 1 << 17;
const T_ER_X: u64 = 1 << 19;
const T_ER_Z: u64 = 1 << 21;
const T_SAE_X: u64 = 1 << 22;
const T_SAE_Z: u64 = 1 << 24;
const T_MUST_EVEX: u64 = 1 << 25;
const T_B32: u64 = 1 << 26;
const T_B64: u64 = 1 << 27;
const T_B16: u64 = (1 << 26) | (1 << 27);
const T_MAP6: u64 = 1 << 31;
const T_F2: u64 = 1 << 37;

const T_N1: u64 = 1;
const T_N2: u64 = 2;
const T_N4: u64 = 3;
const T_N8: u64 = 4;
#[allow(dead_code)]
const T_N16: u64 = 5;

// ─── AVX-512 compare: K result instructions ──────────────────────────────
// Pattern: K, XMM/YMM/ZMM, XMM/YMM/ZMM/M [, imm8]
pub static AVX512_K_X_XM: &[Insn] = &[
    // vcmp with imm
    Insn::avx_k_imm("vcmppd", T_0F | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_YMM | T_66 | T_B64, 0xC2),
    Insn::avx_k_imm("vcmpps", T_0F | T_MUST_EVEX | T_W0 | T_SAE_Z | T_YMM | T_B32, 0xC2),
    Insn::avx_k_imm("vcmpsd", T_0F | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_F2 | T_N8, 0xC2),
    Insn::avx_k_imm("vcmpss", T_0F | T_MUST_EVEX | T_W0 | T_SAE_Z | T_F3 | T_N4, 0xC2),
    Insn::avx_k_imm("vcmpph", T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_Z | T_YMM | T_B16, 0xC2),
    Insn::avx_k_imm("vcmpsh", T_F3 | T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0xC2),

    // vpcmpeq (no imm)
    Insn::avx_k("vpcmpeqb", T_66 | T_0F | T_MUST_EVEX | T_YMM, 0x74),
    Insn::avx_k("vpcmpeqw", T_66 | T_0F | T_MUST_EVEX | T_YMM, 0x75),
    Insn::avx_k("vpcmpeqd", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_B32, 0x76),
    Insn::avx_k("vpcmpeqq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x29),

    // vpcmpgt (no imm)
    Insn::avx_k("vpcmpgtb", T_66 | T_0F | T_MUST_EVEX | T_YMM, 0x64),
    Insn::avx_k("vpcmpgtw", T_66 | T_0F | T_MUST_EVEX | T_YMM, 0x65),
    Insn::avx_k("vpcmpgtd", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0x66),
    Insn::avx_k("vpcmpgtq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x37),

    // vpcmp with imm
    Insn::avx_k_imm("vpcmpb", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_W0, 0x3F),
    Insn::avx_k_imm("vpcmpub", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_W0, 0x3E),
    Insn::avx_k_imm("vpcmpw", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1, 0x3F),
    Insn::avx_k_imm("vpcmpuw", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1, 0x3E),
    Insn::avx_k_imm("vpcmpd", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0x1F),
    Insn::avx_k_imm("vpcmpud", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0x1E),
    Insn::avx_k_imm("vpcmpq", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x1F),
    Insn::avx_k_imm("vpcmpuq", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x1E),

    // vptestm / vptestnm (no imm)
    Insn::avx_k("vptestmb", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0, 0x26),
    Insn::avx_k("vptestmw", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x26),
    Insn::avx_k("vptestmd", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0x27),
    Insn::avx_k("vptestmq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x27),
    Insn::avx_k("vptestnmb", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0, 0x26),
    Insn::avx_k("vptestnmw", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x26),
    Insn::avx_k("vptestnmd", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0x27),
    Insn::avx_k("vptestnmq", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x27),
];

// ─── AVX-512 3-operand EVEX-only ─────────────────────────────────────────
// Pattern: XMM/YMM/ZMM, XMM/YMM/ZMM, XMM/YMM/ZMM/M [, imm8]
pub static AVX512_X_X_XM: &[Insn] = &[
    // align
    Insn::avx_imm("valignd", T_MUST_EVEX | T_66 | T_0F3A | T_W0 | T_YMM, 0x03),
    Insn::avx_imm("valignq", T_MUST_EVEX | T_66 | T_0F3A | T_EW1 | T_YMM, 0x03),

    // logical (integer, d/q width)
    Insn::avx("vpandd", T_MUST_EVEX | T_YMM | T_66 | T_0F | T_W0 | T_B32, 0xDB),
    Insn::avx("vpandq", T_MUST_EVEX | T_YMM | T_66 | T_0F | T_EW1 | T_B64, 0xDB),
    Insn::avx("vpandnd", T_MUST_EVEX | T_YMM | T_66 | T_0F | T_W0 | T_B32, 0xDF),
    Insn::avx("vpandnq", T_MUST_EVEX | T_YMM | T_66 | T_0F | T_EW1 | T_B64, 0xDF),
    Insn::avx("vpord", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0xEB),
    Insn::avx("vporq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0xEB),
    Insn::avx("vpxord", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_W0 | T_B32, 0xEF),
    Insn::avx("vpxorq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0xEF),

    // max/min q
    Insn::avx("vpmaxsq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x3D),
    Insn::avx("vpmaxuq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x3F),
    Insn::avx("vpminsq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x39),
    Insn::avx("vpminuq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x3B),

    // multiply
    Insn::avx("vpmullq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x40),

    // shift
    Insn::avx("vpsraq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_N16, 0xE2),
    Insn::avx("vpsravq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64, 0x46),
    Insn::avx("vpsravw", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x11),
    Insn::avx("vpsllvw", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x12),
    Insn::avx("vpsrlvw", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x10),

    // permute
    Insn::avx("vpermb", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0, 0x8D),
    Insn::avx("vpermw", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x8D),

    // blend mask
    Insn::avx("vblendmpd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x65),
    Insn::avx("vblendmps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x65),
    Insn::avx("vpblendmb", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0, 0x66),
    Insn::avx("vpblendmw", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1, 0x66),
    Insn::avx("vpblendmd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x64),
    Insn::avx("vpblendmq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x64),

    // permt2
    Insn::avx("vpermt2b", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0, 0x7D),
    Insn::avx("vpermt2w", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1, 0x7D),
    Insn::avx("vpermt2d", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x7E),
    Insn::avx("vpermt2q", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x7E),
    Insn::avx("vpermt2ps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x7F),
    Insn::avx("vpermt2pd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x7F),

    // permi2
    Insn::avx("vpermi2b", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0, 0x75),
    Insn::avx("vpermi2w", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1, 0x75),
    Insn::avx("vpermi2d", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x76),
    Insn::avx("vpermi2q", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x76),
    Insn::avx("vpermi2ps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x77),
    Insn::avx("vpermi2pd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x77),

    // multiply-add 52-bit
    Insn::avx("vpmadd52luq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0xB4),
    Insn::avx("vpmadd52huq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0xB5),

    // ternary logic
    Insn::avx_imm("vpternlogd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x25),
    Insn::avx_imm("vpternlogq", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x25),

    // getexp scalar
    Insn::avx("vgetexpsd", T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_SAE_X | T_N8, 0x43),
    Insn::avx("vgetexpss", T_66 | T_0F38 | T_MUST_EVEX | T_W0 | T_SAE_X | T_N4, 0x43),
    Insn::avx("vgetexpsh", T_66 | T_MAP6 | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0x43),

    // getmant scalar
    Insn::avx_imm("vgetmantsd", T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_SAE_X | T_N8, 0x27),
    Insn::avx_imm("vgetmantss", T_66 | T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N4, 0x27),
    Insn::avx_imm("vgetmantsh", T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0x27),

    // fixupimm
    Insn::avx_imm("vfixupimmpd", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0x54),
    Insn::avx_imm("vfixupimmps", T_66 | T_0F3A | T_MUST_EVEX | T_YMM | T_W0 | T_B32 | T_SAE_Z, 0x54),
    Insn::avx_imm("vfixupimmsd", T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_N8, 0x55),
    Insn::avx_imm("vfixupimmss", T_66 | T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_Z | T_N4, 0x55),

    // rcp14 scalar
    Insn::avx("vrcp14sd", T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_N8, 0x4D),
    Insn::avx("vrcp14ss", T_66 | T_0F38 | T_MUST_EVEX | T_W0 | T_N4, 0x4D),
    Insn::avx("vrcpsh", T_66 | T_MAP6 | T_MUST_EVEX | T_W0 | T_N2, 0x4D),

    // rsqrt14 scalar
    Insn::avx("vrsqrt14sd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N8, 0x4F),
    Insn::avx("vrsqrt14ss", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N4, 0x4F),
    Insn::avx("vrsqrtsh", T_66 | T_MAP6 | T_MUST_EVEX | T_W0 | T_N2, 0x4F),

    // sqrt scalar FP16
    Insn::avx("vsqrtsh", T_F3 | T_MAP5 | T_MUST_EVEX | T_W0 | T_ER_X | T_N2, 0x51),

    // rndscale scalar
    Insn::avx_imm("vrndscalesd", T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_N8 | T_SAE_X, 0x0B),
    Insn::avx_imm("vrndscaless", T_66 | T_0F3A | T_MUST_EVEX | T_W0 | T_N4 | T_SAE_X, 0x0A),
    Insn::avx_imm("vrndscalesh", T_0F3A | T_MUST_EVEX | T_W0 | T_N2 | T_SAE_X, 0x0A),

    // scalef
    Insn::avx("vscalefpd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x2C),
    Insn::avx("vscalefps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_ER_Z, 0x2C),
    Insn::avx("vscalefsd", T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_ER_X | T_N8, 0x2D),
    Insn::avx("vscalefss", T_66 | T_0F38 | T_MUST_EVEX | T_W0 | T_ER_X | T_N4, 0x2D),
    Insn::avx("vscalefph", T_66 | T_MAP6 | T_YMM | T_MUST_EVEX | T_W0 | T_B16 | T_ER_Z, 0x2C),
    Insn::avx("vscalefsh", T_66 | T_MAP6 | T_MUST_EVEX | T_W0 | T_ER_X | T_N2, 0x2D),

    // dbpsadbw
    Insn::avx_imm("vdbpsadbw", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0, 0x42),

    // multishift
    Insn::avx("vpmultishiftqb", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x83),

    // rotate variable
    Insn::avx("vprolvd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x15),
    Insn::avx("vprolvq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x15),
    Insn::avx("vprorvd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x14),
    Insn::avx("vprorvq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x14),

    // rcp28/rsqrt28 (AVX-512ER)
    Insn::avx("vrcp28sd", T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_N8 | T_SAE_X, 0xCB),
    Insn::avx("vrcp28ss", T_66 | T_0F38 | T_MUST_EVEX | T_W0 | T_N4 | T_SAE_X, 0xCB),
    Insn::avx("vrsqrt28sd", T_66 | T_0F38 | T_MUST_EVEX | T_EW1 | T_N8 | T_SAE_X, 0xCD),
    Insn::avx("vrsqrt28ss", T_66 | T_0F38 | T_MUST_EVEX | T_W0 | T_N4 | T_SAE_X, 0xCD),

    // range
    Insn::avx_imm("vrangepd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_SAE_Z, 0x50),
    Insn::avx_imm("vrangeps", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_SAE_Z, 0x50),
    Insn::avx_imm("vrangesd", T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_SAE_X | T_N8, 0x51),
    Insn::avx_imm("vrangess", T_66 | T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N4, 0x51),

    // reduce scalar
    Insn::avx_imm("vreducesd", T_66 | T_0F3A | T_MUST_EVEX | T_EW1 | T_SAE_X | T_N8, 0x57),
    Insn::avx_imm("vreducess", T_66 | T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N4, 0x57),
    Insn::avx_imm("vreducesh", T_0F3A | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0x57),

    // shift concatenate (VBMI2)
    Insn::avx_imm("vpshldw", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z, 0x70),
    Insn::avx_imm("vpshldd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x71),
    Insn::avx_imm("vpshldq", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_B64, 0x71),
    Insn::avx("vpshldvw", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z, 0x70),
    Insn::avx("vpshldvd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x71),
    Insn::avx("vpshldvq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_B64, 0x71),
    Insn::avx_imm("vpshrdw", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z, 0x72),
    Insn::avx_imm("vpshrdd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x73),
    Insn::avx_imm("vpshrdq", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_B64, 0x73),
    Insn::avx("vpshrdvw", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z, 0x72),
    Insn::avx("vpshrdvd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x73),
    Insn::avx("vpshrdvq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_B64, 0x73),

    // BF16/NE conversions and dot product
    Insn::avx("vcvtne2ps2bf16", T_F2 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x72),
    Insn::avx("vdpbf16ps", T_F3 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x52),

    // FP16 conversions (3-operand)
    Insn::avx("vcvtsd2sh", T_F2 | T_MAP5 | T_MUST_EVEX | T_EW1 | T_ER_X | T_N8, 0x5A),
    Insn::avx("vcvtsh2sd", T_F3 | T_MAP5 | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0x5A),
    Insn::avx("vcvtsh2ss", T_MAP6 | T_MUST_EVEX | T_W0 | T_SAE_X | T_N2, 0x13),
    Insn::avx("vcvtss2sh", T_MAP5 | T_MUST_EVEX | T_W0 | T_ER_X | T_N4, 0x1D),

    // BF16 arithmetic
    Insn::avx("vaddbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x58),
    Insn::avx("vdivbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x5E),
    Insn::avx("vmaxbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x5F),
    Insn::avx("vminbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x5D),
    Insn::avx("vmulbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x59),
    Insn::avx("vscalefbf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x2C),
    Insn::avx("vsubbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x5C),

    // BF16 FMA
    Insn::avx("vfmadd132bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x98),
    Insn::avx("vfmadd213bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xA8),
    Insn::avx("vfmadd231bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xB8),
    Insn::avx("vfnmadd132bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x9C),
    Insn::avx("vfnmadd213bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xAC),
    Insn::avx("vfnmadd231bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xBC),
    Insn::avx("vfmsub132bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x9A),
    Insn::avx("vfmsub213bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xAA),
    Insn::avx("vfmsub231bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xBA),
    Insn::avx("vfnmsub132bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x9E),
    Insn::avx("vfnmsub213bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xAE),
    Insn::avx("vfnmsub231bf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0xBE),

    // AVX10 conversions (3-op)
    Insn::avx("vcvt2ps2phx", T_MUST_EVEX | T_66 | T_0F38 | T_W0 | T_YMM | T_B32 | T_ER_Z, 0x67),

    // dot product
    Insn::avx("vdpphps", T_MUST_EVEX | T_0F38 | T_W0 | T_YMM | T_SAE_Z | T_B32, 0x52),

    // minmax
    Insn::avx_imm("vminmaxbf16", T_MUST_EVEX | T_F2 | T_0F3A | T_W0 | T_YMM | T_B16, 0x52),
    Insn::avx_imm("vminmaxpd", T_MUST_EVEX | T_66 | T_0F3A | T_EW1 | T_YMM | T_B64 | T_SAE_Z, 0x52),
    Insn::avx_imm("vminmaxph", T_MUST_EVEX | T_0F3A | T_W0 | T_YMM | T_B16 | T_SAE_Z, 0x52),
    Insn::avx_imm("vminmaxps", T_MUST_EVEX | T_66 | T_0F3A | T_W0 | T_YMM | T_B32 | T_SAE_Z, 0x52),
    Insn::avx_imm("vminmaxsd", T_MUST_EVEX | T_66 | T_0F3A | T_EW1 | T_SAE_X | T_N8, 0x53),
    Insn::avx_imm("vminmaxsh", T_MUST_EVEX | T_0F3A | T_W0 | T_SAE_X | T_N2, 0x53),
    Insn::avx_imm("vminmaxss", T_MUST_EVEX | T_66 | T_0F3A | T_W0 | T_SAE_X | T_N4, 0x53),
];

// ─── AVX-512 2-operand EVEX-only ─────────────────────────────────────────
// Pattern: XMM/YMM/ZMM, XMM/YMM/ZMM/M [, imm8]
pub static AVX512_X_XM: &[Insn] = &[
    // conversions
    Insn::vex_xm("vcvtpd2qq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_ER_Z, 0x7B),
    Insn::vex_xm("vcvtpd2uqq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_ER_Z, 0x79),
    Insn::vex_xm("vcvtps2udq", T_0F | T_MUST_EVEX | T_YMM | T_W0 | T_B32 | T_ER_Z, 0x79),
    Insn::vex_xm("vcvtqq2pd", T_F3 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_ER_Z, 0xE6),
    Insn::vex_xm("vcvttpd2qq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0x7A),
    Insn::vex_xm("vcvttpd2uqq", T_66 | T_0F | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0x78),
    Insn::vex_xm("vcvttps2udq", T_0F | T_MUST_EVEX | T_YMM | T_W0 | T_B32 | T_SAE_Z, 0x78),
    Insn::vex_xm("vcvtudq2ps", T_F2 | T_0F | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_ER_Z, 0x7A),
    Insn::vex_xm("vcvtuqq2pd", T_F3 | T_0F | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x7A),

    // expand
    Insn::vex_xm("vexpandpd", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_N8, 0x88),
    Insn::vex_xm("vexpandps", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0 | T_N4, 0x88),
    Insn::vex_xm("vpexpandd", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0 | T_N4, 0x89),
    Insn::vex_xm("vpexpandq", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_N8, 0x89),

    // getexp packed
    Insn::vex_xm("vgetexppd", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1 | T_B64 | T_SAE_Z, 0x42),
    Insn::vex_xm("vgetexpps", T_66 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0 | T_B32 | T_SAE_Z, 0x42),
    Insn::vex_xm("vgetexpph", T_66 | T_MAP6 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_SAE_Z, 0x42),

    // getmant packed (with imm)
    Insn::vex_xm_imm("vgetmantpd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_SAE_Z, 0x26),
    Insn::vex_xm_imm("vgetmantps", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_SAE_Z, 0x26),
    Insn::vex_xm_imm("vgetmantph", T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B16 | T_SAE_Z, 0x26),

    // rcp14 packed
    Insn::vex_xm("vrcp14pd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x4C),
    Insn::vex_xm("vrcp14ps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x4C),
    Insn::vex_xm("vrcpph", T_66 | T_MAP6 | T_MUST_EVEX | T_YMM | T_W0 | T_B16, 0x4C),

    // rsqrt14 packed
    Insn::vex_xm("vrsqrt14pd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x4E),
    Insn::vex_xm("vrsqrt14ps", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x4E),
    Insn::vex_xm("vrsqrtph", T_66 | T_MAP6 | T_YMM | T_MUST_EVEX | T_W0 | T_B16, 0x4E),

    // sqrt packed FP16
    Insn::vex_xm("vsqrtph", T_MAP5 | T_YMM | T_MUST_EVEX | T_W0 | T_ER_Z | T_B16, 0x51),
    Insn::vex_xm("vsqrtbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_YMM | T_B16, 0x51),

    // rndscale packed (with imm)
    Insn::vex_xm_imm("vrndscalepd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_SAE_Z, 0x09),
    Insn::vex_xm_imm("vrndscaleps", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_SAE_Z, 0x08),
    Insn::vex_xm_imm("vrndscaleph", T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B16 | T_SAE_Z, 0x08),

    // conflict / lzcnt
    Insn::vex_xm("vpconflictd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0xC4),
    Insn::vex_xm("vpconflictq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0xC4),
    Insn::vex_xm("vplzcntd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_B32, 0x44),
    Insn::vex_xm("vplzcntq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_B64, 0x44),

    // reduce packed (with imm)
    Insn::vex_xm_imm("vreducepd", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_EW1 | T_B64 | T_SAE_Z, 0x56),
    Insn::vex_xm_imm("vreduceps", T_66 | T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B32 | T_SAE_Z, 0x56),
    Insn::vex_xm_imm("vreduceph", T_0F3A | T_YMM | T_MUST_EVEX | T_W0 | T_B16 | T_SAE_Z, 0x56),

    // popcnt
    Insn::vex_xm("vpopcntb", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z, 0x54),
    Insn::vex_xm("vpopcntw", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z, 0x54),
    Insn::vex_xm("vpopcntd", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_B32, 0x55),
    Insn::vex_xm("vpopcntq", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_B64, 0x55),

    // expand byte/word
    Insn::vex_xm("vpexpandb", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_SAE_Z | T_N1, 0x62),
    Insn::vex_xm("vpexpandw", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_SAE_Z | T_N2, 0x62),

    // FP16 2-operand conversions
    Insn::vex_xm("vcvtph2uw", T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_ER_Z, 0x7D),
    Insn::vex_xm("vcvtph2w", T_66 | T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_ER_Z, 0x7D),
    Insn::vex_xm("vcvttph2uw", T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_SAE_Z, 0x7C),
    Insn::vex_xm("vcvttph2w", T_66 | T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_SAE_Z, 0x7C),
    Insn::vex_xm("vcvtuw2ph", T_F2 | T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_ER_Z, 0x7D),
    Insn::vex_xm("vcvtw2ph", T_F3 | T_MAP5 | T_MUST_EVEX | T_YMM | T_W0 | T_B16 | T_ER_Z, 0x7D),

    // compare FP16/BF16
    Insn::vex_xm("vcomish", T_MUST_EVEX | T_MAP5 | T_W0 | T_SAE_X | T_N2, 0x2F),
    Insn::vex_xm("vucomish", T_MUST_EVEX | T_MAP5 | T_W0 | T_SAE_X | T_N2, 0x2E),
    Insn::vex_xm("vcomisbf16", T_MUST_EVEX | T_66 | T_MAP5 | T_W0 | T_N2, 0x2F),

    // AVX10 comx
    Insn::vex_xm("vcomxsd", T_MUST_EVEX | T_F2 | T_0F | T_EW1 | T_SAE_X | T_N8, 0x2F),
    Insn::vex_xm("vcomxsh", T_MUST_EVEX | T_F3 | T_MAP5 | T_W0 | T_SAE_X | T_N2, 0x2F),
    Insn::vex_xm("vcomxss", T_MUST_EVEX | T_F3 | T_0F | T_W0 | T_SAE_X | T_N4, 0x2F),
    Insn::vex_xm("vucomxsd", T_MUST_EVEX | T_F2 | T_0F | T_EW1 | T_SAE_X | T_N8, 0x2E),
    Insn::vex_xm("vucomxsh", T_MUST_EVEX | T_F3 | T_MAP5 | T_W0 | T_SAE_X | T_N2, 0x2E),
    Insn::vex_xm("vucomxss", T_MUST_EVEX | T_F3 | T_0F | T_W0 | T_SAE_X | T_N4, 0x2E),

    // AVX10 conversion instructions
    Insn::vex_xm("vcvtbf162ibs", T_MUST_EVEX | T_YMM | T_F2 | T_MAP5 | T_W0 | T_B16, 0x69),
    Insn::vex_xm("vcvtbf162iubs", T_MUST_EVEX | T_YMM | T_F2 | T_MAP5 | T_W0 | T_B16, 0x6B),
    Insn::vex_xm("vcvttbf162ibs", T_MUST_EVEX | T_YMM | T_F2 | T_MAP5 | T_W0 | T_B16, 0x68),
    Insn::vex_xm("vcvttbf162iubs", T_MUST_EVEX | T_YMM | T_F2 | T_MAP5 | T_W0 | T_B16, 0x6A),
    Insn::vex_xm("vcvttpd2qqs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_EW1 | T_B64 | T_SAE_Z, 0x6D),
    Insn::vex_xm("vcvttpd2uqqs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_EW1 | T_B64 | T_SAE_Z, 0x6C),
    Insn::vex_xm("vcvtph2ibs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B16 | T_ER_Z, 0x69),
    Insn::vex_xm("vcvtph2iubs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B16 | T_ER_Z, 0x6B),
    Insn::vex_xm("vcvttph2ibs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B16 | T_SAE_Z, 0x68),
    Insn::vex_xm("vcvttph2iubs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B16 | T_SAE_Z, 0x6A),
    Insn::vex_xm("vcvttps2dqs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B32 | T_SAE_Z, 0x6D),
    Insn::vex_xm("vcvtps2ibs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_W0 | T_B32 | T_ER_Z, 0x69),
    Insn::vex_xm("vcvtps2iubs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_W0 | T_B32 | T_ER_Z, 0x6B),
    Insn::vex_xm("vcvttps2ibs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_W0 | T_B32 | T_SAE_Z, 0x68),
    Insn::vex_xm("vcvttps2iubs", T_MUST_EVEX | T_YMM | T_66 | T_MAP5 | T_W0 | T_B32 | T_SAE_Z, 0x6A),
    Insn::vex_xm("vcvttps2udqs", T_MUST_EVEX | T_YMM | T_MAP5 | T_W0 | T_B32 | T_SAE_Z, 0x6C),

    // BF16 misc
    Insn::vex_xm("vgetexpbf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x42),
    Insn::vex_xm_imm("vgetmantbf16", T_MUST_EVEX | T_F2 | T_0F3A | T_W0 | T_YMM | T_B16, 0x26),
    Insn::vex_xm("vrcpbf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x4C),
    Insn::vex_xm_imm("vreducebf16", T_MUST_EVEX | T_F2 | T_0F3A | T_W0 | T_YMM | T_B16, 0x56),
    Insn::vex_xm_imm("vrndscalebf16", T_MUST_EVEX | T_F2 | T_0F3A | T_W0 | T_YMM | T_B16, 0x08),
    Insn::vex_xm("vrsqrtbf16", T_MUST_EVEX | T_MAP6 | T_W0 | T_YMM | T_B16, 0x4E),
];

// ─── FP16 FMA instructions ───────────────────────────────────────────────
// v{name}{132,213,231}{ph,sh}
pub static FP16_FMA: &[Insn] = &[
    // vfmadd ph (packed)
    Insn::avx("vfmadd132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x98),
    Insn::avx("vfmadd213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xA8),
    Insn::avx("vfmadd231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xB8),
    // vfmadd sh (scalar)
    Insn::avx("vfmadd132sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x99),
    Insn::avx("vfmadd213sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xA9),
    Insn::avx("vfmadd231sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xB9),

    // vfmaddsub ph
    Insn::avx("vfmaddsub132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x96),
    Insn::avx("vfmaddsub213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xA6),
    Insn::avx("vfmaddsub231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xB6),

    // vfmsubadd ph
    Insn::avx("vfmsubadd132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x97),
    Insn::avx("vfmsubadd213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xA7),
    Insn::avx("vfmsubadd231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xB7),

    // vfmsub ph/sh
    Insn::avx("vfmsub132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x9A),
    Insn::avx("vfmsub213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xAA),
    Insn::avx("vfmsub231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xBA),
    Insn::avx("vfmsub132sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x9B),
    Insn::avx("vfmsub213sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xAB),
    Insn::avx("vfmsub231sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xBB),

    // vfnmadd ph/sh
    Insn::avx("vfnmadd132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x9C),
    Insn::avx("vfnmadd213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xAC),
    Insn::avx("vfnmadd231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xBC),
    Insn::avx("vfnmadd132sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x9D),
    Insn::avx("vfnmadd213sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xAD),
    Insn::avx("vfnmadd231sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xBD),

    // vfnmsub ph/sh
    Insn::avx("vfnmsub132ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0x9E),
    Insn::avx("vfnmsub213ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xAE),
    Insn::avx("vfnmsub231ph", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B16, 0xBE),
    Insn::avx("vfnmsub132sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x9F),
    Insn::avx("vfnmsub213sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xAF),
    Insn::avx("vfnmsub231sh", T_66 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0xBF),
];

/// Get all AVX-512 instruction tables.
pub fn all_tables() -> Vec<&'static [Insn]> {
    vec![
        AVX512_K_X_XM,
        AVX512_X_X_XM,
        AVX512_X_XM,
        FP16_FMA,
    ]
}
