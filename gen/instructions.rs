/// Instruction tables from gen_code.cpp.
/// These cover SSE, AVX, FMA, and other standard instructions.

use super::Insn;

// ─── TypeFlags constants (sorted to match xbyak avx_type_def.h) ──────────
// Low 3 bits: disp8*N encoding
const T_N1: u64 = 1;
const T_N2: u64 = 2;
const T_N4: u64 = 3;
const T_N8: u64 = 4;
const T_N16: u64 = 5;
const T_N32: u64 = 6;
#[allow(dead_code)]
const T_NX_MASK: u64 = 7;
const T_DUP: u64 = 7; // == T_NX_MASK
const T_N_VL: u64 = 1 << 3;  // N * (1, 2, 4) for VL
#[allow(dead_code)]
const T_APX: u64 = 1 << 4;
const T_66: u64 = 1 << 5;    // pp = 1
const T_F3: u64 = 1 << 6;    // pp = 2
#[allow(dead_code)]
const T_ER_R: u64 = 1 << 7;  // reg{er}
const T_0F: u64 = 1 << 8;
const T_0F38: u64 = 1 << 9;
const T_0F3A: u64 = 1 << 10;
const T_MAP5: u64 = 1 << 11;
#[allow(dead_code)]
const T_L1: u64 = 1 << 12;
const T_W0: u64 = 1 << 13;   // T_EW0 = T_W0
const T_W1: u64 = 1 << 14;   // for VEX
const T_EW1: u64 = 1 << 16;  // for EVEX
const T_YMM: u64 = 1 << 17;  // support YMM, ZMM
const T_EVEX: u64 = 1 << 18;
const T_ER_X: u64 = 1 << 19; // xmm{er}
#[allow(dead_code)]
const T_ER_Y: u64 = 1 << 20; // ymm{er}
const T_ER_Z: u64 = 1 << 21; // zmm{er}
const T_SAE_X: u64 = 1 << 22; // xmm{sae}
const T_SAE_Y: u64 = 1 << 23; // ymm{sae}
const T_SAE_Z: u64 = 1 << 24; // zmm{sae}
const T_MUST_EVEX: u64 = 1 << 25;
const T_B32: u64 = 1 << 26;  // m32bcst
const T_B64: u64 = 1 << 27;  // m64bcst
const T_B16: u64 = (1 << 26) | (1 << 27); // m16bcst
#[allow(dead_code)]
const T_M_K: u64 = 1 << 28;  // mem{k}
#[allow(dead_code)]
const T_VSIB: u64 = 1 << 29;
const T_MEM_EVEX: u64 = 1 << 30; // use evex if mem
const T_MAP6: u64 = 1 << 31;
#[allow(dead_code)]
const T_NF: u64 = 1 << 32;   // T_nf
#[allow(dead_code)]
const T_CODE1_IF1: u64 = 1 << 33;
// bit 34 unused
#[allow(dead_code)]
const T_ND1: u64 = 1 << 35;
#[allow(dead_code)]
const T_ZU: u64 = 1 << 36;
const T_F2: u64 = 1 << 37;   // pp = 3

const T_EW0: u64 = T_W0; // alias

// ─── AVX X_X_XM table (3-operand VEX/EVEX instructions) ─────────────────
// Source: gen_code.cpp putX_X_XM()
// These generate `v<name>(x1, x2, op)` or `v<name>(x1, x2, op, imm8)`
pub static AVX_X_X_XM: &[Insn] = &[
    // blendpd/blendps/dppd/dpps - with imm
    Insn::avx_imm("vblendpd", T_0F3A | T_66 | T_W0 | T_YMM, 0x0D),
    Insn::avx_imm("vblendps", T_0F3A | T_66 | T_W0 | T_YMM, 0x0C),
    Insn::avx_imm("vdppd", T_0F3A | T_66 | T_W0, 0x41),
    Insn::avx_imm("vdpps", T_0F3A | T_66 | T_W0 | T_YMM, 0x40),
    Insn::avx_imm("vmpsadbw", T_0F3A | T_66 | T_W0 | T_YMM, 0x42),
    Insn::avx_imm("vpblendw", T_0F3A | T_66 | T_W0 | T_YMM, 0x0E),
    Insn::avx_imm("vpblendd", T_0F3A | T_66 | T_W0 | T_YMM, 0x02),
    Insn::avx_imm("vroundsd", T_0F3A | T_66 | T_W0, 0x0B),
    Insn::avx_imm("vroundss", T_0F3A | T_66 | T_W0, 0x0A),
    Insn::avx_imm("vpclmulqdq", T_0F3A | T_66 | T_W0 | T_YMM | T_EVEX, 0x44),

    // permil
    Insn::avx("vpermilps", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x0C),
    Insn::avx("vpermilpd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x0D),

    // variable shifts
    Insn::avx("vpsllvd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_B32, 0x47),
    Insn::avx("vpsllvq", T_0F38 | T_66 | T_W1 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x47),
    Insn::avx("vpsravd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_B32, 0x46),
    Insn::avx("vpsrlvd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_B32, 0x45),
    Insn::avx("vpsrlvq", T_0F38 | T_66 | T_W1 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x45),

    // compare with imm
    Insn::avx_imm("vcmppd", T_0F | T_66 | T_YMM, 0xC2),
    Insn::avx_imm("vcmpps", T_0F | T_YMM, 0xC2),
    Insn::avx_imm("vcmpsd", T_0F | T_F2, 0xC2),
    Insn::avx_imm("vcmpss", T_0F | T_F3, 0xC2),

    // conversions
    Insn::avx("vcvtsd2ss", T_0F | T_F2 | T_EVEX | T_EW1 | T_N8 | T_ER_X, 0x5A),
    Insn::avx("vcvtss2sd", T_0F | T_F3 | T_EVEX | T_W0 | T_N4 | T_SAE_X, 0x5A),

    // insertps
    Insn::avx_imm("vinsertps", T_0F3A | T_66 | T_W0 | T_EVEX | T_N4, 0x21),

    // pack
    Insn::avx("vpacksswb", T_0F | T_66 | T_YMM | T_EVEX, 0x63),
    Insn::avx("vpackssdw", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x6B),
    Insn::avx("vpackuswb", T_0F | T_66 | T_YMM | T_EVEX, 0x67),
    Insn::avx("vpackusdw", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x2B),

    // integer arithmetic
    Insn::avx("vpaddb", T_0F | T_66 | T_YMM | T_EVEX, 0xFC),
    Insn::avx("vpaddw", T_0F | T_66 | T_YMM | T_EVEX, 0xFD),
    Insn::avx("vpaddq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0xD4),
    Insn::avx("vpaddsb", T_0F | T_66 | T_YMM | T_EVEX, 0xEC),
    Insn::avx("vpaddsw", T_0F | T_66 | T_YMM | T_EVEX, 0xED),
    Insn::avx("vpaddusb", T_0F | T_66 | T_YMM | T_EVEX, 0xDC),
    Insn::avx("vpaddusw", T_0F | T_66 | T_YMM | T_EVEX, 0xDD),

    Insn::avx_imm("vpalignr", T_0F3A | T_66 | T_YMM | T_EVEX, 0x0F),

    Insn::avx("vpandn", T_0F | T_66 | T_YMM, 0xDF),

    Insn::avx("vpavgb", T_0F | T_66 | T_YMM | T_EVEX, 0xE0),
    Insn::avx("vpavgw", T_0F | T_66 | T_YMM | T_EVEX, 0xE3),

    // compare
    Insn::avx("vpcmpeqb", T_0F | T_66 | T_YMM, 0x74),
    Insn::avx("vpcmpeqw", T_0F | T_66 | T_YMM, 0x75),
    Insn::avx("vpcmpeqd", T_0F | T_66 | T_YMM, 0x76),
    Insn::avx("vpcmpeqq", T_0F38 | T_66 | T_YMM, 0x29),
    Insn::avx("vpcmpgtb", T_0F | T_66 | T_YMM, 0x64),
    Insn::avx("vpcmpgtw", T_0F | T_66 | T_YMM, 0x65),
    Insn::avx("vpcmpgtd", T_0F | T_66 | T_YMM, 0x66),
    Insn::avx("vpcmpgtq", T_0F38 | T_66 | T_YMM, 0x37),

    // horizontal
    Insn::avx("vphaddw", T_0F38 | T_66 | T_YMM, 0x01),
    Insn::avx("vphaddd", T_0F38 | T_66 | T_YMM, 0x02),
    Insn::avx("vphaddsw", T_0F38 | T_66 | T_YMM, 0x03),
    Insn::avx("vphsubw", T_0F38 | T_66 | T_YMM, 0x05),
    Insn::avx("vphsubd", T_0F38 | T_66 | T_YMM, 0x06),
    Insn::avx("vphsubsw", T_0F38 | T_66 | T_YMM, 0x07),

    // multiply-add
    Insn::avx("vpmaddwd", T_0F | T_66 | T_YMM | T_EVEX, 0xF5),
    Insn::avx("vpmaddubsw", T_0F38 | T_66 | T_YMM | T_EVEX, 0x04),

    // max/min
    Insn::avx("vpmaxsb", T_0F38 | T_66 | T_YMM | T_EVEX, 0x3C),
    Insn::avx("vpmaxsw", T_0F | T_66 | T_YMM | T_EVEX, 0xEE),
    Insn::avx("vpmaxsd", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x3D),
    Insn::avx("vpmaxub", T_0F | T_66 | T_YMM | T_EVEX, 0xDE),
    Insn::avx("vpmaxuw", T_0F38 | T_66 | T_YMM | T_EVEX, 0x3E),
    Insn::avx("vpmaxud", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x3F),
    Insn::avx("vpminsb", T_0F38 | T_66 | T_YMM | T_EVEX, 0x38),
    Insn::avx("vpminsw", T_0F | T_66 | T_YMM | T_EVEX, 0xEA),
    Insn::avx("vpminsd", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x39),
    Insn::avx("vpminub", T_0F | T_66 | T_YMM | T_EVEX, 0xDA),
    Insn::avx("vpminuw", T_0F38 | T_66 | T_YMM | T_EVEX, 0x3A),
    Insn::avx("vpminud", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x3B),

    // multiply
    Insn::avx("vpmulhuw", T_0F | T_66 | T_YMM | T_EVEX, 0xE4),
    Insn::avx("vpmulhrsw", T_0F38 | T_66 | T_YMM | T_EVEX, 0x0B),
    Insn::avx("vpmulhw", T_0F | T_66 | T_YMM | T_EVEX, 0xE5),
    Insn::avx("vpmullw", T_0F | T_66 | T_YMM | T_EVEX, 0xD5),
    Insn::avx("vpmulld", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x40),
    Insn::avx("vpmuludq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0xF4),
    Insn::avx("vpmuldq", T_0F38 | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x28),

    Insn::avx("vpsadbw", T_0F | T_66 | T_YMM | T_EVEX, 0xF6),
    Insn::avx("vpshufb", T_0F38 | T_66 | T_YMM | T_EVEX, 0x00),

    // sign
    Insn::avx("vpsignb", T_0F38 | T_66 | T_YMM, 0x08),
    Insn::avx("vpsignw", T_0F38 | T_66 | T_YMM, 0x09),
    Insn::avx("vpsignd", T_0F38 | T_66 | T_YMM, 0x0A),

    // shift reg,reg/mem
    Insn::avx("vpsllw", T_0F | T_66 | T_YMM | T_EVEX | T_N16, 0xF1),
    Insn::avx("vpslld", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_N16, 0xF2),
    Insn::avx("vpsllq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_N16, 0xF3),
    Insn::avx("vpsraw", T_0F | T_66 | T_YMM | T_EVEX | T_N16, 0xE1),
    Insn::avx("vpsrad", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_N16, 0xE2),
    Insn::avx("vpsrlw", T_0F | T_66 | T_YMM | T_EVEX | T_N16, 0xD1),
    Insn::avx("vpsrld", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_N16, 0xD2),
    Insn::avx("vpsrlq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_N16, 0xD3),

    // sub
    Insn::avx("vpsubb", T_0F | T_66 | T_YMM | T_EVEX, 0xF8),
    Insn::avx("vpsubw", T_0F | T_66 | T_YMM | T_EVEX, 0xF9),
    Insn::avx("vpsubq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0xFB),
    Insn::avx("vpsubsb", T_0F | T_66 | T_YMM | T_EVEX, 0xE8),
    Insn::avx("vpsubsw", T_0F | T_66 | T_YMM | T_EVEX, 0xE9),
    Insn::avx("vpsubusb", T_0F | T_66 | T_YMM | T_EVEX, 0xD8),
    Insn::avx("vpsubusw", T_0F | T_66 | T_YMM | T_EVEX, 0xD9),

    // unpack
    Insn::avx("vpunpckhbw", T_0F | T_66 | T_YMM | T_EVEX, 0x68),
    Insn::avx("vpunpckhwd", T_0F | T_66 | T_YMM | T_EVEX, 0x69),
    Insn::avx("vpunpckhdq", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x6A),
    Insn::avx("vpunpckhqdq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x6D),
    Insn::avx("vpunpcklbw", T_0F | T_66 | T_YMM | T_EVEX, 0x60),
    Insn::avx("vpunpcklwd", T_0F | T_66 | T_YMM | T_EVEX, 0x61),
    Insn::avx("vpunpckldq", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x62),
    Insn::avx("vpunpcklqdq", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x6C),

    // reciprocal/rsqrt
    Insn::avx("vrcpss", T_0F | T_F3, 0x53),
    Insn::avx("vrsqrtss", T_0F | T_F3, 0x52),

    // shuffle
    Insn::avx_imm("vshufpd", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0xC6),
    Insn::avx_imm("vshufps", T_0F | T_YMM | T_EVEX | T_W0 | T_B32, 0xC6),

    // sqrt scalar (3-operand AVX forms)
    Insn::avx("vsqrtsd", T_0F | T_F2 | T_EVEX | T_EW1 | T_ER_X | T_N8, 0x51),
    Insn::avx("vsqrtss", T_0F | T_F3 | T_EVEX | T_W0 | T_ER_X | T_N4, 0x51),

    // unpack float
    Insn::avx("vunpckhpd", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x15),
    Insn::avx("vunpckhps", T_0F | T_YMM | T_EVEX | T_W0 | T_B32, 0x15),
    Insn::avx("vunpcklpd", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x14),
    Insn::avx("vunpcklps", T_0F | T_YMM | T_EVEX | T_W0 | T_B32, 0x14),

    // galois field
    Insn::avx_imm("vgf2p8affineinvqb", T_66 | T_0F3A | T_W1 | T_EVEX | T_YMM | T_EW1 | T_SAE_Z | T_B64, 0xCF),
    Insn::avx_imm("vgf2p8affineqb", T_66 | T_0F3A | T_W1 | T_EVEX | T_YMM | T_EW1 | T_SAE_Z | T_B64, 0xCE),
    Insn::avx("vgf2p8mulb", T_66 | T_0F38 | T_W0 | T_EVEX | T_YMM | T_W0 | T_SAE_Z, 0xCF),

    // SM3/SM4
    Insn::avx("vsm3msg1", T_0F38 | T_W0 | T_EVEX | T_W0, 0xDA),
    Insn::avx("vsm3msg2", T_66 | T_0F38 | T_W0 | T_EVEX | T_W0, 0xDA),
    Insn::avx_imm("vsm3rnds2", T_66 | T_0F3A | T_W0 | T_EVEX | T_W0, 0xDE),
    Insn::avx("vsm4key4", T_F3 | T_0F38 | T_W0 | T_EVEX | T_W0, 0xDA),
    Insn::avx("vsm4rnds4", T_F2 | T_0F38 | T_W0 | T_EVEX | T_W0, 0xDA),

    // AES (3-operand AVX forms)
    Insn::avx("vaesenc", T_0F38 | T_66 | T_YMM | T_EVEX, 0xDC),
    Insn::avx("vaesenclast", T_0F38 | T_66 | T_YMM | T_EVEX, 0xDD),
    Insn::avx("vaesdec", T_0F38 | T_66 | T_YMM | T_EVEX, 0xDE),
    Insn::avx("vaesdeclast", T_0F38 | T_66 | T_YMM | T_EVEX, 0xDF),

    // FP16 3-operand arithmetic
    Insn::avx("vaddph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_ER_Z | T_B16, 0x58),
    Insn::avx("vaddsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x58),
    Insn::avx("vsubph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_ER_Z | T_B16, 0x5C),
    Insn::avx("vsubsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x5C),
    Insn::avx("vmulph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_ER_Z | T_B16, 0x59),
    Insn::avx("vmulsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x59),
    Insn::avx("vdivph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_ER_Z | T_B16, 0x5E),
    Insn::avx("vdivsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_ER_X | T_N2, 0x5E),
    Insn::avx("vmaxph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_SAE_Z | T_B16, 0x5F),
    Insn::avx("vmaxsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_SAE_X | T_N2, 0x5F),
    Insn::avx("vminph", T_MAP5 | T_W0 | T_YMM | T_MUST_EVEX | T_SAE_Z | T_B16, 0x5D),
    Insn::avx("vminsh", T_MAP5 | T_F3 | T_W0 | T_MUST_EVEX | T_SAE_X | T_N2, 0x5D),

    // Insert (3-op + imm: ymm/zmm, ymm/zmm, xmm/m, imm8)
    Insn::avx_imm("vinsertf128", T_0F3A | T_66 | T_W0 | T_YMM, 0x18),
    Insn::avx_imm("vinserti128", T_0F3A | T_66 | T_W0 | T_YMM, 0x38),
    Insn::avx_imm("vinsertf32x4", T_0F3A | T_66 | T_MUST_EVEX | T_W0 | T_YMM | T_N16, 0x18),
    Insn::avx_imm("vinsertf64x2", T_0F3A | T_66 | T_MUST_EVEX | T_EW1 | T_YMM | T_N16, 0x18),
    Insn::avx_imm("vinsertf32x8", T_0F3A | T_66 | T_MUST_EVEX | T_W0 | T_YMM | T_N32, 0x1A),
    Insn::avx_imm("vinsertf64x4", T_0F3A | T_66 | T_MUST_EVEX | T_EW1 | T_YMM | T_N32, 0x1A),
    Insn::avx_imm("vinserti32x4", T_0F3A | T_66 | T_MUST_EVEX | T_W0 | T_YMM | T_N16, 0x38),
    Insn::avx_imm("vinserti64x2", T_0F3A | T_66 | T_MUST_EVEX | T_EW1 | T_YMM | T_N16, 0x38),
    Insn::avx_imm("vinserti32x8", T_0F3A | T_66 | T_MUST_EVEX | T_W0 | T_YMM | T_N32, 0x3A),
    Insn::avx_imm("vinserti64x4", T_0F3A | T_66 | T_MUST_EVEX | T_EW1 | T_YMM | T_N32, 0x3A),

    // Permute 3-op (VEX)
    Insn::avx_imm("vperm2f128", T_0F3A | T_66 | T_W0 | T_YMM, 0x06),
    Insn::avx_imm("vperm2i128", T_0F3A | T_66 | T_W0 | T_YMM, 0x46),
    Insn::avx("vpermd", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x36),
    Insn::avx("vpermps", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x16),

    // VNNI dot product
    Insn::avx("vpdpbusd", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x50),
    Insn::avx("vpdpbusds", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x51),
    Insn::avx("vpdpwssd", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x52),
    Insn::avx("vpdpwssds", T_66 | T_0F38 | T_W0 | T_YMM | T_EVEX | T_B32, 0x53),

    // VNNI INT8
    Insn::avx("vpdpbssd", T_F2 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x50),
    Insn::avx("vpdpbssds", T_F2 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x51),
    Insn::avx("vpdpbsud", T_F3 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x50),
    Insn::avx("vpdpbsuds", T_F3 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x51),
    Insn::avx("vpdpbuud", T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x50),
    Insn::avx("vpdpbuuds", T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x51),
    Insn::avx("vpdpwsud", T_F3 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD2),
    Insn::avx("vpdpwsuds", T_F3 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD3),
    Insn::avx("vpdpwusd", T_66 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD2),
    Insn::avx("vpdpwusds", T_66 | T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD3),
    Insn::avx("vpdpwuud", T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD2),
    Insn::avx("vpdpwuuds", T_0F38 | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0xD3),

    // FP16 complex multiply
    Insn::avx("vfmaddcph", T_F3 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B32, 0x56),
    Insn::avx("vfcmaddcph", T_F2 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B32, 0x56),
    Insn::avx("vfmulcph", T_F3 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B32, 0xD6),
    Insn::avx("vfcmulcph", T_F2 | T_MAP6 | T_W0 | T_MUST_EVEX | T_ER_Z | T_YMM | T_B32, 0xD6),

    // SHA-512
    Insn::avx("vsha512msg1", T_F2 | T_0F38 | T_W0 | T_MUST_EVEX, 0xCC),
    Insn::avx("vsha512msg2", T_F2 | T_0F38 | T_W0 | T_MUST_EVEX, 0xCD),
    Insn::avx("vsha512rnds2", T_F2 | T_0F38 | T_W0 | T_MUST_EVEX, 0xCB),
];

// ─── SSE 2-operand instructions (from suffix expansion table) ────────────
// Source: gen_code.cpp, the suffix expansion loop (addps/pd/ss/sd, etc.)
// These generate `<name>(xmm, xmm/m)` — NOT v-prefixed
pub static SSE_2OP: &[Insn] = &[
    // andnps/andnpd
    Insn::sse("andnps", T_0F, 0x55),
    Insn::sse("andnpd", T_0F | T_66, 0x55),
    // max
    Insn::sse("maxps", T_0F, 0x5F),
    Insn::sse("maxss", T_0F | T_F3, 0x5F),
    Insn::sse("maxpd", T_0F | T_66, 0x5F),
    Insn::sse("maxsd", T_0F | T_F2, 0x5F),
    // min
    Insn::sse("minps", T_0F, 0x5D),
    Insn::sse("minss", T_0F | T_F3, 0x5D),
    Insn::sse("minpd", T_0F | T_66, 0x5D),
    Insn::sse("minsd", T_0F | T_F2, 0x5D),
    // rcp/rsqrt
    Insn::sse("rcpps", T_0F, 0x53),
    Insn::sse("rcpss", T_0F | T_F3, 0x53),
    Insn::sse("rsqrtps", T_0F, 0x52),
    Insn::sse("rsqrtss", T_0F | T_F3, 0x52),
    // shuf (with imm)
    Insn::sse_imm("shufps", T_0F, 0xC6),
    Insn::sse_imm("shufpd", T_0F | T_66, 0xC6),
    // cmp (with imm)
    Insn::sse_imm("cmpps", T_0F, 0xC2),
    Insn::sse_imm("cmpss", T_0F | T_F3, 0xC2),
    Insn::sse_imm("cmppd", T_0F | T_66, 0xC2),
    Insn::sse_imm("cmpsd_xmm", T_0F | T_F2, 0xC2), // avoid conflict with x86 cmpsd
    // unpack
    Insn::sse("unpckhps", T_0F, 0x15),
    Insn::sse("unpckhpd", T_0F | T_66, 0x15),
    Insn::sse("unpcklps", T_0F, 0x14),
    Insn::sse("unpcklpd", T_0F | T_66, 0x14),

    // SSE1 special reg-reg moves
    Insn::sse("movhlps", T_0F, 0x12),
    Insn::sse("movlhps", T_0F, 0x16),

    // SSE2 packed integer arithmetic
    Insn::sse("paddb", T_0F | T_66, 0xFC),
    Insn::sse("paddw", T_0F | T_66, 0xFD),
    Insn::sse("paddq", T_0F | T_66, 0xD4),
    Insn::sse("paddsb", T_0F | T_66, 0xEC),
    Insn::sse("paddsw", T_0F | T_66, 0xED),
    Insn::sse("paddusb", T_0F | T_66, 0xDC),
    Insn::sse("paddusw", T_0F | T_66, 0xDD),
    Insn::sse("psubb", T_0F | T_66, 0xF8),
    Insn::sse("psubw", T_0F | T_66, 0xF9),
    Insn::sse("psubq", T_0F | T_66, 0xFB),
    Insn::sse("psubsb", T_0F | T_66, 0xE8),
    Insn::sse("psubsw", T_0F | T_66, 0xE9),
    Insn::sse("psubusb", T_0F | T_66, 0xD8),
    Insn::sse("psubusw", T_0F | T_66, 0xD9),
    Insn::sse("pmullw", T_0F | T_66, 0xD5),
    Insn::sse("pmulhw", T_0F | T_66, 0xE5),
    Insn::sse("pmulhuw", T_0F | T_66, 0xE4),
    Insn::sse("pmuludq", T_0F | T_66, 0xF4),
    Insn::sse("pmaddwd", T_0F | T_66, 0xF5),

    // SSE2 packed compare
    Insn::sse("pcmpeqb", T_0F | T_66, 0x74),
    Insn::sse("pcmpeqw", T_0F | T_66, 0x75),
    Insn::sse("pcmpeqd", T_0F | T_66, 0x76),
    Insn::sse("pcmpgtb", T_0F | T_66, 0x64),
    Insn::sse("pcmpgtw", T_0F | T_66, 0x65),
    Insn::sse("pcmpgtd", T_0F | T_66, 0x66),

    // SSE2 packed misc
    Insn::sse("pandn", T_0F | T_66, 0xDF),
    Insn::sse("pavgb", T_0F | T_66, 0xE0),
    Insn::sse("pavgw", T_0F | T_66, 0xE3),
    Insn::sse("psadbw", T_0F | T_66, 0xF6),
    Insn::sse("packsswb", T_0F | T_66, 0x63),
    Insn::sse("packssdw", T_0F | T_66, 0x6B),
    Insn::sse("packuswb", T_0F | T_66, 0x67),

    // SSE2 packed unpack
    Insn::sse("punpckhbw", T_0F | T_66, 0x68),
    Insn::sse("punpckhwd", T_0F | T_66, 0x69),
    Insn::sse("punpckhdq", T_0F | T_66, 0x6A),
    Insn::sse("punpcklbw", T_0F | T_66, 0x60),
    Insn::sse("punpcklwd", T_0F | T_66, 0x61),
    Insn::sse("punpckldq", T_0F | T_66, 0x62),

    // SSSE3 packed
    Insn::sse("pshufb", T_0F38 | T_66, 0x00),
    Insn::sse("phaddw", T_0F38 | T_66, 0x01),
    Insn::sse("phaddd", T_0F38 | T_66, 0x02),
    Insn::sse("phaddsw", T_0F38 | T_66, 0x03),
    Insn::sse("phsubw", T_0F38 | T_66, 0x05),
    Insn::sse("phsubd", T_0F38 | T_66, 0x06),
    Insn::sse("phsubsw", T_0F38 | T_66, 0x07),
    Insn::sse("psignb", T_0F38 | T_66, 0x08),
    Insn::sse("psignw", T_0F38 | T_66, 0x09),
    Insn::sse("psignd", T_0F38 | T_66, 0x0A),
    Insn::sse("pmaddubsw", T_0F38 | T_66, 0x04),
    Insn::sse("pmulhrsw", T_0F38 | T_66, 0x0B),
    Insn::sse_imm("palignr", T_0F3A | T_66, 0x0F),

    // SSE2 shift by xmm
    Insn::sse("psllw", T_0F | T_66, 0xF1),
    Insn::sse("pslld", T_0F | T_66, 0xF2),
    Insn::sse("psllq", T_0F | T_66, 0xF3),
    Insn::sse("psrlw", T_0F | T_66, 0xD1),
    Insn::sse("psrld", T_0F | T_66, 0xD2),
    Insn::sse("psrlq", T_0F | T_66, 0xD3),
    Insn::sse("psraw", T_0F | T_66, 0xE1),
    Insn::sse("psrad", T_0F | T_66, 0xE2),

    // SSE2 conversions
    Insn::sse("cvtpd2ps", T_0F | T_66, 0x5A),
    Insn::sse("cvtps2pd", T_0F, 0x5A),
    Insn::sse("cvtpd2dq", T_0F | T_F2, 0xE6),
    Insn::sse("cvttpd2dq", T_0F | T_66, 0xE6),
    Insn::sse("cvtdq2pd", T_0F | T_F3, 0xE6),
    Insn::sse("cvtps2dq", T_0F | T_66, 0x5B),
    Insn::sse("cvttps2dq", T_0F | T_F3, 0x5B),
    Insn::sse("cvtdq2ps", T_0F, 0x5B),

    // SSE4.1
    Insn::sse("pabsb", T_0F38 | T_66, 0x1C),
    Insn::sse("pabsw", T_0F38 | T_66, 0x1D),
    Insn::sse("pabsd", T_0F38 | T_66, 0x1E),

    // SSE4.1 with imm
    Insn::sse_imm("blendpd", T_0F3A | T_66, 0x0D),
    Insn::sse_imm("blendps", T_0F3A | T_66, 0x0C),
    Insn::sse_imm("dppd", T_0F3A | T_66, 0x41),
    Insn::sse_imm("dpps", T_0F3A | T_66, 0x40),
    Insn::sse_imm("mpsadbw", T_0F3A | T_66, 0x42),
    Insn::sse_imm("pblendw", T_0F3A | T_66, 0x0E),
    Insn::sse_imm("roundsd", T_0F3A | T_66, 0x0B),
    Insn::sse_imm("roundss", T_0F3A | T_66, 0x0A),
    Insn::sse_imm("roundpd", T_0F3A | T_66, 0x09),
    Insn::sse_imm("roundps", T_0F3A | T_66, 0x08),
    Insn::sse_imm("pclmulqdq", T_0F3A | T_66, 0x44),
    Insn::sse_imm("insertps", T_0F3A | T_66, 0x21),

    // SSE3/SSSE3 2-op (from opSSE pattern)
    Insn::sse("packusdw", T_0F38 | T_66, 0x2B),
    Insn::sse("pcmpeqq", T_0F38 | T_66, 0x29),
    Insn::sse("pcmpgtq", T_0F38 | T_66, 0x37),
    Insn::sse("pmaxsb", T_0F38 | T_66, 0x3C),
    Insn::sse("pmaxsd", T_0F38 | T_66, 0x3D),
    Insn::sse("pmaxuw", T_0F38 | T_66, 0x3E),
    Insn::sse("pmaxud", T_0F38 | T_66, 0x3F),
    Insn::sse("pminsb", T_0F38 | T_66, 0x38),
    Insn::sse("pminsd", T_0F38 | T_66, 0x39),
    Insn::sse("pminuw", T_0F38 | T_66, 0x3A),
    Insn::sse("pminud", T_0F38 | T_66, 0x3B),
    Insn::sse("pmulld", T_0F38 | T_66, 0x40),
    Insn::sse("pmuldq", T_0F38 | T_66, 0x28),

    // SSE3 - only sse, no avx
    Insn::sse("addsubps", T_0F | T_F2, 0xD0),
    Insn::sse("addsubpd", T_0F | T_66, 0xD0),
    Insn::sse("haddps", T_0F | T_F2, 0x7C),
    Insn::sse("haddpd", T_0F | T_66, 0x7C),
    Insn::sse("hsubps", T_0F | T_F2, 0x7D),
    Insn::sse("hsubpd", T_0F | T_66, 0x7D),
    Insn::sse("movshdup", T_0F | T_F3, 0x16),
    Insn::sse("movsldup", T_0F | T_F3, 0x12),
    Insn::sse("movddup", T_0F | T_F2, 0x12),

    // AES (SSE 2-operand)
    Insn::sse("aesenc", T_0F38 | T_66, 0xDC),
    Insn::sse("aesenclast", T_0F38 | T_66, 0xDD),
    Insn::sse("aesdec", T_0F38 | T_66, 0xDE),
    Insn::sse("aesdeclast", T_0F38 | T_66, 0xDF),
    Insn::sse("aesimc", T_0F38 | T_66, 0xDB),
    Insn::sse_imm("aeskeygenassist", T_0F3A | T_66, 0xDF),

    // SSE4.2 string comparison
    Insn::sse_imm("pcmpestri", T_0F3A | T_66, 0x61),
    Insn::sse_imm("pcmpestrm", T_0F3A | T_66, 0x60),
    Insn::sse_imm("pcmpistri", T_0F3A | T_66, 0x63),
    Insn::sse_imm("pcmpistrm", T_0F3A | T_66, 0x62),

    // SSE4.1 test
    Insn::sse("ptest", T_0F38 | T_66, 0x17),

    // SSE4.1 phminposuw
    Insn::sse("phminposuw", T_0F38 | T_66, 0x41),

    // SSE4.1 pmovsx/pmovzx
    Insn::sse("pmovsxbw", T_0F38 | T_66, 0x20),
    Insn::sse("pmovsxbd", T_0F38 | T_66, 0x21),
    Insn::sse("pmovsxbq", T_0F38 | T_66, 0x22),
    Insn::sse("pmovsxwd", T_0F38 | T_66, 0x23),
    Insn::sse("pmovsxwq", T_0F38 | T_66, 0x24),
    Insn::sse("pmovsxdq", T_0F38 | T_66, 0x25),
    Insn::sse("pmovzxbw", T_0F38 | T_66, 0x30),
    Insn::sse("pmovzxbd", T_0F38 | T_66, 0x31),
    Insn::sse("pmovzxbq", T_0F38 | T_66, 0x32),
    Insn::sse("pmovzxwd", T_0F38 | T_66, 0x33),
    Insn::sse("pmovzxwq", T_0F38 | T_66, 0x34),
    Insn::sse("pmovzxdq", T_0F38 | T_66, 0x35),

    // SSE2 misc
    Insn::sse("punpckhqdq", T_0F | T_66, 0x6D),
    Insn::sse("punpcklqdq", T_0F | T_66, 0x6C),

    // SSE2 shuffle with imm
    Insn::sse_imm("pshufd", T_0F | T_66, 0x70),
    Insn::sse_imm("pshufhw", T_0F | T_F3, 0x70),
    Insn::sse_imm("pshuflw", T_0F | T_F2, 0x70),

    // Missing SSE2 basic max/min
    Insn::sse("pmaxsw", T_0F | T_66, 0xEE),
    Insn::sse("pmaxub", T_0F | T_66, 0xDE),
    Insn::sse("pminsw", T_0F | T_66, 0xEA),
    Insn::sse("pminub", T_0F | T_66, 0xDA),
    // SSE4.1 missing
    Insn::sse("pmovmskb", T_0F | T_66, 0xD7),
    // SSE2 lddqu
    Insn::sse("lddqu", T_0F | T_F2, 0xF0),
    // GFNI (legacy SSE forms)
    Insn::sse_imm("gf2p8affineinvqb", T_66 | T_0F3A, 0xCF),
    Insn::sse_imm("gf2p8affineqb", T_66 | T_0F3A, 0xCE),
    Insn::sse("gf2p8mulb", T_66 | T_0F38, 0xCF),
    // SHA (legacy SSE)
    Insn::sse("sha1nexte", T_0F38, 0xC8),
    Insn::sse("sha1msg1", T_0F38, 0xC9),
    Insn::sse("sha1msg2", T_0F38, 0xCA),
    Insn::sse_imm("sha1rnds4", T_0F3A, 0xCC),
    Insn::sse("sha256msg1", T_0F38, 0xCC),
    Insn::sse("sha256msg2", T_0F38, 0xCD),
    Insn::sse("sha256rnds2", T_0F38, 0xCB),
];

// ─── AVX 2-operand (no vvvv) ─────────────────────────────────────────────
// These use opVex(x, 0, op, ...) - 2 operand form with no VEX.vvvv
pub static AVX_X_XM: &[Insn] = &[
    // sqrtps/pd (2-operand AVX form, not scalar)
    Insn::vex_xm("vsqrtps", T_0F | T_YMM | T_EVEX | T_EW0 | T_B32, 0x51),
    Insn::vex_xm("vsqrtpd", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64 | T_ER_Z, 0x51),
    Insn::vex_xm("vrcpps", T_0F | T_YMM, 0x53),
    Insn::vex_xm("vrsqrtps", T_0F | T_YMM, 0x52),

    // permilps/pd imm form
    Insn::vex_xm_imm("vpermilps", T_0F3A | T_66 | T_W0 | T_YMM | T_EVEX | T_EW0 | T_B32, 0x04),
    Insn::vex_xm_imm("vpermilpd", T_0F3A | T_66 | T_W0 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x05),

    // vpermq/vpermpd (imm form)
    Insn::vex_xm_imm("vpermq", T_0F3A | T_66 | T_W1 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x00),
    Insn::vex_xm_imm("vpermpd", T_0F3A | T_66 | T_W1 | T_YMM | T_EVEX | T_EW1 | T_B64, 0x01),

    // shuffle imm (2-operand form, not 3-op)
    Insn::vex_xm_imm("vpshufd", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x70),
    Insn::vex_xm_imm("vpshufhw", T_0F | T_F3 | T_YMM | T_EVEX, 0x70),
    Insn::vex_xm_imm("vpshuflw", T_0F | T_F2 | T_YMM | T_EVEX, 0x70),

    // roundps/pd (2-operand AVX form)
    Insn::vex_xm_imm("vroundps", T_0F3A | T_66 | T_W0 | T_YMM, 0x08),
    Insn::vex_xm_imm("vroundpd", T_0F3A | T_66 | T_W0 | T_YMM, 0x09),

    // abs (2-operand)
    Insn::vex_xm("vpabsb", T_0F38 | T_66 | T_YMM | T_EVEX, 0x1C),
    Insn::vex_xm("vpabsw", T_0F38 | T_66 | T_YMM | T_EVEX, 0x1D),
    Insn::vex_xm("vpabsd", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_B32, 0x1E),

    // broadcast
    Insn::vex_xm("vbroadcastss", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_N4, 0x18),
    Insn::vex_xm("vpbroadcastb", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_N1, 0x78),
    Insn::vex_xm("vpbroadcastw", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_N2, 0x79),
    Insn::vex_xm("vpbroadcastd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_N4, 0x58),
    Insn::vex_xm("vpbroadcastq", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_EW1 | T_N8, 0x59),

    // ptest
    Insn::vex_xm("vptest", T_0F38 | T_66 | T_YMM, 0x17),

    // convert
    Insn::vex_xm("vcvtdq2ps", T_0F | T_YMM | T_EVEX | T_W0 | T_B32 | T_ER_Z, 0x5B),
    Insn::vex_xm("vcvtps2dq", T_0F | T_66 | T_YMM | T_EVEX | T_W0 | T_B32 | T_ER_Z, 0x5B),
    Insn::vex_xm("vcvttps2dq", T_0F | T_F3 | T_YMM | T_EVEX | T_W0 | T_B32 | T_SAE_Z, 0x5B),

    // movshdup/sldup/ddup (AVX)
    Insn::vex_xm("vmovshdup", T_0F | T_F3 | T_YMM | T_EVEX | T_W0, 0x16),
    Insn::vex_xm("vmovsldup", T_0F | T_F3 | T_YMM | T_EVEX | T_W0, 0x12),
    Insn::vex_xm("vmovddup", T_0F | T_F2 | T_YMM | T_EVEX | T_EW1 | T_DUP, 0x12),

    // AES (AVX 2-operand)
    Insn::vex_xm("vaesimc", T_0F38 | T_66 | T_W0, 0xDB),
    Insn::vex_xm_imm("vaeskeygenassist", T_0F3A | T_66, 0xDF),

    // string (AVX 2-operand with imm)
    Insn::vex_xm_imm("vpcmpestri", T_0F3A | T_66, 0x61),
    Insn::vex_xm_imm("vpcmpestrm", T_0F3A | T_66, 0x60),
    Insn::vex_xm_imm("vpcmpistri", T_0F3A | T_66, 0x63),
    Insn::vex_xm_imm("vpcmpistrm", T_0F3A | T_66, 0x62),

    // VEX test
    Insn::vex_xm("vtestps", T_0F38 | T_66 | T_YMM, 0x0E),
    Insn::vex_xm("vtestpd", T_0F38 | T_66 | T_YMM, 0x0F),

    // phminposuw (AVX)
    Insn::vex_xm("vphminposuw", T_0F38 | T_66, 0x41),

    // pmovsxbw etc (AVX)
    Insn::vex_xm("vpmovsxbw", T_0F38 | T_66 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x20),
    Insn::vex_xm("vpmovsxbd", T_0F38 | T_66 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x21),
    Insn::vex_xm("vpmovsxbq", T_0F38 | T_66 | T_YMM | T_EVEX | T_N2 | T_N_VL, 0x22),
    Insn::vex_xm("vpmovsxwd", T_0F38 | T_66 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x23),
    Insn::vex_xm("vpmovsxwq", T_0F38 | T_66 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x24),
    Insn::vex_xm("vpmovsxdq", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_N8 | T_N_VL, 0x25),
    Insn::vex_xm("vpmovzxbw", T_0F38 | T_66 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x30),
    Insn::vex_xm("vpmovzxbd", T_0F38 | T_66 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x31),
    Insn::vex_xm("vpmovzxbq", T_0F38 | T_66 | T_YMM | T_EVEX | T_N2 | T_N_VL, 0x32),
    Insn::vex_xm("vpmovzxwd", T_0F38 | T_66 | T_YMM | T_EVEX | T_N8 | T_N_VL, 0x33),
    Insn::vex_xm("vpmovzxwq", T_0F38 | T_66 | T_YMM | T_EVEX | T_N4 | T_N_VL, 0x34),
    Insn::vex_xm("vpmovzxdq", T_0F38 | T_66 | T_YMM | T_EVEX | T_W0 | T_N8 | T_N_VL, 0x35),

    // Missing broadcasts
    Insn::vex_xm("vbroadcastsd", T_0F38 | T_66 | T_W0 | T_YMM | T_EVEX | T_EW1 | T_N8, 0x19),
    Insn::vex_xm("vbroadcastf128", T_0F38 | T_66 | T_W0 | T_YMM, 0x1A),
    Insn::vex_xm("vbroadcasti128", T_0F38 | T_66 | T_W0 | T_YMM, 0x5A),
    Insn::vex_xm("vbroadcastf32x2", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N8, 0x19),
    Insn::vex_xm("vbroadcastf32x4", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N16, 0x1A),
    Insn::vex_xm("vbroadcastf64x2", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N16, 0x1A),
    Insn::vex_xm("vbroadcastf32x8", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N32, 0x1B),
    Insn::vex_xm("vbroadcastf64x4", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N32, 0x1B),
    Insn::vex_xm("vbroadcasti32x2", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N8, 0x59),
    Insn::vex_xm("vbroadcasti32x4", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N16, 0x5A),
    Insn::vex_xm("vbroadcasti64x2", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N16, 0x5A),
    Insn::vex_xm("vbroadcasti32x8", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_W0 | T_N32, 0x5B),
    Insn::vex_xm("vbroadcasti64x4", T_66 | T_0F38 | T_YMM | T_MUST_EVEX | T_EW1 | T_N32, 0x5B),

    // Widening conversions (2-op, half-width source)
    Insn::vex_xm("vcvtdq2pd", T_0F | T_F3 | T_YMM | T_EVEX | T_W0 | T_B32 | T_N8 | T_N_VL, 0xE6),
    Insn::vex_xm("vcvtps2pd", T_0F | T_YMM | T_EVEX | T_W0 | T_B32 | T_N8 | T_N_VL | T_SAE_Y, 0x5A),
    Insn::vex_xm("vcvtph2ps", T_0F38 | T_66 | T_W0 | T_EVEX | T_YMM | T_N8 | T_N_VL | T_SAE_Y, 0x13),
    Insn::vex_xm("vcvtudq2pd", T_N8 | T_N_VL | T_F3 | T_0F | T_W0 | T_YMM | T_MUST_EVEX | T_B32, 0x7A),

    // Narrowing conversions (2-op, dst narrower than src)
    Insn::vex_xm("vcvtpd2dq", T_0F | T_F2 | T_YMM | T_EVEX | T_EW1 | T_B64 | T_ER_Z | T_N16 | T_N_VL, 0xE6),
    Insn::vex_xm("vcvtpd2ps", T_0F | T_66 | T_YMM | T_EVEX | T_EW1 | T_B64 | T_ER_Z | T_N16 | T_N_VL, 0x5A),
    Insn::vex_xm("vcvttpd2dq", T_66 | T_0F | T_YMM | T_EVEX | T_EW1 | T_B64 | T_SAE_Z | T_N16 | T_N_VL, 0xE6),

    // vlddqu
    Insn::vex_xm("vlddqu", T_0F | T_F2 | T_W0 | T_YMM, 0xF0),

    // vpbroadcastmb2q / vpbroadcastmw2d (k-register source -> xmm/ymm/zmm)
    Insn::vex_xm("vpbroadcastmb2q", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_EW1, 0x2A),
    Insn::vex_xm("vpbroadcastmw2d", T_F3 | T_0F38 | T_MUST_EVEX | T_YMM | T_W0, 0x3A),
];

// ─── AVX VEX move instructions (bidirectional: load + store) ─────────────
pub static AVX_MOV: &[Insn] = &[
    Insn::vex_mov("vmovdqa32", T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW0, 0x6F, 0x7F),
    Insn::vex_mov("vmovdqa64", T_66 | T_0F | T_YMM | T_MUST_EVEX | T_EW1, 0x6F, 0x7F),
    Insn::vex_mov("vmovdqu8", T_F2 | T_0F | T_YMM | T_MUST_EVEX | T_EW0, 0x6F, 0x7F),
    Insn::vex_mov("vmovdqu16", T_F2 | T_0F | T_YMM | T_MUST_EVEX | T_EW1, 0x6F, 0x7F),
    Insn::vex_mov("vmovdqu32", T_F3 | T_0F | T_YMM | T_MUST_EVEX | T_EW0, 0x6F, 0x7F),
    Insn::vex_mov("vmovdqu64", T_F3 | T_0F | T_YMM | T_MUST_EVEX | T_EW1, 0x6F, 0x7F),
];

// ─── FMA instructions ────────────────────────────────────────────────────
// Source: gen_code.cpp FMA section
// 10 base names × 3 orderings × 2 suffixes = 60 instructions
pub static FMA: &[Insn] = &[
    // vfmadd{132,213,231}{pd,ps}
    Insn::avx("vfmadd132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x98),
    Insn::avx("vfmadd213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xA8),
    Insn::avx("vfmadd231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xB8),
    Insn::avx("vfmadd132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x98),
    Insn::avx("vfmadd213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xA8),
    Insn::avx("vfmadd231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xB8),
    // vfmadd{132,213,231}{sd,ss}
    Insn::avx("vfmadd132sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0x99),
    Insn::avx("vfmadd213sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xA9),
    Insn::avx("vfmadd231sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xB9),
    Insn::avx("vfmadd132ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0x99),
    Insn::avx("vfmadd213ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xA9),
    Insn::avx("vfmadd231ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xB9),

    // vfmaddsub
    Insn::avx("vfmaddsub132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x96),
    Insn::avx("vfmaddsub213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xA6),
    Insn::avx("vfmaddsub231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xB6),
    Insn::avx("vfmaddsub132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x96),
    Insn::avx("vfmaddsub213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xA6),
    Insn::avx("vfmaddsub231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xB6),

    // vfmsubadd
    Insn::avx("vfmsubadd132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x97),
    Insn::avx("vfmsubadd213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xA7),
    Insn::avx("vfmsubadd231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xB7),
    Insn::avx("vfmsubadd132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x97),
    Insn::avx("vfmsubadd213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xA7),
    Insn::avx("vfmsubadd231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xB7),

    // vfmsub
    Insn::avx("vfmsub132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x9A),
    Insn::avx("vfmsub213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xAA),
    Insn::avx("vfmsub231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xBA),
    Insn::avx("vfmsub132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x9A),
    Insn::avx("vfmsub213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xAA),
    Insn::avx("vfmsub231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xBA),
    Insn::avx("vfmsub132sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0x9B),
    Insn::avx("vfmsub213sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xAB),
    Insn::avx("vfmsub231sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xBB),
    Insn::avx("vfmsub132ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0x9B),
    Insn::avx("vfmsub213ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xAB),
    Insn::avx("vfmsub231ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xBB),

    // vfnmadd
    Insn::avx("vfnmadd132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x9C),
    Insn::avx("vfnmadd213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xAC),
    Insn::avx("vfnmadd231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xBC),
    Insn::avx("vfnmadd132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x9C),
    Insn::avx("vfnmadd213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xAC),
    Insn::avx("vfnmadd231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xBC),
    Insn::avx("vfnmadd132sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0x9D),
    Insn::avx("vfnmadd213sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xAD),
    Insn::avx("vfnmadd231sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xBD),
    Insn::avx("vfnmadd132ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0x9D),
    Insn::avx("vfnmadd213ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xAD),
    Insn::avx("vfnmadd231ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xBD),

    // vfnmsub
    Insn::avx("vfnmsub132pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0x9E),
    Insn::avx("vfnmsub213pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xAE),
    Insn::avx("vfnmsub231pd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_YMM | T_ER_Z | T_B64, 0xBE),
    Insn::avx("vfnmsub132ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0x9E),
    Insn::avx("vfnmsub213ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xAE),
    Insn::avx("vfnmsub231ps", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_YMM | T_ER_Z | T_B32, 0xBE),
    Insn::avx("vfnmsub132sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0x9F),
    Insn::avx("vfnmsub213sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xAF),
    Insn::avx("vfnmsub231sd", T_0F38 | T_66 | T_EVEX | T_W1 | T_EW1 | T_ER_X | T_N8, 0xBF),
    Insn::avx("vfnmsub132ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0x9F),
    Insn::avx("vfnmsub213ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xAF),
    Insn::avx("vfnmsub231ss", T_0F38 | T_66 | T_EVEX | T_W0 | T_EW0 | T_ER_X | T_N4, 0xBF),
];

// ─── AVX float arithmetic (3-operand) ────────────────────────────────────
// These are the v-prefixed versions of SSE float ops
pub static AVX_FLOAT_3OP: &[Insn] = &[
    // add (already hand-written: vaddps, vaddpd, vaddss, vaddsd)
    Insn::avx("vandnps", T_0F | T_EVEX | T_YMM | T_EW0 | T_B32 | T_N16 | T_N_VL, 0x55),
    Insn::avx("vandnpd", T_66 | T_0F | T_EVEX | T_YMM | T_EW1 | T_B64 | T_N16 | T_N_VL, 0x55),
    Insn::avx("vmaxps", T_0F | T_EVEX | T_YMM | T_EW0 | T_B32 | T_N16 | T_N_VL, 0x5F),
    Insn::avx("vmaxpd", T_66 | T_0F | T_EVEX | T_YMM | T_EW1 | T_B64 | T_N16 | T_N_VL, 0x5F),
    Insn::avx("vmaxss", T_F3 | T_0F | T_EVEX | T_EW0 | T_N4 | T_SAE_X, 0x5F),
    Insn::avx("vmaxsd", T_F2 | T_0F | T_EVEX | T_EW1 | T_N8 | T_SAE_X, 0x5F),
    Insn::avx("vminps", T_0F | T_EVEX | T_YMM | T_EW0 | T_B32 | T_N16 | T_N_VL, 0x5D),
    Insn::avx("vminpd", T_66 | T_0F | T_EVEX | T_YMM | T_EW1 | T_B64 | T_N16 | T_N_VL, 0x5D),
    Insn::avx("vminss", T_F3 | T_0F | T_EVEX | T_EW0 | T_N4 | T_SAE_X, 0x5D),
    Insn::avx("vminsd", T_F2 | T_0F | T_EVEX | T_EW1 | T_N8 | T_SAE_X, 0x5D),
    // vsubss/sd (already hand-written: vsubps, vsubpd but not ss/sd)
    Insn::avx("vsubss", T_F3 | T_0F | T_EVEX | T_EW0 | T_N4 | T_ER_X, 0x5C),
    Insn::avx("vsubsd", T_F2 | T_0F | T_EVEX | T_EW1 | T_N8 | T_ER_X, 0x5C),
    Insn::avx("vmulss", T_F3 | T_0F | T_EVEX | T_EW0 | T_N4 | T_ER_X, 0x59),
    Insn::avx("vmulsd", T_F2 | T_0F | T_EVEX | T_EW1 | T_N8 | T_ER_X, 0x59),
    Insn::avx("vdivss", T_F3 | T_0F | T_EVEX | T_EW0 | T_N4 | T_ER_X, 0x5E),
    Insn::avx("vdivsd", T_F2 | T_0F | T_EVEX | T_EW1 | T_N8 | T_ER_X, 0x5E),

    // SSE3 AVX versions
    Insn::avx("vaddsubps", T_0F | T_F2 | T_YMM, 0xD0),
    Insn::avx("vaddsubpd", T_0F | T_66 | T_YMM, 0xD0),
    Insn::avx("vhaddps", T_0F | T_F2 | T_YMM, 0x7C),
    Insn::avx("vhaddpd", T_0F | T_66 | T_YMM, 0x7C),
    Insn::avx("vhsubps", T_0F | T_F2 | T_YMM, 0x7D),
    Insn::avx("vhsubpd", T_0F | T_66 | T_YMM, 0x7D),
];

// ─── AVX-512 compare (VEX 2-op form) ────────────────────────────────────
pub static AVX_CMP_2OP: &[Insn] = &[
    Insn::vex_xm("vcomiss", T_0F | T_EVEX | T_EW0 | T_SAE_X | T_N4, 0x2F),
    Insn::vex_xm("vcomisd", T_0F | T_66 | T_EVEX | T_EW1 | T_SAE_X | T_N8, 0x2F),
    Insn::vex_xm("vucomiss", T_0F | T_EVEX | T_EW0 | T_SAE_X | T_N4, 0x2E),
    Insn::vex_xm("vucomisd", T_0F | T_66 | T_EVEX | T_EW1 | T_SAE_X | T_N8, 0x2E),
];

/// Get all instruction tables from gen_code.cpp
pub fn all_tables() -> Vec<&'static [Insn]> {
    vec![
        AVX_X_X_XM,
        SSE_2OP,
        AVX_X_XM,
        AVX_MOV,
        FMA,
        AVX_FLOAT_3OP,
        AVX_CMP_2OP,
    ]
}
