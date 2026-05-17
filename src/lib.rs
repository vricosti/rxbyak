pub mod address;
pub mod assembler;
pub mod code_array;
pub mod encode;
pub mod encoding_flags;
pub mod error;
pub mod label;
mod mnemonic;
pub mod operand;
pub mod platform;
pub mod reg;
pub mod util;

// Re-exports for convenient use
pub use address::{Address, RegExp};
pub use assembler::CodeAssembler;
pub use encoding_flags::TypeFlags;
pub use error::{Error, Result};
pub use label::{JmpType, Label, LabelId};
pub use operand::{Kind, Reg, RegMem, RegMemImm, Rounding, Segment};

// Re-export address frame functions
pub use address::{broadcast_ptr, xmmword_ptr, ymmword_ptr, zmmword_ptr};
pub use address::{byte_ptr, dword_ptr, ptr, qword_ptr, word_ptr};

// Re-export all register constants
pub use reg::*;

// Re-export StackFrame utilities
pub use util::stack_frame::{StackFrame, USE_RCX, USE_RDX};
