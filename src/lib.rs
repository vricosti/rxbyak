pub mod error;
pub mod operand;
pub mod reg;
pub mod address;
pub mod encoding_flags;
pub mod code_array;
pub mod label;
pub mod encode;
pub mod assembler;
mod mnemonic;
pub mod platform;
pub mod util;

// Re-exports for convenient use
pub use assembler::CodeAssembler;
pub use error::{Error, Result};
pub use operand::{Reg, RegMem, RegMemImm, Kind, Rounding, Segment};
pub use address::{Address, RegExp};
pub use label::{Label, LabelId, JmpType};
pub use encoding_flags::TypeFlags;

// Re-export address frame functions
pub use address::{ptr, byte_ptr, word_ptr, dword_ptr, qword_ptr};
pub use address::{xmmword_ptr, ymmword_ptr, zmmword_ptr, broadcast_ptr};

// Re-export all register constants
pub use reg::*;
