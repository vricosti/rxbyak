use crate::error::{Error, Result};
use crate::platform::{self, ProtectMode};

const DEFAULT_MAX_SIZE: usize = 4096;

/// Buffer allocation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocMode {
    /// Use a user-provided buffer (no alignment, no protection).
    UserBuf,
    /// Use internally allocated memory (aligned, protected).
    Alloc,
    /// Automatically grow memory as needed.
    AutoGrow,
}

/// Label save mode for address resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelMode {
    /// Use displacement as-is.
    AsIs,
    /// Absolute address (subtract top).
    Abs,
    /// Address + top (for AutoGrow label resolution).
    AddTop,
}

/// Information about an address that needs to be patched.
#[derive(Clone, Debug)]
pub(crate) struct AddrInfo {
    pub code_offset: usize,
    pub jmp_addr: u64,
    pub jmp_size: u8,
    pub mode: LabelMode,
}

impl AddrInfo {
    pub fn get_val(&self, top: *const u8) -> Result<u64> {
        let disp = match self.mode {
            LabelMode::AddTop => self.jmp_addr.wrapping_add(top as u64),
            LabelMode::AsIs => self.jmp_addr,
            LabelMode::Abs => self.jmp_addr.wrapping_sub(top as u64),
        };
        if self.jmp_size == 4 {
            // Verify fits in i32
            let s = disp as i64;
            if !(i32::MIN as i64..=i32::MAX as i64).contains(&s) {
                // Allow unsigned 32-bit range too
                if disp > u32::MAX as u64 {
                    return Err(Error::OffsetIsTooBig);
                }
            }
        }
        Ok(disp)
    }
}

/// A growable code buffer with executable memory support.
pub struct CodeBuffer {
    ptr: *mut u8,
    size: usize,
    capacity: usize,
    mode: AllocMode,
    addr_info_list: Vec<AddrInfo>,
    calc_jmp_called: bool,
}

impl CodeBuffer {
    /// Create a new code buffer.
    ///
    /// - `max_size`: Initial capacity (default 4096).
    /// - `mode`: Allocation mode.
    pub fn new(max_size: usize, mode: AllocMode) -> Result<Self> {
        let cap = if max_size == 0 { DEFAULT_MAX_SIZE } else { max_size };
        let ptr = match mode {
            AllocMode::UserBuf => {
                return Err(Error::BadParameter);
            }
            AllocMode::Alloc | AllocMode::AutoGrow => {
                platform::alloc_exec_mem(cap)?
            }
        };
        Ok(Self {
            ptr,
            size: 0,
            capacity: cap,
            mode,
            addr_info_list: Vec::new(),
            calc_jmp_called: false,
        })
    }

    /// Create a code buffer backed by a user-provided buffer.
    ///
    /// # Safety
    /// The buffer must remain valid for the lifetime of this CodeBuffer.
    pub unsafe fn from_user_buf(buf: *mut u8, size: usize) -> Self {
        Self {
            ptr: buf,
            size: 0,
            capacity: size,
            mode: AllocMode::UserBuf,
            addr_info_list: Vec::new(),
            calc_jmp_called: false,
        }
    }

    /// Get allocation mode.
    pub fn alloc_mode(&self) -> AllocMode { self.mode }

    /// Get current code size (bytes written).
    pub fn size(&self) -> usize { self.size }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize { self.capacity }

    /// Get pointer to start of code buffer.
    pub fn top(&self) -> *const u8 { self.ptr }

    /// Get pointer to current write position.
    pub fn cur(&self) -> *const u8 {
        unsafe { self.ptr.add(self.size) }
    }

    /// Whether AutoGrow jump addresses have been calculated.
    pub fn is_calc_jmp_called(&self) -> bool { self.calc_jmp_called }

    /// Set the calc_jmp_called flag.
    pub fn set_calc_jmp_called(&mut self, v: bool) { self.calc_jmp_called = v; }

    /// Reset the code size to zero.
    pub fn reset_size(&mut self) { self.size = 0; }

    /// Set the code size directly.
    pub fn set_size(&mut self, size: usize) { self.size = size; }

    /// Emit a single byte.
    pub fn db(&mut self, byte: u8) -> Result<()> {
        if self.size >= self.capacity {
            self.grow()?;
        }
        unsafe { *self.ptr.add(self.size) = byte; }
        self.size += 1;
        Ok(())
    }

    /// Emit a slice of bytes.
    pub fn db_slice(&mut self, bytes: &[u8]) -> Result<()> {
        for &b in bytes {
            self.db(b)?;
        }
        Ok(())
    }

    /// Emit N bytes from a u64 (little-endian).
    pub fn db_n(&mut self, code: u64, n: usize) -> Result<()> {
        let bytes = code.to_le_bytes();
        for &b in &bytes[..n] {
            self.db(b)?;
        }
        Ok(())
    }

    /// Emit a 16-bit word (little-endian).
    pub fn dw(&mut self, v: u16) -> Result<()> {
        self.db_n(v as u64, 2)
    }

    /// Emit a 32-bit dword (little-endian).
    pub fn dd(&mut self, v: u32) -> Result<()> {
        self.db_n(v as u64, 4)
    }

    /// Emit a 64-bit qword (little-endian).
    pub fn dq(&mut self, v: u64) -> Result<()> {
        self.db_n(v, 8)
    }

    /// Rewrite bytes at a specific offset.
    pub fn rewrite(&mut self, offset: usize, val: u64, size: usize) {
        let bytes = val.to_le_bytes();
        for i in 0..size {
            unsafe { *self.ptr.add(offset + i) = bytes[i]; }
        }
    }

    /// Save an address info for later resolution (labels).
    pub fn save(&mut self, code_offset: usize, jmp_addr: u64, jmp_size: u8, mode: LabelMode) {
        self.addr_info_list.push(AddrInfo {
            code_offset,
            jmp_addr,
            jmp_size,
            mode,
        });
    }

    /// Calculate jump addresses for AutoGrow mode.
    pub fn calc_jmp_address(&mut self) -> Result<()> {
        if self.calc_jmp_called { return Ok(()); }
        for info in &self.addr_info_list {
            let val = info.get_val(self.ptr)?;
            let bytes = val.to_le_bytes();
            let n = info.jmp_size as usize;
            for i in 0..n {
                unsafe { *self.ptr.add(info.code_offset + i) = bytes[i]; }
            }
        }
        self.calc_jmp_called = true;
        Ok(())
    }

    /// Set memory protection to Read+Execute.
    pub fn protect_rx(&mut self) -> Result<()> {
        if self.mode == AllocMode::UserBuf { return Ok(()); }
        unsafe { platform::protect(self.ptr, self.capacity, ProtectMode::ReadExec) }
    }

    /// Set memory protection to Read+Write.
    pub fn protect_rw(&mut self) -> Result<()> {
        if self.mode == AllocMode::UserBuf { return Ok(()); }
        unsafe { platform::protect(self.ptr, self.capacity, ProtectMode::ReadWrite) }
    }

    /// Set memory protection to Read+Write+Execute.
    pub fn protect_rwe(&mut self) -> Result<()> {
        if self.mode == AllocMode::UserBuf { return Ok(()); }
        unsafe { platform::protect(self.ptr, self.capacity, ProtectMode::ReadWriteExec) }
    }

    /// Grow the buffer (for AutoGrow mode).
    fn grow(&mut self) -> Result<()> {
        if self.mode != AllocMode::AutoGrow {
            return Err(Error::CodeIsTooBig);
        }
        let new_cap = self.capacity * 2;
        let new_ptr = platform::alloc_exec_mem(new_cap)?;

        // Copy existing code
        unsafe {
            core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.size);
            platform::free_exec_mem(self.ptr, self.capacity)?;
        }

        // Update addr_info pointers: adjust code offsets if needed
        // (they're already relative offsets so no adjustment needed)

        self.ptr = new_ptr;
        self.capacity = new_cap;
        Ok(())
    }

    /// Get the code as a slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get a typed function pointer from the code buffer.
    ///
    /// # Safety
    /// The caller must ensure the code buffer contains valid machine code
    /// and that the type F matches the calling convention of the generated code.
    pub unsafe fn as_fn<F>(&self) -> F {
        assert_eq!(core::mem::size_of::<F>(), core::mem::size_of::<*const u8>());
        core::mem::transmute_copy(&self.ptr)
    }
}

impl Drop for CodeBuffer {
    fn drop(&mut self) {
        if self.mode != AllocMode::UserBuf && !self.ptr.is_null() {
            let _ = unsafe { platform::free_exec_mem(self.ptr, self.capacity) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_emit() {
        let mut buf = CodeBuffer::new(4096, AllocMode::Alloc).unwrap();
        buf.db(0x90).unwrap(); // nop
        buf.db(0xC3).unwrap(); // ret
        assert_eq!(buf.size(), 2);
        assert_eq!(buf.as_slice(), &[0x90, 0xC3]);
    }

    #[test]
    fn test_dw_dd_dq() {
        let mut buf = CodeBuffer::new(4096, AllocMode::Alloc).unwrap();
        buf.dw(0x1234).unwrap();
        assert_eq!(&buf.as_slice()[..2], &[0x34, 0x12]);

        buf.dd(0xDEADBEEF).unwrap();
        assert_eq!(&buf.as_slice()[2..6], &[0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_autogrow() {
        let mut buf = CodeBuffer::new(16, AllocMode::AutoGrow).unwrap();
        for _ in 0..32 {
            buf.db(0x90).unwrap();
        }
        assert_eq!(buf.size(), 32);
        assert!(buf.capacity() >= 32);
    }

    #[test]
    fn test_alloc_overflow() {
        let mut buf = CodeBuffer::new(4, AllocMode::Alloc).unwrap();
        buf.db(0x90).unwrap();
        buf.db(0x90).unwrap();
        buf.db(0x90).unwrap();
        buf.db(0x90).unwrap();
        assert!(buf.db(0x90).is_err());
    }
}
