#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

use crate::error::Result;

/// Memory protection mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtectMode {
    /// Read + Write
    ReadWrite,
    /// Read + Write + Execute
    ReadWriteExec,
    /// Read + Execute
    ReadExec,
}

/// Allocate executable memory.
///
/// Returns a pointer to the allocated memory block.
/// The memory is initially writable (RW).
pub fn alloc_exec_mem(size: usize) -> Result<*mut u8> {
    #[cfg(unix)]
    { unix::alloc_exec_mem(size) }
    #[cfg(windows)]
    { windows::alloc_exec_mem(size) }
    #[cfg(not(any(unix, windows)))]
    { Err(Error::CantAlloc) }
}

/// Free executable memory.
pub unsafe fn free_exec_mem(ptr: *mut u8, size: usize) -> Result<()> {
    #[cfg(unix)]
    { unix::free_exec_mem(ptr, size) }
    #[cfg(windows)]
    { windows::free_exec_mem(ptr, size) }
    #[cfg(not(any(unix, windows)))]
    { Err(Error::Munmap) }
}

/// Change memory protection.
pub unsafe fn protect(ptr: *mut u8, size: usize, mode: ProtectMode) -> Result<()> {
    #[cfg(unix)]
    { unix::protect(ptr, size, mode) }
    #[cfg(windows)]
    { windows::protect(ptr, size, mode) }
    #[cfg(not(any(unix, windows)))]
    { Err(Error::CantProtect) }
}

/// Get the system page size.
pub fn page_size() -> usize {
    #[cfg(unix)]
    { unix::page_size() }
    #[cfg(windows)]
    { windows::page_size() }
    #[cfg(not(any(unix, windows)))]
    { 4096 }
}
