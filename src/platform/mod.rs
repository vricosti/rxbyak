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
    {
        unix::alloc_exec_mem(size)
    }
    #[cfg(windows)]
    {
        windows::alloc_exec_mem(size)
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(Error::CantAlloc)
    }
}

/// Commit a previously reserved portion of executable memory.
///
/// Windows reserves large fixed-address JIT buffers up front and commits them
/// progressively. Unix mappings already provide lazy physical commitment, so
/// this is a no-op there.
///
/// # Safety
///
/// `ptr..ptr + size` must be an uncommitted subrange of a live allocation
/// returned by [`alloc_exec_mem`].
pub unsafe fn commit_exec_mem(ptr: *mut u8, size: usize, mode: ProtectMode) -> Result<()> {
    #[cfg(unix)]
    {
        unix::commit_exec_mem(ptr, size, mode)
    }
    #[cfg(windows)]
    {
        windows::commit_exec_mem(ptr, size, mode)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (ptr, size, mode);
        Err(crate::error::Error::CantAlloc)
    }
}

/// Free executable memory.
///
/// # Safety
///
/// `ptr` and `size` must describe a live allocation previously returned by
/// [`alloc_exec_mem`], and the allocation must not be used after this call.
pub unsafe fn free_exec_mem(ptr: *mut u8, size: usize) -> Result<()> {
    #[cfg(unix)]
    {
        unix::free_exec_mem(ptr, size)
    }
    #[cfg(windows)]
    {
        windows::free_exec_mem(ptr, size)
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(Error::Munmap)
    }
}

/// Change memory protection.
///
/// # Safety
///
/// `ptr` and `size` must describe a live mapped allocation for the full range.
pub unsafe fn protect(ptr: *mut u8, size: usize, mode: ProtectMode) -> Result<()> {
    #[cfg(unix)]
    {
        unix::protect(ptr, size, mode)
    }
    #[cfg(windows)]
    {
        windows::protect(ptr, size, mode)
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(Error::CantProtect)
    }
}

/// Get the system page size.
pub fn page_size() -> usize {
    #[cfg(unix)]
    {
        unix::page_size()
    }
    #[cfg(windows)]
    {
        windows::page_size()
    }
    #[cfg(not(any(unix, windows)))]
    {
        4096
    }
}
