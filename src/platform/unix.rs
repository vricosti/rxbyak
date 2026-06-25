use super::ProtectMode;
use crate::error::{Error, Result};

pub fn alloc_exec_mem(size: usize) -> Result<*mut u8> {
    // Match upstream Dynarmic's x64 allocator: pages are writable while code is
    // emitted and are made executable later through the protection helpers.
    // RWX MAP_JIT mappings fault on Apple Silicon when per-thread JIT write
    // protection is enabled.
    let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

    #[cfg(target_os = "macos")]
    {
        // Hardened macOS processes require MAP_JIT for writable executable
        // mappings. This is harmless for non-hardened runs and keeps the
        // x64/Rosetta path from failing before the backend can initialize.
        flags |= libc::MAP_JIT;
    }

    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(Error::CantAlloc);
    }
    Ok(ptr as *mut u8)
}

pub unsafe fn free_exec_mem(ptr: *mut u8, size: usize) -> Result<()> {
    let ret = libc::munmap(ptr as *mut libc::c_void, size);
    if ret != 0 {
        return Err(Error::Munmap);
    }
    Ok(())
}

pub unsafe fn protect(ptr: *mut u8, size: usize, mode: ProtectMode) -> Result<()> {
    let prot = match mode {
        ProtectMode::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        ProtectMode::ReadWriteExec => libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        ProtectMode::ReadExec => libc::PROT_READ | libc::PROT_EXEC,
    };
    // Align to page boundary
    let page = page_size();
    let addr = (ptr as usize) & !(page - 1);
    let end = ((ptr as usize) + size + page - 1) & !(page - 1);
    let ret = libc::mprotect(addr as *mut libc::c_void, end - addr, prot);
    if ret != 0 {
        return Err(Error::CantProtect);
    }
    Ok(())
}

pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
