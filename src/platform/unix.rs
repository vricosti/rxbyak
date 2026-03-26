use crate::error::{Error, Result};
use super::ProtectMode;

pub fn alloc_exec_mem(size: usize) -> Result<*mut u8> {
    // Allocate as RWX matching upstream dynarmic behavior when
    // DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT is OFF (the default).
    // This eliminates all mprotect toggles during JIT compilation.
    let ptr = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
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
