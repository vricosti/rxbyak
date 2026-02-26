use crate::error::{Error, Result};
use super::ProtectMode;

pub fn alloc_exec_mem(size: usize) -> Result<*mut u8> {
    let ptr = unsafe {
        windows_sys::Win32::System::Memory::VirtualAlloc(
            core::ptr::null(),
            size,
            windows_sys::Win32::System::Memory::MEM_COMMIT
                | windows_sys::Win32::System::Memory::MEM_RESERVE,
            windows_sys::Win32::System::Memory::PAGE_READWRITE,
        )
    };
    if ptr.is_null() {
        return Err(Error::CantAlloc);
    }
    Ok(ptr as *mut u8)
}

pub unsafe fn free_exec_mem(ptr: *mut u8, _size: usize) -> Result<()> {
    let ret = windows_sys::Win32::System::Memory::VirtualFree(
        ptr as *mut core::ffi::c_void,
        0,
        windows_sys::Win32::System::Memory::MEM_RELEASE,
    );
    if ret == 0 {
        return Err(Error::Munmap);
    }
    Ok(())
}

pub unsafe fn protect(ptr: *mut u8, size: usize, mode: ProtectMode) -> Result<()> {
    let prot = match mode {
        ProtectMode::ReadWrite => windows_sys::Win32::System::Memory::PAGE_READWRITE,
        ProtectMode::ReadWriteExec => windows_sys::Win32::System::Memory::PAGE_EXECUTE_READWRITE,
        ProtectMode::ReadExec => windows_sys::Win32::System::Memory::PAGE_EXECUTE_READ,
    };
    let mut old_prot = 0u32;
    let ret = windows_sys::Win32::System::Memory::VirtualProtect(
        ptr as *mut core::ffi::c_void,
        size,
        prot,
        &mut old_prot,
    );
    if ret == 0 {
        return Err(Error::CantProtect);
    }
    Ok(())
}

pub fn page_size() -> usize {
    let mut si = unsafe { core::mem::zeroed::<windows_sys::Win32::System::SystemInformation::SYSTEM_INFO>() };
    unsafe {
        windows_sys::Win32::System::SystemInformation::GetSystemInfo(&mut si);
    }
    si.dwPageSize as usize
}
