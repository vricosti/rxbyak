/// CPU topology detection.
///
/// Port of xbyak_util.h `CpuTopology` class.
/// Reads cache hierarchy and core topology from the OS.
use std::collections::BTreeSet;

use crate::error::{Error, Result};
use crate::util::cpu::{Cpu, HYBRID};

const CPU_MASK_LIMIT: u32 = 1 << 10;

/// Ordered set of logical CPU indices.
///
/// This is the Rust counterpart of Xbyak's non-compact `CpuMask`
/// implementation. It preserves the same ordering, range and parsing
/// contracts without exposing the container as the public API.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CpuMask {
    indices: BTreeSet<u32>,
}

impl CpuMask {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Append a monotonically increasing CPU index.
    pub fn append(&mut self, index: u32) -> Result<()> {
        if index >= CPU_MASK_LIMIT
            || self
                .indices
                .last()
                .is_some_and(|previous| *previous >= index)
        {
            return Err(Error::InvalidCpumaskIndex);
        }
        self.indices.insert(index);
        Ok(())
    }

    /// Append every CPU index in the inclusive range `[first, last]`.
    pub fn append_range(&mut self, first: u32, last: u32) -> Result<()> {
        if first > last || last >= CPU_MASK_LIMIT {
            return Err(Error::InvalidCpumaskIndex);
        }
        for index in first..=last {
            self.append(index)?;
        }
        Ok(())
    }

    /// Append indices parsed from `(integer|range)[,(integer|range)]*`.
    pub fn append_str(&mut self, value: &str) -> Result<()> {
        if value.is_empty() {
            return Ok(());
        }
        for item in value.split(',') {
            if item.is_empty() {
                return Err(Error::InvalidCpumaskIndex);
            }
            if let Some((first, last)) = item.split_once('-') {
                if first.is_empty() || last.is_empty() || last.contains('-') {
                    return Err(Error::InvalidCpumaskIndex);
                }
                let first = first
                    .parse::<u32>()
                    .map_err(|_| Error::InvalidCpumaskIndex)?;
                let last = last
                    .parse::<u32>()
                    .map_err(|_| Error::InvalidCpumaskIndex)?;
                self.append_range(first, last)?;
            } else {
                let index = item
                    .parse::<u32>()
                    .map_err(|_| Error::InvalidCpumaskIndex)?;
                self.append(index)?;
            }
        }
        Ok(())
    }

    pub fn range_string(&self) -> String {
        let mut result = String::new();
        let mut values = self.indices.iter().copied().peekable();
        while let Some(first) = values.next() {
            let mut last = first;
            while values.peek().is_some_and(|next| *next == last + 1) {
                last = values.next().unwrap();
            }
            if !result.is_empty() {
                result.push(',');
            }
            result.push_str(&first.to_string());
            if last != first {
                result.push('-');
                result.push_str(&last.to_string());
            }
        }
        result
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        self.indices.iter().nth(index).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.indices.iter().copied()
    }

    pub fn put(&self, label: Option<&str>) {
        if let Some(label) = label {
            print!("{label}: ");
        }
        println!("{}", self.range_string());
    }
}

impl<'a> IntoIterator for &'a CpuMask {
    type Item = u32;
    type IntoIter = std::iter::Copied<std::collections::btree_set::Iter<'a, u32>>;

    fn into_iter(self) -> Self::IntoIter {
        self.indices.iter().copied()
    }
}

/// Core type for hybrid architectures (Intel Alder Lake+).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreType {
    Unknown,
    Performance, // P-core
    Efficient,   // E-core
    Standard,    // Non-hybrid
}

impl CoreType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Performance => "P-core",
            Self::Efficient => "E-core",
            Self::Standard => "Standard",
            Self::Unknown => "Unknown",
        }
    }
}

/// Cache type identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheType {
    L1i,
    L1d,
    L2,
    L3,
}

impl CacheType {
    pub const ALL: [CacheType; 4] = [CacheType::L1i, CacheType::L1d, CacheType::L2, CacheType::L3];

    fn index(self) -> usize {
        match self {
            CacheType::L1i => 0,
            CacheType::L1d => 1,
            CacheType::L2 => 2,
            CacheType::L3 => 3,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L1i => "L1i",
            Self::L1d => "L1d",
            Self::L2 => "L2",
            Self::L3 => "L3",
        }
    }
}

/// Information about a single cache level.
#[derive(Clone, Debug, Default)]
pub struct CpuCache {
    /// Cache size in bytes.
    pub size: u32,
    /// Number of ways of associativity.
    pub associativity: u32,
    /// Set of logical CPU indices sharing this cache.
    pub shared_cpu_indices: CpuMask,
}

impl CpuCache {
    /// Whether this cache is shared across multiple logical CPUs.
    pub fn is_shared(&self) -> bool {
        self.shared_cpu_indices.len() > 1
    }

    /// Number of logical CPUs sharing this cache.
    pub fn shared_cpu_count(&self) -> usize {
        self.shared_cpu_indices.len()
    }

    pub fn put(&self, label: Option<&str>) {
        if let Some(label) = label {
            print!("{label}: ");
        }
        print!(
            "{} KiB, assoc. {}, shared ",
            self.size / 1024,
            self.associativity
        );
        self.shared_cpu_indices.put(None);
    }
}

/// Information about a single logical CPU.
#[derive(Clone, Debug)]
pub struct LogicalCpu {
    /// Physical core ID.
    pub core_id: u32,
    /// Core type (for hybrid architectures).
    pub core_type: CoreType,
    /// Cache information (indexed by CacheType).
    caches: [CpuCache; 4],
}

impl LogicalCpu {
    pub fn new() -> Self {
        Self {
            core_id: 0,
            core_type: CoreType::Unknown,
            caches: Default::default(),
        }
    }

    /// Get cache information for a specific cache type.
    pub fn cache(&self, ct: CacheType) -> &CpuCache {
        &self.caches[ct.index()]
    }

    pub fn siblings(&self) -> &CpuMask {
        &self.caches[CacheType::L1i.index()].shared_cpu_indices
    }

    pub fn put(&self, label: Option<&str>) {
        if let Some(label) = label {
            print!("{label}: ");
        }
        println!("coreId {}, type {}", self.core_id, self.core_type.as_str());
        for cache_type in CacheType::ALL {
            self.cache(cache_type).put(Some(cache_type.as_str()));
        }
    }
}

impl Default for LogicalCpu {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU topology information.
pub struct CpuTopology {
    logical_cpus: Vec<LogicalCpu>,
    physical_core_num: usize,
    line_size: u32,
    is_hybrid: bool,
}

impl CpuTopology {
    pub fn new(cpu: &Cpu) -> Result<Self> {
        let mut topo = CpuTopology {
            logical_cpus: Vec::new(),
            physical_core_num: 0,
            line_size: 0,
            is_hybrid: cpu.has(HYBRID),
        };
        if init_topology(&mut topo) {
            Ok(topo)
        } else {
            Err(Error::CantInitCpuTopology)
        }
    }

    /// Number of logical CPUs.
    pub fn logical_cpu_count(&self) -> usize {
        self.logical_cpus.len()
    }

    /// Number of physical cores.
    pub fn physical_core_count(&self) -> usize {
        self.physical_core_num
    }

    /// Cache line size in bytes.
    pub fn line_size(&self) -> u32 {
        self.line_size
    }

    /// Whether this is a hybrid system.
    pub fn is_hybrid(&self) -> bool {
        self.is_hybrid
    }

    /// Get logical CPU information.
    pub fn logical_cpu(&self, idx: usize) -> Option<&LogicalCpu> {
        self.logical_cpus.get(idx)
    }

    /// Get cache information for a specific logical CPU and cache type.
    pub fn cache(&self, cpu_idx: usize, ct: CacheType) -> Option<&CpuCache> {
        self.logical_cpus.get(cpu_idx).map(|lc| lc.cache(ct))
    }
}

// --- Platform-specific implementation ---

#[cfg(target_os = "linux")]
fn init_topology(topo: &mut CpuTopology) -> bool {
    use std::fs;

    let logical_cpu_num = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if logical_cpu_num <= 0 || logical_cpu_num as u64 >= u64::from(CPU_MASK_LIMIT) {
        return false;
    }
    let logical_cpu_num = logical_cpu_num as u32;

    topo.logical_cpus
        .resize_with(logical_cpu_num as usize, LogicalCpu::new);
    let mut max_physical_idx = 0u32;

    for cpu_idx in 0..logical_cpu_num {
        let base = format!("/sys/devices/system/cpu/cpu{}", cpu_idx);

        // Read core ID
        let core_id = read_int_from_file(&format!("{}/topology/core_id", base));
        topo.logical_cpus[cpu_idx as usize].core_id = core_id;
        topo.logical_cpus[cpu_idx as usize].core_type = CoreType::Standard;
        max_physical_idx = max_physical_idx.max(core_id);

        // Read cache hierarchy
        for cache_idx in 0..CacheType::ALL.len() as u32 {
            let cache_base = format!("{}/cache/index{}", base, cache_idx);

            // Determine cache type
            let cache_type = match fs::read_to_string(format!("{}/type", cache_base)) {
                Ok(s) => {
                    let s = s.trim();
                    if s.starts_with("Instruction") {
                        Some(CacheType::L1i)
                    } else {
                        let level = read_int_from_file(&format!("{}/level", cache_base));
                        match (s, level) {
                            ("Data", 1) => Some(CacheType::L1d),
                            ("Data", 2) | ("Unified", 2) => Some(CacheType::L2),
                            ("Data", 3) | ("Unified", 3) => Some(CacheType::L3),
                            _ => None,
                        }
                    }
                }
                Err(_) => continue,
            };

            let ct = match cache_type {
                Some(ct) => ct,
                None => continue,
            };

            let cache = &mut topo.logical_cpus[cpu_idx as usize].caches[ct.index()];

            // Read cache size
            if let Ok(s) = fs::read_to_string(format!("{}/size", cache_base)) {
                cache.size = parse_size(s.trim());
            }

            // Read associativity
            cache.associativity =
                read_int_from_file(&format!("{}/ways_of_associativity", cache_base));

            // Read shared CPU list
            if let Ok(s) = fs::read_to_string(format!("{}/shared_cpu_list", cache_base)) {
                if let Ok(mask) = parse_cpu_list(s.trim()) {
                    cache.shared_cpu_indices = mask;
                }
            }
        }
    }

    // Assign core types for hybrid architectures.
    if topo.is_hybrid {
        let p_cores = read_cpu_mask_from_file("/sys/devices/cpu_core/cpus");
        if let Some(mask) = &p_cores {
            for idx in mask {
                if (idx as usize) < topo.logical_cpus.len() {
                    topo.logical_cpus[idx as usize].core_type = CoreType::Performance;
                }
            }
        }
        let e_cores = read_cpu_mask_from_file("/sys/devices/cpu_atom/cpus");
        if let Some(mask) = &e_cores {
            for idx in mask {
                if (idx as usize) < topo.logical_cpus.len() {
                    topo.logical_cpus[idx as usize].core_type = CoreType::Efficient;
                }
            }
        }

        if p_cores.is_none() || e_cores.is_none() {
            update_core_types_with_affinity(topo);
        }
    }

    // Read cache line size
    topo.line_size =
        read_int_from_file("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");

    topo.physical_core_num = (max_physical_idx + 1) as usize;
    true
}

pub fn core_type() -> CoreType {
    let eax = Cpu::cpuid_with_subleaf(0x1a, 0)[0];
    match (eax >> 24) & 0xff {
        0x40 => CoreType::Performance,
        0x20 => CoreType::Efficient,
        _ => CoreType::Standard,
    }
}

#[cfg(target_os = "linux")]
fn read_cpu_mask_from_file(path: &str) -> Option<CpuMask> {
    let value = std::fs::read_to_string(path).ok()?;
    parse_cpu_list(value.trim()).ok()
}

#[cfg(target_os = "linux")]
fn core_type_with_affinity(cpu: u32) -> CoreType {
    unsafe {
        let mut mask: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut mask);
        libc::CPU_SET(cpu as usize, &mut mask);
        if libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mask as *const libc::cpu_set_t,
        ) != 0
        {
            return CoreType::Standard;
        }
    }
    core_type()
}

#[cfg(target_os = "linux")]
fn update_core_types_with_affinity(topo: &mut CpuTopology) {
    unsafe {
        let mut original: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut original);
        if libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut original as *mut libc::cpu_set_t,
        ) != 0
        {
            return;
        }
        for (index, cpu) in topo.logical_cpus.iter_mut().enumerate() {
            cpu.core_type = core_type_with_affinity(index as u32);
        }
        let _ = libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &original as *const libc::cpu_set_t,
        );
    }
}

#[cfg(target_os = "windows")]
fn init_topology(topo: &mut CpuTopology) -> bool {
    use windows_sys::Win32::System::SystemInformation::RelationCache;

    let Some((group_acc, logical_cpu_num)) = windows_group_accumulators() else {
        return false;
    };
    if logical_cpu_num == 0 || logical_cpu_num >= CPU_MASK_LIMIT {
        return false;
    }
    topo.logical_cpus
        .resize_with(logical_cpu_num as usize, LogicalCpu::new);
    topo.physical_core_num =
        windows_populate_cores(&mut topo.logical_cpus, topo.is_hybrid, &group_acc);
    if topo.physical_core_num == 0 {
        return false;
    }

    let Some(buffer) = windows_query_relationship(RelationCache) else {
        return false;
    };
    for entry in buffer.entries() {
        let Ok(entry) = entry else {
            return false;
        };
        if entry.relationship() != RelationCache {
            continue;
        }
        let Some(cache) = entry.cache() else {
            return false;
        };
        let cache_type = match cache.Level {
            1 if cache.Type == windows_sys::Win32::System::SystemInformation::CacheInstruction => {
                Some(CacheType::L1i)
            }
            1 if cache.Type == windows_sys::Win32::System::SystemInformation::CacheData => {
                Some(CacheType::L1d)
            }
            2 => Some(CacheType::L2),
            3 => Some(CacheType::L3),
            _ => None,
        };
        let Some(cache_type) = cache_type else {
            continue;
        };
        if !entry.contains_flexible_array::<
            windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY,
        >(
            std::mem::offset_of!(
                windows_sys::Win32::System::SystemInformation::CACHE_RELATIONSHIP,
                Anonymous
            ),
            WINDOWS_CACHE_GROUP_COUNT,
        ) {
            return false;
        }
        let Some(mask) = windows_cache_mask(cache, &group_acc) else {
            return false;
        };
        for index in &mask {
            let Some(logical_cpu) = topo.logical_cpus.get_mut(index as usize) else {
                return false;
            };
            let target = &mut logical_cpu.caches[cache_type.index()];
            target.size = cache.CacheSize;
            if topo.line_size == 0 {
                topo.line_size = u32::from(cache.LineSize);
            }
            target.associativity = u32::from(cache.Associativity);
            target.shared_cpu_indices = mask.clone();
        }
    }
    true
}

#[cfg(target_os = "windows")]
struct WindowsTopologyBuffer {
    storage: Vec<usize>,
    byte_len: usize,
}

#[cfg(target_os = "windows")]
impl WindowsTopologyBuffer {
    fn entries(&self) -> impl Iterator<Item = std::result::Result<WindowsTopologyEntry<'_>, ()>> {
        WindowsTopologyEntries {
            buffer: self,
            offset: 0,
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Clone, Copy)]
struct WindowsTopologyEntry<'a> {
    bytes: &'a [u8],
    relationship: windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
}

#[cfg(target_os = "windows")]
impl<'a> WindowsTopologyEntry<'a> {
    const HEADER_SIZE: usize = std::mem::size_of::<
        windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
    >() + std::mem::size_of::<u32>();

    fn relationship(
        self,
    ) -> windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP {
        self.relationship
    }

    fn payload<T>(self) -> Option<*const T> {
        let required = Self::HEADER_SIZE.checked_add(std::mem::size_of::<T>())?;
        if self.bytes.len() < required {
            return None;
        }
        let ptr = unsafe { self.bytes.as_ptr().add(Self::HEADER_SIZE).cast::<T>() };
        if ptr.align_offset(std::mem::align_of::<T>()) != 0 {
            return None;
        }
        Some(ptr)
    }

    fn group(
        self,
    ) -> Option<&'a windows_sys::Win32::System::SystemInformation::GROUP_RELATIONSHIP> {
        let ptr =
            self.payload::<windows_sys::Win32::System::SystemInformation::GROUP_RELATIONSHIP>()?;
        Some(unsafe { &*ptr })
    }

    fn processor(
        self,
    ) -> Option<&'a windows_sys::Win32::System::SystemInformation::PROCESSOR_RELATIONSHIP> {
        let ptr = self
            .payload::<windows_sys::Win32::System::SystemInformation::PROCESSOR_RELATIONSHIP>()?;
        Some(unsafe { &*ptr })
    }

    fn cache(
        self,
    ) -> Option<&'a windows_sys::Win32::System::SystemInformation::CACHE_RELATIONSHIP> {
        let ptr =
            self.payload::<windows_sys::Win32::System::SystemInformation::CACHE_RELATIONSHIP>()?;
        Some(unsafe { &*ptr })
    }

    fn contains_flexible_array<U>(self, offset: usize, count: usize) -> bool {
        let Some(elements_size) = std::mem::size_of::<U>().checked_mul(count) else {
            return false;
        };
        let Some(payload_size) = offset.checked_add(elements_size) else {
            return false;
        };
        let Some(required) = Self::HEADER_SIZE.checked_add(payload_size) else {
            return false;
        };
        self.bytes.len() >= required
    }
}

#[cfg(target_os = "windows")]
struct WindowsTopologyEntries<'a> {
    buffer: &'a WindowsTopologyBuffer,
    offset: usize,
}

#[cfg(target_os = "windows")]
impl<'a> Iterator for WindowsTopologyEntries<'a> {
    type Item = std::result::Result<WindowsTopologyEntry<'a>, ()>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.buffer.byte_len {
            return None;
        }
        let Some(header_end) = self.offset.checked_add(WindowsTopologyEntry::HEADER_SIZE) else {
            self.offset = self.buffer.byte_len;
            return Some(Err(()));
        };
        if header_end > self.buffer.byte_len {
            self.offset = self.buffer.byte_len;
            return Some(Err(()));
        }
        let start = unsafe { (self.buffer.storage.as_ptr() as *const u8).add(self.offset) };
        let relationship = unsafe {
            start
                .cast::<windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP>()
                .read_unaligned()
        };
        let size = unsafe {
            start
                .add(std::mem::size_of::<
                    windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
                >())
                .cast::<u32>()
                .read_unaligned()
        } as usize;
        let Some(end) = self.offset.checked_add(size) else {
            self.offset = self.buffer.byte_len;
            return Some(Err(()));
        };
        if size < WindowsTopologyEntry::HEADER_SIZE || end > self.buffer.byte_len {
            self.offset = self.buffer.byte_len;
            return Some(Err(()));
        }
        let bytes = unsafe { std::slice::from_raw_parts(start, size) };
        self.offset += size;
        Some(Ok(WindowsTopologyEntry {
            bytes,
            relationship,
        }))
    }
}

#[cfg(target_os = "windows")]
fn windows_query_relationship(
    relationship: windows_sys::Win32::System::SystemInformation::LOGICAL_PROCESSOR_RELATIONSHIP,
) -> Option<WindowsTopologyBuffer> {
    use windows_sys::Win32::System::SystemInformation::GetLogicalProcessorInformationEx;

    let mut byte_len = 0u32;
    unsafe {
        GetLogicalProcessorInformationEx(relationship, std::ptr::null_mut(), &mut byte_len);
    }
    if byte_len == 0 {
        return None;
    }
    let word_count = (byte_len as usize).div_ceil(std::mem::size_of::<usize>());
    let mut storage = vec![0usize; word_count];
    let mut returned_len = byte_len;
    let result = unsafe {
        GetLogicalProcessorInformationEx(
            relationship,
            storage.as_mut_ptr()
                as *mut windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut returned_len,
        )
    };
    if result == 0 || returned_len == 0 || returned_len > byte_len {
        return None;
    }
    Some(WindowsTopologyBuffer {
        storage,
        byte_len: returned_len as usize,
    })
}

#[cfg(target_os = "windows")]
fn windows_group_accumulators() -> Option<(Vec<u32>, u32)> {
    use windows_sys::Win32::System::SystemInformation::{RelationGroup, PROCESSOR_GROUP_INFO};

    let buffer = windows_query_relationship(RelationGroup)?;
    let mut group_entry = None;
    for entry in buffer.entries() {
        let entry = entry.ok()?;
        if entry.relationship() == RelationGroup {
            group_entry = Some(entry);
            break;
        }
    }
    let entry = group_entry?;
    let group = entry.group()?;
    if group.ActiveGroupCount == 0 {
        return None;
    }
    if !entry.contains_flexible_array::<PROCESSOR_GROUP_INFO>(
        std::mem::offset_of!(
            windows_sys::Win32::System::SystemInformation::GROUP_RELATIONSHIP,
            GroupInfo
        ),
        group.ActiveGroupCount as usize,
    ) {
        return None;
    }
    let group_info = std::ptr::addr_of!(group.GroupInfo) as *const PROCESSOR_GROUP_INFO;
    let mut accumulators = Vec::with_capacity(group.ActiveGroupCount as usize);
    let mut total = 0u32;
    for index in 0..group.ActiveGroupCount as usize {
        accumulators.push(total);
        total = total.checked_add(u32::from(unsafe {
            (*group_info.add(index)).ActiveProcessorCount
        }))?;
    }
    Some((accumulators, total))
}

#[cfg(target_os = "windows")]
fn windows_populate_cores(
    logical_cpus: &mut [LogicalCpu],
    is_hybrid: bool,
    group_acc: &[u32],
) -> usize {
    use windows_sys::Win32::System::SystemInformation::{RelationProcessorCore, GROUP_AFFINITY};

    let Some(buffer) = windows_query_relationship(RelationProcessorCore) else {
        return 0;
    };
    let mut core_index = 0u32;
    for entry in buffer.entries() {
        let Ok(entry) = entry else {
            return 0;
        };
        if entry.relationship() != RelationProcessorCore {
            continue;
        }
        let Some(core) = entry.processor() else {
            return 0;
        };
        if !entry.contains_flexible_array::<GROUP_AFFINITY>(
            std::mem::offset_of!(
                windows_sys::Win32::System::SystemInformation::PROCESSOR_RELATIONSHIP,
                GroupMask
            ),
            core.GroupCount as usize,
        ) {
            return 0;
        }
        let logical = LogicalCpu {
            core_id: core_index,
            core_type: if !is_hybrid {
                CoreType::Standard
            } else if core.EfficiencyClass > 0 {
                CoreType::Performance
            } else {
                CoreType::Efficient
            },
            caches: Default::default(),
        };
        core_index += 1;
        let masks = std::ptr::addr_of!(core.GroupMask) as *const GROUP_AFFINITY;
        for mask_index in 0..core.GroupCount as usize {
            let mask = unsafe { *masks.add(mask_index) };
            let Some(base) = group_acc.get(mask.Group as usize).copied() else {
                return 0;
            };
            for bit in 0..usize::BITS {
                if mask.Mask & (1usize << bit) == 0 {
                    continue;
                }
                let index = base + bit;
                let Some(target) = logical_cpus.get_mut(index as usize) else {
                    return 0;
                };
                *target = logical.clone();
            }
        }
    }
    core_index as usize
}

#[cfg(target_os = "windows")]
fn windows_cache_mask(
    cache: &windows_sys::Win32::System::SystemInformation::CACHE_RELATIONSHIP,
    group_acc: &[u32],
) -> Option<CpuMask> {
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;

    let masks = std::ptr::addr_of!(cache.Anonymous.GroupMasks) as *const GROUP_AFFINITY;
    let mut result = CpuMask::new();
    for index in 0..WINDOWS_CACHE_GROUP_COUNT {
        let group_mask = unsafe { *masks.add(index) };
        let base = group_acc.get(group_mask.Group as usize).copied()?;
        for bit in 0..usize::BITS {
            if group_mask.Mask & (1usize << bit) != 0 {
                result.append(base + bit).ok()?;
            }
        }
    }
    Some(result)
}

// Xbyak uses the legacy single GroupMask unless NTDDI_VERSION explicitly
// selects Windows 10 20H1 or newer. Rust does not receive that C SDK macro, so
// preserve Xbyak's conservative default.
#[cfg(target_os = "windows")]
const WINDOWS_CACHE_GROUP_COUNT: usize = 1;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn init_topology(topo: &mut CpuTopology) -> bool {
    let _ = topo;
    false
}

// --- Helper functions ---

#[cfg(target_os = "linux")]
fn read_int_from_file(path: &str) -> u32 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Parse a size string like "32K", "1M", "512" into bytes.
#[cfg(any(target_os = "linux", test))]
fn parse_size(s: &str) -> u32 {
    let s = s.trim();
    if s.is_empty() {
        return 0;
    }
    let (num_str, suffix) = if s.ends_with('K') || s.ends_with('k') {
        (&s[..s.len() - 1], 1024u32)
    } else if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1024 * 1024)
    } else {
        (s, 1)
    };
    num_str.trim().parse::<u32>().unwrap_or(0) * suffix
}

/// Parse a CPU list string like "0-3,5,7,10-12" into a set of indices.
#[cfg(any(target_os = "linux", test))]
fn parse_cpu_list(s: &str) -> Result<CpuMask> {
    let mut mask = CpuMask::new();
    mask.append_str(s)?;
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_mask_upstream_contract() {
        let mut mask = CpuMask::new();
        assert!(mask.is_empty());
        mask.append(1).unwrap();
        mask.append(3).unwrap();
        mask.append_range(5, 7).unwrap();
        assert_eq!(mask.range_string(), "1,3,5-7");
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.get(3), Some(6));
        assert_eq!(mask.get(5), None);

        assert_eq!(mask.append(7).unwrap_err(), Error::InvalidCpumaskIndex);
        assert_eq!(mask.append(1024).unwrap_err(), Error::InvalidCpumaskIndex);
        assert_eq!(
            mask.append_range(9, 8).unwrap_err(),
            Error::InvalidCpumaskIndex
        );
        mask.clear();
        assert!(mask.is_empty());
    }

    #[test]
    fn test_cpu_mask_strict_parser() {
        for invalid in ["1,", ",1", "2-1", "1--2", "a", "1,1", "1024"] {
            let mut mask = CpuMask::new();
            assert_eq!(
                mask.append_str(invalid).unwrap_err(),
                Error::InvalidCpumaskIndex,
                "input={invalid}"
            );
        }
    }

    #[test]
    fn test_parse_cpu_list() {
        let mask = parse_cpu_list("0-3,5,7,10-12").unwrap();
        assert_eq!(mask.range_string(), "0-3,5,7,10-12");
        assert_eq!(
            mask.iter().collect::<Vec<_>>(),
            [0, 1, 2, 3, 5, 7, 10, 11, 12]
        );
    }

    #[test]
    fn test_parse_cpu_list_single() {
        let mask = parse_cpu_list("0").unwrap();
        assert_eq!(mask.len(), 1);
        assert_eq!(mask.get(0), Some(0));
    }

    #[test]
    fn test_parse_size() {
        assert_eq!(parse_size("32K"), 32768);
        assert_eq!(parse_size("1M"), 1048576);
        assert_eq!(parse_size("512"), 512);
        assert_eq!(parse_size(""), 0);
    }

    #[test]
    fn test_cache_type_index() {
        assert_eq!(CacheType::L1i.index(), 0);
        assert_eq!(CacheType::L1d.index(), 1);
        assert_eq!(CacheType::L2.index(), 2);
        assert_eq!(CacheType::L3.index(), 3);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_linux_topology_contract() {
        let topology = CpuTopology::new(&Cpu::new()).unwrap();
        assert!(topology.logical_cpu_count() > 0);
        assert!(topology.logical_cpu_count() < CPU_MASK_LIMIT as usize);
        assert!(topology.physical_core_count() > 0);
        assert!(matches!(
            core_type(),
            CoreType::Performance | CoreType::Efficient | CoreType::Standard
        ));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_windows_topology_contract() {
        let topology = CpuTopology::new(&Cpu::new()).unwrap();
        assert!(topology.logical_cpu_count() > 0);
        assert!(topology.logical_cpu_count() < CPU_MASK_LIMIT as usize);
        assert!(topology.physical_core_count() > 0);
    }
}
