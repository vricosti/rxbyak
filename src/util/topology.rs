/// CPU topology detection.
///
/// Port of xbyak_util.h `CpuTopology` class.
/// Reads cache hierarchy and core topology from the OS.
use std::collections::BTreeSet;

use crate::error::{Error, Result};

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
    pub fn set_str(&mut self, value: &str) -> Result<()> {
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

    pub fn get_str(&self) -> String {
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
        println!("{}", self.get_str());
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
    fn new() -> Self {
        Self {
            core_id: 0,
            core_type: CoreType::Standard,
            caches: Default::default(),
        }
    }

    /// Get cache information for a specific cache type.
    pub fn cache(&self, ct: CacheType) -> &CpuCache {
        &self.caches[ct.index()]
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
    /// Detect CPU topology. Returns `None` if detection fails.
    pub fn detect(is_hybrid: bool) -> Option<Self> {
        let mut topo = CpuTopology {
            logical_cpus: Vec::new(),
            physical_core_num: 0,
            line_size: 0,
            is_hybrid,
        };
        if init_topology(&mut topo) {
            Some(topo)
        } else {
            None
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

    // Get number of online CPUs
    let logical_cpu_num = match fs::read_to_string("/sys/devices/system/cpu/online") {
        Ok(s) => count_from_cpu_list(s.trim()),
        Err(_) => {
            // Fallback to sysconf
            #[cfg(unix)]
            unsafe {
                libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as u32
            }
            #[cfg(not(unix))]
            {
                return false;
            }
        }
    };

    if logical_cpu_num == 0 {
        return false;
    }

    topo.logical_cpus
        .resize_with(logical_cpu_num as usize, LogicalCpu::new);
    let mut max_physical_idx = 0u32;

    for cpu_idx in 0..logical_cpu_num {
        let base = format!("/sys/devices/system/cpu/cpu{}", cpu_idx);

        // Read core ID
        let core_id = read_int_from_file(&format!("{}/topology/core_id", base));
        topo.logical_cpus[cpu_idx as usize].core_id = core_id;
        max_physical_idx = max_physical_idx.max(core_id);

        // Read cache hierarchy
        for cache_idx in 0..8u32 {
            let cache_base = format!("{}/cache/index{}", base, cache_idx);

            // Determine cache type
            let cache_type = match fs::read_to_string(format!("{}/type", cache_base)) {
                Ok(s) => {
                    let s = s.trim();
                    let level = read_int_from_file(&format!("{}/level", cache_base));
                    match (s, level) {
                        ("Instruction", 1) => Some(CacheType::L1i),
                        ("Data", 1) => Some(CacheType::L1d),
                        ("Data", 2) | ("Unified", 2) => Some(CacheType::L2),
                        ("Data", 3) | ("Unified", 3) => Some(CacheType::L3),
                        _ => None,
                    }
                }
                Err(_) => break, // No more cache indices
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

    // Hybrid core types
    if topo.is_hybrid {
        if let Ok(s) = fs::read_to_string("/sys/devices/cpu_core/cpus") {
            for idx in parse_cpu_list(s.trim()).unwrap_or_default().iter() {
                if (idx as usize) < topo.logical_cpus.len() {
                    topo.logical_cpus[idx as usize].core_type = CoreType::Performance;
                }
            }
        }
        if let Ok(s) = fs::read_to_string("/sys/devices/cpu_atom/cpus") {
            for idx in parse_cpu_list(s.trim()).unwrap_or_default().iter() {
                if (idx as usize) < topo.logical_cpus.len() {
                    topo.logical_cpus[idx as usize].core_type = CoreType::Efficient;
                }
            }
        }
    }

    // Read cache line size
    topo.line_size =
        read_int_from_file("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");

    topo.physical_core_num = (max_physical_idx + 1) as usize;
    true
}

#[cfg(target_os = "windows")]
fn init_topology(topo: &mut CpuTopology) -> bool {
    // Windows topology detection is complex (GetLogicalProcessorInformationEx).
    // For now, provide basic info from the Cpu struct.
    let _ = topo;
    false
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn init_topology(topo: &mut CpuTopology) -> bool {
    let _ = topo;
    false
}

// --- Helper functions ---

fn read_int_from_file(path: &str) -> u32 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Parse a size string like "32K", "1M", "512" into bytes.
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
fn parse_cpu_list(s: &str) -> Result<CpuMask> {
    let mut mask = CpuMask::new();
    mask.set_str(s)?;
    Ok(mask)
}

/// Count total CPUs from an "online" cpu list string (e.g., "0-7" → 8).
fn count_from_cpu_list(s: &str) -> u32 {
    parse_cpu_list(s).map_or(0, |mask| mask.len() as u32)
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
        assert_eq!(mask.get_str(), "1,3,5-7");
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
                mask.set_str(invalid).unwrap_err(),
                Error::InvalidCpumaskIndex,
                "input={invalid}"
            );
        }
    }

    #[test]
    fn test_parse_cpu_list() {
        let mask = parse_cpu_list("0-3,5,7,10-12").unwrap();
        assert_eq!(mask.get_str(), "0-3,5,7,10-12");
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
    fn test_count_from_cpu_list() {
        assert_eq!(count_from_cpu_list("0-7"), 8);
        assert_eq!(count_from_cpu_list("0-3,8-11"), 8);
    }

    #[test]
    fn test_cache_type_index() {
        assert_eq!(CacheType::L1i.index(), 0);
        assert_eq!(CacheType::L1d.index(), 1);
        assert_eq!(CacheType::L2.index(), 2);
        assert_eq!(CacheType::L3.index(), 3);
    }
}
