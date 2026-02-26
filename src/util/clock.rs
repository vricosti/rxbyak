/// RDTSC-based cycle counter for benchmarking.
///
/// Port of xbyak_util.h `Clock` class.

/// High-resolution cycle counter using the x86 RDTSC instruction.
pub struct Clock {
    clock: u64,
    count: u32,
}

impl Clock {
    pub fn new() -> Self {
        Clock { clock: 0, count: 0 }
    }

    /// Read the current timestamp counter.
    #[inline]
    pub fn rdtsc() -> u64 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::x86_64::_rdtsc() as u64
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        { 0 }
    }

    /// Start a measurement interval.
    #[inline]
    pub fn begin(&mut self) {
        self.clock = self.clock.wrapping_sub(Self::rdtsc());
    }

    /// End a measurement interval.
    #[inline]
    pub fn end(&mut self) {
        self.clock = self.clock.wrapping_add(Self::rdtsc());
        self.count += 1;
    }

    /// Number of completed measurement intervals.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Total accumulated clock cycles.
    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// Average cycles per interval (0 if no intervals).
    pub fn average(&self) -> u64 {
        if self.count == 0 { 0 } else { self.clock / self.count as u64 }
    }

    /// Reset the counter.
    pub fn clear(&mut self) {
        self.clock = 0;
        self.count = 0;
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_basic() {
        let mut clk = Clock::new();
        assert_eq!(clk.count(), 0);
        assert_eq!(clk.clock(), 0);

        clk.begin();
        // Do some work
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(i);
        }
        let _ = x;
        clk.end();

        assert_eq!(clk.count(), 1);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        assert!(clk.clock() > 0);
    }

    #[test]
    fn test_clock_clear() {
        let mut clk = Clock::new();
        clk.begin();
        clk.end();
        clk.clear();
        assert_eq!(clk.count(), 0);
        assert_eq!(clk.clock(), 0);
    }
}
