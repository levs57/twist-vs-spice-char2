//! A seeded synthetic trace with program-like locality, and reuse-distance
//! histograms for any trace.
//!
//! Why synthetic: none of the included guest programs can produce `2^20`
//! memory operations without repeating. Their input ABI holds `2^12` words,
//! which caps ERC20 at about 55K operations (1,000 transfers over 1,024
//! accounts) and binary search at about 30K (255 queries). Pointer chase
//! reaches about 240K operations before the harness's `2^20`-cycle limit,
//! and 99.99% of its operations are reads.
//!
//! The model mixes three access streams, each a standard ingredient of
//! memory-trace models:
//!
//! - **Stack (45%).** The top 1/16 of memory is a downward-growing stack,
//!   of which frames may use half. A call pushes a frame of 2–16 words
//!   (probability 1/48 per stack access, skipped if the frame would not
//!   fit). Otherwise a return pops one with probability 1/48 when more than
//!   one frame exists, so 47/48 · 1/48 per access in all. An access
//!   touches the current frame at a geometric offset (p = 1/4), so a few slots
//!   are hot. 65% reads: locals are read more often than spilled.
//! - **Sequential scans (30%).** Half of memory holds arrays. Up to four
//!   concurrent loops each scan `2^U(4..10)` words from a random base with
//!   stride 1 (90%) or 8 (10%, struct fields), and a finished loop is
//!   replaced. A new loop is read-only (60%), write-only (20%,
//!   initialisation) or an in-place update (20%) that reads each word and
//!   then writes it. An update loop takes two operations per word, so it
//!   holds its stream twice as long, and scan operations are 2/3 reads.
//! - **Heap (25%).** The remaining 7/16 of memory holds 8-word objects. The
//!   object is Zipf-distributed (s = 1) over a seeded permutation, and the
//!   field offset is geometric (p = 1/2), so headers are hot. 75% reads.
//!   Every `2^16` operations a new phase rotates which objects are hot.
//!
//! Overall, 0.45 * 0.65 + 0.30 * 2/3 + 0.25 * 0.75 = 68% of operations are
//! reads. Written values are random 64-bit words half of the time and random
//! 16-bit values otherwise.

use serde::{Deserialize, Serialize};

use crate::operations::{OperationMetadata, OperationPayload};
use crate::{PayloadError, SplitMix64, dimensions};

pub const KIND: &str = "synthetic-locality-operations";
const STACK_SHARE: u64 = 45;
const SEQUENTIAL_SHARE: u64 = 30;
const PHASE_OPERATIONS: usize = 1 << 16;
const STREAMS: usize = 4;
const OBJECT_WORDS: usize = 8;

/// Counts of reuse distances in log2 buckets.
///
/// The reuse distance of an access is the number of operations since the
/// previous access to the same word: `buckets[b]` counts distances in
/// `[2^b, 2^(b+1))`. First accesses to a word are counted separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseDistance {
    pub first_accesses: usize,
    pub buckets: Vec<usize>,
    /// Bucket index containing the median repeated access, if any.
    pub median_bucket: Option<usize>,
}

pub fn reuse_distance(addresses: &[u32], words: usize) -> ReuseDistance {
    let mut last = vec![usize::MAX; words];
    let mut buckets = Vec::new();
    let mut first_accesses = 0;
    for (i, &address) in addresses.iter().enumerate() {
        let slot = &mut last[address as usize];
        if *slot == usize::MAX {
            first_accesses += 1;
        } else {
            let bucket = (i - *slot).ilog2() as usize;
            if buckets.len() <= bucket {
                buckets.resize(bucket + 1, 0);
            }
            buckets[bucket] += 1;
        }
        *slot = i;
    }
    let repeated: usize = buckets.iter().sum();
    let median_bucket = (repeated > 0).then(|| {
        let mut seen = 0;
        buckets
            .iter()
            .position(|&count| {
                seen += count;
                2 * seen >= repeated
            })
            .unwrap()
    });
    ReuseDistance {
        first_accesses,
        buckets,
        median_bucket,
    }
}

impl SplitMix64 {
    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn percent(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    /// Geometric number of failures before a success of probability `1/2^k`.
    fn geometric(&mut self, k: u32, cap: usize) -> usize {
        let mut count = 0;
        while count + 1 < cap && self.next() & ((1 << k) - 1) != 0 {
            count += 1;
        }
        count
    }

    /// Unit-interval double from 53 random bits.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy)]
struct Stream {
    next: usize,
    remaining: usize,
    stride: usize,
    /// 0 read-only, 1 write-only, 2 alternate read/write.
    mode: u8,
    toggle: bool,
}

struct Layout {
    stack_top: usize,
    stack_words: usize,
    array_base: usize,
    array_words: usize,
    heap_base: usize,
    objects: usize,
}

impl Layout {
    fn new(words: usize) -> Self {
        let stack_words = words / 16;
        let array_words = words / 2;
        let heap_base = array_words;
        let heap_words = words - stack_words - array_words;
        Self {
            stack_top: words,
            stack_words,
            array_base: 0,
            array_words,
            heap_base,
            objects: heap_words / OBJECT_WORDS,
        }
    }
}

fn new_stream(rng: &mut SplitMix64, layout: &Layout) -> Stream {
    let length = 1usize << (4 + rng.below(7));
    let stride = if rng.percent(10) { 8 } else { 1 };
    let mode = match rng.below(10) {
        0..=5 => 0,
        6 | 7 => 1,
        _ => 2,
    };
    let span = (length * stride).min(layout.array_words);
    Stream {
        next: layout.array_base + rng.below(layout.array_words - span + 1),
        remaining: span / stride,
        stride,
        mode,
        toggle: false,
    }
}

/// `2^log_operations` operations over `2^address_bits` words with the
/// locality model in the module documentation.
pub fn locality_operations(
    log_operations: u32,
    address_bits: u32,
    seed: u64,
) -> Result<OperationPayload, PayloadError> {
    let (count, words) = dimensions(log_operations, address_bits)?;
    if address_bits < 6 {
        return Err(PayloadError(
            "the locality workload needs at least 2^6 words".into(),
        ));
    }
    let layout = Layout::new(words);
    let mut rng = SplitMix64(seed ^ 0x10ca_1175);
    let initial_memory: Vec<u64> = (0..words)
        .map(|w| {
            if w >= layout.stack_top - layout.stack_words {
                0
            } else {
                rng.next()
            }
        })
        .collect();
    // Zipf(s = 1) cumulative weights over object ranks.
    let mut cumulative = Vec::with_capacity(layout.objects);
    let mut total = 0.0;
    for rank in 1..=layout.objects {
        total += 1.0 / rank as f64;
        cumulative.push(total);
    }
    let mut permutation: Vec<usize> = (0..layout.objects).collect();
    for i in (1..permutation.len()).rev() {
        permutation.swap(i, rng.below(i + 1));
    }
    let mut memory = initial_memory.clone();
    let (mut addresses, mut values, mut is_write) = (
        Vec::with_capacity(count),
        Vec::with_capacity(count),
        Vec::with_capacity(count),
    );
    let mut frames: Vec<usize> = Vec::new();
    let mut sp = layout.stack_top;
    let mut streams: Vec<Stream> = (0..STREAMS)
        .map(|_| new_stream(&mut rng, &layout))
        .collect();
    let mut phase_shift = 0;
    for i in 0..count {
        if i % PHASE_OPERATIONS == 0 {
            phase_shift = rng.below(layout.objects);
        }
        let choice = rng.below(100) as u64;
        let (address, write) = if choice < STACK_SHARE {
            let frame_limit = layout.stack_words / 2;
            if rng.below(48) == 0 && sp - (layout.stack_top - layout.stack_words) > 16 {
                let size = 2 + rng.below(15);
                if frames.iter().sum::<usize>() + size <= frame_limit {
                    frames.push(size);
                    sp -= size;
                }
            } else if rng.below(48) == 0 && frames.len() > 1 {
                sp += frames.pop().unwrap();
            }
            if frames.is_empty() {
                frames.push(8.min(frame_limit.max(1)));
                sp -= frames[0];
            }
            let frame = *frames.last().unwrap();
            (sp + rng.geometric(2, frame), !rng.percent(65))
        } else if choice < STACK_SHARE + SEQUENTIAL_SHARE {
            let k = rng.below(STREAMS);
            if streams[k].remaining == 0 {
                streams[k] = new_stream(&mut rng, &layout);
            }
            let s = &mut streams[k];
            let address = s.next;
            let write = match s.mode {
                0 => false,
                1 => true,
                _ => {
                    // In-place update: read a word, then write it back.
                    s.toggle = !s.toggle;
                    !s.toggle
                }
            };
            if s.mode != 2 || !s.toggle {
                s.next += s.stride;
                s.remaining -= 1;
            }
            (address, write)
        } else {
            let u = rng.unit() * total;
            let rank = cumulative
                .partition_point(|&c| c < u)
                .min(layout.objects - 1);
            let object = permutation[(rank + phase_shift) % layout.objects];
            let field = rng.geometric(1, OBJECT_WORDS);
            (
                layout.heap_base + object * OBJECT_WORDS + field,
                !rng.percent(75),
            )
        };
        let value = if write {
            if rng.next() & 1 == 0 {
                rng.next()
            } else {
                rng.next() & 0xffff
            }
        } else {
            memory[address]
        };
        memory[address] = value;
        addresses.push(address as u32);
        values.push(value);
        is_write.push(write);
    }
    let writes = is_write.iter().filter(|&&w| w).count();
    Ok(OperationPayload {
        initial_memory,
        addresses,
        values,
        is_write,
        metadata: OperationMetadata {
            kind: KIND.into(),
            seed,
            address_bits,
            operations: count,
            read_operations: count - writes,
            write_operations: writes,
            memory_operations: count,
            dummy_read_operations: 0,
            narrow_store_read_operations: 0,
            narrow_store_write_operations: 0,
            trace_mode: "synthetic-locality".into(),
            source_vm_cycles: 0,
            selected_source_rows: 0,
            source_load_rows: 0,
            source_full_store_rows: 0,
            source_narrow_store_rows: 0,
            pattern_operations: count,
            full_pattern_repetitions: 1,
            tail_pattern_operations: 0,
            replayed_operations: 0,
            padding_operations: 0,
            completed_source_rows: 0,
            partial_source_operation: None,
            guest: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_distance_buckets_by_log2_gap() {
        // a b a c a b: a at 0,2,4 (gaps 2,2), b at 1,5 (gap 4).
        let r = reuse_distance(&[0, 1, 0, 2, 0, 1], 4);
        assert_eq!(r.first_accesses, 3);
        assert_eq!(r.buckets, vec![0, 2, 1]);
        assert_eq!(r.median_bucket, Some(1));
        assert_eq!(reuse_distance(&[3, 3], 4).buckets, vec![1]);
        assert_eq!(reuse_distance(&[1], 4).median_bucket, None);
    }

    #[test]
    fn locality_trace_is_deterministic_valid_and_mostly_reads() {
        let a = locality_operations(16, 16, 7).unwrap();
        assert_eq!(a, locality_operations(16, 16, 7).unwrap());
        assert_ne!(
            a.addresses,
            locality_operations(16, 16, 8).unwrap().addresses
        );
        a.replay().unwrap();
        let reads = a.metadata.read_operations as f64 / a.metadata.operations as f64;
        assert!((0.66..0.72).contains(&reads), "read fraction {reads}");
        assert_eq!(a.metadata.pattern_operations, 1 << 16);
        assert!(a.addresses.iter().all(|&x| (x as usize) < 1 << 16));
        // The small-memory edge still produces a valid trace.
        locality_operations(10, 6, 1).unwrap().replay().unwrap();
        assert!(locality_operations(10, 5, 1).is_err());
    }

    #[test]
    fn locality_trace_has_short_and_long_reuse_and_does_not_repeat() {
        let p = locality_operations(18, 16, 3).unwrap();
        let r = reuse_distance(&p.addresses, 1 << 16);
        let repeated: usize = r.buckets.iter().sum();
        assert_eq!(repeated + r.first_accesses, 1 << 18);
        // Hot stack slots and in-place updates: many reuses within 8 operations.
        let short: usize = r.buckets[..3].iter().sum();
        assert!(short * 5 > repeated, "short {short} of {repeated}");
        // Scans and the Zipf tail: some reuses span more than 2^14 operations.
        let long: usize = r.buckets.iter().skip(14).sum();
        assert!(long * 50 > repeated, "long {long} of {repeated}");
        // More distinct words than any guest touches in 2^18 operations
        // (at most 5,885, for ERC20).
        assert!(r.first_accesses > 10_000, "{}", r.first_accesses);
        // No period: the second half is not a copy of the first.
        let half = p.addresses.len() / 2;
        assert_ne!(p.addresses[..half], p.addresses[half..]);
    }
}
