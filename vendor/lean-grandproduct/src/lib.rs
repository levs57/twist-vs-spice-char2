//! Batched radix-four grand products adapted from leanVM's pinned GKR prover.
//!
//! The normalized equality sumcheck sends four coefficients per quartic round.
//! It divides by no equality prefix, including when a challenge makes that
//! prefix zero. Two trees reduce to a shared, low-coordinate-first leaf point.
//! Callers must authenticate the returned leaf evaluations against their input
//! commitments and, for a permutation argument, check equality of the roots.
//!
//! Source: leanEthereum/leanVM e5d75f858e91161921c0afe3d9f56300e9a3a03c,
//! crates/leanvm_core/src/gkr.rs and crates/verifier/src/{bus,transcript}.rs.
//! The F192 scalar/packed kernels are translated to OM GF128, the thread pool
//! to Rayon, and the proof transport to an explicitly labelled OM transcript.

use std::fmt;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Range;

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_transcript::Transcript;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 1 << 11;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Error {}

fn require(condition: bool, message: &str) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}

fn label(parent: &[u8], suffix: &[u8]) -> Vec<u8> {
    [parent, suffix].concat()
}

fn layer_label(parent: &[u8], layer: usize) -> Vec<u8> {
    label(parent, format!("/layer/{layer}").as_bytes())
}

#[inline(always)]
fn mul4(a: [K; 4], b: [K; 4]) -> [K; 4] {
    let low = mul_pairs([a[0], a[1]], [b[0], b[1]]);
    let high = mul_pairs([a[2], a[3]], [b[2], b[3]]);
    [low[0], low[1], high[0], high[1]]
}

#[inline(always)]
fn raw4(a: [K; 4], b: [K; 4]) -> [U256; 4] {
    let low = clmul_pairs([a[0], a[1]], [b[0], b[1]]);
    let high = clmul_pairs([a[2], a[3]], [b[2], b[3]]);
    [low[0], low[1], high[0], high[1]]
}

fn add4(a: [K; 4], b: [K; 4]) -> [K; 4] {
    std::array::from_fn(|i| a[i] + b[i])
}

fn eq_table(point: &[K]) -> Vec<K> {
    let mut values = vec![K::ONE];
    for &r in point {
        let len = values.len();
        values.resize(2 * len, K::ZERO);
        for j in 0..len {
            let high = values[j] * r;
            values[len + j] = high;
            values[j] += high;
        }
    }
    values
}

fn interpolate(a: K, b: K, r: K) -> K {
    a + r * (a + b)
}

fn polynomial(coefficients: &[K], r: K) -> K {
    coefficients.iter().rev().fold(K::ZERO, |v, &c| v * r + c)
}

fn coarsen(ones: &[Range<usize>], shift: usize) -> Vec<Range<usize>> {
    ones.iter()
        .map(|r| r.start.div_ceil(1 << shift)..r.end >> shift)
        .filter(|r| !r.is_empty())
        .collect()
}

fn covered(ones: &[Range<usize>], start: usize, end: usize) -> bool {
    let i = ones.partition_point(|r| r.start <= start);
    i > 0 && ones[i - 1].end >= end
}

/// Split a window at the known identity ranges, as in the donor's scalar path.
fn runs<'a>(
    range: Range<usize>,
    ones: &'a [Range<usize>],
) -> impl Iterator<Item = (Range<usize>, bool)> + 'a {
    let first = ones.partition_point(|r| r.end <= range.start);
    let mut ones = ones[first..].iter().peekable();
    let mut at = range.start;
    std::iter::from_fn(move || {
        if at >= range.end {
            return None;
        }
        let run = match ones.peek() {
            Some(r) if r.start <= at => {
                let end = r.end.min(range.end);
                ones.next();
                (at..end, true)
            }
            Some(r) => (at..r.start.min(range.end), false),
            None => (at..range.end, false),
        };
        at = run.0.end;
        Some(run)
    })
}

fn next_level(current: &[K], ones: &[Range<usize>]) -> Vec<K> {
    let rows = current.len().div_ceil(4);
    let identity = coarsen(ones, 2);
    let product = |row: usize| {
        if covered(&identity, row, row + 1) {
            return K::ONE;
        }
        let child = |i| current.get(4 * row + i).copied().unwrap_or(K::ONE);
        let pair = mul_pairs([child(0), child(2)], [child(1), child(3)]);
        pair[0] * pair[1]
    };
    if rows >= PAR_THRESHOLD {
        (0..rows).into_par_iter().map(product).collect()
    } else {
        (0..rows).map(product).collect()
    }
}

fn identity_ranges(size: usize, ones: Vec<Range<usize>>) -> Result<Vec<Range<usize>>, Error> {
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(ones.len());
    for range in ones {
        require(
            range.start <= range.end && range.end <= size,
            "invalid identity ranges",
        )?;
        if range.is_empty() {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            require(last.end <= range.start, "overlapping identity ranges")?;
            if last.end == range.start {
                last.end = range.end;
                continue;
            }
        }
        merged.push(range);
    }
    Ok(merged)
}

fn explicit_extent(size: usize, ones: &[Range<usize>]) -> usize {
    ones.last()
        .filter(|r| r.end == size)
        .map_or(size, |r| r.start.max(1))
}

#[inline(always)]
fn sixteen(values: &[K; 16]) -> ([K; 4], K) {
    let left = mul4(
        [values[0], values[4], values[8], values[12]],
        [values[1], values[5], values[9], values[13]],
    );
    let right = mul4(
        [values[2], values[6], values[10], values[14]],
        [values[3], values[7], values[11], values[15]],
    );
    let fours = mul4(left, right);
    let pair = mul_pairs([fours[0], fours[2]], [fours[1], fours[3]]);
    (fours, pair[0] * pair[1])
}

/// Prepare both levels together while each sixteen-leaf block is hot.
fn prepare_levels(values: &[K], ones: &[Range<usize>]) -> (Vec<K>, Vec<K>) {
    const ROWS: usize = 64;
    let mut fours = output_buffer(Vec::new(), values.len().div_ceil(4));
    let mut sixteens = output_buffer(Vec::new(), values.len().div_ceil(16));
    let fill = |(tile, (four, six)): (usize, (&mut [MaybeUninit<K>], &mut [MaybeUninit<K>]))| {
        for (row, dst) in six.iter_mut().enumerate() {
            let start = (tile * ROWS + row) * 16;
            let end = (start + 16).min(values.len());
            let (products, product) = if covered(ones, start, end) {
                ([K::ONE; 4], K::ONE)
            } else {
                let mut block = [K::ONE; 16];
                block[..end - start].copy_from_slice(&values[start..end]);
                sixteen(&block)
            };
            for (dst, &src) in four[4 * row..].iter_mut().take(4).zip(&products) {
                dst.write(src);
            }
            dst.write(product);
        }
    };
    if values.len() >= 4 * PAR_THRESHOLD {
        fours
            .par_chunks_mut(4 * ROWS)
            .zip(sixteens.par_chunks_mut(ROWS))
            .enumerate()
            .for_each(fill);
    } else {
        fours
            .chunks_mut(4 * ROWS)
            .zip(sixteens.chunks_mut(ROWS))
            .enumerate()
            .for_each(fill);
    }
    // SAFETY: every destination entry is written by exactly one tile.
    unsafe { (initialized(fours), initialized(sixteens)) }
}

/// Prepared leaves and their first two radix-four levels.
///
/// Missing suffix entries up to the caller's logical tree size are identities.
/// Explicit identity ranges describe structural padding, never content-based
/// merging. Their values are checked once at construction.
#[derive(Clone, Debug)]
pub struct Leaves {
    input_len: usize,
    values: Vec<K>,
    fours: Vec<K>,
    sixteens: Vec<K>,
    ones: Vec<Range<usize>>,
}

impl Leaves {
    pub fn new(values: Vec<K>) -> Self {
        Self::with_identity_ranges(values, Vec::new()).expect("nonempty grand-product leaves")
    }

    pub fn with_identity_ranges(
        mut values: Vec<K>,
        ones: Vec<Range<usize>>,
    ) -> Result<Self, Error> {
        require(!values.is_empty(), "empty grand-product leaves")?;
        let input_len = values.len();
        let ones = identity_ranges(input_len, ones)?;
        for range in &ones {
            require(
                values[range.clone()].iter().all(|&v| v == K::ONE),
                "nonidentity value in declared padding",
            )?;
        }
        values.truncate(explicit_extent(input_len, &ones));
        let (fours, sixteens) = prepare_levels(&values, &ones);
        Ok(Self {
            input_len,
            values,
            fours,
            sixteens,
            ones,
        })
    }

    /// Generate both trees and their prepared levels in one tiled pass. The
    /// callback is called only for nonpadding rows, and the known suffix is
    /// stored implicitly rather than allocated.
    pub fn from_fn_pair(
        size: usize,
        ones: Vec<Range<usize>>,
        row: impl Fn(usize) -> [K; 2] + Sync,
    ) -> Result<[Self; 2], Error> {
        require(size > 0, "empty grand-product leaves")?;
        let ones = identity_ranges(size, ones)?;
        let len = explicit_extent(size, &ones);
        let [mut left, mut right] = [0, 1].map(|_| output_buffer(Vec::new(), len));
        let [mut left_four, mut right_four] =
            [0, 1].map(|_| output_buffer(Vec::new(), len.div_ceil(4)));
        let [mut left_six, mut right_six] =
            [0, 1].map(|_| output_buffer(Vec::new(), len.div_ceil(16)));
        const ROWS: usize = 64;
        type Output<'a> = &'a mut [MaybeUninit<K>];
        type Outputs<'a> = (
            (
                (((Output<'a>, Output<'a>), Output<'a>), Output<'a>),
                Output<'a>,
            ),
            Output<'a>,
        );
        let fill = |(tile, (((((left, right), left_four), right_four), left_six), right_six)): (
            usize,
            Outputs<'_>,
        )| {
            for block in 0..left_six.len() {
                let start = (tile * ROWS + block) * 16;
                let count = (len - start).min(16);
                let mut values = [[K::ONE; 16]; 2];
                let all_one = covered(&ones, start, start + count);
                if !all_one {
                    for i in 0..count {
                        if !covered(&ones, start + i, start + i + 1) {
                            let pair = row(start + i);
                            values[0][i] = pair[0];
                            values[1][i] = pair[1];
                        }
                    }
                }
                for (side, (leaves, fours, six)) in [
                    (left.as_mut(), left_four.as_mut(), left_six.as_mut()),
                    (right.as_mut(), right_four.as_mut(), right_six.as_mut()),
                ]
                .into_iter()
                .enumerate()
                {
                    for (dst, &src) in leaves[16 * block..]
                        .iter_mut()
                        .take(count)
                        .zip(&values[side])
                    {
                        dst.write(src);
                    }
                    let (products, product) = if all_one {
                        ([K::ONE; 4], K::ONE)
                    } else {
                        sixteen(&values[side])
                    };
                    for (dst, &src) in fours[4 * block..].iter_mut().take(4).zip(&products) {
                        dst.write(src);
                    }
                    six[block].write(product);
                }
            }
        };
        if len >= 4 * PAR_THRESHOLD {
            left.par_chunks_mut(16 * ROWS)
                .zip(right.par_chunks_mut(16 * ROWS))
                .zip(left_four.par_chunks_mut(4 * ROWS))
                .zip(right_four.par_chunks_mut(4 * ROWS))
                .zip(left_six.par_chunks_mut(ROWS))
                .zip(right_six.par_chunks_mut(ROWS))
                .enumerate()
                .for_each(fill);
        } else {
            left.chunks_mut(16 * ROWS)
                .zip(right.chunks_mut(16 * ROWS))
                .zip(left_four.chunks_mut(4 * ROWS))
                .zip(right_four.chunks_mut(4 * ROWS))
                .zip(left_six.chunks_mut(ROWS))
                .zip(right_six.chunks_mut(ROWS))
                .enumerate()
                .for_each(fill);
        }
        // SAFETY: the disjoint tiles initialize all six destination streams.
        let values = unsafe { [initialized(left), initialized(right)] };
        let fours = unsafe { [initialized(left_four), initialized(right_four)] };
        let sixteens = unsafe { [initialized(left_six), initialized(right_six)] };
        Ok(values
            .into_iter()
            .zip(fours)
            .zip(sixteens)
            .map(|((values, fours), sixteens)| Self {
                input_len: size,
                values,
                fours,
                sixteens,
                ones: ones.clone(),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap())
    }

    pub fn values(&self) -> &[K] {
        &self.values
    }
}

fn build_layers(leaves: Leaves, log_size: usize) -> Vec<Vec<K>> {
    let mut layers: Vec<Vec<K>> = (0..=log_size).map(|_| Vec::new()).collect();
    layers[0] = leaves.values;
    let mut level = 0;
    for (given, below) in [(2, leaves.fours), (4, leaves.sixteens)] {
        if given <= log_size {
            layers[given] = below;
            level = given;
        }
    }
    while level + 2 <= log_size {
        let next = next_level(&layers[level], &coarsen(&leaves.ones, level));
        level += 2;
        layers[level] = next;
    }
    if level < log_size {
        layers[log_size] = vec![layers[level].iter().copied().fold(K::ONE, |a, b| a * b)];
    }
    layers
}

struct SplitEq {
    lo: Vec<K>,
    hi: Vec<K>,
    lo_bits: usize,
}

impl SplitEq {
    fn new(point: &[K]) -> Self {
        let lo_bits = point.len().div_ceil(2).max(point.len().min(3));
        Self {
            lo: eq_table(&point[..lo_bits]),
            hi: eq_table(&point[lo_bits..]),
            lo_bits,
        }
    }

    fn sum(&self, range: Range<usize>, term: impl Fn(usize, K) -> [U256; 4]) -> [K; 4] {
        let mask = (1usize << self.lo_bits) - 1;
        let mut total = [K::ZERO; 4];
        let mut start = range.start;
        while start < range.end {
            let block = start >> self.lo_bits;
            let end = range.end.min((block + 1) << self.lo_bits);
            let sum = (start..end).fold([U256::ZERO; 4], |mut acc, pair| {
                let term = term(pair, self.lo[pair & mask]);
                for i in 0..4 {
                    acc[i] += term[i];
                }
                acc
            });
            total = add4(total, mul4([self.hi[block]; 4], sum.map(U256::reduce)));
            start = end;
        }
        total
    }
}

/// Donor quartic identity: q(0)+q(1), c2, c3, c4, with delayed reductions.
#[inline(always)]
fn quartic_summand(lines: [[K; 2]; 4], equality: K) -> [U256; 4] {
    let [[g0, s0], [g1, s1], [g2, s2], [g3, s3]] = lines;
    let [a0, d0, right0, right2] = mul4([equality, equality, g2, s2], [g0, s0, g3, s3]);
    let [left0, left2, left_at_one, right_at_one] =
        mul4([a0, d0, a0 + d0, g2 + s2], [g1, s1, g1 + s1, g3 + s3]);
    let left1 = left_at_one + left0 + left2;
    let right1 = right_at_one + right0 + right2;
    let [c0, c4, middle, at_one] = raw4(
        [left0, left2, left1, left_at_one],
        [right0, right2, right1, right_at_one],
    );
    let cross_even = (left0 + left2).clmul(right0 + right2);
    let cross_high = (left1 + left2).clmul(right1 + right2);
    [
        c0 + at_one,
        cross_even + c0 + c4 + middle,
        cross_high + middle + c4,
        c4,
    ]
}

fn window_rows(total: usize) -> usize {
    total
        .div_ceil(rayon::current_num_threads().max(1) * 16)
        .clamp(64, 1 << 10)
}

fn sum_tasks(tasks: usize, parallel: bool, f: impl Fn(usize) -> [K; 4] + Sync + Send) -> [K; 4] {
    if parallel {
        (0..tasks)
            .into_par_iter()
            .map(f)
            .reduce(|| [K::ZERO; 4], add4)
    } else {
        (0..tasks).map(f).fold([K::ZERO; 4], add4)
    }
}

// Preserve the old allocation while making every output write explicit.
fn output_buffer(spare: Vec<K>, len: usize) -> Vec<MaybeUninit<K>> {
    let mut spare = ManuallyDrop::new(spare);
    // SAFETY: MaybeUninit<K> has exactly K's layout; old initialized elements
    // remain valid and newly extended entries are never read before writing.
    let mut output =
        unsafe { Vec::from_raw_parts(spare.as_mut_ptr().cast(), spare.len(), spare.capacity()) };
    output.resize_with(len, MaybeUninit::uninit);
    output
}

unsafe fn initialized(output: Vec<MaybeUninit<K>>) -> Vec<K> {
    let mut output = ManuallyDrop::new(output);
    // SAFETY: the caller has written all len elements; layouts agree.
    unsafe { Vec::from_raw_parts(output.as_mut_ptr().cast(), output.len(), output.capacity()) }
}

struct LayerState {
    values: Vec<K>,
    spare: Vec<K>,
    logical_rows: usize,
    ones: Vec<Range<usize>>,
}

impl LayerState {
    fn new(mut values: Vec<K>, logical_rows: usize, ones: Vec<Range<usize>>) -> Self {
        values.resize(4 * values.len().max(1).div_ceil(4), K::ONE);
        Self {
            values,
            spare: Vec::new(),
            logical_rows,
            ones,
        }
    }

    fn round_message(&self, eq: &SplitEq) -> [K; 4] {
        let stored_rows = self.values.len() / 4;
        let full_pairs = stored_rows / 2;
        let identity = coarsen(&self.ones, 1);
        let summand = |pair: usize, weight: K| {
            let lo = 8 * pair;
            quartic_summand(
                std::array::from_fn(|i| {
                    [
                        self.values[lo + i],
                        self.values[lo + i] + self.values[lo + 4 + i],
                    ]
                }),
                weight,
            )
        };
        let rows = window_rows(full_pairs);
        let window = |task: usize| {
            runs(task * rows..((task + 1) * rows).min(full_pairs), &identity)
                .filter(|(_, identity)| !identity)
                .fold([K::ZERO; 4], |sum, (range, _)| {
                    add4(sum, eq.sum(range, summand))
                })
        };
        let mut result = sum_tasks(
            full_pairs.div_ceil(rows),
            full_pairs >= PAR_THRESHOLD,
            window,
        );
        if stored_rows % 2 != 0 {
            result = add4(
                result,
                eq.sum(full_pairs..full_pairs + 1, |pair, weight| {
                    let lo = 8 * pair;
                    quartic_summand(
                        std::array::from_fn(|i| {
                            [self.values[lo + i], self.values[lo + i] + K::ONE]
                        }),
                        weight,
                    )
                }),
            );
        }
        result
    }

    fn fold<const MESSAGE: bool>(&mut self, challenge: K, eq: Option<&SplitEq>) -> [K; 4] {
        const PAIRS: usize = 64;
        let stored_rows = self.values.len() / 4;
        let rows = stored_rows.div_ceil(2);
        let identity = coarsen(&self.ones, 1);
        let identity_pairs = coarsen(&identity, 1);
        let mut output = output_buffer(std::mem::take(&mut self.spare), 4 * rows);
        let values = &self.values;
        let task = |(index, out): (usize, &mut [MaybeUninit<K>])| {
            let first = index * PAIRS;
            let end_row = (2 * first + out.len() / 4).min(rows);
            let end_pair = end_row.div_ceil(2);
            let mut stage = [K::ONE; 8 * PAIRS];
            for (range, identity) in runs(2 * first..end_row, &identity) {
                if identity {
                    continue;
                }
                for row in range {
                    let lo = 8 * row;
                    let left = &values[lo..lo + 4];
                    let right = values.get(lo + 4..lo + 8).unwrap_or(&[K::ONE; 4]);
                    let products =
                        mul4(std::array::from_fn(|i| left[i] + right[i]), [challenge; 4]);
                    for i in 0..4 {
                        stage[4 * (row - 2 * first) + i] = left[i] + products[i];
                    }
                }
            }
            let message = if MESSAGE {
                let eq = eq.expect("next-round equality");
                let summand = |pair: usize, weight: K| {
                    let lo = 8 * (pair - first);
                    let left = &stage[lo..lo + 4];
                    let right = if 2 * pair + 1 < rows {
                        &stage[lo + 4..lo + 8]
                    } else {
                        &[K::ONE; 4]
                    };
                    quartic_summand(
                        std::array::from_fn(|i| [left[i], left[i] + right[i]]),
                        weight,
                    )
                };
                runs(first..end_pair, &identity_pairs)
                    .filter(|(_, identity)| !identity)
                    .fold([K::ZERO; 4], |sum, (range, _)| {
                        add4(sum, eq.sum(range, summand))
                    })
            } else {
                [K::ZERO; 4]
            };
            for (dst, &src) in out.iter_mut().zip(&stage) {
                dst.write(src);
            }
            message
        };
        let message = if rows >= PAR_THRESHOLD {
            output
                .par_chunks_mut(8 * PAIRS)
                .enumerate()
                .map(task)
                .reduce(|| [K::ZERO; 4], add4)
        } else {
            output
                .chunks_mut(8 * PAIRS)
                .enumerate()
                .map(task)
                .fold([K::ZERO; 4], add4)
        };
        // SAFETY: every disjoint output chunk writes every one of its entries.
        let next = unsafe { initialized(output) };
        self.spare = std::mem::replace(&mut self.values, next);
        self.logical_rows /= 2;
        self.ones = identity;
        message
    }

    fn children(&self) -> [K; 4] {
        assert_eq!(self.logical_rows, 1);
        self.values[..4].try_into().unwrap()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerProof {
    /// The normalized cofactor coefficients c1,c2,c3,c4 in each round.
    pub rounds: Vec<[K; 4]>,
    /// Two children for an initial binary layer, four otherwise, tree-major.
    pub children: [Vec<K>; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrandProductProof {
    pub roots: [K; 2],
    pub layers: Vec<LayerProof>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafClaims {
    /// Low variables first; all logical identity-padding variables are retained.
    pub point: Vec<K>,
    pub values: [K; 2],
}

fn bind_context(log_size: usize, roots: &[K; 2], tr: &mut Transcript, parent: &[u8]) {
    tr.absorb_bytes(&label(parent, b"/protocol"), b"leanVM/GF128/radix4/v1");
    tr.absorb_bytes(
        &label(parent, b"/shape"),
        &[(log_size as u64).to_le_bytes(), 2u64.to_le_bytes()].concat(),
    );
    tr.absorb_ks(&label(parent, b"/roots"), roots);
}

fn bind_children(
    children: &[Vec<K>; 2],
    tr: &mut Transcript,
    parent: &[u8],
    step: usize,
) -> Vec<K> {
    tr.absorb_ks(&label(parent, b"/children/0"), &children[0]);
    tr.absorb_ks(&label(parent, b"/children/1"), &children[1]);
    tr.challenge_ks(&label(parent, b"/child-point"), step)
}

fn child_value(children: &[K], point: &[K]) -> K {
    if point.len() == 1 {
        interpolate(children[0], children[1], point[0])
    } else {
        interpolate(
            interpolate(children[0], children[1], point[0]),
            interpolate(children[2], children[3], point[0]),
            point[1],
        )
    }
}

/// Prove two identity-padded trees. Roots need not agree; the caller checks the
/// desired root relation and authenticates both returned leaf evaluations.
pub fn prove(
    log_size: usize,
    leaves: [Leaves; 2],
    tr: &mut Transcript,
    parent: &[u8],
) -> Result<(GrandProductProof, LeafClaims), Error> {
    require(
        log_size < usize::BITS as usize,
        "grand-product size overflow",
    )?;
    require(
        leaves.iter().all(|l| l.input_len <= 1usize << log_size),
        "leaves exceed logical size",
    )?;
    let identities = leaves.each_ref().map(|l| l.ones.clone());
    let mut layers = leaves.map(|l| build_layers(l, log_size));
    let roots = [layers[0][log_size][0], layers[1][log_size][0]];
    bind_context(log_size, &roots, tr, parent);
    let mut proof = GrandProductProof {
        roots,
        layers: Vec::with_capacity(log_size.div_ceil(2)),
    };
    let mut point = Vec::new();
    let mut values = roots;
    let mut layer = log_size;
    while layer > 0 {
        let ll = layer_label(parent, layer);
        let lambda = tr.challenge_k(&label(&ll, b"/batch"));
        let round_count = log_size - layer;
        if layer % 2 != 0 {
            let children = [0, 1].map(|tree| {
                let below = &layers[tree][layer - 1];
                vec![below[0], below.get(1).copied().unwrap_or(K::ONE)]
            });
            let next = bind_children(&children, tr, &ll, 1);
            values = [0, 1].map(|tree| child_value(&children[tree], &next));
            proof.layers.push(LayerProof {
                rounds: Vec::new(),
                children,
            });
            point = next;
            layer -= 1;
            continue;
        }
        let mut trees = [0, 1].map(|tree| {
            LayerState::new(
                std::mem::take(&mut layers[tree][layer - 2]),
                1usize << round_count,
                coarsen(&identities[tree], layer),
            )
        });
        let mut messages = if round_count > 0 {
            let eq = SplitEq::new(&point[1..]);
            trees.each_ref().map(|tree| tree.round_message(&eq))
        } else {
            [[K::ZERO; 4]; 2]
        };
        let mut rounds = Vec::with_capacity(round_count);
        let mut round_point = Vec::with_capacity(round_count);
        for round in 0..round_count {
            let mut coefficients =
                std::array::from_fn(|i| messages[0][i] + lambda * messages[1][i]);
            coefficients[0] += coefficients[1] + coefficients[2] + coefficients[3];
            tr.absorb_ks(&label(&ll, b"/round"), &coefficients);
            let challenge = tr.challenge_k(&label(&ll, b"/challenge"));
            rounds.push(coefficients);
            round_point.push(challenge);
            if round + 1 < round_count {
                let eq = SplitEq::new(&point[2 + round..]);
                messages = trees
                    .each_mut()
                    .map(|tree| tree.fold::<true>(challenge, Some(&eq)));
            } else {
                for tree in &mut trees {
                    tree.fold::<false>(challenge, None);
                }
            }
        }
        let children = trees.each_ref().map(|tree| tree.children().to_vec());
        let next = bind_children(&children, tr, &ll, 2);
        values = [0, 1].map(|tree| child_value(&children[tree], &next));
        point = next;
        point.extend(round_point);
        proof.layers.push(LayerProof { rounds, children });
        layer -= 2;
    }
    Ok((proof, LeafClaims { point, values }))
}

pub fn verify(
    log_size: usize,
    proof: &GrandProductProof,
    tr: &mut Transcript,
    parent: &[u8],
) -> Result<LeafClaims, Error> {
    require(
        log_size < usize::BITS as usize,
        "grand-product size overflow",
    )?;
    require(
        proof.layers.len() == log_size.div_ceil(2),
        "incorrect grand-product layer count",
    )?;
    let mut layer = log_size;
    for current in &proof.layers {
        let step = if layer % 2 != 0 { 1 } else { 2 };
        require(
            current.rounds.len() == log_size - layer
                && current.children.iter().all(|c| c.len() == 1 << step),
            "malformed grand-product layer",
        )?;
        layer -= step;
    }
    bind_context(log_size, &proof.roots, tr, parent);
    let mut point = Vec::new();
    let mut values = proof.roots;
    let mut layer = log_size;
    for current in &proof.layers {
        let step = if layer % 2 != 0 { 1 } else { 2 };
        let ll = layer_label(parent, layer);
        let lambda = tr.challenge_k(&label(&ll, b"/batch"));
        let mut claim = values[0] + lambda * values[1];
        let mut round_point = Vec::with_capacity(point.len());
        for (&at, tail) in point.iter().zip(&current.rounds) {
            let mut coefficients = [K::ZERO; 5];
            coefficients[1..].copy_from_slice(tail);
            coefficients[0] = claim + at * tail.iter().copied().fold(K::ZERO, |a, b| a + b);
            tr.absorb_ks(&label(&ll, b"/round"), tail);
            let challenge = tr.challenge_k(&label(&ll, b"/challenge"));
            round_point.push(challenge);
            claim = polynomial(&coefficients, challenge);
        }
        let products = current
            .children
            .each_ref()
            .map(|c| c.iter().copied().fold(K::ONE, |a, b| a * b));
        require(
            claim == products[0] + lambda * products[1],
            "grand-product layer relation failed",
        )?;
        let next = bind_children(&current.children, tr, &ll, step);
        values = [0, 1].map(|tree| child_value(&current.children[tree], &next));
        point = next;
        point.extend(round_point);
        layer -= step;
    }
    Ok(LeafClaims { point, values })
}

#[cfg(test)]
mod tests;
