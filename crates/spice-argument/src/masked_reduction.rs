//! One Eq-factored sumcheck for the two cycle masks and public-initial mask.
//!
//! A smaller source domain is repeated along high variables, not zero-padded.
//! Its Eq weights therefore sum to one, including at Boolean challenge points.
//! Terminal scalars are conditional claims: the caller MUST authenticate the
//! three cycle sources at point[..n] and initial flag at point[..a] with PCS.

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_memcheck::eq_factor::{self, Cofactor, Endpoint};
use om_sumcheck::{
    SumcheckProof, eq_eval, eq_table, mle_eval_bits, sumcheck_init, sumcheck_verify,
};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskClaims {
    pub cycle: [K; 2],
    pub initial: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskProof {
    pub sc: SumcheckProof,
    pub value: K,
    pub is_write: K,
    pub next_is_write: K,
    pub initial_next_is_write: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceClaims {
    /// Full common point; the outer statement supplies each source dimension.
    pub point: Vec<K>,
    pub value: K,
    pub is_write: K,
    pub next_is_write: K,
    pub initial_next_is_write: K,
}

fn require(ok: bool, message: &str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
fn label(base: &[u8], suffix: &[u8]) -> Vec<u8> {
    [base, suffix].concat()
}

fn public_shape(initial: &[u64], gp_point: &[K]) -> Result<(usize, usize), Error> {
    require(
        initial.len().is_power_of_two(),
        "masked reduction: initial domain must be a nonzero power of two",
    )?;
    require(
        !gp_point.is_empty() && gp_point.len() <= 32,
        "masked reduction: invalid GP point dimension",
    )?;
    let a = initial.len().trailing_zeros() as usize;
    let l = gp_point.len() - 1;
    require(
        a <= l,
        "masked reduction: initial domain exceeds common point",
    )?;
    Ok((a, l))
}

fn setup(
    a: usize,
    l: usize,
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> [K; 3] {
    tr.absorb_bytes(
        &label(base, b"/protocol"),
        b"spice/masked-reduction/replicated-eq/v1",
    );
    tr.absorb_bytes(
        &label(base, b"/dimensions"),
        &[a as u64, l as u64, 64]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    tr.absorb_ks(&label(base, b"/gp-point"), gp_point);
    tr.absorb_ks(
        &label(base, b"/claims"),
        &[claims.cycle[0], claims.cycle[1], claims.initial],
    );
    // All three claimed values are fixed before any batching coefficient.
    std::array::from_fn(|_| tr.challenge_k(&label(base, b"/batch")))
}

fn batched_claim(claims: &MaskClaims, weights: [K; 3]) -> K {
    let values = [claims.cycle[0], claims.cycle[1], claims.initial];
    (0..3)
        .fold(U256::ZERO, |sum, i| sum + weights[i].clmul(values[i]))
        .reduce()
}

struct Products;
impl Cofactor for Products {
    const DEGREE: usize = 2;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        4
    }
    fn value(&self, row: &[K]) -> K {
        row[0] * row[1] + row[2] * row[3]
    }
    fn prepare(&self, low_eq: &[K]) -> Vec<K> {
        low_eq.to_vec()
    }
    fn accumulate(
        &self,
        prepared: &Vec<K>,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    ) {
        let mut add = |slot: usize, a: [K; 2], b: [K; 2]| {
            let scaled = mul_pairs(a, [prepared[low]; 2]);
            let products = clmul_pairs(scaled, b);
            out[slot] += products[0] + products[1];
        };
        if matches!(endpoint, Endpoint::Both | Endpoint::Zero) {
            add(0, [left[0], left[2]], [left[1], left[3]]);
        }
        if matches!(endpoint, Endpoint::Both | Endpoint::One) {
            add(1, [right[0], right[2]], [right[1], right[3]]);
        }
        add(
            2,
            [left[0] + right[0], left[2] + right[2]],
            [left[1] + right[1], left[3] + right[3]],
        );
    }
}

const PAR_THRESHOLD: usize = 1 << 14;

/// Statistics of one product under the suffix equality of its own domain.
/// Coordinates beyond that domain have equality weights summing to one.
struct NativeRound {
    lo: Vec<K>,
    hi: Vec<K>,
    h: usize,
    at_one: bool,
}

impl NativeRound {
    fn new(suffix: &[K], at_one: bool) -> Self {
        let h = suffix.len() / 2;
        Self {
            lo: eq_table(&suffix[..h]),
            hi: eq_table(&suffix[h..]),
            h,
            at_one,
        }
    }
}

struct NativeSums<'a> {
    round: &'a NativeRound,
    high: usize,
    partial: [U256; 2],
    total: [U256; 2],
}

impl<'a> NativeSums<'a> {
    fn new(round: &'a NativeRound) -> Self {
        Self {
            round,
            high: 0,
            partial: [U256::ZERO; 2],
            total: [U256::ZERO; 2],
        }
    }

    #[inline]
    fn pair(&mut self, index: usize, left: [K; 2], right: [K; 2]) {
        let high = index >> self.round.h;
        if high != self.high {
            self.flush();
            self.high = high;
        }
        let low = index & ((1usize << self.round.h) - 1);
        let endpoint = if self.round.at_one { right } else { left };
        // Fill both SIMD lanes even though only one native product is being
        // scanned: selected endpoint in lane 0, quadratic coefficient in 1.
        let a = [endpoint[0], left[0] + right[0]];
        let b = [endpoint[1], left[1] + right[1]];
        let scaled = mul_pairs(a, [self.round.lo[low]; 2]);
        let products = clmul_pairs(scaled, b);
        self.partial[0] += products[0];
        self.partial[1] += products[1];
    }

    fn flush(&mut self) {
        let products = clmul_pairs(
            [self.round.hi[self.high]; 2],
            self.partial.map(U256::reduce),
        );
        self.total[0] += products[0];
        self.total[1] += products[1];
        self.partial = [U256::ZERO; 2];
    }

    fn finish(mut self) -> [U256; 2] {
        self.flush();
        self.total
    }
}

fn add_raw(a: [U256; 2], b: [U256; 2]) -> [U256; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn native_statistics(raw: [U256; 2], at_one: bool) -> [K; 3] {
    let mut out = [K::ZERO; 3];
    out[usize::from(at_one)] = raw[0].reduce();
    out[2] = raw[1].reduce();
    out
}

fn add_statistics(a: [K; 3], b: [K; 3]) -> [K; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}

/// A pair of columns with only its native coordinates. The initial rows are
/// streamed; bound columns first appear at half the native source length.
/// Once all native variables are bound, no dummy-coordinate tables are made.
struct NativeProduct<F> {
    dimension: usize,
    source: F,
    columns: [Vec<K>; 2],
    scratch: [Vec<K>; 2],
    scalar: Option<[K; 2]>,
}

impl<F: Fn(usize) -> [K; 2] + Sync> NativeProduct<F> {
    fn new(dimension: usize, source: F) -> Self {
        let scalar = (dimension == 0).then(|| source(0));
        Self {
            dimension,
            source,
            columns: std::array::from_fn(|_| Vec::new()),
            scratch: std::array::from_fn(|_| Vec::new()),
            scalar,
        }
    }

    fn constant_statistics(&self, at_one: bool) -> [K; 3] {
        let values = self
            .scalar
            .expect("native product has no remaining variables");
        let mut out = [K::ZERO; 3];
        out[usize::from(at_one)] = values[0] * values[1];
        out
    }

    fn initial_statistics(&self, point: &[K], at_one: bool) -> [K; 3] {
        if self.dimension == 0 {
            return self.constant_statistics(at_one);
        }
        const PAIRS: usize = 1024;
        let round = NativeRound::new(&point[1..self.dimension], at_one);
        let pairs = 1usize << (self.dimension - 1);
        let process = |chunk: usize| {
            let mut sums = NativeSums::new(&round);
            let first = chunk * PAIRS;
            for pair in first..(first + PAIRS).min(pairs) {
                sums.pair(pair, (self.source)(2 * pair), (self.source)(2 * pair + 1));
            }
            sums.finish()
        };
        let chunks = pairs.div_ceil(PAIRS);
        let raw = if 1usize << self.dimension >= PAR_THRESHOLD {
            (0..chunks)
                .into_par_iter()
                .map(process)
                .reduce(|| [U256::ZERO; 2], add_raw)
        } else {
            (0..chunks).map(process).fold([U256::ZERO; 2], add_raw)
        };
        native_statistics(raw, at_one)
    }

    fn bind_and_statistics(
        &mut self,
        round: usize,
        challenge: K,
        point: &[K],
        next_at_one: Option<bool>,
    ) -> [K; 3] {
        if round >= self.dimension {
            return next_at_one.map_or([K::ZERO; 3], |at_one| self.constant_statistics(at_one));
        }
        const ROWS: usize = 1024;
        let remaining = self.dimension - round - 1;
        let size = 1usize << remaining;
        for column in &mut self.scratch {
            if column.len() > size {
                column.truncate(size);
            } else if column.len() < size && size >= PAR_THRESHOLD / 2 {
                (0..size)
                    .into_par_iter()
                    .map(|_| K::ZERO)
                    .collect_into_vec(column);
            } else {
                column.resize(size, K::ZERO);
            }
        }
        let next = if remaining > 0 {
            next_at_one.map(|at_one| NativeRound::new(&point[round + 2..self.dimension], at_one))
        } else {
            None
        };
        let [out0, out1] = &mut self.scratch;
        let previous = &self.columns;
        let source = &self.source;
        let tile = |(tile, (out0, out1)): (usize, (&mut [K], &mut [K]))| {
            let base = tile * ROWS;
            if round == 0 {
                for j in 0..out0.len() {
                    let mut left = source(2 * (base + j));
                    let right = source(2 * (base + j) + 1);
                    om_field::fold_slices(&mut left, &right, challenge);
                    out0[j] = left[0];
                    out1[j] = left[1];
                }
            } else {
                let range = 2 * base..2 * (base + out0.len());
                om_field::fold_pairs(&previous[0][range.clone()], challenge, out0);
                om_field::fold_pairs(&previous[1][range], challenge, out1);
            }
            if let Some(next) = &next {
                let mut sums = NativeSums::new(next);
                for pair in 0..out0.len() / 2 {
                    sums.pair(
                        base / 2 + pair,
                        [out0[2 * pair], out1[2 * pair]],
                        [out0[2 * pair + 1], out1[2 * pair + 1]],
                    );
                }
                sums.finish()
            } else {
                [U256::ZERO; 2]
            }
        };
        let raw = if 2 * size >= PAR_THRESHOLD {
            out0.par_chunks_mut(ROWS)
                .zip(out1.par_chunks_mut(ROWS))
                .enumerate()
                .map(tile)
                .reduce(|| [U256::ZERO; 2], add_raw)
        } else {
            out0.chunks_mut(ROWS)
                .zip(out1.chunks_mut(ROWS))
                .enumerate()
                .map(tile)
                .fold([U256::ZERO; 2], add_raw)
        };
        std::mem::swap(&mut self.columns, &mut self.scratch);
        if remaining == 0 {
            self.scalar = Some([self.columns[0][0], self.columns[1][0]]);
            next_at_one.map_or([K::ZERO; 3], |at_one| self.constant_statistics(at_one))
        } else {
            next_at_one.map_or([K::ZERO; 3], |at_one| native_statistics(raw, at_one))
        }
    }

    fn terminal(&self) -> [K; 2] {
        self.scalar.expect("all native product variables are bound")
    }
}

/// Exact joint sumcheck, with each product stored on its own domain. The two
/// suffix-weighted statistics are added before recovering the omitted endpoint,
/// so the finite-point messages and global challenge point stay unchanged.
#[allow(clippy::too_many_arguments)]
fn prove_native(
    point: &[K],
    n: usize,
    a: usize,
    cycle_row: impl Fn(usize) -> [K; 2] + Sync,
    memory_row: impl Fn(usize) -> [K; 2] + Sync,
    initial: K,
    tr: &mut Transcript,
    base: &[u8],
    mut sample: impl FnMut(&mut Transcript, &[u8]) -> K,
) -> (K, SumcheckProof, Vec<K>, [K; 4]) {
    let l = point.len();
    let mut cycle = NativeProduct::new(n, cycle_row);
    let mut memory = NativeProduct::new(a, memory_row);
    if l == 0 {
        let c = cycle.terminal();
        let m = memory.terminal();
        let actual = c[0] * c[1] + m[0] * m[1];
        sumcheck_init(tr, base, actual, 0, 3);
        return (
            actual,
            SumcheckProof {
                round_msgs: Vec::new(),
            },
            Vec::new(),
            [c[0], c[1], m[0], m[1]],
        );
    }
    let inverses = eq_factor::batch_inverse_one_plus(point);
    let at_one = point[0] != K::ONE;
    let stats = add_statistics(
        cycle.initial_statistics(point, at_one),
        memory.initial_statistics(point, at_one),
    );
    let mut coefficients =
        eq_factor::reconstruct_cofactor(&stats, point[0], Some(initial), inverses[0]);
    let evaluations = eq_factor::factored_evaluations(&coefficients, point[0], K::ONE);
    let claim = evaluations[0] + evaluations[1];
    sumcheck_init(tr, base, claim, l, 3);
    let msg_label = label(base, b"/msg");
    let chal_label = label(base, b"/chal");
    let mut prefix = K::ONE;
    let mut messages = Vec::with_capacity(l);
    let mut challenges = Vec::with_capacity(l);
    for round in 0..l {
        let evaluations = eq_factor::factored_evaluations(&coefficients, point[round], prefix);
        let message = vec![evaluations[0], evaluations[2], evaluations[3]];
        tr.absorb_ks(&msg_label, &message);
        let challenge = sample(tr, &chal_label);
        messages.push(message);
        challenges.push(challenge);
        let normalized_next = eq_factor::eval_coefficients(&coefficients, challenge);
        prefix *= K::ONE + point[round] + challenge;
        let next_at_one = (round + 1 < l && prefix != K::ZERO).then(|| point[round + 1] != K::ONE);
        // Even a zero global equality prefix only suppresses statistics.
        // Every genuine source variable must still be folded for its opening.
        let stats = add_statistics(
            cycle.bind_and_statistics(round, challenge, point, next_at_one),
            memory.bind_and_statistics(round, challenge, point, next_at_one),
        );
        if round + 1 < l {
            coefficients = if prefix == K::ZERO {
                vec![K::ZERO; 3]
            } else {
                eq_factor::reconstruct_cofactor(
                    &stats,
                    point[round + 1],
                    Some(normalized_next),
                    inverses[round + 1],
                )
            };
        }
    }
    let c = cycle.terminal();
    let m = memory.terminal();
    (
        claim,
        SumcheckProof {
            round_msgs: messages,
        },
        challenges,
        [c[0], c[1], m[0], m[1]],
    )
}

fn evaluate_flags(
    is_write: &[bool],
    next_is_write: &[bool],
    initial_next: &[bool],
    point: &[K],
    n: usize,
    a: usize,
) -> [K; 3] {
    // Share the cycle Eq table and bit scan. The public-initial flag has its
    // own smaller-domain point; existing bit-MLE code handles that domain.
    let eq = eq_table(&point[..n]);
    let add = |mut sum: [K; 2], ((&first, &next), &weight): ((&bool, &bool), &K)| {
        if first {
            sum[0] += weight;
        }
        if next {
            sum[1] += weight;
        }
        sum
    };
    let pair = if is_write.len() >= 1 << 14 {
        is_write
            .par_iter()
            .zip(next_is_write)
            .zip(eq.par_iter())
            .fold(|| [K::ZERO; 2], add)
            .reduce(|| [K::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    } else {
        is_write
            .iter()
            .zip(next_is_write)
            .zip(&eq)
            .fold([K::ZERO; 2], add)
    };
    [pair[0], pair[1], mle_eval_bits(initial_next, &point[..a])]
}

fn absorb_sources(tr: &mut Transcript, base: &[u8], proof: &MaskProof) {
    tr.absorb_ks(
        &label(base, b"/sources"),
        &[
            proof.value,
            proof.is_write,
            proof.next_is_write,
            proof.initial_next_is_write,
        ],
    );
}

/// `(proof, point, weights, value, folded masks)` from [`prove_sumcheck`].
pub type FusedSumcheck = (SumcheckProof, Vec<K>, [K; 3], K, [K; 2]);

/// Run only the sum-check, returning `(proof, point, weights, value,
/// folded masks)`. `value` is the word source at `point`. The caller sends it
/// and the flag sources later, absorbs them before any challenge that depends
/// on them, and the verifier then calls [`Deferred::check`].
#[allow(clippy::too_many_arguments)]
pub fn prove_sumcheck(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<FusedSumcheck, Error> {
    prove_sumcheck_sampled(
        values,
        is_write,
        next_is_write,
        initial_memory,
        initial_next_is_write,
        gp_point,
        claims,
        tr,
        base,
        false,
        |tr, label| tr.challenge_k(label),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_sumcheck_sampled(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
    force_native: bool,
    sample: impl FnMut(&mut Transcript, &[u8]) -> K,
) -> Result<FusedSumcheck, Error> {
    let (a, l) = public_shape(initial_memory, gp_point)?;
    require(
        values.len().is_power_of_two(),
        "masked reduction: cycle domain must be a nonzero power of two",
    )?;
    let n = values.len().trailing_zeros() as usize;
    require(
        l == n.max(a),
        "masked reduction: GP point does not match source domains",
    )?;
    require(
        is_write.len() == values.len()
            && next_is_write.len() == values.len()
            && initial_next_is_write.len() == initial_memory.len(),
        "masked reduction: flag shape mismatch",
    )?;
    let weights = setup(a, l, gp_point, claims, tr, base);
    let initial = batched_claim(claims, weights);
    let cycle_row = |cycle: usize| {
        let mut mask = weights[0] + weights[1];
        if is_write[cycle] {
            mask += weights[0];
        }
        if next_is_write[cycle] {
            mask += weights[1];
        }
        [K::from_u128(values[cycle] as u128), mask]
    };
    let memory_row = |memory: usize| {
        [
            K::from_u128(initial_memory[memory] as u128),
            if initial_next_is_write[memory] {
                K::ZERO
            } else {
                weights[2]
            },
        ]
    };
    let (actual, sc, point, finals) = {
        let _span = Span::new("spice.masked.sumcheck");
        if n == a && !force_native {
            // There is no repeated domain to remove. Retain the joint scan
            // rather than paying for two same-size native passes.
            let row = |j: usize, out: &mut [K]| {
                out[..2].copy_from_slice(&cycle_row(j));
                out[2..].copy_from_slice(&memory_row(j));
            };
            let (actual, sc, point, finals) = eq_factor::prove_with_initial_claim(
                &gp_point[..l],
                &Products,
                row,
                Some(initial),
                tr,
                &label(base, b"/sc"),
            );
            (
                actual,
                sc,
                point,
                [finals[0], finals[1], finals[2], finals[3]],
            )
        } else {
            prove_native(
                &gp_point[..l],
                n,
                a,
                cycle_row,
                memory_row,
                initial,
                tr,
                &label(base, b"/sc"),
                sample,
            )
        }
    };
    require(
        actual == initial,
        "masked reduction: incorrect zero-dimensional claim",
    )?;
    // finals[1] and finals[3] are the folded mask columns; the caller can
    // compare them with the flag sources as a prover-side self-check.
    Ok((sc, point, weights, finals[0], [finals[1], finals[3]]))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn prove_sumcheck_with_challenges(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
    forced: &[K],
) -> Result<FusedSumcheck, Error> {
    require(
        forced.len() + 1 == gp_point.len(),
        "masked reduction: forced challenge shape mismatch",
    )?;
    let mut at = 0;
    prove_sumcheck_sampled(
        values,
        is_write,
        next_is_write,
        initial_memory,
        initial_next_is_write,
        gp_point,
        claims,
        tr,
        base,
        true,
        |tr, label| {
            tr.challenge_k(label);
            let result = forced[at];
            at += 1;
            result
        },
    )
}

/// The masked-reduction terminal relation, awaiting the flag sources.
#[derive(Clone, Debug)]
pub struct Deferred {
    pub point: Vec<K>,
    weights: [K; 3],
    equality: K,
    expected: K,
    public_value: K,
}

impl Deferred {
    /// Check the terminal product relation given the sources at `point`.
    pub fn check(
        &self,
        value: K,
        is_write: K,
        next_is_write: K,
        initial_next_is_write: K,
    ) -> Result<(), Error> {
        let w = self.weights;
        let cofactor = value * (w[0] + w[1] + w[0] * is_write + w[1] * next_is_write)
            + self.public_value * w[2] * (K::ONE + initial_next_is_write);
        require(
            self.equality * cofactor == self.expected,
            "masked reduction: terminal product check failed",
        )
    }
}

/// Verifier half of [`prove_sumcheck`]. Initial memory is public and is
/// evaluated here; nothing after the sum-check is absorbed.
pub fn verify_sumcheck(
    initial_memory: &[u64],
    gp_point: &[K],
    claims: &MaskClaims,
    sc: &SumcheckProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<Deferred, Error> {
    let (a, l) = public_shape(initial_memory, gp_point)?;
    require(
        sc.round_msgs.len() == l && sc.round_msgs.iter().all(|m| m.len() == 3),
        "masked reduction: malformed sumcheck shape",
    )?;
    let weights = setup(a, l, gp_point, claims, tr, base);
    let initial = batched_claim(claims, weights);
    let (point, expected) = sumcheck_verify(initial, l, 3, sc, tr, &label(base, b"/sc"))?;
    Ok(Deferred {
        public_value: crate::word_eval(initial_memory, &point[..a], 6),
        equality: eq_eval(&gp_point[..l], &point),
        point,
        weights,
        expected,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn prove(
    values: &[u64],
    is_write: &[bool],
    next_is_write: &[bool],
    initial_memory: &[u64],
    initial_next_is_write: &[bool],
    gp_point: &[K],
    claims: &MaskClaims,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(MaskProof, SourceClaims), Error> {
    let (sc, point, weights, value, masks) = prove_sumcheck(
        values,
        is_write,
        next_is_write,
        initial_memory,
        initial_next_is_write,
        gp_point,
        claims,
        tr,
        base,
    )?;
    let (a, _) = public_shape(initial_memory, gp_point)?;
    let n = values.len().trailing_zeros() as usize;
    let _span = Span::new("spice.masked.claims");
    let [first, next, initial_next] =
        evaluate_flags(is_write, next_is_write, initial_next_is_write, &point, n, a);
    require(
        masks[0] == weights[0] + weights[1] + weights[0] * first + weights[1] * next
            && masks[1] == weights[2] * (K::ONE + initial_next),
        "masked reduction: linear source binding failed",
    )?;
    let proof = MaskProof {
        sc,
        value,
        is_write: first,
        next_is_write: next,
        initial_next_is_write: initial_next,
    };
    let sources = SourceClaims {
        point,
        value: proof.value,
        is_write: first,
        next_is_write: next,
        initial_next_is_write: initial_next,
    };
    absorb_sources(tr, base, &proof);
    Ok((proof, sources))
}

/// The outer verifier MUST authenticate the returned values at its statement's
/// cycle/initial prefixes. No source dimensions or evaluation points are taken
/// from the prover. Initial memory is public and evaluated independently here.
pub fn verify(
    initial_memory: &[u64],
    gp_point: &[K],
    claims: &MaskClaims,
    proof: &MaskProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<SourceClaims, Error> {
    let deferred = verify_sumcheck(initial_memory, gp_point, claims, &proof.sc, tr, base)?;
    deferred.check(
        proof.value,
        proof.is_write,
        proof.next_is_write,
        proof.initial_next_is_write,
    )?;
    absorb_sources(tr, base, proof);
    Ok(SourceClaims {
        point: deferred.point,
        value: proof.value,
        is_write: proof.is_write,
        next_is_write: proof.next_is_write,
        initial_next_is_write: proof.initial_next_is_write,
    })
}

#[cfg(test)]
#[path = "masked_reduction_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "masked_reduction_native_tests.rs"]
mod native_tests;
