//! Spice ordering through Flock's packed Boolean zerocheck, unchanged.
//!
//! Donor: succinctlabs/flock at daa0cf426563b01f9e744f6265ebae37c97afcb7,
//! `crates/flock-core/src/zerocheck.rs` and its optimized skip/tail kernels.
//! For large traces the cycle index occupies the low coordinates and the
//! gate axis is rounded only to the comparison width's next power of two.
//! Flock's six skipped cycle coordinates give a general Lagrange functional;
//! its weights are returned explicitly for source evaluation and Mosaic.
//! Small traces retain the original gate-low layout and ordinary MLE claims.
//!
//! This module does not authenticate its terminal source values by itself.
//! The caller must commit predecessor/carry bitplanes before entering, then
//! authenticate ALL returned claims. Flock's non-grinding schedule is used.

pub use flock_core::zerocheck::{ZerocheckClaim as FlockClaim, ZerocheckProof as FlockProof};
use flock_core::{
    challenger::Challenger,
    field::F128,
    zerocheck::{
        K_SKIP, PaddingSpec, ZerocheckClaim, ZerocheckProof, multilinear::lagrange_weights_naive,
        prove_packed_padded, verify as verify_flock_zerocheck,
    },
};
use om_field::{K, U256};
use om_sumcheck::{PackedBits, eq_table};
use om_transcript::{Transcript, profiling::Span};
use rayon::prelude::*;

use crate::order::{OrderError, width};

const MIN_CYCLE_BITS: usize = 7;
const GATE_SLOTS: usize = 1 << K_SKIP;
const COMPACT_MIN_CYCLE_BITS: usize = K_SKIP + MIN_CYCLE_BITS;

fn compact(n: usize) -> bool {
    n >= COMPACT_MIN_CYCLE_BITS
}

fn gate_log(w: usize) -> usize {
    w.next_power_of_two().trailing_zeros() as usize
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderProof {
    pub zerocheck: ZerocheckProof,
    pub previous_ab: Vec<K>,
    pub carries_ab: Vec<K>,
    pub carries_c: Vec<K>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderClaims {
    pub point_ab: Vec<K>,
    pub point_c: Vec<K>,
    pub previous_ab: Vec<K>,
    pub carries_ab: Vec<K>,
    pub carries_c: Vec<K>,
    /// Replaces Eq on the low six cycle coordinates in the compact layout.
    /// The corresponding entries of point_ab/point_c are placeholders.
    pub prefix_weights: Option<Vec<K>>,
}

fn from_flock(x: F128) -> K {
    K::from_u128(x.lo as u128 | ((x.hi as u128) << 64))
}

fn to_flock(x: K) -> F128 {
    F128 {
        lo: x.value() as u64,
        hi: (x.value() >> 64) as u64,
    }
}

fn label(base: &[u8], suffix: &[u8]) -> Vec<u8> {
    [base, suffix].concat()
}

/// The actual donor Challenger seam, with explicit OM hash-chain framing.
/// Owning the small transcript also satisfies the donor's fork interface;
/// this particular zerocheck does not fork or request a hash-specific sampler.
struct OmChallenger {
    tr: Transcript,
}

impl Challenger for OmChallenger {
    fn observe_label(&mut self, bytes: &[u8]) {
        self.tr.absorb_bytes(b"spice/flock/label", bytes);
    }
    fn observe_f128(&mut self, value: F128) {
        self.tr.absorb_k(b"spice/flock/field", from_flock(value));
    }
    fn observe_f128_slice(&mut self, values: &[F128]) {
        let values: Vec<K> = values.iter().copied().map(from_flock).collect();
        self.tr.absorb_ks(b"spice/flock/field-slice", &values);
    }
    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.tr.absorb_bytes(b"spice/flock/bytes", bytes);
    }
    fn sample_f128(&mut self) -> F128 {
        to_flock(self.tr.challenge_k(b"spice/flock/challenge"))
    }
    fn fork_from_seed(&self, seed: [F128; 2], domain: &'static [u8]) -> Self {
        let mut tr = Transcript::new(b"spice/flock/fork/v1", self.tr.hasher());
        tr.absorb_bytes(b"domain", domain);
        tr.absorb_ks(b"seed", &seed.map(from_flock));
        Self { tr }
    }
    // Never silently inherit the trait's non-cryptographic PoW defaults if a
    // future caller accidentally chooses a grinding entry point.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        panic!("Spice Flock adapter supports only the non-grinding schedule")
    }
    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        false
    }
}

fn setup(n: usize, a: usize, tr: &mut Transcript, base: &[u8]) -> Result<usize, OrderError> {
    let w = width(n, a)?;
    if compact(n) {
        let g = gate_log(w);
        tr.absorb_bytes(
            &label(base, b"/protocol"),
            b"spice/order/flock-boolean/cycle-low/v2",
        );
        let dimensions: Vec<u8> = [n, a, w, n, K_SKIP, g]
            .into_iter()
            .flat_map(|v| (v as u64).to_le_bytes())
            .collect();
        tr.absorb_bytes(&label(base, b"/dimensions"), &dimensions);
        return Ok(n + g - K_SKIP);
    }
    let cycle_bits = n.max(MIN_CYCLE_BITS);
    tr.absorb_bytes(&label(base, b"/protocol"), b"spice/order/flock-boolean/v1");
    let dimensions: Vec<u8> = [n, a, w, cycle_bits, K_SKIP]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(&label(base, b"/dimensions"), &dimensions);
    Ok(cycle_bits)
}

/// Each cycle contributes one native 64-bit gate row to each Boolean column.
/// Gate b is `(p_b+c_b)(1+i_b+c_b)=c_b+c_(b+1)`. No field-element
/// expansion, random gate weighting, or witness-content merging is performed.
fn packed_rows(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
) -> Result<([Vec<u8>; 3], usize), OrderError> {
    let w = width(n, a)?;
    if previous.len() != 1usize << n || carries.len() != previous.len() {
        return Err(OrderError(
            "order/flock: malformed source word lengths".into(),
        ));
    }
    let cycles = 1usize << n.max(MIN_CYCLE_BITS);
    let mut packed = std::array::from_fn(|_| vec![0u8; cycles * (GATE_SLOTS / 8)]);
    let [a_rows, b_rows, c_rows] = &mut packed;
    let mask = (1u64 << w) - 1;
    let timestamp_base = 1u64 << (w - 1);
    a_rows[..previous.len() * 8]
        .par_chunks_mut(8)
        .zip(b_rows[..previous.len() * 8].par_chunks_mut(8))
        .zip(c_rows[..previous.len() * 8].par_chunks_mut(8))
        .enumerate()
        .try_for_each(|(j, ((aa, bb), cc))| {
            let p = previous[j] as u64;
            let c = carries[j] as u64;
            let time = timestamp_base + j as u64;
            if p >= time || c >> (w - 1) != 0 {
                return Err(OrderError(
                    "order/flock: invalid predecessor/carry word".into(),
                ));
            }
            let cin = 1 | (c << 1);
            let av = (p ^ cin) & mask;
            let bv = (!time ^ cin) & mask;
            let cv = (cin ^ c) & mask;
            if av & bv != cv {
                return Err(OrderError("order/flock: noncanonical carry witness".into()));
            }
            aa.copy_from_slice(&av.to_le_bytes());
            bb.copy_from_slice(&bv.to_le_bytes());
            cc.copy_from_slice(&cv.to_le_bytes());
            Ok(())
        })?;
    Ok((packed, w))
}

/// Gate-major packed Boolean columns: bit j + 2^n*b is gate b at cycle j.
/// Unused gates form one declared zero suffix rather than gaps in every row.
fn packed_rows_compact(
    previous: &[PackedBits],
    carries: &[PackedBits],
    n: usize,
    a: usize,
) -> Result<([Vec<u8>; 3], usize), OrderError> {
    let w = width(n, a)?;
    let cycles = 1usize << n;
    if !compact(n)
        || previous.len() != w
        || carries.len() != w - 1
        || previous.iter().chain(carries).any(|p| p.len() != cycles)
    {
        return Err(OrderError(
            "order/flock: malformed packed source planes".into(),
        ));
    }
    let bytes_per_gate = cycles / 8;
    let mut packed = std::array::from_fn(|_| vec![0u8; bytes_per_gate * w.next_power_of_two()]);
    let [a_rows, b_rows, c_rows] = &mut packed;
    const TIME_LOW: [u64; 6] = [
        0xaaaa_aaaa_aaaa_aaaa,
        0xcccc_cccc_cccc_cccc,
        0xf0f0_f0f0_f0f0_f0f0,
        0xff00_ff00_ff00_ff00,
        0xffff_0000_ffff_0000,
        0xffff_ffff_0000_0000,
    ];
    a_rows
        .par_chunks_mut(bytes_per_gate)
        .zip(b_rows.par_chunks_mut(bytes_per_gate))
        .zip(c_rows.par_chunks_mut(bytes_per_gate))
        .enumerate()
        .try_for_each(|(bit, ((aa, bb), cc))| {
            if bit >= w {
                return Ok(());
            }
            for (word, ((aa, bb), cc)) in aa
                .chunks_exact_mut(8)
                .zip(bb.chunks_exact_mut(8))
                .zip(cc.chunks_exact_mut(8))
                .enumerate()
            {
                let cin = if bit == 0 {
                    u64::MAX
                } else {
                    carries[bit - 1].words()[word]
                };
                let cout = if bit + 1 == w {
                    0
                } else {
                    carries[bit].words()[word]
                };
                let time = if bit + 1 == w {
                    u64::MAX
                } else if bit >= n {
                    0
                } else if bit < 6 {
                    TIME_LOW[bit]
                } else {
                    0u64.wrapping_sub(((word >> (bit - 6)) & 1) as u64)
                };
                let av = previous[bit].words()[word] ^ cin;
                let bv = !time ^ cin;
                let cv = cin ^ cout;
                if av & bv != cv {
                    return Err(OrderError(
                        "order/flock: noncanonical packed carry witness".into(),
                    ));
                }
                aa.copy_from_slice(&av.to_le_bytes());
                bb.copy_from_slice(&bv.to_le_bytes());
                cc.copy_from_slice(&cv.to_le_bytes());
            }
            Ok(())
        })?;
    Ok((packed, w))
}

/// Evaluate the committed word bitplanes at two points with shared Eq tables.
/// The native-word masked XOR loop has no per-row field multiplication or
/// temporary field-valued bitplanes.
fn evaluate_sources(
    previous: &[u32],
    carries: &[u32],
    point_ab: &[K],
    point_c: &[K],
    w: usize,
) -> (Vec<K>, Vec<K>, Vec<K>) {
    let eq_ab = eq_table(point_ab);
    let eq_c = eq_table(point_c);
    // Maximum w=32: 32 predecessor bits plus 2*31 internal carry bits.
    let sum = previous
        .par_chunks(2048)
        .zip(carries.par_chunks(2048))
        .zip(eq_ab.par_chunks(2048).zip(eq_c.par_chunks(2048)))
        .map(|((previous, carries), (eq_ab, eq_c))| {
            let mut sum = [K::ZERO; 94];
            for ((&p, &c), (&ab, &at_c)) in previous.iter().zip(carries).zip(eq_ab.iter().zip(eq_c))
            {
                for b in 0..w {
                    let mask = 0u128.wrapping_sub(((p >> b) & 1) as u128);
                    sum[b] += K::from_u128(ab.value() & mask);
                }
                for b in 0..w - 1 {
                    let mask = 0u128.wrapping_sub(((c >> b) & 1) as u128);
                    sum[w + b] += K::from_u128(ab.value() & mask);
                    sum[2 * w - 1 + b] += K::from_u128(at_c.value() & mask);
                }
            }
            sum
        })
        .reduce(
            || [K::ZERO; 94],
            |mut left, right| {
                for i in 0..3 * w - 2 {
                    left[i] += right[i];
                }
                left
            },
        );
    (
        sum[..w].to_vec(),
        sum[w..2 * w - 1].to_vec(),
        sum[2 * w - 1..3 * w - 2].to_vec(),
    )
}

fn time_bit(point: &[K], w: usize, b: usize) -> K {
    if b + 1 == w {
        K::ONE
    } else {
        point.get(b).copied().unwrap_or(K::ZERO)
    }
}

fn carry_bit(carry: &[K], w: usize, b: usize) -> K {
    if b == 0 {
        K::ONE
    } else if b == w {
        K::ZERO
    } else {
        carry[b - 1]
    }
}

fn padding_mask(point: &[K], n: usize) -> K {
    point[n..].iter().fold(K::ONE, |v, &r| v * (K::ONE + r))
}

fn check_terminals(
    n: usize,
    w: usize,
    claim: &ZerocheckClaim,
    previous_ab: &[K],
    carries_ab: &[K],
    carries_c: &[K],
) -> Result<OrderClaims, OrderError> {
    let rounds = if compact(n) {
        n + gate_log(w) - K_SKIP
    } else {
        n.max(MIN_CYCLE_BITS)
    };
    if claim.mlv_challenges.len() != rounds
        || claim.r_rest.len() != rounds
        || previous_ab.len() != w
        || carries_ab.len() != w - 1
        || carries_c.len() != w - 1
    {
        return Err(OrderError("order/flock: malformed terminal claims".into()));
    }
    let all_ab: Vec<K> = claim
        .mlv_challenges
        .iter()
        .copied()
        .map(from_flock)
        .collect();
    let all_c: Vec<K> = claim.r_rest.iter().copied().map(from_flock).collect();
    let prefix_weights = compact(n).then(|| {
        lagrange_weights_naive(K_SKIP, claim.z)
            .into_iter()
            .map(from_flock)
            .collect::<Vec<_>>()
    });
    let (point_ab, point_c, weights_ab, weights_c, mask_ab, mask_c) = if compact(n) {
        let rest = n - K_SKIP;
        let point = |tail: &[K]| {
            let mut point = vec![K::ZERO; K_SKIP];
            point.extend_from_slice(&tail[..rest]);
            point
        };
        (
            point(&all_ab),
            point(&all_c),
            eq_table(&all_ab[rest..]),
            eq_table(&all_c[rest..]),
            K::ONE,
            K::ONE,
        )
    } else {
        let weights: Vec<K> = lagrange_weights_naive(K_SKIP, claim.z)
            .into_iter()
            .map(from_flock)
            .collect();
        (
            all_ab[..n].to_vec(),
            all_c[..n].to_vec(),
            weights.clone(),
            weights,
            padding_mask(&all_ab, n),
            padding_mask(&all_c, n),
        )
    };
    // A skipped cycle block is a general 64-weight functional, not an Eq
    // point. Public low timestamp bits therefore use its binary moments.
    let moments: [K; K_SKIP] = std::array::from_fn(|bit| {
        prefix_weights.as_ref().map_or(K::ZERO, |weights| {
            weights
                .iter()
                .enumerate()
                .filter(|(j, _)| (j >> bit) & 1 != 0)
                .fold(K::ZERO, |sum, (_, &weight)| sum + weight)
        })
    });
    let mut aa = U256::ZERO;
    let mut bb = U256::ZERO;
    let mut cc = U256::ZERO;
    for bit in 0..w {
        let cab = carry_bit(carries_ab, w, bit);
        let time = if prefix_weights.is_some() && bit < K_SKIP {
            moments[bit]
        } else {
            time_bit(&point_ab, w, bit)
        };
        aa += weights_ab[bit].clmul(previous_ab[bit] + cab);
        bb += weights_ab[bit].clmul(K::ONE + time + cab);
        cc += weights_c[bit].clmul(carry_bit(carries_c, w, bit) + carry_bit(carries_c, w, bit + 1));
    }
    if aa.reduce() * mask_ab != from_flock(claim.a_eval)
        || bb.reduce() * mask_ab != from_flock(claim.b_eval)
        || cc.reduce() * mask_c != from_flock(claim.c_eval)
    {
        return Err(OrderError(
            "order/flock: terminal source binding failed".into(),
        ));
    }
    Ok(OrderClaims {
        point_ab,
        point_c,
        previous_ab: previous_ab.to_vec(),
        carries_ab: carries_ab.to_vec(),
        carries_c: carries_c.to_vec(),
        prefix_weights,
    })
}

fn absorb_sources(tr: &mut Transcript, base: &[u8], proof: &OrderProof) {
    tr.absorb_ks(&label(base, b"/previous-ab"), &proof.previous_ab);
    tr.absorb_ks(&label(base, b"/carries-ab"), &proof.carries_ab);
    tr.absorb_ks(&label(base, b"/carries-c"), &proof.carries_c);
}

/// The ordering zerocheck alone. The caller evaluates the sources at the
/// returned points later (see [`terminal_points`] and [`check_sources`]) and
/// must absorb them before any challenge that depends on them.
fn prove_zerocheck_legacy(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(ZerocheckProof, ZerocheckClaim), OrderError> {
    let ([a_rows, b_rows, c_rows], w) = {
        let _span = Span::new("spice.order.pack");
        packed_rows(previous, carries, n, a)?
    };
    let padded_n = setup(n, a, tr, base)?;
    let m = K_SKIP + padded_n;
    let mut challenger = OmChallenger { tr: tr.clone() };
    // One useful gate prefix per cycle. All remaining gate positions and any
    // padded cycles are actually zero, as required by the donor's support hint.
    let padding = PaddingSpec::uniform(K_SKIP, w, previous.len());
    let (zerocheck, terminal) = {
        let _span = Span::new("spice.order.zerocheck");
        prove_packed_padded(&a_rows, &b_rows, &c_rows, m, &padding, &mut challenger)
    };
    *tr = challenger.tr;
    Ok((zerocheck, terminal))
}

/// Run the ordering proof from native words. Large traces transpose the
/// sources once; callers that already committed planes should use the packed
/// entry point to reuse that work.
pub fn prove_zerocheck(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(ZerocheckProof, ZerocheckClaim), OrderError> {
    if !compact(n) {
        return prove_zerocheck_legacy(previous, carries, n, a, tr, base);
    }
    let w = width(n, a)?;
    let (previous_planes, carry_planes) = {
        let _span = Span::new("spice.order.pack");
        (
            crate::pack_columns(previous, w),
            crate::pack_columns(carries, w - 1),
        )
    };
    prove_zerocheck_packed(
        previous,
        carries,
        &previous_planes,
        &carry_planes,
        n,
        a,
        tr,
        base,
    )
}

/// Reuse the committed predecessor and carry planes. The source planes are
/// authoritative; callers authenticate those same columns downstream. Raw
/// words supply the legacy fallback and must describe the same trace.
#[allow(clippy::too_many_arguments)]
pub fn prove_zerocheck_packed(
    previous: &[u32],
    carries: &[u32],
    previous_planes: &[PackedBits],
    carry_planes: &[PackedBits],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(ZerocheckProof, ZerocheckClaim), OrderError> {
    width(n, a)?;
    if !compact(n) {
        return prove_zerocheck_legacy(previous, carries, n, a, tr, base);
    }
    if previous.len() != 1usize << n || carries.len() != previous.len() {
        return Err(OrderError(
            "order/flock: malformed source word lengths".into(),
        ));
    }
    let ([a_rows, b_rows, c_rows], w) = {
        let _span = Span::new("spice.order.pack");
        packed_rows_compact(previous_planes, carry_planes, n, a)?
    };
    let m = K_SKIP + setup(n, a, tr, base)?;
    let mut challenger = OmChallenger { tr: tr.clone() };
    // Gate-major planes make every unused gate a contiguous zero suffix.
    let padding = PaddingSpec::uniform(m, w * previous.len(), 1);
    let (proof, terminal) = {
        let _span = Span::new("spice.order.zerocheck");
        prove_packed_padded(&a_rows, &b_rows, &c_rows, m, &padding, &mut challenger)
    };
    *tr = challenger.tr;
    Ok((proof, terminal))
}

/// Verifier half of [`prove_zerocheck`].
pub fn verify_zerocheck_with(
    n: usize,
    a: usize,
    proof: &ZerocheckProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<ZerocheckClaim, OrderError> {
    let padded_n = setup(n, a, tr, base)?;
    let mut challenger = OmChallenger { tr: tr.clone() };
    let terminal = verify_flock_zerocheck(K_SKIP + padded_n, proof, &mut challenger)
        .map_err(|e| OrderError(format!("order/flock: {e:?}")))?;
    *tr = challenger.tr;
    Ok(terminal)
}

/// `(skip variables, multilinear rounds, fixed equality coordinates)` of the
/// Flock zerocheck for `2^n` cycles. The fixed coordinates are the seven
/// public inner challenges that `flock_core::zerocheck::verify` places at
/// `r[K_SKIP..K_SKIP + 7]`.
pub fn zerocheck_shape(n: usize, a: usize) -> (usize, usize, usize) {
    let rounds = if compact(n) {
        n + gate_log(n.max(a) + 1) - K_SKIP
    } else {
        n.max(MIN_CYCLE_BITS)
    };
    (K_SKIP, rounds, MIN_CYCLE_BITS)
}

/// The A/B and C source bases. In the compact layout their first six
/// coordinates are placeholders, replaced by terminal_prefix_weights.
pub fn terminal_points(n: usize, _a: usize, claim: &ZerocheckClaim) -> (Vec<K>, Vec<K>) {
    let point = |xs: &[F128]| {
        let mut out = if compact(n) {
            vec![K::ZERO; K_SKIP]
        } else {
            Vec::new()
        };
        let rest = if compact(n) { n - K_SKIP } else { n };
        out.extend(xs.iter().take(rest).copied().map(from_flock));
        out
    };
    (point(&claim.mlv_challenges), point(&claim.r_rest))
}

/// Flock's skipped-cycle Lagrange weights, or None for the legacy layout.
/// These are not generally the Eq table of any six-coordinate field point.
pub fn terminal_prefix_weights(n: usize, _a: usize, claim: &ZerocheckClaim) -> Option<Vec<K>> {
    compact(n).then(|| {
        lagrange_weights_naive(K_SKIP, claim.z)
            .into_iter()
            .map(from_flock)
            .collect()
    })
}

/// Check the zerocheck's terminal values against source evaluations.
pub fn check_sources(
    n: usize,
    a: usize,
    claim: &ZerocheckClaim,
    previous_ab: &[K],
    carries_ab: &[K],
    carries_c: &[K],
) -> Result<OrderClaims, OrderError> {
    check_terminals(n, width(n, a)?, claim, previous_ab, carries_ab, carries_c)
}

pub fn prove(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(OrderProof, OrderClaims), OrderError> {
    let w = width(n, a)?;
    let (previous_planes, carry_planes) = {
        let _span = Span::new("spice.order.pack");
        (
            crate::pack_columns(previous, w),
            crate::pack_columns(carries, w - 1),
        )
    };
    prove_packed(
        previous,
        carries,
        &previous_planes,
        &carry_planes,
        n,
        a,
        tr,
        base,
    )
}

/// Full ordering proof with authenticated-column preparation shared with RAM.
#[allow(clippy::too_many_arguments)]
pub fn prove_packed(
    previous: &[u32],
    carries: &[u32],
    previous_planes: &[PackedBits],
    carry_planes: &[PackedBits],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(OrderProof, OrderClaims), OrderError> {
    let (zerocheck, terminal) = prove_zerocheck_packed(
        previous,
        carries,
        previous_planes,
        carry_planes,
        n,
        a,
        tr,
        base,
    )?;
    let w = width(n, a)?;
    let _span = Span::new("spice.order.claims");
    let (point_ab, point_c) = terminal_points(n, a, &terminal);
    let prefix_weights = terminal_prefix_weights(n, a, &terminal);
    let (previous_ab, carries_ab, carries_c) = if compact(n) {
        let mut evaluations = crate::planes::evaluate_with_prefixes(
            &[&point_ab, &point_c],
            &[prefix_weights.as_deref(), prefix_weights.as_deref()],
            &[
                crate::planes::Request {
                    planes: previous_planes.iter().collect(),
                    point: 0,
                },
                crate::planes::Request {
                    planes: carry_planes.iter().collect(),
                    point: 0,
                },
                crate::planes::Request {
                    planes: carry_planes.iter().collect(),
                    point: 1,
                },
            ],
        )
        .into_iter();
        (
            evaluations.next().unwrap(),
            evaluations.next().unwrap(),
            evaluations.next().unwrap(),
        )
    } else {
        evaluate_sources(previous, carries, &point_ab, &point_c, w)
    };
    let claims = check_terminals(n, w, &terminal, &previous_ab, &carries_ab, &carries_c)?;
    let proof = OrderProof {
        zerocheck,
        previous_ab,
        carries_ab,
        carries_c,
    };
    absorb_sources(tr, base, &proof);
    Ok((proof, claims))
}

pub fn verify(
    n: usize,
    a: usize,
    proof: &OrderProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<OrderClaims, OrderError> {
    let w = width(n, a)?;
    if proof.previous_ab.len() != w
        || proof.carries_ab.len() != w - 1
        || proof.carries_c.len() != w - 1
    {
        return Err(OrderError(
            "order/flock: malformed source evaluations".into(),
        ));
    }
    let terminal = verify_zerocheck_with(n, a, &proof.zerocheck, tr, base)?;
    let claims = check_terminals(
        n,
        w,
        &terminal,
        &proof.previous_ab,
        &proof.carries_ab,
        &proof.carries_c,
    )?;
    absorb_sources(tr, base, proof);
    Ok(claims)
}

#[cfg(test)]
#[path = "order_flock_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "order_flock_compact_tests.rs"]
mod compact_tests;
