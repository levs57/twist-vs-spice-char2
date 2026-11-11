//! Typed single-value RAM events: a read observes `value`, a write replaces it.
//! Masked predecessor/successor values are authenticated by a shared product
//! sumcheck; public memory boundaries have the same contract as BinaryTwist.
//! Ordering retains the existing Flock proof and committed carry bitplanes.

use lean_grandproduct::{GrandProductProof, LeafClaims, Leaves};
use om_field::{K, mul_pairs};
use om_memcheck::onehot::EvalClaim;
use om_mosaic::{BatchProof, batch_prove_packed, batch_verify, compute_layout};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, Commitment};
use om_sumcheck::{PackedBits, SumcheckProof};
use om_transcript::{
    Transcript,
    profiling::{self, Span},
};
use std::collections::BTreeMap;

use crate::{
    ComponentSummary, Config, Error, OrderingMode, Schema, Statement, Summary, combine_bits, dims,
    masked_reduction::{self, MaskClaims},
    order, order_flock, pack_columns, pack_values, planes, require, validate, word_claim,
    word_eval,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessTrace {
    pub initial: Vec<u64>,
    pub addresses: Vec<u32>,
    /// A read's observed value or a write's replacement value.
    pub values: Vec<u64>,
    pub is_write: Vec<bool>,
}

/// Protocol options for typed Spice. The default is the original setting:
/// one fingerprint and public final memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Independent fingerprint challenge sets `(gamma, alpha, beta, delta)`,
    /// all drawn after the commitment: 1 or 2. Two sets square the
    /// fingerprint collision term; both share one batched GKR.
    pub fingerprints: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { fingerprints: 1 }
    }
}

impl Options {
    /// The `spice-typed-2fp` method: two independent fingerprints.
    pub const TWO_FINGERPRINTS: Self = Self { fingerprints: 2 };

    fn validate(&self) -> Result<(), Error> {
        require(
            matches!(self.fingerprints, 1 | 2),
            "typed Spice supports one or two fingerprints",
        )
    }
    fn log_fingerprints(&self) -> usize {
        self.fingerprints.trailing_zeros() as usize
    }
    fn absorb(&self, tr: &mut Transcript) {
        tr.absorb_bytes(
            b"spice/typed/options",
            &(self.fingerprints as u64).to_le_bytes(),
        );
    }
}

/// One fingerprint challenge set: shift, address, value and write-tag
/// coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fingerprint {
    gamma: K,
    alpha: K,
    beta: K,
    delta: K,
}

fn fingerprints(tr: &mut Transcript, options: &Options) -> Vec<Fingerprint> {
    // Set 0 keeps the original labels; set 1 uses distinct labels. Every set
    // is drawn after the commitment has been absorbed.
    const LABELS: [[&[u8]; 4]; 2] = [
        [
            b"spice/typed/fingerprint/shift",
            b"spice/typed/fingerprint/address",
            b"spice/typed/fingerprint/value",
            b"spice/typed/fingerprint/write",
        ],
        [
            b"spice/typed/fingerprint2/shift",
            b"spice/typed/fingerprint2/address",
            b"spice/typed/fingerprint2/value",
            b"spice/typed/fingerprint2/write",
        ],
    ];
    LABELS[..options.fingerprints]
        .iter()
        .map(|[g, a, b, d]| Fingerprint {
            gamma: tr.challenge_k(g),
            alpha: tr.challenge_k(a),
            beta: tr.challenge_k(b),
            delta: tr.challenge_k(d),
        })
        .collect()
}

/// The two multiset products (left, right) of every fingerprint set.
///
/// With one set these are the GKR roots. With two sets the trees hold
/// `[set 0 | set 1]`, so the first GKR layer's children are the per-set
/// products (binary first layer) or their halves (radix-four first layer).
/// The GKR binds these children exactly as it binds every other layer.
fn fingerprint_products(
    proof: &GrandProductProof,
    options: &Options,
) -> Result<Vec<[K; 2]>, Error> {
    if options.fingerprints == 1 {
        return Ok(vec![proof.roots]);
    }
    let first = proof
        .layers
        .first()
        .ok_or_else(|| Error("typed permutation has no layers".into()))?;
    let [left, right] = &first.children;
    match left.len() {
        2 if right.len() == 2 => Ok((0..2).map(|f| [left[f], right[f]]).collect()),
        4 if right.len() == 4 => Ok((0..2)
            .map(|f| {
                [
                    left[2 * f] * left[2 * f + 1],
                    right[2 * f] * right[2 * f + 1],
                ]
            })
            .collect()),
        _ => Err(Error("malformed typed permutation first layer".into())),
    }
}

fn check_products(proof: &GrandProductProof, options: &Options) -> Result<(), Error> {
    for [left, right] in fingerprint_products(proof, options)? {
        require(left == right, "typed RAM permutation products disagree")?;
    }
    Ok(())
}

/// Every committed-plane evaluation, sent in one message after the GKR, the
/// masked-reduction sum-check and the ordering zerocheck have fixed their
/// points. One fused pass over the bitplanes computes all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evaluations {
    /// At the GKR leaf point's cycle prefix.
    pub address: Vec<K>,
    pub previous: Vec<K>,
    /// `[is_write, next_is_write]`.
    pub flags: [K; 2],
    /// At the GKR leaf point's memory prefix.
    pub final_previous: Vec<K>,
    pub initial_next_is_write: K,
    /// At the masked-reduction point: the word value (from the sum-check's
    /// last round), `[is_write, next_is_write]` and the initial flag.
    pub mask_value: K,
    pub mask_flags: [K; 2],
    pub mask_initial_next_is_write: K,
    /// At Flock's A/B point and C point.
    pub previous_ab: Vec<K>,
    pub carries_ab: Vec<K>,
    pub carries_c: Vec<K>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    pub permutation: GrandProductProof,
    /// Masked-value claims at the GKR leaf point, sent before the sum-check.
    pub masked: MaskClaims,
    pub masks: SumcheckProof,
    pub order: order_flock::FlockProof,
    pub evaluations: Evaluations,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    pub proof: Proof,
    pub metrics: Vec<(&'static str, std::time::Duration)>,
    pub summary: Summary,
}

/// Committed bitplane groups: name prefix, plane count and plane log size.
///
/// Each group is split into aligned power-of-two chunks (21 = 16 + 4 + 1).
/// A chunk of `2^k` planes is one Mosaic component of log size
/// `plane_log + k`, with plane `i` at offset `i * 2^plane_log`, so the plane
/// index occupies the high `k` coordinates. The committed bits are the same
/// bitplanes as before, only grouped.
fn plane_groups(s: &Statement) -> [(&'static str, usize, usize); 5] {
    let (n, a, _, b) = dims(s);
    [
        ("input/address", a, n),
        ("input/previous", b, n),
        ("order/carry", b - 1, n),
        ("boundary/previous", b, a),
        // Plane 0 is is_write and plane 1 is next_is_write.
        ("input/flags", 2, n),
    ]
}

/// Aligned power-of-two chunks `(first plane, log chunk size)`, largest first.
fn plane_chunks(count: usize) -> Vec<(usize, usize)> {
    let mut start = 0;
    (0..usize::BITS as usize)
        .rev()
        .filter(|&k| (count >> k) & 1 == 1)
        .map(|k| {
            let chunk = (start, k);
            start += 1 << k;
            chunk
        })
        .collect()
}

fn chunk_name(prefix: &str, start: usize, log_len: usize) -> String {
    format!("{prefix}/{start}+{}", 1usize << log_len)
}

/// Combination coordinates drawn for every plane chunk (chunks have at most
/// 32 planes because predecessor words have at most 32 bits).
const RHO_LEN: usize = 5;

fn schema(s: &Statement) -> Schema {
    let (n, a, _, _) = dims(s);
    let mut components = vec![ComponentSummary {
        name: "input/value".into(),
        log_size: n + s.config.word_log_bits,
    }];
    for (prefix, count, log_size) in plane_groups(s) {
        components.extend(
            plane_chunks(count)
                .into_iter()
                .map(|(start, k)| ComponentSummary {
                    name: chunk_name(prefix, start, k),
                    log_size: log_size + k,
                }),
        );
    }
    components.push(ComponentSummary {
        name: "initial/next_is_write".into(),
        log_size: a,
    });
    let ids: BTreeMap<_, _> = components
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.clone(), i as i64))
        .collect();
    let layout = compute_layout(
        &components
            .iter()
            .enumerate()
            .map(|(i, c)| (i as i64, c.log_size))
            .collect::<Vec<_>>(),
        14,
    );
    let pcs = BitPCS::new(make_params(
        layout.ell - 7,
        &s.config.profile,
        s.config.log_rate,
    ));
    Schema {
        layout,
        ids,
        components,
        pcs,
    }
}

/// Write bitplane `planes[i]` of group `prefix` into its chunk component.
fn place_planes(g: &mut PackedBits, schema: &Schema, prefix: &str, planes: &[PackedBits]) {
    let placements = schema.layout.by_id();
    for (start, k) in plane_chunks(planes.len()) {
        let placement = placements[&schema.ids[&chunk_name(prefix, start, k)]];
        let plane_len = 1usize << (placement.log_size - k);
        for (i, plane) in planes[start..start + (1 << k)].iter().enumerate() {
            assert_eq!(plane.len(), plane_len, "bitplane length");
            let offset = placement.offset + i * plane_len;
            let words = g.words_mut();
            if offset % 64 == 0 && plane_len >= 64 {
                words[offset / 64..offset / 64 + plane.words().len()]
                    .copy_from_slice(plane.words());
            } else {
                for bit in plane.iter_ones() {
                    words[(offset + bit) / 64] |= 1u64 << ((offset + bit) % 64);
                }
            }
        }
    }
}

/// A claim before the chunk combination challenge is drawn.
enum Pending {
    /// All planes of one group evaluated at one point.
    Planes {
        prefix: &'static str,
        point: Vec<K>,
        values: Vec<K>,
    },
    Direct(EvalClaim),
}

fn group(prefix: &'static str, point: &[K], values: &[K]) -> Pending {
    Pending::Planes {
        prefix,
        point: point.to_vec(),
        values: values.to_vec(),
    }
}

/// Replace each group of per-plane values by one claim per chunk:
/// `sum_i eq(rho[..k], i) * value_i` at `point || rho[..k]`. `rho` must be
/// drawn after every per-plane value has been absorbed.
fn finish_claims(pending: Vec<Pending>, rho: &[K]) -> Vec<EvalClaim> {
    pending
        .into_iter()
        .flat_map(|claim| match claim {
            Pending::Direct(claim) => vec![claim],
            Pending::Planes {
                prefix,
                point,
                values,
            } => plane_chunks(values.len())
                .into_iter()
                .map(|(start, k)| {
                    let weights = om_sumcheck::eq_table(&rho[..k]);
                    let value = values[start..start + (1 << k)]
                        .iter()
                        .zip(&weights)
                        .fold(K::ZERO, |sum, (&v, &w)| sum + v * w);
                    let mut at = point.clone();
                    at.extend_from_slice(&rho[..k]);
                    EvalClaim {
                        poly: chunk_name(prefix, start, k),
                        point: at,
                        value,
                    }
                })
                .collect(),
        })
        .collect()
}

fn group_rho(tr: &mut Transcript) -> Vec<K> {
    tr.challenge_ks(b"spice/typed/plane-chunks/rho", RHO_LEN)
}

fn transcript(s: &Statement, schema: &Schema, options: &Options) -> Transcript {
    let mut tr = crate::transcript(s, schema, &OrderingMode::FlockBoolean);
    tr.absorb_bytes(b"spice/trace-semantics", b"typed-single-value-masked/v1");
    options.absorb(&mut tr);
    tr
}

fn pack_flags(flags: &[bool]) -> PackedBits {
    PackedBits::from_words(
        flags
            .chunks(64)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u64, |v, (i, &b)| v | u64::from(b) << i)
            })
            .collect(),
        flags.len(),
    )
}

fn scalar_claim(name: &str, point: &[K], value: K) -> EvalClaim {
    EvalClaim {
        poly: name.into(),
        point: point.to_vec(),
        value,
    }
}

/// All Mosaic claims before chunk combination: every plane group at every
/// point where the protocol needs it, the word value and the initial flag.
fn pending_claims(
    s: &Statement,
    leaf: &[K],
    mask: &[K],
    order: &order_flock::OrderClaims,
    v: &Evaluations,
) -> Vec<Pending> {
    let (n, a, _, _) = dims(s);
    vec![
        group("input/address", &leaf[..n], &v.address),
        group("input/previous", &leaf[..n], &v.previous),
        group("input/flags", &leaf[..n], &v.flags),
        group("boundary/previous", &leaf[..a], &v.final_previous),
        Pending::Direct(scalar_claim(
            "initial/next_is_write",
            &leaf[..a],
            v.initial_next_is_write,
        )),
        Pending::Direct(word_claim(
            "input/value",
            &mask[..n],
            v.mask_value,
            s.config.word_log_bits,
        )),
        group("input/flags", &mask[..n], &v.mask_flags),
        Pending::Direct(scalar_claim(
            "initial/next_is_write",
            &mask[..a],
            v.mask_initial_next_is_write,
        )),
        group("input/previous", &order.point_ab, &order.previous_ab),
        group("order/carry", &order.point_ab, &order.carries_ab),
        // Flock extracts the linear C column at a second cycle point.
        group("order/carry", &order.point_c, &order.carries_c),
    ]
}

fn absorb_masked(tr: &mut Transcript, v: &MaskClaims) {
    tr.absorb_ks(b"spice/typed/masked", &[v.cycle[0], v.cycle[1], v.initial]);
}

fn absorb_evaluations(tr: &mut Transcript, v: &Evaluations) {
    tr.absorb_ks(b"spice/typed/address", &v.address);
    tr.absorb_ks(b"spice/typed/previous", &v.previous);
    tr.absorb_ks(b"spice/typed/flags", &v.flags);
    tr.absorb_ks(b"spice/typed/final-previous", &v.final_previous);
    tr.absorb_k(b"spice/typed/initial-flag", v.initial_next_is_write);
    tr.absorb_k(b"spice/typed/mask/value", v.mask_value);
    tr.absorb_ks(b"spice/typed/mask/flags", &v.mask_flags);
    tr.absorb_k(
        b"spice/typed/mask/initial-flag",
        v.mask_initial_next_is_write,
    );
    tr.absorb_ks(b"spice/typed/order/previous-ab", &v.previous_ab);
    tr.absorb_ks(b"spice/typed/order/carries-ab", &v.carries_ab);
    tr.absorb_ks(b"spice/typed/order/carries-c", &v.carries_c);
}

fn check_shapes(s: &Statement, v: &Evaluations) -> Result<(), Error> {
    let (_, a, _, b) = dims(s);
    require(
        v.address.len() == a
            && v.previous.len() == b
            && v.final_previous.len() == b
            && v.previous_ab.len() == b
            && v.carries_ab.len() == b - 1
            && v.carries_c.len() == b - 1,
        "malformed typed evaluations",
    )
}

fn check_leaves(
    s: &Statement,
    leaves: &LeafClaims,
    masked: &MaskClaims,
    v: &Evaluations,
    fps: &[Fingerprint],
) -> Result<(), Error> {
    let (n, a, l, _) = dims(s);
    let log_f = fps.len().trailing_zeros() as usize;
    require(
        leaves.point.len() == l + 1 + log_f,
        "malformed typed permutation terminals",
    )?;
    let p = &leaves.point;
    let cm = p[n..l].iter().fold(K::ONE, |v, &x| v * (K::ONE + x));
    let mm = p[a..l].iter().fold(K::ONE, |v, &x| v * (K::ONE + x));
    let addr = combine_bits(&v.address);
    let previous = combine_bits(&v.previous);
    let final_previous = combine_bits(&v.final_previous);
    let memory_addr = combine_bits(&p[..a]);
    let time = K::from_u128(1u128 << l) + combine_bits(&p[..n]);
    let final_value = word_eval(&s.final_memory, &p[..a], s.config.word_log_bits);
    // Set f occupies the aligned block selected by the top log_f coordinates.
    let selectors = om_sumcheck::eq_table(&p[l + 1..]);
    let mut expected = [K::ZERO; 2];
    for (fp, &select) in fps.iter().zip(&selectors) {
        let Fingerprint {
            gamma,
            alpha,
            beta,
            delta,
        } = *fp;
        let cycles = [
            gamma + previous + alpha * addr + beta * masked.cycle[0] + delta * v.flags[0],
            gamma + time + alpha * addr + beta * masked.cycle[1] + delta * v.flags[1],
        ];
        let memory = [
            gamma + final_previous + alpha * memory_addr + beta * final_value,
            gamma
                + memory_addr
                + alpha * memory_addr
                + beta * masked.initial
                + delta * v.initial_next_is_write,
        ];
        for side in 0..2 {
            expected[side] += select
                * ((K::ONE + p[l]) * (K::ONE + cm * (K::ONE + cycles[side]))
                    + p[l] * (K::ONE + mm * (K::ONE + memory[side])));
        }
    }
    for side in 0..2 {
        require(
            expected[side] == leaves.values[side],
            "typed permutation leaf binding failed",
        )?;
    }
    Ok(())
}

/// The verifier's checks on the evaluation message, in one place for the
/// prover's self-check and the verifier.
#[allow(clippy::too_many_arguments)]
fn check_evaluations(
    s: &Statement,
    leaves: &LeafClaims,
    masked: &MaskClaims,
    mask: &masked_reduction::Deferred,
    order: &order_flock::FlockClaim,
    v: &Evaluations,
    fps: &[Fingerprint],
) -> Result<order_flock::OrderClaims, Error> {
    let (n, a, _, _) = dims(s);
    check_shapes(s, v)?;
    check_leaves(s, leaves, masked, v, fps)?;
    mask.check(
        v.mask_value,
        v.mask_flags[0],
        v.mask_flags[1],
        v.mask_initial_next_is_write,
    )?;
    Ok(order_flock::check_sources(
        n,
        a,
        order,
        &v.previous_ab,
        &v.carries_ab,
        &v.carries_c,
    )?)
}

pub fn prove(config: &Config, trace: &AccessTrace) -> Result<ProveResult, Error> {
    prove_with(config, trace, &Options::default())
}

pub fn prove_with(
    config: &Config,
    trace: &AccessTrace,
    options: &Options,
) -> Result<ProveResult, Error> {
    profiling::clear();
    options.validate()?;
    let t = trace.addresses.len();
    require(
        t.is_power_of_two(),
        "typed operation count must be a nonzero power of two",
    )?;
    require(
        trace.values.len() == t && trace.is_write.len() == t,
        "typed operation shape mismatch",
    )?;
    let n = t.trailing_zeros() as usize;
    let mut statement = Statement {
        config: config.clone(),
        log_cycles: n,
        initial_memory: trace.initial.clone(),
        final_memory: trace.initial.clone(),
    };
    validate(&statement)?;
    let (_, a, l, b) = dims(&statement);
    let cells = 1usize << a;
    require(
        trace.addresses.iter().all(|&k| (k as usize) < cells),
        "typed address out of range",
    )?;
    require(
        trace
            .values
            .iter()
            .all(|&v| config.word_log_bits == 6 || v >> (1usize << config.word_log_bits) == 0),
        "typed value exceeds configured width",
    )?;
    let (previous, next, initial_next, last, carries) = {
        let _span = Span::new("spice.witness");
        let mut memory = trace.initial.clone();
        let mut last: Vec<u32> = (0..cells as u32).collect();
        let mut previous = Vec::with_capacity(t);
        let mut next = vec![false; t];
        let mut initial_next = vec![false; cells];
        let base = 1u32 << l;
        for j in 0..t {
            let k = trace.addresses[j] as usize;
            if trace.is_write[j] {
                memory[k] = trace.values[j];
            } else {
                require(trace.values[j] == memory[k], "incorrect typed read value")?;
            }
            let prev = last[k];
            previous.push(prev);
            if prev < base {
                initial_next[prev as usize] = trace.is_write[j];
            } else {
                next[(prev - base) as usize] = trace.is_write[j];
            }
            last[k] = base | j as u32;
        }
        statement.final_memory = memory;
        let carries = order::build_carries(&previous, n, a)?;
        (previous, next, initial_next, last, carries)
    };
    let schema = schema(&statement);
    let mut tr = transcript(&statement, &schema, options);
    let (address_columns, previous_columns, carry_columns, last_columns, flags, initial_flags, g) = {
        let _span = Span::new("spice.pack");
        let address = pack_columns(&trace.addresses, a);
        let prev = pack_columns(&previous, b);
        let last_columns = pack_columns(&last, b);
        let carry = pack_columns(&carries, b - 1);
        let flags = [pack_flags(&trace.is_write), pack_flags(&next)];
        let initial_flags = pack_flags(&initial_next);
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        schema.place(
            &mut g,
            "input/value",
            &pack_values(&trace.values, config.word_log_bits),
        );
        for (prefix, cols) in [
            ("input/address", &address[..]),
            ("input/previous", &prev[..]),
            ("boundary/previous", &last_columns[..]),
            ("order/carry", &carry[..]),
            ("input/flags", &flags[..]),
        ] {
            place_planes(&mut g, &schema, prefix, cols);
        }
        schema.place(&mut g, "initial/next_is_write", &initial_flags);
        (address, prev, carry, last_columns, flags, initial_flags, g)
    };
    let (commitment, state) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let fps = fingerprints(&mut tr, options);
    let log_f = options.log_fingerprints();
    let leaves = {
        let _span = Span::new("spice.permutation.leaves");
        let half = 1usize << l;
        let set = 2 * half;
        let padding = (0..fps.len())
            .flat_map(|f| {
                [
                    f * set + t..f * set + half,
                    f * set + half + cells..(f + 1) * set,
                ]
            })
            .filter(|x| !x.is_empty())
            .collect();
        Leaves::from_fn_pair(set << log_f, padding, |row| {
            let Fingerprint {
                gamma,
                alpha,
                beta,
                delta,
            } = fps[row / set];
            let row = row % set;
            if row < half {
                let j = row;
                let common = gamma + alpha * K::from_u128(trace.addresses[j] as u128);
                let masks = [trace.is_write[j], next[j]];
                let values = mul_pairs(
                    [beta; 2],
                    masks.map(|flag| {
                        K::from_u128(
                            (trace.values[j] & 0u64.wrapping_sub(u64::from(!flag))) as u128,
                        )
                    }),
                );
                let tags = masks
                    .map(|flag| K::from_u128(delta.value() & 0u128.wrapping_sub(u128::from(flag))));
                [
                    common + K::from_u128(previous[j] as u128) + values[0] + tags[0],
                    common + K::from_u128(((1usize << l) | j) as u128) + values[1] + tags[1],
                ]
            } else {
                let k = row - half;
                let addr = K::from_u128(k as u128);
                let common = gamma + alpha * addr;
                let init = trace.initial[k] & 0u64.wrapping_sub(u64::from(!initial_next[k]));
                let values = mul_pairs(
                    [beta; 2],
                    [
                        K::from_u128(statement.final_memory[k] as u128),
                        K::from_u128(init as u128),
                    ],
                );
                let tag =
                    K::from_u128(delta.value() & 0u128.wrapping_sub(u128::from(initial_next[k])));
                [
                    common + K::from_u128(last[k] as u128) + values[0],
                    common + addr + values[1] + tag,
                ]
            }
        })?
    };
    let (permutation, leaves) = {
        let _span = Span::new("spice.permutation.gkr");
        lean_grandproduct::prove(l + 1 + log_f, leaves, &mut tr, b"spice/typed/permutation")?
    };
    check_products(&permutation, options)?;
    let masked = {
        let _span = Span::new("spice.masked.claims");
        let cycle = planes::masked_word_evals(
            &trace.values,
            [&trace.is_write, &next],
            &leaves.point[..n],
            config.word_log_bits,
        );
        let initial: Vec<u64> = trace
            .initial
            .iter()
            .zip(&initial_next)
            .map(|(&v, &w)| v & 0u64.wrapping_sub(u64::from(!w)))
            .collect();
        MaskClaims {
            cycle,
            initial: word_eval(&initial, &leaves.point[..a], config.word_log_bits),
        }
    };
    absorb_masked(&mut tr, &masked);
    let (masks, mask_point, mask_weights, mask_value, folded) = {
        let _span = Span::new("spice.mask");
        masked_reduction::prove_sumcheck(
            &trace.values,
            &trace.is_write,
            &next,
            &trace.initial,
            &initial_next,
            &leaves.point[..l + 1],
            &masked,
            &mut tr,
            b"spice/typed/masks",
        )?
    };
    let (order, order_terminal) = {
        let _span = Span::new("spice.order");
        order_flock::prove_zerocheck(&previous, &carries, n, a, &mut tr, b"spice/typed/order")?
    };
    let evaluations = {
        let _span = Span::new("spice.claims");
        let (point_ab, point_c) = order_flock::terminal_points(n, &order_terminal);
        let cycle_points = [&leaves.point[..n], &mask_point[..n], &point_ab, &point_c];
        fn request(columns: &[PackedBits], point: usize) -> planes::Request<'_> {
            planes::Request {
                planes: columns.iter().collect(),
                point,
            }
        }
        let mut cycle = planes::evaluate(
            &cycle_points,
            &[
                request(&address_columns, 0),
                request(&previous_columns, 0),
                request(&flags, 0),
                request(&flags, 1),
                request(&previous_columns, 2),
                request(&carry_columns, 2),
                request(&carry_columns, 3),
            ],
        )
        .into_iter();
        let mut memory = planes::evaluate(
            &[&leaves.point[..a], &mask_point[..a]],
            &[
                request(&last_columns, 0),
                planes::Request {
                    planes: vec![&initial_flags],
                    point: 0,
                },
                planes::Request {
                    planes: vec![&initial_flags],
                    point: 1,
                },
            ],
        )
        .into_iter();
        let take = |it: &mut std::vec::IntoIter<Vec<K>>| it.next().expect("request");
        let pair = |v: Vec<K>| [v[0], v[1]];
        Evaluations {
            address: take(&mut cycle),
            previous: take(&mut cycle),
            flags: pair(take(&mut cycle)),
            mask_flags: pair(take(&mut cycle)),
            previous_ab: take(&mut cycle),
            carries_ab: take(&mut cycle),
            carries_c: take(&mut cycle),
            final_previous: take(&mut memory),
            initial_next_is_write: take(&mut memory)[0],
            mask_initial_next_is_write: take(&mut memory)[0],
            mask_value,
        }
    };
    require(
        folded[0]
            == mask_weights[0]
                + mask_weights[1]
                + mask_weights[0] * evaluations.mask_flags[0]
                + mask_weights[1] * evaluations.mask_flags[1]
            && folded[1] == mask_weights[2] * (K::ONE + evaluations.mask_initial_next_is_write),
        "typed masked reduction: folded masks disagree with flag sources",
    )?;
    absorb_evaluations(&mut tr, &evaluations);
    check_shapes(&statement, &evaluations)?;
    check_leaves(&statement, &leaves, &masked, &evaluations, &fps)?;
    let order_claims = order_flock::check_sources(
        n,
        a,
        &order_terminal,
        &evaluations.previous_ab,
        &evaluations.carries_ab,
        &evaluations.carries_c,
    )?;
    let claims = finish_claims(
        pending_claims(
            &statement,
            &leaves.point,
            &mask_point,
            &order_claims,
            &evaluations,
        ),
        &group_rho(&mut tr),
    );
    let (mosaic, r, value) = {
        let _span = Span::new("mosaic.batch");
        batch_prove_packed(&schema.layout, &schema.claims(&claims)?, &g, &mut tr)
    };
    let (opened, opening) = {
        let _span = Span::new("pcs.open");
        schema.pcs.open_packed(state, &g, &r, &mut tr)
    };
    require(opened == value, "typed Mosaic/PCS mismatch")?;
    let summary = Summary {
        cycles: t,
        memory_cells: cells,
        address_components: a,
        chunk_bits: 1,
        committed_bits: g.len(),
        packed_field_elements: g.len() / 128,
        mosaic_claims: claims.len(),
        commitment_components: schema.components,
    };
    Ok(ProveResult {
        statement,
        proof: Proof {
            commitment,
            permutation,
            masked,
            masks,
            order,
            evaluations,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
    })
}

fn verify_inner(s: &Statement, p: &Proof, options: &Options) -> Result<(), Error> {
    validate(s)?;
    options.validate()?;
    let (n, a, l, _) = dims(s);
    let schema = schema(s);
    require(
        p.commitment.n == schema.layout.ell - 7 && p.commitment.root.len() == 32,
        "malformed typed commitment",
    )?;
    let mut tr = transcript(s, &schema, options);
    schema.pcs.absorb_commitment(&p.commitment, &mut tr);
    let fps = fingerprints(&mut tr, options);
    let leaves = lean_grandproduct::verify(
        l + 1 + options.log_fingerprints(),
        &p.permutation,
        &mut tr,
        b"spice/typed/permutation",
    )?;
    check_products(&p.permutation, options)?;
    absorb_masked(&mut tr, &p.masked);
    let mask = masked_reduction::verify_sumcheck(
        &s.initial_memory,
        &leaves.point[..l + 1],
        &p.masked,
        &p.masks,
        &mut tr,
        b"spice/typed/masks",
    )?;
    let order = order_flock::verify_zerocheck_with(n, a, &p.order, &mut tr, b"spice/typed/order")?;
    absorb_evaluations(&mut tr, &p.evaluations);
    let order_claims =
        check_evaluations(s, &leaves, &p.masked, &mask, &order, &p.evaluations, &fps)?;
    let claims = finish_claims(
        pending_claims(s, &leaves.point, &mask.point, &order_claims, &p.evaluations),
        &group_rho(&mut tr),
    );
    let (r, value) = batch_verify(&schema.layout, &schema.claims(&claims)?, &p.mosaic, &mut tr)?;
    schema
        .pcs
        .verify(&p.commitment, &r, value, &p.opening, &mut tr)?;
    Ok(())
}

pub fn verify(s: &Statement, p: &Proof) -> Result<(), Error> {
    verify_with(s, p, &Options::default())
}

/// Verify under explicit options. The verifier, not the proof, chooses the
/// number of fingerprints, so a one-fingerprint proof never passes as 2fp.
pub fn verify_with(s: &Statement, p: &Proof, options: &Options) -> Result<(), Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| verify_inner(s, p, options)))
        .map_err(|_| Error("malformed typed RAM proof".into()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use lean_grandproduct::LayerProof;

    fn proof(children: [Vec<K>; 2]) -> GrandProductProof {
        GrandProductProof {
            roots: [K::ONE; 2],
            layers: vec![LayerProof {
                rounds: Vec::new(),
                children,
            }],
        }
    }

    #[test]
    fn chunks_are_aligned_and_keep_the_primary_commitment_size() {
        assert_eq!(plane_chunks(21), vec![(0, 4), (16, 2), (20, 0)]);
        assert_eq!(plane_chunks(20), vec![(0, 4), (16, 2)]);
        assert_eq!(plane_chunks(16), vec![(0, 4)]);
        assert_eq!(plane_chunks(2), vec![(0, 1)]);
        let s = Statement {
            config: Config {
                address_bits: 16,
                chunk_bits: 1,
                ..Config::default()
            },
            log_cycles: 20,
            initial_memory: vec![0; 1 << 16],
            final_memory: vec![0; 1 << 16],
        };
        let schema = schema(&s);
        let populated: usize = schema.components.iter().map(|c| 1usize << c.log_size).sum();
        // 124.375 populated bits per row; the padded commitment stays 2^27.
        assert_eq!(populated, 130_416_640);
        assert_eq!(schema.layout.ell, 27);
        for p in &schema.layout.placements {
            assert_eq!(p.offset % (1usize << p.log_size), 0);
        }
    }

    #[test]
    fn per_set_products_come_from_the_first_layer() {
        let k = |v: u128| K::from_u128(v);
        let two = Options::TWO_FINGERPRINTS;
        // Binary first layer: one child per set.
        let p = proof([vec![k(2), k(3)], vec![k(2), k(3)]]);
        assert_eq!(
            fingerprint_products(&p, &two).unwrap(),
            vec![[k(2), k(2)], [k(3), k(3)]]
        );
        assert!(check_products(&p, &two).is_ok());
        // Equal overall products, unequal per-set products: rejected.
        let p = proof([vec![k(2) * k(5), k(3)], vec![k(2), k(3) * k(5)]]);
        assert!(check_products(&p, &two).is_err());
        // Radix-four first layer: cycle and memory halves of each set.
        let p = proof([vec![k(2), k(3), k(5), k(7)], vec![k(6), K::ONE, k(7), k(5)]]);
        assert!(check_products(&p, &two).is_ok());
        let p = proof([vec![k(2), k(3), k(5), k(7)], vec![k(5), k(7), k(2), k(3)]]);
        // Swapping the sets keeps the overall product but not the per-set ones.
        assert!(check_products(&p, &two).is_err());
        let p = proof([vec![k(2), k(3), k(5), k(7)], vec![k(2), k(5), k(3), k(7)]]);
        assert!(check_products(&p, &two).is_err());
        assert!(fingerprint_products(&proof([vec![k(1)], vec![k(1)]]), &two).is_err());
    }
}
