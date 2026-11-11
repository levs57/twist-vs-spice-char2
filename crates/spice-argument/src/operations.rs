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
    order, order_flock, pack_columns, pack_values, planes, require, validate, validate_memory,
    word_claim, word_eval,
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
    /// Commit final memory (one word per cell) instead of reading it from the
    /// statement. The leaf binding then uses a Mosaic claim on the committed
    /// words. `Statement::final_memory` must then be empty: the verifier
    /// rejects a statement that carries one, so a caller cannot mistake it
    /// for a checked value. `ProveResult::final_memory` holds the prover's.
    pub private_final_memory: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            fingerprints: 1,
            private_final_memory: false,
        }
    }
}

impl Options {
    /// The `spice-typed-2fp` method: two independent fingerprints.
    pub const TWO_FINGERPRINTS: Self = Self {
        fingerprints: 2,
        private_final_memory: false,
    };

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
            &[
                self.fingerprints as u64,
                u64::from(self.private_final_memory),
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>(),
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
    /// With private final memory: the committed final words at the GKR leaf
    /// point's memory prefix. `None` when final memory is public.
    pub final_value: Option<K>,
}

/// The committed columns at the CPU point `r_c`, sent right after the
/// commitment. A CPU proof would compare these with its own columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuEvaluations {
    /// Address bitplane `b` at `r_c`, low bit first.
    pub address: Vec<K>,
    /// `W[value](r_c) = Σ_j eq(r_c, j) enc(value_j)`.
    pub value: K,
    /// `Σ_j eq(r_c, j) is_write_j`.
    pub is_write: K,
}

/// What the verifier hands the CPU: the point `r_c` and the word-level
/// claims `W[addr](r_c)`, `W[value](r_c)` and the `is_write` evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interface {
    pub point: Vec<K>,
    pub address: K,
    pub value: K,
    pub is_write: K,
}

impl Interface {
    fn new(point: Vec<K>, cpu: &CpuEvaluations) -> Self {
        Interface {
            point,
            address: combine_bits(&cpu.address),
            value: cpu.value,
            is_write: cpu.is_write,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    /// The committed columns at the CPU point.
    pub cpu: CpuEvaluations,
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
    /// The public statement. With private final memory its `final_memory`
    /// is empty.
    pub statement: Statement,
    /// The final memory the prover computed, in both modes.
    pub final_memory: Vec<u64>,
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

fn schema(s: &Statement, options: &Options) -> Schema {
    let (n, a, _, _) = dims(s);
    let mut components = vec![ComponentSummary {
        name: "input/value".into(),
        log_size: n + s.config.word_log_bits,
    }];
    if options.private_final_memory {
        components.push(ComponentSummary {
            name: "boundary/final".into(),
            log_size: a + s.config.word_log_bits,
        });
    }
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
            if offset.is_multiple_of(64) && plane_len >= 64 {
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

/// `span` names the profiling stage that hashes the public statement.
fn transcript(s: &Statement, schema: &Schema, options: &Options, span: &'static str) -> Transcript {
    let mut tr = crate::transcript_with(
        s,
        schema,
        &OrderingMode::FlockBoolean,
        !options.private_final_memory,
        span,
    );
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
    cpu_point: &[K],
    cpu: &CpuEvaluations,
    leaf: &[K],
    mask: &[K],
    order: &order_flock::OrderClaims,
    v: &Evaluations,
) -> Vec<Pending> {
    let (n, a, _, _) = dims(s);
    let final_value = v.final_value.map(|value| {
        Pending::Direct(word_claim(
            "boundary/final",
            &leaf[..a],
            value,
            s.config.word_log_bits,
        ))
    });
    final_value
        .into_iter()
        .chain([
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
            // The CPU interface: address planes, the word value and
            // is_write (plane 0 of input/flags, the high coordinate) at r_c.
            group("input/address", cpu_point, &cpu.address),
            Pending::Direct(word_claim(
                "input/value",
                cpu_point,
                cpu.value,
                s.config.word_log_bits,
            )),
            Pending::Direct(scalar_claim(
                &chunk_name("input/flags", 0, 1),
                &[cpu_point, &[K::ZERO]].concat(),
                cpu.is_write,
            )),
        ])
        .collect()
}

/// `r_c`: `n` nonzero draws right after the commitment.
fn cpu_point(tr: &mut Transcript, n: usize) -> Vec<K> {
    (0..n)
        .map(|_| tr.challenge_nonzero_k(b"spice/typed/cpu_point"))
        .collect()
}

fn absorb_cpu(tr: &mut Transcript, cpu: &CpuEvaluations) {
    let mut values = cpu.address.clone();
    values.extend([cpu.value, cpu.is_write]);
    tr.absorb_ks(b"spice/typed/cpu_claims", &values);
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
    if let Some(value) = v.final_value {
        tr.absorb_k(b"spice/typed/final-value", value);
    }
}

fn check_shapes(s: &Statement, v: &Evaluations, options: &Options) -> Result<(), Error> {
    let (_, a, _, b) = dims(s);
    require(
        v.final_value.is_some() == options.private_final_memory
            && v.address.len() == a
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
    // Private final memory: the committed words' claimed evaluation, which
    // Mosaic authenticates. Public: the verifier evaluates it directly.
    let final_value = match v.final_value {
        Some(value) => value,
        None => word_eval(&s.final_memory, &p[..a], s.config.word_log_bits),
    };
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
    options: &Options,
) -> Result<order_flock::OrderClaims, Error> {
    let (n, a, _, _) = dims(s);
    check_shapes(s, v, options)?;
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
    prove_hooked(config, trace, options, None)
}

/// A test hook that edits the CPU evaluations before they are sent and
/// absorbed. The prover's own Mosaic claims keep the computed values, so a
/// wrong evaluation reaches the verifier instead of tripping the prover.
type CpuHook<'a> = Option<&'a dyn Fn(&mut CpuEvaluations)>;

fn prove_hooked(
    config: &Config,
    trace: &AccessTrace,
    options: &Options,
    hook: CpuHook<'_>,
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
    let schema = schema(&statement, options);
    let mut tr = transcript(&statement, &schema, options, "spice.statement");
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
        if options.private_final_memory {
            schema.place(
                &mut g,
                "boundary/final",
                &pack_values(&statement.final_memory, config.word_log_bits),
            );
        }
        (address, prev, carry, last_columns, flags, initial_flags, g)
    };
    let (commitment, state) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let r_c = cpu_point(&mut tr, n);
    let cpu = {
        let _span = Span::new("spice.interface");
        let planes = planes::evaluate(
            &[&r_c],
            &[planes::Request {
                planes: address_columns.iter().chain([&flags[0]]).collect(),
                point: 0,
            }],
        )
        .pop()
        .expect("request");
        CpuEvaluations {
            address: planes[..a].to_vec(),
            is_write: planes[a],
            value: ram_argument::word::mle(&trace.values, &r_c),
        }
    };
    let computed = cpu.clone();
    let mut cpu = cpu;
    if let Some(hook) = hook {
        hook(&mut cpu);
    }
    absorb_cpu(&mut tr, &cpu);
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
            final_value: options.private_final_memory.then(|| {
                word_eval(
                    &statement.final_memory,
                    &leaves.point[..a],
                    config.word_log_bits,
                )
            }),
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
    check_shapes(&statement, &evaluations, options)?;
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
            &r_c,
            &computed,
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
    let final_memory = if options.private_final_memory {
        std::mem::take(&mut statement.final_memory)
    } else {
        statement.final_memory.clone()
    };
    Ok(ProveResult {
        statement,
        final_memory,
        proof: Proof {
            commitment,
            cpu,
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

fn verify_inner(s: &Statement, p: &Proof, options: &Options) -> Result<Interface, Error> {
    // Spans carry the `verify.` prefix because the harness clears the
    // profiler before verifying and reports only these stages.
    let (n, a, l, schema, mut tr, fps, r_c) = {
        let _span = Span::new("verify.setup");
        validate_memory(s, !options.private_final_memory)?;
        require(
            !options.private_final_memory || s.final_memory.is_empty(),
            "a statement with private final memory must not carry final memory",
        )?;
        options.validate()?;
        let (n, a, l, _) = dims(s);
        let schema = schema(s, options);
        require(
            p.commitment.n == schema.layout.ell - 7 && p.commitment.root.len() == 32,
            "malformed typed commitment",
        )?;
        let mut tr = transcript(s, &schema, options, "verify.statement");
        schema.pcs.absorb_commitment(&p.commitment, &mut tr);
        require(p.cpu.address.len() == a, "malformed typed CPU evaluations")?;
        let r_c = cpu_point(&mut tr, n);
        absorb_cpu(&mut tr, &p.cpu);
        let fps = fingerprints(&mut tr, options);
        (n, a, l, schema, tr, fps, r_c)
    };
    let leaves = {
        let _span = Span::new("verify.gkr");
        let leaves = lean_grandproduct::verify(
            l + 1 + options.log_fingerprints(),
            &p.permutation,
            &mut tr,
            b"spice/typed/permutation",
        )?;
        check_products(&p.permutation, options)?;
        leaves
    };
    let mask = {
        let _span = Span::new("verify.mask");
        absorb_masked(&mut tr, &p.masked);
        masked_reduction::verify_sumcheck(
            &s.initial_memory,
            &leaves.point[..l + 1],
            &p.masked,
            &p.masks,
            &mut tr,
            b"spice/typed/masks",
        )?
    };
    let order = {
        let _span = Span::new("verify.order");
        order_flock::verify_zerocheck_with(n, a, &p.order, &mut tr, b"spice/typed/order")?
    };
    let claims = {
        let _span = Span::new("verify.claims");
        absorb_evaluations(&mut tr, &p.evaluations);
        let order_claims = check_evaluations(
            s,
            &leaves,
            &p.masked,
            &mask,
            &order,
            &p.evaluations,
            &fps,
            options,
        )?;
        let claims = finish_claims(
            pending_claims(
                s,
                &r_c,
                &p.cpu,
                &leaves.point,
                &mask.point,
                &order_claims,
                &p.evaluations,
            ),
            &group_rho(&mut tr),
        );
        schema.claims(&claims)?
    };
    let (r, value) = {
        let _span = Span::new("verify.mosaic");
        batch_verify(&schema.layout, &claims, &p.mosaic, &mut tr)?
    };
    let _span = Span::new("verify.pcs");
    schema
        .pcs
        .verify(&p.commitment, &r, value, &p.opening, &mut tr)?;
    Ok(Interface::new(r_c, &p.cpu))
}

/// Serialized proof size in bytes, split by component. A field element
/// counts 16 bytes, a hash or Merkle root 32, an index or small integer 4,
/// a PoW nonce 8, and length prefixes nothing: every length follows from
/// the statement and options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofSize {
    /// Commitment root and variable count, ring-switching and Ligerito.
    pub pcs_bytes: usize,
    /// The Mosaic batching sum-check and the claimed value of `G`.
    pub mosaic_bytes: usize,
    /// The CPU evaluations, GKR, masked reduction, Flock zerocheck and the
    /// evaluation message.
    pub iop_bytes: usize,
}

impl ProofSize {
    pub fn total(&self) -> usize {
        self.pcs_bytes + self.mosaic_bytes + self.iop_bytes
    }
}

const FIELD_BYTES: usize = 16;
const HASH_BYTES: usize = 32;
const INDEX_BYTES: usize = 4;
const NONCE_BYTES: usize = 8;

fn sumcheck_bytes(sc: &SumcheckProof) -> usize {
    FIELD_BYTES * sc.round_msgs.iter().map(Vec::len).sum::<usize>()
}

pub fn proof_size(p: &Proof) -> ProofSize {
    let (ring, ligerito) = &p.opening;
    let pcs_bytes = HASH_BYTES
        + INDEX_BYTES
        + FIELD_BYTES * ring.s.len()
        + sumcheck_bytes(&ring.sumcheck)
        + HASH_BYTES * ligerito.roots.len()
        + ligerito.sumchecks.iter().map(sumcheck_bytes).sum::<usize>()
        + ligerito
            .openings
            .iter()
            .map(|o| {
                FIELD_BYTES * o.columns.iter().map(Vec::len).sum::<usize>()
                    + HASH_BYTES * o.auth.len()
            })
            .sum::<usize>()
        + FIELD_BYTES * ligerito.final_poly.len();
    let mosaic_bytes = sumcheck_bytes(&p.mosaic.sumcheck) + FIELD_BYTES;
    let gkr = FIELD_BYTES
        * (p.permutation.roots.len()
            + p.permutation
                .layers
                .iter()
                .map(|layer| {
                    4 * layer.rounds.len() + layer.children.iter().map(Vec::len).sum::<usize>()
                })
                .sum::<usize>());
    let masked = FIELD_BYTES * (p.masked.cycle.len() + 1) + sumcheck_bytes(&p.masks);
    let o = &p.order;
    let order = FIELD_BYTES
        * (o.round1_ab.len() + o.round1_c.len() + 2 * o.multilinear_rounds.len() + 3)
        + NONCE_BYTES * o.grinding_nonces.len();
    let v = &p.evaluations;
    let evaluations = FIELD_BYTES
        * (v.address.len()
            + v.previous.len()
            + v.flags.len()
            + v.final_previous.len()
            + 1
            + 1
            + v.mask_flags.len()
            + 1
            + v.previous_ab.len()
            + v.carries_ab.len()
            + v.carries_c.len()
            + usize::from(v.final_value.is_some()));
    let cpu = FIELD_BYTES * (p.cpu.address.len() + 2);
    ProofSize {
        pcs_bytes,
        mosaic_bytes,
        iop_bytes: cpu + gkr + masked + order + evaluations,
    }
}

/// Soundness error terms of typed Spice outside the PCS, as probabilities
/// over `K = GF(2^128)`. Each term is a union-bound summand; the PCS and
/// Fiat-Shamir terms are added by the caller. `docs/soundness-spice.md`
/// states what each step proves.
pub fn soundness_terms(s: &Statement, options: &Options) -> Vec<(String, f64)> {
    let field = 2f64.powi(-128);
    let (n, a, l, _) = dims(s);
    let mut terms: Vec<(String, f64)> = Vec::new();

    // Fingerprints (`fingerprints`, `check_products`): each set compares two
    // products of `T + K` affine forms in (gamma, alpha, beta, delta). Distinct
    // multisets give distinct polynomials of total degree `T + K`, so one set
    // collides with probability at most `(T + K) / |K|`. Sets use independent
    // challenges, so the errors multiply.
    let fingerprint = ((1u64 << n) + (1u64 << a)) as f64 * field;
    terms.push((
        "fingerprint".into(),
        fingerprint.powi(options.fingerprints as i32),
    ));
    let mut push = |name: &str, numerator: f64| terms.push((name.to_string(), numerator * field));

    // GKR (`lean_grandproduct::verify` over `l + 1 + log_f` variables): per
    // layer, one batching challenge for the two trees (1), one normalized
    // quartic sum-check round per coordinate already bound (4 each), and the
    // child point that reduces 2 or 4 children to one claim (1 or 2).
    let log_size = l + 1 + options.log_fingerprints();
    let (mut batch, mut rounds, mut children) = (0usize, 0usize, 0usize);
    let mut layer = log_size;
    while layer > 0 {
        let step = if layer.is_multiple_of(2) { 2 } else { 1 };
        batch += 1;
        rounds += log_size - layer;
        children += step;
        layer -= step;
    }
    push("gkr.batch", batch as f64);
    push("gkr.sumcheck", 4.0 * rounds as f64);
    push("gkr.child-point", children as f64);

    // Masked reduction (`masked_reduction::verify_sumcheck`): three claims
    // batched with independent weights (1), then `l` cubic rounds.
    push("mask.batch", 1.0);
    push("mask.sumcheck", 3.0 * l as f64);

    // Flock zerocheck (`order_flock::verify_zerocheck_with`, then
    // `flock_core::zerocheck::verify`). The seven fixed equality coordinates
    // have F_2-independent weights (test `fixed_inner_equality_weights_form_
    // an_f2_basis`), so a Boolean gate error survives them and only the random
    // coordinates count: at most `skip + rounds - fixed`. The univariate-skip
    // fold point `z` checks a polynomial of degree below `2 * 2^skip`; each
    // multilinear round is quadratic.
    let (skip, flock_rounds, fixed) = order_flock::zerocheck_shape(n);
    push("flock.zerocheck", (skip + flock_rounds - fixed) as f64);
    push("flock.skip", ((2usize << skip) - 1) as f64);
    push("flock.sumcheck", 2.0 * flock_rounds as f64);

    // Plane chunks (`finish_claims`, `group_rho`): a chunk of `2^k` planes is
    // one claim weighted by `eq(rho[..k], i)`, multilinear of degree `k` in
    // rho. Every wrong plane value makes its chunk's claim wrong unless rho
    // hits a root, so the largest `k` bounds the error.
    let k_max = plane_groups(s)
        .iter()
        .flat_map(|&(_, count, _)| plane_chunks(count))
        .map(|(_, k)| k)
        .max()
        .unwrap_or(0);
    push("rho", k_max as f64);

    // Mosaic (`om_mosaic::batch_verify`): powers-of-gamma batching of `c`
    // claims (c - 1), then `ell` quadratic rounds over the committed `G`.
    // The count includes the three CPU-interface claims at `r_c` (address
    // chunks, value and is_write).
    let schema = schema(s, options);
    let claims = claim_count(s, options);
    push("mosaic.batch", (claims - 1) as f64);
    push("mosaic.sumcheck", 2.0 * schema.layout.ell as f64);
    terms
}

/// The number of Mosaic claims typed Spice makes, from the same code path
/// the prover and verifier use.
fn claim_count(s: &Statement, options: &Options) -> usize {
    let (n, a, l, b) = dims(s);
    let zeros = |len: usize| vec![K::ZERO; len];
    let v = Evaluations {
        address: zeros(a),
        previous: zeros(b),
        flags: [K::ZERO; 2],
        final_previous: zeros(b),
        initial_next_is_write: K::ZERO,
        mask_value: K::ZERO,
        mask_flags: [K::ZERO; 2],
        mask_initial_next_is_write: K::ZERO,
        previous_ab: zeros(b),
        carries_ab: zeros(b - 1),
        carries_c: zeros(b - 1),
        final_value: options.private_final_memory.then_some(K::ZERO),
    };
    let order = order_flock::OrderClaims {
        point_ab: zeros(n),
        point_c: zeros(n),
        previous_ab: v.previous_ab.clone(),
        carries_ab: v.carries_ab.clone(),
        carries_c: v.carries_c.clone(),
    };
    let point = zeros(l + 1);
    let cpu = CpuEvaluations {
        address: zeros(a),
        value: K::ZERO,
        is_write: K::ZERO,
    };
    finish_claims(
        pending_claims(s, &zeros(n), &cpu, &point, &point, &order, &v),
        &[K::ZERO; RHO_LEN],
    )
    .len()
}

pub fn verify(s: &Statement, p: &Proof) -> Result<Interface, Error> {
    verify_with(s, p, &Options::default())
}

/// Verify under explicit options. The verifier, not the proof, chooses the
/// number of fingerprints, so a one-fingerprint proof never passes as 2fp.
pub fn verify_with(s: &Statement, p: &Proof, options: &Options) -> Result<Interface, Error> {
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
        let schema = schema(&s, &Options::default());
        let populated: usize = schema.components.iter().map(|c| 1usize << c.log_size).sum();
        // 124.375 populated bits per row; the padded commitment stays 2^27.
        assert_eq!(populated, 130_416_640);
        assert_eq!(schema.layout.ell, 27);
        for p in &schema.layout.placements {
            assert_eq!(p.offset % (1usize << p.log_size), 0);
        }
    }

    #[test]
    fn private_final_memory_adds_one_word_component_and_one_claim() {
        let s = Statement {
            config: Config {
                address_bits: 16,
                chunk_bits: 1,
                ..Config::default()
            },
            log_cycles: 20,
            initial_memory: vec![0; 1 << 16],
            final_memory: Vec::new(),
        };
        let options = Options {
            private_final_memory: true,
            ..Options::default()
        };
        let schema = schema(&s, &options);
        let populated: usize = schema.components.iter().map(|c| 1usize << c.log_size).sum();
        // 64 bits per cell: 2^22 more populated bits, which crosses 2^27.
        assert_eq!(populated, 130_416_640 + (1 << 22));
        assert_eq!(schema.layout.ell, 28);
    }

    #[test]
    fn private_final_value_is_a_mosaic_claim_on_the_committed_words() {
        let config = Config {
            address_bits: 2,
            chunk_bits: 1,
            ..Config::default()
        };
        let trace = AccessTrace {
            initial: vec![7, 11, 19, 23],
            addresses: vec![0, 1, 1, 3],
            values: vec![9, 11, 4, 23],
            is_write: vec![true, false, true, false],
        };
        let options = Options {
            private_final_memory: true,
            ..Options::default()
        };
        let r = prove_with(&config, &trace, &options).unwrap();
        let p = &r.proof;
        let (n, _, l, _) = dims(&r.statement);
        let order = order_flock::OrderClaims {
            point_ab: vec![K::ZERO; n],
            point_c: vec![K::ZERO; n],
            previous_ab: p.evaluations.previous_ab.clone(),
            carries_ab: p.evaluations.carries_ab.clone(),
            carries_c: p.evaluations.carries_c.clone(),
        };
        let leaf: Vec<K> = (0..l + 1).map(|i| K::from_u128(5 + i as u128)).collect();
        let pending = pending_claims(
            &r.statement,
            &leaf[..n],
            &p.cpu,
            &leaf,
            &leaf,
            &order,
            &p.evaluations,
        );
        let claims = finish_claims(pending, &[K::ONE; RHO_LEN]);
        let finals: Vec<_> = claims
            .iter()
            .filter(|c| c.poly == "boundary/final")
            .collect();
        let a = r.statement.config.address_bits;
        let expected = word_claim(
            "boundary/final",
            &leaf[..a],
            p.evaluations.final_value.unwrap(),
            r.statement.config.word_log_bits,
        );
        assert_eq!(finals, vec![&expected]);
        let public = finish_claims(
            pending_claims(
                &r.statement,
                &leaf[..n],
                &p.cpu,
                &leaf,
                &leaf,
                &order,
                &Evaluations {
                    final_value: None,
                    ..p.evaluations.clone()
                },
            ),
            &[K::ONE; RHO_LEN],
        );
        assert_eq!(public.len() + 1, claims.len());
    }

    fn shipped(final_memory: Vec<u64>) -> Statement {
        Statement {
            config: Config {
                address_bits: 16,
                chunk_bits: 1,
                ..Config::default()
            },
            log_cycles: 20,
            initial_memory: vec![0; 1 << 16],
            final_memory,
        }
    }

    #[test]
    fn soundness_terms_match_the_issue_estimate() {
        let s = shipped(vec![0; 1 << 16]);
        let total = |terms: &[(String, f64)]| terms.iter().map(|t| t.1).sum::<f64>();
        let one = soundness_terms(&s, &Options::default());
        let get = |terms: &[(String, f64)], name: &str| {
            terms.iter().find(|t| t.0 == name).unwrap().1 * 2f64.powi(128)
        };
        // l = 20: a binary top layer, then radix-four layers with 1, 3, ..., 19 rounds.
        assert_eq!(get(&one, "gkr.batch"), 11.0);
        assert_eq!(get(&one, "gkr.sumcheck"), 400.0);
        assert_eq!(get(&one, "gkr.child-point"), 21.0);
        assert_eq!(get(&one, "flock.zerocheck"), 19.0);
        assert_eq!(get(&one, "rho"), 4.0);
        // 22 claims, three of them for the CPU interface.
        assert_eq!(get(&one, "mosaic.batch"), 21.0);
        assert_eq!(get(&one, "mosaic.sumcheck"), 54.0);
        // The fingerprint dominates: (2^20 + 2^16) / 2^128 = 2^-107.91.
        assert!(
            (total(&one).log2() + 107.9).abs() < 0.02,
            "{}",
            total(&one).log2()
        );
        let rest: f64 = one.iter().skip(1).map(|t| t.1).sum();
        assert!(rest.log2() < -118.0);
        let two = soundness_terms(&s, &Options::TWO_FINGERPRINTS);
        assert!(two[0].1.log2() < -215.0);
        assert_eq!(get(&two, "gkr.sumcheck"), 440.0);
        assert!(total(&two).log2() < -118.0);
        let private = Options {
            private_final_memory: true,
            ..Options::default()
        };
        let terms = soundness_terms(&shipped(Vec::new()), &private);
        assert_eq!(get(&terms, "mosaic.batch"), 22.0);
        assert_eq!(get(&terms, "mosaic.sumcheck"), 56.0);
    }

    #[test]
    fn claim_count_matches_the_proof() {
        let config = Config {
            address_bits: 2,
            chunk_bits: 1,
            ..Config::default()
        };
        let trace = AccessTrace {
            initial: vec![7, 11, 19, 23],
            addresses: vec![0, 1, 1, 3],
            values: vec![9, 11, 4, 23],
            is_write: vec![true, false, true, false],
        };
        for options in [
            Options::default(),
            Options {
                private_final_memory: true,
                ..Options::default()
            },
        ] {
            let r = prove_with(&config, &trace, &options).unwrap();
            assert_eq!(claim_count(&r.statement, &options), r.summary.mosaic_claims,);
        }
        // At a = 16: 19 protocol claims plus one address chunk, the value
        // and is_write at the CPU point.
        let s = shipped(vec![0; 1 << 16]);
        assert_eq!(claim_count(&s, &Options::default()), 22);
    }

    #[test]
    fn mosaic_rejects_wrong_cpu_evaluations_sent_before_absorption() {
        let config = Config {
            address_bits: 2,
            chunk_bits: 1,
            ..Config::default()
        };
        let trace = AccessTrace {
            initial: vec![7, 11, 19, 23],
            addresses: vec![0, 1, 1, 3],
            values: vec![9, 11, 4, 23],
            is_write: vec![true, false, true, false],
        };
        let options = Options::default();
        let honest = prove_with(&config, &trace, &options).unwrap();
        verify(&honest.statement, &honest.proof).unwrap();
        let edits: [&dyn Fn(&mut CpuEvaluations); 3] =
            [&|c| c.value += K::ONE, &|c| c.is_write += K::ONE, &|c| {
                c.address[1] += K::ONE
            }];
        for edit in edits {
            let r = prove_hooked(&config, &trace, &options, Some(edit)).unwrap();
            let err = verify(&r.statement, &r.proof).unwrap_err();
            assert!(format!("{err:?}").contains("mosaic"), "{err:?}");
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
