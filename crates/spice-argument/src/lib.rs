//! Ordered linked-list RAM, using the specified Spice-style permutation.
//!
//! Private predecessor pointers and ripple carries are authenticated by BitPCS.
//! `prev < timestamp` replaces the counter column. Public memory and shared-address
//! semantics match ram-argument. This GF128 benchmark configuration has a
//! size-dependent grand-product soundness bound, not 128-bit total security.

pub mod masked_reduction;
pub mod operations;
pub mod order;
pub mod order_flock;
pub mod planes;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;
use std::time::Duration;

use lean_grandproduct::{GrandProductProof, LeafClaims, Leaves};
use om_field::{K, mul_pairs};
use om_memcheck::onehot::EvalClaim;
use om_memcheck::twist::word_mle_eval;
use om_mosaic::{
    BatchProof, Claim, Layout, batch_prove_packed, batch_verify, compute_layout, place_in_g_packed,
};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, Commitment};
use om_sumcheck::{PackedBits, VerificationError};
use om_transcript::{
    Transcript,
    profiling::{self, Span},
};
use rayon::prelude::*;

pub use ram_argument::{AccessTrace, ComponentSummary, Config, Statement, Summary};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<VerificationError> for Error {
    fn from(e: VerificationError) -> Self {
        Self(e.message().into())
    }
}
impl From<lean_grandproduct::Error> for Error {
    fn from(e: lean_grandproduct::Error) -> Self {
        Self(e.to_string())
    }
}
impl From<order::OrderError> for Error {
    fn from(e: order::OrderError) -> Self {
        Self(e.to_string())
    }
}
fn require(ok: bool, message: &str) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputEvaluations {
    pub address: Vec<K>,
    pub previous: Vec<K>,
    pub final_previous: Vec<K>,
    pub read: K,
    pub write: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderingMode {
    FlockBoolean,
    Ripple,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderingProof {
    Flock(order_flock::OrderProof),
    Ripple(order::OrderProof),
}

impl OrderingProof {
    pub fn mode(&self) -> OrderingMode {
        match self {
            Self::Flock(_) => OrderingMode::FlockBoolean,
            Self::Ripple(_) => OrderingMode::Ripple,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    pub permutation: GrandProductProof,
    pub inputs: InputEvaluations,
    pub order: OrderingProof,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    pub proof: Proof,
    pub metrics: Vec<(&'static str, Duration)>,
    pub summary: Summary,
}

fn dims(s: &Statement) -> (usize, usize, usize, usize) {
    let n = s.log_cycles;
    let a = s.config.address_bits;
    let l = n.max(a);
    (n, a, l, l + 1)
}

fn validate(s: &Statement) -> Result<(), Error> {
    validate_memory(s, true)
}

/// With `public_final` false the statement's final memory is ignored: it is
/// neither checked here nor read by the verifier.
fn validate_memory(s: &Statement, public_final: bool) -> Result<(), Error> {
    let c = &s.config;
    require(
        (1..=31).contains(&c.address_bits) && s.log_cycles <= 31,
        "invalid RAM dimensions",
    )?;
    require(
        c.shared_address && c.read_ports == 1,
        "Spice requires one shared-address read/write port",
    )?;
    require(c.word_log_bits <= 6, "word_log_bits must be at most 6")?;
    require(
        c.profile == "provable-120" || c.profile == "conjectured-100",
        "unsupported PCS profile",
    )?;
    require(
        (if c.profile == "provable-120" { 2 } else { 1 }..=8).contains(&c.log_rate),
        "invalid PCS log_rate",
    )?;
    require(
        s.initial_memory.len() == 1usize << c.address_bits
            && (!public_final || s.final_memory.len() == s.initial_memory.len()),
        "public memory shape mismatch",
    )?;
    let word_fits = |&v: &u64| c.word_log_bits == 6 || v >> (1usize << c.word_log_bits) == 0;
    let final_memory: &[u64] = if public_final { &s.final_memory } else { &[] };
    require(
        s.initial_memory.iter().chain(final_memory).all(word_fits),
        "public memory word exceeds configured width",
    )
}

struct Schema {
    layout: Layout,
    ids: BTreeMap<String, i64>,
    components: Vec<ComponentSummary>,
    pcs: BitPCS,
}
impl Schema {
    // Same Mosaic/BitPCS placement and opening pipeline as ram-argument and
    // lookup-argument; no alternate commitment implementation is introduced.
    fn new(s: &Statement) -> Self {
        let (n, a, _, w) = dims(s);
        let mut components = vec![
            ComponentSummary {
                name: "input/read".into(),
                log_size: n + s.config.word_log_bits,
            },
            ComponentSummary {
                name: "input/write".into(),
                log_size: n + s.config.word_log_bits,
            },
        ];
        for (prefix, count, log_size) in [
            ("input/address", a, n),
            ("input/previous", w, n),
            ("order/carry", w - 1, n),
            ("boundary/previous", w, a),
        ] {
            components.extend((0..count).map(|i| ComponentSummary {
                name: format!("{prefix}/{i}"),
                log_size,
            }));
        }
        let ids = components
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
        Self {
            layout,
            ids,
            components,
            pcs,
        }
    }
    fn place(&self, g: &mut PackedBits, name: &str, bits: &PackedBits) {
        place_in_g_packed(g, &self.layout, self.ids[name], bits);
    }
    fn claims(&self, claims: &[EvalClaim]) -> Result<Vec<Claim>, Error> {
        require(
            claims.len() <= om_mosaic::MAX_CLAIMS,
            "too many Mosaic claims",
        )?;
        claims
            .iter()
            .map(|c| {
                self.ids
                    .get(&c.poly)
                    .map(|&i| c.to_mosaic_claim(i))
                    .ok_or_else(|| Error("unknown component".into()))
            })
            .collect()
    }
}

fn transcript(s: &Statement, schema: &Schema, ordering: &OrderingMode) -> Transcript {
    transcript_with(s, schema, ordering, true, "spice.statement")
}

/// The statement transcript. Final memory is absorbed only when public.
fn transcript_with(
    s: &Statement,
    schema: &Schema,
    ordering: &OrderingMode,
    public_final: bool,
    span: &'static str,
) -> Transcript {
    let _span = Span::new(span);
    let mut tr = Transcript::new_default(b"ordered-spice-ram/gf128/v2");
    tr.absorb_bytes(
        b"spice/ordering",
        match ordering {
            OrderingMode::FlockBoolean => b"flock-boolean/v1",
            OrderingMode::Ripple => b"ripple-eq-factor/v1",
        },
    );
    tr.absorb_bytes(
        b"spice/config",
        &[
            s.config.address_bits,
            s.log_cycles,
            s.config.word_log_bits,
            s.config.log_rate,
            schema.layout.ell,
            s.config.chunk_bits,
            s.config.read_ports,
            usize::from(s.config.shared_address),
        ]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect::<Vec<_>>(),
    );
    tr.absorb_bytes(
        b"spice/config/mode",
        format!("{:?}", s.config.virtualization).as_bytes(),
    );
    tr.absorb_bytes(b"spice/profile", s.config.profile.as_bytes());
    let empty = Vec::new();
    for (label, values) in [
        (b"spice/initial".as_slice(), &s.initial_memory),
        if public_final {
            (b"spice/final".as_slice(), &s.final_memory)
        } else {
            (b"spice/final-committed".as_slice(), &empty)
        },
    ] {
        tr.absorb_bytes(
            label,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
    }
    tr
}

/// Bit planes `0..bits` of `words`, each as a packed column.
///
/// Each group of 64 words is transposed once with two 32x32 bit-matrix
/// transposes, so the input is read once instead of once per plane.
fn pack_columns(words: &[u32], bits: usize) -> Vec<PackedBits> {
    assert!(bits <= 32);
    let chunks = words.len().div_ceil(64);
    let mut planes = vec![0u64; chunks * bits];
    if bits > 0 {
        planes
            .par_chunks_mut(bits)
            .zip(words.par_chunks(64))
            .for_each(|(out, chunk)| {
                let mut halves = [[0u32; 32]; 2];
                for (i, &w) in chunk.iter().enumerate() {
                    halves[i / 32][i % 32] = w;
                }
                halves.iter_mut().for_each(transpose32);
                for (b, plane) in out.iter_mut().enumerate() {
                    let half = |h: &[u32; 32]| u64::from(h[31 - b].reverse_bits());
                    *plane = half(&halves[0]) | half(&halves[1]) << 32;
                }
            });
    }
    (0..bits)
        .into_par_iter()
        .map(|b| {
            PackedBits::from_words(
                (0..chunks).map(|c| planes[c * bits + b]).collect(),
                words.len(),
            )
        })
        .collect()
}

/// Transpose a 32x32 bit matrix whose row `r` is `a[r]`, most significant
/// bit first (Hacker's Delight, figure 7-6): afterwards bit `31 - c` of
/// `a[r]` is the old bit `31 - r` of `a[c]`.
fn transpose32(a: &mut [u32; 32]) {
    let mut j = 16;
    let mut m: u32 = 0x0000_ffff;
    while j != 0 {
        let mut k = 0;
        while k < 32 {
            let t = (a[k] ^ (a[k + j] >> j)) & m;
            a[k] ^= t;
            a[k + j] ^= t << j;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

fn pack_values(values: &[u64], log_bits: usize) -> PackedBits {
    if log_bits == 6 {
        return PackedBits::from_words(values.to_vec(), values.len() * 64);
    }
    let width = 1usize << log_bits;
    let mut words = vec![0u64; (values.len() * width).div_ceil(64)];
    for (i, &v) in values.iter().enumerate() {
        words[i * width / 64] |= v << (i * width % 64);
    }
    PackedBits::from_words(words, values.len() * width)
}
fn column_claims(prefix: &str, values: &[K], point: &[K]) -> Vec<EvalClaim> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| EvalClaim {
            poly: format!("{prefix}/{i}"),
            point: point.to_vec(),
            value,
        })
        .collect()
}
fn word_claim(name: &str, point: &[K], value: K, log_bits: usize) -> EvalClaim {
    let (mut p, factor) = word_encoding(log_bits);
    p.extend_from_slice(point);
    EvalClaim {
        poly: name.into(),
        point: p,
        value: value / factor,
    }
}
fn word_eval(values: &[u64], point: &[K], log_bits: usize) -> K {
    let (p, factor) = word_encoding(log_bits);
    factor * word_mle_eval(values, &p, point)
}
fn word_encoding(log_bits: usize) -> (Vec<K>, K) {
    static ENCODINGS: OnceLock<Vec<(Vec<K>, K)>> = OnceLock::new();
    ENCODINGS.get_or_init(|| {
        (0..=6)
            .map(|log_bits| {
                let mut power = K::X;
                let mut factor = K::ONE;
                let mut p = Vec::new();
                for _ in 0..log_bits {
                    let sum = K::ONE + power;
                    factor *= sum;
                    p.push(power / sum);
                    power = power.square();
                }
                (p, factor)
            })
            .collect()
    })[log_bits]
        .clone()
}
fn combine_bits(values: &[K]) -> K {
    values
        .iter()
        .enumerate()
        .fold(K::ZERO, |s, (i, &v)| s + K::from_u128(1u128 << i) * v)
}
fn eval_columns(columns: &[PackedBits], point: &[K]) -> Vec<K> {
    // Share the equality table across bitplanes, rather than rebuilding it for
    // every requested predecessor/address bit (the PCS tables stay Boolean).
    let weights = om_sumcheck::eq_table(point);
    eval_columns_weighted(columns, &weights)
}
fn eval_columns_weighted(columns: &[PackedBits], weights: &[K]) -> Vec<K> {
    columns
        .par_iter()
        .map(|column| {
            column
                .words()
                .iter()
                .enumerate()
                .fold(K::ZERO, |mut sum, (i, &word)| {
                    let mut word = word;
                    while word != 0 {
                        let b = word.trailing_zeros() as usize;
                        sum += weights[64 * i + b];
                        word &= word - 1;
                    }
                    sum
                })
        })
        .collect()
}

fn source_claims(s: &Statement, p: &[K], inputs: &InputEvaluations) -> Vec<EvalClaim> {
    let (n, a, _, _) = dims(s);
    let mut claims = column_claims("input/address", &inputs.address, &p[..n]);
    claims.extend(column_claims("input/previous", &inputs.previous, &p[..n]));
    claims.extend(column_claims(
        "boundary/previous",
        &inputs.final_previous,
        &p[..a],
    ));
    claims.push(word_claim(
        "input/read",
        &p[..n],
        inputs.read,
        s.config.word_log_bits,
    ));
    claims.push(word_claim(
        "input/write",
        &p[..n],
        inputs.write,
        s.config.word_log_bits,
    ));
    claims
}
fn absorb_inputs(tr: &mut Transcript, v: &InputEvaluations) {
    tr.absorb_ks(b"spice/address", &v.address);
    tr.absorb_ks(b"spice/previous", &v.previous);
    tr.absorb_ks(b"spice/final-previous", &v.final_previous);
    tr.absorb_ks(b"spice/values", &[v.read, v.write]);
}

fn ripple_order_claims(terminal: &order::OrderClaims) -> Vec<EvalClaim> {
    let mut claims = column_claims("input/previous", &terminal.previous, &terminal.point);
    claims.extend(column_claims(
        "order/carry",
        &terminal.carries,
        &terminal.point,
    ));
    claims
}

fn flock_order_claims(terminal: &order_flock::OrderClaims) -> Vec<EvalClaim> {
    let mut claims = column_claims("input/previous", &terminal.previous_ab, &terminal.point_ab);
    claims.extend(column_claims(
        "order/carry",
        &terminal.carries_ab,
        &terminal.point_ab,
    ));
    // Flock extracts the linear C column in the skip round. Its cycle point
    // differs from A/B, so this second group is a real authenticated opening.
    claims.extend(column_claims(
        "order/carry",
        &terminal.carries_c,
        &terminal.point_c,
    ));
    claims
}

fn check_leaves(
    s: &Statement,
    leaves: &LeafClaims,
    v: &InputEvaluations,
    gamma: K,
    alpha: K,
    beta: K,
) -> Result<(), Error> {
    let (n, a, l, w) = dims(s);
    require(
        leaves.point.len() == l + 1
            && v.address.len() == a
            && v.previous.len() == w
            && v.final_previous.len() == w,
        "malformed permutation terminals",
    )?;
    let p = &leaves.point;
    let cycle_mask = p[n..l].iter().fold(K::ONE, |v, &r| v * (K::ONE + r));
    let memory_mask = p[a..l].iter().fold(K::ONE, |v, &r| v * (K::ONE + r));
    let boundary = p[l];
    let addr = combine_bits(&v.address);
    let time = K::from_u128(1u128 << l) + combine_bits(&p[..n]);
    let memory_addr = combine_bits(&p[..a]);
    let cycles = [
        gamma + combine_bits(&v.previous) + alpha * addr + beta * v.read,
        gamma + time + alpha * addr + beta * v.write,
    ];
    let memories = [
        gamma
            + combine_bits(&v.final_previous)
            + alpha * memory_addr
            + beta * word_eval(&s.final_memory, &p[..a], s.config.word_log_bits),
        gamma
            + memory_addr
            + alpha * memory_addr
            + beta * word_eval(&s.initial_memory, &p[..a], s.config.word_log_bits),
    ];
    for side in 0..2 {
        let expected = (K::ONE + boundary) * (K::ONE + cycle_mask * (K::ONE + cycles[side]))
            + boundary * (K::ONE + memory_mask * (K::ONE + memories[side]));
        require(
            expected == leaves.values[side],
            "permutation leaf does not match authenticated inputs",
        )?;
    }
    Ok(())
}

pub fn prove(config: &Config, trace: &AccessTrace) -> Result<ProveResult, Error> {
    prove_with_ordering(config, trace, OrderingMode::FlockBoolean)
}

/// The default uses Flock's packed Boolean argument. Ripple is retained only
/// as an explicitly selected performance/reference baseline.
pub fn prove_with_ordering(
    config: &Config,
    trace: &AccessTrace,
    ordering: OrderingMode,
) -> Result<ProveResult, Error> {
    profiling::clear();
    let t = trace.writes.len();
    require(
        t.is_power_of_two(),
        "cycle count must be a nonzero power of two",
    )?;
    let n = t.trailing_zeros() as usize;
    let mut statement = Statement {
        config: config.clone(),
        log_cycles: n,
        initial_memory: trace.initial.clone(),
        final_memory: trace.initial.clone(),
    };
    validate(&statement)?;
    require(
        trace.reads.len() == 1 && trace.reads[0] == trace.writes && trace.new_values.len() == t,
        "shared-address trace shape mismatch",
    )?;
    let (_, a, l, w) = dims(&statement);
    let cells = 1usize << a;
    require(
        trace.writes.iter().all(|&v| (v as usize) < cells),
        "address out of range",
    )?;
    let (reads, previous, last) = {
        let _span = Span::new("spice.witness");
        let mut memory = trace.initial.clone();
        let mut last: Vec<u32> = (0..cells as u32).collect();
        let mut previous = Vec::with_capacity(t);
        let mut reads = Vec::with_capacity(t);
        for (j, (&addr, &value)) in trace.writes.iter().zip(&trace.new_values).enumerate() {
            require(
                config.word_log_bits == 6 || value >> (1usize << config.word_log_bits) == 0,
                "write word exceeds configured width",
            )?;
            let k = addr as usize;
            previous.push(last[k]);
            reads.push(memory[k]);
            last[k] = (1u32 << l) | j as u32;
            memory[k] = value;
        }
        statement.final_memory = memory;
        (reads, previous, last)
    };
    if let Some(values) = &trace.read_values {
        require(
            values.len() == 1 && values[0] == reads,
            "incorrect supplied read values",
        )?;
    }
    let schema = Schema::new(&statement);
    let mut tr = transcript(&statement, &schema, &ordering);
    // Carry bit j is c_(j+1); endpoints c0=1 and cw=0 remain public.
    let carries = {
        let _span = Span::new("spice.witness");
        order::build_carries(&previous, n, a)?
    };
    let (address_columns, previous_columns, last_columns, g) = {
        let _span = Span::new("spice.pack");
        let addresses = pack_columns(&trace.writes, a);
        let prev = pack_columns(&previous, w);
        let carry = pack_columns(&carries, w - 1);
        let last = pack_columns(&last, w);
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        schema.place(
            &mut g,
            "input/read",
            &pack_values(&reads, config.word_log_bits),
        );
        schema.place(
            &mut g,
            "input/write",
            &pack_values(&trace.new_values, config.word_log_bits),
        );
        for (prefix, columns) in [
            ("input/address", &addresses),
            ("input/previous", &prev),
            ("order/carry", &carry),
            ("boundary/previous", &last),
        ] {
            for (i, column) in columns.iter().enumerate() {
                schema.place(&mut g, &format!("{prefix}/{i}"), column);
            }
        }
        (addresses, prev, last, g)
    };
    let (commitment, state) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let gamma = tr.challenge_k(b"spice/fingerprint/shift");
    let alpha = tr.challenge_k(b"spice/fingerprint/address");
    let beta = tr.challenge_k(b"spice/fingerprint/value");
    let permutation_inputs = {
        let _span = Span::new("spice.permutation.leaves");
        let half = 1usize << l;
        let padding: Vec<_> = [t..half, half + cells..2 * half]
            .into_iter()
            .filter(|r| !r.is_empty())
            .collect();
        // Reuse the donor's tiled joint leaf/fours/sixteens schedule. Declared
        // padding never invokes the row callback and a known suffix is implicit.
        Leaves::from_fn_pair(2 * half, padding, |row| {
            if row < half {
                let j = row;
                let common = gamma + alpha * K::from_u128(trace.writes[j] as u128);
                let values = mul_pairs(
                    [beta; 2],
                    [
                        K::from_u128(reads[j] as u128),
                        K::from_u128(trace.new_values[j] as u128),
                    ],
                );
                [
                    common + K::from_u128(previous[j] as u128) + values[0],
                    common + K::from_u128(((1usize << l) | j) as u128) + values[1],
                ]
            } else {
                let k = row - half;
                let addr = K::from_u128(k as u128);
                let common = gamma + alpha * addr;
                let values = mul_pairs(
                    [beta; 2],
                    [
                        K::from_u128(statement.final_memory[k] as u128),
                        K::from_u128(statement.initial_memory[k] as u128),
                    ],
                );
                [
                    common + K::from_u128(last[k] as u128) + values[0],
                    common + addr + values[1],
                ]
            }
        })?
    };
    let (permutation, leaves) = {
        let _span = Span::new("spice.permutation.gkr");
        lean_grandproduct::prove(l + 1, permutation_inputs, &mut tr, b"spice/permutation")?
    };
    require(
        permutation.roots[0] == permutation.roots[1],
        "RAM tuple products do not agree",
    )?;
    let inputs = {
        let _span = Span::new("spice.permutation.claims");
        let cycle_weights = om_sumcheck::eq_table(&leaves.point[..n]);
        InputEvaluations {
            address: eval_columns_weighted(&address_columns, &cycle_weights),
            previous: eval_columns_weighted(&previous_columns, &cycle_weights),
            final_previous: eval_columns(&last_columns, &leaves.point[..a]),
            read: word_eval(&reads, &leaves.point[..n], config.word_log_bits),
            write: word_eval(&trace.new_values, &leaves.point[..n], config.word_log_bits),
        }
    };
    check_leaves(&statement, &leaves, &inputs, gamma, alpha, beta)?;
    absorb_inputs(&mut tr, &inputs);
    let mut claims = source_claims(&statement, &leaves.point, &inputs);
    let order = {
        let _span = Span::new("spice.order");
        match ordering {
            OrderingMode::FlockBoolean => {
                let (proof, terminal) =
                    order_flock::prove(&previous, &carries, n, a, &mut tr, b"spice/order")?;
                claims.extend(flock_order_claims(&terminal));
                OrderingProof::Flock(proof)
            }
            OrderingMode::Ripple => {
                let (proof, terminal) =
                    order::prove(&previous, &carries, n, a, &mut tr, b"spice/order")?;
                claims.extend(ripple_order_claims(&terminal));
                OrderingProof::Ripple(proof)
            }
        }
    };
    let (mosaic, r, value) = {
        let _span = Span::new("mosaic.batch");
        batch_prove_packed(&schema.layout, &schema.claims(&claims)?, &g, &mut tr)
    };
    let (opened, opening) = {
        let _span = Span::new("pcs.open");
        schema.pcs.open_packed(state, &g, &r, &mut tr)
    };
    require(opened == value, "Mosaic/PCS mismatch")?;
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
            inputs,
            order,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
    })
}

fn verify_inner(statement: &Statement, proof: &Proof) -> Result<(), Error> {
    validate(statement)?;
    let (n, a, l, _) = dims(statement);
    let schema = Schema::new(statement);
    require(
        proof.commitment.n == schema.layout.ell - 7 && proof.commitment.root.len() == 32,
        "malformed commitment",
    )?;
    let mut tr = transcript(statement, &schema, &proof.order.mode());
    schema.pcs.absorb_commitment(&proof.commitment, &mut tr);
    let gamma = tr.challenge_k(b"spice/fingerprint/shift");
    let alpha = tr.challenge_k(b"spice/fingerprint/address");
    let beta = tr.challenge_k(b"spice/fingerprint/value");
    require(
        proof.permutation.roots[0] == proof.permutation.roots[1],
        "RAM tuple products do not agree",
    )?;
    let leaves =
        lean_grandproduct::verify(l + 1, &proof.permutation, &mut tr, b"spice/permutation")?;
    check_leaves(statement, &leaves, &proof.inputs, gamma, alpha, beta)?;
    absorb_inputs(&mut tr, &proof.inputs);
    let mut claims = source_claims(statement, &leaves.point, &proof.inputs);
    match &proof.order {
        OrderingProof::Flock(p) => claims.extend(flock_order_claims(&order_flock::verify(
            n,
            a,
            p,
            &mut tr,
            b"spice/order",
        )?)),
        OrderingProof::Ripple(p) => claims.extend(ripple_order_claims(&order::verify(
            n,
            a,
            p,
            &mut tr,
            b"spice/order",
        )?)),
    }
    let (r, value) = batch_verify(
        &schema.layout,
        &schema.claims(&claims)?,
        &proof.mosaic,
        &mut tr,
    )?;
    schema
        .pcs
        .verify(&proof.commitment, &r, value, &proof.opening, &mut tr)?;
    Ok(())
}

pub fn verify(statement: &Statement, proof: &Proof) -> Result<(), Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| Error("malformed Spice proof".into()))?
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    #[test]
    fn transposed_packing_matches_per_bit_extraction() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (len, bits) in [(1, 1), (5, 3), (64, 32), (100, 21), (4096, 17), (256, 0)] {
            let words: Vec<u32> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed as u32) & (((1u64 << bits) - 1) as u32)
                })
                .collect();
            let columns = pack_columns(&words, bits);
            assert_eq!(columns.len(), bits);
            for (b, column) in columns.iter().enumerate() {
                assert_eq!(column.len(), len);
                for (i, &w) in words.iter().enumerate() {
                    assert_eq!(
                        column.words()[i / 64] >> (i % 64) & 1 == 1,
                        w >> b & 1 == 1,
                        "len {len} bit {b} row {i}"
                    );
                }
            }
        }
    }
}
