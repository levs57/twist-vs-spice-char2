//! Twist v2: the RAM argument of issue #1, §1–§2.
//!
//! The trace has one read and one write per cycle at the same address, as in
//! the shared-address upstream Twist (`crate::prove`). Twist v2 commits only
//! the one-hot address components `A_1..A_d` and the increment `Inc` (plus
//! final memory when it is private). The read value is virtual, as in Jolt,
//! which commits only `RamInc` and `RamRa`: after the commitment the verifier
//! draws a CPU point `r_c`, and the proof outputs word-level claims on the
//! address, the read value `rv` and the write value `wv` at `r_c`
//! ([`Interface`]). A CPU proof would consume them; here the caller checks
//! them against the trace. The protocol has:
//!
//! * a read check with no write term on the claim `v_rv + γ_a v_addr`, with
//!   the address claim folded into the memory values (`read.rs`, items 1–3);
//! * one Mosaic claim on `Inc` at `r_c` for the write claim;
//! * one cycle sum-check for the value claim, the final-memory claim at the
//!   same address point, the virtualisation of `ra` and Wambsgans' mirror
//!   test, which replaces the degree-4 one-hot check (`cycle.rs`, items 4–6);
//! * word-level memory evaluation in the verifier, no evaluation points in
//!   the proof and no address-bit claims (§2);
//! * a one-pass witness builder (item 7).
//!
//! The soundness argument is in `docs/soundness-twist.md`.

pub mod cycle;
pub(crate) mod cycle_prover;
pub mod logup;
pub mod read;
pub(crate) mod util;

use std::borrow::Cow;
use std::collections::BTreeMap;

use om_field::{K, U256};
use om_memcheck::onehot::AddrComps;
use om_mosaic::{
    BatchProof, Claim, Layout, batch_prove_packed, batch_verify, compute_layout, place_in_g_packed,
};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, Commitment};
use om_sumcheck::PackedBits;
use om_transcript::Transcript;
use om_transcript::profiling::{self, Span};
use rayon::prelude::*;

pub use cycle::CycleProof;
pub use read::{ReadOrder, ReadProof};

use crate::word;
use crate::{AccessTrace, ComponentSummary, RamError, Summary};
use util::SplitEq;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub address_bits: usize,
    /// Bits in each address component; must divide `address_bits`.
    pub chunk_bits: usize,
    /// A word contains `2^word_log_bits` bits, at most 64.
    pub word_log_bits: usize,
    pub profile: String,
    pub log_rate: usize,
    /// Binding order of the read check. Part of the statement.
    pub order: ReadOrder,
    /// Merge loads of a cell between stores in the address-first prover
    /// (issue item 8). A prover-side choice; the proof is the same.
    pub merge_loads: bool,
    /// Final memory is public (in the statement) or committed (issue §5).
    pub public_final: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            address_bits: 8,
            chunk_bits: 4,
            word_log_bits: 6,
            profile: "provable-120".into(),
            log_rate: 2,
            order: ReadOrder::AddressFirst,
            merge_loads: true,
            public_final: true,
        }
    }
}

impl Config {
    /// The v2 configuration matching an upstream shared-address config.
    pub fn from_legacy(config: &crate::Config) -> Self {
        Self {
            address_bits: config.address_bits,
            chunk_bits: config.chunk_bits,
            word_log_bits: config.word_log_bits,
            profile: config.profile.clone(),
            log_rate: config.log_rate,
            ..Self::default()
        }
    }

    fn components(&self) -> usize {
        self.address_bits / self.chunk_bits
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub config: Config,
    pub log_cycles: usize,
    pub initial_memory: Vec<u64>,
    /// Present exactly when `config.public_final`.
    pub final_memory: Option<Vec<u64>>,
}

/// The word-level claims a CPU proof consumes, at the CPU point `r_c`:
/// `W[x](r_c) = Σ_j eq(r_c, j) enc(x_j)` for the address, read-value and
/// write-value columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuClaims {
    pub address: K,
    pub read: K,
    pub write: K,
}

/// What an accepting verifier outputs: the CPU point and the claims at it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interface {
    pub point: Vec<K>,
    pub claims: CpuClaims,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    /// The CPU claims at `r_c`.
    pub claims: CpuClaims,
    pub read: ReadProof,
    /// `W[Mf](r_k)` when final memory is committed.
    pub final_eval: Option<K>,
    pub cycle: CycleProof,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    /// Final memory, whether or not it is public.
    pub final_memory: Vec<u64>,
    pub proof: Proof,
    /// The prover's view of the interface; callers should use the one
    /// `verify` returns.
    pub interface: Interface,
    pub metrics: Vec<(&'static str, std::time::Duration)>,
    pub summary: Summary,
}

/// Proof size by part, counting 16 bytes per field element, 32 per hash and
/// 4 per index, with no length prefixes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProofSize {
    pub pcs_bytes: usize,
    pub mosaic_bytes: usize,
    pub iop_bytes: usize,
}

impl ProofSize {
    pub fn total(&self) -> usize {
        self.pcs_bytes + self.mosaic_bytes + self.iop_bytes
    }
}

pub(crate) fn pcs_bytes(commitment: &Commitment, opening: &BitProof) -> usize {
    let (rs, lig) = opening;
    let rs_elems = rs.s.len() + rs.sumcheck.round_msgs.iter().map(Vec::len).sum::<usize>();
    commitment.root.len() + 16 * rs_elems + lig.to_bytes().len()
}

pub(crate) fn mosaic_bytes(mosaic: &BatchProof) -> usize {
    16 * (1 + mosaic
        .sumcheck
        .round_msgs
        .iter()
        .map(Vec::len)
        .sum::<usize>())
}

pub fn proof_size(proof: &Proof) -> ProofSize {
    let read = 2 * (proof.read.cycle.len() + proof.read.address.len()) + 2;
    let cycle = proof
        .cycle
        .sumcheck
        .round_msgs
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        + proof.cycle.a.len()
        + proof.cycle.a_mirror.len()
        + 1;
    // Three CPU claims.
    let iop = 3 + read + cycle + usize::from(proof.final_eval.is_some());
    ProofSize {
        pcs_bytes: pcs_bytes(&proof.commitment, &proof.opening),
        mosaic_bytes: mosaic_bytes(&proof.mosaic),
        iop_bytes: 16 * iop,
    }
}

/// Soundness error terms of Twist v2 outside the PCS and ring switch, as
/// probabilities over `K = GF(2^128)`. Each term is one union-bound summand;
/// Fiat-Shamir and PCS terms are added by the caller. `docs/soundness-twist.md`
/// derives each term.
pub fn soundness_terms(statement: &Statement) -> Vec<(String, f64)> {
    let config = &statement.config;
    let (n, a, c) = (statement.log_cycles, config.address_bits, config.chunk_bits);
    let d = config.components();
    let schema = Schema::new(config, n);
    // Two per component, Inc at z, Inc at r_c, and committed final memory.
    let claims = 2 * d + 2 + usize::from(!config.public_final);
    let field = 2f64.powi(-128);
    let terms: [(&str, usize); 8] = [
        // The prover fixes v_rv and v_addr before gamma_a; if either is
        // wrong, v_rv + gamma_a v_addr equals the true sum for at most one
        // gamma_a.
        ("read.batch", 1),
        // n cycle rounds of a cubic (eq times a quadratic) and a address
        // rounds of a quadratic.
        ("read.sumcheck", 3 * n + 2 * a),
        // The word-level final-memory identity W[Mf] = W[M0] + sum ra Inc_w
        // at r_k, multilinear in a variables.
        ("final.point", a),
        // gamma, delta and eps_1..eps_d combine d + 2 identities linearly.
        ("cycle.batch", 1),
        // n rounds of degree d + 2.
        ("cycle.sumcheck", (d + 2) * n),
        // A_i(r, j) A_i(1 + r, j) - pi(r) is zero as a polynomial only for
        // unit rows, and has degree 2c in r = r_k^i; the per-row errors are
        // combined by eq(r_j, .), multilinear in n variables.
        ("mirror.point", 2 * c + n),
        // Powers of one gamma over all Mosaic claims.
        ("mosaic.batch", claims - 1),
        // ell quadratic rounds over the committed G.
        ("mosaic.sumcheck", 2 * schema.layout.ell),
    ];
    terms
        .iter()
        .map(|&(name, numerator)| (name.to_string(), numerator as f64 * field))
        .collect()
}

fn require(condition: bool, message: &str) -> Result<(), RamError> {
    if condition {
        Ok(())
    } else {
        Err(RamError(message.into()))
    }
}

fn validate_config(config: &Config, n: usize) -> Result<(), RamError> {
    require(
        (1..=31).contains(&config.address_bits),
        "address_bits must be in 1..=31",
    )?;
    require(
        config.chunk_bits > 0
            && config.chunk_bits <= config.address_bits
            && config.address_bits.is_multiple_of(config.chunk_bits),
        "chunk_bits must be a positive divisor of address_bits",
    )?;
    require(config.word_log_bits <= 6, "word_log_bits must be at most 6")?;
    require((1..=31).contains(&n), "log_cycles must be in 1..=31")?;
    require(
        config.profile == "provable-120" || config.profile == "conjectured-100",
        "unsupported Ligerito profile",
    )?;
    let minimum_rate = if config.profile == "provable-120" {
        2
    } else {
        1
    };
    require(
        (minimum_rate..=8).contains(&config.log_rate),
        "invalid Ligerito log_rate",
    )
}

fn word_fits(word: u64, log_bits: usize) -> bool {
    log_bits == 6 || word >> (1usize << log_bits) == 0
}

pub(crate) struct Schema {
    layout: Layout,
    ids: BTreeMap<String, i64>,
    components: Vec<ComponentSummary>,
    pcs: BitPCS,
}

fn comp_name(i: usize) -> String {
    format!("ram/ra0/{}", i + 1)
}

impl Schema {
    fn new(config: &Config, n: usize) -> Self {
        let mut components: Vec<ComponentSummary> = (0..config.components())
            .map(|i| ComponentSummary {
                name: comp_name(i),
                log_size: config.chunk_bits + n,
            })
            .collect();
        components.push(ComponentSummary {
            name: "ram/inc".into(),
            log_size: config.word_log_bits + n,
        });
        if !config.public_final {
            components.push(ComponentSummary {
                name: "ram/final".into(),
                log_size: config.word_log_bits + config.address_bits,
            });
        }
        let ids = components
            .iter()
            .enumerate()
            .map(|(id, c)| (c.name.clone(), id as i64))
            .collect();
        let layout = compute_layout(
            &components
                .iter()
                .enumerate()
                .map(|(id, c)| (id as i64, c.log_size))
                .collect::<Vec<_>>(),
            14,
        );
        let pcs = BitPCS::new(make_params(
            layout.ell - 7,
            &config.profile,
            config.log_rate,
        ));
        Self {
            layout,
            ids,
            components,
            pcs,
        }
    }

    fn place(&self, g: &mut PackedBits, name: &str, table: &PackedBits) {
        place_in_g_packed(g, &self.layout, self.ids[name], table);
    }

    fn place_words(&self, g: &mut PackedBits, name: &str, words: &[u64], log_bits: usize) {
        if log_bits != 6 {
            self.place(g, name, &pack_words(words, log_bits));
            return;
        }
        let placement = self
            .layout
            .placements
            .iter()
            .find(|p| p.comp_id == self.ids[name])
            .expect("word component in layout");
        assert_eq!(words.len() * 64, 1usize << placement.log_size);
        assert_eq!(placement.offset % 64, 0);
        let start = placement.offset / 64;
        g.words_mut()[start..start + words.len()].copy_from_slice(words);
    }

    fn claim(&self, name: &str, point: Vec<K>, value: K) -> Claim {
        Claim {
            comp_id: self.ids[name],
            point,
            value,
        }
    }
}

fn pack_words(words: &[u64], log_bits: usize) -> PackedBits {
    let width = 1usize << log_bits;
    if width == 64 {
        return PackedBits::from_words(words.to_vec(), words.len() * 64);
    }
    let per_word = 64 / width;
    let packed = words
        .par_chunks(per_word)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0u64, |value, (i, &word)| value | (word << (i * width)))
        })
        .collect();
    PackedBits::from_words(packed, words.len() * width)
}

fn absorb_memory(tr: &mut Transcript, label: &[u8], words: &[u64]) {
    tr.absorb_u64s(label, words);
}

fn statement_transcript(statement: &Statement, schema: &Schema) -> Transcript {
    let config = &statement.config;
    let mut tr = Transcript::new_default(b"ram-argument/twist-v2/virtual-read");
    let params = [
        config.address_bits,
        config.chunk_bits,
        config.word_log_bits,
        config.log_rate,
        statement.log_cycles,
        schema.layout.ell,
        config.order.code() as usize,
        usize::from(config.public_final),
    ];
    let bytes: Vec<u8> = params
        .iter()
        .flat_map(|&v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(b"ram/config", &bytes);
    tr.absorb_bytes(b"ram/profile", config.profile.as_bytes());
    let layout: Vec<u8> = schema
        .layout
        .placements
        .iter()
        .flat_map(|p| {
            [p.comp_id as u64, p.log_size as u64, p.offset as u64]
                .into_iter()
                .flat_map(u64::to_le_bytes)
        })
        .collect();
    tr.absorb_bytes(b"ram/layout", &layout);
    absorb_memory(&mut tr, b"ram/initial", &statement.initial_memory);
    if let Some(f) = &statement.final_memory {
        absorb_memory(&mut tr, b"ram/final", f);
    }
    tr
}

/// The witness in one pass (issue item 7). With supplied read values, `Inc`
/// is a parallel map and only stores update the final memory. Without them
/// the trace is replayed once. `rv` is prover-only: it is not committed.
#[cfg(test)]
pub(crate) struct Witness {
    pub addr: Vec<u32>,
    pub rv: Vec<u64>,
    pub inc: Vec<u64>,
    pub final_memory: Vec<u64>,
}

struct PreparedWitness<'a> {
    addr: &'a [u32],
    rv: Cow<'a, [u64]>,
    inc: Vec<u64>,
    final_memory: Vec<u64>,
}

fn prepare_witness(trace: &AccessTrace) -> PreparedWitness<'_> {
    let addr = trace.writes.as_slice();
    match &trace.read_values {
        Some(values) => {
            let rv = values[0].as_slice();
            let inc: Vec<u64> = rv
                .par_iter()
                .zip(&trace.new_values)
                .map(|(&r, &w)| r ^ w)
                .collect();
            let mut final_memory = trace.initial.clone();
            for (j, &x) in inc.iter().enumerate() {
                if x != 0 {
                    final_memory[addr[j] as usize] = trace.new_values[j];
                }
            }
            PreparedWitness {
                addr,
                rv: Cow::Borrowed(rv),
                inc,
                final_memory,
            }
        }
        None => {
            let mut memory = trace.initial.clone();
            let mut rv = Vec::with_capacity(addr.len());
            let mut inc = Vec::with_capacity(addr.len());
            for (&a, &w) in addr.iter().zip(&trace.new_values) {
                let old = memory[a as usize];
                rv.push(old);
                inc.push(old ^ w);
                memory[a as usize] = w;
            }
            PreparedWitness {
                addr,
                rv: Cow::Owned(rv),
                inc,
                final_memory: memory,
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn build_witness(trace: &AccessTrace) -> Witness {
    let witness = prepare_witness(trace);
    Witness {
        addr: witness.addr.to_vec(),
        rv: witness.rv.into_owned(),
        inc: witness.inc,
        final_memory: witness.final_memory,
    }
}

/// The CPU point `r_c`, drawn after the commitment. Nonzero so the split-eq
/// read rounds can always derive `q(1)`.
fn cpu_point(tr: &mut Transcript, n: usize) -> Vec<K> {
    (0..n)
        .map(|_| tr.challenge_nonzero_k(b"ram/point_c"))
        .collect()
}

fn absorb_claims(tr: &mut Transcript, claims: &CpuClaims) {
    tr.absorb_ks(
        b"ram/cpu_claims",
        &[claims.address, claims.read, claims.write],
    );
}

/// `W[addr](r_c)`, `W[rv](r_c)` and `W[rv ⊕ Inc](r_c)` in one pass.
fn cpu_claims(w: &PreparedWitness<'_>, r_c: &[K]) -> CpuClaims {
    let se = SplitEq::new(r_c);
    let block = |hi: usize| -> [K; 3] {
        let base = hi << se.h;
        let mut acc = [U256::ZERO; 3];
        for (lo, &e) in se.lo.iter().enumerate() {
            let j = base | lo;
            acc[0] += e.clmul(word::enc(w.addr[j] as u64));
            acc[1] += e.clmul(word::enc(w.rv[j]));
            acc[2] += e.clmul(word::enc(w.inc[j]));
        }
        acc.map(|x| se.hi[hi] * x.reduce())
    };
    let [address, read, inc] = (0..se.hi.len()).into_par_iter().map(block).reduce(
        || [K::ZERO; 3],
        |x, y| [x[0] + y[0], x[1] + y[1], x[2] + y[2]],
    );
    CpuClaims {
        address,
        read,
        write: read + inc,
    }
}

/// The Mosaic claims: `A_i` at `(r_k^i ‖ z)` and `(1 + r_k^i ‖ z)`, `Inc` at
/// `(p_w ‖ z)` and `(p_w ‖ r_c)`, and committed final memory at `(p_w ‖ r_k)`.
#[allow(clippy::too_many_arguments)]
fn output_claims(
    schema: &Schema,
    config: &Config,
    r_c: &[K],
    claims: &CpuClaims,
    r_k: &[K],
    z: &[K],
    cycle: &CycleProof,
    final_eval: Option<K>,
) -> Vec<Claim> {
    let (c, w) = (config.chunk_bits, config.word_log_bits);
    let join = |a: &[K], b: &[K]| -> Vec<K> { a.iter().chain(b).copied().collect() };
    let word_claim = |name: &str, point: &[K], value: K| {
        let (point, value) = word::claim_at(point, value, w);
        schema.claim(name, point, value)
    };
    let mut out = Vec::with_capacity(2 * config.components() + 3);
    for i in 0..config.components() {
        let rk = cycle::chunk_point(r_k, c, i);
        out.push(schema.claim(&comp_name(i), join(rk, z), cycle.a[i]));
        out.push(schema.claim(
            &comp_name(i),
            join(&cycle::mirrored(rk), z),
            cycle.a_mirror[i],
        ));
    }
    out.push(word_claim("ram/inc", z, cycle.inc));
    // The write claim: W[Inc](r_c) = v_wv + v_rv, because enc is F2-linear.
    out.push(word_claim("ram/inc", r_c, claims.write + claims.read));
    if let Some(v) = final_eval {
        out.push(word_claim("ram/final", r_k, v));
    }
    out
}

/// Prove the RAM trace with Twist v2. The trace must use one shared
/// read/write address per cycle.
pub fn prove(config: &Config, trace: &AccessTrace) -> Result<ProveResult, RamError> {
    profiling::clear();
    require(
        !trace.writes.is_empty() && trace.writes.len().is_power_of_two(),
        "cycle count must be a nonzero power of two",
    )?;
    let n = trace.writes.len().trailing_zeros() as usize;
    validate_config(config, n)?;
    let cells = 1usize << config.address_bits;
    require(
        trace.initial.len() == cells,
        "initial memory length does not match address_bits",
    )?;
    require(
        trace.reads.len() == 1 && trace.reads[0] == trace.writes,
        "Twist v2 requires one read port at the write address",
    )?;
    require(
        trace.new_values.len() == trace.writes.len(),
        "write value shape mismatch",
    )?;
    require(
        trace
            .initial
            .iter()
            .chain(&trace.new_values)
            .all(|&w| word_fits(w, config.word_log_bits)),
        "trace word exceeds configured width",
    )?;
    require(
        trace.writes.iter().all(|&a| (a as usize) < cells),
        "trace address out of range",
    )?;
    if let Some(values) = &trace.read_values {
        require(
            values.len() == 1 && values[0].len() == trace.writes.len(),
            "read value shape mismatch",
        )?;
    }
    let schema = Schema::new(config, n);
    let witness = {
        let _span = Span::new("ram.witness");
        prepare_witness(trace)
    };
    prove_prepared(config, n, &trace.initial, witness, schema, None)
}

/// Tests use `hook` to change the CPU claims before they are absorbed.
pub(crate) type ClaimsHook<'a> = Option<&'a dyn Fn(&mut CpuClaims)>;

#[cfg(test)]
pub(crate) fn prove_witness(
    config: &Config,
    n: usize,
    initial: &[u64],
    witness: Witness,
    schema: Schema,
    hook: ClaimsHook<'_>,
) -> Result<ProveResult, RamError> {
    let Witness {
        addr,
        rv,
        inc,
        final_memory,
    } = witness;
    prove_prepared(
        config,
        n,
        initial,
        PreparedWitness {
            addr: &addr,
            rv: Cow::Owned(rv),
            inc,
            final_memory,
        },
        schema,
        hook,
    )
}

fn prove_prepared(
    config: &Config,
    n: usize,
    initial: &[u64],
    witness: PreparedWitness<'_>,
    schema: Schema,
    hook: ClaimsHook<'_>,
) -> Result<ProveResult, RamError> {
    let d = config.components();
    let statement = Statement {
        config: config.clone(),
        log_cycles: n,
        initial_memory: initial.to_vec(),
        final_memory: config.public_final.then(|| witness.final_memory.clone()),
    };
    let mut tr = {
        let _span = Span::new("ram.statement");
        statement_transcript(&statement, &schema)
    };
    let g = {
        let _span = Span::new("ram.pack");
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        for i in 0..d {
            schema.place(
                &mut g,
                &comp_name(i),
                &AddrComps::packed_from_addrs(witness.addr, config.chunk_bits, i),
            );
        }
        schema.place_words(&mut g, "ram/inc", &witness.inc, config.word_log_bits);
        if !config.public_final {
            schema.place_words(
                &mut g,
                "ram/final",
                &witness.final_memory,
                config.word_log_bits,
            );
        }
        g
    };
    let (commitment, committed) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let r_c = cpu_point(&mut tr, n);
    let computed = {
        let _span = Span::new("twist.interface");
        cpu_claims(&witness, &r_c)
    };
    // The claims sent. They equal `computed` unless a test hook changes them;
    // the Mosaic claims below use `computed`, so that a test's wrong claim
    // reaches the verifier's checks instead of the Mosaic prover's
    // consistency assertion.
    let mut claims = computed;
    if let Some(hook) = hook {
        hook(&mut claims);
    }
    absorb_claims(&mut tr, &claims);
    let gamma_a = tr.challenge_k(b"ram/address_batch");
    let (rv_e, inc_e, m0_e, store) = {
        // The read check runs on V' = V + gamma_a enc(k): the memory table
        // built from M0'(k) = enc(M0[k]) + gamma_a enc(k) with the same
        // increments. Its claim is v_rv + gamma_a v_addr.
        let _span = Span::new("ram.claims");
        let (m0_e, rv_e): (Vec<K>, Vec<K>) = match config.order {
            ReadOrder::AddressFirst => (
                initial
                    .par_iter()
                    .enumerate()
                    .with_min_len(1 << 12)
                    .map(|(k, &x)| word::enc(x) + gamma_a * word::enc(k as u64))
                    .collect(),
                Vec::new(),
            ),
            ReadOrder::CycleFirst => {
                let shift: Vec<K> = (0..cells_of(config))
                    .into_par_iter()
                    .with_min_len(1 << 12)
                    .map(|k| gamma_a * word::enc(k as u64))
                    .collect();
                let m0_e = initial
                    .par_iter()
                    .zip(&shift)
                    .with_min_len(1 << 12)
                    .map(|(&x, &t)| word::enc(x) + t)
                    .collect();
                let rv_e = witness
                    .rv
                    .par_iter()
                    .zip(witness.addr.par_iter())
                    .with_min_len(1 << 12)
                    .map(|(&x, &k)| word::enc(x) + shift[k as usize])
                    .collect();
                (m0_e, rv_e)
            }
        };
        let inc_e: Vec<K> = witness
            .inc
            .par_iter()
            .with_min_len(1 << 12)
            .map(|&x| word::enc(x))
            .collect();
        let store: Vec<bool> = witness.inc.par_iter().map(|&x| x != 0).collect();
        (rv_e, inc_e, m0_e, store)
    };
    let read = {
        let _span = Span::new("twist.read");
        read::prove(
            &read::ReadWitness {
                addr: witness.addr,
                rv_e: &rv_e,
                inc_e: &inc_e,
                store: &store,
                m0_e: &m0_e,
                a: config.address_bits,
            },
            &r_c,
            claims.read + gamma_a * claims.address,
            config.order,
            config.merge_loads,
            &mut tr,
        )
    };
    drop((rv_e, m0_e, store));
    let r_k = read.r_k.clone();
    let (m0_at, final_at) = {
        let _span = Span::new("twist.memory");
        (
            word::mle(initial, &r_k),
            word::mle(&witness.final_memory, &r_k),
        )
    };
    let final_eval = (!config.public_final).then_some(final_at);
    if let Some(v) = final_eval {
        tr.absorb_k(b"ram/final_eval", v);
    }
    let cycle = {
        let _span = Span::new("twist.cycle");
        let ch = cycle::challenges(&mut tr, d);
        // V(r_k, r_j) = V'(r_k, r_j) + gamma_a enc~(r_k).
        let val = read.proof.val + gamma_a * word::enc_mle(&r_k);
        let claim = cycle::claimed_sum(
            &ch,
            val + m0_at,
            final_at + m0_at,
            read.proof.ra,
            &r_k,
            config.chunk_bits,
        );
        cycle::prove(
            &cycle::CycleWitness {
                addr: witness.addr,
                inc_e: &inc_e,
                c: config.chunk_bits,
                d,
            },
            &r_k,
            &read.r_j,
            claim,
            &ch,
            &mut tr,
        )
    };
    let mosaic_claims = output_claims(
        &schema,
        config,
        &r_c,
        &computed,
        &r_k,
        &cycle.z,
        &cycle.proof,
        final_eval,
    );
    let (mosaic, r_g, v_g) = {
        let _span = Span::new("mosaic.batch");
        batch_prove_packed(&schema.layout, &mosaic_claims, &g, &mut tr)
    };
    let (opened, opening) = {
        let _span = Span::new("pcs.open");
        schema.pcs.open_packed(committed, &g, &r_g, &mut tr)
    };
    require(opened == v_g, "Mosaic and PCS opening disagree")?;
    let summary = Summary {
        cycles: witness.addr.len(),
        memory_cells: cells_of(config),
        address_components: d,
        chunk_bits: config.chunk_bits,
        committed_bits: g.len(),
        packed_field_elements: g.len() / 128,
        mosaic_claims: mosaic_claims.len(),
        commitment_components: schema.components,
    };
    Ok(ProveResult {
        statement,
        final_memory: witness.final_memory,
        proof: Proof {
            commitment,
            claims,
            read: read.proof,
            final_eval,
            cycle: cycle.proof,
            mosaic,
            opening,
        },
        interface: Interface { point: r_c, claims },
        metrics: profiling::take(),
        summary,
    })
}

fn cells_of(config: &Config) -> usize {
    1usize << config.address_bits
}

fn verify_inner(statement: &Statement, proof: &Proof) -> Result<Interface, RamError> {
    let config = &statement.config;
    let n = statement.log_cycles;
    validate_config(config, n)?;
    let cells = cells_of(config);
    require(
        statement.initial_memory.len() == cells,
        "initial memory length does not match address_bits",
    )?;
    require(
        statement.final_memory.is_some() == config.public_final
            && statement
                .final_memory
                .as_ref()
                .is_none_or(|f| f.len() == cells),
        "final memory does not match the configuration",
    )?;
    require(
        statement
            .initial_memory
            .iter()
            .chain(statement.final_memory.iter().flatten())
            .all(|&w| word_fits(w, config.word_log_bits)),
        "memory word exceeds configured width",
    )?;
    require(
        proof.final_eval.is_some() != config.public_final,
        "final evaluation presence does not match the configuration",
    )?;
    let d = config.components();
    let schema = Schema::new(config, n);
    require(
        proof.commitment.n == schema.layout.ell - 7 && proof.commitment.root.len() == 32,
        "malformed commitment",
    )?;
    let mut tr = {
        let _span = Span::new("verify.statement");
        statement_transcript(statement, &schema)
    };
    schema.pcs.absorb_commitment(&proof.commitment, &mut tr);
    let r_c = cpu_point(&mut tr, n);
    absorb_claims(&mut tr, &proof.claims);
    let gamma_a = tr.challenge_k(b"ram/address_batch");
    let read = {
        let _span = Span::new("verify.read");
        read::verify(
            proof.claims.read + gamma_a * proof.claims.address,
            &r_c,
            config.address_bits,
            config.order,
            &proof.read,
            &mut tr,
        )?
    };
    let (m0_at, final_minus_m0) = {
        // Word-level evaluation of M0 and Mf + M0 over 2^a words (issue §2).
        let _span = Span::new("verify.memory");
        let m0_at = word::mle(&statement.initial_memory, &read.r_k);
        let diff = match (&statement.final_memory, proof.final_eval) {
            (Some(f), _) => word::mle(f, &read.r_k) + m0_at,
            (None, Some(v)) => v + m0_at,
            (None, None) => unreachable!(),
        };
        (m0_at, diff)
    };
    if let Some(v) = proof.final_eval {
        tr.absorb_k(b"ram/final_eval", v);
    }
    let z = {
        let _span = Span::new("verify.cycle");
        let ch = cycle::challenges(&mut tr, d);
        let val = read.val + gamma_a * word::enc_mle(&read.r_k);
        let claim = cycle::claimed_sum(
            &ch,
            val + m0_at,
            final_minus_m0,
            read.ra,
            &read.r_k,
            config.chunk_bits,
        );
        cycle::verify(&proof.cycle, &read.r_j, claim, &ch, d, &mut tr)?
    };
    let claims = output_claims(
        &schema,
        config,
        &r_c,
        &proof.claims,
        &read.r_k,
        &z,
        &proof.cycle,
        proof.final_eval,
    );
    let (r_g, v_g) = {
        let _span = Span::new("verify.mosaic");
        batch_verify(&schema.layout, &claims, &proof.mosaic, &mut tr)?
    };
    let _span = Span::new("verify.pcs");
    schema
        .pcs
        .verify(&proof.commitment, &r_g, v_g, &proof.opening, &mut tr)?;
    Ok(Interface {
        point: r_c,
        claims: proof.claims,
    })
}

/// Verify a Twist v2 proof against the public statement. On success, return
/// the CPU point and the claims at it, which the caller must check against
/// its own columns: the RAM argument proves them only relative to the
/// committed trace.
pub fn verify(statement: &Statement, proof: &Proof) -> Result<Interface, RamError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| RamError("malformed RAM proof".into()))?
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod preparation_tests {
    use super::*;

    #[test]
    fn borrowed_memory_bytes_preserve_exact_transcript() {
        for words in [vec![], vec![0], vec![1, u64::MAX, 0x0123_4567_89ab_cdef]] {
            let mut actual = Transcript::new_default(b"memory-borrow-oracle");
            let mut expected = actual.clone();
            absorb_memory(&mut actual, b"memory", &words);
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            expected.absorb_bytes(b"memory", &bytes);
            assert_eq!(actual, expected);
            assert_eq!(actual.challenge_k(b"after"), expected.challenge_k(b"after"));
        }
    }

    #[test]
    fn prepared_trace_borrows_inputs_and_direct_word_placement_matches_packed_bits() {
        let trace = AccessTrace {
            initial: vec![1, 2, 3, 4],
            reads: vec![vec![0, 1, 0, 3]],
            writes: vec![0, 1, 0, 3],
            new_values: vec![5, 2, 7, 0],
            read_values: Some(vec![vec![1, 2, 5, 4]]),
        };
        let witness = prepare_witness(&trace);
        assert_eq!(witness.addr.as_ptr(), trace.writes.as_ptr());
        assert!(matches!(&witness.rv, Cow::Borrowed(_)));
        assert_eq!(
            witness.rv.as_ptr(),
            trace.read_values.as_ref().unwrap()[0].as_ptr()
        );
        assert_eq!(witness.inc, vec![4, 0, 2, 4]);
        assert_eq!(witness.final_memory, vec![7, 2, 3, 0]);
        for word_log_bits in 0..=6 {
            let config = Config {
                address_bits: 2,
                chunk_bits: 1,
                word_log_bits,
                ..Config::default()
            };
            let schema = Schema::new(&config, 2);
            let mask = if word_log_bits == 6 {
                u64::MAX
            } else {
                (1u64 << (1usize << word_log_bits)) - 1
            };
            let words = [1 & mask, 0, 0x1234_5678_9abc_def0 & mask, mask];
            let mut direct = PackedBits::zero(1usize << schema.layout.ell);
            let mut expected = direct.clone();
            schema.place_words(&mut direct, "ram/inc", &words, word_log_bits);
            schema.place(&mut expected, "ram/inc", &pack_words(&words, word_log_bits));
            assert_eq!(direct, expected, "word_log_bits={word_log_bits}");
        }
    }
}
