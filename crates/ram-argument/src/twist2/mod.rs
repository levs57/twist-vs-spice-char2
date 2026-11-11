//! Twist v2: the RAM argument of issue #1, §1–§2.
//!
//! The statement and committed trace are the same as the shared-address
//! upstream Twist (`crate::prove`): one read and one write per cycle at the
//! same address, committed one-hot address components `A_1..A_d`, the
//! increment `Inc` and the read value `RV`. The protocol differs:
//!
//! * a read check with no write term (`read.rs`, items 1–3);
//! * one cycle sum-check for the value claim, the final-memory claim at the
//!   same address point, the virtualisation of `ra` and Wambsgans' mirror
//!   test, which replaces the degree-4 one-hot check (`cycle.rs`, items 4–6);
//! * word-level final-memory evaluation in the verifier, no evaluation points
//!   in the proof and no address-bit claims (§2);
//! * a one-pass witness builder (item 7).
//!
//! The soundness argument is in `docs/soundness-twist.md`.

pub mod cycle;
pub mod read;
pub(crate) mod util;

use std::collections::BTreeMap;

use om_field::K;
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

use crate::{AccessTrace, ComponentSummary, RamError, Summary};
use util::{SplitEq, WordEval, word_mle};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    /// `RV(r_o, r_c)`.
    pub rv: K,
    pub read: ReadProof,
    /// `Mf(r_o, r_k)` when final memory is committed.
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
    let iop = 1 + read + cycle + usize::from(proof.final_eval.is_some());
    ProofSize {
        pcs_bytes: pcs_bytes(&proof.commitment, &proof.opening),
        mosaic_bytes: mosaic_bytes(&proof.mosaic),
        iop_bytes: 16 * iop,
    }
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
        components.push(ComponentSummary {
            name: "input/rv0".into(),
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

fn statement_transcript(statement: &Statement, schema: &Schema) -> Transcript {
    let config = &statement.config;
    let mut tr = Transcript::new_default(b"ram-argument/twist-v2");
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
    let words = |w: &[u64]| -> Vec<u8> { w.iter().flat_map(|x| x.to_le_bytes()).collect() };
    tr.absorb_bytes(b"ram/initial", &words(&statement.initial_memory));
    if let Some(f) = &statement.final_memory {
        tr.absorb_bytes(b"ram/final", &words(f));
    }
    tr
}

/// The witness in one pass (issue item 7). With supplied read values, `Inc`
/// is a parallel map and only stores update the final memory. Without them
/// the trace is replayed once.
pub(crate) struct Witness {
    pub addr: Vec<u32>,
    pub rv: Vec<u64>,
    pub inc: Vec<u64>,
    pub final_memory: Vec<u64>,
}

pub(crate) fn build_witness(trace: &AccessTrace) -> Witness {
    let addr = trace.writes.clone();
    match &trace.read_values {
        Some(values) => {
            let rv = values[0].clone();
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
            Witness {
                addr,
                rv,
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
            Witness {
                addr,
                rv,
                inc,
                final_memory: memory,
            }
        }
    }
}

fn post_commit_points(tr: &mut Transcript, ell: usize, n: usize) -> (Vec<K>, Vec<K>) {
    let r_o = tr.challenge_ks(b"ram/point_o", ell);
    // Nonzero so the split-eq read rounds can always derive q(1).
    let r_c = (0..n)
        .map(|_| tr.challenge_nonzero_k(b"ram/point_c"))
        .collect();
    (r_o, r_c)
}

#[allow(clippy::too_many_arguments)]
fn output_claims(
    schema: &Schema,
    config: &Config,
    r_o: &[K],
    r_c: &[K],
    rv: K,
    r_k: &[K],
    z: &[K],
    cycle: &CycleProof,
    final_eval: Option<K>,
) -> Vec<Claim> {
    let c = config.chunk_bits;
    let join = |a: &[K], b: &[K]| -> Vec<K> { a.iter().chain(b).copied().collect() };
    let mut claims = vec![schema.claim("input/rv0", join(r_o, r_c), rv)];
    for i in 0..config.components() {
        let rk = cycle::chunk_point(r_k, c, i);
        claims.push(schema.claim(&comp_name(i), join(rk, z), cycle.a[i]));
        claims.push(schema.claim(
            &comp_name(i),
            join(&cycle::mirrored(rk), z),
            cycle.a_mirror[i],
        ));
    }
    claims.push(schema.claim("ram/inc", join(r_o, z), cycle.inc));
    if let Some(v) = final_eval {
        claims.push(schema.claim("ram/final", join(r_o, r_k), v));
    }
    claims
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
        build_witness(trace)
    };
    prove_witness(config, n, &trace.initial, witness, schema)
}

pub(crate) fn prove_witness(
    config: &Config,
    n: usize,
    initial: &[u64],
    witness: Witness,
    schema: Schema,
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
        let comps = AddrComps::from_addrs(witness.addr.clone(), config.address_bits, d);
        for i in 0..d {
            schema.place(&mut g, &comp_name(i), &comps.packed(i));
        }
        schema.place(
            &mut g,
            "ram/inc",
            &pack_words(&witness.inc, config.word_log_bits),
        );
        schema.place(
            &mut g,
            "input/rv0",
            &pack_words(&witness.rv, config.word_log_bits),
        );
        if !config.public_final {
            schema.place(
                &mut g,
                "ram/final",
                &pack_words(&witness.final_memory, config.word_log_bits),
            );
        }
        g
    };
    let (commitment, committed) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let (r_o, r_c) = post_commit_points(&mut tr, config.word_log_bits, n);
    let (rv_e, inc_e, m0_e, store, rv) = {
        let _span = Span::new("ram.claims");
        let we = WordEval::new(&r_o);
        let rv_e = we.eval_all(&witness.rv);
        let inc_e = we.eval_all(&witness.inc);
        let m0_e = we.eval_all(initial);
        let store: Vec<bool> = witness.inc.par_iter().map(|&x| x != 0).collect();
        let eq = SplitEq::new(&r_c);
        let rv = (0..rv_e.len())
            .into_par_iter()
            .with_min_len(1 << 12)
            .map(|j| eq.at(j) * rv_e[j])
            .reduce(|| K::ZERO, |a, b| a + b);
        (rv_e, inc_e, m0_e, store, rv)
    };
    tr.absorb_k(b"ram/rv", rv);
    let read = {
        let _span = Span::new("twist.read");
        read::prove(
            &read::ReadWitness {
                addr: &witness.addr,
                rv_e: &rv_e,
                inc_e: &inc_e,
                store: &store,
                m0_e: &m0_e,
                a: config.address_bits,
            },
            &r_c,
            rv,
            config.order,
            config.merge_loads,
            &mut tr,
        )
    };
    let r_k = read.r_k.clone();
    let (m0_at, final_at) = {
        let _span = Span::new("twist.memory");
        let eq = SplitEq::new(&r_k);
        let m0_at = (0..cells_of(config))
            .map(|k| eq.at(k) * m0_e[k])
            .fold(K::ZERO, |a, b| a + b);
        (m0_at, word_mle(&witness.final_memory, &r_o, &r_k))
    };
    let final_eval = (!config.public_final).then_some(final_at);
    if let Some(v) = final_eval {
        tr.absorb_k(b"ram/final_eval", v);
    }
    let cycle = {
        let _span = Span::new("twist.cycle");
        let ch = cycle::challenges(&mut tr, d);
        let claim = cycle::claimed_sum(
            &ch,
            read.proof.val + m0_at,
            final_at + m0_at,
            read.proof.ra,
            &r_k,
            config.chunk_bits,
        );
        cycle::prove(
            &cycle::CycleWitness {
                addr: &witness.addr,
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
    let claims = output_claims(
        &schema,
        config,
        &r_o,
        &r_c,
        rv,
        &r_k,
        &cycle.z,
        &cycle.proof,
        final_eval,
    );
    let (mosaic, r_g, v_g) = {
        let _span = Span::new("mosaic.batch");
        batch_prove_packed(&schema.layout, &claims, &g, &mut tr)
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
        mosaic_claims: claims.len(),
        commitment_components: schema.components,
    };
    Ok(ProveResult {
        statement,
        final_memory: witness.final_memory,
        proof: Proof {
            commitment,
            rv,
            read: read.proof,
            final_eval,
            cycle: cycle.proof,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
    })
}

fn cells_of(config: &Config) -> usize {
    1usize << config.address_bits
}

fn verify_inner(statement: &Statement, proof: &Proof) -> Result<(), RamError> {
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
    let (r_o, r_c) = post_commit_points(&mut tr, config.word_log_bits, n);
    tr.absorb_k(b"ram/rv", proof.rv);
    let read = {
        let _span = Span::new("verify.read");
        read::verify(
            proof.rv,
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
        let m0_at = word_mle(&statement.initial_memory, &r_o, &read.r_k);
        let diff = match (&statement.final_memory, proof.final_eval) {
            (Some(f), _) => {
                let x: Vec<u64> = f
                    .iter()
                    .zip(&statement.initial_memory)
                    .map(|(a, b)| a ^ b)
                    .collect();
                word_mle(&x, &r_o, &read.r_k)
            }
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
        let claim = cycle::claimed_sum(
            &ch,
            read.val + m0_at,
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
        &r_o,
        &r_c,
        proof.rv,
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
    Ok(())
}

/// Verify a Twist v2 proof against the public statement.
pub fn verify(statement: &Statement, proof: &Proof) -> Result<(), RamError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| RamError("malformed RAM proof".into()))?
}

#[cfg(test)]
mod tests;
