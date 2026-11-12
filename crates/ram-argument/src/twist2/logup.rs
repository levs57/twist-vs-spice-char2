//! Twist's existing read reduction with a direct LogUp* address pushforward.
//!
//! Main commitment: binary address columns and Inc, plus private final memory.
//! The cubic cycle check reduces the value/final/read-selector claims to
//! `f(z) = sum_j Eq(z,j) Eq(r_k,addr_j)`. Commit the field-valued pushforward
//! `M[k] = sum_{j:addr_j=k} Eq(z,j)` before its denominator challenge, prove
//! the weighted rational identity, and open M directly at r_k. There is no
//! dualization or separate inner-product sumcheck: `M(r_k)` is exactly f(z).
//! Both M evaluations and the rational tree's source address leaves reach PCS.

use std::borrow::Cow;
use std::collections::BTreeMap;

use bitz_logup::{
    FractionTreeWitness, RationalClaims, RationalProof, prove_rational, verify_rational,
};
use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_memcheck::twist::{lt_eval, lt_table};
use om_mosaic::{
    BatchProof, Claim, Layout, batch_prove_packed, batch_verify, compute_layout, place_in_g_packed,
};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, BitStatePacked, Commitment};
use om_sumcheck::{PackedBits, SumcheckProof, eq_eval, eq_table, sumcheck_init, sumcheck_verify};
use om_transcript::{
    Transcript,
    profiling::{self, Span},
};
use rayon::prelude::*;

use super::util::{PAR, SplitEq, checks_enabled, emit_round, zeros};
use super::{Config, CpuClaims, Interface, PreparedWitness, ReadOrder, ReadProof, Statement};
use crate::{AccessTrace, ComponentSummary, RamError, Summary, word};

const CYCLE_LABEL: &[u8] = b"ram/logup/cycle";
const RATIONAL_LABEL: &[u8] = b"ram/logup/rational";
const PUSHFORWARD: &str = "ram/logup/pushforward";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleProof {
    pub sumcheck: SumcheckProof,
    /// `sum_j Eq(z,j) Eq(r_k,addr_j)`.
    pub ra: K,
    /// Word-level Inc(z).
    pub inc: K,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticationProof {
    pub commitment: Commitment,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PushforwardProof {
    pub rational: RationalProof,
    pub address_evals: Vec<K>,
    pub auxiliary: AuthenticationProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    pub claims: CpuClaims,
    pub read: ReadProof,
    pub final_eval: Option<K>,
    pub cycle: CycleProof,
    pub pushforward: PushforwardProof,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    pub final_memory: Vec<u64>,
    pub proof: Proof,
    pub interface: Interface,
    pub metrics: Vec<(&'static str, std::time::Duration)>,
    pub summary: Summary,
    /// The post-challenge pushforward has its own commitment and opening.
    pub auxiliary_summary: Summary,
}

fn address_name(bit: usize) -> String {
    format!("ram/address/{bit}")
}

struct Schema {
    layout: Layout,
    ids: BTreeMap<String, i64>,
    components: Vec<ComponentSummary>,
    pcs: BitPCS,
}

impl Schema {
    fn new(config: &Config, components: Vec<ComponentSummary>) -> Self {
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

    fn main(config: &Config, n: usize) -> Self {
        let mut components: Vec<_> = (0..config.address_bits)
            .map(|bit| ComponentSummary {
                name: address_name(bit),
                log_size: n,
            })
            .collect();
        components.push(ComponentSummary {
            name: "ram/inc".into(),
            log_size: n + config.word_log_bits,
        });
        if !config.public_final {
            components.push(ComponentSummary {
                name: "ram/final".into(),
                log_size: config.address_bits + config.word_log_bits,
            });
        }
        Self::new(config, components)
    }

    fn auxiliary(config: &Config) -> Self {
        Self::new(
            config,
            vec![ComponentSummary {
                name: PUSHFORWARD.into(),
                log_size: config.address_bits + 7,
            }],
        )
    }

    fn place(&self, g: &mut PackedBits, name: &str, bits: &PackedBits) {
        place_in_g_packed(g, &self.layout, self.ids[name], bits);
    }

    fn place_words(&self, g: &mut PackedBits, name: &str, words: &[u64], log_bits: usize) {
        if log_bits != 6 {
            self.place(g, name, &super::pack_words(words, log_bits));
            return;
        }
        let p = self
            .layout
            .placements
            .iter()
            .find(|p| p.comp_id == self.ids[name])
            .expect("word component");
        assert_eq!(p.offset % 64, 0);
        assert_eq!(words.len() * 64, 1usize << p.log_size);
        g.words_mut()[p.offset / 64..p.offset / 64 + words.len()].copy_from_slice(words);
    }

    fn place_fields(&self, g: &mut PackedBits, values: &[K]) {
        let p = self
            .layout
            .placements
            .iter()
            .find(|p| p.comp_id == self.ids[PUSHFORWARD])
            .expect("pushforward component");
        assert_eq!(p.offset % 128, 0);
        assert_eq!(values.len() * 128, 1usize << p.log_size);
        g.words_mut()[p.offset / 64..p.offset / 64 + 2 * values.len()]
            .par_chunks_mut(2)
            .zip(values.par_iter())
            .for_each(|(dst, value)| {
                dst[0] = value.value() as u64;
                dst[1] = (value.value() >> 64) as u64;
            });
    }

    fn claim(&self, name: &str, point: Vec<K>, value: K) -> Claim {
        Claim {
            comp_id: self.ids[name],
            point,
            value,
        }
    }

    fn word_claim(&self, name: &str, point: &[K], value: K, log_bits: usize) -> Claim {
        let (point, value) = word::claim_at(point, value, log_bits);
        self.claim(name, point, value)
    }

    fn valid_commitment(&self, cm: &Commitment) -> Result<(), RamError> {
        super::require(
            cm.n == self.layout.ell - 7 && cm.root.len() == 32,
            "malformed LogUp address commitment",
        )
    }

    fn open(
        &self,
        g: &PackedBits,
        state: BitStatePacked,
        claims: &[Claim],
        tr: &mut Transcript,
        auxiliary: bool,
    ) -> Result<(BatchProof, BitProof), RamError> {
        let (mosaic, point, value) = {
            let _span = Span::new(if auxiliary {
                "mosaic.aux.batch"
            } else {
                "mosaic.batch"
            });
            batch_prove_packed(&self.layout, claims, g, tr)
        };
        let (opened, opening) = {
            let _span = Span::new(if auxiliary {
                "logup.aux.open"
            } else {
                "pcs.open"
            });
            self.pcs.open_packed(state, g, &point, tr)
        };
        super::require(
            opened == value,
            "LogUp address Mosaic/PCS evaluation mismatch",
        )?;
        Ok((mosaic, opening))
    }

    fn verify(
        &self,
        auth: &AuthenticationProof,
        claims: &[Claim],
        tr: &mut Transcript,
    ) -> Result<(), RamError> {
        self.valid_commitment(&auth.commitment)?;
        let (point, value) = batch_verify(&self.layout, claims, &auth.mosaic, tr)?;
        self.pcs
            .verify(&auth.commitment, &point, value, &auth.opening, tr)?;
        Ok(())
    }

    fn summary(
        &self,
        cycles: usize,
        memory_cells: usize,
        address_components: usize,
        claims: usize,
    ) -> Summary {
        Summary {
            cycles,
            memory_cells,
            address_components,
            chunk_bits: 0,
            committed_bits: 1usize << self.layout.ell,
            packed_field_elements: (1usize << self.layout.ell) / 128,
            mosaic_claims: claims,
            commitment_components: self.components.clone(),
        }
    }
}

fn transcript(statement: &Statement, main: &Schema, aux: &Schema) -> Transcript {
    let c = &statement.config;
    let mut tr = Transcript::new_default(b"ram-argument/twist-v2/logup-address/v1");
    let dimensions = [
        c.address_bits,
        c.chunk_bits,
        c.word_log_bits,
        c.log_rate,
        statement.log_cycles,
        main.layout.ell,
        aux.layout.ell,
        c.order.code() as usize,
        usize::from(c.public_final),
    ];
    let bytes: Vec<_> = dimensions
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(b"ram/config", &bytes);
    tr.absorb_bytes(b"ram/profile", c.profile.as_bytes());
    for (label, schema) in [
        (b"ram/main-layout".as_slice(), main),
        (b"ram/aux-layout".as_slice(), aux),
    ] {
        let bytes: Vec<_> = schema
            .layout
            .placements
            .iter()
            .flat_map(|p| {
                [p.comp_id as u64, p.log_size as u64, p.offset as u64]
                    .into_iter()
                    .flat_map(u64::to_le_bytes)
            })
            .collect();
        tr.absorb_bytes(label, &bytes);
    }
    super::absorb_memory(&mut tr, b"ram/initial", &statement.initial_memory);
    if let Some(memory) = &statement.final_memory {
        super::absorb_memory(&mut tr, b"ram/final", memory);
    }
    tr
}

/// Coefficients of three structured dot products. For a pair, P=f*Inc is
/// quadratic. Its endpoints and leading coefficient cost three multiplies,
/// rather than evaluating four generic cubic nodes. Equality and LT suffixes
/// stay split, and carryless dot accumulators reduce once per weight block.
#[derive(Clone, Default)]
struct Dots {
    lt: [K; 3],
    eq: [K; 3],
    f: [K; 2],
}

impl Dots {
    fn add(mut self, other: Self) -> Self {
        for i in 0..3 {
            self.lt[i] += other.lt[i];
            self.eq[i] += other.eq[i];
        }
        for i in 0..2 {
            self.f[i] += other.f[i];
        }
        self
    }
}

struct RoundWeights {
    eq: SplitEq,
    lt_lo: Vec<K>,
    lt_hi: Vec<K>,
}

impl RoundWeights {
    fn new(suffix: &[K]) -> Self {
        let eq = SplitEq::new(suffix);
        let lt_lo = lt_table(&suffix[..eq.h]);
        let lt_hi = lt_table(&suffix[eq.h..]);
        Self { eq, lt_lo, lt_hi }
    }
}

fn round_dots(
    f: &[K],
    inc: &[K],
    bind: Option<K>,
    weights: &RoundWeights,
    gamma: K,
    out_f: &mut Vec<K>,
    out_inc: &mut Vec<K>,
) -> Dots {
    assert_eq!(f.len(), inc.len());
    let stride = if bind.is_some() { 4 } else { 2 };
    let block = stride * weights.eq.lo.len();
    let process = |sf: &[K],
                   si: &[K],
                   mut df: Option<&mut [K]>,
                   mut di: Option<&mut [K]>,
                   hi: usize|
     -> Dots {
        let mut plain = [K::ZERO; 3];
        let mut lt = [U256::ZERO; 3];
        let mut eq = [U256::ZERO; 3];
        let mut ef = [U256::ZERO; 2];
        for (j, (&e, &l)) in weights.eq.lo.iter().zip(&weights.lt_lo).enumerate() {
            let (fa, fb, ia, ib) = if let Some(r) = bind {
                let at = stride * j;
                let [fa, fb] = mul_pairs([r; 2], [sf[at] + sf[at + 1], sf[at + 2] + sf[at + 3]]);
                let [ia, ib] = mul_pairs([r; 2], [si[at] + si[at + 1], si[at + 2] + si[at + 3]]);
                let (fa, fb, ia, ib) = (sf[at] + fa, sf[at + 2] + fb, si[at] + ia, si[at + 2] + ib);
                if let Some(ref mut dst) = df {
                    dst[2 * j] = fa;
                    dst[2 * j + 1] = fb;
                }
                if let Some(ref mut dst) = di {
                    dst[2 * j] = ia;
                    dst[2 * j + 1] = ib;
                }
                (fa, fb, ia, ib)
            } else {
                (sf[2 * j], sf[2 * j + 1], si[2 * j], si[2 * j + 1])
            };
            // As in Twist's existing cycle kernel, unchanged accesses make
            // the complete value/final product vanish for this pair. The
            // selector's Eq-weighted claim still contributes below.
            if !(ia.is_zero() && ib.is_zero()) {
                let [p0, p1] = mul_pairs([fa, fb], [ia, ib]);
                let pinf = (fa + fb) * (ia + ib);
                for (i, p) in [p0, p1, pinf].into_iter().enumerate() {
                    plain[i] += p;
                    let [lp, ep] = clmul_pairs([l, e], [p; 2]);
                    lt[i] += lp;
                    eq[i] += ep;
                }
            }
            let [f0, fd] = clmul_pairs([e; 2], [fa, fa + fb]);
            ef[0] += f0;
            ef[1] += fd;
        }
        let h = weights.eq.hi[hi];
        let l = weights.lt_hi[hi] + gamma;
        let mut out = Dots::default();
        for i in 0..3 {
            let [left, right] = mul_pairs([l, h], [plain[i], lt[i].reduce()]);
            out.lt[i] = left + right;
            out.eq[i] = h * eq[i].reduce();
        }
        for i in 0..2 {
            out.f[i] = h * ef[i].reduce();
        }
        // Endpoints -> ordinary monomial coefficients of P.
        out.lt[1] += out.lt[0] + out.lt[2];
        out.eq[1] += out.eq[0] + out.eq[2];
        out
    };
    if bind.is_some() {
        out_f.resize(f.len() / 2, K::ZERO);
        out_inc.resize(inc.len() / 2, K::ZERO);
        if f.len() >= PAR {
            f.par_chunks(block)
                .zip(inc.par_chunks(block))
                .zip(out_f.par_chunks_mut(block / 2))
                .zip(out_inc.par_chunks_mut(block / 2))
                .enumerate()
                .map(|(hi, (((sf, si), df), di))| process(sf, si, Some(df), Some(di), hi))
                .reduce(Dots::default, Dots::add)
        } else {
            f.chunks(block)
                .zip(inc.chunks(block))
                .zip(out_f.chunks_mut(block / 2))
                .zip(out_inc.chunks_mut(block / 2))
                .enumerate()
                .map(|(hi, (((sf, si), df), di))| process(sf, si, Some(df), Some(di), hi))
                .fold(Dots::default(), Dots::add)
        }
    } else if f.len() >= PAR {
        f.par_chunks(block)
            .zip(inc.par_chunks(block))
            .enumerate()
            .map(|(hi, (sf, si))| process(sf, si, None, None, hi))
            .reduce(Dots::default, Dots::add)
    } else {
        f.chunks(block)
            .zip(inc.chunks(block))
            .enumerate()
            .map(|(hi, (sf, si))| process(sf, si, None, None, hi))
            .fold(Dots::default(), Dots::add)
    }
}

fn cubic_values(dots: &Dots, ri: K, lt_prefix: K, eq_prefix: K, delta: K) -> [K; 4] {
    let l0 = ri + (K::ONE + ri) * lt_prefix;
    let l1 = ri + lt_prefix;
    let scale = delta * eq_prefix;
    let v0 = scale * (K::ONE + ri) * dots.f[0];
    let v1 = scale * ((K::ONE + ri) * dots.f[1] + dots.f[0]);
    let v2 = scale * dots.f[1];
    let c = [
        dots.lt[0] + l0 * dots.eq[0] + v0,
        dots.lt[1] + l0 * dots.eq[1] + l1 * dots.eq[0] + v1,
        dots.lt[2] + l0 * dots.eq[2] + l1 * dots.eq[1] + v2,
        l1 * dots.eq[2],
    ];
    let evaluate = |x: K| c[0] + x * (c[1] + x * (c[2] + x * c[3]));
    [
        c[0],
        c[0] + c[1] + c[2] + c[3],
        evaluate(K::from_u128(2)),
        evaluate(K::from_u128(3)),
    ]
}

fn cycle_challenges(tr: &mut Transcript) -> (K, K) {
    (
        tr.challenge_k(b"ram/logup/cycle/gamma"),
        tr.challenge_k(b"ram/logup/cycle/delta"),
    )
}

fn prove_cycle(
    addr: &[u32],
    inc: &[K],
    r_k: &[K],
    r_j: &[K],
    claim: K,
    gamma: K,
    delta: K,
    tr: &mut Transcript,
) -> (CycleProof, Vec<K>) {
    let n = r_j.len();
    let selector = eq_table(r_k);
    let mut f: Vec<_> = addr.par_iter().map(|&ad| selector[ad as usize]).collect();
    drop(selector);
    let mut inc: Cow<'_, [K]> = Cow::Borrowed(inc);
    let (mut out_f, mut out_inc) = (Vec::new(), Vec::new());
    sumcheck_init(tr, CYCLE_LABEL, claim, n, 3);
    let mut msgs = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut current = claim;
    let (mut lt_prefix, mut eq_prefix) = (K::ZERO, K::ONE);
    let mut dots = round_dots(
        &f,
        &inc,
        None,
        &RoundWeights::new(&r_j[1..]),
        gamma,
        &mut out_f,
        &mut out_inc,
    );
    for i in 0..n {
        let values = cubic_values(&dots, r_j[i], lt_prefix, eq_prefix, delta);
        if checks_enabled() {
            assert_eq!(
                values[0] + values[1],
                current,
                "direct-address cycle round {i}"
            );
        }
        let (r, next) = emit_round(&values, tr, CYCLE_LABEL, &mut msgs);
        z.push(r);
        current = next;
        let eq = K::ONE + r_j[i] + r;
        lt_prefix = lt_prefix * eq + (K::ONE + r) * r_j[i];
        eq_prefix *= eq;
        if i + 1 == n {
            let [ff, ii] = mul_pairs([r; 2], [f[0] + f[1], inc[0] + inc[1]]);
            let (ff, ii) = (f[0] + ff, inc[0] + ii);
            if checks_enabled() {
                assert_eq!(
                    current,
                    (lt_prefix + gamma) * ff * ii + delta * eq_prefix * ff,
                    "direct-address cycle terminal"
                );
            }
            tr.absorb_ks(b"ram/logup/cycle/evals", &[ff, ii]);
            return (
                CycleProof {
                    sumcheck: SumcheckProof { round_msgs: msgs },
                    ra: ff,
                    inc: ii,
                },
                z,
            );
        }
        dots = round_dots(
            &f,
            &inc,
            Some(r),
            &RoundWeights::new(&r_j[i + 2..]),
            gamma,
            &mut out_f,
            &mut out_inc,
        );
        std::mem::swap(&mut f, &mut out_f);
        let previous = std::mem::replace(&mut inc, Cow::Owned(std::mem::take(&mut out_inc)));
        if let Cow::Owned(previous) = previous {
            out_inc = previous;
        }
    }
    unreachable!("configuration requires at least one cycle variable")
}

fn verify_cycle(
    r_j: &[K],
    claim: K,
    gamma: K,
    delta: K,
    proof: &CycleProof,
    tr: &mut Transcript,
) -> Result<Vec<K>, RamError> {
    let (z, current) = sumcheck_verify(claim, r_j.len(), 3, &proof.sumcheck, tr, CYCLE_LABEL)?;
    let expected =
        (lt_eval(&z, r_j) + gamma) * proof.ra * proof.inc + delta * eq_eval(&z, r_j) * proof.ra;
    super::require(
        current == expected,
        "direct-address cycle final evaluation mismatch",
    )?;
    tr.absorb_ks(b"ram/logup/cycle/evals", &[proof.ra, proof.inc]);
    Ok(z)
}

fn pushforward(addr: &[u32], weights: &[K], cells: usize) -> Vec<K> {
    let workers = rayon::current_num_threads().max(1);
    if addr.len() >= PAR && cells * workers <= 4 * addr.len() {
        let per = addr.len().div_ceil(workers);
        addr.par_chunks(per)
            .zip(weights.par_chunks(per))
            .map(|(addresses, weights)| {
                let mut result = zeros(cells);
                for (&ad, &weight) in addresses.iter().zip(weights) {
                    result[ad as usize] += weight;
                }
                result
            })
            .reduce_with(|mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += b;
                }
                a
            })
            .expect("nonempty validated access trace")
    } else {
        let mut result = zeros(cells);
        for (&ad, &weight) in addr.iter().zip(weights) {
            result[ad as usize] += weight;
        }
        result
    }
}

fn bit_evals(addr: &[u32], point: &[K], bits: usize) -> Vec<K> {
    let weights = SplitEq::new(point);
    let block = |start: usize, addresses: &[u32]| {
        let mut out = vec![K::ZERO; bits];
        for (offset, &address) in addresses.iter().enumerate() {
            let value = weights.at(start + offset);
            let mut address = address;
            while address != 0 {
                let bit = address.trailing_zeros() as usize;
                out[bit] += value;
                address &= address - 1;
            }
        }
        out
    };
    addr.par_chunks(PAR)
        .enumerate()
        .map(|(i, chunk)| block(i * PAR, chunk))
        .reduce(
            || vec![K::ZERO; bits],
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b) {
                    *a += b;
                }
                a
            },
        )
}

fn rational_leaves(
    z: &[K],
    tau: K,
    leaves: &RationalClaims,
    address_evals: &[K],
) -> Result<(), RamError> {
    let address = address_evals
        .iter()
        .enumerate()
        .fold(tau, |sum, (bit, &value)| {
            sum + K::from_u128(1u128 << bit) * value
        });
    super::require(
        leaves.left.num == eq_eval(z, &leaves.left.point),
        "LogUp source weight mismatch",
    )?;
    super::require(leaves.left.den == address, "LogUp source address mismatch")?;
    super::require(
        leaves.right.den == tau + word::enc_mle(&leaves.right.point),
        "LogUp table index mismatch",
    )
}

fn aux_claims(schema: &Schema, r_k: &[K], ra: K, leaves: &RationalClaims) -> Vec<Claim> {
    vec![
        schema.word_claim(PUSHFORWARD, r_k, ra, 7),
        schema.word_claim(PUSHFORWARD, &leaves.right.point, leaves.right.num, 7),
    ]
}

fn main_claims(
    schema: &Schema,
    c: &Config,
    r_c: &[K],
    cpu: &CpuClaims,
    r_k: &[K],
    z: &[K],
    cycle: &CycleProof,
    source_point: &[K],
    address_evals: &[K],
    final_eval: Option<K>,
) -> Vec<Claim> {
    let mut out: Vec<_> = address_evals
        .iter()
        .enumerate()
        .map(|(bit, &value)| schema.claim(&address_name(bit), source_point.to_vec(), value))
        .collect();
    out.push(schema.word_claim("ram/inc", z, cycle.inc, c.word_log_bits));
    out.push(schema.word_claim("ram/inc", r_c, cpu.read + cpu.write, c.word_log_bits));
    if let Some(value) = final_eval {
        out.push(schema.word_claim("ram/final", r_k, value, c.word_log_bits));
    }
    out
}

fn map_bitz(error: bitz_logup::Error) -> RamError {
    RamError(format!("LogUp address reduction: {error}"))
}

fn prove_pushforward(
    c: &Config,
    addr: &[u32],
    z: &[K],
    r_k: &[K],
    ra: K,
    schema: &Schema,
    tr: &mut Transcript,
) -> Result<(PushforwardProof, Vec<K>), RamError> {
    let _span = Span::new("logup.pushforward");
    let weights = {
        let _span = Span::new("logup.pushforward.weights");
        eq_table(z)
    };
    let values = {
        let _span = Span::new("logup.pushforward.histogram");
        pushforward(addr, &weights, 1usize << c.address_bits)
    };
    let g = {
        let _span = Span::new("logup.aux.pack");
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        schema.place_fields(&mut g, &values);
        g
    };
    tr.absorb_bytes(b"ram/logup/auxiliary", b"weighted-pushforward/v1");
    let (commitment, state) = {
        let _span = Span::new("logup.aux.commit");
        schema.pcs.commit_packed(&g, tr)
    };
    let tau = tr.challenge_k(b"ram/logup/tau");
    super::require(
        tau.value() >= 1u128 << c.address_bits,
        "LogUp denominator challenge is a table index",
    )?;
    let (mut left, mut right) = {
        let _span = Span::new("logup.fraction.witness");
        let source_den: Vec<_> = addr
            .par_iter()
            .map(|&address| tau + word::enc(address as u64))
            .collect();
        let table_den: Vec<_> = (0..values.len())
            .into_par_iter()
            .map(|k| tau + word::enc(k as u64))
            .collect();
        let (mut left, mut right) = (
            FractionTreeWitness::default(),
            FractionTreeWitness::default(),
        );
        left.rebuild(&weights, &source_den, z.len(), K::ZERO, K::ONE)
            .map_err(map_bitz)?;
        right
            .rebuild(&values, &table_den, c.address_bits, K::ZERO, K::ONE)
            .map_err(map_bitz)?;
        (left, right)
    };
    drop((weights, values));
    let (rational, leaves) = {
        let _span = Span::new("logup.rational");
        prove_rational(&mut left, &mut right, tr, RATIONAL_LABEL).map_err(map_bitz)?
    };
    drop((left, right));
    let address_evals = {
        let _span = Span::new("logup.address.claims");
        bit_evals(addr, &leaves.left.point, c.address_bits)
    };
    rational_leaves(z, tau, &leaves, &address_evals)?;
    tr.absorb_ks(b"ram/logup/address-evals", &address_evals);
    let (mosaic, opening) =
        schema.open(&g, state, &aux_claims(schema, r_k, ra, &leaves), tr, true)?;
    Ok((
        PushforwardProof {
            rational,
            address_evals,
            auxiliary: AuthenticationProof {
                commitment,
                mosaic,
                opening,
            },
        },
        leaves.left.point,
    ))
}

fn verify_pushforward(
    c: &Config,
    n: usize,
    z: &[K],
    r_k: &[K],
    ra: K,
    schema: &Schema,
    proof: &PushforwardProof,
    tr: &mut Transcript,
) -> Result<Vec<K>, RamError> {
    super::require(
        proof.address_evals.len() == c.address_bits,
        "malformed LogUp address evaluations",
    )?;
    schema.valid_commitment(&proof.auxiliary.commitment)?;
    tr.absorb_bytes(b"ram/logup/auxiliary", b"weighted-pushforward/v1");
    schema
        .pcs
        .absorb_commitment(&proof.auxiliary.commitment, tr);
    let tau = tr.challenge_k(b"ram/logup/tau");
    super::require(
        tau.value() >= 1u128 << c.address_bits,
        "LogUp denominator challenge is a table index",
    )?;
    let leaves = verify_rational(n, c.address_bits, &proof.rational, tr, RATIONAL_LABEL)
        .map_err(map_bitz)?;
    rational_leaves(z, tau, &leaves, &proof.address_evals)?;
    tr.absorb_ks(b"ram/logup/address-evals", &proof.address_evals);
    schema.verify(&proof.auxiliary, &aux_claims(schema, r_k, ra, &leaves), tr)?;
    Ok(leaves.left.point)
}

pub fn prove(c: &Config, trace: &AccessTrace) -> Result<ProveResult, RamError> {
    profiling::clear();
    super::require(
        !trace.writes.is_empty() && trace.writes.len().is_power_of_two(),
        "cycle count must be a nonzero power of two",
    )?;
    let n = trace.writes.len().trailing_zeros() as usize;
    super::validate_config(c, n)?;
    let cells = 1usize << c.address_bits;
    super::require(
        trace.initial.len() == cells,
        "initial memory length does not match address_bits",
    )?;
    super::require(
        trace.reads.len() == 1 && trace.reads[0] == trace.writes,
        "LogUp Twist requires one shared read/write address",
    )?;
    super::require(
        trace.new_values.len() == trace.writes.len(),
        "write value shape mismatch",
    )?;
    super::require(
        trace
            .initial
            .iter()
            .chain(&trace.new_values)
            .all(|&word| super::word_fits(word, c.word_log_bits)),
        "trace word exceeds configured width",
    )?;
    super::require(
        trace
            .writes
            .iter()
            .all(|&address| (address as usize) < cells),
        "trace address out of range",
    )?;
    if let Some(values) = &trace.read_values {
        super::require(
            values.len() == 1 && values[0].len() == trace.writes.len(),
            "read value shape mismatch",
        )?;
    }
    let witness = {
        let _span = Span::new("ram.witness");
        super::prepare_witness(trace)
    };
    prove_prepared(c, n, &trace.initial, witness, None)
}

fn prove_prepared(
    c: &Config,
    n: usize,
    initial: &[u64],
    w: PreparedWitness<'_>,
    hook: super::ClaimsHook<'_>,
) -> Result<ProveResult, RamError> {
    let (main, aux) = (Schema::main(c, n), Schema::auxiliary(c));
    let statement = Statement {
        config: c.clone(),
        log_cycles: n,
        initial_memory: initial.to_vec(),
        final_memory: c.public_final.then(|| w.final_memory.clone()),
    };
    let mut tr = {
        let _span = Span::new("ram.statement");
        transcript(&statement, &main, &aux)
    };
    let g = {
        let _span = Span::new("ram.pack");
        let mut g = PackedBits::zero(1usize << main.layout.ell);
        for bit in 0..c.address_bits {
            let packed = w
                .addr
                .par_chunks(64)
                .map(|chunk| {
                    chunk.iter().enumerate().fold(0u64, |word, (j, &address)| {
                        word | (u64::from((address >> bit) & 1) << j)
                    })
                })
                .collect();
            main.place(
                &mut g,
                &address_name(bit),
                &PackedBits::from_words(packed, w.addr.len()),
            );
        }
        main.place_words(&mut g, "ram/inc", &w.inc, c.word_log_bits);
        if !c.public_final {
            main.place_words(&mut g, "ram/final", &w.final_memory, c.word_log_bits);
        }
        g
    };
    let (commitment, committed) = {
        let _span = Span::new("pcs.commit");
        main.pcs.commit_packed(&g, &mut tr)
    };
    let r_c = super::cpu_point(&mut tr, n);
    let computed = {
        let _span = Span::new("twist.interface");
        super::cpu_claims(&w, &r_c)
    };
    let mut claims = computed;
    if let Some(hook) = hook {
        hook(&mut claims);
    }
    super::absorb_claims(&mut tr, &claims);
    let gamma_a = tr.challenge_k(b"ram/address_batch");
    let (rv_e, inc_e, m0_e, store) = {
        let _span = Span::new("ram.claims");
        let m0_e = initial
            .par_iter()
            .enumerate()
            .map(|(k, &value)| word::enc(value) + gamma_a * word::enc(k as u64))
            .collect::<Vec<_>>();
        let rv_e = if c.order == ReadOrder::CycleFirst {
            w.rv.par_iter()
                .zip(w.addr.par_iter())
                .map(|(&value, &address)| word::enc(value) + gamma_a * word::enc(address as u64))
                .collect()
        } else {
            Vec::new()
        };
        let inc_e = w
            .inc
            .par_iter()
            .map(|&value| word::enc(value))
            .collect::<Vec<_>>();
        let store = w
            .inc
            .par_iter()
            .map(|&value| value != 0)
            .collect::<Vec<_>>();
        (rv_e, inc_e, m0_e, store)
    };
    let read = {
        let _span = Span::new("twist.read");
        super::read::prove(
            &super::read::ReadWitness {
                addr: w.addr,
                rv_e: &rv_e,
                inc_e: &inc_e,
                store: &store,
                m0_e: &m0_e,
                a: c.address_bits,
            },
            &r_c,
            claims.read + gamma_a * claims.address,
            c.order,
            c.merge_loads,
            &mut tr,
        )
    };
    drop((rv_e, m0_e, store));
    let r_k = &read.r_k;
    let (m0_at, final_at) = {
        let _span = Span::new("twist.memory");
        (word::mle(initial, r_k), word::mle(&w.final_memory, r_k))
    };
    let final_eval = (!c.public_final).then_some(final_at);
    if let Some(value) = final_eval {
        tr.absorb_k(b"ram/final_eval", value);
    }
    let (cycle, z) = {
        let _span = Span::new("twist.cycle");
        let (gamma, delta) = cycle_challenges(&mut tr);
        let val = read.proof.val + gamma_a * word::enc_mle(r_k);
        let claim = val + m0_at + gamma * (final_at + m0_at) + delta * read.proof.ra;
        prove_cycle(w.addr, &inc_e, r_k, &read.r_j, claim, gamma, delta, &mut tr)
    };
    drop(inc_e);
    let (pushforward, source_point) =
        prove_pushforward(c, w.addr, &z, r_k, cycle.ra, &aux, &mut tr)?;
    let main_claims = main_claims(
        &main,
        c,
        &r_c,
        &computed,
        r_k,
        &z,
        &cycle,
        &source_point,
        &pushforward.address_evals,
        final_eval,
    );
    let (mosaic, opening) = main.open(&g, committed, &main_claims, &mut tr, false)?;
    let summary = main.summary(
        w.addr.len(),
        1usize << c.address_bits,
        c.address_bits,
        main_claims.len(),
    );
    let auxiliary_summary = aux.summary(1usize << c.address_bits, 1usize << c.address_bits, 0, 2);
    Ok(ProveResult {
        statement,
        final_memory: w.final_memory,
        proof: Proof {
            commitment,
            claims,
            read: read.proof,
            final_eval,
            cycle,
            pushforward,
            mosaic,
            opening,
        },
        interface: Interface { point: r_c, claims },
        metrics: profiling::take(),
        summary,
        auxiliary_summary,
    })
}

fn verify_inner(statement: &Statement, proof: &Proof) -> Result<Interface, RamError> {
    let c = &statement.config;
    let n = statement.log_cycles;
    super::validate_config(c, n)?;
    let cells = 1usize << c.address_bits;
    super::require(
        statement.initial_memory.len() == cells,
        "initial memory shape mismatch",
    )?;
    super::require(
        statement.final_memory.is_some() == c.public_final
            && statement
                .final_memory
                .as_ref()
                .is_none_or(|memory| memory.len() == cells),
        "final memory shape mismatch",
    )?;
    super::require(
        statement
            .initial_memory
            .iter()
            .chain(statement.final_memory.iter().flatten())
            .all(|&word| super::word_fits(word, c.word_log_bits)),
        "memory word exceeds configured width",
    )?;
    super::require(
        proof.final_eval.is_some() != c.public_final,
        "final evaluation presence mismatch",
    )?;
    let (main, aux) = (Schema::main(c, n), Schema::auxiliary(c));
    main.valid_commitment(&proof.commitment)?;
    let mut tr = {
        let _span = Span::new("verify.statement");
        transcript(statement, &main, &aux)
    };
    main.pcs.absorb_commitment(&proof.commitment, &mut tr);
    let r_c = super::cpu_point(&mut tr, n);
    super::absorb_claims(&mut tr, &proof.claims);
    let gamma_a = tr.challenge_k(b"ram/address_batch");
    let read = {
        let _span = Span::new("verify.read");
        super::read::verify(
            proof.claims.read + gamma_a * proof.claims.address,
            &r_c,
            c.address_bits,
            c.order,
            &proof.read,
            &mut tr,
        )?
    };
    let (m0_at, final_diff) = {
        let _span = Span::new("verify.memory");
        let m0 = word::mle(&statement.initial_memory, &read.r_k);
        let final_at = match (&statement.final_memory, proof.final_eval) {
            (Some(memory), _) => word::mle(memory, &read.r_k),
            (None, Some(value)) => value,
            _ => unreachable!(),
        };
        (m0, final_at + m0)
    };
    if let Some(value) = proof.final_eval {
        tr.absorb_k(b"ram/final_eval", value);
    }
    let z = {
        let _span = Span::new("verify.cycle");
        let (gamma, delta) = cycle_challenges(&mut tr);
        let val = read.val + gamma_a * word::enc_mle(&read.r_k);
        verify_cycle(
            &read.r_j,
            val + m0_at + gamma * final_diff + delta * read.ra,
            gamma,
            delta,
            &proof.cycle,
            &mut tr,
        )?
    };
    let source_point = {
        let _span = Span::new("verify.logup");
        verify_pushforward(
            c,
            n,
            &z,
            &read.r_k,
            proof.cycle.ra,
            &aux,
            &proof.pushforward,
            &mut tr,
        )?
    };
    let claims = main_claims(
        &main,
        c,
        &r_c,
        &proof.claims,
        &read.r_k,
        &z,
        &proof.cycle,
        &source_point,
        &proof.pushforward.address_evals,
        proof.final_eval,
    );
    let (point, value) = {
        let _span = Span::new("verify.mosaic");
        batch_verify(&main.layout, &claims, &proof.mosaic, &mut tr)?
    };
    {
        let _span = Span::new("verify.pcs");
        main.pcs
            .verify(&proof.commitment, &point, value, &proof.opening, &mut tr)?;
    }
    Ok(Interface {
        point: r_c,
        claims: proof.claims,
    })
}

pub fn verify(statement: &Statement, proof: &Proof) -> Result<Interface, RamError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| RamError("malformed LogUp Twist proof".into()))?
}

pub fn proof_size(proof: &Proof) -> super::ProofSize {
    let read = 2 * (proof.read.cycle.len() + proof.read.address.len()) + 2;
    let cycle = proof
        .cycle
        .sumcheck
        .round_msgs
        .iter()
        .map(Vec::len)
        .sum::<usize>()
        + 2;
    let rational = &proof.pushforward.rational;
    let mut rational_fields = rational.roots.len();
    for tree in [&rational.left, &rational.right] {
        for layer in &tree.layers {
            rational_fields += 1
                + layer.numerators.len()
                + 2 * layer.product.rounds.len()
                + layer.product.evaluations.len();
        }
    }
    let iop = 3
        + read
        + cycle
        + rational_fields
        + proof.pushforward.address_evals.len()
        + usize::from(proof.final_eval.is_some());
    super::ProofSize {
        pcs_bytes: super::pcs_bytes(&proof.commitment, &proof.opening)
            + super::pcs_bytes(
                &proof.pushforward.auxiliary.commitment,
                &proof.pushforward.auxiliary.opening,
            ),
        mosaic_bytes: super::mosaic_bytes(&proof.mosaic)
            + super::mosaic_bytes(&proof.pushforward.auxiliary.mosaic),
        iop_bytes: 16 * iop,
    }
}

/// IOP error terms excluding both PCS and RingSwitch instances. The caller
/// composes those on their actual, separately padded main/auxiliary domains.
pub fn soundness_terms(statement: &Statement) -> Vec<(String, f64)> {
    let c = &statement.config;
    let (n, a) = (statement.log_cycles, c.address_bits);
    let (main, aux) = (Schema::main(c, n), Schema::auxiliary(c));
    let main_claims = a + 2 + usize::from(!c.public_final);
    let f = 2f64.powi(-128);
    let terms = [
        ("read.batch", 1usize),
        ("read.sumcheck", 3 * n + 2 * a),
        ("final.point", a),
        ("cycle.batch", 1),
        ("cycle.sumcheck", 3 * n),
        ("rational.tree.batch", 2 * (n + a)),
        (
            "rational.tree.sumcheck",
            3 * (n * (n - 1) + a * (a - 1)) / 2,
        ),
        ("rational.tree.selector", n + a),
        ("mosaic.main.batch", main_claims - 1),
        ("mosaic.main.sumcheck", 2 * main.layout.ell),
        ("mosaic.aux.batch", 1),
        ("mosaic.aux.sumcheck", 2 * aux.layout.ell),
    ];
    let mut out: Vec<_> = terms
        .into_iter()
        .map(|(name, numerator)| (name.to_string(), numerator as f64 * f))
        .collect();
    out.push(("pushforward.rational".into(), ((1u64 << a) - 1) as f64 * f));
    out
}

#[cfg(test)]
#[path = "logup_tests.rs"]
mod tests;
