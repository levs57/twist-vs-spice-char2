//! Indexed read-only lookup proofs over GF128, sharing Mosaic/RingSwitch/Ligerito.
//! Shout commits one-hot selectors as well as the common binary-address/value
//! input. Logup* commits a post-challenge weighted pushforward, never integer
//! multiplicities reduced modulo two.

mod logup;
mod shout;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use om_field::K;
pub use om_memcheck::onehot::VirtualizationMode;
use om_memcheck::onehot::{AddrComps, EvalClaim};
use om_memcheck::shout::{ShoutParams, ShoutProof};
use om_mosaic::{
    BatchProof, Claim, Layout, batch_prove_packed, batch_verify, compute_layout, place_in_g_packed,
};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, BitStatePacked, Commitment};
use om_sumcheck::{PackedBits, VerificationError};
use om_transcript::{
    Transcript,
    profiling::{self, Span},
};
use rayon::prelude::*;

pub use logup::LogupProof;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Shout,
    LogupStar,
}

impl Method {
    pub fn name(self) -> &'static str {
        match self {
            Self::Shout => "shout",
            Self::LogupStar => "logup-star",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub address_bits: usize,
    pub chunk_bits: usize,
    pub virtualization: VirtualizationMode,
    pub profile: String,
    pub log_rate: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            address_bits: 12,
            chunk_bits: 4,
            virtualization: VirtualizationMode::Flat,
            profile: "provable-120".into(),
            log_rate: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupTrace {
    pub table: Vec<u64>,
    pub addresses: Vec<u32>,
    pub read_values: Option<Vec<u64>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub config: Config,
    pub method: Method,
    pub log_lookups: usize,
    pub table: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputClaims {
    pub read_value: K,
    /// Needed by Shout to link its one-hot address encoding to the common
    /// binary-address input; empty for logup*, whose rational leaves do so.
    pub address_bits: Vec<K>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentProof {
    Shout(ShoutProof),
    LogupStar(LogupProof),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticationProof {
    pub commitment: Commitment,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LookupProof {
    pub commitment: Commitment,
    pub input: InputClaims,
    pub argument: ArgumentProof,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub lookups: usize,
    pub table_rows: usize,
    pub core_bits: usize,
    pub argument_bits: usize,
    pub main_committed_bits: usize,
    pub auxiliary_bits: usize,
    pub auxiliary_committed_bits: usize,
    pub main_claims: usize,
    pub auxiliary_claims: usize,
}

/// Inclusive elapsed times; `metrics` separately holds exclusive stage spans.
#[derive(Clone, Debug, Default)]
pub struct CommitmentTimings {
    pub main_commit: Duration,
    pub auxiliary_commit: Duration,
    pub main_open: Duration,
    pub auxiliary_open: Duration,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    pub proof: LookupProof,
    pub metrics: Vec<(&'static str, Duration)>,
    pub timings: CommitmentTimings,
    pub summary: Summary,
}

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
impl From<bitz_logup::Error> for Error {
    fn from(e: bitz_logup::Error) -> Self {
        Self(e.to_string())
    }
}

fn require(condition: bool, message: &str) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}

fn validate(statement: &Statement) -> Result<(), Error> {
    let c = &statement.config;
    require(
        (1..=31).contains(&c.address_bits),
        "address_bits must be in 1..=31",
    )?;
    require(
        statement.log_lookups <= 31,
        "log_lookups must be at most 31",
    )?;
    require(
        statement.table.len() == 1usize << c.address_bits,
        "public table shape mismatch",
    )?;
    require(
        c.profile == "provable-120" || c.profile == "conjectured-100",
        "unsupported PCS profile",
    )?;
    require(
        (if c.profile == "provable-120" { 2 } else { 1 }..=8).contains(&c.log_rate),
        "invalid PCS log_rate",
    )?;
    if statement.method == Method::Shout {
        require(
            c.chunk_bits > 0
                && c.chunk_bits <= c.address_bits
                && c.address_bits.is_multiple_of(c.chunk_bits),
            "chunk_bits must divide address_bits",
        )?;
        require(
            c.virtualization != VirtualizationMode::GkrTree
                || (c.address_bits / c.chunk_bits).is_power_of_two(),
            "GKR requires a power-of-two chunk count",
        )?;
    }
    Ok(())
}

pub(crate) struct Schema {
    layout: Layout,
    ids: BTreeMap<String, i64>,
    pcs: BitPCS,
}
impl Schema {
    fn new(components: Vec<(String, usize)>, config: &Config) -> Self {
        let ids = components
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i as i64))
            .collect();
        let layout = compute_layout(
            &components
                .iter()
                .enumerate()
                .map(|(i, (_, n))| (i as i64, *n))
                .collect::<Vec<_>>(),
            14,
        );
        let pcs = BitPCS::new(make_params(
            layout.ell - 7,
            &config.profile,
            config.log_rate,
        ));
        Self { layout, ids, pcs }
    }
    fn main(statement: &Statement) -> Self {
        let n = statement.log_lookups;
        let c = &statement.config;
        let mut components = vec![("input/rv".into(), n + 6)];
        components.extend((0..c.address_bits).map(|i| (format!("input/address/{i}"), n)));
        if statement.method == Method::Shout {
            components.extend(
                (1..=c.address_bits / c.chunk_bits)
                    .map(|i| (format!("lookup/ra/{i}"), n + c.chunk_bits)),
            );
        }
        Self::new(components, c)
    }
    fn auxiliary(statement: &Statement) -> Self {
        Self::new(
            vec![("aux/pushforward".into(), statement.config.address_bits + 7)],
            &statement.config,
        )
    }
    fn place(&self, g: &mut PackedBits, name: &str, table: &PackedBits) {
        place_in_g_packed(g, &self.layout, self.ids[name], table)
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
                    .map(|&id| c.to_mosaic_claim(id))
                    .ok_or_else(|| Error(format!("unknown component {}", c.poly)))
            })
            .collect()
    }
    fn valid_commitment(&self, cm: &Commitment) -> Result<(), Error> {
        require(
            cm.n == self.layout.ell - 7 && cm.root.len() == 32,
            "malformed commitment",
        )
    }
    fn open(
        &self,
        g: &PackedBits,
        state: BitStatePacked,
        claims: &[EvalClaim],
        tr: &mut Transcript,
        label: &'static str,
    ) -> Result<(BatchProof, BitProof, Duration), Error> {
        let started = Instant::now();
        let _span = Span::new(label);
        let claims = self.claims(claims)?;
        let (batch, r, v) = {
            let _span = Span::new("lookup.mosaic");
            batch_prove_packed(&self.layout, &claims, g, tr)
        };
        let (opened, proof) = self.pcs.open_packed(state, g, &r, tr);
        require(opened == v, "Mosaic/PCS evaluation mismatch")?;
        Ok((batch, proof, started.elapsed()))
    }
    fn verify(
        &self,
        cm: &Commitment,
        batch: &BatchProof,
        opening: &BitProof,
        claims: &[EvalClaim],
        tr: &mut Transcript,
    ) -> Result<(), Error> {
        self.valid_commitment(cm)?;
        let (r, v) = batch_verify(&self.layout, &self.claims(claims)?, batch, tr)?;
        self.pcs.verify(cm, &r, v, opening, tr)?;
        Ok(())
    }
}

fn transcript(statement: &Statement, schema: &Schema) -> Transcript {
    let _span = Span::new("lookup.statement");
    let c = &statement.config;
    let mut tr = Transcript::new_default(b"indexed-rom-lookup/v1");
    tr.absorb_bytes(b"lookup/method", statement.method.name().as_bytes());
    let dims = [
        c.address_bits,
        c.chunk_bits,
        c.log_rate,
        statement.log_lookups,
        schema.layout.ell,
    ];
    tr.absorb_bytes(
        b"lookup/config",
        &dims
            .into_iter()
            .flat_map(|v| (v as u64).to_le_bytes())
            .collect::<Vec<_>>(),
    );
    tr.absorb_bytes(b"lookup/profile", c.profile.as_bytes());
    tr.absorb_bytes(
        b"lookup/virtualization",
        match c.virtualization {
            VirtualizationMode::Flat => b"flat",
            VirtualizationMode::FlatLegacy => b"legacy-flat",
            VirtualizationMode::GkrTree => b"gkr",
        },
    );
    // Bind the actual arbitrary ROM, including duplicate/zero values.
    tr.absorb_bytes(
        b"lookup/table",
        &statement
            .table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    tr
}

/// Polynomial-basis encoding of a Boolean word is a scaled ordinary MLE:
/// C*Eq(r,i)=X^i. This authenticates a field word using one BitPCS opening.
pub(crate) fn word_encoding(log_bits: usize) -> (Vec<K>, K) {
    static ENCODINGS: OnceLock<Vec<(Vec<K>, K)>> = OnceLock::new();
    assert!(log_bits <= 7);
    ENCODINGS.get_or_init(|| (0..=7).map(build_word_encoding).collect())[log_bits].clone()
}

fn build_word_encoding(log_bits: usize) -> (Vec<K>, K) {
    let mut factor = K::ONE;
    let mut point = Vec::with_capacity(log_bits);
    let mut power = K::X;
    for _ in 0..log_bits {
        let sum = K::ONE + power;
        factor *= sum;
        point.push(power / sum);
        power = power.square();
    }
    (point, factor)
}

pub(crate) fn encoded_word_eval(words: &[u64], point: &[K]) -> K {
    let (word_point, factor) = word_encoding(6);
    factor * om_memcheck::twist::word_mle_eval(words, &word_point, point)
}

fn word_claim(point: &[K], value: K, name: &str, log_bits: usize) -> EvalClaim {
    let (mut p, factor) = word_encoding(log_bits);
    p.extend_from_slice(point);
    EvalClaim {
        poly: name.into(),
        point: p,
        value: value / factor,
    }
}

fn input_claims(statement: &Statement, cycle: &[K], input: &InputClaims) -> Vec<EvalClaim> {
    let mut claims = vec![word_claim(cycle, input.read_value, "input/rv", 6)];
    if statement.method == Method::Shout {
        claims.extend(
            input
                .address_bits
                .iter()
                .enumerate()
                .map(|(i, &value)| EvalClaim {
                    poly: format!("input/address/{i}"),
                    point: cycle.to_vec(),
                    value,
                }),
        );
    }
    claims
}

/// Encode binary-address inputs word by word without data-dependent branches.
/// Columns initialize on workers; the layout and committed bits are unchanged.
fn pack_address_columns(addresses: &[u32], address_bits: usize) -> Vec<PackedBits> {
    let column = |bit: usize| {
        let words = addresses
            .chunks(64)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u64, |word, (j, &address)| {
                    word | (u64::from((address >> bit) & 1) << j)
                })
            })
            .collect();
        PackedBits::from_words(words, addresses.len())
    };
    if addresses.len().saturating_mul(address_bits) >= 1 << 16 {
        (0..address_bits).into_par_iter().map(column).collect()
    } else {
        (0..address_bits).map(column).collect()
    }
}

pub fn prove(method: Method, config: &Config, trace: &LookupTrace) -> Result<ProveResult, Error> {
    profiling::clear();
    require(
        !trace.addresses.is_empty() && trace.addresses.len().is_power_of_two(),
        "lookup count must be a nonzero power of two",
    )?;
    let n = trace.addresses.len().ilog2() as usize;
    let statement = Statement {
        config: config.clone(),
        method,
        log_lookups: n,
        table: trace.table.clone(),
    };
    validate(&statement)?;
    require(
        trace
            .addresses
            .iter()
            .all(|&v| (v as usize) < trace.table.len()),
        "address out of range",
    )?;
    let mut timings = CommitmentTimings::default();
    let values: Vec<u64> = {
        let _span = Span::new("lookup.witness");
        trace
            .addresses
            .iter()
            .map(|&a| trace.table[a as usize])
            .collect()
    };
    if let Some(supplied) = &trace.read_values {
        require(*supplied == values, "incorrect supplied read values")?;
    }
    let schema = Schema::main(&statement);
    let mut tr = transcript(&statement, &schema);
    let comps = AddrComps::from_addrs(
        trace.addresses.clone(),
        config.address_bits,
        if method == Method::Shout {
            config.address_bits / config.chunk_bits
        } else {
            1
        },
    );
    let g = {
        let _span = Span::new("lookup.pack");
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        schema.place(
            &mut g,
            "input/rv",
            &PackedBits::from_words(values.clone(), values.len() * 64),
        );
        for (i, column) in pack_address_columns(&trace.addresses, config.address_bits)
            .iter()
            .enumerate()
        {
            schema.place(&mut g, &format!("input/address/{i}"), column);
        }
        if method == Method::Shout {
            for i in 0..config.address_bits / config.chunk_bits {
                schema.place(&mut g, &format!("lookup/ra/{}", i + 1), &comps.packed(i));
            }
        }
        g
    };
    let (commitment, state) = {
        let started = Instant::now();
        let _span = Span::new("lookup.main.commit");
        let result = schema.pcs.commit_packed(&g, &mut tr);
        timings.main_commit = started.elapsed();
        result
    };
    let cycle = tr.challenge_ks(b"lookup/cycle", n);
    let input = {
        let _span = Span::new("lookup.claims");
        InputClaims {
            read_value: encoded_word_eval(&values, &cycle),
            address_bits: if method == Method::Shout {
                comps.bit_evals(config.address_bits, &cycle)
            } else {
                Vec::new()
            },
        }
    };
    tr.absorb_k(b"lookup/read-claim", input.read_value);
    tr.absorb_ks(b"lookup/address-claims", &input.address_bits);
    let mut claims = input_claims(&statement, &cycle, &input);
    let t = trace.addresses.len();
    let mut summary = Summary {
        lookups: t,
        table_rows: trace.table.len(),
        core_bits: (config.address_bits + 64) * t,
        argument_bits: if method == Method::Shout {
            (config.address_bits / config.chunk_bits) * (1usize << config.chunk_bits) * t
        } else {
            0
        },
        main_committed_bits: g.len(),
        ..Summary::default()
    };
    let argument = match method {
        Method::Shout => {
            let params = ShoutParams {
                name: "lookup".into(),
                a: config.address_bits,
                d: config.address_bits / config.chunk_bits,
                n,
            };
            let (proof, output) = shout::prove(
                config,
                &params,
                &trace.table,
                &comps,
                &cycle,
                &input,
                &mut tr,
            );
            claims.extend(output);
            ArgumentProof::Shout(proof)
        }
        Method::LogupStar => {
            let (proof, output) = logup::prove(
                &statement,
                &comps,
                &cycle,
                input.read_value,
                &mut tr,
                &mut summary,
                &mut timings,
            )?;
            claims.extend(output);
            ArgumentProof::LogupStar(proof)
        }
    };
    summary.main_claims = claims.len();
    let (mosaic, opening, elapsed) =
        schema.open(&g, state, &claims, &mut tr, "lookup.main.open")?;
    timings.main_open = elapsed;
    Ok(ProveResult {
        statement,
        proof: LookupProof {
            commitment,
            input,
            argument,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
        timings,
    })
}

fn verify_inner(statement: &Statement, proof: &LookupProof) -> Result<(), Error> {
    validate(statement)?;
    let config = &statement.config;
    let n = statement.log_lookups;
    require(
        proof.input.address_bits.len()
            == if statement.method == Method::Shout {
                config.address_bits
            } else {
                0
            },
        "malformed input claims",
    )?;
    let schema = Schema::main(statement);
    schema.valid_commitment(&proof.commitment)?;
    let mut tr = transcript(statement, &schema);
    schema.pcs.absorb_commitment(&proof.commitment, &mut tr);
    let cycle = tr.challenge_ks(b"lookup/cycle", n);
    tr.absorb_k(b"lookup/read-claim", proof.input.read_value);
    tr.absorb_ks(b"lookup/address-claims", &proof.input.address_bits);
    let mut claims = input_claims(statement, &cycle, &proof.input);
    claims.extend(match (&proof.argument, statement.method) {
        (ArgumentProof::Shout(p), Method::Shout) => {
            let params = ShoutParams {
                name: "lookup".into(),
                a: config.address_bits,
                d: config.address_bits / config.chunk_bits,
                n,
            };
            shout::verify(
                config,
                &params,
                &statement.table,
                &cycle,
                &proof.input,
                p,
                &mut tr,
            )?
        }
        (ArgumentProof::LogupStar(p), Method::LogupStar) => {
            logup::verify(statement, &cycle, proof.input.read_value, p, &mut tr)?
        }
        _ => return Err(Error("wrong lookup proof branch".into())),
    });
    schema.verify(
        &proof.commitment,
        &proof.mosaic,
        &proof.opening,
        &claims,
        &mut tr,
    )
}

pub fn verify(statement: &Statement, proof: &LookupProof) -> Result<(), Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| Error("malformed lookup proof".into()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_opening_conversion_matches_every_polynomial_basis_coefficient() {
        for log_bits in 0..=7 {
            let (point, factor) = word_encoding(log_bits);
            for (i, eq) in om_sumcheck::eq_table(&point).into_iter().enumerate() {
                assert_eq!(factor * eq, K::from_u128(1u128 << i));
            }
        }
    }

    #[test]
    fn encoded_word_mle_is_an_opening_of_the_actual_committed_bits() {
        let words = [0, u64::MAX, 1 << 63, 0xfedc_ba98_7654_3210];
        let cycle = [K::from_u128(0xdead_beef), K::from_u128(0x1234_5678)];
        let claim = word_claim(&cycle, encoded_word_eval(&words, &cycle), "input/rv", 6);
        let bits: Vec<bool> = words
            .iter()
            .flat_map(|word| (0..64).map(move |i| word >> i & 1 != 0))
            .collect();
        assert_eq!(claim.value, om_sumcheck::mle_eval_bits(&bits, &claim.point));
    }

    #[test]
    fn address_columns_preserve_bit_order_at_word_boundaries_and_on_workers() {
        for length in [1, 2, 63, 64, 65, 127, 128, 1 << 15] {
            let addresses: Vec<u32> = (0..length)
                .map(|j| {
                    (j as u32)
                        .wrapping_mul(0xa135_d789)
                        .rotate_left((j % 32) as u32)
                })
                .collect();
            for bits in [1, 16, 31, 32] {
                let expected: Vec<_> = (0..bits)
                    .map(|i| {
                        PackedBits::from_bools(
                            &addresses
                                .iter()
                                .map(|address| (address >> i) & 1 != 0)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    assert_eq!(
                        pool.install(|| pack_address_columns(&addresses, bits)),
                        expected
                    );
                }
            }
        }
    }
}
