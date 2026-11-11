//! Standalone committed RAM-trace consistency using open-magnetar's BinaryTwist,
//! Mosaic batching, RingSwitch, and Ligerito implementations over GF(2^128).
//!
//! Every cycle reads all ports before its one write. The public statement binds
//! initial and final memory; the proof commits to the private access/value trace.
//! This proves RAM consistency, not execution of a CPU or an R1CS program.
//! Addresses are defined by the committed one-hot components; this interface
//! does not promise a separate binary-address column for a CPU proof to open.
//! The default shared-address read/modify/write mode commits RV and Inc, derives
//! WV = RV + Inc, and shares read/write address components as upstream does.

pub mod twist2;
pub mod word;

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use om_field::K;
use om_memcheck::onehot::EvalClaim;
pub use om_memcheck::onehot::VirtualizationMode;
use om_memcheck::twist::{
    FinalMem, TwistClaims, TwistParams, TwistProof, addr_bit_claims, build_witness, comp_names_tw,
    twist_prove_with_virtualization_and_shared_address, twist_verify_with_virtualization,
    word_mle_eval,
};
use om_mosaic::{
    BatchProof, Claim, Layout, batch_prove_packed, batch_verify, compute_layout, place_in_g_packed,
};
use om_pcs::ligerito::make_params;
use om_pcs::{BitPCS, BitProof, Commitment};
use om_sumcheck::{PackedBits, VerificationError};
use om_transcript::Transcript;
use om_transcript::profiling::{self, Span};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub address_bits: usize,
    /// Bits in each address component; must divide `address_bits`.
    /// Four bits means one 16-entry one-hot lane, not four one-hot entries.
    pub chunk_bits: usize,
    /// Aggregated flat chunk uniting, binary GKR, or the legacy flat reference.
    pub virtualization: VirtualizationMode,
    pub read_ports: usize,
    /// One read and one write at the same address in every cycle. This shares
    /// the committed address components and derives WV from RV and Inc.
    /// Set false for independent read/write addresses or multiple read ports.
    pub shared_address: bool,
    /// A word contains `2^word_log_bits` bits, at most 64.
    pub word_log_bits: usize,
    /// Upstream Ligerito profile, e.g. `provable-120`; not a 128-bit security claim.
    pub profile: String,
    pub log_rate: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            address_bits: 8,
            chunk_bits: 4,
            virtualization: VirtualizationMode::Flat,
            read_ports: 1,
            shared_address: true,
            word_log_bits: 6,
            profile: "provable-120".into(),
            log_rate: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessTrace {
    pub initial: Vec<u64>,
    /// Port-major addresses: `reads[port][cycle]`.
    pub reads: Vec<Vec<u32>>,
    pub writes: Vec<u32>,
    pub new_values: Vec<u64>,
    /// If supplied, values must match reads before the cycle's write.
    /// If absent, the adapter derives them by replaying the access trace.
    pub read_values: Option<Vec<Vec<u64>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub config: Config,
    pub log_cycles: usize,
    pub initial_memory: Vec<u64>,
    pub final_memory: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamProof {
    pub commitment: Commitment,
    pub input_claims: TwistClaims,
    pub twist: TwistProof,
    pub mosaic: BatchProof,
    pub opening: BitProof,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSummary {
    pub name: String,
    pub log_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub cycles: usize,
    pub memory_cells: usize,
    pub address_components: usize,
    pub chunk_bits: usize,
    pub committed_bits: usize,
    pub packed_field_elements: usize,
    pub mosaic_claims: usize,
    pub commitment_components: Vec<ComponentSummary>,
}

#[derive(Clone, Debug)]
pub struct ProveResult {
    pub statement: Statement,
    pub proof: RamProof,
    /// Exclusive stage durations when the profiling feature is enabled.
    pub metrics: Vec<(&'static str, Duration)>,
    pub summary: Summary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamError(pub String);

impl fmt::Display for RamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for RamError {}

impl From<VerificationError> for RamError {
    fn from(error: VerificationError) -> Self {
        Self(error.message().to_owned())
    }
}

fn require(condition: bool, message: &str) -> Result<(), RamError> {
    if condition {
        Ok(())
    } else {
        Err(RamError(message.into()))
    }
}

fn validate_config(config: &Config, log_cycles: usize) -> Result<(), RamError> {
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
    require(
        config.virtualization != VirtualizationMode::GkrTree
            || (config.address_bits / config.chunk_bits).is_power_of_two(),
        "tree virtualization requires a power-of-two chunk count",
    )?;
    require(config.read_ports > 0, "at least one read port is required")?;
    require(
        !config.shared_address || config.read_ports == 1,
        "shared_address requires exactly one read port",
    )?;
    require(config.word_log_bits <= 6, "word_log_bits must be at most 6")?;
    require(log_cycles <= 31, "log_cycles must be at most 31")?;
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
    )?;
    // The actual combined claim count is checked before Mosaic batching.
    require(config.read_ports <= 32, "read_ports must be at most 32")
}

fn word_fits(word: u64, log_bits: usize) -> bool {
    log_bits == 6 || word >> (1usize << log_bits) == 0
}

fn validate_statement(statement: &Statement) -> Result<(), RamError> {
    validate_config(&statement.config, statement.log_cycles)?;
    let cells = 1usize << statement.config.address_bits;
    require(
        statement.initial_memory.len() == cells && statement.final_memory.len() == cells,
        "initial/final memory length does not match address_bits",
    )?;
    require(
        statement
            .initial_memory
            .iter()
            .chain(&statement.final_memory)
            .all(|&word| word_fits(word, statement.config.word_log_bits)),
        "initial/final memory word exceeds configured word width",
    )
}

struct Schema {
    params: TwistParams,
    layout: Layout,
    ids: BTreeMap<String, i64>,
    components: Vec<ComponentSummary>,
    shared_address: bool,
    pcs: BitPCS,
}

impl Schema {
    fn new(config: &Config, n: usize) -> Self {
        let params = TwistParams {
            name: "ram".into(),
            a: config.address_bits,
            d: config.address_bits / config.chunk_bits,
            p: config.read_ports,
            n,
            ell: config.word_log_bits,
        };
        let mut components = Vec::new();
        for name in comp_names_tw(&params) {
            if config.shared_address && name.starts_with("ram/wa/") {
                continue;
            }
            components.push(ComponentSummary {
                name,
                log_size: config.chunk_bits + n,
            });
        }
        components.push(ComponentSummary {
            name: "ram/inc".into(),
            log_size: config.word_log_bits + n,
        });
        for port in 0..config.read_ports {
            components.push(ComponentSummary {
                name: format!("input/rv{port}"),
                log_size: config.word_log_bits + n,
            });
        }
        if !config.shared_address {
            components.push(ComponentSummary {
                name: "input/wv".into(),
                log_size: config.word_log_bits + n,
            });
        }
        let mut ids: BTreeMap<String, i64> = components
            .iter()
            .enumerate()
            .map(|(id, component)| (component.name.clone(), id as i64))
            .collect();
        if config.shared_address {
            for component in 1..=params.d {
                let id = ids[&format!("ram/ra0/{component}")];
                ids.insert(format!("ram/wa/{component}"), id);
            }
        }
        let layout = compute_layout(
            &components
                .iter()
                .enumerate()
                .map(|(id, c)| (id as i64, c.log_size))
                .collect::<Vec<_>>(),
            // RingSwitch packs seven variables and make_params requires n>6.
            14,
        );
        let pcs = BitPCS::new(make_params(
            layout.ell - 7,
            &config.profile,
            config.log_rate,
        ));
        Self {
            params,
            layout,
            ids,
            components,
            shared_address: config.shared_address,
            pcs,
        }
    }

    fn place(&self, g: &mut PackedBits, name: &str, table: &PackedBits) {
        place_in_g_packed(g, &self.layout, self.ids[name], table);
    }

    fn output_claims(&self, claims: &[EvalClaim]) -> Result<Vec<Claim>, RamError> {
        claims
            .iter()
            .map(|claim| {
                self.ids
                    .get(&claim.poly)
                    .map(|&id| claim.to_mosaic_claim(id))
                    .ok_or_else(|| {
                        RamError(format!("unknown Twist output component {}", claim.poly))
                    })
            })
            .collect()
    }

    fn input_authentication_claims(&self, input: &TwistClaims) -> Vec<Claim> {
        let mut point = input.r_o.clone();
        point.extend_from_slice(&input.r_c);
        let mut claims: Vec<Claim> = input
            .rv
            .iter()
            .enumerate()
            .map(|(port, &value)| Claim {
                comp_id: self.ids[&format!("input/rv{port}")],
                point: point.clone(),
                value,
            })
            .collect();
        // With shared addresses, the old cell value is RV and Inc is old XOR
        // new. Linearity makes WV = RV + Inc at this same MLE point. Open the
        // actual Inc column here; its other Twist output claims are elsewhere.
        claims.push(if self.shared_address {
            Claim {
                comp_id: self.ids["ram/inc"],
                point,
                value: input.wv + input.rv[0],
            }
        } else {
            Claim {
                comp_id: self.ids["input/wv"],
                point,
                value: input.wv,
            }
        });
        // Address bits are internal Twist messages, fixed before its random
        // phi coefficients. That existing relation binds them to the committed
        // one-hot trace; no second binary-address oracle is needed here.
        claims
    }
}

// This is only trace encoding. All field arithmetic and proof kernels are the
// original upstream routines. Packing avoids a temporary Vec<bool> per word.
fn pack_words(words: &[u64], log_bits: usize) -> PackedBits {
    let width = 1usize << log_bits;
    if width == 64 {
        return PackedBits::from_words(words.to_vec(), words.len() * 64);
    }
    let per_word = 64 / width;
    let packed = words
        .chunks(per_word)
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
    let _span = Span::new("ram.statement");
    let config = &statement.config;
    let mut tr = Transcript::new_default(match config.virtualization {
        VirtualizationMode::FlatLegacy => b"ram-argument/v2",
        VirtualizationMode::Flat => b"ram-argument/v4",
        VirtualizationMode::GkrTree => b"ram-argument/v3",
    });
    let params = [
        config.address_bits,
        config.chunk_bits,
        config.read_ports,
        usize::from(config.shared_address),
        config.word_log_bits,
        config.log_rate,
        statement.log_cycles,
        schema.layout.ell,
    ];
    let mut bytes: Vec<u8> = params
        .iter()
        .flat_map(|&v| (v as u64).to_le_bytes())
        .collect();
    match config.virtualization {
        VirtualizationMode::FlatLegacy => {}
        VirtualizationMode::GkrTree => bytes.extend_from_slice(&1u64.to_le_bytes()),
        VirtualizationMode::Flat => bytes.extend_from_slice(&2u64.to_le_bytes()),
    }
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
    for (label, words) in [
        (b"ram/initial".as_slice(), &statement.initial_memory),
        (b"ram/final".as_slice(), &statement.final_memory),
    ] {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        tr.absorb_bytes(label, &bytes);
    }
    tr
}

/// Prove the trace's read-before-write RAM consistency, including public final memory.
/// Optional supplied read values are checked against the upstream witness replay.
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
        trace.reads.len() == config.read_ports
            && trace.reads.iter().all(|r| r.len() == trace.writes.len()),
        "read address shape mismatch",
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
            .all(|&word| word_fits(word, config.word_log_bits)),
        "trace word exceeds configured width",
    )?;
    require(
        trace
            .reads
            .iter()
            .flatten()
            .chain(&trace.writes)
            .all(|&address| (address as usize) < cells),
        "trace address out of range",
    )?;
    require(
        !config.shared_address || trace.reads[0] == trace.writes,
        "shared_address requires equal read and write address streams",
    )?;
    if let Some(values) = &trace.read_values {
        require(
            values.len() == config.read_ports
                && values.iter().all(|v| v.len() == trace.writes.len()),
            "read value shape mismatch",
        )?;
    }
    let schema = Schema::new(config, n);
    let (witness, values, final_memory) = {
        let _span = Span::new("ram.witness");
        let reads: Vec<Vec<usize>> = trace
            .reads
            .iter()
            .map(|addresses| addresses.iter().map(|&a| a as usize).collect())
            .collect();
        let writes: Vec<usize> = trace.writes.iter().map(|&a| a as usize).collect();
        // The upstream wrapper computes read values and final memory and calls
        // build_witness_lean internally, preserving its compact address form.
        let (witness, info) = build_witness(
            &schema.params,
            &trace.initial,
            &reads,
            &writes,
            &trace.new_values,
        );
        if let Some(supplied) = &trace.read_values {
            require(
                *supplied == info.rv,
                "supplied read value is inconsistent with RAM trace",
            )?;
        }
        (witness, info.rv, info.final_state)
    };
    let statement = Statement {
        config: config.clone(),
        log_cycles: n,
        initial_memory: trace.initial.clone(),
        final_memory,
    };
    let mut tr = statement_transcript(&statement, &schema);
    let g = {
        let _span = Span::new("ram.pack");
        let mut g = PackedBits::zero(1usize << schema.layout.ell);
        for (access, address) in witness
            .ra
            .iter()
            .chain(std::iter::once(&witness.wa))
            .enumerate()
        {
            if config.shared_address && access == config.read_ports {
                continue;
            }
            for component in 0..schema.params.d {
                let name = if access < config.read_ports {
                    format!("ram/ra{access}/{}", component + 1)
                } else {
                    format!("ram/wa/{}", component + 1)
                };
                schema.place(&mut g, &name, &address.packed(component));
            }
        }
        schema.place(&mut g, "ram/inc", &witness.inc_packed(config.word_log_bits));
        for (port, words) in values.iter().enumerate() {
            schema.place(
                &mut g,
                &format!("input/rv{port}"),
                &pack_words(words, config.word_log_bits),
            );
        }
        if !config.shared_address {
            schema.place(
                &mut g,
                "input/wv",
                &pack_words(&trace.new_values, config.word_log_bits),
            );
        }
        g
    };
    let (commitment, committed) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    // The witness root and all public parameters/memory precede these points.
    let r_o = tr.challenge_ks(b"ram/point_o", config.word_log_bits);
    let r_c = tr.challenge_ks(b"ram/point_c", n);
    let input_claims = {
        let _span = Span::new("ram.claims");
        let rv: Vec<K> = values
            .iter()
            .map(|words| word_mle_eval(words, &r_o, &r_c))
            .collect();
        let wv = if config.shared_address {
            rv[0] + word_mle_eval(&witness.inc, &r_o, &r_c)
        } else {
            word_mle_eval(&trace.new_values, &r_o, &r_c)
        };
        let bits = if config.shared_address {
            let bits = witness.ra[0].bit_evals(config.address_bits, &r_c);
            vec![bits.clone(), bits]
        } else {
            addr_bit_claims(&schema.params, &witness, &r_c)
        };
        TwistClaims {
            r_o: r_o.clone(),
            r_c: r_c.clone(),
            rv,
            wv,
            bits,
        }
    };
    let final_mem = FinalMem {
        m: config.address_bits,
        s_out: 0,
        words: statement.final_memory.clone(),
    };
    let (twist, output) = {
        let _span = Span::new("ram.twist");
        twist_prove_with_virtualization_and_shared_address(
            &schema.params,
            &statement.initial_memory,
            &witness,
            &input_claims,
            &mut tr,
            b"ram/twist",
            Some(&final_mem),
            false,
            config.virtualization,
            config.shared_address,
        )
    };
    let mut claims = schema.input_authentication_claims(&input_claims);
    claims.extend(schema.output_claims(&output.claims)?);
    require(
        claims.len() <= om_mosaic::MAX_CLAIMS,
        "too many total claims for upstream Mosaic batching",
    )?;
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
        cycles: trace.writes.len(),
        memory_cells: cells,
        address_components: schema.params.d,
        chunk_bits: config.chunk_bits,
        committed_bits: g.len(),
        packed_field_elements: g.len() / 128,
        mosaic_claims: claims.len(),
        commitment_components: schema.components,
    };
    Ok(ProveResult {
        statement,
        proof: RamProof {
            commitment,
            input_claims,
            twist,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
    })
}

fn verify_inner(statement: &Statement, proof: &RamProof) -> Result<(), RamError> {
    validate_statement(statement)?;
    let config = &statement.config;
    let schema = Schema::new(config, statement.log_cycles);
    require(
        proof.commitment.n == schema.layout.ell - 7 && proof.commitment.root.len() == 32,
        "malformed commitment",
    )?;
    let mut tr = statement_transcript(statement, &schema);
    schema.pcs.absorb_commitment(&proof.commitment, &mut tr);
    let r_o = tr.challenge_ks(b"ram/point_o", config.word_log_bits);
    let r_c = tr.challenge_ks(b"ram/point_c", statement.log_cycles);
    let input = &proof.input_claims;
    require(
        input.r_o == r_o && input.r_c == r_c,
        "input claim point does not match postcommit challenges",
    )?;
    require(
        input.rv.len() == config.read_ports
            && input.bits.len() == config.read_ports + 1
            && input.bits.iter().all(|v| v.len() == config.address_bits),
        "malformed input claims",
    )?;
    require(
        !config.shared_address || input.bits[0] == input.bits[1],
        "shared-address read and write bit claims disagree",
    )?;
    let final_mem = FinalMem {
        m: config.address_bits,
        s_out: 0,
        words: statement.final_memory.clone(),
    };
    let output = twist_verify_with_virtualization(
        &schema.params,
        &|point: &[K]| {
            word_mle_eval(
                &statement.initial_memory,
                &point[config.address_bits..],
                &point[..config.address_bits],
            )
        },
        input,
        &proof.twist,
        &mut tr,
        b"ram/twist",
        Some(&final_mem),
        false,
        config.virtualization,
    )?;
    let mut claims = schema.input_authentication_claims(input);
    claims.extend(schema.output_claims(&output.claims)?);
    let (r_g, v_g) = batch_verify(&schema.layout, &claims, &proof.mosaic, &mut tr)?;
    schema
        .pcs
        .verify(&proof.commitment, &r_g, v_g, &proof.opening, &mut tr)?;
    Ok(())
}

/// Verify using only the public statement and proof, without the access trace.
/// Malformed upstream proof shapes are converted into a verification error.
pub fn verify(statement: &Statement, proof: &RamProof) -> Result<(), RamError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_inner(statement, proof)
    }))
    .map_err(|_| RamError("malformed RAM proof".into()))?
}
