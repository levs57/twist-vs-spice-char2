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
use om_sumcheck::PackedBits;
use om_transcript::{
    Transcript,
    profiling::{self, Span},
};
use rayon::prelude::*;
use std::collections::BTreeMap;

use crate::{
    ComponentSummary, Config, Error, OrderingMode, Schema, Statement, Summary, column_claims,
    combine_bits, dims, eval_columns, eval_columns_weighted, flock_order_claims,
    masked_reduction::{self, MaskClaims, MaskProof},
    order, order_flock, pack_columns, pack_values, require, validate, word_claim, word_eval,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessTrace {
    pub initial: Vec<u64>,
    pub addresses: Vec<u32>,
    /// A read's observed value or a write's replacement value.
    pub values: Vec<u64>,
    pub is_write: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inputs {
    pub address: Vec<K>,
    pub previous: Vec<K>,
    pub final_previous: Vec<K>,
    pub is_write: K,
    pub next_is_write: K,
    pub initial_next_is_write: K,
    pub masked: MaskClaims,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub commitment: Commitment,
    pub permutation: GrandProductProof,
    pub inputs: Inputs,
    pub masks: MaskProof,
    pub order: order_flock::OrderProof,
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

fn schema(s: &Statement) -> Schema {
    let (n, a, _, b) = dims(s);
    let mut components = vec![ComponentSummary {
        name: "input/value".into(),
        log_size: n + s.config.word_log_bits,
    }];
    for (prefix, count, log_size) in [
        ("input/address", a, n),
        ("input/previous", b, n),
        ("order/carry", b - 1, n),
        ("boundary/previous", b, a),
    ] {
        components.extend((0..count).map(|i| ComponentSummary {
            name: format!("{prefix}/{i}"),
            log_size,
        }));
    }
    for (name, log_size) in [
        ("input/is_write", n),
        ("input/next_is_write", n),
        ("initial/next_is_write", a),
    ] {
        components.push(ComponentSummary {
            name: name.into(),
            log_size,
        });
    }
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

fn transcript(s: &Statement, schema: &Schema) -> Transcript {
    let mut tr = crate::transcript(s, schema, &OrderingMode::FlockBoolean);
    tr.absorb_bytes(b"spice/trace-semantics", b"typed-single-value-masked/v1");
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

fn input_claims(s: &Statement, p: &[K], v: &Inputs) -> Vec<EvalClaim> {
    let (n, a, _, _) = dims(s);
    let mut out = column_claims("input/address", &v.address, &p[..n]);
    out.extend(column_claims("input/previous", &v.previous, &p[..n]));
    out.extend(column_claims(
        "boundary/previous",
        &v.final_previous,
        &p[..a],
    ));
    out.push(scalar_claim("input/is_write", &p[..n], v.is_write));
    out.push(scalar_claim(
        "input/next_is_write",
        &p[..n],
        v.next_is_write,
    ));
    out.push(scalar_claim(
        "initial/next_is_write",
        &p[..a],
        v.initial_next_is_write,
    ));
    out
}

fn mask_source_claims(s: &Statement, v: &masked_reduction::SourceClaims) -> Vec<EvalClaim> {
    let (n, a, _, _) = dims(s);
    vec![
        word_claim(
            "input/value",
            &v.point[..n],
            v.value,
            s.config.word_log_bits,
        ),
        scalar_claim("input/is_write", &v.point[..n], v.is_write),
        scalar_claim("input/next_is_write", &v.point[..n], v.next_is_write),
        scalar_claim(
            "initial/next_is_write",
            &v.point[..a],
            v.initial_next_is_write,
        ),
    ]
}

fn absorb_inputs(tr: &mut Transcript, v: &Inputs) {
    tr.absorb_ks(b"spice/typed/address", &v.address);
    tr.absorb_ks(b"spice/typed/previous", &v.previous);
    tr.absorb_ks(b"spice/typed/final-previous", &v.final_previous);
    tr.absorb_ks(
        b"spice/typed/flags",
        &[v.is_write, v.next_is_write, v.initial_next_is_write],
    );
    tr.absorb_ks(
        b"spice/typed/masked",
        &[v.masked.cycle[0], v.masked.cycle[1], v.masked.initial],
    );
}

fn check_leaves(
    s: &Statement,
    leaves: &LeafClaims,
    v: &Inputs,
    gamma: K,
    alpha: K,
    beta: K,
    delta: K,
) -> Result<(), Error> {
    let (n, a, l, b) = dims(s);
    require(
        leaves.point.len() == l + 1
            && v.address.len() == a
            && v.previous.len() == b
            && v.final_previous.len() == b,
        "malformed typed permutation terminals",
    )?;
    let p = &leaves.point;
    let cm = p[n..l].iter().fold(K::ONE, |v, &x| v * (K::ONE + x));
    let mm = p[a..l].iter().fold(K::ONE, |v, &x| v * (K::ONE + x));
    let addr = combine_bits(&v.address);
    let memory_addr = combine_bits(&p[..a]);
    let time = K::from_u128(1u128 << l) + combine_bits(&p[..n]);
    let cycles = [
        gamma
            + combine_bits(&v.previous)
            + alpha * addr
            + beta * v.masked.cycle[0]
            + delta * v.is_write,
        gamma + time + alpha * addr + beta * v.masked.cycle[1] + delta * v.next_is_write,
    ];
    let memory = [
        gamma
            + combine_bits(&v.final_previous)
            + alpha * memory_addr
            + beta * word_eval(&s.final_memory, &p[..a], s.config.word_log_bits),
        gamma
            + memory_addr
            + alpha * memory_addr
            + beta * v.masked.initial
            + delta * v.initial_next_is_write,
    ];
    for side in 0..2 {
        let expected = (K::ONE + p[l]) * (K::ONE + cm * (K::ONE + cycles[side]))
            + p[l] * (K::ONE + mm * (K::ONE + memory[side]));
        require(
            expected == leaves.values[side],
            "typed permutation leaf binding failed",
        )?;
    }
    Ok(())
}

pub fn prove(config: &Config, trace: &AccessTrace) -> Result<ProveResult, Error> {
    profiling::clear();
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
    let mut tr = transcript(&statement, &schema);
    let (address_columns, previous_columns, last_columns, flags, initial_flags, g) = {
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
            ("input/address", &address),
            ("input/previous", &prev),
            ("boundary/previous", &last_columns),
            ("order/carry", &carry),
        ] {
            for (i, col) in cols.iter().enumerate() {
                schema.place(&mut g, &format!("{prefix}/{i}"), col);
            }
        }
        schema.place(&mut g, "input/is_write", &flags[0]);
        schema.place(&mut g, "input/next_is_write", &flags[1]);
        schema.place(&mut g, "initial/next_is_write", &initial_flags);
        (address, prev, last_columns, flags, initial_flags, g)
    };
    let (commitment, state) = {
        let _span = Span::new("pcs.commit");
        schema.pcs.commit_packed(&g, &mut tr)
    };
    let gamma = tr.challenge_k(b"spice/typed/fingerprint/shift");
    let alpha = tr.challenge_k(b"spice/typed/fingerprint/address");
    let beta = tr.challenge_k(b"spice/typed/fingerprint/value");
    let delta = tr.challenge_k(b"spice/typed/fingerprint/write");
    let leaves = {
        let _span = Span::new("spice.permutation.leaves");
        let half = 1usize << l;
        let padding = [t..half, half + cells..2 * half]
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect();
        Leaves::from_fn_pair(2 * half, padding, |row| {
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
        lean_grandproduct::prove(l + 1, leaves, &mut tr, b"spice/typed/permutation")?
    };
    require(
        permutation.roots[0] == permutation.roots[1],
        "typed RAM permutation products disagree",
    )?;
    let inputs = {
        let _span = Span::new("spice.claims");
        let eq = om_sumcheck::eq_table(&leaves.point[..n]);
        let fv = eval_columns_weighted(&flags, &eq);
        let masked: Vec<u64> = trace
            .values
            .par_iter()
            .zip(&trace.is_write)
            .map(|(&v, &w)| v & 0u64.wrapping_sub(u64::from(!w)))
            .collect();
        let masked_next: Vec<u64> = trace
            .values
            .par_iter()
            .zip(&next)
            .map(|(&v, &w)| v & 0u64.wrapping_sub(u64::from(!w)))
            .collect();
        let initial: Vec<u64> = trace
            .initial
            .iter()
            .zip(&initial_next)
            .map(|(&v, &w)| v & 0u64.wrapping_sub(u64::from(!w)))
            .collect();
        Inputs {
            address: eval_columns_weighted(&address_columns, &eq),
            previous: eval_columns_weighted(&previous_columns, &eq),
            final_previous: eval_columns(&last_columns, &leaves.point[..a]),
            is_write: fv[0],
            next_is_write: fv[1],
            initial_next_is_write: om_sumcheck::mle_eval_packed(&initial_flags, &leaves.point[..a]),
            masked: MaskClaims {
                cycle: [
                    word_eval(&masked, &leaves.point[..n], config.word_log_bits),
                    word_eval(&masked_next, &leaves.point[..n], config.word_log_bits),
                ],
                initial: word_eval(&initial, &leaves.point[..a], config.word_log_bits),
            },
        }
    };
    check_leaves(&statement, &leaves, &inputs, gamma, alpha, beta, delta)?;
    absorb_inputs(&mut tr, &inputs);
    let mut claims = input_claims(&statement, &leaves.point, &inputs);
    let (masks, mask_sources) = {
        let _span = Span::new("spice.mask");
        masked_reduction::prove(
            &trace.values,
            &trace.is_write,
            &next,
            &trace.initial,
            &initial_next,
            &leaves.point,
            &inputs.masked,
            &mut tr,
            b"spice/typed/masks",
        )?
    };
    claims.extend(mask_source_claims(&statement, &mask_sources));
    let (order, order_sources) = {
        let _span = Span::new("spice.order");
        order_flock::prove(&previous, &carries, n, a, &mut tr, b"spice/typed/order")?
    };
    claims.extend(flock_order_claims(&order_sources));
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
            inputs,
            masks,
            order,
            mosaic,
            opening,
        },
        metrics: profiling::take(),
        summary,
    })
}

fn verify_inner(s: &Statement, p: &Proof) -> Result<(), Error> {
    validate(s)?;
    let (n, a, l, _) = dims(s);
    let schema = schema(s);
    require(
        p.commitment.n == schema.layout.ell - 7 && p.commitment.root.len() == 32,
        "malformed typed commitment",
    )?;
    let mut tr = transcript(s, &schema);
    schema.pcs.absorb_commitment(&p.commitment, &mut tr);
    let gamma = tr.challenge_k(b"spice/typed/fingerprint/shift");
    let alpha = tr.challenge_k(b"spice/typed/fingerprint/address");
    let beta = tr.challenge_k(b"spice/typed/fingerprint/value");
    let delta = tr.challenge_k(b"spice/typed/fingerprint/write");
    require(
        p.permutation.roots[0] == p.permutation.roots[1],
        "typed permutation roots disagree",
    )?;
    let leaves =
        lean_grandproduct::verify(l + 1, &p.permutation, &mut tr, b"spice/typed/permutation")?;
    check_leaves(s, &leaves, &p.inputs, gamma, alpha, beta, delta)?;
    absorb_inputs(&mut tr, &p.inputs);
    let mut claims = input_claims(s, &leaves.point, &p.inputs);
    let sources = masked_reduction::verify(
        &s.initial_memory,
        &leaves.point,
        &p.inputs.masked,
        &p.masks,
        &mut tr,
        b"spice/typed/masks",
    )?;
    claims.extend(mask_source_claims(s, &sources));
    let terminal = order_flock::verify(n, a, &p.order, &mut tr, b"spice/typed/order")?;
    claims.extend(flock_order_claims(&terminal));
    let (r, value) = batch_verify(&schema.layout, &schema.claims(&claims)?, &p.mosaic, &mut tr)?;
    schema
        .pcs
        .verify(&p.commitment, &r, value, &p.opening, &mut tr)?;
    Ok(())
}

pub fn verify(s: &Statement, p: &Proof) -> Result<(), Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| verify_inner(s, p)))
        .map_err(|_| Error("malformed typed RAM proof".into()))?
}
