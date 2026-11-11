//! Structural proof sizes, split into PCS, Mosaic and IOP parts.
//!
//! LOCAL FALLBACK. The agreed interface is `pub fn proof_size(&Proof) ->
//! ProofSize` in each argument crate (see CONVENTIONS.md). Neither crate has
//! it on this branch, so this module counts the public fields of the current
//! proof structs. When the crates gain `proof_size`, replace the bodies of
//! `twist`, `spice` and `spice_typed` with calls to it and delete the
//! walkers below.
//!
//! Counting rule (CONVENTIONS.md): a field element is 16 bytes, a hash or
//! Merkle root is 32 bytes (counted as its actual digest length), an index is
//! 4 bytes, and a length prefix is free. Values the verifier recomputes but
//! the proof still carries (for example Twist's `r_o`, `r_c`) are counted,
//! because they are in the proof object.

use om_mosaic::BatchProof;
use om_pcs::{BitProof, Commitment, LigeritoProof, RingSwitchProof};
use om_sumcheck::SumcheckProof;
use serde::Serialize;

pub const FIELD: usize = 16;
pub const INDEX: usize = 4;
/// Flock grinding nonces are 64-bit values, not indices.
pub const NONCE: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ProofSize {
    /// Commitment (root and size index) plus the RingSwitch and Ligerito opening.
    pub pcs_bytes: usize,
    /// Mosaic batching sum-check and its claimed value.
    pub mosaic_bytes: usize,
    /// Everything else: the RAM argument's own messages and evaluations.
    pub iop_bytes: usize,
    pub total_bytes: usize,
    /// Where the numbers came from: `local-structural-fallback` or `crate`.
    pub source: &'static str,
}

impl ProofSize {
    fn new(pcs_bytes: usize, mosaic_bytes: usize, iop_bytes: usize) -> Self {
        Self {
            pcs_bytes,
            mosaic_bytes,
            iop_bytes,
            total_bytes: pcs_bytes + mosaic_bytes + iop_bytes,
            source: "local-structural-fallback",
        }
    }
}

fn fields(values: &[impl Sized]) -> usize {
    values.len() * FIELD
}

pub fn sumcheck(p: &SumcheckProof) -> usize {
    p.round_msgs.iter().map(|m| fields(m)).sum()
}

fn commitment(c: &Commitment) -> usize {
    c.root.len() + INDEX
}

pub fn ligerito(p: &LigeritoProof) -> usize {
    p.roots.iter().map(Vec::len).sum::<usize>()
        + p.sumchecks.iter().map(sumcheck).sum::<usize>()
        + p.openings
            .iter()
            .map(|o| {
                o.columns.iter().map(|c| fields(c)).sum::<usize>()
                    + o.auth.iter().map(Vec::len).sum::<usize>()
            })
            .sum::<usize>()
        + fields(&p.final_poly)
}

fn ring_switch(p: &RingSwitchProof) -> usize {
    fields(&p.s) + sumcheck(&p.sumcheck)
}

fn pcs(c: &Commitment, opening: &BitProof) -> usize {
    commitment(c) + ring_switch(&opening.0) + ligerito(&opening.1)
}

fn mosaic(p: &BatchProof) -> usize {
    sumcheck(&p.sumcheck) + FIELD
}

/// Upstream BinaryTwist (`ram_argument::RamProof`).
/// Twist v2: PCS and Mosaic counted as for the other methods, the rest from
/// `ram_argument::twist2::proof_size`.
pub fn twist2(p: &ram_argument::twist2::Proof) -> ProofSize {
    let mut size = ProofSize::new(
        pcs(&p.commitment, &p.opening),
        mosaic(&p.mosaic),
        ram_argument::twist2::proof_size(p).iop_bytes,
    );
    size.source = "crate";
    size
}

pub fn twist(p: &ram_argument::RamProof) -> ProofSize {
    let c = &p.input_claims;
    let claims = fields(&c.r_o)
        + fields(&c.r_c)
        + fields(&c.rv)
        + FIELD
        + c.bits.iter().map(|b| fields(b)).sum::<usize>();
    let t = &p.twist;
    let mut iop = claims
        + sumcheck(&t.rw)
        + fields(&t.rw_evals)
        + sumcheck(&t.val)
        + fields(&t.val_evals)
        + t.fin.as_ref().map_or(0, sumcheck)
        + t.fin_evals.as_ref().map_or(0, |v| fields(v))
        + t.inc_extra.map_or(0, |_| FIELD)
        + sumcheck(&t.onehot.sc)
        + t.onehot.evals.iter().map(|v| fields(v)).sum::<usize>();
    if let Some(v) = &t.virt {
        iop += sumcheck(&v.sc) + fields(&v.evals);
    }
    if let Some(v) = &t.virt_tree {
        iop += v
            .layers
            .iter()
            .map(|l| sumcheck(&l.sumcheck) + fields(&l.child_evals))
            .sum::<usize>();
    }
    if let Some(v) = &t.virt_flat {
        iop += sumcheck(&v.reduction.sc)
            + fields(&v.reduction.evals)
            + sumcheck(&v.sc)
            + fields(&v.evals);
    }
    ProofSize::new(pcs(&p.commitment, &p.opening), mosaic(&p.mosaic), iop)
}

fn grand_product(p: &lean_grandproduct::GrandProductProof) -> usize {
    fields(&p.roots)
        + p.layers
            .iter()
            .map(|l| {
                l.rounds.len() * 4 * FIELD + l.children.iter().map(|c| fields(c)).sum::<usize>()
            })
            .sum::<usize>()
}

fn flock_order(p: &spice_argument::order_flock::OrderProof) -> usize {
    let z = &p.zerocheck;
    fields(&z.round1_ab)
        + fields(&z.round1_c)
        + z.multilinear_rounds.len() * 2 * FIELD
        + 3 * FIELD
        + z.grinding_nonces.len() * NONCE
        + fields(&p.previous_ab)
        + fields(&p.carries_ab)
        + fields(&p.carries_c)
}

/// Untyped Spice (`spice` and `spice-ripple`).
pub fn spice(p: &spice_argument::Proof) -> ProofSize {
    let i = &p.inputs;
    let inputs = fields(&i.address) + fields(&i.previous) + fields(&i.final_previous) + 2 * FIELD;
    let order = match &p.order {
        spice_argument::OrderingProof::Flock(o) => flock_order(o),
        spice_argument::OrderingProof::Ripple(o) => {
            sumcheck(&o.sc) + fields(&o.previous) + fields(&o.carries)
        }
    };
    ProofSize::new(
        pcs(&p.commitment, &p.opening),
        mosaic(&p.mosaic),
        grand_product(&p.permutation) + inputs + order,
    )
}

/// Typed Spice (`spice-typed`).
pub fn spice_typed(p: &spice_argument::operations::Proof) -> ProofSize {
    let i = &p.inputs;
    let inputs = fields(&i.address)
        + fields(&i.previous)
        + fields(&i.final_previous)
        + 3 * FIELD
        + fields(&i.masked.cycle)
        + FIELD;
    let masks = sumcheck(&p.masks.sc) + 4 * FIELD;
    ProofSize::new(
        pcs(&p.commitment, &p.opening),
        mosaic(&p.mosaic),
        grand_product(&p.permutation) + inputs + masks + flock_order(&p.order),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ligerito_count_matches_reference_byte_layout_and_parts_add_up() {
        let trace = ram_argument::AccessTrace {
            initial: (0..16).collect(),
            reads: vec![vec![3, 5, 3, 9]],
            writes: vec![3, 5, 3, 9],
            new_values: vec![7, 5, 8, 1],
            read_values: None,
        };
        let config = ram_argument::Config {
            address_bits: 4,
            chunk_bits: 2,
            ..Default::default()
        };
        let result = ram_argument::prove(&config, &trace).unwrap();
        let opening = &result.proof.opening.1;
        // The reference layout (spec §4.8) has no length prefixes, matching
        // the counting rule exactly.
        assert_eq!(ligerito(opening), opening.to_bytes().len());
        let size = twist(&result.proof);
        assert_eq!(
            size.total_bytes,
            size.pcs_bytes + size.mosaic_bytes + size.iop_bytes
        );
        assert!(size.pcs_bytes > size.iop_bytes && size.iop_bytes > 0);
        assert_eq!(
            size.mosaic_bytes,
            sumcheck(&result.proof.mosaic.sumcheck) + FIELD
        );
        // Input claims: r_o (6) + r_c (2) + rv (1) + wv (1) + 2 x 4 bits.
        let c = &result.proof.input_claims;
        assert_eq!(
            (c.r_o.len(), c.r_c.len(), c.bits.len()),
            (6, 2, 2),
            "test assumes w=6, n=2, one read port"
        );
    }

    #[test]
    fn typed_spice_size_counts_every_part() {
        let trace = spice_argument::operations::AccessTrace {
            initial: (0..16).collect(),
            addresses: vec![3, 5, 3, 9],
            values: vec![3, 77, 3, 9],
            is_write: vec![false, true, false, false],
        };
        let config = ram_argument::Config {
            address_bits: 4,
            chunk_bits: 2,
            ..Default::default()
        };
        let result = spice_argument::operations::prove(&config, &trace).unwrap();
        let size = spice_typed(&result.proof);
        assert_eq!(
            size.total_bytes,
            size.pcs_bytes + size.mosaic_bytes + size.iop_bytes
        );
        assert!(size.iop_bytes > grand_product(&result.proof.permutation));
        assert_eq!(
            size.pcs_bytes,
            32 + INDEX + ring_switch(&result.proof.opening.0) + ligerito(&result.proof.opening.1)
        );
    }
}
