//! Composed soundness error per method: PCS terms plus IOP terms, union-bounded.
//!
//! PCS terms come from `om_pcs::ligerito::soundness_bits` on the parameters
//! the provers actually use, `make_params(ell - 7, profile, log_rate)`, where
//! `2^ell` is the committed (padded) bit count. Every level contributes its
//! sum-check, beta-batching, proximity-gap (already including repetitions)
//! and query terms.
//!
//! IOP terms: `twist` (Twist v2) and the typed Spice methods use each
//! crate's `soundness_terms`, plus the ring switch, which the crates leave to
//! the caller. `twist-legacy` uses a local fallback derived by reading the
//! verifier at 632855e; each term names the verifier code it comes from.
//! `spice_typed_iop` is the same kind of derivation for typed Spice at
//! 632855e, kept so the baseline's soundness can be reported next to its
//! timings. Untyped `spice`/`spice-ripple` get only the grand-product
//! fingerprint term and are marked partial.
//!
//! Conventions: every term is a probability. A sum-check with `r` rounds of
//! degree `d` contributes `r*d/|K|`; a random linear combination of claims
//! contributes `(claims-1)/|K|` (powers of one challenge) or `1/|K|`
//! (independent weights); a Schwartz-Zippel check of a multilinear identity
//! at a random point in `K^v` contributes `v/|K|`. `|K| = 2^128`. Fiat-Shamir
//! and hash-collision terms are out of scope, as are Flock grinding bits
//! (grinding is disabled in this build: the proofs carry no nonces).

use serde::Serialize;

const FIELD_BITS: i32 = 128;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Term {
    pub name: String,
    /// `pcs` or `iop`.
    pub group: &'static str,
    pub probability: f64,
    pub bits: f64,
    /// Verifier code the term comes from.
    pub location: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Soundness {
    /// `local-fallback`, or `partial` when IOP terms beyond the fingerprint
    /// have not been derived for this method.
    pub iop_source: &'static str,
    pub pcs_probability: f64,
    pub iop_probability: f64,
    pub total_probability: f64,
    /// `-log2(total_probability)`.
    pub bits: f64,
    pub pcs_bits: f64,
    pub iop_bits: f64,
    pub terms: Vec<Term>,
}

/// The dimensions the terms depend on, all taken from the run.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    /// `log2 T`.
    pub n: usize,
    /// `log2 K` (memory words).
    pub a: usize,
    /// Twist address chunk bits (`d = a / c` components).
    pub chunk_bits: usize,
    /// `log2` bits per word (6 for 64-bit words).
    pub word_log_bits: usize,
    /// `log2` of the committed (padded) bit count.
    pub ell: usize,
    /// Claims passed to Mosaic.
    pub mosaic_claims: usize,
}

fn over_field(numerator: f64) -> f64 {
    numerator * 2f64.powi(-FIELD_BITS)
}

fn iop(name: &str, numerator: usize, location: &'static str) -> Term {
    let probability = over_field(numerator as f64);
    Term {
        name: name.into(),
        group: "iop",
        probability,
        bits: -probability.log2(),
        location,
    }
}

/// Ligerito terms for the actual parameters, one per level and kind.
pub fn pcs_terms(ell: usize, profile: &str, log_rate: usize) -> Vec<Term> {
    let params = om_pcs::ligerito::make_params(ell - 7, profile, log_rate);
    om_pcs::ligerito::soundness_bits(&params, profile)
        .into_iter()
        .flat_map(|level| {
            [
                ("sumcheck", level.sumcheck),
                ("batch", level.batch),
                ("gap", level.gap),
                ("query", level.query),
            ]
            .map(|(kind, bits)| Term {
                name: format!("ligerito.level{}.{kind}", level.level),
                group: "pcs",
                probability: 2f64.powf(-bits),
                bits,
                location: "vendor/open-magnetar/crates/om-pcs/src/ligerito.rs soundness_bits",
            })
        })
        .collect()
}

/// Mosaic batching and RingSwitch, shared by every method.
fn mosaic_and_ring_switch(d: &Dims) -> Vec<Term> {
    let mut terms = vec![
        // Powers of one gamma over all claims.
        iop(
            "mosaic.gamma_batch",
            d.mosaic_claims.saturating_sub(1),
            "om-mosaic/src/mosaic.rs batch_verify (gamma powers)",
        ),
        // ell rounds, degree 2.
        iop(
            "mosaic.sumcheck",
            2 * d.ell,
            "om-mosaic/src/mosaic.rs batch_verify (sumcheck_verify)",
        ),
    ];
    terms.extend(ring_switch(d));
    terms
}

fn ring_switch(d: &Dims) -> Vec<Term> {
    vec![
        // rho in K^7 combines the 128 partial evaluations (multilinear in rho).
        iop(
            "ringswitch.rho",
            7,
            "om-pcs/src/ring_switch.rs verify (rs/rho)",
        ),
        // ell - 7 rounds, degree 2.
        iop(
            "ringswitch.sumcheck",
            2 * (d.ell - 7),
            "om-pcs/src/ring_switch.rs verify (sumcheck_verify)",
        ),
    ]
}

/// Upstream BinaryTwist with shared address, one read port, Flat
/// virtualization and public final memory, as `ram_argument::verify` runs it.
/// Twist v2, from `ram_argument::twist2::soundness_terms` (which includes
/// Mosaic), plus the ring switch.
pub fn twist2_iop(d: &Dims, public_final: bool) -> Vec<Term> {
    let config = ram_argument::twist2::Config {
        address_bits: d.a,
        chunk_bits: d.chunk_bits,
        word_log_bits: d.word_log_bits,
        public_final,
        ..Default::default()
    };
    let statement = ram_argument::twist2::Statement {
        config,
        log_cycles: d.n,
        initial_memory: Vec::new(),
        final_memory: None,
    };
    let mut terms: Vec<Term> = ram_argument::twist2::soundness_terms(&statement)
        .into_iter()
        .map(|(name, probability)| Term {
            name: format!("twist2.{name}"),
            group: "iop",
            probability,
            bits: -probability.log2(),
            location: "crates/ram-argument/src/twist2/mod.rs soundness_terms",
        })
        .collect();
    terms.extend(ring_switch(d));
    terms
}

pub fn twist_iop(d: &Dims) -> Vec<Term> {
    let (n, a, c, w) = (d.n, d.a, d.chunk_bits, d.word_log_bits);
    let components = a / c;
    let mut terms = vec![
        // rv, wv and the address-bit claims are checked as multilinear
        // identities at the post-commitment point (r_o, r_c) in K^(w+n).
        iop(
            "twist.input_point",
            w + n,
            "crates/ram-argument/src/lib.rs verify_inner (ram/point_o, ram/point_c)",
        ),
        // alpha0, alpha and beta fold the per-access claims linearly.
        iop(
            "twist.rw.combination",
            2,
            "om-memcheck/src/twist.rs twist_verify_with_virtualization (coeffs)",
        ),
        iop(
            "twist.rw.sumcheck",
            3 * (n + a),
            "om-memcheck/src/twist.rs sumcheck_verify(.., n + a, 3, /rw)",
        ),
        iop(
            "twist.val.sumcheck",
            3 * n,
            "om-memcheck/src/twist.rs sumcheck_verify(.., n, 3, /val)",
        ),
        // Final memory compared as an MLE at (r_low, r_o) in K^(a+w).
        iop(
            "twist.final.point",
            a + w,
            "om-memcheck/src/twist.rs /fin/r_low, /fin/r_o",
        ),
        iop(
            "twist.final.sumcheck",
            2 * n,
            "om-memcheck/src/twist.rs sumcheck_verify(.., n, 2, /fin)",
        ),
        iop(
            "virt.flat.claim_weights",
            1,
            "om-memcheck/src/virt_flat.rs verify (nu)",
        ),
        iop(
            "virt.flat.reduction.sumcheck",
            2 * n,
            "om-memcheck/src/virt_claim_reduction.rs verify (degree 2, n rounds)",
        ),
        iop(
            "virt.flat.product_weights",
            1,
            "om-memcheck/src/virt_flat.rs verify (lambda)",
        ),
        // Product of d component evaluations times eq: degree d + 1.
        iop(
            "virt.flat.product.sumcheck",
            (components + 1) * n,
            "om-memcheck/src/virt_flat.rs verify (degree d+1, n rounds)",
        ),
        // In characteristic 2, C*eq(r',u) = eq(r,u)^3; the difference for a
        // row of weight >= 2 has total degree <= 3c in r.
        iop(
            "onehot.point",
            3 * c,
            "om-memcheck/src/onehot.rs onehot_verify (point r in K^c)",
        ),
        iop(
            "onehot.zerocheck_point",
            n,
            "om-memcheck/src/onehot.rs onehot_verify (rho in K^n)",
        ),
        iop(
            "onehot.component_batch",
            1,
            "om-memcheck/src/onehot.rs onehot_verify (lambda)",
        ),
        iop(
            "onehot.sumcheck",
            4 * n,
            "om-memcheck/src/onehot.rs onehot_verify (degree 4, n rounds)",
        ),
    ];
    terms.extend(mosaic_and_ring_switch(d));
    terms
}

#[allow(dead_code)]
/// Rounds summed over the batched radix-4 GKR layers of a grand product
/// with `2^log_size` leaves (lean-grandproduct `verify`).
pub fn grand_product_rounds(log_size: usize) -> (usize, usize) {
    let (mut layer, mut layers, mut rounds) = (log_size, 0, 0);
    while layer > 0 {
        rounds += log_size - layer;
        layers += 1;
        layer -= if layer % 2 != 0 { 1 } else { 2 };
    }
    (layers, rounds)
}

fn fingerprint(d: &Dims) -> Term {
    // Each leaf is affine in (gamma, alpha, beta, delta) with coefficient 1
    // on gamma, so each tree's product has degree T + K.
    iop(
        "spice.fingerprint",
        (1usize << d.n) + (1usize << d.a),
        "crates/spice-argument/src/operations.rs check_leaves (gamma, alpha, beta, delta)",
    )
}

/// Typed Spice, from `spice_argument::operations::soundness_terms` (which
/// includes Mosaic), plus the ring switch.
pub fn spice_typed_crate(d: &Dims, fingerprints: usize, public_final: bool) -> Vec<Term> {
    let statement = ram_argument::Statement {
        config: ram_argument::Config {
            address_bits: d.a,
            word_log_bits: d.word_log_bits,
            ..Default::default()
        },
        log_cycles: d.n,
        initial_memory: Vec::new(),
        final_memory: Vec::new(),
    };
    let options = spice_argument::operations::Options {
        fingerprints,
        private_final_memory: !public_final,
    };
    let mut terms: Vec<Term> = spice_argument::operations::soundness_terms(&statement, &options)
        .into_iter()
        .map(|(name, probability)| Term {
            name: format!("spice.{name}"),
            group: "iop",
            probability,
            bits: -probability.log2(),
            location: "crates/spice-argument/src/operations.rs soundness_terms",
        })
        .collect();
    terms.extend(ring_switch(d));
    terms
}

/// Typed Spice as `spice_argument::operations::verify` ran it at 632855e
/// (one fingerprint, one Mosaic claim per bit plane).
#[allow(dead_code)]
pub fn spice_typed_iop(d: &Dims) -> Vec<Term> {
    let l = d.n.max(d.a);
    let (layers, rounds) = grand_product_rounds(l + 1);
    let padded_n = d.n.max(7);
    let mut terms = vec![
        fingerprint(d),
        iop(
            "gkr.tree_batch",
            layers,
            "vendor/lean-grandproduct/src/lib.rs verify (/batch lambda per layer)",
        ),
        // Round messages carry 4 coefficients; the eq factor is divided out.
        iop(
            "gkr.sumcheck",
            4 * rounds,
            "vendor/lean-grandproduct/src/lib.rs verify (degree 4 per round)",
        ),
        iop(
            "gkr.child_point",
            l + 1,
            "vendor/lean-grandproduct/src/lib.rs bind_children",
        ),
        iop(
            "mask.weights",
            1,
            "crates/spice-argument/src/masked_reduction.rs verify (weights)",
        ),
        iop(
            "mask.sumcheck",
            3 * l,
            "crates/spice-argument/src/masked_reduction.rs verify (degree 3, l rounds)",
        ),
        // Seven of the eq coordinates are fixed constants; only the
        // remaining random ones count.
        iop(
            "flock.zerocheck_point",
            padded_n - 7,
            "vendor/flock-boolean/crates/flock-core/src/zerocheck.rs verify (r_outer)",
        ),
        // The verifier interpolates the skip round from 2 * 2^6 points.
        iop(
            "flock.univariate_skip",
            2 * 64 - 1,
            "vendor/flock-boolean/crates/flock-core/src/zerocheck.rs verify (round1_ab)",
        ),
        iop(
            "flock.multilinear_rounds",
            2 * padded_n,
            "vendor/flock-boolean/crates/flock-core/src/zerocheck.rs verify (degree 2 rounds)",
        ),
    ];
    terms.extend(mosaic_and_ring_switch(d));
    terms
}

fn sum(terms: &[Term], group: &str) -> f64 {
    terms
        .iter()
        .filter(|t| t.group == group)
        .map(|t| t.probability)
        .sum()
}

/// Union bound over all PCS and IOP terms for one case.
pub fn compose(
    method: &str,
    d: &Dims,
    profile: &str,
    log_rate: usize,
    public_final: bool,
) -> Soundness {
    let (iop_source, mut terms) = match method {
        "twist" => ("crate", twist2_iop(d, public_final)),
        "twist-legacy" => ("local-fallback", twist_iop(d)),
        "spice-typed" => ("crate", spice_typed_crate(d, 1, public_final)),
        "spice-typed-2fp" => ("crate", spice_typed_crate(d, 2, public_final)),
        _ => ("partial", vec![fingerprint(d)]),
    };
    terms.extend(pcs_terms(d.ell, profile, log_rate));
    let pcs_probability = sum(&terms, "pcs");
    let iop_probability = sum(&terms, "iop");
    let total_probability = pcs_probability + iop_probability;
    Soundness {
        iop_source,
        pcs_probability,
        iop_probability,
        total_probability,
        bits: -total_probability.log2(),
        pcs_bits: -pcs_probability.log2(),
        iop_bits: -iop_probability.log2(),
        terms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerator(terms: &[Term]) -> f64 {
        terms.iter().map(|t| t.probability).sum::<f64>() * 2f64.powi(FIELD_BITS)
    }

    fn twist_dims(n: usize, a: usize, ell: usize, claims: usize) -> Dims {
        Dims {
            n,
            a,
            chunk_bits: 4,
            word_log_bits: 6,
            ell,
            mosaic_claims: claims,
        }
    }

    #[test]
    fn grand_product_round_schedule_matches_the_verifier() {
        // log_size 21: layers 21 (radix 2), then 20, 18, ..., 2 (radix 4).
        assert_eq!(grand_product_rounds(21), (11, 100));
        assert_eq!(grand_product_rounds(17), (9, 64));
        assert_eq!(grand_product_rounds(4), (2, 2));
    }

    #[test]
    fn iop_totals_match_the_hand_derivation() {
        // n=20, a=16, c=4: d=4, ell=28, 7d+5=33 Mosaic claims.
        assert_eq!(
            numerator(&twist_iop(&twist_dims(20, 16, 28, 33))).round(),
            650.0
        );
        // n=16, a=12: d=3, ell=24, 26 claims.
        assert_eq!(
            numerator(&twist_iop(&twist_dims(16, 12, 24, 26))).round(),
            511.0
        );
        // Typed Spice: ell=27, 126 claims; ell=23, 102 claims.
        let typed = spice_typed_iop(&twist_dims(20, 16, 27, 126));
        assert_eq!(numerator(&typed).round(), (1_114_112 + 899) as f64);
        let typed = spice_typed_iop(&twist_dims(16, 12, 23, 102));
        assert_eq!(numerator(&typed).round(), (69_632 + 685) as f64);
    }

    #[test]
    fn composed_bits_at_the_shipped_parameters() {
        let twist = compose(
            "twist-legacy",
            &twist_dims(20, 16, 28, 33),
            "provable-120",
            2,
            true,
        );
        let typed = compose(
            "spice-typed",
            &twist_dims(20, 16, 27, 19),
            "provable-120",
            2,
            true,
        );
        assert!((twist.bits - 117.06).abs() < 0.01, "{}", twist.bits);
        assert!((typed.bits - 107.91).abs() < 0.01, "{}", typed.bits);
        let two = compose(
            "spice-typed-2fp",
            &twist_dims(20, 16, 27, 19),
            "provable-120",
            2,
            true,
        );
        assert!((two.bits - 116.92).abs() < 0.01, "{}", two.bits);
        assert!(twist.pcs_probability > twist.iop_probability);
        assert!(typed.iop_probability > typed.pcs_probability);
        assert_eq!(twist.terms.iter().filter(|t| t.group == "pcs").count(), 12);
        let partial = compose(
            "spice",
            &twist_dims(20, 16, 28, 120),
            "provable-120",
            2,
            true,
        );
        assert_eq!(partial.iop_source, "partial");
        // Twist v2 at n=20, a=16, c=4 with 10 Mosaic claims.
        let v2 = compose(
            "twist",
            &twist_dims(20, 16, 28, 10),
            "provable-120",
            2,
            true,
        );
        assert_eq!(v2.iop_source, "crate");
        assert!(v2.iop_probability < twist.iop_probability);
    }
}
