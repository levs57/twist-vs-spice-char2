//! BinaryShout (spec 07 §7.4). Mirrors `reference/openmagnetar/shout.py`
//! byte for byte in its transcript interaction.

use om_field::K;
use om_sumcheck::{
    SumcheckProof, VerificationError, mle_eval, sumcheck_prove_weighted_mul2, sumcheck_verify,
};
use om_transcript::{Transcript, u64le};
use rayon::prelude::*;

use crate::onehot::PAR_THRESHOLD;
use crate::onehot::{
    AddrComps, EvalClaim, OneHotProof, VirtGroup, VirtProof, VirtTerm, VirtualizationMode,
    onehot_prove, onehot_verify, virt_prove, virt_verify,
};
use crate::sublabel;
use crate::virt_flat::{VirtFlatProof, virt_flat_prove, virt_flat_verify};
use crate::virt_tree::{VirtTreeProof, virt_tree_prove, virt_tree_verify};
use om_transcript::profiling::Span;

/// Parameters of one BinaryShout instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShoutParams {
    pub name: String,
    /// Table has `2^a` rows.
    pub a: usize,
    /// Number of address components, `d | a`.
    pub d: usize,
    /// `T = 2^n` cycles.
    pub n: usize,
}

/// One claim group: all claims at one cycle point `r_c`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShoutClaims {
    /// Cycle point, `K^n`.
    pub r_c: Vec<K>,
    /// Indices of the table columns claimed (increasing).
    pub cols: Vec<usize>,
    /// `rv_h~(r_c)` for `h` in `cols`.
    pub rv: Vec<K>,
    /// `f_t~(r_c)`, `t = 0..a-1` (`t = 0` is the LSB), or `None`.
    pub bits: Option<Vec<K>>,
}

/// Flat virtualization uses the already-common point for a single incoming
/// read claim, and the general reduction for several incoming cycle points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShoutFlatProof {
    SinglePoint(VirtProof),
    MultiPoint(VirtFlatProof),
}

/// Proof of one BinaryShout instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShoutProof {
    pub read: SumcheckProof,
    pub ra_evals: Vec<K>,
    pub virt: Option<VirtProof>,
    pub virt_flat: Option<ShoutFlatProof>,
    pub virt_tree: Option<VirtTreeProof>,
    pub onehot: OneHotProof,
}

/// The component polynomial names `{name}/ra/{i}`, `i` 1-based.
pub fn comp_names(params: &ShoutParams) -> Vec<String> {
    (0..params.d)
        .map(|i| format!("{}/ra/{}", params.name, i + 1))
        .collect()
}

/// `ra~(k, r_c)` for all Boolean `k` (length `2^a`): a histogram of
/// the addresses weighted by `eq(r_c, j)` (§7.10), `O(T + 2^a)`.
fn ra_at(params: &ShoutParams, comps: &AddrComps, r_c: &[K]) -> Vec<K> {
    comps.ra_at(params.a, r_c)
}

/// The honest claim group at cycle point `r_c` for the given table and
/// address components. `cols = None` claims every column.
pub fn shout_honest_claims(
    params: &ShoutParams,
    table: &[Vec<K>],
    comps: &AddrComps,
    r_c: &[K],
    cols: Option<&[usize]>,
    with_bits: bool,
) -> ShoutClaims {
    let cols: Vec<usize> = match cols {
        None => (0..table.len()).collect(),
        Some(c) => c.to_vec(),
    };
    let ra = ra_at(params, comps, r_c);
    let mut rv = Vec::with_capacity(cols.len());
    for &h in &cols {
        let mut acc = K::ZERO;
        for (k, &v) in ra.iter().enumerate() {
            acc += v * table[h][k];
        }
        rv.push(acc);
    }
    let bits = if with_bits {
        let mut bits = Vec::with_capacity(params.a);
        for t in 0..params.a {
            let mut acc = K::ZERO;
            for (k, &v) in ra.iter().enumerate() {
                if (k >> t) & 1 == 1 {
                    acc += v;
                }
            }
            bits.push(acc);
        }
        Some(bits)
    } else {
        None
    };
    ShoutClaims {
        r_c: r_c.to_vec(),
        cols,
        rv,
        bits,
    }
}

fn phi_table(
    alpha0: K,
    alphas: Option<&[K]>,
    gammas: &[K],
    table: &[Vec<K>],
    cols: &[usize],
    a: usize,
) -> Vec<K> {
    let nk = 1usize << a;
    // Build the affine bit contribution with one XOR per new entry rather
    // than scanning all address bits for every row. ROM columns below remain
    // arbitrary multilinear tables and are retained in full.
    let mut out = Vec::with_capacity(nk);
    out.push(alpha0);
    for t in 0..a {
        let alpha = alphas.and_then(|xs| xs.get(t)).copied().unwrap_or(K::ZERO);
        let len = out.len();
        for i in 0..len {
            out.push(out[i] + alpha);
        }
    }
    for (&g, &h) in gammas.iter().zip(cols) {
        if nk >= PAR_THRESHOLD {
            out.par_iter_mut()
                .zip(&table[h])
                .for_each(|(v, &tv)| *v += g * tv);
        } else {
            for (v, &tv) in out.iter_mut().zip(&table[h]) {
                *v += g * tv;
            }
        }
    }
    out
}

fn absorb(tr: &mut Transcript, label: &[u8], groups: &[ShoutClaims]) {
    tr.absorb_bytes(&sublabel(label, b"/ngroups"), &u64le(groups.len() as u64));
    for g in groups {
        tr.absorb_ks(&sublabel(label, b"/point"), &g.r_c);
        let mut cols_bytes = Vec::with_capacity(8 * g.cols.len() + 1);
        for &h in &g.cols {
            cols_bytes.extend_from_slice(&u64le(h as u64));
        }
        cols_bytes.push(if g.bits.is_some() { 0x01 } else { 0x00 });
        tr.absorb_bytes(&sublabel(label, b"/cols"), &cols_bytes);
        tr.absorb_ks(&sublabel(label, b"/rv"), &g.rv);
        if let Some(bits) = &g.bits {
            tr.absorb_ks(&sublabel(label, b"/bits"), bits);
        }
    }
}

/// Per group `(α_{g,0}, α_{g,1..a} or None, γ_g)`, then `β`.
type GroupCoeffs = (K, Option<Vec<K>>, Vec<K>);

fn coeffs(
    tr: &mut Transcript,
    label: &[u8],
    a: usize,
    groups: &[ShoutClaims],
) -> (Vec<GroupCoeffs>, Vec<K>) {
    let mut out = Vec::with_capacity(groups.len());
    for g in groups {
        let alpha0 = tr.challenge_nonzero_k(&sublabel(label, b"/alpha0"));
        let alphas = if g.bits.is_some() {
            Some(tr.challenge_ks(&sublabel(label, b"/alpha"), a))
        } else {
            None
        };
        let gammas = tr.challenge_ks(&sublabel(label, b"/gamma"), g.cols.len());
        out.push((alpha0, alphas, gammas));
    }
    let beta = tr.challenge_ks(&sublabel(label, b"/beta"), groups.len());
    (out, beta)
}

/// The verifier's per-group claim `alpha0 + Σ α f + Σ γ rv`.
fn group_claim(g: &ShoutClaims, co: &GroupCoeffs) -> K {
    let (alpha0, alphas, gammas) = co;
    let mut c = *alpha0;
    if let (Some(alphas), Some(bits)) = (alphas, &g.bits) {
        for (&al, &f) in alphas.iter().zip(bits) {
            c += al * f;
        }
    }
    for (&ga, &v) in gammas.iter().zip(&g.rv) {
        c += ga * v;
    }
    c
}

fn validate_shape(
    params: &ShoutParams,
    table: &[Vec<K>],
    groups: &[ShoutClaims],
) -> Result<(), VerificationError> {
    let (a, d, n) = (params.a, params.d, params.n);
    if a == 0 || a > 32 || d == 0 || !a.is_multiple_of(d) || n >= usize::BITS as usize {
        return Err(VerificationError::new("shout: invalid dimensions"));
    }
    if groups.is_empty()
        || table.iter().any(|column| column.len() != 1usize << a)
        || groups.iter().any(|group| {
            group.r_c.len() != n
                || group.rv.len() != group.cols.len()
                || group.bits.as_ref().is_some_and(|bits| bits.len() != a)
                || group.cols.iter().any(|&h| h >= table.len())
        })
    {
        return Err(VerificationError::new("shout: malformed claims or table"));
    }
    Ok(())
}

fn absorb_virtualization_mode(tr: &mut Transcript, label: &[u8], mode: VirtualizationMode) {
    match mode {
        VirtualizationMode::FlatLegacy => {}
        VirtualizationMode::Flat => tr.absorb_bytes(
            &sublabel(label, b"/virtualization-mode"),
            b"flat/common-cycle/v1",
        ),
        VirtualizationMode::GkrTree => {
            tr.absorb_bytes(&sublabel(label, b"/virtualization-mode"), b"gkr-tree/v1")
        }
    }
}

/// Prove one BinaryShout instance (spec 07 §7.4). `table` is the public
/// table: `c` columns of `2^a` field elements. Returns the proof and
/// the output claims (virtualization claims, then one-hot claims).
pub fn shout_prove(
    params: &ShoutParams,
    table: &[Vec<K>],
    comps: &AddrComps,
    groups: &[ShoutClaims],
    transcript: &mut Transcript,
    label: &[u8],
) -> (ShoutProof, Vec<EvalClaim>) {
    shout_prove_with_virtualization(
        params,
        table,
        comps,
        groups,
        transcript,
        label,
        VirtualizationMode::FlatLegacy,
    )
}

/// Prove Shout with the selected address virtualization protocol. The legacy
/// entry point retains its original proof messages and transcript interaction.
pub fn shout_prove_with_virtualization(
    params: &ShoutParams,
    table: &[Vec<K>],
    comps: &AddrComps,
    groups: &[ShoutClaims],
    transcript: &mut Transcript,
    label: &[u8],
    mode: VirtualizationMode,
) -> (ShoutProof, Vec<EvalClaim>) {
    validate_shape(params, table, groups).expect("shout: input shape");
    assert!(
        mode != VirtualizationMode::GkrTree || params.d.is_power_of_two(),
        "tree virtualization requires a power-of-two component count"
    );
    assert!(
        crate::virt_tree::component_shape(comps, params.a, params.d, params.n),
        "shout: malformed address components"
    );
    absorb_virtualization_mode(transcript, label, mode);
    let (a, d, n) = (params.a, params.d, params.n);
    absorb(transcript, label, groups);
    let (cos, beta) = coeffs(transcript, label, a, groups);
    let (sc, r_k, mles) = {
        let _span = Span::new("shout.read");
        let ra_rs: Vec<Vec<K>> = groups
            .iter()
            .map(|g| ra_at(params, comps, &g.r_c))
            .collect();
        let phis: Vec<Vec<K>> = groups
            .iter()
            .zip(&cos)
            .map(|(g, co)| phi_table(co.0, co.1.as_deref(), &co.2, table, &g.cols, a))
            .collect();
        let mut tables = ra_rs;
        tables.extend(phis);
        // Sum actual tables, retaining the unscaled RA terminal evaluations.
        sumcheck_prove_weighted_mul2(tables, &beta, transcript, &sublabel(label, b"/read"))
    };
    let ra_evals = mles[..groups.len()].to_vec();
    transcript.absorb_ks(&sublabel(label, b"/ra_evals"), &ra_evals);
    let names = comp_names(params);
    let grp = VirtGroup {
        names: names.clone(),
        r_k,
        terms: groups
            .iter()
            .zip(&ra_evals)
            .map(|(g, &v)| VirtTerm {
                cyc: g.r_c.clone(),
                value: v,
            })
            .collect(),
    };
    let (vproof, flat_proof, tree_proof, mut claims) = {
        let _span = Span::new("shout.virtualization");
        match mode {
            VirtualizationMode::FlatLegacy => {
                let (proof, claims) = virt_prove(
                    &[grp],
                    &[comps],
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (proof, None, None, claims)
            }
            VirtualizationMode::Flat if d > 1 && groups.len() == 1 => {
                let (proof, claims) = crate::virt_single::prove(
                    &grp,
                    comps,
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (None, Some(ShoutFlatProof::SinglePoint(proof)), None, claims)
            }
            VirtualizationMode::Flat => {
                let (proof, claims) = virt_flat_prove(
                    &[grp],
                    &[comps],
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (None, proof.map(ShoutFlatProof::MultiPoint), None, claims)
            }
            VirtualizationMode::GkrTree => {
                let (proof, claims) = virt_tree_prove(
                    &[grp],
                    &[comps],
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (None, None, proof, claims)
            }
        }
    };
    let (oproof, oclaims) = {
        let _span = Span::new("shout.onehot");
        onehot_prove(
            &[comps],
            &names,
            a / d,
            n,
            transcript,
            &sublabel(label, b"/onehot"),
        )
    };
    claims.extend(oclaims);
    (
        ShoutProof {
            read: sc,
            ra_evals,
            virt: vproof,
            virt_flat: flat_proof,
            virt_tree: tree_proof,
            onehot: oproof,
        },
        claims,
    )
}

/// Verify one BinaryShout instance, returning the output claims.
/// `table` is the public table: `c` columns of `2^a` field elements.
///
/// # Errors
/// Returns [`VerificationError`] on malformed claims or proof or any
/// failing check.
pub fn shout_verify(
    params: &ShoutParams,
    table: &[Vec<K>],
    groups: &[ShoutClaims],
    proof: &ShoutProof,
    transcript: &mut Transcript,
    label: &[u8],
) -> Result<Vec<EvalClaim>, VerificationError> {
    shout_verify_with_virtualization(
        params,
        table,
        groups,
        proof,
        transcript,
        label,
        VirtualizationMode::FlatLegacy,
    )
}

/// Verify the explicit virtualization mode and return all committed-selector
/// evaluation claims for the caller to authenticate with its PCS.
pub fn shout_verify_with_virtualization(
    params: &ShoutParams,
    table: &[Vec<K>],
    groups: &[ShoutClaims],
    proof: &ShoutProof,
    transcript: &mut Transcript,
    label: &[u8],
    mode: VirtualizationMode,
) -> Result<Vec<EvalClaim>, VerificationError> {
    validate_shape(params, table, groups)?;
    let (a, d, n) = (params.a, params.d, params.n);
    if proof.ra_evals.len() != groups.len()
        || proof.read.round_msgs.len() != a
        || proof
            .read
            .round_msgs
            .iter()
            .any(|message| message.len() != 2)
        || (d == 1
            && (proof.virt.is_some() || proof.virt_flat.is_some() || proof.virt_tree.is_some()))
        || match mode {
            VirtualizationMode::FlatLegacy => {
                proof.virt_flat.is_some() || proof.virt_tree.is_some()
            }
            VirtualizationMode::Flat => {
                proof.virt.is_some()
                    || proof.virt_tree.is_some()
                    || (d > 1 && proof.virt_flat.is_none())
            }
            VirtualizationMode::GkrTree => {
                !d.is_power_of_two()
                    || proof.virt.is_some()
                    || proof.virt_flat.is_some()
                    || (d > 1 && proof.virt_tree.is_none())
            }
        }
    {
        return Err(VerificationError::new(
            "shout: malformed proof or virtualization mode",
        ));
    }
    absorb_virtualization_mode(transcript, label, mode);
    absorb(transcript, label, groups);
    let (cos, beta) = coeffs(transcript, label, a, groups);
    let mut claim = K::ZERO;
    for ((&b, g), co) in beta.iter().zip(groups).zip(&cos) {
        claim += b * group_claim(g, co);
    }
    let (r_k, expected) = sumcheck_verify(
        claim,
        a,
        2,
        &proof.read,
        transcript,
        &sublabel(label, b"/read"),
    )?;
    let mut val = K::ZERO;
    for (((&b, g), co), &ra) in beta.iter().zip(groups).zip(&cos).zip(&proof.ra_evals) {
        let (alpha0, alphas, gammas) = co;
        let mut phi = *alpha0;
        if let Some(alphas) = alphas {
            for (&al, &x) in alphas.iter().zip(&r_k) {
                phi += al * x;
            }
        }
        for (&ga, &h) in gammas.iter().zip(&g.cols) {
            phi += ga * mle_eval(&table[h], &r_k);
        }
        val += b * ra * phi;
    }
    if val != expected {
        return Err(VerificationError::new("shout: read check failed"));
    }
    transcript.absorb_ks(&sublabel(label, b"/ra_evals"), &proof.ra_evals);
    let names = comp_names(params);
    let grp = VirtGroup {
        names: names.clone(),
        r_k,
        terms: groups
            .iter()
            .zip(&proof.ra_evals)
            .map(|(g, &v)| VirtTerm {
                cyc: g.r_c.clone(),
                value: v,
            })
            .collect(),
    };
    let mut claims = match mode {
        VirtualizationMode::FlatLegacy => virt_verify(
            &[grp],
            a,
            d,
            n,
            proof.virt.as_ref(),
            transcript,
            &sublabel(label, b"/virt"),
        )?,
        VirtualizationMode::Flat => match (d, groups.len(), proof.virt_flat.as_ref()) {
            (1, _, None) => virt_flat_verify(
                &[grp],
                a,
                d,
                n,
                None,
                transcript,
                &sublabel(label, b"/virt"),
            )?,
            (_, 1, Some(ShoutFlatProof::SinglePoint(proof))) => crate::virt_single::verify(
                &grp,
                a,
                d,
                n,
                proof,
                transcript,
                &sublabel(label, b"/virt"),
            )?,
            (_, count, Some(ShoutFlatProof::MultiPoint(proof))) if count > 1 => virt_flat_verify(
                &[grp],
                a,
                d,
                n,
                Some(proof),
                transcript,
                &sublabel(label, b"/virt"),
            )?,
            _ => {
                return Err(VerificationError::new(
                    "shout: wrong flat virtualization branch",
                ));
            }
        },
        VirtualizationMode::GkrTree => virt_tree_verify(
            &[grp],
            a,
            d,
            n,
            proof.virt_tree.as_ref(),
            transcript,
            &sublabel(label, b"/virt"),
        )?,
    };
    let oclaims = onehot_verify(
        &names,
        a / d,
        n,
        &proof.onehot,
        transcript,
        &sublabel(label, b"/onehot"),
    )?;
    claims.extend(oclaims);
    Ok(claims)
}

#[cfg(test)]
mod optimized_tests {
    use super::*;

    #[test]
    fn phi_affine_base_matches_bitwise_definition_with_arbitrary_rom() {
        for a in [0usize, 1, 3, 8, 14] {
            let n = 1usize << a;
            let alpha0 = K::from_u128(0xdead_beef);
            let alphas: Vec<K> = (0..a).map(|i| K::from_u128(0xbad5 << i)).collect();
            let table = vec![
                (0..n)
                    .map(|k| K::from_u128((k as u128 + 7).pow(3)))
                    .collect(),
            ];
            let gamma = K::from_u128(0xcafe_f00d);
            for bits in [None, Some(alphas.as_slice())] {
                let got = phi_table(alpha0, bits, &[gamma], &table, &[0], a);
                for (k, &value) in got.iter().enumerate() {
                    let mut expected = alpha0 + gamma * table[0][k];
                    if let Some(bits) = bits {
                        for (t, &alpha) in bits.iter().enumerate() {
                            if (k >> t) & 1 != 0 {
                                expected += alpha;
                            }
                        }
                    }
                    assert_eq!(value, expected);
                }
            }
        }
    }
}
