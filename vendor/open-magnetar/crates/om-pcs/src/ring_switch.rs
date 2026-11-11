//! Ring switching for the monomial packing basis `β_j = X^j` (spec 04 §4.5).
//! Mirrors `reference/openmagnetar/ring_switch.py`.
//!
//! Reduces `w̃(r) = y` for a bit vector of length `2^ℓ` (`ℓ ≥ 8`) to
//! `P̃(r_s) = v` for the packed vector of length `2^(ℓ-7)`.

use om_field::K;
use om_sumcheck::{
    PackedBits, SumcheckProof, VerificationError, eq_table, product_sum,
    sumcheck_prove_mul2_generated_first, sumcheck_verify,
};
use om_transcript::Transcript;
use rayon::prelude::*;

/// Tables at least this large use rayon (exact elementwise maps).
const PAR_THRESHOLD: usize = 1 << 14;

/// Groups of eight 128-bit rows with at least this many set bits (of 1024)
/// use subset-sum lookups in [`partial_evals_packed`]; sparser groups add
/// bit by bit. Both paths give the same result.
const DENSE_GROUP_ONES: u32 = 160;

/// `κ = 7` (§4.5).
pub const KAPPA: usize = 7;
/// Packing width `128 = 2^κ` (Python `W`).
pub const W: usize = 1 << KAPPA;

/// `RingSwitchProof = (s, sum-check messages)` (§4.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingSwitchProof {
    /// The 128 partial evaluations `s_j`.
    pub s: Vec<K>,
    /// The sum-check proof (`ℓ-7` rounds, degree 2).
    pub sumcheck: SumcheckProof,
}

/// `P[v] = Σ_j X^j · bits[j + 128v]` (§4.5, C1 packing).
///
/// # Panics
/// Panics if `bits.len()` is not a multiple of 128 (Python `ValueError`).
pub fn pack(bits: &[bool]) -> Vec<K> {
    assert!(
        bits.len().is_multiple_of(W),
        "bit vector length must be a multiple of 128"
    );
    if bits.len() >= PAR_THRESHOLD {
        bits.par_chunks(W).map(K::pack).collect()
    } else {
        bits.chunks(W).map(K::pack).collect()
    }
}

/// [`pack`] from packed words: word pair `(lo, hi)` is the packed
/// element `K(hi · 2^64 + lo)`, the same value `K::pack` produces
/// from the corresponding 128 bools.
///
/// # Panics
/// If the packed length is not a multiple of 128.
pub fn pack_packed(bits: &PackedBits) -> Vec<K> {
    assert!(
        bits.len().is_multiple_of(W),
        "bit vector length must be a multiple of 128"
    );
    let words = bits.words();
    let pair_to_k = |pair: &[u64]| {
        let lo = u128::from(pair[0]);
        let hi = pair.get(1).map_or(0, |&h| u128::from(h));
        K::from_u128(lo | (hi << 64))
    };
    if bits.len() >= PAR_THRESHOLD {
        words.par_chunks(2).map(pair_to_k).collect()
    } else {
        words.chunks(2).map(pair_to_k).collect()
    }
}

/// [`partial_evals`] from packed bits: only set bits are visited, and the
/// high eq factor `hi[vh]` is multiplied once per 128-entry block rather than
/// on every `vl`.
///
/// # Panics
/// If the packed length is not a multiple of 128.
pub fn partial_evals_packed(bits: &PackedBits, r_hi: &[K]) -> Vec<K> {
    assert!(
        bits.len().is_multiple_of(W),
        "bit vector length must be a multiple of 128"
    );
    // eq(r_hi, v) = lo[v_lo] * hi[v_hi]: two square-root tables
    // instead of one of 2^len(r_hi) entries.
    let h = r_hi.len() / 2;
    let (lo, hi) = (eq_table(&r_hi[..h]), eq_table(&r_hi[h..]));
    let words = bits.words();
    // Add `e` into `local[b]` for every set bit `b` of row `v`.
    let sparse_row = |local: &mut [u128; W], v: usize, e: u128| {
        for (half, &w) in [words[2 * v], words[2 * v + 1]].iter().enumerate() {
            let mut m = w;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                m &= m - 1;
                local[64 * half + b] ^= e;
            }
        }
    };
    let block = |vh: usize| -> [u128; W] {
        let mut local = [0u128; W];
        let v_base = vh << h;
        let rows = lo.len();
        let mut vl = 0;
        // Eight rows at a time: dense groups use one subset-sum lookup per
        // column (see `om_field::subset`), sparse ones add bit by bit. Both
        // give the same sums.
        while vl + 8 <= rows {
            let v = v_base | vl;
            let group = &words[2 * v..2 * v + 16];
            let ones: u32 = group.iter().map(|w| w.count_ones()).sum();
            if ones >= DENSE_GROUP_ONES {
                let t = om_field::subset::subset_sums8(&lo[vl..vl + 8]);
                for half in 0..2 {
                    let column_rows: [u64; 8] = std::array::from_fn(|k| group[2 * k + half]);
                    for (q, bytes) in om_field::subset::column_bytes(&column_rows)
                        .into_iter()
                        .enumerate()
                    {
                        for j in 0..8 {
                            let s = ((bytes >> (8 * j)) & 0xff) as usize;
                            local[64 * half + 8 * q + j] ^= t[s].value();
                        }
                    }
                }
            } else if ones != 0 {
                for k in 0..8 {
                    sparse_row(&mut local, v + k, lo[vl + k].value());
                }
            }
            vl += 8;
        }
        for vl in vl..rows {
            sparse_row(&mut local, v_base | vl, lo[vl].value());
        }
        let hi_v = hi[vh];
        let mut acc = [0u128; W];
        for i in 0..W {
            if local[i] != 0 {
                acc[i] = (K::from_u128(local[i]) * hi_v).value();
            }
        }
        acc
    };
    let merge = |mut a: [u128; W], b: [u128; W]| {
        for (x, y) in a.iter_mut().zip(b) {
            *x ^= y;
        }
        a
    };
    let acc = if bits.len() >= PAR_THRESHOLD {
        (0..hi.len())
            .into_par_iter()
            .map(block)
            .reduce(|| [0u128; W], merge)
    } else {
        (0..hi.len()).map(block).fold([0u128; W], merge)
    };
    acc.into_iter().map(K::from_u128).collect()
}

/// [`rs_prove`] over packed bits: the same proof, without expanding
/// the bit vector.
///
/// # Panics
/// As [`rs_prove`].
pub fn rs_prove_packed(
    bits: &PackedBits,
    r: &[K],
    y: K,
    tr: &mut Transcript,
) -> (RingSwitchProof, Vec<K>, K) {
    let s = partial_evals_packed_checked(bits, r);
    rs_prove_from_s(bits, r, y, s, tr)
}

/// [`rs_prove_packed`] that also returns the value `y = w~(r)`,
/// computed from the partial evaluations as `sum_j eq(r_lo, j) s_j`
/// (the identity the verifier checks) instead of from a table over
/// all `2^ell` positions. Returns `(proof, r_s, v, y)`.
///
/// # Panics
/// As [`rs_prove_packed`].
pub fn rs_prove_packed_value(
    bits: &PackedBits,
    r: &[K],
    tr: &mut Transcript,
) -> (RingSwitchProof, Vec<K>, K, K) {
    let s = partial_evals_packed_checked(bits, r);
    let y = product_sum(&eq_table(&r[..KAPPA]), &s);
    let (proof, r_s, v) = rs_prove_from_s(bits, r, y, s, tr);
    (proof, r_s, v, y)
}

fn partial_evals_packed_checked(bits: &PackedBits, r: &[K]) -> Vec<K> {
    let ell = r.len();
    assert!(
        ell > KAPPA && bits.len() == 1 << ell,
        "need ell >= 8 and 2^ell bits"
    );
    partial_evals_packed(bits, &r[KAPPA..])
}

fn rs_prove_from_s(
    bits: &PackedBits,
    r: &[K],
    y: K,
    s: Vec<K>,
    tr: &mut Transcript,
) -> (RingSwitchProof, Vec<K>, K) {
    let (target, a) = prepare_from_s(r, y, &s, tr);
    // The first field table is already present as pairs of packed u64 words.
    // Read it directly for round zero and its first fold; allocating a full
    // packed field table would duplicate this immutable input representation.
    let words = bits.words();
    let first = |i| K::from_u128(u128::from(words[2 * i]) | (u128::from(words[2 * i + 1]) << 64));
    let (proof, r_s, finals) =
        sumcheck_prove_mul2_generated_first(target, a.len(), first, a, tr, b"rs/sumcheck");
    (RingSwitchProof { s, sumcheck: proof }, r_s, finals[0])
}

/// Frozen allocation-based RingSwitch driver for proof/transcript regression
/// tests. The production packed driver generates its first table on demand.
#[cfg(test)]
pub(crate) fn rs_prove_packed_value_owned_reference(
    bits: &PackedBits,
    r: &[K],
    tr: &mut Transcript,
) -> (RingSwitchProof, Vec<K>, K, K) {
    let s = partial_evals_packed_checked(bits, r);
    let y = product_sum(&eq_table(&r[..KAPPA]), &s);
    let (target, a) = prepare_from_s(r, y, &s, tr);
    let (sumcheck, r_s, finals) =
        om_sumcheck::sumcheck_prove_mul2(target, vec![pack_packed(bits), a], tr, b"rs/sumcheck");
    (RingSwitchProof { s, sumcheck }, r_s, finals[0], y)
}

fn prepare_from_s(r: &[K], y: K, s: &[K], tr: &mut Transcript) -> (K, Vec<K>) {
    absorb_claim(tr, r, y);
    tr.absorb_ks(b"rs/s", s);
    let rho = tr.challenge_ks(b"rs/rho", KAPPA);
    let target = target_of(s, &rho);
    let a = a_table(&r[KAPPA..], &rho);
    (target, a)
}

/// `s_j = Σ_v eq(r_hi, v) · bits[j + 128v]`.
pub fn partial_evals(bits: &[bool], r_hi: &[K]) -> Vec<K> {
    let eqt = eq_table(r_hi);
    let at = |j: usize| {
        let mut acc = 0u128;
        for (v, e) in eqt.iter().enumerate() {
            if bits[j + W * v] {
                acc ^= e.value();
            }
        }
        K::from_u128(acc)
    };
    if bits.len() >= PAR_THRESHOLD {
        (0..W).into_par_iter().map(at).collect()
    } else {
        (0..W).map(at).collect()
    }
}

/// `c_i = pack_j(bit_i(s_j))` (left coordinates → right coordinates).
pub fn transpose_bits(s: &[K]) -> Vec<K> {
    (0..W)
        .map(|i| {
            let mut v = 0u128;
            for (j, sj) in s.iter().enumerate().take(W) {
                v |= ((sj.value() >> i) & 1) << j;
            }
            K::from_u128(v)
        })
        .collect()
}

/// `a[v] = Σ_i eq(ρ, i) · bit_i(eq(r_hi, v))`.
pub fn a_table(r_hi: &[K], rho: &[K]) -> Vec<K> {
    let er: Vec<u128> = eq_table(rho).iter().map(|e| e.value()).collect();
    let mut base_lut = [[0u128; 256]; 16];
    for (byte_idx, table) in base_lut.iter_mut().enumerate() {
        for b in 1..256usize {
            let bit = b.trailing_zeros() as usize;
            let idx = 8 * byte_idx + bit;
            let val = if idx < er.len() { er[idx] } else { 0 };
            table[b] = table[b & (b - 1)] ^ val;
        }
    }
    let eval_l = |z: u128| -> u128 {
        let bytes = z.to_le_bytes();
        let mut acc = 0u128;
        for (byte_idx, &b) in bytes.iter().enumerate() {
            acc ^= base_lut[byte_idx][b as usize];
        }
        acc
    };

    let n = 1usize << r_hi.len();
    if r_hi.len() >= 14 {
        // Since L(z) = Σ_i eq(ρ, i) bit_i(z) is GF(2)-linear in z, for each
        // fixed y = hi[v_hi] the composite map x ↦ L(x · y) is GF(2)-linear
        // in x = lo[v_lo]. Building a 16-byte LUT per `v_hi` eliminates all
        // 2^len(r_hi) field multiplications.
        let h_lo = (r_hi.len() - 6).clamp(10, 18);
        let (lo, hi) = (eq_table(&r_hi[..h_lo]), eq_table(&r_hi[h_lo..]));
        let lo_len = 1usize << h_lo;
        let mut out = vec![K::ZERO; n];
        out.par_chunks_mut(lo_len)
            .zip(hi.par_iter())
            .for_each(|(chunk, &y)| {
                let mut lut_y = [[0u128; 256]; 16];
                for (byte_idx, table) in lut_y.iter_mut().enumerate() {
                    let mut basis = [0u128; 8];
                    for (bit, b_val) in basis.iter_mut().enumerate() {
                        let k = 8 * byte_idx + bit;
                        let prod = (K::from_u128(1u128 << k) * y).value();
                        *b_val = eval_l(prod);
                    }
                    for b in 1..256usize {
                        let bit = b.trailing_zeros() as usize;
                        table[b] = table[b & (b - 1)] ^ basis[bit];
                    }
                }
                for (dst, &x) in chunk.iter_mut().zip(&lo) {
                    let bytes = x.value().to_le_bytes();
                    let mut acc = 0u128;
                    for (byte_idx, &b) in bytes.iter().enumerate() {
                        acc ^= lut_y[byte_idx][b as usize];
                    }
                    *dst = K::from_u128(acc);
                }
            });
        return out;
    }

    // Small tables: eq(r_hi, v) = lo[v_lo] * hi[v_hi] per entry.
    let h = r_hi.len() / 2;
    let (lo, hi) = (eq_table(&r_hi[..h]), eq_table(&r_hi[h..]));
    let mask = (1usize << h) - 1;
    let at = |v: usize| K::from_u128(eval_l((lo[v & mask] * hi[v >> h]).value()));
    if n >= PAR_THRESHOLD {
        (0..n).into_par_iter().map(at).collect()
    } else {
        (0..n).map(at).collect()
    }
}

fn mul_tensor(t: &[K], x: K, y: K) -> Vec<K> {
    let mut acc = [0u128; W];
    for (i, ti) in t.iter().enumerate() {
        if ti.is_zero() {
            continue;
        }
        let mut xb = (x * K::from_u128(1u128 << i)).value();
        let mut m = 0usize;
        while xb != 0 {
            if xb & 1 == 1 {
                acc[m] ^= ti.value();
            }
            xb >>= 1;
            m += 1;
        }
    }
    acc.iter().map(|&a| K::from_u128(a) * y).collect()
}

/// MLE of `a` at `z` via the tensor recursion of §4.5 step 5:
/// `T = Π_u [(1+r_u)⊗(1+z_u) + r_u⊗z_u]`.
pub fn a_eval(r_hi: &[K], rho: &[K], z: &[K]) -> K {
    let mut t = vec![K::ZERO; W];
    t[0] = K::ONE;
    for (&rk, &zk) in r_hi.iter().zip(z) {
        let t1 = mul_tensor(&t, K::ONE + rk, K::ONE + zk);
        let t2 = mul_tensor(&t, rk, zk);
        t = t1.iter().zip(&t2).map(|(&p, &q)| p + q).collect();
    }
    product_sum(&eq_table(rho), &t)
}

fn absorb_claim(tr: &mut Transcript, r: &[K], y: K) {
    tr.absorb_ks(b"rs/point", r);
    tr.absorb_k(b"rs/value", y);
}

fn target_of(s: &[K], rho: &[K]) -> K {
    let c = transpose_bits(s);
    product_sum(&eq_table(rho), &c)
}

/// Prover. Returns `(proof, r_s, v)` with `v = P̃(r_s)`; the caller then
/// proves it with the PCS on the same transcript.
///
/// # Panics
/// Panics unless `ℓ ≥ 8` and `bits.len() == 2^ℓ` (Python `ValueError`).
pub fn rs_prove(bits: &[bool], r: &[K], y: K, tr: &mut Transcript) -> (RingSwitchProof, Vec<K>, K) {
    // Reuse the split-equality, set-bit prover for the bool convenience API.
    // This avoids its former full equality table and 128 column-wise scans.
    rs_prove_packed(&PackedBits::from_bools(bits), r, y, tr)
}

/// Verifier. Returns `(r_s, v)`: the PCS must now certify `P̃(r_s) = v`.
///
/// # Errors
/// Returns [`VerificationError`] on a too-small point, a malformed `s`, an
/// `s` inconsistent with `y`, a failed sum-check, or `ã(r_s) = 0` (§4.5).
pub fn rs_verify(
    r: &[K],
    y: K,
    proof: &RingSwitchProof,
    tr: &mut Transcript,
) -> Result<(Vec<K>, K), VerificationError> {
    let ell = r.len();
    if ell <= KAPPA {
        return Err(VerificationError::new("ell too small"));
    }
    if proof.s.len() != W {
        return Err(VerificationError::new("s must have 128 entries"));
    }
    absorb_claim(tr, r, y);
    let acc = product_sum(&eq_table(&r[..KAPPA]), &proof.s);
    if acc != y {
        return Err(VerificationError::new(
            "ring switching: s inconsistent with claimed value",
        ));
    }
    tr.absorb_ks(b"rs/s", &proof.s);
    let rho = tr.challenge_ks(b"rs/rho", KAPPA);
    let target = target_of(&proof.s, &rho);
    let (r_s, final_claim) =
        sumcheck_verify(target, ell - KAPPA, 2, &proof.sumcheck, tr, b"rs/sumcheck")?;
    let a_val = a_eval(&r[KAPPA..], &rho, &r_s);
    if a_val.is_zero() {
        return Err(VerificationError::new(
            "ring switching: degenerate a(r_s) = 0",
        ));
    }
    Ok((r_s, final_claim / a_val))
}

#[cfg(test)]
mod generated_table_tests {
    use super::*;

    #[test]
    fn generated_packed_table_preserves_owned_proof_and_transcript() {
        let mut seed = 0x704a_91e8_4b38_c625u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Packed dimensions 1/2 cover the final-fold edge cases; 14 reaches
        // both generated parallel kernels without materializing their input.
        for (n, pattern, point_kind) in [
            (1usize, 0, 0),
            (1, 1, 1),
            (1, 2, 2),
            (2, 3, 2),
            (7, 0, 2),
            (7, 1, 2),
            (7, 2, 1),
            (7, 3, 0),
            (14, 3, 2),
        ] {
            let words = (0..1usize << (n + 1))
                .map(|i| match pattern {
                    0 => 0,
                    1 => u64::MAX,
                    2 => 1u64 << (i % 64),
                    _ => next(),
                })
                .collect();
            let bits = PackedBits::from_words(words, 1 << (n + KAPPA));
            let point: Vec<K> = (0..n + KAPPA)
                .map(|i| match point_kind {
                    0 => K::ZERO,
                    1 => K::from_bit(i % 2 == 0),
                    _ => K::from_u128(u128::from(next()) | (u128::from(next()) << 64)),
                })
                .collect();
            let mut owned_tr = Transcript::new_default(b"rs-generated");
            let mut generated_tr = owned_tr.clone();
            let owned = rs_prove_packed_value_owned_reference(&bits, &point, &mut owned_tr);
            let generated = rs_prove_packed_value(&bits, &point, &mut generated_tr);
            assert_eq!(
                owned, generated,
                "n={n} pattern={pattern} point={point_kind}"
            );
            let expected_after = owned_tr.challenge_k(b"after");
            assert_eq!(expected_after, generated_tr.challenge_k(b"after"));

            let mut explicit_tr = Transcript::new_default(b"rs-generated");
            let explicit = rs_prove_packed(&bits, &point, owned.3, &mut explicit_tr);
            assert_eq!(explicit, (owned.0, owned.1, owned.2));
            assert_eq!(expected_after, explicit_tr.challenge_k(b"after"));
        }
    }
}
