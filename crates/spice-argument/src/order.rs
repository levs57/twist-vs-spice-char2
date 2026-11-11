//! Committed predecessor ordering, using OM's subtraction carry gadget.
//!
//! The gate is specialized from open-magnetar revision
//! febfc0e66f5809e034dd1afac78202bd4882de27,
//! `crates/om-circuit/src/circuit.rs:622-640` and spec 10 sections 10.3-10.4:
//! `(p_b+c_b)(1+i_b+c_b)=c_b+c_{b+1}`, with c_0=1 and c_w=0.
//! It uses the existing statically dispatched GF128 Eq-factor engine, including
//! split equality, delayed reductions, packed folds and fused next messages.
//!
//! The caller commits all predecessor and internal-carry bitplanes BEFORE this
//! protocol, then authenticates every returned evaluation with that commitment.
//! This module alone does not provide polynomial commitment authentication.

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_memcheck::eq_factor::{self, Cofactor, Endpoint, scale_low_weights};
use om_sumcheck::{SumcheckProof, eq_eval, sumcheck_verify};
use om_transcript::Transcript;
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderError(pub String);
impl core::fmt::Display for OrderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for OrderError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderProof {
    pub sc: SumcheckProof,
    pub previous: Vec<K>,
    pub carries: Vec<K>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderClaims {
    pub point: Vec<K>,
    pub previous: Vec<K>,
    pub carries: Vec<K>,
}

/// Width of both tagged timestamps and committed predecessor words.
pub fn width(n: usize, a: usize) -> Result<usize, OrderError> {
    if n > 31 || a > 31 {
        return Err(OrderError("order: n and a must be at most 31".into()));
    }
    Ok(n.max(a) + 1)
}

/// Bit k-1 of each returned word is c_k, for 1 <= k < w. Endpoints are
/// constants: c_0=1, c_w=0. No difference or final-carry column is committed.
pub fn build_carries(previous: &[u32], n: usize, a: usize) -> Result<Vec<u32>, OrderError> {
    let w = width(n, a)?;
    if previous.len() != 1usize << n {
        return Err(OrderError("order: predecessor length".into()));
    }
    let base = 1u64 << (w - 1);
    let mask = (1u64 << w) - 1;
    previous
        .par_iter()
        .enumerate()
        .map(|(j, &p)| carry_word(p, base + j as u64, mask))
        .collect()
}

#[inline]
fn carry_word(previous: u32, time: u64, mask: u64) -> Result<u32, OrderError> {
    if previous as u64 >= time {
        return Err(OrderError(
            "order: predecessor must precede its access".into(),
        ));
    }
    // OM's s = A xor B' xor Cin, rearranged using a native word sum.
    // This also matches Flock's add_carry_parts witness-generation identity.
    let bp = time ^ mask;
    let sum = previous as u64 + bp + 1;
    let cin = (sum & mask) ^ previous as u64 ^ bp;
    Ok((cin >> 1) as u32)
}

fn label(base: &[u8], suffix: &[u8]) -> Vec<u8> {
    [base, suffix].concat()
}

fn setup(
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(Vec<K>, Vec<K>), OrderError> {
    let w = width(n, a)?;
    tr.absorb_bytes(
        &label(base, b"/protocol"),
        b"spice/order/ripple-majority/v1",
    );
    let dimensions: Vec<u8> = [n, a, w]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(&label(base, b"/dimensions"), &dimensions);
    let point = tr.challenge_ks(&label(base, b"/equality"), n);
    let weights = tr.challenge_ks(&label(base, b"/gate-weights"), w);
    Ok((point, weights))
}

/// The MLE of bit b of timestamp B+j, where B=2^max(n,a), at a cycle point.
fn time_bit(point: &[K], w: usize, b: usize) -> K {
    if b + 1 == w {
        K::ONE
    } else {
        point.get(b).copied().unwrap_or(K::ZERO)
    }
}

fn carry_at(carries: &[K], w: usize, b: usize) -> K {
    if b == 0 {
        K::ONE
    } else if b == w {
        K::ZERO
    } else {
        carries[b - 1]
    }
}

struct Gates<'a> {
    weights: &'a [K],
    validated_zero: bool,
}
impl Gates<'_> {
    #[inline(always)]
    fn accumulate_mode<const BOTH: bool, const AT_ONE: bool>(
        &self,
        weights: &[K],
        left: &[K],
        right: &[K],
        out: &mut [U256],
    ) {
        for (b, &weight) in weights[..self.weights.len()].iter().enumerate() {
            let (l0, l1, r0, r1) = (left[2 * b], right[2 * b], left[2 * b + 1], right[2 * b + 1]);
            if BOTH {
                let scaled = mul_pairs([weight; 2], [l0, l1]);
                let endpoints = clmul_pairs(scaled, [r0, r1]);
                out[0] += endpoints[0];
                out[1] += endpoints[1];
                out[2] += (weight * (l0 + l1)).clmul(r0 + r1);
            } else {
                let (l, r) = if AT_ONE { (l1, r1) } else { (l0, r0) };
                let scaled = mul_pairs([weight; 2], [l, l0 + l1]);
                let products = clmul_pairs(scaled, [r, r0 + r1]);
                out[usize::from(AT_ONE)] += products[0];
                out[2] += products[1];
            }
        }
        // D is a linear column, so it has no quadratic high coefficient.
        let i = 2 * self.weights.len();
        let eq = weights[self.weights.len()];
        if BOTH {
            let terms = clmul_pairs([eq; 2], [left[i], right[i]]);
            out[0] += terms[0];
            out[1] += terms[1];
        } else {
            out[usize::from(AT_ONE)] += eq.clmul(if AT_ONE { right[i] } else { left[i] });
        }
    }
}
impl Cofactor for Gates<'_> {
    const DEGREE: usize = 2;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        2 * self.weights.len() + 1
    }
    fn value(&self, row: &[K]) -> K {
        self.weights
            .iter()
            .enumerate()
            .fold(
                U256::from_k(row[2 * self.weights.len()]),
                |sum, (b, &weight)| sum + weight.clmul(row[2 * b] * row[2 * b + 1]),
            )
            .reduce()
    }
    fn initial_endpoints_zero(&self) -> bool {
        self.validated_zero
    }
    fn prepare(&self, low_eq: &[K]) -> Self::Prepared {
        let mut weights = self.weights.to_vec();
        weights.push(K::ONE);
        scale_low_weights(&weights, low_eq)
    }
    #[inline]
    fn accumulate(
        &self,
        prepared: &Self::Prepared,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    ) {
        let count = self.weights.len() + 1;
        let weights = &prepared[low * count..(low + 1) * count];
        match endpoint {
            Endpoint::Both => self.accumulate_mode::<true, false>(weights, left, right, out),
            Endpoint::Zero => self.accumulate_mode::<false, false>(weights, left, right, out),
            Endpoint::One => self.accumulate_mode::<false, true>(weights, left, right, out),
            Endpoint::Neither => {
                let mut b = 0;
                while b + 1 < self.weights.len() {
                    let slopes = [
                        left[2 * b] + right[2 * b],
                        left[2 * b + 2] + right[2 * b + 2],
                    ];
                    let scaled = mul_pairs([weights[b], weights[b + 1]], slopes);
                    let products = clmul_pairs(
                        scaled,
                        [
                            left[2 * b + 1] + right[2 * b + 1],
                            left[2 * b + 3] + right[2 * b + 3],
                        ],
                    );
                    out[2] += products[0] + products[1];
                    b += 2;
                }
                if b < self.weights.len() {
                    out[2] += (weights[b] * (left[2 * b] + right[2 * b]))
                        .clmul(left[2 * b + 1] + right[2 * b + 1]);
                }
            }
        }
    }
}

/// Fill the subtraction's factor rows. D=Σ_b λ_b(c_b+c_{b+1}) is formed
/// from packed carry bits using masked XORs, without per-row field multiplies.
fn source_row(previous: u32, carry: u32, time: u64, weights: &[K], out: &mut [K]) {
    let w = weights.len();
    let mut d = weights[0];
    for b in 0..w {
        let c = if b == 0 {
            1
        } else {
            ((carry >> (b - 1)) & 1) as u64
        };
        out[2 * b] = K::from_u128((((previous as u64 >> b) & 1) ^ c) as u128);
        out[2 * b + 1] = K::from_u128((1 ^ ((time >> b) & 1) ^ c) as u128);
        if b > 0 {
            let mask = 0u128.wrapping_sub(c as u128);
            d += K::from_u128((weights[b - 1] + weights[b]).value() & mask);
        }
    }
    out[2 * w] = d;
}

fn absorb_claims(tr: &mut Transcript, base: &[u8], previous: &[K], carries: &[K]) {
    tr.absorb_ks(&label(base, b"/previous"), previous);
    tr.absorb_ks(&label(base, b"/carries"), carries);
}

/// Prove all w gates in one degree-three, Eq-weighted sumcheck over cycles.
/// Native-word validation proves every Boolean-row gate zero, allowing the
/// existing structural-zero shortcut to omit both initial endpoint sums.
pub fn prove(
    previous: &[u32],
    carries: &[u32],
    n: usize,
    a: usize,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<(OrderProof, OrderClaims), OrderError> {
    let w = width(n, a)?;
    if previous.len() != 1usize << n || carries.len() != previous.len() {
        return Err(OrderError(
            "order: malformed predecessor/carry words".into(),
        ));
    }
    let timestamp_base = 1u64 << (w - 1);
    let mask = (1u64 << w) - 1;
    for (j, (&p, &c)) in previous.iter().zip(carries).enumerate() {
        if c != carry_word(p, timestamp_base + j as u64, mask)? {
            return Err(OrderError("order: noncanonical carry witness".into()));
        }
    }
    let (rho, weights) = setup(n, a, tr, base)?;
    let row = |j: usize, out: &mut [K]| {
        source_row(
            previous[j],
            carries[j],
            timestamp_base + j as u64,
            &weights,
            out,
        )
    };
    let (initial, sc, point, values) = eq_factor::prove(
        &rho,
        &Gates {
            weights: &weights,
            validated_zero: true,
        },
        row,
        tr,
        &label(base, b"/sc"),
    );
    if initial != K::ZERO {
        return Err(OrderError(
            "order: carry equations are not satisfied".into(),
        ));
    }
    if values[1] != time_bit(&point, w, 0) {
        return Err(OrderError("order: initial carry is not one".into()));
    }
    let carry_values: Vec<K> = (1..w)
        .map(|b| values[2 * b + 1] + K::ONE + time_bit(&point, w, b))
        .collect();
    let previous_values: Vec<K> = (0..w)
        .map(|b| values[2 * b] + carry_at(&carry_values, w, b))
        .collect();
    let d = (0..w)
        .fold(U256::ZERO, |sum, b| {
            sum + weights[b]
                .clmul(carry_at(&carry_values, w, b) + carry_at(&carry_values, w, b + 1))
        })
        .reduce();
    if values[2 * w] != d {
        return Err(OrderError("order: linear carry binding".into()));
    }
    absorb_claims(tr, base, &previous_values, &carry_values);
    let claims = OrderClaims {
        point,
        previous: previous_values.clone(),
        carries: carry_values.clone(),
    };
    Ok((
        OrderProof {
            sc,
            previous: previous_values,
            carries: carry_values,
        },
        claims,
    ))
}

pub fn verify(
    n: usize,
    a: usize,
    proof: &OrderProof,
    tr: &mut Transcript,
    base: &[u8],
) -> Result<OrderClaims, OrderError> {
    let w = width(n, a)?;
    if proof.previous.len() != w
        || proof.carries.len() != w - 1
        || proof.sc.round_msgs.len() != n
        || proof.sc.round_msgs.iter().any(|m| m.len() != 3)
    {
        return Err(OrderError("order: malformed proof".into()));
    }
    let (rho, weights) = setup(n, a, tr, base)?;
    let (point, expected) = sumcheck_verify(K::ZERO, n, 3, &proof.sc, tr, &label(base, b"/sc"))
        .map_err(|e| OrderError(e.to_string()))?;
    let cofactor = (0..w)
        .fold(U256::ZERO, |sum, b| {
            let c = carry_at(&proof.carries, w, b);
            let next = carry_at(&proof.carries, w, b + 1);
            let gate = (proof.previous[b] + c) * (K::ONE + time_bit(&point, w, b) + c) + c + next;
            sum + weights[b].clmul(gate)
        })
        .reduce();
    if eq_eval(&rho, &point) * cofactor != expected {
        return Err(OrderError("order: terminal carry equation".into()));
    }
    absorb_claims(tr, base, &proof.previous, &proof.carries);
    Ok(OrderClaims {
        point,
        previous: proof.previous.clone(),
        carries: proof.carries.clone(),
    })
}

#[cfg(test)]
#[path = "order_tests.rs"]
mod tests;
