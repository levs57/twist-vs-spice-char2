//! Word-level encoding shared by Twist v2 and typed Spice.
//!
//! A word `x` of up to 64 bits enters the field as `enc(x) = Σ_o x_o X^o`
//! (`K::from_u128`). `enc` is F2-linear and injective. For a bit table
//! `B(o, j)` holding bit `o` of word `x_j`, with the `2^w` word bits as the
//! low coordinates, the fixed point `p_w` of [`encoding_point`] satisfies
//! `eq(p_w, o) = X^o / φ_w`, so `B̃(p_w ‖ r) = W[x](r) / φ_w`, where
//! `W[x](r) = Σ_j eq(r, j) enc(x_j)` is the word MLE of the column.

use std::sync::OnceLock;

use om_field::{K, U256};
use rayon::prelude::*;

use crate::twist2::util::{PAR, SplitEq};

/// `enc(x) = Σ_o x_o X^o`.
#[inline]
pub fn enc(x: u64) -> K {
    K::from_u128(x as u128)
}

/// `(p_w, φ_w)` for words of `2^log_bits` bits: `p_i = X^{2^i} / (1 + X^{2^i})`
/// and `φ_w = Π_i (1 + X^{2^i})`, so `eq(p_w, o) · φ_w = X^o` for every
/// `o < 2^log_bits`. Every `1 + X^{2^i}` is nonzero because `|K^*|` is odd.
pub fn encoding_point(log_bits: usize) -> (Vec<K>, K) {
    static ENCODINGS: OnceLock<Vec<(Vec<K>, K)>> = OnceLock::new();
    ENCODINGS.get_or_init(|| {
        (0..=6)
            .map(|log_bits| {
                let mut power = K::X;
                let mut factor = K::ONE;
                let mut p = Vec::new();
                for _ in 0..log_bits {
                    let sum = K::ONE + power;
                    factor *= sum;
                    p.push(power / sum);
                    power = power.square();
                }
                (p, factor)
            })
            .collect()
    })[log_bits]
        .clone()
}

/// The bit-table claim that a word MLE equals `value` at `point`: the
/// evaluation point `p_w ‖ point` and the bit-table value `value / φ_w`.
pub fn claim_at(point: &[K], value: K, log_bits: usize) -> (Vec<K>, K) {
    let (mut p, factor) = encoding_point(log_bits);
    p.extend_from_slice(point);
    (p, value / factor)
}

/// `W[words](r) = Σ_k eq(r, k) enc(words[k])`, with `r[0]` the low bit of `k`.
pub fn mle(words: &[u64], r: &[K]) -> K {
    assert_eq!(words.len(), 1usize << r.len(), "word::mle: length");
    let se = SplitEq::new(r);
    let block = |hi: usize| -> K {
        let base = hi << se.h;
        let mut acc = U256::ZERO;
        for (lo, &e) in se.lo.iter().enumerate() {
            acc += e.clmul(enc(words[base | lo]));
        }
        se.hi[hi] * acc.reduce()
    };
    if words.len() >= PAR {
        (0..se.hi.len())
            .into_par_iter()
            .map(block)
            .reduce(|| K::ZERO, |a, b| a + b)
    } else {
        (0..se.hi.len()).map(block).fold(K::ZERO, |a, b| a + b)
    }
}

/// The multilinear extension of `k ↦ enc(k)` on `{0,1}^len(r)`:
/// `Σ_b r[b] X^b`, with `r[b]` the coordinate of bit `b` of `k`.
pub fn enc_mle(r: &[K]) -> K {
    assert!(r.len() <= 128, "word::enc_mle: at most 128 coordinates");
    r.iter()
        .enumerate()
        .fold(K::ZERO, |s, (b, &x)| s + K::from_u128(1u128 << b) * x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use om_sumcheck::eq_table;

    fn rand_k(s: &mut u64) -> K {
        let mut next = || {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        K::from_u128(((next() as u128) << 64) | next() as u128)
    }

    #[test]
    fn word_point_encodes_words() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for log_bits in 0..=6 {
            let (p, phi) = encoding_point(log_bits);
            let eq = eq_table(&p);
            for (o, &e) in eq.iter().enumerate() {
                assert_eq!(e * phi, K::from_u128(1u128 << o), "eq(p, {o})");
            }
            // A random bit table opened at (p ‖ r) equals W[x](r) / φ.
            let n = 3;
            let width = 1usize << log_bits;
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let words: Vec<u64> = (0..1 << n)
                .map(|_| rand_k(&mut s).value() as u64 & mask)
                .collect();
            let r: Vec<K> = (0..n).map(|_| rand_k(&mut s)).collect();
            let (point, value) = claim_at(&r, mle(&words, &r), log_bits);
            let full = eq_table(&point);
            let table: K = (0..width << n)
                .filter(|&i| (words[i >> log_bits] >> (i & (width - 1))) & 1 == 1)
                .map(|i| full[i])
                .fold(K::ZERO, |a, b| a + b);
            assert_eq!(table, value, "log_bits {log_bits}");
        }
    }

    #[test]
    fn enc_mle_matches_bruteforce() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for a in [1usize, 4, 8, 12] {
            let r: Vec<K> = (0..a).map(|_| rand_k(&mut s)).collect();
            let eq = eq_table(&r);
            let brute = eq
                .iter()
                .enumerate()
                .fold(K::ZERO, |acc, (k, &e)| acc + e * enc(k as u64));
            assert_eq!(brute, enc_mle(&r), "a = {a}");
        }
    }

    #[test]
    fn mle_matches_bruteforce() {
        let mut s = 7u64;
        for n in [1usize, 5, 13] {
            let words: Vec<u64> = (0..1 << n).map(|_| rand_k(&mut s).value() as u64).collect();
            let r: Vec<K> = (0..n).map(|_| rand_k(&mut s)).collect();
            let eq = eq_table(&r);
            let brute = words
                .iter()
                .zip(&eq)
                .fold(K::ZERO, |acc, (&w, &e)| acc + e * enc(w));
            assert_eq!(brute, mle(&words, &r));
        }
    }
}
