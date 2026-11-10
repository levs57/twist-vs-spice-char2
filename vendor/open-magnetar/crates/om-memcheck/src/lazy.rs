//! Memory-lean prover helpers (spec 07 §7.10, informative): a sum-check
//! whose tables are never materialised at full length, split `eq`
//! tables, and word evaluation by byte lookup. None of this changes a
//! transcript byte: every round message is the same field element the
//! dense prover computes (field arithmetic is exact, so the order of
//! the sums does not matter).

use om_field::K;
#[cfg(test)]
use om_sumcheck::bind_low_and_round_evals_into;
use om_sumcheck::{
    SumcheckProof, bind_low_and_round_evals_mul2_into, eq_table, interpolate_eval, sumcheck_init,
};
use om_transcript::Transcript;
use rayon::prelude::*;

use crate::onehot::PAR_THRESHOLD;
use crate::sublabel;
use crate::sumcheck_kernels::{self, Relation};

/// `eq(r, j)` for all `j < 2^len(r)` from two tables of size about
/// `2^(len/2)`: one multiplication per lookup instead of a `2^len(r)`
/// table.
pub(crate) struct SplitEq {
    pub(crate) lo: Vec<K>,
    pub(crate) hi: Vec<K>,
    pub(crate) h: usize,
}

impl SplitEq {
    pub(crate) fn new(r: &[K]) -> Self {
        let h = r.len() / 2;
        SplitEq {
            lo: eq_table(&r[..h]),
            hi: eq_table(&r[h..]),
            h,
        }
    }

    #[inline]
    pub(crate) fn at(&self, j: usize) -> K {
        self.lo[j & ((1usize << self.h) - 1)] * self.hi[j >> self.h]
    }
}

/// `LT~(bits(j), y)` for all `j < 2^len(y)` from split tables, using
/// `LT(x, y) = LT(x_hi, y_hi) + eq(x_hi, y_hi) · LT(x_lo, y_lo)`.
pub(crate) struct SplitLt {
    lt_lo: Vec<K>,
    lt_hi: Vec<K>,
    eq_hi: Vec<K>,
    h: usize,
}

impl SplitLt {
    pub(crate) fn new(y: &[K]) -> Self {
        let h = y.len() / 2;
        SplitLt {
            lt_lo: crate::twist::lt_table(&y[..h]),
            lt_hi: crate::twist::lt_table(&y[h..]),
            eq_hi: eq_table(&y[h..]),
            h,
        }
    }

    #[inline]
    pub(crate) fn at(&self, j: usize) -> K {
        let (lo, hi) = (j & ((1usize << self.h) - 1), j >> self.h);
        self.lt_hi[hi] + self.eq_hi[hi] * self.lt_lo[lo]
    }
}

/// `w ↦ Σ_{o < 2^ell} eq(r_o, o) · bit_o(w)` by byte-table lookup: the
/// MLE in the `o` variables of one word of a word table.
pub(crate) struct WordEval {
    tabs: Vec<[K; 256]>,
    mask: u64,
}

impl WordEval {
    /// # Panics
    /// Panics if `len(r_o) > 6` (words are `u64`).
    pub(crate) fn new(r_o: &[K]) -> Self {
        assert!(r_o.len() <= 6, "WordEval: ell must be <= 6");
        let eo = eq_table(r_o);
        let l = eo.len();
        let nbytes = l.div_ceil(8);
        let mut tabs = vec![[K::ZERO; 256]; nbytes];
        for (b, tab) in tabs.iter_mut().enumerate() {
            for v in 1..256usize {
                let low = v & (v - 1);
                let bit = v.trailing_zeros() as usize;
                let o = 8 * b + bit;
                tab[v] = tab[low] + if o < l { eo[o] } else { K::ZERO };
            }
        }
        let mask = if l >= 64 { u64::MAX } else { (1u64 << l) - 1 };
        WordEval { tabs, mask }
    }

    #[inline]
    pub(crate) fn eval(&self, w: u64) -> K {
        let w = w & self.mask;
        let mut acc = K::ZERO;
        for (b, tab) in self.tabs.iter().enumerate() {
            acc += tab[((w >> (8 * b)) & 0xff) as usize];
        }
        acc
    }
}

/// `Σ_j eq(r_c, j) · f(j)` over `j < 2^len(r_c)`.
pub(crate) fn eq_weighted_sum(r_c: &[K], f: &(dyn Fn(usize) -> K + Sync)) -> K {
    use om_field::U256;
    let t = 1usize << r_c.len();
    let se = SplitEq::new(r_c);
    let block = |hi_idx: usize| -> K {
        let base = hi_idx << se.h;
        let mut u = U256::ZERO;
        for (lo_idx, &e_lo) in se.lo.iter().enumerate() {
            u += e_lo.clmul(f(base | lo_idx));
        }
        se.hi[hi_idx] * u.reduce()
    };
    if t >= PAR_THRESHOLD {
        (0..se.hi.len())
            .into_par_iter()
            .map(block)
            .reduce(|| K::ZERO, |a, b| a + b)
    } else {
        (0..se.hi.len()).fold(K::ZERO, |acc, hi_idx| acc + block(hi_idx))
    }
}

/// A row function: `fill(j, out)` writes the `width` table entries of
/// row (cycle) `j` into `out`.
pub(crate) type RowFn<'a> = dyn Fn(usize, &mut [K]) + Sync + 'a;

/// Generic reference interface for checking the specialized row kernels.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sumcheck_prove_rows(
    n: usize,
    width: usize,
    row: &RowFn<'_>,
    combine: &(dyn Fn(&[K]) -> K + Sync),
    degree: usize,
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_rows(
        n,
        width,
        row,
        Relation::Generic { combine, degree },
        transcript,
        label,
    )
}

/// Explicit product-of-two relation; arbitrary closures never select this path.
pub(crate) fn sumcheck_prove_rows_mul2(
    n: usize,
    row: &RowFn<'_>,
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_rows(n, 2, row, Relation::Mul2, transcript, label)
}

#[cfg(test)]
pub(crate) fn sumcheck_prove_rows_onehot(
    n: usize,
    row: &RowFn<'_>,
    weights: &[K],
    c: K,
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_rows(
        n,
        1 + 2 * weights.len(),
        row,
        Relation::OneHot { weights, c },
        transcript,
        label,
    )
}

pub(crate) fn sumcheck_prove_rows_products(
    n: usize,
    row: &RowFn<'_>,
    groups: usize,
    components: usize,
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    assert!(components > 0);
    prove_rows(
        n,
        groups * (components + 1),
        row,
        Relation::Products { groups, components },
        transcript,
        label,
    )
}

#[cfg(test)]
pub(crate) fn sumcheck_prove_rows_weighted_products(
    n: usize,
    row: &RowFn<'_>,
    weights: &[K],
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    prove_rows(
        n,
        1 + 2 * weights.len(),
        row,
        Relation::WeightedProducts { weights },
        transcript,
        label,
    )
}

pub(crate) fn prove_rows(
    n: usize,
    width: usize,
    row: &RowFn<'_>,
    relation: Relation<'_>,
    transcript: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    let degree = relation.degree();
    assert!(degree >= 1, "degree must be >= 1");
    if n == 0 {
        let mut values = vec![K::ZERO; width];
        row(0, &mut values);
        let claim = relation.value(&values);
        sumcheck_init(transcript, label, claim, 0, degree);
        return (
            claim,
            SumcheckProof {
                round_msgs: Vec::new(),
            },
            Vec::new(),
            values,
        );
    }
    assert!(width > 0);
    let msg_label = sublabel(label, b"/msg");
    let chal_label = sublabel(label, b"/chal");
    let mut msgs = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let mut evaluations = sumcheck_kernels::first_round(row, width, n, relation);
    let claim = evaluations[0] + evaluations[1];
    let mut current = claim;
    sumcheck_init(transcript, label, claim, n, degree);
    let mut tables = Vec::new();
    let mut scratch = Vec::new();
    for round in 0..n {
        assert_eq!(
            evaluations[0] + evaluations[1],
            current,
            "claim inconsistent with tables (g(0)+g(1) != claim)"
        );
        let mut message = Vec::with_capacity(degree);
        message.push(evaluations[0]);
        message.extend_from_slice(&evaluations[2..]);
        transcript.absorb_ks(&msg_label, &message);
        let challenge = transcript.challenge_k(&chal_label);
        msgs.push(message);
        challenges.push(challenge);
        current = interpolate_eval(&evaluations, challenge);
        if round == 0 {
            // Original rows are regenerated once to retain the half-sized peak
            // tables, but the next message is emitted during this same fill.
            (tables, evaluations) = sumcheck_kernels::materialize_and_round(
                row, width, n, challenge, relation, current,
            );
        } else {
            evaluations = match relation {
                #[cfg(test)]
                Relation::Generic { combine, degree } => bind_low_and_round_evals_into(
                    &tables,
                    &mut scratch,
                    challenge,
                    combine,
                    degree,
                    Some(current),
                ),
                Relation::Mul2 => bind_low_and_round_evals_mul2_into(
                    &tables,
                    &mut scratch,
                    challenge,
                    Some(current),
                ),
                _ => sumcheck_kernels::bind_and_round_into(
                    &tables,
                    &mut scratch,
                    challenge,
                    relation,
                    current,
                ),
            };
            std::mem::swap(&mut tables, &mut scratch);
        }
    }
    let mles = tables.iter().map(|table| table[0]).collect();
    (claim, SumcheckProof { round_msgs: msgs }, challenges, mles)
}

/// `f_t~(r_c) = Σ_j eq(r_c, j) · bit_t(addr(j))` for `t < a`, over
/// `j < 2^len(r_c)`.
pub(crate) fn addr_bit_evals(
    a: usize,
    r_c: &[K],
    addr: &(dyn Fn(usize) -> usize + Sync),
) -> Vec<K> {
    let t = 1usize << r_c.len();
    let se = SplitEq::new(r_c);
    let mask = if a >= usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << a) - 1
    };
    let block = |hi_idx: usize| -> Vec<K> {
        let mut local = vec![0u128; a];
        let base = hi_idx << se.h;
        for (lo_idx, &e_lo) in se.lo.iter().enumerate() {
            let mut ad = addr(base | lo_idx) & mask;
            if ad == 0 {
                continue;
            }
            let ev = e_lo.value();
            while ad != 0 {
                let b = ad.trailing_zeros() as usize;
                local[b] ^= ev;
                ad &= ad - 1;
            }
        }
        let hi_v = se.hi[hi_idx];
        local
            .into_iter()
            .map(|v| {
                if v == 0 {
                    K::ZERO
                } else {
                    K::from_u128(v) * hi_v
                }
            })
            .collect()
    };
    if t >= PAR_THRESHOLD {
        (0..se.hi.len()).into_par_iter().map(block).reduce(
            || vec![K::ZERO; a],
            |mut x, y| {
                for (p, q) in x.iter_mut().zip(&y) {
                    *p += *q;
                }
                x
            },
        )
    } else {
        (0..se.hi.len()).fold(vec![K::ZERO; a], |mut acc, hi_idx| {
            for (p, q) in acc.iter_mut().zip(block(hi_idx)) {
                *p += q;
            }
            acc
        })
    }
}
