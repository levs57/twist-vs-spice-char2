//! The cycle check of Twist v2: one sum-check over the cycle variables that
//! proves, at the address point `r_k` and cycle point `r_j` from the read
//! check,
//!
//! 1. the value claim `V(r_o, r_k, r_j) = M0(r_o, r_k) + Σ_j LT(j, r_j) ra(r_k, j) Inc(r_o, j)`;
//! 2. the final-memory claim `Mf(r_o, r_k) = M0(r_o, r_k) + Σ_j ra(r_k, j) Inc(r_o, j)`
//!    at the same `(r_o, r_k)` (issue item 4, as in Jolt's `RamValCheck`);
//! 3. the virtualisation `ra(r_k, r_j) = Σ_j eq(r_j, j) Π_i A_i(r_k^i, j)`,
//!    run directly on the read check's claim with no separate claim
//!    reduction (issue item 6);
//! 4. the mirror test `Σ_j eq(r_j, j) A_i(r_k^i, j) A_i(1 + r_k^i, j) = π(r_k^i)`
//!    for each component `i`, which proves every row of `A_i` one-hot
//!    (issue item 5, Wambsgans' test).
//!
//! The summand is
//! `(LT(j, r_j) + γ) · Π_i A_i · Inc + δ eq(r_j, j) · (Π_i A_i + Σ_i ε_i' A_i A_i')`
//! with `γ`, `δ`, `ε_i` sampled after the read check and `ε_i' = ε_i / δ`.
//! It has degree `d + 2` in each cycle variable. The sum-check ends with
//! claims on `A_i(r_k^i, z)`, `A_i(1 + r_k^i, z)` and `Inc(r_o, z)` at one new
//! cycle point `z`: `2d + 1` Mosaic claims.

use om_field::{K, U256};
use om_sumcheck::{
    SumcheckProof, VerificationError, eq_eval, eq_table, interpolate_eval, sumcheck_init,
    sumcheck_verify,
};
use om_transcript::Transcript;
use rayon::prelude::*;

use super::util::{PAR, SplitEq, SplitLt, checks_enabled, emit_round, node, par_build};

const LABEL: &[u8] = b"ram/cycle";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleProof {
    pub sumcheck: SumcheckProof,
    /// `A_i(r_k^i, z)` for each component.
    pub a: Vec<K>,
    /// `A_i(1 + r_k^i, z)` for each component.
    pub a_mirror: Vec<K>,
    /// `Inc(r_o, z)`.
    pub inc: K,
}

pub(crate) struct Challenges {
    pub gamma: K,
    pub delta: K,
    pub eps: Vec<K>,
}

pub(crate) fn challenges(tr: &mut Transcript, d: usize) -> Challenges {
    let gamma = tr.challenge_k(b"ram/cycle/gamma");
    let delta = tr.challenge_nonzero_k(b"ram/cycle/delta");
    let eps = (0..d)
        .map(|_| tr.challenge_nonzero_k(b"ram/cycle/eps"))
        .collect();
    Challenges { gamma, delta, eps }
}

/// `π(r) = Π_h r_h (1 + r_h)`.
pub(crate) fn mirror_rhs(r: &[K]) -> K {
    r.iter().fold(K::ONE, |acc, &x| acc * x * (K::ONE + x))
}

/// Component `i`'s slice of the address point.
pub(crate) fn chunk_point(r_k: &[K], c: usize, i: usize) -> &[K] {
    &r_k[i * c..(i + 1) * c]
}

pub(crate) fn mirrored(r: &[K]) -> Vec<K> {
    r.iter().map(|&x| K::ONE + x).collect()
}

/// The claimed sum, given `val_minus_m0 = V − M0` and `final_minus_m0 =
/// Mf − M0` at `(r_o, r_k)`.
pub(crate) fn claimed_sum(
    ch: &Challenges,
    val_minus_m0: K,
    final_minus_m0: K,
    ra: K,
    r_k: &[K],
    c: usize,
) -> K {
    let mut s = val_minus_m0 + ch.gamma * final_minus_m0 + ch.delta * ra;
    for (i, &e) in ch.eps.iter().enumerate() {
        s += e * mirror_rhs(chunk_point(r_k, c, i));
    }
    s
}

pub(crate) struct CycleWitness<'a> {
    pub addr: &'a [u32],
    pub inc_e: &'a [K],
    pub c: usize,
    pub d: usize,
}

pub(crate) struct CycleResult {
    pub proof: CycleProof,
    pub z: Vec<K>,
}

/// Per-thread round accumulator: the high-degree part at nodes
/// `t_0, t_2..t_D` (slot `e` holds node `e`; slot 1 is unused) and the
/// eq-weighted mirror part at `t_0..t_3`.
#[derive(Clone)]
struct Acc {
    high: Vec<U256>,
    low: [U256; 4],
}

impl Acc {
    fn new(degree: usize) -> Self {
        Acc {
            high: vec![U256::ZERO; degree + 1],
            low: [U256::ZERO; 4],
        }
    }

    fn merge(mut self, o: Acc) -> Acc {
        for (a, b) in self.high.iter_mut().zip(o.high) {
            *a += b;
        }
        for (a, b) in self.low.iter_mut().zip(o.low) {
            *a += b;
        }
        self
    }

    /// `[g(t_0), g(t_1), …, g(t_D)]`, with `g(t_1) = cur + g(t_0)`.
    fn values(&self, cur: K) -> Vec<K> {
        let low: Vec<K> = self.low.iter().map(|x| x.reduce()).collect();
        let mut ys: Vec<K> = self
            .high
            .iter()
            .enumerate()
            .map(|(e, h)| {
                let l = if e < 4 {
                    low[e]
                } else {
                    interpolate_eval(&low, node(e))
                };
                h.reduce() + l
            })
            .collect();
        ys[1] = cur + ys[0];
        ys
    }
}

/// Row-major tables: row `j` is `[tl, ed, inc, A_1..A_d, Ã_1..Ã_d](j)` with
/// `tl = LT(·, r_j) + γ`, `ed = δ eq(r_j, ·)` and `Ã_i = ε_i' A_i(1 + r_k^i, ·)`.
/// A pair of rows is contiguous, which keeps each round's reads sequential.
pub(crate) struct Rows {
    d: usize,
    width: usize,
    data: Vec<K>,
}

const TL: usize = 0;
const ED: usize = 1;
const INC: usize = 2;
const A0: usize = 3;

impl Rows {
    fn len(&self) -> usize {
        self.data.len() / self.width
    }

    fn row(&self, j: usize) -> &[K] {
        &self.data[j * self.width..(j + 1) * self.width]
    }
}

/// Add one pair's contribution, given the pair's two rows.
#[inline]
fn pair_kernel(x0: &[K], x1: &[K], d: usize, degree: usize, acc: &mut Acc, scratch: &mut [K]) {
    let width = x0.len();
    let (delta, val) = scratch.split_at_mut(width);
    for t in 0..width {
        delta[t] = x0[t] + x1[t];
    }
    let top = degree.max(3);
    for e in 0..=top {
        match e {
            0 => val.copy_from_slice(x0),
            1 => val.copy_from_slice(x1),
            _ => {
                for t in 0..width {
                    val[t] = x0[t] + delta[t].mul_t(e as u128);
                }
            }
        }
        if e != 1 && e <= degree {
            let mut prod = val[A0];
            for i in 1..d {
                prod *= val[A0 + i];
            }
            let lin = val[TL] * val[INC] + val[ED];
            acc.high[e] += lin.clmul(prod);
        }
        if e < 4 {
            let mut mir = U256::ZERO;
            for i in 0..d {
                mir += val[A0 + i].clmul(val[A0 + d + i]);
            }
            acc.low[e] += val[ED].clmul(mir.reduce());
        }
    }
}

const CHUNK: usize = 1 << 9;

/// Round accumulator over all pairs of `rows`.
fn eval_rows(rows: &Rows, degree: usize) -> Acc {
    let (d, w) = (rows.d, rows.width);
    let half = rows.len() / 2;
    let fold = |range: std::ops::Range<usize>| -> Acc {
        let mut acc = Acc::new(degree);
        let mut scratch = vec![K::ZERO; 2 * w];
        for m in range {
            let pair = &rows.data[2 * m * w..(2 * m + 2) * w];
            let (x0, x1) = pair.split_at(w);
            pair_kernel(x0, x1, d, degree, &mut acc, &mut scratch);
        }
        acc
    };
    if half >= PAR {
        (0..half.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| fold(c * CHUNK..((c + 1) * CHUNK).min(half)))
            .reduce(|| Acc::new(degree), Acc::merge)
    } else {
        fold(0..half)
    }
}

#[inline]
fn bind_row(src0: &[K], src1: &[K], r: K, dst: &mut [K]) {
    for ((o, &a), &b) in dst.iter_mut().zip(src0).zip(src1) {
        *o = a + r * (a + b);
    }
}

/// Bind the low variable to `r` and, unless `last`, accumulate the next
/// round over the bound rows in the same pass.
fn bind_eval(rows: &Rows, r: K, degree: usize, last: bool) -> (Rows, Acc) {
    let (d, w) = (rows.d, rows.width);
    let out_len = rows.len() / 2;
    if out_len == 1 {
        let mut out = vec![K::ZERO; w];
        bind_row(rows.row(0), rows.row(1), r, &mut out);
        return (
            Rows {
                d,
                width: w,
                data: out,
            },
            Acc::new(degree),
        );
    }
    // Each unit is one output pair (two output rows from four input rows).
    let units = out_len / 2;
    let chunk = if units >= PAR / 2 { CHUNK } else { units };
    let accs = std::sync::Mutex::new(Vec::new());
    // SAFETY: `out_len` is even here, so every chunk of `chunk * 2 * w`
    // elements is a whole number of output pairs, and each pair writes all
    // `2 * w` of its elements through the zip with `pair` below.
    let data = unsafe {
        par_build(out_len * w, chunk * 2 * w, |start, dst| {
            debug_assert_eq!(dst.len() % (2 * w), 0);
            let p0 = start / (2 * w);
            let mut acc = Acc::new(degree);
            let mut scratch = vec![K::ZERO; 2 * w];
            let mut pair = vec![K::ZERO; 2 * w];
            for (k, out_pair) in dst.chunks_mut(2 * w).enumerate() {
                let p = p0 + k;
                let src = &rows.data[4 * p * w..(4 * p + 4) * w];
                let (o0, o1) = pair.split_at_mut(w);
                bind_row(&src[..w], &src[w..2 * w], r, o0);
                bind_row(&src[2 * w..3 * w], &src[3 * w..], r, o1);
                for (o, &x) in out_pair.iter_mut().zip(pair.iter()) {
                    o.write(x);
                }
                if !last {
                    let (o0, o1) = pair.split_at(w);
                    pair_kernel(o0, o1, d, degree, &mut acc, &mut scratch);
                }
            }
            accs.lock().unwrap().push(acc);
        })
    };
    let acc = accs
        .into_inner()
        .unwrap()
        .into_iter()
        .fold(Acc::new(degree), Acc::merge);
    (Rows { d, width: w, data }, acc)
}

/// Round-0 accumulator over rows that `make_row(j, out)` writes on demand,
/// so the full-size table is never stored.
fn eval_virtual<G: Fn(usize, &mut [K]) + Sync>(
    size: usize,
    d: usize,
    w: usize,
    degree: usize,
    make_row: &G,
) -> Acc {
    let half = size / 2;
    let fold = |range: std::ops::Range<usize>| -> Acc {
        let mut acc = Acc::new(degree);
        let mut scratch = vec![K::ZERO; 2 * w];
        let mut pair = vec![K::ZERO; 2 * w];
        for m in range {
            let (x0, x1) = pair.split_at_mut(w);
            make_row(2 * m, x0);
            make_row(2 * m + 1, x1);
            pair_kernel(x0, x1, d, degree, &mut acc, &mut scratch);
        }
        acc
    };
    (0..half.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| fold(c * CHUNK..((c + 1) * CHUNK).min(half)))
        .reduce(|| Acc::new(degree), Acc::merge)
}

/// Bind round 0 to `r` over rows from `make_row` and accumulate round 1 in
/// the same pass. Needs at least four rows.
fn bind_eval_virtual<G: Fn(usize, &mut [K]) + Sync>(
    size: usize,
    d: usize,
    w: usize,
    r: K,
    degree: usize,
    make_row: &G,
) -> (Rows, Acc) {
    let out_len = size / 2;
    let last = out_len == 1;
    let units = out_len / 2;
    let chunk = if units >= PAR / 2 {
        CHUNK
    } else {
        units.max(1)
    };
    let accs = std::sync::Mutex::new(Vec::new());
    // SAFETY: `out_len` is even here, so every chunk of `chunk * 2 * w`
    // elements is a whole number of output pairs, and each pair writes all
    // `2 * w` of its elements through the zip with `pair` below.
    let data = unsafe {
        par_build(out_len * w, chunk * 2 * w, |start, dst| {
            debug_assert_eq!(dst.len() % (2 * w), 0);
            let p0 = start / (2 * w);
            let mut acc = Acc::new(degree);
            let mut scratch = vec![K::ZERO; 2 * w];
            let mut src = vec![K::ZERO; 4 * w];
            let mut pair = vec![K::ZERO; 2 * w];
            for (k, out_pair) in dst.chunks_mut(2 * w).enumerate() {
                let p = p0 + k;
                for (q, row) in src.chunks_mut(w).enumerate() {
                    make_row(4 * p + q, row);
                }
                let (o0, o1) = pair.split_at_mut(w);
                bind_row(&src[..w], &src[w..2 * w], r, o0);
                bind_row(&src[2 * w..3 * w], &src[3 * w..], r, o1);
                for (o, &x) in out_pair.iter_mut().zip(pair.iter()) {
                    o.write(x);
                }
                if !last {
                    let (o0, o1) = pair.split_at(w);
                    pair_kernel(o0, o1, d, degree, &mut acc, &mut scratch);
                }
            }
            accs.lock().unwrap().push(acc);
        })
    };
    let acc = accs
        .into_inner()
        .unwrap()
        .into_iter()
        .fold(Acc::new(degree), Acc::merge);
    (Rows { d, width: w, data }, acc)
}

/// Column-form tables, used by tests to build adversarial inputs.
#[cfg(test)]
pub(crate) struct Tables {
    pub(crate) tl: Vec<K>,
    pub(crate) ed: Vec<K>,
    pub(crate) inc: Vec<K>,
    pub(crate) a: Vec<Vec<K>>,
    pub(crate) am: Vec<Vec<K>>,
}

#[cfg(test)]
impl Tables {
    pub(crate) fn into_rows(self) -> Rows {
        let d = self.a.len();
        let width = A0 + 2 * d;
        let len = self.inc.len();
        let mut data = Vec::with_capacity(len * width);
        for j in 0..len {
            data.extend([self.tl[j], self.ed[j], self.inc[j]]);
            data.extend(self.a.iter().map(|t| t[j]));
            data.extend(self.am.iter().map(|t| t[j]));
        }
        Rows { d, width, data }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove(
    w: &CycleWitness<'_>,
    r_k: &[K],
    r_j: &[K],
    claim: K,
    ch: &Challenges,
    tr: &mut Transcript,
) -> CycleResult {
    let n = r_j.len();
    let (c, d) = (w.c, w.d);
    let tables_span = om_transcript::profiling::Span::new("twist.cycle.tables");
    let delta_inv = ch.delta.inverse();
    let eps_scaled: Vec<K> = ch.eps.iter().map(|&e| e * delta_inv).collect();
    let mask = (1u32 << c) - 1;
    let eq_a: Vec<Vec<K>> = (0..d).map(|i| eq_table(chunk_point(r_k, c, i))).collect();
    let eq_am: Vec<Vec<K>> = (0..d)
        .map(|i| {
            eq_table(&mirrored(chunk_point(r_k, c, i)))
                .into_iter()
                .map(|x| x * eps_scaled[i])
                .collect()
        })
        .collect();
    let lt = SplitLt::new(r_j);
    let eq = SplitEq::new(r_j);
    let size = 1usize << n;
    let width = A0 + 2 * d;
    let make_row = |j: usize, row: &mut [K]| {
        let ad = w.addr[j];
        row[TL] = lt.at(j) + ch.gamma;
        row[ED] = ch.delta * eq.at(j);
        row[INC] = w.inc_e[j];
        for i in 0..d {
            let u = ((ad >> (i * c)) & mask) as usize;
            row[A0 + i] = eq_a[i][u];
            row[A0 + d + i] = eq_am[i][u];
        }
    };
    drop(tables_span);
    let _span = om_transcript::profiling::Span::new("twist.cycle.rounds");
    if size < 4 {
        let mut data = vec![K::ZERO; size * width];
        for (j, row) in data.chunks_mut(width).enumerate() {
            make_row(j, row);
        }
        return prove_rows(Rows { d, width, data }, &eps_scaled, claim, tr);
    }
    // Round 0 reads rows built on the fly, and its bind writes the first
    // stored (half-size) table.
    let degree = d + 2;
    sumcheck_init(tr, LABEL, claim, n, degree);
    let mut msgs = Vec::with_capacity(n);
    let acc = eval_virtual(size, d, width, degree, &make_row);
    let ys = acc.values(claim);
    let (r, cur) = emit_round(&ys, tr, LABEL, &mut msgs);
    let (rows, acc) = bind_eval_virtual(size, d, width, r, degree, &make_row);
    finish_rounds(rows, acc, vec![r], msgs, cur, &eps_scaled, tr)
}

/// The sum-check on already built rows; `eps_scaled[i]` is the factor
/// folded into `Ã_i`.
pub(crate) fn prove_rows(
    rows: Rows,
    eps_scaled: &[K],
    claim: K,
    tr: &mut Transcript,
) -> CycleResult {
    let n = rows.len().trailing_zeros() as usize;
    let degree = rows.d + 2;
    sumcheck_init(tr, LABEL, claim, n, degree);
    let acc = if n > 0 {
        eval_rows(&rows, degree)
    } else {
        Acc::new(degree)
    };
    finish_rounds(rows, acc, Vec::new(), Vec::new(), claim, eps_scaled, tr)
}

/// The remaining rounds, given the rows bound to `z` so far, the next
/// round's accumulator and the current claim.
fn finish_rounds(
    mut rows: Rows,
    mut acc: Acc,
    mut z: Vec<K>,
    mut msgs: Vec<Vec<K>>,
    mut cur: K,
    eps_scaled: &[K],
    tr: &mut Transcript,
) -> CycleResult {
    let d = rows.d;
    let degree = d + 2;
    let rest = rows.len().trailing_zeros() as usize;
    for i in 0..rest {
        let ys = acc.values(cur);
        let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
        cur = next;
        z.push(r);
        (rows, acc) = bind_eval(&rows, r, degree, i + 1 == rest);
    }
    if checks_enabled() {
        // The last bound row must reproduce the final claim.
        let row = rows.row(0);
        let prod: K = (0..d).fold(K::ONE, |p, i| p * row[A0 + i]);
        let mir: K = (0..d).fold(K::ZERO, |s, i| s + row[A0 + i] * row[A0 + d + i]);
        let value = (row[TL] * row[INC] + row[ED]) * prod + row[ED] * mir;
        assert_eq!(value, cur, "cycle check final value");
    }
    let row = rows.row(0);
    let a_z: Vec<K> = row[A0..A0 + d].to_vec();
    let am_z: Vec<K> = row[A0 + d..A0 + 2 * d]
        .iter()
        .zip(eps_scaled)
        .map(|(&t, &e)| t * e.inverse())
        .collect();
    let inc_z = row[INC];
    let mut evals = a_z.clone();
    evals.extend_from_slice(&am_z);
    evals.push(inc_z);
    tr.absorb_ks(b"ram/cycle/evals", &evals);
    CycleResult {
        proof: CycleProof {
            sumcheck: SumcheckProof { round_msgs: msgs },
            a: a_z,
            a_mirror: am_z,
            inc: inc_z,
        },
        z,
    }
}

pub(crate) fn verify(
    proof: &CycleProof,
    r_j: &[K],
    claim: K,
    ch: &Challenges,
    d: usize,
    tr: &mut Transcript,
) -> Result<Vec<K>, VerificationError> {
    let n = r_j.len();
    if proof.a.len() != d || proof.a_mirror.len() != d {
        return Err(VerificationError::new("cycle check: malformed evaluations"));
    }
    let (z, cur) = sumcheck_verify(claim, n, d + 2, &proof.sumcheck, tr, LABEL)?;
    let prod: K = proof.a.iter().fold(K::ONE, |acc, &x| acc * x);
    let mirror: K = proof
        .a
        .iter()
        .zip(&proof.a_mirror)
        .zip(&ch.eps)
        .fold(K::ZERO, |acc, ((&x, &y), &e)| acc + e * x * y);
    let lt = om_memcheck::twist::lt_eval(&z, r_j);
    let eq = eq_eval(&z, r_j);
    let expected = (lt + ch.gamma) * prod * proof.inc + eq * (ch.delta * prod + mirror);
    if cur != expected {
        return Err(VerificationError::new(
            "cycle check: final evaluation mismatch",
        ));
    }
    let mut evals = proof.a.clone();
    evals.extend_from_slice(&proof.a_mirror);
    evals.push(proof.inc);
    tr.absorb_ks(b"ram/cycle/evals", &evals);
    Ok(z)
}
