//! The prover of the cycle check (protocol in `cycle.rs`). It sends the same
//! messages as the reference prover there, which stores
//! `LT(·, r_j) + γ` and `δ eq(r_j, ·)` as columns, but it does not store
//! `δ eq(r_j, ·)`.
//!
//! **Split eq.** In round `i`, with `z_{<i}` bound and `X` the current
//! variable, `eq(r_j, (z_{<i}, X, m)) = E_i · eq(r_i, X) · eq(r_{>i}, m)`
//! with `E_i = eq(r_{<i}, z_{<i})`. The prover accumulates the eq-weighted
//! part `Σ_m eq(r_{>i}, m) (δ Π_i A_i + Σ_i ε_i' A_i Ã_i)(X, m)`, which has
//! degree `d` in `X`, and multiplies by the linear factor `E_i eq(r_i, X)`
//! once per round. The weight `eq(r_{>i}, m)` is split as
//! `eq_lo[m_lo] · eq_hi[m_hi]`: the inner sum over `m_lo` is accumulated
//! unreduced and the outer factor is applied once per block.

use om_field::{K, U256};
use om_sumcheck::{interpolate_eval, sumcheck_init};
use om_transcript::Transcript;
use rayon::prelude::*;

use super::cycle::{
    Challenges, CycleResult, CycleWitness, LABEL, component_tables, conclude, eps_scaled,
};
use super::util::{PAR, SplitEq, SplitLt, checks_enabled, emit_round, node, par_build};

/// Row layout: `[tl, inc, A_1..A_d, Ã_1..Ã_d]` with `tl = LT(·, r_j) + γ`
/// and `Ã_i = ε_i' A_i(1 + r_k^i, ·)`.
const TL: usize = 0;
const INC: usize = 1;
const A0: usize = 2;

/// Pairs per parallel task (rounded up to whole weight blocks).
const CHUNK: usize = 1 << 9;

/// Round sums, also used for one block of pairs. `u[e]` holds
/// `Σ tl·inc·ΠA` at node `e` (slot 1 unused), `pi[e]` the eq-weighted
/// `ΠA` at nodes `0..=d` and `mir[e]` the eq-weighted mirror sum at nodes
/// `0..=2`.
#[derive(Clone)]
struct Acc {
    u: Vec<U256>,
    pi: Vec<U256>,
    mir: [U256; 3],
}

impl Acc {
    fn new(d: usize) -> Self {
        Acc {
            u: vec![U256::ZERO; d + 3],
            pi: vec![U256::ZERO; d + 1],
            mir: [U256::ZERO; 3],
        }
    }

    fn merge(mut self, o: Acc) -> Acc {
        for (a, b) in self.u.iter_mut().zip(o.u) {
            *a += b;
        }
        for (a, b) in self.pi.iter_mut().zip(o.pi) {
            *a += b;
        }
        for (a, b) in self.mir.iter_mut().zip(o.mir) {
            *a += b;
        }
        self
    }

    /// Add block `b`, whose eq weights lack the factor `hi`, and clear it.
    fn flush(&mut self, b: &mut Acc, hi: K) {
        for (a, x) in self.u.iter_mut().zip(b.u.iter_mut()) {
            *a += *x;
            *x = U256::ZERO;
        }
        for (a, x) in self.pi.iter_mut().zip(b.pi.iter_mut()) {
            *a += hi.clmul(x.reduce());
            *x = U256::ZERO;
        }
        for (a, x) in self.mir.iter_mut().zip(b.mir.iter_mut()) {
            *a += hi.clmul(x.reduce());
            *x = U256::ZERO;
        }
    }
}

/// Per-task buffers.
struct Scratch {
    delta: Vec<K>,
    val: Vec<K>,
    pair: Vec<K>,
}

impl Scratch {
    fn new(w: usize) -> Self {
        Scratch {
            delta: vec![K::ZERO; w],
            val: vec![K::ZERO; w],
            pair: vec![K::ZERO; 2 * w],
        }
    }
}

/// Add one pair's contribution with inner eq weight `eq_lo`.
#[inline]
fn kernel(x0: &[K], x1: &[K], d: usize, eq_lo: K, b: &mut Acc, delta: &mut [K], val: &mut [K]) {
    let w = x0.len();
    for t in 0..w {
        delta[t] = x0[t] + x1[t];
    }
    for e in 0..=d + 2 {
        let v: &[K] = match e {
            0 => x0,
            1 => x1,
            _ => {
                let t = e as u128;
                let top = if e <= 2 { w } else { A0 + d };
                for k in 0..top {
                    val[k] = x0[k] + delta[k].mul_t(t);
                }
                val
            }
        };
        let mut prod = v[A0];
        for i in 1..d {
            prod *= v[A0 + i];
        }
        if e != 1 {
            b.u[e] += (v[TL] * v[INC]).clmul(prod);
        }
        if e <= d {
            b.pi[e] += eq_lo.clmul(prod);
        }
        if e <= 2 {
            let mut m = U256::ZERO;
            for i in 0..d {
                m += v[A0 + i].clmul(v[A0 + d + i]);
            }
            b.mir[e] += eq_lo.clmul(m.reduce());
        }
    }
}

/// Pairs per task: whole blocks of `2^h` pairs.
fn task_pairs(pairs: usize, h: usize) -> usize {
    if pairs >= PAR {
        (1usize << h).max(CHUNK).min(pairs)
    } else {
        pairs
    }
}

/// Round sums over `pairs` pairs that `pair(m, out)` writes as two rows.
fn eval_pass<P: Fn(usize, &mut [K]) + Sync>(
    pairs: usize,
    w: usize,
    d: usize,
    wts: &SplitEq,
    pair: &P,
) -> Acc {
    let block = 1usize << wts.h;
    let task = task_pairs(pairs, wts.h);
    (0..pairs / task)
        .into_par_iter()
        .map(|t| {
            let mut acc = Acc::new(d);
            let mut blk = Acc::new(d);
            let mut s = Scratch::new(w);
            for m in t * task..(t + 1) * task {
                pair(m, &mut s.pair);
                let (x0, x1) = s.pair.split_at(w);
                let lo = m & (block - 1);
                kernel(x0, x1, d, wts.lo[lo], &mut blk, &mut s.delta, &mut s.val);
                if lo == block - 1 {
                    acc.flush(&mut blk, wts.hi[m >> wts.h]);
                }
            }
            acc
        })
        .reduce(|| Acc::new(d), Acc::merge)
}

/// Write `pairs` output pairs with `bind(p, out)` and, in the same pass,
/// accumulate the next round's sums over them with weights `wts`.
fn bind_pass<B: Fn(usize, &mut [K]) + Sync>(
    pairs: usize,
    w: usize,
    d: usize,
    wts: &SplitEq,
    bind: &B,
) -> (Vec<K>, Acc) {
    let block = 1usize << wts.h;
    let task = task_pairs(pairs, wts.h);
    let accs = std::sync::Mutex::new(Vec::new());
    // SAFETY: `task` divides `pairs`, so every chunk is `task` whole output
    // pairs of `2 * w` elements, and the zip below writes all of them.
    let data = unsafe {
        par_build(pairs * 2 * w, task * 2 * w, |start, dst| {
            debug_assert_eq!(dst.len(), task * 2 * w);
            let p0 = start / (2 * w);
            let mut acc = Acc::new(d);
            let mut blk = Acc::new(d);
            let mut s = Scratch::new(w);
            for (k, out) in dst.chunks_mut(2 * w).enumerate() {
                let p = p0 + k;
                bind(p, &mut s.pair);
                for (o, &x) in out.iter_mut().zip(s.pair.iter()) {
                    o.write(x);
                }
                let (x0, x1) = s.pair.split_at(w);
                let lo = p & (block - 1);
                kernel(x0, x1, d, wts.lo[lo], &mut blk, &mut s.delta, &mut s.val);
                if lo == block - 1 {
                    acc.flush(&mut blk, wts.hi[p >> wts.h]);
                }
            }
            accs.lock().unwrap().push(acc);
        })
    };
    let acc = accs
        .into_inner()
        .unwrap()
        .into_iter()
        .fold(Acc::new(d), Acc::merge);
    (data, acc)
}

#[inline]
fn bind_row(src0: &[K], src1: &[K], r: K, dst: &mut [K]) {
    for ((o, &a), &b) in dst.iter_mut().zip(src0).zip(src1) {
        *o = a + r * (a + b);
    }
}

/// The round message values `[g(t_0), …, g(t_{d+2})]`, given the eq prefix
/// `e_pref = eq(r_{<i}, z_{<i})` and `r_i`.
fn values(acc: &Acc, d: usize, cur: K, e_pref: K, r_i: K, delta: K) -> Vec<K> {
    let pi: Vec<K> = acc.pi.iter().map(|x| x.reduce()).collect();
    let mir: Vec<K> = acc.mir.iter().map(|x| x.reduce()).collect();
    let mut ys = vec![K::ZERO; d + 3];
    for (e, y) in ys.iter_mut().enumerate() {
        if e == 1 {
            continue;
        }
        let x = node(e);
        let p = if e <= d {
            pi[e]
        } else {
            interpolate_eval(&pi, x)
        };
        let m = if e <= 2 {
            mir[e]
        } else {
            interpolate_eval(&mir, x)
        };
        let kappa = e_pref * (K::ONE + x + r_i);
        *y = acc.u[e].reduce() + kappa * delta * (p + m);
    }
    ys[1] = cur + ys[0];
    ys
}

pub(crate) fn prove(
    w: &CycleWitness<'_>,
    r_k: &[K],
    r_j: &[K],
    claim: K,
    ch: &Challenges,
    tr: &mut Transcript,
) -> CycleResult {
    let n = r_j.len();
    assert!(n >= 2, "cycle prover: at least four cycles");
    let (c, d) = (w.c, w.d);
    let width = A0 + 2 * d;
    let tables_span = om_transcript::profiling::Span::new("twist.cycle.tables");
    let eps_scaled = eps_scaled(ch);
    let (eq_a, eq_am) = component_tables(r_k, c, d, &eps_scaled);
    let mask = (1u32 << c) - 1;
    let lt = SplitLt::new(r_j);
    let make_row = |j: usize, row: &mut [K]| {
        let ad = w.addr[j];
        row[TL] = lt.at(j) + ch.gamma;
        row[INC] = w.inc_e[j];
        for i in 0..d {
            let u = ((ad >> (i * c)) & mask) as usize;
            row[A0 + i] = eq_a[i][u];
            row[A0 + d + i] = eq_am[i][u];
        }
    };
    // Rows `j` and `j + 1` bound to `r`, written without building them.
    let bound_row = |j: usize, r: K, out: &mut [K]| {
        let bind = |a: K, b: K| a + r * (a + b);
        let (a0, a1) = (w.addr[j], w.addr[j + 1]);
        out[TL] = bind(lt.at(j), lt.at(j + 1)) + ch.gamma;
        out[INC] = bind(w.inc_e[j], w.inc_e[j + 1]);
        for i in 0..d {
            let u0 = ((a0 >> (i * c)) & mask) as usize;
            let u1 = ((a1 >> (i * c)) & mask) as usize;
            out[A0 + i] = bind(eq_a[i][u0], eq_a[i][u1]);
            out[A0 + d + i] = bind(eq_am[i][u0], eq_am[i][u1]);
        }
    };
    drop(tables_span);
    let _span = om_transcript::profiling::Span::new("twist.cycle.rounds");
    sumcheck_init(tr, LABEL, claim, n, d + 2);
    let mut msgs = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut cur = claim;
    let mut e_pref = K::ONE;

    // Round 0 over rows built on the fly.
    let acc = eval_pass(
        1 << (n - 1),
        width,
        d,
        &SplitEq::new(&r_j[1..]),
        &|m, out: &mut [K]| {
            let (x0, x1) = out.split_at_mut(width);
            make_row(2 * m, x0);
            make_row(2 * m + 1, x1);
        },
    );
    let mut ys = values(&acc, d, cur, e_pref, r_j[0], ch.delta);
    let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
    (cur, e_pref) = (next, e_pref * (K::ONE + r + r_j[0]));
    z.push(r);
    // Bind round 0 and accumulate round 1 in one pass.
    let (mut rows, mut acc) = bind_pass(
        1 << (n - 2),
        width,
        d,
        &SplitEq::new(&r_j[2..]),
        &|p, out: &mut [K]| {
            let (o0, o1) = out.split_at_mut(width);
            bound_row(4 * p, r, o0);
            bound_row(4 * p + 2, r, o1);
        },
    );
    for i in 1..n {
        ys = values(&acc, d, cur, e_pref, r_j[i], ch.delta);
        let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
        (cur, e_pref) = (next, e_pref * (K::ONE + r + r_j[i]));
        z.push(r);
        let src = &rows;
        if i + 1 < n {
            (rows, acc) = bind_pass(
                src.len() / (4 * width),
                width,
                d,
                &SplitEq::new(&r_j[i + 2..]),
                &|p, out: &mut [K]| {
                    let s = &src[4 * p * width..(4 * p + 4) * width];
                    let (o0, o1) = out.split_at_mut(width);
                    bind_row(&s[..width], &s[width..2 * width], r, o0);
                    bind_row(&s[2 * width..3 * width], &s[3 * width..], r, o1);
                },
            );
        } else {
            let mut out = vec![K::ZERO; width];
            bind_row(&src[..width], &src[width..], r, &mut out);
            rows = out;
        }
    }
    let row = &rows[..width];
    if checks_enabled() {
        let prod: K = (0..d).fold(K::ONE, |p, i| p * row[A0 + i]);
        let mir: K = (0..d).fold(K::ZERO, |s, i| s + row[A0 + i] * row[A0 + d + i]);
        let value = row[TL] * row[INC] * prod + e_pref * ch.delta * (prod + mir);
        assert_eq!(value, cur, "cycle check final value");
    }
    conclude(
        row[A0..A0 + d].to_vec(),
        &row[A0 + d..A0 + 2 * d],
        &eps_scaled,
        row[INC],
        msgs,
        z,
        tr,
    )
}
