//! The prover of the cycle check (protocol in `cycle.rs`). It sends the same
//! messages as the reference prover there, which stores `LT(·, r_j) + γ`
//! and `δ eq(r_j, ·)` as columns, but stores neither.
//!
//! **Split eq.** In round `i`, with `z_{<i}` bound and `X` the current
//! variable, `eq(r_j, (z_{<i}, X, m)) = E_i · eq(r_i, X) · eq(r_{>i}, m)`
//! with `E_i = eq(r_{<i}, z_{<i})`. The prover accumulates the eq-weighted
//! part `Σ_m eq(r_{>i}, m) (δ Π_i A_i + Σ_i ε_i' A_i Ã_i)(X, m)`, which has
//! degree `d` in `X`, and multiplies by the linear factor `E_i eq(r_i, X)`
//! once per round. The weight `eq(r_{>i}, m)` is split as
//! `eq_lo[m_lo] · eq_hi[m_hi]`: the inner sum over `m_lo` is accumulated
//! unreduced and the outer factor is applied once per block.
//!
//! **Split LT.** With the low variables least significant,
//! `LT((z_{<i}, X, m), r_j) = LT(m, r_{>i}) + eq(m, r_{>i}) · L_i(X)` with
//! `L_i(X) = LT((z_{<i}, X), r_{≤i}) = (1 + X) r_i + eq(X, r_i) ℓ_i` and
//! `ℓ_i = LT(z_{<i}, r_{<i})`. With `P = Π_i A_i · Inc`, the value part is
//! `Σ_m (LT(m, r_{>i}) + γ) P(X, m) + L_i(X) Σ_m eq(r_{>i}, m) P(X, m)`; both
//! sums have degree `d + 1`. `LT(m, ·)` is split as
//! `lt_hi[m_hi] + eq_hi[m_hi] · lt_lo[m_lo]`, so the inner sums need
//! `Σ P`, `Σ lt_lo P` and `Σ eq_lo P`.

use om_field::{K, U256};
use om_sumcheck::{interpolate_eval, sumcheck_init};
use om_transcript::Transcript;
use rayon::prelude::*;

use super::cycle::{
    Challenges, CycleResult, CycleWitness, LABEL, component_tables, conclude, eps_scaled,
};
use super::util::{PAR, SplitEq, checks_enabled, emit_round, node, par_build};

/// Row layout: `[inc, A_1..A_d, Ã_1..Ã_d]` with `Ã_i = ε_i' A_i(1 + r_k^i, ·)`.
const INC: usize = 0;
const A0: usize = 1;

/// Pairs per parallel task (rounded up to whole weight blocks).
const CHUNK: usize = 1 << 9;

/// Split `eq(r_{>i}, ·)` and `LT(·, r_{>i})` tables for one round.
struct Weights {
    eq: SplitEq,
    lt_lo: Vec<K>,
    lt_hi: Vec<K>,
}

impl Weights {
    fn new(s: &[K]) -> Self {
        let eq = SplitEq::new(s);
        let h = eq.h;
        Weights {
            lt_lo: om_memcheck::twist::lt_table(&s[..h]),
            lt_hi: om_memcheck::twist::lt_table(&s[h..]),
            eq,
        }
    }
}

/// Round sums, also used for one block of pairs. At node `e`, for
/// `e ≤ d + 1`: `p0` is `Σ P`, `plt` is `Σ lt·P` and `peq` is `Σ eq·P`; in a
/// round total, `p0` is unused and `plt` holds `Σ (LT + γ) P`. `pi[e]` is
/// the eq-weighted `ΠA` at nodes `0..=d` and `mir[e]` the eq-weighted mirror
/// sum at nodes `0..=2`.
#[derive(Clone)]
struct Acc {
    p0: Vec<K>,
    plt: Vec<U256>,
    peq: Vec<U256>,
    pi: Vec<U256>,
    mir: [U256; 3],
}

impl Acc {
    fn new(d: usize) -> Self {
        Acc {
            p0: vec![K::ZERO; d + 2],
            plt: vec![U256::ZERO; d + 2],
            peq: vec![U256::ZERO; d + 2],
            pi: vec![U256::ZERO; d + 1],
            mir: [U256::ZERO; 3],
        }
    }

    fn merge(mut self, o: Acc) -> Acc {
        for (a, b) in self.plt.iter_mut().zip(o.plt) {
            *a += b;
        }
        for (a, b) in self.peq.iter_mut().zip(o.peq) {
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

    /// Add block `b`, whose weights lack the outer factors, and clear it.
    /// `lt_hi` already includes `γ`.
    fn flush(&mut self, b: &mut Acc, eq_hi: K, lt_hi: K) {
        for e in 0..b.p0.len() {
            self.plt[e] += lt_hi.clmul(b.p0[e]) + eq_hi.clmul(b.plt[e].reduce());
            self.peq[e] += eq_hi.clmul(b.peq[e].reduce());
            b.p0[e] = K::ZERO;
            b.plt[e] = U256::ZERO;
            b.peq[e] = U256::ZERO;
        }
        for (a, x) in self.pi.iter_mut().zip(b.pi.iter_mut()) {
            *a += eq_hi.clmul(x.reduce());
            *x = U256::ZERO;
        }
        for (a, x) in self.mir.iter_mut().zip(b.mir.iter_mut()) {
            *a += eq_hi.clmul(x.reduce());
            *x = U256::ZERO;
        }
    }
}

/// Per-task buffers.
struct Scratch {
    delta: Vec<K>,
    val: Vec<K>,
    pair: Vec<K>,
    keys: Vec<usize>,
}

impl Scratch {
    fn new(w: usize) -> Self {
        Scratch {
            delta: vec![K::ZERO; w],
            val: vec![K::ZERO; w],
            pair: vec![K::ZERO; 2 * w],
            keys: vec![0; (w - 1) / 2],
        }
    }
}

/// Add one pair's contribution with inner weights `eq_lo` and `lt_lo`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn kernel(
    x0: &[K],
    x1: &[K],
    d: usize,
    eq_lo: K,
    lt_lo: K,
    b: &mut Acc,
    delta: &mut [K],
    val: &mut [K],
) {
    let w = x0.len();
    for t in 0..w {
        delta[t] = x0[t] + x1[t];
    }
    // With Inc zero on both rows (two loads), `P` vanishes and only the
    // degree-d eq-weighted part remains.
    let store = !(x0[INC].is_zero() && x1[INC].is_zero());
    for e in 0..=d + store as usize {
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
        if store {
            let p = prod * v[INC];
            b.p0[e] += p;
            b.plt[e] += lt_lo.clmul(p);
            b.peq[e] += eq_lo.clmul(p);
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

/// Round sums over `pairs` pairs; `pair(m, eq_lo, lt_lo, blk, scratch)`
/// adds pair `m`'s contribution to `blk`.
fn eval_pass<P: Fn(usize, K, K, &mut Acc, &mut Scratch) + Sync>(
    pairs: usize,
    w: usize,
    d: usize,
    wts: &Weights,
    gamma: K,
    pair: &P,
) -> Acc {
    let h = wts.eq.h;
    let block = 1usize << h;
    let task = task_pairs(pairs, h);
    (0..pairs / task)
        .into_par_iter()
        .map(|t| {
            let mut acc = Acc::new(d);
            let mut blk = Acc::new(d);
            let mut s = Scratch::new(w);
            for m in t * task..(t + 1) * task {
                let lo = m & (block - 1);
                pair(m, wts.eq.lo[lo], wts.lt_lo[lo], &mut blk, &mut s);
                if lo == block - 1 {
                    acc.flush(&mut blk, wts.eq.hi[m >> h], wts.lt_hi[m >> h] + gamma);
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
    wts: &Weights,
    gamma: K,
    bind: &B,
) -> (Vec<K>, Acc) {
    let h = wts.eq.h;
    let block = 1usize << h;
    let task = task_pairs(pairs, h);
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
                kernel(
                    x0,
                    x1,
                    d,
                    wts.eq.lo[lo],
                    wts.lt_lo[lo],
                    &mut blk,
                    &mut s.delta,
                    &mut s.val,
                );
                if lo == block - 1 {
                    acc.flush(&mut blk, wts.eq.hi[p >> h], wts.lt_hi[p >> h] + gamma);
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

/// Round-0 tables for one-hot rows. Before binding, `A_i(·, j)` takes one
/// of `2^c` values, so on a pair of rows it is fixed by the pair's chunk
/// values `k = u_0 + 2^c u_1`. `a[(i 2^{2c} + k)(d + 2) + e]` is `A_i` at
/// node `e ≤ d + 1` and `m[(i 2^{2c} + k) 3 + e]` is `A_i Ã_i` at node
/// `e ≤ 2`.
struct PairTables {
    c: usize,
    d: usize,
    a: Vec<K>,
    m: Vec<K>,
}

/// Largest chunk width that uses round-0 tables: `2^{2c}` pairs of values
/// per component.
const LOOKUP_MAX_C: usize = 5;

impl PairTables {
    fn new(eq_a: &[Vec<K>], eq_am: &[Vec<K>], c: usize) -> Self {
        let d = eq_a.len();
        let (side, pts) = (1usize << c, d + 2);
        let mut a = Vec::with_capacity(d * side * side * pts);
        let mut m = Vec::with_capacity(d * side * side * 3);
        for i in 0..d {
            for k in 0..side * side {
                let (u0, u1) = (k & (side - 1), k >> c);
                let (a0, a1) = (eq_a[i][u0], eq_a[i][u1]);
                let (m0, m1) = (eq_am[i][u0], eq_am[i][u1]);
                for e in 0..pts {
                    a.push(a0 + (a0 + a1).mul_t(e as u128));
                }
                for e in 0..3u128 {
                    m.push((a0 + (a0 + a1).mul_t(e)) * (m0 + (m0 + m1).mul_t(e)));
                }
            }
        }
        PairTables { c, d, a, m }
    }

    /// Pair `(j, j + 1)`'s contribution in round 0.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn kernel(
        &self,
        addr: &[u32],
        inc: &[K],
        j: usize,
        eq_lo: K,
        lt_lo: K,
        b: &mut Acc,
        keys: &mut [usize],
    ) {
        let (c, d) = (self.c, self.d);
        let mask = (1u32 << c) - 1;
        let (a0, a1) = (addr[j], addr[j + 1]);
        for (i, key) in keys.iter_mut().enumerate() {
            let k = (((a0 >> (i * c)) & mask) | (((a1 >> (i * c)) & mask) << c)) as usize;
            *key = (i << (2 * c)) + k;
        }
        let (i0, i1) = (inc[j], inc[j + 1]);
        let di = i0 + i1;
        let pts = d + 2;
        let store = !(i0.is_zero() && i1.is_zero());
        for e in 0..=d + store as usize {
            let mut prod = self.a[keys[0] * pts + e];
            for &key in &keys[1..] {
                prod *= self.a[key * pts + e];
            }
            if store {
                let p = prod * (i0 + di.mul_t(e as u128));
                b.p0[e] += p;
                b.plt[e] += lt_lo.clmul(p);
                b.peq[e] += eq_lo.clmul(p);
            }
            if e <= d {
                b.pi[e] += eq_lo.clmul(prod);
            }
            if e <= 2 {
                let mut m = K::ZERO;
                for &key in keys.iter() {
                    m += self.m[key * 3 + e];
                }
                b.mir[e] += eq_lo.clmul(m);
            }
        }
    }
}

/// `t[i 2^{2c} + k] = x_i(u_0) + r (x_i(u_0) + x_i(u_1))` for `k = u_0 + 2^c u_1`:
/// component tables bound to the round-0 challenge.
fn bound_pair_table(x: &[Vec<K>], c: usize, r: K) -> Vec<K> {
    let side = 1usize << c;
    let mut t = Vec::with_capacity(x.len() * side * side);
    for xi in x {
        for k in 0..side * side {
            let (a, b) = (xi[k & (side - 1)], xi[k >> c]);
            t.push(a + r * (a + b));
        }
    }
    t
}

#[inline]
fn bind_row(src0: &[K], src1: &[K], r: K, dst: &mut [K]) {
    for ((o, &a), &b) in dst.iter_mut().zip(src0).zip(src1) {
        *o = a + r * (a + b);
    }
}

/// Running prefix factors: `e = eq(r_{<i}, z_{<i})` and
/// `l = LT(z_{<i}, r_{<i})`.
#[derive(Clone, Copy)]
struct Prefix {
    e: K,
    l: K,
}

impl Prefix {
    /// `L_i(x) = LT((z_{<i}, x), r_{≤i})`.
    fn lt_at(self, x: K, r_i: K) -> K {
        (K::ONE + x) * r_i + (K::ONE + x + r_i) * self.l
    }

    fn eq_at(self, x: K, r_i: K) -> K {
        self.e * (K::ONE + x + r_i)
    }

    fn bind(self, z_i: K, r_i: K) -> Prefix {
        Prefix {
            e: self.eq_at(z_i, r_i),
            l: self.lt_at(z_i, r_i),
        }
    }
}

/// The round message values `[g(t_0), …, g(t_{d+2})]`.
fn values(acc: &Acc, d: usize, cur: K, pre: Prefix, r_i: K, delta: K) -> Vec<K> {
    let red = |v: &[U256]| -> Vec<K> { v.iter().map(|x| x.reduce()).collect() };
    let (plt, peq, pi, mir) = (red(&acc.plt), red(&acc.peq), red(&acc.pi), red(&acc.mir));
    let at = |v: &[K], e: usize| -> K {
        if e < v.len() {
            v[e]
        } else {
            interpolate_eval(v, node(e))
        }
    };
    let mut ys = vec![K::ZERO; d + 3];
    for (e, y) in ys.iter_mut().enumerate() {
        if e == 1 {
            continue;
        }
        let x = node(e);
        *y = at(&plt, e)
            + pre.lt_at(x, r_i) * at(&peq, e)
            + pre.eq_at(x, r_i) * delta * (at(&pi, e) + at(&mir, e));
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
    let make_row = |j: usize, row: &mut [K]| {
        let ad = w.addr[j];
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
    let mut pre = Prefix {
        e: K::ONE,
        l: K::ZERO,
    };

    // Round 0 reads one-hot rows through pair tables, or builds the rows on
    // the fly for wide chunks.
    let lookups = (c <= LOOKUP_MAX_C).then(|| PairTables::new(&eq_a, &eq_am, c));
    let w0 = Weights::new(&r_j[1..]);
    let acc = match &lookups {
        Some(t) => eval_pass(
            1 << (n - 1),
            width,
            d,
            &w0,
            ch.gamma,
            &|m, eq_lo, lt_lo, blk: &mut Acc, s: &mut Scratch| {
                t.kernel(w.addr, w.inc_e, 2 * m, eq_lo, lt_lo, blk, &mut s.keys);
            },
        ),
        None => eval_pass(
            1 << (n - 1),
            width,
            d,
            &w0,
            ch.gamma,
            &|m, eq_lo, lt_lo, blk: &mut Acc, s: &mut Scratch| {
                let (x0, x1) = s.pair.split_at_mut(width);
                make_row(2 * m, x0);
                make_row(2 * m + 1, x1);
                kernel(x0, x1, d, eq_lo, lt_lo, blk, &mut s.delta, &mut s.val);
            },
        ),
    };
    let mut ys = values(&acc, d, cur, pre, r_j[0], ch.delta);
    let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
    (cur, pre) = (next, pre.bind(r, r_j[0]));
    z.push(r);
    // Bind round 0 and accumulate round 1 in one pass. With pair tables, the
    // bound one-hot columns are lookups into tables bound to `r`.
    let w1 = Weights::new(&r_j[2..]);
    let (mut rows, mut acc) = match lookups {
        Some(_) => {
            let ba = bound_pair_table(&eq_a, c, r);
            let bm = bound_pair_table(&eq_am, c, r);
            let bound = |j: usize, out: &mut [K]| {
                let (a0, a1) = (w.addr[j], w.addr[j + 1]);
                let (i0, i1) = (w.inc_e[j], w.inc_e[j + 1]);
                out[INC] = i0 + r * (i0 + i1);
                for i in 0..d {
                    let k = (((a0 >> (i * c)) & mask) | (((a1 >> (i * c)) & mask) << c)) as usize;
                    out[A0 + i] = ba[(i << (2 * c)) + k];
                    out[A0 + d + i] = bm[(i << (2 * c)) + k];
                }
            };
            bind_pass(
                1 << (n - 2),
                width,
                d,
                &w1,
                ch.gamma,
                &|p, out: &mut [K]| {
                    let (o0, o1) = out.split_at_mut(width);
                    bound(4 * p, o0);
                    bound(4 * p + 2, o1);
                },
            )
        }
        None => bind_pass(
            1 << (n - 2),
            width,
            d,
            &w1,
            ch.gamma,
            &|p, out: &mut [K]| {
                let (o0, o1) = out.split_at_mut(width);
                bound_row(4 * p, r, o0);
                bound_row(4 * p + 2, r, o1);
            },
        ),
    };
    for i in 1..n {
        ys = values(&acc, d, cur, pre, r_j[i], ch.delta);
        let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
        (cur, pre) = (next, pre.bind(r, r_j[i]));
        z.push(r);
        let src = &rows;
        if i + 1 < n {
            (rows, acc) = bind_pass(
                src.len() / (4 * width),
                width,
                d,
                &Weights::new(&r_j[i + 2..]),
                ch.gamma,
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
        let value = (pre.l + ch.gamma) * row[INC] * prod + pre.e * ch.delta * (prod + mir);
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
