//! The read check of Twist v2.
//!
//! It proves `RV(r_o, r_c) = Σ_{k,j} eq(r_c, j) · ra(k, j) · V(r_o, k, j)`,
//! where `ra(k, j) = Π_i A_i(k_i, j)` and `V(o, k, j)` is bit `o` of cell `k`
//! before cycle `j`'s write. There is no write term: with a shared
//! read/write address, the written value is `RV + Inc` by definition, so
//! nothing else needs to be proved about it (issue item 3).
//!
//! The sum-check has `n` cycle rounds and `a` address rounds, in either order
//! (issue items 1 and 2). Both orders produce the same proof shape and the
//! same verifier; the order is bound in the statement transcript.
//!
//! * Cycle rounds use the split-eq factorisation: the prover sends `q̃(0)` and
//!   the leading coefficient of `q̃`, where the round polynomial is
//!   `g(X) = eq(r_c[i], X) · q̃(X)` and `q̃` has degree 2. The verifier derives
//!   `q̃(1)` from the running claim; `r_c` is sampled nonzero so this division
//!   is always defined.
//! * Address rounds send `g(0)` and the leading coefficient of the degree-2
//!   round polynomial.

use om_field::{K, U256};
use om_sumcheck::{VerificationError, eq_eval};
use om_transcript::Transcript;
use rayon::prelude::*;

use super::util::{PAR, SplitEq, checks_enabled, sublabel};

/// Which variables the read sum-check binds first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReadOrder {
    /// Cycle variables first, over a sparse matrix whose entries carry the
    /// memory value before and after their accesses (issue item 1).
    #[default]
    CycleFirst,
    /// Address variables first, with per-block memory checkpoints
    /// (issue item 2).
    AddressFirst,
}

impl ReadOrder {
    pub(crate) fn code(self) -> u64 {
        match self {
            ReadOrder::CycleFirst => 0,
            ReadOrder::AddressFirst => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadProof {
    /// Cycle rounds: `[q̃(0), leading coefficient of q̃]`.
    pub cycle: Vec<[K; 2]>,
    /// Address rounds: `[g(0), leading coefficient of g]`.
    pub address: Vec<[K; 2]>,
    /// `ra(r_k, r_j)`.
    pub ra: K,
    /// `V(r_o, r_k, r_j)`.
    pub val: K,
}

/// The prover's per-cycle data, already evaluated at `r_o` where needed.
pub(crate) struct ReadWitness<'a> {
    pub addr: &'a [u32],
    /// `WE(rv_j)`: the value each cycle reads.
    pub rv_e: &'a [K],
    /// `WE(inc_j)`: the change each cycle makes to its cell.
    pub inc_e: &'a [K],
    /// `inc_j != 0`: the cycle changes memory.
    pub store: &'a [bool],
    /// `WE(M0[k])` for every cell.
    pub m0_e: &'a [K],
    pub a: usize,
}

pub(crate) struct ReadResult {
    pub proof: ReadProof,
    /// Address point, low bit first.
    pub r_k: Vec<K>,
    /// Cycle point, low bit first.
    pub r_j: Vec<K>,
}

const LABEL: &[u8] = b"ram/read";

fn init(tr: &mut Transcript, claim: K, n: usize, a: usize, order: ReadOrder) {
    tr.absorb_k(&sublabel(LABEL, b"/claim"), claim);
    let params: Vec<u8> = [n as u64, a as u64, order.code()]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    tr.absorb_bytes(&sublabel(LABEL, b"/params"), &params);
}

fn cycle_round(tr: &mut Transcript, msg: [K; 2]) -> K {
    tr.absorb_ks(b"ram/read/cycle/msg", &msg);
    tr.challenge_k(b"ram/read/cycle/chal")
}

fn address_round(tr: &mut Transcript, msg: [K; 2]) -> K {
    tr.absorb_ks(b"ram/read/address/msg", &msg);
    tr.challenge_k(b"ram/read/address/chal")
}

fn finish(tr: &mut Transcript, ra: K, val: K) {
    tr.absorb_ks(b"ram/read/evals", &[ra, val]);
}

/// `q(r)` for the quadratic with `q(0)`, `q(1)` and leading coefficient.
#[inline]
fn quad_at(q0: K, q1: K, qinf: K, r: K) -> K {
    q0 + r * (q0 + q1 + qinf + r * qinf)
}

/// `eq(a, X) = 1 + a + X`.
#[inline]
fn eq_lin(a: K, x: K) -> K {
    K::ONE + a + x
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub(crate) struct ReadClaims {
    pub r_k: Vec<K>,
    pub r_j: Vec<K>,
    pub ra: K,
    pub val: K,
}

pub(crate) fn verify(
    claim: K,
    r_c: &[K],
    a: usize,
    order: ReadOrder,
    proof: &ReadProof,
    tr: &mut Transcript,
) -> Result<ReadClaims, VerificationError> {
    let n = r_c.len();
    if proof.cycle.len() != n || proof.address.len() != a {
        return Err(VerificationError::new("read check: wrong number of rounds"));
    }
    init(tr, claim, n, a, order);
    let mut cur = claim;
    let mut r_j = Vec::with_capacity(n);
    let mut r_k = Vec::with_capacity(a);
    let mut cycle_phase = |cur: &mut K, tr: &mut Transcript| -> Result<(), VerificationError> {
        for (i, &[q0, qinf]) in proof.cycle.iter().enumerate() {
            let ai = r_c[i];
            let inv = ai
                .inv()
                .ok_or_else(|| VerificationError::new("read check: zero cycle point"))?;
            // cur = (1 + a) q(0) + a q(1).
            let q1 = (*cur + (K::ONE + ai) * q0) * inv;
            let r = cycle_round(tr, [q0, qinf]);
            *cur = eq_lin(ai, r) * quad_at(q0, q1, qinf, r);
            r_j.push(r);
        }
        Ok(())
    };
    let mut address_phase = |cur: &mut K, tr: &mut Transcript| {
        for &[g0, ginf] in &proof.address {
            let g1 = *cur + g0;
            let r = address_round(tr, [g0, ginf]);
            *cur = quad_at(g0, g1, ginf, r);
            r_k.push(r);
        }
    };
    match order {
        ReadOrder::CycleFirst => {
            cycle_phase(&mut cur, tr)?;
            address_phase(&mut cur, tr);
        }
        ReadOrder::AddressFirst => {
            address_phase(&mut cur, tr);
            cycle_phase(&mut cur, tr)?;
        }
    }
    if cur != eq_eval(r_c, &r_j) * proof.ra * proof.val {
        return Err(VerificationError::new(
            "read check: final evaluation mismatch",
        ));
    }
    finish(tr, proof.ra, proof.val);
    Ok(ReadClaims {
        r_k,
        r_j,
        ra: proof.ra,
        val: proof.val,
    })
}

// ---------------------------------------------------------------------------
// Shared prover pieces
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Acc3 {
    q0: U256,
    q1: U256,
    qinf: U256,
}

impl Acc3 {
    fn add(mut self, o: Acc3) -> Acc3 {
        self.q0 += o.q0;
        self.q1 += o.q1;
        self.qinf += o.qinf;
        self
    }

    fn reduce(self) -> (K, K, K) {
        (self.q0.reduce(), self.q1.reduce(), self.qinf.reduce())
    }
}

/// Dense split-eq cycle rounds over `ra(j)` and `val(j)` (both of length
/// `2^(n - start)`), where `r_c[..start]` is already bound into `scale`.
/// Used by the address-first prover after its address rounds.
fn dense_cycle_rounds(
    mut ra: Vec<K>,
    mut val: Vec<K>,
    r_c: &[K],
    mut cur: K,
    tr: &mut Transcript,
    msgs: &mut Vec<[K; 2]>,
    r_j: &mut Vec<K>,
) -> (K, K, K) {
    let n = r_c.len();
    let mut scale = K::ONE;
    for i in 0..n {
        let half = ra.len() / 2;
        let w = SplitEq::new(&r_c[i + 1..]);
        let pair = |m: usize| -> Acc3 {
            let (a0, a1) = (ra[2 * m], ra[2 * m + 1]);
            let (v0, v1) = (val[2 * m], val[2 * m + 1]);
            let wm = w.at(m);
            Acc3 {
                q0: (wm * a0).clmul(v0),
                q1: (wm * a1).clmul(v1),
                qinf: (wm * (a0 + a1)).clmul(v0 + v1),
            }
        };
        let acc = if half >= PAR {
            (0..half)
                .into_par_iter()
                .with_min_len(1 << 10)
                .map(pair)
                .reduce(Acc3::default, Acc3::add)
        } else {
            (0..half).map(pair).fold(Acc3::default(), Acc3::add)
        };
        let (q0, q1, qinf) = acc.reduce();
        let (q0, q1, qinf) = (scale * q0, scale * q1, scale * qinf);
        if checks_enabled() {
            assert_eq!(
                (K::ONE + r_c[i]) * q0 + r_c[i] * q1,
                cur,
                "read cycle round {i}"
            );
        }
        let r = cycle_round(tr, [q0, qinf]);
        msgs.push([q0, qinf]);
        r_j.push(r);
        let e = eq_lin(r_c[i], r);
        cur = e * quad_at(q0, q1, qinf, r);
        scale *= e;
        let bind = |t: &[K]| -> Vec<K> {
            if half >= PAR {
                (0..half)
                    .into_par_iter()
                    .with_min_len(1 << 10)
                    .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
                    .collect()
            } else {
                (0..half)
                    .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
                    .collect()
            }
        };
        ra = bind(&ra);
        val = bind(&val);
    }
    (cur, ra[0], val[0])
}

/// Split the sorted `keys` into about `parts` ranges whose boundaries never
/// separate two keys with the same `key >> 1`.
fn pair_aligned_ranges(keys: &[u32], parts: usize) -> Vec<std::ops::Range<usize>> {
    let len = keys.len();
    let parts = parts.max(1);
    let mut cuts = vec![0];
    for p in 1..parts {
        let mut b = (len * p / parts).max(*cuts.last().unwrap());
        while b > 0 && b < len && keys[b] >> 1 == keys[b - 1] >> 1 {
            b += 1;
        }
        if b > *cuts.last().unwrap() && b < len {
            cuts.push(b);
        }
    }
    cuts.push(len);
    cuts.windows(2).map(|w| w[0]..w[1]).collect()
}

fn bind_dense(t: &[K], r: K) -> Vec<K> {
    let half = t.len() / 2;
    if half >= PAR {
        (0..half)
            .into_par_iter()
            .with_min_len(1 << 10)
            .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
            .collect()
    } else {
        (0..half)
            .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
            .collect()
    }
}

/// Sparse address rounds after the cycle rounds: `Σ_k scale · ra(k) · V(k)`
/// with `ra` supported on the sorted `keys`, and `V(k) = M0(k)` off them.
#[allow(clippy::too_many_arguments)]
fn sparse_address_rounds(
    mut keys: Vec<u32>,
    mut ra: Vec<K>,
    mut val: Vec<K>,
    m0_e: &[K],
    scale: K,
    mut cur: K,
    tr: &mut Transcript,
    msgs: &mut Vec<[K; 2]>,
    r_k: &mut Vec<K>,
    a: usize,
) -> (K, K, K) {
    let mut m0: Vec<K> = Vec::new();
    for s in 0..a {
        let m0s: &[K] = if s == 0 { m0_e } else { &m0 };
        let ranges = pair_aligned_ranges(&keys, if keys.len() >= PAR { 256 } else { 1 });
        let round = |range: &std::ops::Range<usize>| -> Acc3 {
            let mut acc = Acc3::default();
            let mut i = range.start;
            while i < range.end {
                let t = keys[i] >> 1;
                let (ra0, v0, ra1, v1);
                if keys[i] & 1 == 0 {
                    if i + 1 < range.end && keys[i + 1] >> 1 == t {
                        (ra0, v0, ra1, v1) = (ra[i], val[i], ra[i + 1], val[i + 1]);
                        i += 2;
                    } else {
                        (ra0, v0, ra1, v1) = (ra[i], val[i], K::ZERO, m0s[2 * t as usize + 1]);
                        i += 1;
                    }
                } else {
                    (ra0, v0, ra1, v1) = (K::ZERO, m0s[2 * t as usize], ra[i], val[i]);
                    i += 1;
                }
                acc.q0 += ra0.clmul(v0);
                acc.q1 += ra1.clmul(v1);
                acc.qinf += (ra0 + ra1).clmul(v0 + v1);
            }
            acc
        };
        let acc = if ranges.len() > 1 {
            ranges
                .par_iter()
                .map(round)
                .reduce(Acc3::default, Acc3::add)
        } else {
            round(&ranges[0])
        };
        let (g0, g1, ginf) = acc.reduce();
        let (g0, g1, ginf) = (scale * g0, scale * g1, scale * ginf);
        if checks_enabled() {
            assert_eq!(g0 + g1, cur, "read address round {s}");
        }
        let r = address_round(tr, [g0, ginf]);
        msgs.push([g0, ginf]);
        r_k.push(r);
        cur = quad_at(g0, g1, ginf, r);
        // Bind the sparse entries.
        let bind = |range: &std::ops::Range<usize>| -> (Vec<u32>, Vec<K>, Vec<K>) {
            let cap = range.len();
            let (mut nk, mut nr, mut nv) = (
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
                Vec::with_capacity(cap),
            );
            let mut i = range.start;
            while i < range.end {
                let t = keys[i] >> 1;
                let (ra0, v0, ra1, v1);
                if keys[i] & 1 == 0 {
                    if i + 1 < range.end && keys[i + 1] >> 1 == t {
                        (ra0, v0, ra1, v1) = (ra[i], val[i], ra[i + 1], val[i + 1]);
                        i += 2;
                    } else {
                        (ra0, v0, ra1, v1) = (ra[i], val[i], K::ZERO, m0s[2 * t as usize + 1]);
                        i += 1;
                    }
                } else {
                    (ra0, v0, ra1, v1) = (K::ZERO, m0s[2 * t as usize], ra[i], val[i]);
                    i += 1;
                }
                nk.push(t);
                nr.push(ra0 + r * (ra0 + ra1));
                nv.push(v0 + r * (v0 + v1));
            }
            (nk, nr, nv)
        };
        let parts: Vec<(Vec<u32>, Vec<K>, Vec<K>)> = if ranges.len() > 1 {
            ranges.par_iter().map(bind).collect()
        } else {
            vec![bind(&ranges[0])]
        };
        let (mut nk, mut nr, mut nv) = (Vec::new(), Vec::new(), Vec::new());
        for (k, r_, v) in parts {
            nk.extend(k);
            nr.extend(r_);
            nv.extend(v);
        }
        m0 = bind_dense(m0s, r);
        keys = nk;
        ra = nr;
        val = nv;
    }
    debug_assert!(keys.len() <= 1);
    let (ra_end, val_end) = if keys.is_empty() {
        (K::ZERO, m0[0])
    } else {
        (ra[0], val[0])
    };
    (cur, ra_end, val_end)
}

// ---------------------------------------------------------------------------
// Cycle-first prover (issue item 1)
// ---------------------------------------------------------------------------

/// The sparse matrix at one level of cycle binding. Row `m` covers the
/// `2^level` cycles `[m 2^level, (m+1) 2^level)` and stores one entry for
/// each distinct address accessed there, sorted by address, at offset
/// `m 2^level` in each column.
struct Level {
    level: usize,
    len: Vec<u32>,
    key: Vec<u32>,
    /// `ra(key, row)` bound in the low cycle variables.
    ra: Vec<K>,
    /// `V(key, row)` bound in the low cycle variables.
    val: Vec<K>,
    /// The cell's value before the row's first access to it.
    prev: Vec<K>,
    /// The cell's value after the row's last access to it.
    next: Vec<K>,
}

/// One entry of the merge of a row pair: values on the left and right rows,
/// with a missing side filled from its neighbour.
#[inline]
fn merged_entry(lv: &Level, li: Option<usize>, ri: Option<usize>) -> (u32, K, K, K, K, K, K) {
    // Returns (key, ra0, ra1, v0, v1, prev, next).
    match (li, ri) {
        (Some(l), Some(r)) => (
            lv.key[l], lv.ra[l], lv.ra[r], lv.val[l], lv.val[r], lv.prev[l], lv.next[r],
        ),
        (Some(l), None) => (
            lv.key[l],
            lv.ra[l],
            K::ZERO,
            lv.val[l],
            lv.next[l],
            lv.prev[l],
            lv.next[l],
        ),
        (None, Some(r)) => (
            lv.key[r],
            K::ZERO,
            lv.ra[r],
            lv.prev[r],
            lv.val[r],
            lv.prev[r],
            lv.next[r],
        ),
        (None, None) => unreachable!(),
    }
}

/// Walk the sorted merge of rows `2m` and `2m+1` of `lv`.
#[inline]
fn for_each_merged(lv: &Level, m: usize, mut f: impl FnMut(Option<usize>, Option<usize>)) {
    let cap = 1usize << lv.level;
    let (lo, ro) = (2 * m * cap, (2 * m + 1) * cap);
    let (le, re) = (lo + lv.len[2 * m] as usize, ro + lv.len[2 * m + 1] as usize);
    let (mut i, mut j) = (lo, ro);
    while i < le && j < re {
        let (ki, kj) = (lv.key[i], lv.key[j]);
        if ki == kj {
            f(Some(i), Some(j));
            i += 1;
            j += 1;
        } else if ki < kj {
            f(Some(i), None);
            i += 1;
        } else {
            f(None, Some(j));
            j += 1;
        }
    }
    while i < le {
        f(Some(i), None);
        i += 1;
    }
    while j < re {
        f(None, Some(j));
        j += 1;
    }
}

fn level_round(lv: &Level, w: &SplitEq) -> Acc3 {
    let pairs = lv.len.len() / 2;
    let row = |m: usize| -> Acc3 {
        let mut s = Acc3::default();
        for_each_merged(lv, m, |li, ri| {
            let (_, a0, a1, v0, v1, _, _) = merged_entry(lv, li, ri);
            s.q0 += a0.clmul(v0);
            s.q1 += a1.clmul(v1);
            s.qinf += (a0 + a1).clmul(v0 + v1);
        });
        let wm = w.at(m);
        let (q0, q1, qinf) = s.reduce();
        Acc3 {
            q0: wm.clmul(q0),
            q1: wm.clmul(q1),
            qinf: wm.clmul(qinf),
        }
    };
    if pairs * (1 << lv.level) >= PAR && pairs > 1 {
        let min = (1usize << 11 >> lv.level).max(1);
        (0..pairs)
            .into_par_iter()
            .with_min_len(min)
            .map(row)
            .reduce(Acc3::default, Acc3::add)
    } else {
        (0..pairs).map(row).fold(Acc3::default(), Acc3::add)
    }
}

fn level_bind(lv: Level, r: K) -> Level {
    let pairs = lv.len.len() / 2;
    let total = lv.key.len();
    let cap = 2usize << lv.level;
    let mut out = Level {
        level: lv.level + 1,
        len: vec![0; pairs],
        key: vec![0; total],
        ra: vec![K::ZERO; total],
        val: vec![K::ZERO; total],
        prev: vec![K::ZERO; total],
        next: vec![K::ZERO; total],
    };
    let fill = |m: usize,
                len: &mut u32,
                key: &mut [u32],
                ra: &mut [K],
                val: &mut [K],
                prev: &mut [K],
                next: &mut [K]| {
        let mut o = 0;
        for_each_merged(&lv, m, |li, ri| {
            let (k, a0, a1, v0, v1, p, nx) = merged_entry(&lv, li, ri);
            key[o] = k;
            ra[o] = a0 + r * (a0 + a1);
            val[o] = v0 + r * (v0 + v1);
            prev[o] = p;
            next[o] = nx;
            o += 1;
        });
        *len = o as u32;
    };
    if total >= PAR && pairs > 1 {
        let min = (1usize << 11 >> lv.level).max(1);
        out.len
            .par_iter_mut()
            .zip(out.key.par_chunks_mut(cap))
            .zip(out.ra.par_chunks_mut(cap))
            .zip(out.val.par_chunks_mut(cap))
            .zip(out.prev.par_chunks_mut(cap))
            .zip(out.next.par_chunks_mut(cap))
            .enumerate()
            .with_min_len(min)
            .for_each(|(m, (((((len, key), ra), val), prev), next))| {
                fill(m, len, key, ra, val, prev, next)
            });
    } else {
        for (m, (((((len, key), ra), val), prev), next)) in out
            .len
            .iter_mut()
            .zip(out.key.chunks_mut(cap))
            .zip(out.ra.chunks_mut(cap))
            .zip(out.val.chunks_mut(cap))
            .zip(out.prev.chunks_mut(cap))
            .zip(out.next.chunks_mut(cap))
            .enumerate()
        {
            fill(m, len, key, ra, val, prev, next);
        }
    }
    out
}

fn prove_cycle_first(w: &ReadWitness<'_>, r_c: &[K], claim: K, tr: &mut Transcript) -> ReadResult {
    let n = r_c.len();
    let t = w.addr.len();
    let mut lv = Level {
        level: 0,
        len: vec![1; t],
        key: w.addr.to_vec(),
        ra: vec![K::ONE; t],
        val: w.rv_e.to_vec(),
        prev: w.rv_e.to_vec(),
        next: if t >= PAR {
            w.rv_e
                .par_iter()
                .zip(w.inc_e)
                .map(|(&v, &i)| v + i)
                .collect()
        } else {
            w.rv_e.iter().zip(w.inc_e).map(|(&v, &i)| v + i).collect()
        },
    };
    let mut cycle = Vec::with_capacity(n);
    let mut r_j = Vec::with_capacity(n);
    let mut cur = claim;
    let mut scale = K::ONE;
    for i in 0..n {
        let weights = SplitEq::new(&r_c[i + 1..]);
        let (q0, q1, qinf) = level_round(&lv, &weights).reduce();
        let (q0, q1, qinf) = (scale * q0, scale * q1, scale * qinf);
        if checks_enabled() {
            assert_eq!(
                (K::ONE + r_c[i]) * q0 + r_c[i] * q1,
                cur,
                "read cycle round {i}"
            );
        }
        let r = cycle_round(tr, [q0, qinf]);
        cycle.push([q0, qinf]);
        r_j.push(r);
        let e = eq_lin(r_c[i], r);
        cur = e * quad_at(q0, q1, qinf, r);
        scale *= e;
        lv = level_bind(lv, r);
    }
    let len = lv.len[0] as usize;
    lv.key.truncate(len);
    lv.ra.truncate(len);
    lv.val.truncate(len);
    let mut address = Vec::with_capacity(w.a);
    let mut r_k = Vec::with_capacity(w.a);
    let (_cur, ra, val) = sparse_address_rounds(
        lv.key,
        lv.ra,
        lv.val,
        w.m0_e,
        scale,
        cur,
        tr,
        &mut address,
        &mut r_k,
        w.a,
    );
    finish(tr, ra, val);
    ReadResult {
        proof: ReadProof {
            cycle,
            address,
            ra,
            val,
        },
        r_k,
        r_j,
    }
}

// ---------------------------------------------------------------------------
// Address-first prover (issue items 2 and 8)
// ---------------------------------------------------------------------------

/// Accesses grouped for the address rounds. Without load merging there is
/// one unit per cycle. With it (issue item 8), the cycles between two stores
/// (to any address) that access the same cell form one unit, whose weight is
/// the sum of their `eq(r_c, j)`. They read the same value in every address
/// round, because no store intervenes. A unit's store, if any, happens after
/// every read in its segment, so it is placed last in the segment.
struct Units {
    addr: Vec<u32>,
    weight: Vec<K>,
    inc: Vec<K>,
    store: Vec<bool>,
    /// The unit of each cycle (empty when units are cycles).
    of_cycle: Vec<u32>,
}

fn build_units(w: &ReadWitness<'_>, eq_c: Vec<K>, merge: bool) -> Units {
    let t = w.addr.len();
    if !merge {
        return Units {
            addr: w.addr.to_vec(),
            weight: eq_c,
            inc: w.inc_e.to_vec(),
            store: w.store.to_vec(),
            of_cycle: Vec::new(),
        };
    }
    // Segments end at (and include) each store.
    let mut ends: Vec<usize> = (0..t).filter(|&j| w.store[j]).map(|j| j + 1).collect();
    if ends.last() != Some(&t) {
        ends.push(t);
    }
    let starts: Vec<usize> = std::iter::once(0)
        .chain(ends.iter().copied())
        .take(ends.len())
        .collect();
    type Seg = (Vec<u32>, Vec<K>, Vec<K>, Vec<bool>, Vec<(usize, u32)>);
    let seg = |(s, e): (usize, usize)| -> Seg {
        let mut idx: Vec<usize> = (s..e).collect();
        let last_store = if w.store[e - 1] { Some(e - 1) } else { None };
        idx.sort_unstable_by_key(|&j| (w.addr[j], j));
        let (mut ad, mut wt, mut inc, mut st) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut map = Vec::with_capacity(e - s);
        let mut store_unit = None;
        for &j in &idx {
            if ad.last() != Some(&w.addr[j]) || ad.is_empty() {
                ad.push(w.addr[j]);
                wt.push(K::ZERO);
                inc.push(K::ZERO);
                st.push(false);
            }
            let u = ad.len() - 1;
            wt[u] += eq_c[j];
            if Some(j) == last_store {
                inc[u] = w.inc_e[j];
                st[u] = true;
                store_unit = Some(u);
            }
            map.push((j, u as u32));
        }
        if let Some(u) = store_unit {
            // Move the storing unit last.
            let last = ad.len() - 1;
            ad.swap(u, last);
            wt.swap(u, last);
            inc.swap(u, last);
            st.swap(u, last);
            for m in &mut map {
                if m.1 as usize == u {
                    m.1 = last as u32;
                } else if m.1 as usize == last {
                    m.1 = u as u32;
                }
            }
        }
        (ad, wt, inc, st, map)
    };
    let segs: Vec<Seg> = starts
        .into_par_iter()
        .zip(ends.into_par_iter())
        .with_min_len(64)
        .map(seg)
        .collect();
    let total: usize = segs.iter().map(|s| s.0.len()).sum();
    let mut units = Units {
        addr: Vec::with_capacity(total),
        weight: Vec::with_capacity(total),
        inc: Vec::with_capacity(total),
        store: Vec::with_capacity(total),
        of_cycle: vec![0; t],
    };
    for (ad, wt, inc, st, map) in segs {
        let base = units.addr.len() as u32;
        for (j, u) in map {
            units.of_cycle[j] = base + u;
        }
        units.addr.extend(ad);
        units.weight.extend(wt);
        units.inc.extend(inc);
        units.store.extend(st);
    }
    units
}

#[allow(clippy::needless_range_loop)]
fn prove_address_first(
    w: &ReadWitness<'_>,
    r_c: &[K],
    claim: K,
    merge_loads: bool,
    tr: &mut Transcript,
) -> ReadResult {
    let n = r_c.len();
    let a = w.a;
    let eq_c = SplitEq::new(r_c).table();
    let units = build_units(w, eq_c, merge_loads);
    let nu = units.addr.len();
    // Blocks of units, each with a checkpoint of the bound memory state at
    // its start. Their total size is capped so large memories stay serial.
    let threads = rayon::current_num_threads().max(1);
    let budget = 1usize << 24;
    let blocks = (4 * threads)
        .min((budget >> a).max(1))
        .min(nu.div_ceil(1 << 10))
        .max(1);
    let bounds: Vec<usize> = (0..=blocks).map(|b| nu * b / blocks).collect();
    let mut starts: Vec<Vec<K>> = Vec::with_capacity(blocks);
    {
        let mut state = w.m0_e.to_vec();
        for b in 0..blocks {
            starts.push(state.clone());
            for u in bounds[b]..bounds[b + 1] {
                if units.store[u] {
                    state[units.addr[u] as usize] += units.inc[u];
                }
            }
        }
    }
    let mut c = vec![K::ONE; nu];
    let mut address = Vec::with_capacity(a);
    let mut r_k = Vec::with_capacity(a);
    let mut cur = claim;
    for s in 0..a {
        let block = |b: usize, start: &Vec<K>| -> Acc3 {
            let mut state = start.clone();
            let mut acc = Acc3::default();
            for u in bounds[b]..bounds[b + 1] {
                let k = (units.addr[u] >> s) as usize;
                let t = k & !1;
                let (v0, v1) = (state[t], state[t + 1]);
                let cu = c[u];
                let wc = units.weight[u] * cu;
                if k & 1 == 0 {
                    acc.q0 += wc.clmul(v0);
                } else {
                    acc.q1 += wc.clmul(v1);
                }
                acc.qinf += wc.clmul(v0 + v1);
                if units.store[u] {
                    state[k] += cu * units.inc[u];
                }
            }
            acc
        };
        let acc = starts
            .par_iter()
            .enumerate()
            .map(|(b, st)| block(b, st))
            .reduce(Acc3::default, Acc3::add);
        let (g0, g1, ginf) = acc.reduce();
        if checks_enabled() {
            assert_eq!(g0 + g1, cur, "read address round {s}");
        }
        let r = address_round(tr, [g0, ginf]);
        address.push([g0, ginf]);
        r_k.push(r);
        cur = quad_at(g0, g1, ginf, r);
        let (e0, e1) = (K::ONE + r, r);
        c.par_iter_mut()
            .zip(units.addr.par_iter())
            .with_min_len(1 << 12)
            .for_each(|(cu, &ad)| {
                *cu *= if (ad >> s) & 1 == 0 { e0 } else { e1 };
            });
        starts.par_iter_mut().for_each(|st| *st = bind_dense(st, r));
    }
    // Values per unit at r_k: a running scalar per block.
    let mut val_u = vec![K::ZERO; nu];
    {
        let mut chunks: Vec<&mut [K]> = Vec::with_capacity(blocks);
        let mut rest = val_u.as_mut_slice();
        for b in 0..blocks {
            let (head, tail) = rest.split_at_mut(bounds[b + 1] - bounds[b]);
            chunks.push(head);
            rest = tail;
        }
        chunks.into_par_iter().enumerate().for_each(|(b, out)| {
            let mut v = starts[b][0];
            for (o, u) in out.iter_mut().zip(bounds[b]..bounds[b + 1]) {
                *o = v;
                if units.store[u] {
                    v += c[u] * units.inc[u];
                }
            }
        });
    }
    let (ra_j, val_j) = if units.of_cycle.is_empty() {
        (c, val_u)
    } else {
        let map = &units.of_cycle;
        (
            map.par_iter().map(|&u| c[u as usize]).collect(),
            map.par_iter().map(|&u| val_u[u as usize]).collect(),
        )
    };
    let mut cycle = Vec::with_capacity(n);
    let mut r_j = Vec::with_capacity(n);
    let (_cur, ra, val) = dense_cycle_rounds(ra_j, val_j, r_c, cur, tr, &mut cycle, &mut r_j);
    finish(tr, ra, val);
    ReadResult {
        proof: ReadProof {
            cycle,
            address,
            ra,
            val,
        },
        r_k,
        r_j,
    }
}

pub(crate) fn prove(
    w: &ReadWitness<'_>,
    r_c: &[K],
    claim: K,
    order: ReadOrder,
    merge_loads: bool,
    tr: &mut Transcript,
) -> ReadResult {
    init(tr, claim, r_c.len(), w.a, order);
    match order {
        ReadOrder::CycleFirst => prove_cycle_first(w, r_c, claim, tr),
        ReadOrder::AddressFirst => prove_address_first(w, r_c, claim, merge_loads, tr),
    }
}
