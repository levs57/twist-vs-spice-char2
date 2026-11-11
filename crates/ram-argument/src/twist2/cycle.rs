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

use super::util::{PAR, SplitEq, SplitLt, checks_enabled, emit_round, node};

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

/// Round values `g(t_0..t_D)` of the summand over pairs of the bound tables.
///
/// Table order: `tl = LT + γ`, `ed = δ eq(r_j, ·)`, `inc`, `A_1..A_d`,
/// `Ã_1..Ã_d` where `Ã_i = ε_i' A_i(1 + r_k^i, ·)`.
#[allow(clippy::needless_range_loop)]
fn round_values(t: &Tables, degree: usize) -> Vec<K> {
    let d = t.a.len();
    let half = t.inc.len() / 2;
    let pair = |m: usize, high: &mut [U256], low: &mut [U256; 4], av: &mut [K]| {
        let at = |tab: &[K], e: usize| -> K {
            let (f0, f1) = (tab[2 * m], tab[2 * m + 1]);
            match e {
                0 => f0,
                1 => f1,
                _ => f0 + (f0 + f1).mul_t(e as u128),
            }
        };
        for e in 0..=degree {
            let mut prod = at(&t.a[0], e);
            if e < 4 {
                av[0] = prod;
            }
            for i in 1..d {
                let x = at(&t.a[i], e);
                if e < 4 {
                    av[i] = x;
                }
                prod *= x;
            }
            let lin = at(&t.tl, e) * at(&t.inc, e) + at(&t.ed, e);
            high[e] += lin.clmul(prod);
            if e < 4 {
                let mut mir = U256::ZERO;
                for i in 0..d {
                    mir += av[i].clmul(at(&t.am[i], e));
                }
                low[e] += at(&t.ed, e).clmul(mir.reduce());
            }
        }
    };
    let fold = |range: std::ops::Range<usize>| -> (Vec<U256>, [U256; 4]) {
        let mut high = vec![U256::ZERO; degree + 1];
        let mut low = [U256::ZERO; 4];
        let mut av = vec![K::ZERO; d];
        for m in range {
            pair(m, &mut high, &mut low, &mut av);
        }
        (high, low)
    };
    let merge = |mut x: (Vec<U256>, [U256; 4]), y: (Vec<U256>, [U256; 4])| {
        for (a, b) in x.0.iter_mut().zip(y.0) {
            *a += b;
        }
        for (a, b) in x.1.iter_mut().zip(y.1) {
            *a += b;
        }
        x
    };
    let (high, low) = if half >= PAR {
        let chunk = 1 << 10;
        (0..half.div_ceil(chunk))
            .into_par_iter()
            .map(|ch| fold(ch * chunk..((ch + 1) * chunk).min(half)))
            .reduce(|| (vec![U256::ZERO; degree + 1], [U256::ZERO; 4]), merge)
    } else {
        fold(0..half)
    };
    let low: Vec<K> = low.iter().map(|x| x.reduce()).collect();
    high.into_iter()
        .enumerate()
        .map(|(e, h)| {
            let l = if e < 4 {
                low[e]
            } else {
                interpolate_eval(&low, node(e))
            };
            h.reduce() + l
        })
        .collect()
}

pub(crate) struct Tables {
    pub(crate) tl: Vec<K>,
    pub(crate) ed: Vec<K>,
    pub(crate) inc: Vec<K>,
    pub(crate) a: Vec<Vec<K>>,
    pub(crate) am: Vec<Vec<K>>,
}

impl Tables {
    fn bind(&mut self, r: K) {
        let bind = |t: &mut Vec<K>| {
            let half = t.len() / 2;
            let out: Vec<K> = if half >= PAR {
                (0..half)
                    .into_par_iter()
                    .with_min_len(1 << 10)
                    .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
                    .collect()
            } else {
                (0..half)
                    .map(|m| t[2 * m] + r * (t[2 * m] + t[2 * m + 1]))
                    .collect()
            };
            *t = out;
        };
        let mut all: Vec<&mut Vec<K>> = vec![&mut self.tl, &mut self.ed, &mut self.inc];
        all.extend(self.a.iter_mut());
        all.extend(self.am.iter_mut());
        all.into_par_iter().for_each(bind);
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
    let lookup = |tab: Vec<K>, i: usize| -> Vec<K> {
        let shift = i * c;
        w.addr
            .par_iter()
            .with_min_len(1 << 12)
            .map(|&ad| tab[((ad >> shift) & mask) as usize])
            .collect()
    };
    let a: Vec<Vec<K>> = (0..d)
        .map(|i| lookup(eq_table(chunk_point(r_k, c, i)), i))
        .collect();
    let am: Vec<Vec<K>> = (0..d)
        .map(|i| {
            let tab: Vec<K> = eq_table(&mirrored(chunk_point(r_k, c, i)))
                .into_iter()
                .map(|x| x * eps_scaled[i])
                .collect();
            lookup(tab, i)
        })
        .collect();
    let lt = SplitLt::new(r_j);
    let eq = SplitEq::new(r_j);
    let size = 1usize << n;
    let tl: Vec<K> = (0..size)
        .into_par_iter()
        .with_min_len(1 << 12)
        .map(|j| lt.at(j) + ch.gamma)
        .collect();
    let ed: Vec<K> = (0..size)
        .into_par_iter()
        .with_min_len(1 << 12)
        .map(|j| ch.delta * eq.at(j))
        .collect();
    let tables = Tables {
        tl,
        ed,
        inc: w.inc_e.to_vec(),
        a,
        am,
    };
    drop(tables_span);
    let _span = om_transcript::profiling::Span::new("twist.cycle.rounds");
    prove_tables(tables, &eps_scaled, claim, tr)
}

/// The sum-check on already built tables; `eps_scaled[i]` is the factor
/// folded into `Ã_i`.
pub(crate) fn prove_tables(
    mut tables: Tables,
    eps_scaled: &[K],
    claim: K,
    tr: &mut Transcript,
) -> CycleResult {
    let n = tables.inc.len().trailing_zeros() as usize;
    let degree = tables.a.len() + 2;
    sumcheck_init(tr, LABEL, claim, n, degree);
    let mut msgs = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut cur = claim;
    for _ in 0..n {
        let ys = round_values(&tables, degree);
        if checks_enabled() {
            assert_eq!(ys[0] + ys[1], cur, "cycle check round sum");
        }
        let (r, next) = emit_round(&ys, tr, LABEL, &mut msgs);
        cur = next;
        z.push(r);
        tables.bind(r);
    }
    let a_z: Vec<K> = tables.a.iter().map(|t| t[0]).collect();
    let am_z: Vec<K> = tables
        .am
        .iter()
        .zip(eps_scaled)
        .map(|(t, &e)| t[0] * e.inverse())
        .collect();
    let inc_z = tables.inc[0];
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
