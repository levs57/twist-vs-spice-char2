//! BinaryTwist (spec 07 §7.5–7.7). Mirrors
//! `reference/openmagnetar/twist.py` byte for byte in its transcript
//! interaction.
//!
//! Address component tables: see [`crate::onehot`]. Inc table: index
//! `o + 2^ell * j`. Memory `M0` table: index `k + 2^a * o` (`k` low, C2).
//! Words are `u64`s (`2^ell <= 64` bits used, `ell <= 6` in v1).

use om_field::{K, U256};
use om_sumcheck::{
    PackedBits, SumcheckProof, VerificationError, eq_eval, eq_table, interpolate_eval,
    mle_eval_bits, sumcheck_init, sumcheck_verify,
};
use om_transcript::Transcript;
use rayon::prelude::*;

use crate::eq_factor::{
    batch_inverse_one_plus, eval_coefficients, factored_evaluations, reconstruct_cofactor,
    scale_low_weights,
};
use crate::lazy::{
    SplitEq, SplitLt, WordEval, eq_weighted_sum, sumcheck_prove_rows_mul2,
    sumcheck_prove_rows_products,
};
use crate::onehot::PAR_THRESHOLD;
use crate::onehot::{
    AddrComps, EvalClaim, OneHotProof, VirtGroup, VirtProof, VirtTerm, VirtualizationMode, bits_of,
    onehot_prove, onehot_verify, virt_prove, virt_verify,
};
use crate::sublabel;
use crate::virt_flat::{VirtFlatProof, virt_flat_prove, virt_flat_verify};
use crate::virt_tree::{VirtTreeProof, virt_tree_prove, virt_tree_verify};

/// Parameters of one BinaryTwist instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistParams {
    pub name: String,
    /// Memory has `2^a` cells.
    pub a: usize,
    /// Number of address components, `d | a`.
    pub d: usize,
    /// Reads per cycle (then one write).
    pub p: usize,
    /// `T = 2^n` cycles.
    pub n: usize,
    /// Words have `2^ell` bits (default 6 in Python).
    pub ell: usize,
}

impl TwistParams {
    /// Parameters with the Python default `ell = 6`.
    pub fn new(name: &str, a: usize, d: usize, p: usize, n: usize) -> Self {
        TwistParams {
            name: name.to_string(),
            a,
            d,
            p,
            n,
            ell: 6,
        }
    }
}

/// The input claims of one BinaryTwist instance, all at the common
/// value point `r_o` and cycle point `r_c`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistClaims {
    /// `K^ell`.
    pub r_o: Vec<K>,
    /// `K^n`.
    pub r_c: Vec<K>,
    /// `p` claimed values `rv_s~(r_o, r_c)`.
    pub rv: Vec<K>,
    /// `wv~(r_o, r_c)`.
    pub wv: K,
    /// `bits[x][t]`: address bit `t` of address `x` (`x < p` reads,
    /// `x = p` write) at `r_c`.
    pub bits: Vec<Vec<K>>,
}

/// Final-memory claim spec (spec 07 §7.6, C6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalMem {
    /// Output region has `2^m` words.
    pub m: usize,
    /// High `(a - m)` bits of the region's first word index.
    pub s_out: usize,
    /// `2^m` public output words.
    pub words: Vec<u64>,
}

/// The committed witness of one BinaryTwist instance, in the compact
/// form of spec 07 §7.10: the address components as one address per
/// cycle (or arbitrary bit tables, for cheating-prover tests), and the
/// `Inc` table as one word per cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistWitness {
    /// `ra[s]`: the `d` components of read `s`.
    pub ra: Vec<AddrComps>,
    /// The `d` components of the write address.
    pub wa: AddrComps,
    /// `inc[j]`: the increment word of cycle `j`; the committed `Inc`
    /// table is bit `o` of `inc[j]` at index `o + 2^ell * j`, `o <
    /// 2^ell` (higher bits are ignored).
    pub inc: Vec<u64>,
}

impl TwistWitness {
    /// The `Inc` bit table, index `o + 2^ell * j` (as [`word_table`]).
    pub fn inc_table(&self, ell: usize) -> Vec<bool> {
        word_table(&self.inc, ell)
    }

    /// The `Inc` bit table, packed.
    pub fn inc_packed(&self, ell: usize) -> PackedBits {
        if ell == 6 {
            return PackedBits::from_words(self.inc.clone(), self.inc.len() * 64);
        }
        let l = 1usize << ell;
        let mut out = PackedBits::zero(self.inc.len() * l);
        for (j, &w) in self.inc.iter().enumerate() {
            let mut w = if l >= 64 { w } else { w & ((1u64 << l) - 1) };
            while w != 0 {
                let o = w.trailing_zeros() as usize;
                out.set(o + l * j);
                w &= w - 1;
            }
        }
        out
    }
}

/// Proof of one BinaryTwist instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistProof {
    pub rw: SumcheckProof,
    /// `[ra_0..ra_{p-1}, wa, V, I]`.
    pub rw_evals: Vec<K>,
    pub val: SumcheckProof,
    /// `[wa~(r_k, s), Inc~(r_o, s)]`.
    pub val_evals: Vec<K>,
    pub fin: Option<SumcheckProof>,
    pub fin_evals: Option<Vec<K>>,
    pub inc_extra: Option<K>,
    pub virt: Option<VirtProof>,
    pub virt_tree: Option<VirtTreeProof>,
    pub virt_flat: Option<VirtFlatProof>,
    pub onehot: OneHotProof,
}

/// Output of one BinaryTwist instance: the evaluation claims on
/// committed polynomials (spec 07 §7.5 step 9, in order) and the
/// optional Inc-at-a-random-point claim (§7.7, also the last entry of
/// `claims` when present).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistOutput {
    pub claims: Vec<EvalClaim>,
    pub inc_extra: Option<EvalClaim>,
}

/// The honest values computed by [`build_witness`] (Python's `info`
/// dict).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwistInfo {
    /// `rv[s][j]`: word read by read `s` in cycle `j`.
    pub rv: Vec<Vec<u64>>,
    /// `wv[j]`: word written in cycle `j`.
    pub wv: Vec<u64>,
    /// `inc[j]`: XOR of old and new content of the written cell.
    pub inc: Vec<u64>,
    /// Memory state after the last cycle (Python key `"final"`).
    pub final_state: Vec<u64>,
    /// `reads[s][j]`: address of read `s` in cycle `j`.
    pub reads: Vec<Vec<usize>>,
    /// `writes[j]`: address written in cycle `j`.
    pub writes: Vec<usize>,
}

// ----------------------------------------------------------------- helpers

/// MLE of `LT(x, y) = [x < y]` (C2 order, `x_1` = LSB): `O(n)` field
/// operations. `LT~(x,y) = Σ_i (1 + x_i) y_i Π_{m > i} (1 + x_m + y_m)`.
///
/// # Panics
/// Panics if `x` and `y` have different lengths (indexing, as in
/// Python's `range(len(x))` against `y[i]`).
pub fn lt_eval(x: &[K], y: &[K]) -> K {
    let (mut acc, mut suffix) = (K::ZERO, K::ONE);
    for i in (0..x.len()).rev() {
        acc += (K::ONE + x[i]) * y[i] * suffix;
        suffix *= K::ONE + x[i] + y[i];
    }
    acc
}

/// The table `j ↦ LT~(bits(j), y)` of length `2^len(y)`.
pub fn lt_table(y: &[K]) -> Vec<K> {
    let n = y.len();
    let size = 1usize << n;
    if size >= PAR_THRESHOLD {
        (0..size)
            .into_par_iter()
            .map(|j| lt_eval(&bits_of(j as u64, n), y))
            .collect()
    } else {
        (0..size)
            .map(|j| lt_eval(&bits_of(j as u64, n), y))
            .collect()
    }
}

/// All component names in one-hot order: reads `0..p-1` each with
/// components `1..d`, then the write components `1..d`.
pub fn comp_names_tw(params: &TwistParams) -> Vec<String> {
    let mut nm = Vec::with_capacity((params.p + 1) * params.d);
    for s in 0..params.p {
        nm.extend((0..params.d).map(|i| format!("{}/ra{}/{}", params.name, s, i + 1)));
    }
    nm.extend((0..params.d).map(|i| format!("{}/wa/{}", params.name, i + 1)));
    nm
}

/// Bit table of a list of words, index `o + 2^ell * j` (bit `o` of
/// `words[j]`).
pub fn word_table(words: &[u64], ell: usize) -> Vec<bool> {
    let l = 1usize << ell;
    let total = words.len() * l;
    let at = |idx: usize| (words[idx / l] >> (idx % l)) & 1 == 1;
    if total >= PAR_THRESHOLD {
        (0..total).into_par_iter().map(at).collect()
    } else {
        (0..total).map(at).collect()
    }
}

/// From an access trace (`reads[s][j]`, `writes[j]`, `new_words[j]`)
/// build the witness and the honest values, replaying memory starting
/// from `m0` (one word per cell).
pub fn build_witness(
    params: &TwistParams,
    m0: &[u64],
    reads: &[Vec<usize>],
    writes: &[usize],
    new_words: &[u64],
) -> (TwistWitness, TwistInfo) {
    let (p, n) = (params.p, params.n);
    let t = 1usize << n;
    let mut state = m0.to_vec();
    let mut rv = vec![vec![0u64; t]; p];
    let (mut wv, mut inc) = (vec![0u64; t], vec![0u64; t]);
    for j in 0..t {
        for s in 0..p {
            rv[s][j] = state[reads[s][j]];
        }
        let old = state[writes[j]];
        inc[j] = old ^ new_words[j];
        state[writes[j]] = new_words[j];
        wv[j] = new_words[j];
    }
    let to32 = |v: &[usize]| -> Vec<u32> {
        v.iter()
            .map(|&x| u32::try_from(x).expect("address fits in u32"))
            .collect()
    };
    let wit = build_witness_lean(
        params,
        m0,
        reads.iter().map(|r| to32(r)).collect(),
        to32(writes),
        new_words,
    );
    debug_assert_eq!(wit.inc, inc);
    let info = TwistInfo {
        rv,
        wv,
        inc,
        final_state: state,
        reads: reads.to_vec(),
        writes: writes.to_vec(),
    };
    (wit, info)
}

/// The witness alone (no honest-value side tables): what the prover
/// commits to. `reads[s][j]` and `writes[j]` are word addresses below
/// `2^a`, moved into the witness.
///
/// # Panics
/// Panics if a sequence does not have `2^n` entries or an address is
/// out of range.
pub fn build_witness_lean(
    params: &TwistParams,
    m0: &[u64],
    reads: Vec<Vec<u32>>,
    writes: Vec<u32>,
    new_words: &[u64],
) -> TwistWitness {
    let (a, d, p, n) = (params.a, params.d, params.p, params.n);
    let t = 1usize << n;
    assert!(
        reads.len() == p && reads.iter().all(|r| r.len() == t),
        "build_witness: need p read sequences of 2^n addresses"
    );
    assert!(
        writes.len() == t && new_words.len() == t,
        "build_witness: need 2^n writes"
    );
    let mut state = m0.to_vec();
    let mut inc = vec![0u64; t];
    for j in 0..t {
        let k = writes[j] as usize;
        inc[j] = state[k] ^ new_words[j];
        state[k] = new_words[j];
    }
    TwistWitness {
        ra: reads
            .into_iter()
            .map(|r| AddrComps::from_addrs(r, a, d))
            .collect(),
        wa: AddrComps::from_addrs(writes, a, d),
        inc,
    }
}

/// `MLE(word_table(words, ell))(r_o ‖ r_c)` (`ell = len(r_o)`), in
/// `O(T)` without the `2^ell · T` table.
pub fn word_mle_eval(words: &[u64], r_o: &[K], r_c: &[K]) -> K {
    assert_eq!(words.len(), 1usize << r_c.len(), "word_mle_eval: length");
    let we = WordEval::new(r_o);
    eq_weighted_sum(r_c, &|j| we.eval(words[j]))
}

/// The address-bit claims `bits[x][t] = f_t~(r_c)` of every access
/// (`x < p` reads, `x = p` the write), as in [`honest_claims`].
pub fn addr_bit_claims(params: &TwistParams, wit: &TwistWitness, r_c: &[K]) -> Vec<Vec<K>> {
    wit.ra
        .iter()
        .chain(std::iter::once(&wit.wa))
        .map(|c| c.bit_evals(params.a, r_c))
        .collect()
}

/// The honest input claims at `(r_o, r_c)` for the values in `info`.
pub fn honest_claims(params: &TwistParams, info: &TwistInfo, r_o: &[K], r_c: &[K]) -> TwistClaims {
    let (a, p) = (params.a, params.p);
    let rv: Vec<K> = (0..p)
        .map(|s| word_mle_eval(&info.rv[s], r_o, r_c))
        .collect();
    let wv = word_mle_eval(&info.wv, r_o, r_c);
    let mut addrs: Vec<&Vec<usize>> = info.reads.iter().collect();
    addrs.push(&info.writes);
    let bits: Vec<Vec<K>> = (0..=p)
        .map(|x| crate::lazy::addr_bit_evals(a, r_c, &|j| addrs[x][j]))
        .collect();
    TwistClaims {
        r_o: r_o.to_vec(),
        r_c: r_c.to_vec(),
        rv,
        wv,
        bits,
    }
}

/// The `M_0` bit table, index `k + 2^a * o` (bit `o` of `m0[k]`).
pub fn m0_table(m0: &[u64], a: usize, ell: usize) -> Vec<bool> {
    let rows = m0.len().min(1usize << a);
    if rows == 0 {
        return Vec::new();
    }
    let total = rows << ell;
    let at = |idx: usize| (m0[idx % rows] >> (idx / rows)) & 1 == 1;
    if total >= PAR_THRESHOLD {
        (0..total).into_par_iter().map(at).collect()
    } else {
        (0..total).map(at).collect()
    }
}

/// The output-region bit table, index `k_low + 2^m * o` (bit `o` of
/// `words[k_low]`), where `2^m = words.len()`.
pub fn out_table(words: &[u64], ell: usize) -> Vec<bool> {
    if words.is_empty() {
        return Vec::new();
    }
    let total = words.len() << ell;
    let at = |idx: usize| (words[idx % words.len()] >> (idx / words.len())) & 1 == 1;
    if total >= PAR_THRESHOLD {
        (0..total).into_par_iter().map(at).collect()
    } else {
        (0..total).map(at).collect()
    }
}

fn absorb(tr: &mut Transcript, label: &[u8], c: &TwistClaims) {
    tr.absorb_ks(&sublabel(label, b"/point_o"), &c.r_o);
    tr.absorb_ks(&sublabel(label, b"/point_c"), &c.r_c);
    tr.absorb_ks(&sublabel(label, b"/rv"), &c.rv);
    tr.absorb_k(&sublabel(label, b"/wv"), c.wv);
    let flat: Vec<K> = c.bits.iter().flatten().copied().collect();
    tr.absorb_ks(&sublabel(label, b"/bits"), &flat);
}

fn absorb_virtualization_mode(tr: &mut Transcript, label: &[u8], mode: VirtualizationMode) {
    match mode {
        VirtualizationMode::FlatLegacy => {}
        VirtualizationMode::GkrTree => {
            tr.absorb_bytes(&sublabel(label, b"/virtualization-mode"), b"gkr-tree/v1")
        }
        VirtualizationMode::Flat => tr.absorb_bytes(
            &sublabel(label, b"/virtualization-mode"),
            b"flat/common-cycle/v1",
        ),
    }
}

fn coeffs(tr: &mut Transcript, label: &[u8], p: usize, a: usize) -> (Vec<K>, Vec<Vec<K>>, Vec<K>) {
    let (mut alpha0, mut alphas) = (Vec::with_capacity(p + 1), Vec::with_capacity(p + 1));
    for _ in 0..=p {
        alpha0.push(tr.challenge_nonzero_k(&sublabel(label, b"/alpha0")));
        alphas.push(tr.challenge_ks(&sublabel(label, b"/alpha"), a));
    }
    let beta = tr.challenge_ks(&sublabel(label, b"/beta"), p + 1);
    (alpha0, alphas, beta)
}

/// `Φ(k) = alpha0 + Σ_t alphas[t] k_t` as a table over Boolean `k`.
fn phi_k(alpha0: K, alphas: &[K], a: usize) -> Vec<K> {
    let nk = 1usize << a;
    let mut table = vec![alpha0; nk];
    for bit in 0..a {
        let alpha = alphas.get(bit).copied().unwrap_or(K::ZERO);
        let half = 1usize << bit;
        let (low, high) = table[..2 * half].split_at_mut(half);
        if half >= PAR_THRESHOLD {
            high.par_iter_mut()
                .zip(low.par_iter())
                .for_each(|(dst, &src)| *dst = src + alpha);
        } else {
            for (dst, &src) in high.iter_mut().zip(low.iter()) {
                *dst = src + alpha;
            }
        }
    }
    table
}

/// Table layout: `[eq, V, I, ra_0..ra_p, phi_0..phi_p]`.
#[cfg(test)]
fn rw_combine(p: usize, beta: &[K], v: &[K]) -> K {
    let (eq, big_v, big_i) = (v[0], v[1], v[2]);
    let ra = &v[3..4 + p];
    let phi = &v[4 + p..];
    let mut acc = K::ZERO;
    for x in 0..=p {
        acc += beta[x] * ra[x] * (big_v + phi[x]);
    }
    acc += beta[p] * ra[p] * big_i;
    eq * acc
}

fn s_out_bits(final_mem: &FinalMem, a: usize) -> Vec<K> {
    bits_of(final_mem.s_out as u64, a - final_mem.m)
}

#[allow(clippy::too_many_arguments)]
fn groups(
    params: &TwistParams,
    has_final: bool,
    r_k: &[K],
    r_j: &[K],
    rw_ev: &[K],
    s_v: &[K],
    val_ev: &[K],
    r_kf: Option<&[K]>,
    s_f: Option<&[K]>,
    fin_ev: Option<&[K]>,
) -> Vec<VirtGroup> {
    let (p, d) = (params.p, params.d);
    let nm = comp_names_tw(params);
    let mut groups = Vec::new();
    for s in 0..p {
        groups.push(VirtGroup {
            names: nm[s * d..(s + 1) * d].to_vec(),
            r_k: r_k.to_vec(),
            terms: vec![VirtTerm {
                cyc: r_j.to_vec(),
                value: rw_ev[s],
            }],
        });
    }
    groups.push(VirtGroup {
        names: nm[p * d..].to_vec(),
        r_k: r_k.to_vec(),
        terms: vec![
            VirtTerm {
                cyc: r_j.to_vec(),
                value: rw_ev[p],
            },
            VirtTerm {
                cyc: s_v.to_vec(),
                value: val_ev[0],
            },
        ],
    });
    if has_final {
        groups.push(VirtGroup {
            names: nm[p * d..].to_vec(),
            r_k: r_kf.expect("final group needs r_kf").to_vec(),
            terms: vec![VirtTerm {
                cyc: s_f.expect("final group needs s_f").to_vec(),
                value: fin_ev.expect("final group needs fin_evals")[0],
            }],
        });
    }
    groups
}

#[allow(clippy::too_many_arguments)]
fn inc_claims(
    params: &TwistParams,
    claims: &TwistClaims,
    r_j: &[K],
    rw_ev: &[K],
    s_v: &[K],
    val_ev: &[K],
    r_of: Option<&[K]>,
    s_f: Option<&[K]>,
    fin_ev: Option<&[K]>,
) -> Vec<EvalClaim> {
    let nm = format!("{}/inc", params.name);
    let pt = |pre: &[K], post: &[K]| {
        let mut v = pre.to_vec();
        v.extend_from_slice(post);
        v
    };
    let mut out = vec![
        EvalClaim {
            poly: nm.clone(),
            point: pt(&claims.r_o, r_j),
            value: *rw_ev.last().expect("rw_evals nonempty"),
        },
        EvalClaim {
            poly: nm.clone(),
            point: pt(&claims.r_o, s_v),
            value: val_ev[1],
        },
    ];
    if let Some(fin_ev) = fin_ev {
        out.push(EvalClaim {
            poly: nm,
            point: pt(
                r_of.expect("final inc claim needs r_of"),
                s_f.expect("final inc claim needs s_f"),
            ),
            value: fin_ev[1],
        });
    }
    out
}

// -----------------------------------------------------------------
// Sparse rw prover (spec 07 §7.10, informative). Same transcript and
// proof bytes as the dense definition in §7.5; only the prover's
// internal representation differs.
//
// The dense prover materialises tables of size `T * 2^a`. Here the
// first `n` sum-check rounds (cycle variables) work on one sparse
// group per remaining cycle suffix. A group stores, per access `x`,
// the sparse support of the bound one-hot table `ra_x` (at most one
// address per original cycle in the group) and the scalar
// `A = sum_k ra_x(k) V(k)`, where `V` is the bound value table;
// `sum_k ra_x(k) Phi_x(k)` and `sum_k ra_x(k)` are recomputed from the
// support. Binding two sibling groups needs the cross term
// `sum_k ra_L(k) V_R(k) + ra_R(k) V_L(k)`, where `V` of a group at an
// arbitrary address is recomputed from that address's write list:
// the state of a cell is constant between its writes, and the
// eq-weight of a constant segment is a difference of prefix sums of
// the bound cycle eq table. Round 1 reads the per-cycle data directly
// (no groups are built); later rounds keep the groups in flat CSR
// arrays. After the cycle rounds, one group remains and the last `a`
// rounds run densely on tables of size `2^a` only.

/// Per-address value-change lists (CSR, group order) with the eq-weighted average
/// value `v_avg` of the cell on that group and the value `v_after` of the
/// cell right after that group's last cycle. Bound in-place after each cycle
/// sum-check round so `v_group` is always a single binary search with zero
/// inner loops over writes.
struct WriteLog {
    off: Vec<usize>,
    len: Vec<usize>,
    active: Vec<usize>,
    grp: Vec<u32>,
    v_avg: Vec<K>,
    v_after: Vec<K>,
    /// `Val~` of `M_0` per cell.
    m0val: Vec<K>,
}

impl WriteLog {
    /// Builds the initial (cycle-level) write log along with `(aval0, inc_val)`:
    /// `aval0[j * np + x]` is the `Val~` of cell `addrs[x][j]` just before
    /// cycle `j`, and `inc_val[j] = we.eval(inc[j])`.
    fn new(m0: &[u64], addrs: &[&[u32]], inc: &[u64], we: &WordEval) -> (Self, Vec<K>, Vec<K>) {
        let nk = m0.len();
        let np = addrs.len();
        let writes = addrs[np - 1];
        let mut off = vec![0usize; nk + 1];
        // Unchanged writes remain proof accesses but need no historical value segment.
        for (j, &k) in writes.iter().enumerate() {
            if inc[j] != 0 {
                off[k as usize + 1] += 1;
            }
        }
        let len: Vec<usize> = (0..nk).map(|k| off[k + 1]).collect();
        let active = len
            .iter()
            .enumerate()
            .filter_map(|(k, &count)| (count != 0).then_some(k))
            .collect();
        for k in 0..nk {
            off[k + 1] += off[k];
        }
        let eval = |w: &u64| we.eval(*w);
        let (inc_val, m0val): (Vec<K>, Vec<K>) = if writes.len() + nk >= PAR_THRESHOLD {
            (
                inc.par_iter().map(eval).collect(),
                m0.par_iter().map(eval).collect(),
            )
        } else {
            (
                inc.iter().map(eval).collect(),
                m0.iter().map(eval).collect(),
            )
        };
        let mut pos = off[..nk].to_vec();
        let changes = off[nk];
        let mut grp = vec![0u32; changes];
        let mut v_avg = vec![K::ZERO; changes];
        let mut v_after = vec![K::ZERO; changes];
        let mut aval0 = vec![K::ZERO; writes.len() * np];
        let mut cur = m0val.clone();
        for (j, &k_u32) in writes.iter().enumerate() {
            let base_a = j * np;
            for (x, ad) in addrs.iter().enumerate() {
                aval0[base_a + x] = cur[ad[j] as usize];
            }
            let k = k_u32 as usize;
            let before = cur[k];
            let after = before + inc_val[j];
            cur[k] = after;
            if inc[j] != 0 {
                let idx = pos[k];
                grp[idx] = u32::try_from(j).expect("T fits in u32");
                v_avg[idx] = before;
                v_after[idx] = after;
                pos[k] = idx + 1;
            }
        }
        (
            WriteLog {
                off,
                len,
                active,
                grp,
                v_avg,
                v_after,
                m0val,
            },
            aval0,
            inc_val,
        )
    }

    /// `V~(k)` of group `g` at the current cycle-binding round.
    #[inline]
    fn v_group(&self, k: usize, g: usize) -> K {
        let lo = self.off[k];
        let hi = lo + self.len[k];
        let g32 = g as u32;
        let idx = self.grp[lo..hi].partition_point(|&c| c < g32);
        if lo + idx < hi && self.grp[lo + idx] == g32 {
            self.v_avg[lo + idx]
        } else if idx == 0 {
            self.m0val[k]
        } else {
            self.v_after[lo + idx - 1]
        }
    }

    /// Bind the low cycle variable of all write segments with challenge `chal`.
    fn bind(&mut self, chal: K) {
        for &k in &self.active {
            let lo = self.off[k];
            let n_k = self.len[k];
            if n_k == 0 {
                continue;
            }
            let mut prev_after = self.m0val[k];
            let mut r_idx = 0usize;
            let mut w_idx = 0usize;
            while r_idx < n_k {
                let g0 = self.grp[lo + r_idx];
                let m = g0 >> 1;
                let gl = m << 1;
                let gr = gl | 1;
                let (avg_l, after_l) = if g0 == gl {
                    let a = self.v_avg[lo + r_idx];
                    let f = self.v_after[lo + r_idx];
                    r_idx += 1;
                    (a, f)
                } else {
                    (prev_after, prev_after)
                };
                let (avg_r, after_r) = if r_idx < n_k && self.grp[lo + r_idx] == gr {
                    let a = self.v_avg[lo + r_idx];
                    let f = self.v_after[lo + r_idx];
                    r_idx += 1;
                    (a, f)
                } else {
                    (after_l, after_l)
                };
                self.grp[lo + w_idx] = m;
                self.v_avg[lo + w_idx] = avg_l + chal * (avg_l + avg_r);
                self.v_after[lo + w_idx] = after_r;
                prev_after = after_r;
                w_idx += 1;
            }
            self.len[k] = w_idx;
        }
    }
}

/// The remaining cycle groups of the rw sum-check.
enum RwView<'a> {
    /// Before the first binding: one group per cycle, read from the
    /// per-cycle data.
    Cycles {
        addrs: &'a [&'a [u32]],
        inc_val: Vec<K>,
        aval0: Vec<K>,
    },
    /// Bound groups: per group `inc`, `A_x` (`aval[g * (p+1) +
    /// x]`), and per access a CSR list of `(address, ra)` sorted by
    /// address.
    Csr {
        inc: Vec<K>,
        aval: Vec<K>,
        off: Vec<Vec<usize>>,
        keys: Vec<Vec<u32>>,
        vals: Vec<Vec<K>>,
    },
}

#[derive(Default)]
struct RwScratch {
    inc: Vec<K>,
    aval: Vec<K>,
    off: Vec<Vec<usize>>,
    keys: Vec<Vec<u32>>,
    vals: Vec<Vec<K>>,
}

impl RwScratch {
    fn recycle(&mut self, view: RwView<'_>) {
        if let RwView::Csr {
            inc,
            aval,
            off,
            keys,
            vals,
        } = view
        {
            *self = Self {
                inc,
                aval,
                off,
                keys,
                vals,
            };
        }
    }
}

impl RwView<'_> {
    fn len(&self) -> usize {
        match self {
            RwView::Cycles { inc_val, .. } => inc_val.len(),
            RwView::Csr { inc, .. } => inc.len(),
        }
    }

    #[inline]
    fn inc(&self, g: usize) -> K {
        match self {
            RwView::Cycles { inc_val, .. } => inc_val[g],
            RwView::Csr { inc, .. } => inc[g],
        }
    }

    #[inline]
    fn aval(&self, g: usize, x: usize, np: usize) -> K {
        match self {
            RwView::Cycles { aval0, .. } => aval0[g * np + x],
            RwView::Csr { aval, .. } => aval[g * np + x],
        }
    }

    /// Support of group `g`, access `x`: keys and values (`None`: all
    /// values are one).
    #[inline]
    fn entries(&self, g: usize, x: usize) -> (&[u32], Option<&[K]>) {
        match self {
            RwView::Cycles { addrs, .. } => (&addrs[x][g..g + 1], None),
            RwView::Csr {
                off, keys, vals, ..
            } => {
                let (lo, hi) = (off[x][g], off[x][g + 1]);
                (&keys[x][lo..hi], Some(&vals[x][lo..hi]))
            }
        }
    }
}

#[inline]
fn entry_val(vals: Option<&[K]>, i: usize) -> K {
    vals.map_or(K::ONE, |v| v[i])
}

/// Number of distinct keys in two sorted lists.
fn merged_len(l: &[u32], r: &[u32]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < l.len() && j < r.len() {
        match l[i].cmp(&r[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
        c += 1;
    }
    c + (l.len() - i) + (r.len() - j)
}

/// Merge two sorted `(address, value)` lists with coefficients into
/// `keys`/`vals` (exactly `merged_len` slots).
#[allow(clippy::too_many_arguments)]
fn merge_into(
    lk: &[u32],
    lv: Option<&[K]>,
    rk: &[u32],
    rv: Option<&[K]>,
    c0: K,
    c1: K,
    keys: &mut [u32],
    vals: &mut [K],
) {
    let (mut i, mut j, mut o) = (0, 0, 0);
    while i < lk.len() || j < rk.len() {
        let take_l = j >= rk.len() || (i < lk.len() && lk[i] <= rk[j]);
        let take_r = i >= lk.len() || (j < rk.len() && rk[j] <= lk[i]);
        let k = if take_l { lk[i] } else { rk[j] };
        let v = if take_l && take_r {
            // A shared key needs a single affine fold, rather than two
            // separately weighted products. Initial Boolean rows need none.
            match (lv, rv) {
                (None, None) => K::ONE,
                _ => {
                    let left = entry_val(lv, i);
                    left + c1 * (left + entry_val(rv, j))
                }
            }
        } else if take_l {
            lv.map_or(c0, |values| c0 * values[i])
        } else {
            rv.map_or(c1, |values| c1 * values[j])
        };
        if take_l {
            i += 1;
        }
        if take_r {
            j += 1;
        }
        keys[o] = k;
        vals[o] = v;
        o += 1;
    }
    debug_assert_eq!(o, keys.len());
}

/// Pairs per parallel work item when building the next CSR view.
const BIND_CHUNK: usize = 1 << 10;

/// Bind the sibling groups of `view` at `chal` into the next CSR view.
fn rw_bind(
    view: &RwView<'_>,
    crosses: &[K],
    np: usize,
    chal: K,
    scratch: &mut RwScratch,
) -> RwView<'static> {
    let half = view.len() / 2;
    let (c0, c1) = (K::ONE + chal, chal);
    let chal2 = chal.square();
    let par = view.len() >= PAR_THRESHOLD;
    let mut inc = std::mem::take(&mut scratch.inc);
    let mut aval = std::mem::take(&mut scratch.aval);
    inc.resize(half, K::ZERO);
    aval.resize(half * np, K::ZERO);
    let scal = |(m, (ic, av)): (usize, (&mut K, &mut [K]))| {
        let (il, ir) = (view.inc(2 * m), view.inc(2 * m + 1));
        *ic = il + chal * (il + ir);
        for (x, slot) in av.iter_mut().enumerate() {
            let (al, ar, cross) = (
                view.aval(2 * m, x, np),
                view.aval(2 * m + 1, x, np),
                crosses[m * np + x],
            );
            *slot = al + (chal.clmul(cross) + chal2.clmul(al + ar + cross)).reduce();
        }
    };
    if par {
        inc.par_iter_mut()
            .zip(aval.par_chunks_mut(np))
            .enumerate()
            .for_each(scal);
    } else {
        inc.iter_mut()
            .zip(aval.chunks_mut(np))
            .enumerate()
            .for_each(scal);
    }
    let (mut offs, mut keys_all, mut vals_all) = (
        std::mem::take(&mut scratch.off),
        std::mem::take(&mut scratch.keys),
        std::mem::take(&mut scratch.vals),
    );
    offs.resize_with(np, Vec::new);
    keys_all.resize_with(np, Vec::new);
    vals_all.resize_with(np, Vec::new);
    for x in 0..np {
        // Exact sizes first, then fill disjoint chunks in parallel.
        let off = &mut offs[x];
        off.resize(half + 1, 0);
        off[0] = 0;
        let count = |(m, c): (usize, &mut usize)| {
            *c = merged_len(view.entries(2 * m, x).0, view.entries(2 * m + 1, x).0);
        };
        if par {
            off[1..].par_iter_mut().enumerate().for_each(count);
        } else {
            off[1..].iter_mut().enumerate().for_each(count);
        }
        for m in 0..half {
            off[m + 1] += off[m];
        }
        let total = off[half];
        let keys = &mut keys_all[x];
        let vals = &mut vals_all[x];
        keys.resize(total, 0);
        vals.resize(total, K::ZERO);
        let mut pieces: Vec<(usize, &mut [u32], &mut [K])> = Vec::new();
        let (mut krest, mut vrest) = (&mut keys[..], &mut vals[..]);
        let mut m0 = 0;
        while m0 < half {
            let m1 = (m0 + BIND_CHUNK).min(half);
            let sz = off[m1] - off[m0];
            let (kh, kt) = krest.split_at_mut(sz);
            let (vh, vt) = vrest.split_at_mut(sz);
            pieces.push((m0, kh, vh));
            krest = kt;
            vrest = vt;
            m0 = m1;
        }
        let fill = |(m0, ks, vs): (usize, &mut [u32], &mut [K])| {
            let m1 = (m0 + BIND_CHUNK).min(half);
            let base = off[m0];
            for m in m0..m1 {
                let (lk, lv) = view.entries(2 * m, x);
                let (rk, rv) = view.entries(2 * m + 1, x);
                let (a, b) = (off[m] - base, off[m + 1] - base);
                merge_into(lk, lv, rk, rv, c0, c1, &mut ks[a..b], &mut vs[a..b]);
            }
        };
        if par {
            pieces.into_par_iter().for_each(fill);
        } else {
            pieces.into_iter().for_each(fill);
        }
    }
    RwView::Csr {
        inc,
        aval,
        off: offs,
        keys: keys_all,
        vals: vals_all,
    }
}

/// The sparse rw sum-check of `twist_prove`. Returns the same triple
/// as `sumcheck_prove` on the dense tables of §7.5: proof, all
/// challenges (`r_j` then `r_k`), and the table MLEs at the point.
#[allow(clippy::too_many_arguments)]
fn rw_sumcheck_sparse(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    alpha0: &[K],
    alphas: &[Vec<K>],
    beta: &[K],
    transcript: &mut Transcript,
    label: &[u8],
) -> (SumcheckProof, Vec<K>, Vec<K>) {
    let setup_span = om_transcript::profiling::Span::new("twist.rw.setup");
    let (a, p, n) = (params.a, params.p, params.n);
    let (nk, _t, np) = (1usize << a, 1usize << n, params.p + 1);
    assert_eq!(m0.len(), nk, "twist_prove: m0 must have 2^a words");

    // Address of every access in every cycle (`p` is the write).
    let addr_cows: Vec<std::borrow::Cow<'_, [u32]>> = wit
        .ra
        .iter()
        .chain(std::iter::once(&wit.wa))
        .map(|c| c.addrs())
        .collect();
    let addrs: Vec<&[u32]> = addr_cows.iter().map(|c| &c[..]).collect();
    let ie = WordEval::new(&claims.r_o);
    let (mut log, aval0, inc_val) = WriteLog::new(m0, &addrs, &wit.inc, &ie);
    let phis: Vec<Vec<K>> = (0..np).map(|x| phi_k(alpha0[x], &alphas[x], a)).collect();

    let mut view = RwView::Cycles {
        addrs: &addrs,
        inc_val,
        aval0,
    };
    let msg_label = sublabel(label, b"/msg");
    let chal_label = sublabel(label, b"/chal");
    let (mut msgs, mut rs): (Vec<Vec<K>>, Vec<K>) = (Vec::new(), Vec::new());
    let mut cur = K::ZERO;
    let mut scratch = RwScratch::default();
    let mut crosses = Vec::new();
    let mut inverses = vec![K::ZERO];
    inverses.extend(batch_inverse_one_plus(&claims.r_c[n.min(1)..]));
    let mut eq_prefix = K::ONE;
    let mut normalized_claim = None;

    drop(setup_span);
    let cycles_span = om_transcript::profiling::Span::new("twist.rw.cycles");
    if n == 0 {
        // No cycle rounds: the claim is the single group's total.
        let mut inner = beta[p] * view.inc(0);
        for x in 0..np {
            let (ks, vs) = view.entries(0, x);
            let mut b = K::ZERO;
            for (i, &k) in ks.iter().enumerate() {
                b += entry_val(vs, i) * phis[x][k as usize];
            }
            inner += beta[x] * (view.aval(0, x, np) + b);
        }
        cur = inner;
        sumcheck_init(transcript, label, cur, a, 3);
    }

    // Cycle rounds over the sparse groups.
    for r in 0..n {
        let half = view.len() / 2;
        let par = view.len() >= PAR_THRESHOLD;
        crosses.resize(half * np, K::ZERO);
        // The transparent equality polynomial is rank one. Accumulate the
        // quadratic cofactor against a split suffix table, then attach its
        // current-variable linear factor once for the entire round (Gruen).
        // Later rounds need just one endpoint and the quadratic coefficient.
        let z = claims.r_c[r];
        let need_zero = r == 0 || z == K::ONE;
        let need_one = r == 0 || z != K::ONE;
        let suffix = SplitEq::new(&claims.r_c[r + 1..]);
        let block_pairs = suffix.lo.len();
        // Batch coefficients and low Eq factors are invariant across high
        // blocks. Combining them once removes per-pair intermediate reductions.
        let weighted_low = scale_low_weights(beta, &suffix.lo);
        let process = |(high, cross_block): (usize, &mut [K])| -> [K; 3] {
            let mut acc = [U256::ZERO; 3];
            for (low, cx) in cross_block.chunks_mut(np).enumerate() {
                let m = high * block_pairs + low;
                let (gl, gr) = (2 * m, 2 * m + 1);
                let (il, ir) = (view.inc(gl), view.inc(gr));
                let low_weights = &weighted_low[low * np..(low + 1) * np];
                let mut core = [U256::ZERO; 3];
                let (mut sl, mut sr) = (K::ZERO, K::ZERO);
                if r == 0 {
                    let w_l = addrs[p][gl];
                    for x in 0..np {
                        let k_l = addrs[x][gl];
                        let k_r = addrs[x][gr];
                        let v_l = view.aval(gl, x, np);
                        let v_r = view.aval(gr, x, np);
                        let mut cross = v_l + v_r;
                        if k_l == w_l {
                            cross += il;
                        }
                        if k_r == w_l {
                            cross += il;
                        }
                        cx[x] = cross;
                        let bx = low_weights[x];
                        if need_zero {
                            core[0] += bx.clmul(v_l + phis[x][k_l as usize]);
                        }
                        if need_one {
                            core[1] += bx.clmul(v_r + phis[x][k_r as usize]);
                        }
                        core[2] += bx.clmul(v_l + v_r + cross);
                    }
                    // The Boolean address tables sum to one in each row.
                    if need_zero {
                        core[0] += low_weights[p].clmul(il);
                    }
                    if need_one {
                        core[1] += low_weights[p].clmul(ir);
                    }
                } else {
                    for x in 0..np {
                        let (lk, lv) = view.entries(gl, x);
                        let (rk, rv) = view.entries(gr, x);
                        let mut cross = U256::ZERO;
                        let (mut bl_x, mut br_x) = (U256::ZERO, U256::ZERO);
                        for (i, &k) in lk.iter().enumerate() {
                            let v = entry_val(lv, i);
                            cross += v.clmul(log.v_group(k as usize, gr));
                            if need_zero {
                                bl_x += v.clmul(phis[x][k as usize]);
                            }
                            if x == p {
                                sl += v;
                            }
                        }
                        for (i, &k) in rk.iter().enumerate() {
                            let v = entry_val(rv, i);
                            cross += v.clmul(log.v_group(k as usize, gl));
                            if need_one {
                                br_x += v.clmul(phis[x][k as usize]);
                            }
                            if x == p {
                                sr += v;
                            }
                        }
                        let cross = cross.reduce();
                        cx[x] = cross;
                        let (al, ar) = (view.aval(gl, x, np), view.aval(gr, x, np));
                        let bx = low_weights[x];
                        if need_zero {
                            core[0] += bx.clmul(al + bl_x.reduce());
                        }
                        if need_one {
                            core[1] += bx.clmul(ar + br_x.reduce());
                        }
                        core[2] += bx.clmul(al + ar + cross);
                    }
                    if need_zero {
                        core[0] += low_weights[p].clmul(il * sl);
                    }
                    if need_one {
                        core[1] += low_weights[p].clmul(ir * sr);
                    }
                    core[2] += low_weights[p].clmul((il + ir) * (sl + sr));
                }
                for (sum, term) in acc.iter_mut().zip(core) {
                    *sum += term;
                }
            }
            let high_weight = suffix.hi[high];
            acc.map(|value| high_weight * value.reduce())
        };
        let add3 = |left: [K; 3], right: [K; 3]| core::array::from_fn(|i| left[i] + right[i]);
        let stats = if par {
            crosses
                .par_chunks_mut(block_pairs * np)
                .enumerate()
                .map(process)
                .reduce(|| [K::ZERO; 3], add3)
        } else {
            crosses
                .chunks_mut(block_pairs * np)
                .enumerate()
                .map(process)
                .fold([K::ZERO; 3], add3)
        };
        let cofactor = reconstruct_cofactor(&stats, z, normalized_claim, inverses[r]);
        let ev = factored_evaluations(&cofactor, z, eq_prefix);
        if r == 0 {
            cur = ev[0] + ev[1];
            sumcheck_init(transcript, label, cur, n + a, 3);
        }
        assert_eq!(
            ev[0] + ev[1],
            cur,
            "claim inconsistent with tables (g(0)+g(1) != claim)"
        );
        let mut msg = vec![ev[0]];
        msg.extend_from_slice(&ev[2..]);
        transcript.absorb_ks(&msg_label, &msg);
        let chal = transcript.challenge_k(&chal_label);
        msgs.push(msg);
        rs.push(chal);
        cur = interpolate_eval(&ev, chal);
        normalized_claim = Some(eval_coefficients(&cofactor, chal));
        eq_prefix *= K::ONE + z + chal;
        let next = rw_bind(&view, &crosses, np, chal, &mut scratch);
        scratch.recycle(std::mem::replace(&mut view, next));
        log.bind(chal);
    }

    drop(crosses);
    drop(scratch);
    drop(cycles_span);
    let _address_span = om_transcript::profiling::Span::new("twist.rw.address");
    // Address rounds: dense tables of size `2^a` only.
    let mut hist: Vec<Vec<K>> = vec![vec![K::ZERO; nk]; np];
    for (x, h) in hist.iter_mut().enumerate() {
        let (ks, vs) = view.entries(0, x);
        for (i, &k) in ks.iter().enumerate() {
            h[k as usize] += entry_val(vs, i);
        }
    }
    let (eq0, inc0) = (eq_prefix, view.inc(0));
    drop(view);
    let vat = |k: usize| log.v_group(k, 0);
    let vtab: Vec<K> = if nk >= PAR_THRESHOLD {
        (0..nk).into_par_iter().map(vat).collect()
    } else {
        (0..nk).map(vat).collect()
    };
    drop(log);
    let mut tables: Vec<Vec<K>> = vec![vtab];
    tables.extend(hist);
    tables.extend(phis);
    // Equality and Inc are scalars after all cycle variables are bound.
    // Fold the scalar equality into the batching weights once and use the
    // actual quadratic relation, retaining the declared cubic wire format.
    let weighted_beta: Vec<K> = beta.iter().map(|&weight| eq0 * weight).collect();
    let relation = crate::sumcheck_kernels::Relation::RwAddress {
        weights: &weighted_beta,
        inc: inc0,
    };
    let mut bound = Vec::new();
    let mut ev = if a == 0 {
        Vec::new()
    } else {
        crate::sumcheck_kernels::dense_round(&tables, relation, cur)
    };
    for bit in 0..a {
        assert_eq!(
            ev[0] + ev[1],
            cur,
            "claim inconsistent with tables (g(0)+g(1) != claim)"
        );
        let mut msg = vec![ev[0]];
        msg.extend_from_slice(&ev[2..]);
        transcript.absorb_ks(&msg_label, &msg);
        let chal = transcript.challenge_k(&chal_label);
        msgs.push(msg);
        rs.push(chal);
        cur = interpolate_eval(&ev, chal);
        let phi_offsets: Vec<K> = alphas.iter().map(|alpha| chal * alpha[bit]).collect();
        ev = crate::sumcheck_kernels::bind_and_round_with_affine_into(
            &tables,
            &mut bound,
            chal,
            relation,
            cur,
            &phi_offsets,
        );
        std::mem::swap(&mut tables, &mut bound);
    }
    // Keep the original table-MLE order while constants stay outside dense tables.
    let mut mles = vec![eq0, tables[0][0], inc0];
    mles.extend(tables[1..].iter().map(|tb| tb[0]));
    (SumcheckProof { round_msgs: msgs }, rs, mles)
}

/// Prove one BinaryTwist instance (spec 07 §7.5), with the optional
/// final-memory claim (§7.6) and Inc-at-a-random-point output (§7.7).
///
/// Every sum-check runs on per-cycle data (addresses and `Inc` words)
/// without full-length tables (§7.10); the bytes are those of the
/// dense definition.
#[allow(clippy::too_many_arguments)]
pub fn twist_prove(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
) -> (TwistProof, TwistOutput) {
    twist_prove_with_virtualization(
        params,
        m0,
        wit,
        claims,
        transcript,
        label,
        final_mem,
        inc_extra_flag,
        VirtualizationMode::FlatLegacy,
    )
}

/// Prove with the selected address-component reduction. Flat retains the original transcript.
#[allow(clippy::too_many_arguments)]
pub fn twist_prove_with_virtualization(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: VirtualizationMode,
) -> (TwistProof, TwistOutput) {
    twist_prove_with_virtualization_and_shared_address(
        params,
        m0,
        wit,
        claims,
        transcript,
        label,
        final_mem,
        inc_extra_flag,
        mode,
        false,
    )
}

/// Prove with explicit structural sharing of the read/write address source.
///
/// When `shared_address` is true, the caller declares that the sole read and
/// write components are the same committed polynomials. This selects canonical
/// source references for prover computation without comparing witness values;
/// every original claim slot and the verifier's checks remain unchanged.
#[allow(clippy::too_many_arguments)]
pub fn twist_prove_with_virtualization_and_shared_address(
    params: &TwistParams,
    m0: &[u64],
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: VirtualizationMode,
    shared_address: bool,
) -> (TwistProof, TwistOutput) {
    let _other_span = om_transcript::profiling::Span::new("twist.other");
    assert!(
        !shared_address || params.p == 1,
        "shared address requires exactly one read port"
    );
    assert!(
        mode != VirtualizationMode::GkrTree || params.d.is_power_of_two(),
        "tree virtualization requires a power-of-two component count"
    );
    absorb_virtualization_mode(transcript, label, mode);
    let (a, d, p, n, ell) = (params.a, params.d, params.p, params.n, params.ell);
    let nk = 1usize << a;
    assert_eq!(wit.inc.len(), 1usize << n, "twist_prove: Inc length");
    absorb(transcript, label, claims);
    let (alpha0, alphas, beta) = coeffs(transcript, label, p, a);
    assert_eq!(m0.len(), nk, "twist_prove: m0 must have 2^a words");
    // The rw sum-check runs sparsely (§7.10): no `T * 2^a` tables.
    let (rw, r, mles) = rw_sumcheck_sparse(
        params,
        m0,
        wit,
        claims,
        &alpha0,
        &alphas,
        &beta,
        transcript,
        &sublabel(label, b"/rw"),
    );
    let (r_j, r_k) = (r[..n].to_vec(), r[n..].to_vec());
    let mut rw_ev: Vec<K> = (0..=p).map(|x| mles[3 + x]).collect();
    rw_ev.push(mles[1]);
    rw_ev.push(mles[2]);
    transcript.absorb_ks(&sublabel(label, b"/rw/evals"), &rw_ev);

    // Val evaluation: tables `LT~(j, r_j)`, `eq(r_k, wa(j))`,
    // `Inc~(r_o, j)`, evaluated per cycle.
    let val_span = om_transcript::profiling::Span::new("twist.val");
    let writes_addr = wit.wa.addrs();
    let ie = WordEval::new(&claims.r_o);
    {
        let (lt, ek) = (SplitLt::new(&r_j), eq_table(&r_k));
        let row = |j: usize, out: &mut [K]| {
            out[0] = lt.at(j);
            out[1] = ek[writes_addr[j] as usize];
            out[2] = ie.eval(wit.inc[j]);
        };
        let (_, val, s_v, mles) =
            sumcheck_prove_rows_products(n, &row, 1, 2, transcript, &sublabel(label, b"/val"));
        let val_ev = vec![mles[1], mles[2]];
        transcript.absorb_ks(&sublabel(label, b"/val/evals"), &val_ev);
        drop(val_span);
        twist_prove_tail(
            params,
            wit,
            claims,
            transcript,
            label,
            final_mem,
            inc_extra_flag,
            &writes_addr,
            (rw, rw_ev, r_j, r_k),
            (val, val_ev, s_v),
            (d, ell),
            mode,
            shared_address,
        )
    }
}

/// The steps of [`twist_prove`] after the Val sum-check: final memory,
/// `Inc` at a random point, virtualization and the one-hot check.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn twist_prove_tail(
    params: &TwistParams,
    wit: &TwistWitness,
    claims: &TwistClaims,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    writes_addr: &[u32],
    (rw, rw_ev, r_j, r_k): (SumcheckProof, Vec<K>, Vec<K>, Vec<K>),
    (val, val_ev, s_v): (SumcheckProof, Vec<K>, Vec<K>),
    (d, ell): (usize, usize),
    mode: VirtualizationMode,
    shared_address: bool,
) -> (TwistProof, TwistOutput) {
    let (a, n) = (params.a, params.n);
    // Final memory
    let (mut fin, mut fin_ev) = (None, None);
    let (mut r_kf, mut s_f, mut r_of) = (None, None, None);
    if let Some(final_mem) = final_mem {
        let _final_span = om_transcript::profiling::Span::new("twist.final");
        let r_low = transcript.challenge_ks(&sublabel(label, b"/fin/r_low"), final_mem.m);
        let rof = transcript.challenge_ks(&sublabel(label, b"/fin/r_o"), ell);
        let mut rkf = r_low;
        rkf.extend(s_out_bits(final_mem, a));
        let (ekf, ief) = (eq_table(&rkf), WordEval::new(&rof));
        let row = |j: usize, out: &mut [K]| {
            out[0] = ekf[writes_addr[j] as usize];
            out[1] = ief.eval(wit.inc[j]);
        };
        let (_, fin_sc, sf, mles) =
            sumcheck_prove_rows_mul2(n, &row, transcript, &sublabel(label, b"/fin"));
        let fev = vec![mles[0], mles[1]];
        transcript.absorb_ks(&sublabel(label, b"/fin/evals"), &fev);
        fin = Some(fin_sc);
        fin_ev = Some(fev);
        r_kf = Some(rkf);
        s_f = Some(sf);
        r_of = Some(rof);
    }

    let (mut extra, mut extra_claim) = (None, None);
    if inc_extra_flag {
        let xo = transcript.challenge_ks(&sublabel(label, b"/incx/o"), ell);
        let xc = transcript.challenge_ks(&sublabel(label, b"/incx/c"), n);
        let value = word_mle_eval(&wit.inc, &xo, &xc);
        let mut pt = xo;
        pt.extend_from_slice(&xc);
        transcript.absorb_k(&sublabel(label, b"/incx/val"), value);
        extra = Some(value);
        extra_claim = Some(EvalClaim {
            poly: format!("{}/inc", params.name),
            point: pt,
            value,
        });
    }

    let grps = groups(
        params,
        final_mem.is_some(),
        &r_k,
        &r_j,
        &rw_ev,
        &s_v,
        &val_ev,
        r_kf.as_deref(),
        s_f.as_deref(),
        fin_ev.as_deref(),
    );
    let write_comps = if shared_address { &wit.ra[0] } else { &wit.wa };
    let mut comps_per_group: Vec<&AddrComps> = wit.ra.iter().collect();
    comps_per_group.push(write_comps);
    if final_mem.is_some() {
        comps_per_group.push(write_comps);
    }
    let (vproof, tree_proof, flat_proof, mut out_claims) = {
        let _virtualization_span = om_transcript::profiling::Span::new("twist.virtualization");
        match mode {
            VirtualizationMode::FlatLegacy => {
                let (proof, claims) = virt_prove(
                    &grps,
                    &comps_per_group,
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (proof, None, None, claims)
            }
            VirtualizationMode::Flat => {
                let (proof, claims) = virt_flat_prove(
                    &grps,
                    &comps_per_group,
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (None, None, proof, claims)
            }
            VirtualizationMode::GkrTree => {
                let (proof, claims) = virt_tree_prove(
                    &grps,
                    &comps_per_group,
                    a,
                    d,
                    n,
                    transcript,
                    &sublabel(label, b"/virt"),
                );
                (None, proof, None, claims)
            }
        }
    };
    let mut all_comps: Vec<&AddrComps> = wit.ra.iter().collect();
    all_comps.push(write_comps);
    let (oproof, oclaims) = {
        let _onehot_span = om_transcript::profiling::Span::new("twist.onehot");
        onehot_prove(
            &all_comps,
            &comp_names_tw(params),
            a / d,
            n,
            transcript,
            &sublabel(label, b"/onehot"),
        )
    };
    out_claims.extend(oclaims);
    out_claims.extend(inc_claims(
        params,
        claims,
        &r_j,
        &rw_ev,
        &s_v,
        &val_ev,
        r_of.as_deref(),
        s_f.as_deref(),
        fin_ev.as_deref(),
    ));
    if let Some(ec) = &extra_claim {
        out_claims.push(ec.clone());
    }
    let out = TwistOutput {
        claims: out_claims,
        inc_extra: extra_claim,
    };
    (
        TwistProof {
            rw,
            rw_evals: rw_ev,
            val,
            val_evals: val_ev,
            fin,
            fin_evals: fin_ev,
            inc_extra: extra,
            virt: vproof,
            virt_tree: tree_proof,
            virt_flat: flat_proof,
            onehot: oproof,
        },
        out,
    )
}

/// Verify one BinaryTwist instance, returning its output claims.
/// `m0_eval(point)` must return `M0~(point)` for
/// `point = r_k ‖ r_o` (`a + ell` coordinates).
///
/// # Errors
/// Returns [`VerificationError`] on malformed claims or proof, a proof
/// whose shape does not match the instance, or any failing check.
#[allow(clippy::too_many_arguments)]
pub fn twist_verify(
    params: &TwistParams,
    m0_eval: &dyn Fn(&[K]) -> K,
    claims: &TwistClaims,
    proof: &TwistProof,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
) -> Result<TwistOutput, VerificationError> {
    twist_verify_with_virtualization(
        params,
        m0_eval,
        claims,
        proof,
        transcript,
        label,
        final_mem,
        inc_extra_flag,
        VirtualizationMode::FlatLegacy,
    )
}

/// Verify the selected reduction, rejecting proofs of another mode.
#[allow(clippy::too_many_arguments)]
pub fn twist_verify_with_virtualization(
    params: &TwistParams,
    m0_eval: &dyn Fn(&[K]) -> K,
    claims: &TwistClaims,
    proof: &TwistProof,
    transcript: &mut Transcript,
    label: &[u8],
    final_mem: Option<&FinalMem>,
    inc_extra_flag: bool,
    mode: VirtualizationMode,
) -> Result<TwistOutput, VerificationError> {
    let (a, d, p, n, ell) = (params.a, params.d, params.p, params.n, params.ell);
    if (d == 1 && (proof.virt.is_some() || proof.virt_tree.is_some() || proof.virt_flat.is_some()))
        || match mode {
            VirtualizationMode::FlatLegacy => {
                proof.virt_tree.is_some() || proof.virt_flat.is_some()
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
            "twist: virtualization mode/proof shape mismatch",
        ));
    }
    if claims.rv.len() != p
        || claims.bits.len() != p + 1
        || claims.r_o.len() != ell
        || claims.r_c.len() != n
        || claims.bits.iter().any(|b| b.len() != a)
    {
        return Err(VerificationError::new("twist: malformed claims"));
    }
    if proof.rw_evals.len() != p + 3 || proof.val_evals.len() != 2 {
        return Err(VerificationError::new("twist: malformed proof"));
    }
    if final_mem.is_none() != proof.fin.is_none() || inc_extra_flag != proof.inc_extra.is_some() {
        return Err(VerificationError::new(
            "twist: proof shape does not match instance",
        ));
    }
    absorb_virtualization_mode(transcript, label, mode);
    absorb(transcript, label, claims);
    let (alpha0, alphas, beta) = coeffs(transcript, label, p, a);
    let mut claim = K::ZERO;
    for x in 0..=p {
        let v = if x < p { claims.rv[x] } else { claims.wv };
        let mut t_acc = v + alpha0[x];
        for (&al, &f) in alphas[x].iter().zip(&claims.bits[x]) {
            t_acc += al * f;
        }
        claim += beta[x] * t_acc;
    }
    let (r, expected) = sumcheck_verify(
        claim,
        n + a,
        3,
        &proof.rw,
        transcript,
        &sublabel(label, b"/rw"),
    )?;
    let (r_j, r_k) = (r[..n].to_vec(), r[n..].to_vec());
    let ev = &proof.rw_evals;
    transcript.absorb_ks(&sublabel(label, b"/rw/evals"), ev);
    let (big_v, big_i) = (ev[p + 1], ev[p + 2]);
    let mut acc = K::ZERO;
    for x in 0..=p {
        let mut phi = alpha0[x];
        for (&al, &rk) in alphas[x].iter().zip(&r_k) {
            phi += al * rk;
        }
        acc += beta[x] * ev[x] * (big_v + phi);
    }
    acc += beta[p] * ev[p] * big_i;
    if eq_eval(&claims.r_c, &r_j) * acc != expected {
        return Err(VerificationError::new("twist: read/write check failed"));
    }

    let mut m0_pt = r_k.clone();
    m0_pt.extend_from_slice(&claims.r_o);
    let claim = big_v + m0_eval(&m0_pt);
    let (s_v, expected) = sumcheck_verify(
        claim,
        n,
        3,
        &proof.val,
        transcript,
        &sublabel(label, b"/val"),
    )?;
    transcript.absorb_ks(&sublabel(label, b"/val/evals"), &proof.val_evals);
    if lt_eval(&s_v, &r_j) * proof.val_evals[0] * proof.val_evals[1] != expected {
        return Err(VerificationError::new("twist: val check failed"));
    }

    let (mut r_kf, mut s_f, mut r_of, mut fin_ev) = (None, None, None, None);
    if let Some(final_mem) = final_mem {
        if final_mem.words.len() != 1usize << final_mem.m
            || final_mem.s_out >= 1usize << (a - final_mem.m)
        {
            return Err(VerificationError::new("twist: bad final-memory spec"));
        }
        let r_low = transcript.challenge_ks(&sublabel(label, b"/fin/r_low"), final_mem.m);
        let rof = transcript.challenge_ks(&sublabel(label, b"/fin/r_o"), ell);
        let mut rkf = r_low.clone();
        rkf.extend(s_out_bits(final_mem, a));
        let mut out_pt = r_low;
        out_pt.extend_from_slice(&rof);
        let mut m0_pt = rkf.clone();
        m0_pt.extend_from_slice(&rof);
        let claim = mle_eval_bits(&out_table(&final_mem.words, ell), &out_pt) + m0_eval(&m0_pt);
        let fin_sc = proof
            .fin
            .as_ref()
            .ok_or_else(|| VerificationError::new("twist: proof shape does not match instance"))?;
        let (sf, expected) =
            sumcheck_verify(claim, n, 2, fin_sc, transcript, &sublabel(label, b"/fin"))?;
        let fev = proof
            .fin_evals
            .as_ref()
            .ok_or_else(|| VerificationError::new("twist: malformed final evals"))?;
        if fev.len() != 2 {
            return Err(VerificationError::new("twist: malformed final evals"));
        }
        transcript.absorb_ks(&sublabel(label, b"/fin/evals"), fev);
        if fev[0] * fev[1] != expected {
            return Err(VerificationError::new("twist: final-memory check failed"));
        }
        fin_ev = Some(fev.clone());
        r_kf = Some(rkf);
        s_f = Some(sf);
        r_of = Some(rof);
    }

    let mut extra_claim = None;
    if inc_extra_flag {
        let xo = transcript.challenge_ks(&sublabel(label, b"/incx/o"), ell);
        let xc = transcript.challenge_ks(&sublabel(label, b"/incx/c"), n);
        let value = proof
            .inc_extra
            .ok_or_else(|| VerificationError::new("twist: proof shape does not match instance"))?;
        transcript.absorb_k(&sublabel(label, b"/incx/val"), value);
        let mut pt = xo;
        pt.extend_from_slice(&xc);
        extra_claim = Some(EvalClaim {
            poly: format!("{}/inc", params.name),
            point: pt,
            value,
        });
    }

    let grps = groups(
        params,
        final_mem.is_some(),
        &r_k,
        &r_j,
        ev,
        &s_v,
        &proof.val_evals,
        r_kf.as_deref(),
        s_f.as_deref(),
        fin_ev.as_deref(),
    );
    let mut out_claims = match mode {
        VirtualizationMode::FlatLegacy => virt_verify(
            &grps,
            a,
            d,
            n,
            proof.virt.as_ref(),
            transcript,
            &sublabel(label, b"/virt"),
        )?,
        VirtualizationMode::Flat => virt_flat_verify(
            &grps,
            a,
            d,
            n,
            proof.virt_flat.as_ref(),
            transcript,
            &sublabel(label, b"/virt"),
        )?,
        VirtualizationMode::GkrTree => virt_tree_verify(
            &grps,
            a,
            d,
            n,
            proof.virt_tree.as_ref(),
            transcript,
            &sublabel(label, b"/virt"),
        )?,
    };
    let oclaims = onehot_verify(
        &comp_names_tw(params),
        a / d,
        n,
        &proof.onehot,
        transcript,
        &sublabel(label, b"/onehot"),
    )?;
    out_claims.extend(oclaims);
    out_claims.extend(inc_claims(
        params,
        claims,
        &r_j,
        ev,
        &s_v,
        &proof.val_evals,
        r_of.as_deref(),
        s_f.as_deref(),
        fin_ev.as_deref(),
    ));
    if let Some(ec) = &extra_claim {
        out_claims.push(ec.clone());
    }
    Ok(TwistOutput {
        claims: out_claims,
        inc_extra: extra_claim,
    })
}

#[cfg(test)]
mod optimized_tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn word(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            x ^ (x >> 31)
        }

        fn field(&mut self) -> K {
            K::from_u128(u128::from(self.word()) | (u128::from(self.word()) << 64))
        }
    }

    #[test]
    fn active_change_history_matches_dense_values_at_every_binding() {
        for (a, n) in [(0, 0), (1, 1), (3, 4), (5, 6)] {
            let (nk, t) = (1 << a, 1 << n);
            for seed in 0..6 {
                for mode in 0..3 {
                    let mut rng = Rng(seed + 19);
                    let m0: Vec<u64> = (0..nk).map(|_| rng.word()).collect();
                    let writes: Vec<u32> = (0..t)
                        .map(|_| {
                            if mode == 2 {
                                0
                            } else {
                                (rng.word() as usize % nk) as u32
                            }
                        })
                        .collect();
                    let reads: Vec<u32> =
                        (0..t).map(|_| (rng.word() as usize % nk) as u32).collect();
                    let inc: Vec<u64> = (0..t)
                        .map(|j| {
                            if mode == 0 || (j % 5 != 2 && j % 7 != 3 && j + 1 != t) {
                                0
                            } else {
                                rng.word()
                            }
                        })
                        .collect();
                    let r_o: Vec<K> = (0..6)
                        .map(|_| if seed == 0 { K::ZERO } else { rng.field() })
                        .collect();
                    let we = WordEval::new(&r_o);
                    let addrs = [&reads[..], &writes[..]];
                    let (mut log, aval, inc_values) = WriteLog::new(&m0, &addrs, &inc, &we);
                    let mut current: Vec<K> = m0.iter().map(|&w| we.eval(w)).collect();
                    let mut before = vec![vec![K::ZERO; t]; nk];
                    let mut after = before.clone();
                    for j in 0..t {
                        for k in 0..nk {
                            before[k][j] = current[k];
                        }
                        assert_eq!(aval[2 * j], current[reads[j] as usize]);
                        assert_eq!(aval[2 * j + 1], current[writes[j] as usize]);
                        assert_eq!(inc_values[j], we.eval(inc[j]));
                        current[writes[j] as usize] += inc_values[j];
                        for k in 0..nk {
                            after[k][j] = current[k];
                        }
                    }
                    let active: Vec<usize> = writes
                        .iter()
                        .zip(&inc)
                        .filter_map(|(&k, &v)| (v != 0).then_some(k as usize))
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    assert_eq!(log.active, active);
                    assert_eq!(log.grp.len(), inc.iter().filter(|&&v| v != 0).count());
                    if mode == 0 {
                        assert!(log.active.is_empty());
                    }
                    let mut challenges = Vec::new();
                    for round in 0..=n {
                        let weights = eq_table(&challenges);
                        let block = 1 << round;
                        for k in 0..nk {
                            for group in 0..t / block {
                                let expected =
                                    weights.iter().enumerate().fold(K::ZERO, |acc, (u, &w)| {
                                        acc + w * before[k][group * block + u]
                                    });
                                assert_eq!(
                                    log.v_group(k, group),
                                    expected,
                                    "a={a}, n={n}, seed={seed}, mode={mode}, round={round}, k={k}, group={group}"
                                );
                            }
                            for i in 0..log.len[k] {
                                let at = log.off[k] + i;
                                let group = log.grp[at] as usize;
                                assert_eq!(log.v_after[at], after[k][(group + 1) * block - 1]);
                            }
                        }
                        if round < n {
                            let challenge = match round % 4 {
                                0 => K::ZERO,
                                1 => K::ONE,
                                2 => K::t(2),
                                _ => rng.field(),
                            };
                            log.bind(challenge);
                            challenges.push(challenge);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn affine_phi_matches_bitwise_definition() {
        let mut rng = Rng(77);
        for a in 0..=10 {
            let alpha0 = rng.field();
            let alphas: Vec<K> = (0..a).map(|_| rng.field()).collect();
            for len in [0, a / 2, a] {
                let table = phi_k(alpha0, &alphas[..len], a);
                for (k, &value) in table.iter().enumerate() {
                    let expected =
                        alphas[..len]
                            .iter()
                            .enumerate()
                            .fold(
                                alpha0,
                                |acc, (bit, &alpha)| {
                                    if k >> bit & 1 != 0 { acc + alpha } else { acc }
                                },
                            );
                    assert_eq!(value, expected);
                }
            }
        }
    }

    #[test]
    fn sparse_rw_messages_and_transcript_match_dense_reference() {
        for (a, n, p) in [
            (1, 0, 0),
            (1, 3, 0),
            (1, 0, 1),
            (2, 1, 1),
            (3, 3, 1),
            (3, 4, 2),
        ] {
            for seed in 0..8 {
                let mut rng = Rng(seed + 101);
                let (nk, t, np) = (1 << a, 1 << n, p + 1);
                let params = TwistParams {
                    name: "rw-test".into(),
                    a,
                    d: 1,
                    p,
                    n,
                    ell: 3,
                };
                let m0: Vec<u64> = (0..nk).map(|_| rng.word() & 255).collect();
                let writes: Vec<usize> = (0..t)
                    .map(|_| {
                        if seed % 3 == 0 {
                            0
                        } else {
                            rng.word() as usize % nk
                        }
                    })
                    .collect();
                let reads: Vec<Vec<usize>> = (0..p)
                    .map(|_| {
                        if seed % 2 == 0 {
                            writes.clone()
                        } else {
                            (0..t).map(|_| rng.word() as usize % nk).collect()
                        }
                    })
                    .collect();
                let mut state = m0.clone();
                let new: Vec<u64> = writes
                    .iter()
                    .enumerate()
                    .map(|(j, &k)| {
                        let value = if j % 3 == 0 && seed != 0 {
                            rng.word() & 255
                        } else {
                            state[k]
                        };
                        state[k] = value;
                        value
                    })
                    .collect();
                let (wit, info) = build_witness(&params, &m0, &reads, &writes, &new);
                let r_o: Vec<K> = (0..3).map(|_| rng.field()).collect();
                let r_c: Vec<K> = (0..n)
                    .map(|i| match seed {
                        1 => K::ZERO,
                        2 => K::ONE,
                        3 => K::from_bit(i % 2 == 0),
                        _ => rng.field(),
                    })
                    .collect();
                let claims = honest_claims(&params, &info, &r_o, &r_c);
                let alpha0: Vec<K> = (0..np).map(|_| rng.field()).collect();
                let alphas: Vec<Vec<K>> = (0..np)
                    .map(|_| (0..a).map(|_| rng.field()).collect())
                    .collect();
                let beta: Vec<K> = (0..np).map(|_| rng.field()).collect();
                let we = WordEval::new(&r_o);
                let eq = eq_table(&r_c);
                let mut tables = vec![vec![K::ZERO; nk * t]; 3 + 2 * np];
                let mut values: Vec<K> = m0.iter().map(|&w| we.eval(w)).collect();
                for j in 0..t {
                    let inc = we.eval(wit.inc[j]);
                    for k in 0..nk {
                        let at = j + t * k;
                        tables[0][at] = eq[j];
                        tables[1][at] = values[k];
                        tables[2][at] = inc;
                        for x in 0..np {
                            let address = if x == p { writes[j] } else { reads[x][j] };
                            tables[3 + x][at] = K::from_bit(address == k);
                            tables[3 + np + x][at] = alphas[x].iter().enumerate().fold(
                                alpha0[x],
                                |acc, (bit, &alpha)| {
                                    if k >> bit & 1 != 0 { acc + alpha } else { acc }
                                },
                            );
                        }
                    }
                    values[writes[j]] += inc;
                }
                let combine = |v: &[K]| rw_combine(p, &beta, v);
                let claim = om_sumcheck::combine_sum(&tables, &combine);
                let mut dense_tr = Transcript::new_default(b"rw-equivalence");
                let dense =
                    om_sumcheck::sumcheck_prove(claim, tables, &combine, 3, &mut dense_tr, b"rw");
                let mut sparse_tr = Transcript::new_default(b"rw-equivalence");
                let sparse = rw_sumcheck_sparse(
                    &params,
                    &m0,
                    &wit,
                    &claims,
                    &alpha0,
                    &alphas,
                    &beta,
                    &mut sparse_tr,
                    b"rw",
                );
                assert_eq!(sparse, dense, "a={a}, n={n}, p={p}, seed={seed}");
                assert_eq!(
                    sparse_tr.challenge_k(b"after"),
                    dense_tr.challenge_k(b"after")
                );
            }
        }
    }
}
