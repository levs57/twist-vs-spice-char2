//! Layout, lifting (Eq. 4) and the batching sum-check (spec 05
//! §5.2–5.4). Mirrors `reference/openmagnetar/mosaic.py` byte for byte
//! in its transcript interaction: base label `mos/batch` (absorbing
//! `mos/batch/claims`, squeezing `mos/batch/gamma`, then the sum-check's
//! own sublabels) and `mos/vg` for the final value `v_G`.

use std::collections::BTreeMap;

use om_field::{K, U256, clmul_pairs, mul_pairs};
use om_sumcheck::{
    PackedBits, SumcheckProof, VerificationError, batch_claims, bind_low_in_place, eq_eval,
    eq_table, interpolate_eval, mle_eval_bits, sumcheck_init, sumcheck_verify,
};
use om_transcript::Transcript;
use rayon::prelude::*;

/// Transcript label prefix of this section (Python `mosaic.LABEL`).
pub const LABEL: &[u8] = b"mos";

/// Cap on batched claims (spec 05 §5.4): powers-of-`γ` batching loses
/// `(n-1)/|K|`; `n <= 256` keeps 120 bits.
pub const MAX_CLAIMS: usize = 256;

/// An evaluation claim `p~_{comp_id}(point) = value` on a component
/// polynomial (spec 05 §5.3).
///
/// `comp_id` is `i64` because `om-spartan`'s no-layout mode emits a
/// whole-witness claim with Python's sentinel `comp_id = -1`; genuine
/// component ids are in `[0, 2^32)` (enforced by [`compute_layout`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim {
    pub comp_id: i64,
    pub point: Vec<K>,
    pub value: K,
}

/// Where one component sits in `G`: the aligned subcube starting at
/// `offset`, with `suffix = offset >> log_size` (spec 05 §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub comp_id: i64,
    pub log_size: usize,
    /// First index of the subcube in `G`.
    pub offset: usize,
    /// `offset >> log_size`.
    pub suffix: usize,
}

/// The layout of all components into one hypercube of dimension `ell`
/// (spec 05 §5.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub ell: usize,
    /// In canonical (layout) order: log size descending, id ascending.
    pub placements: Vec<Placement>,
}

impl Layout {
    /// Placements keyed by component id (Python `Layout.by_id`).
    pub fn by_id(&self) -> BTreeMap<i64, &Placement> {
        self.placements.iter().map(|p| (p.comp_id, p)).collect()
    }

    fn placement(&self, comp_id: i64) -> Option<&Placement> {
        self.placements.iter().find(|p| p.comp_id == comp_id)
    }

    /// The suffix as a little-endian bit vector of length
    /// `ell - log_size` (C2): bit `k` is variable `x_{log_size + k + 1}`.
    pub fn suffix_bits(&self, p: &Placement) -> Vec<bool> {
        (0..self.ell - p.log_size)
            .map(|k| (p.suffix >> k) & 1 == 1)
            .collect()
    }
}

/// Compute the layout of `components` (`(id, log_size)` pairs): sort by
/// (log size descending, id ascending) and pack sequentially. Because
/// sizes are non-increasing powers of two, every offset is a multiple
/// of its component's size (spec 05 §5.2). `ell` is the maximum of
/// `ceil_log2(sum 2^log_size)` (with `ceil_log2(1) = 0`), the largest
/// log size, and `ell_min`.
///
/// # Panics
/// Panics where Python raises `ValueError`: an empty component list, a
/// duplicate id, or an id outside `[0, 2^32)`.
pub fn compute_layout(components: &[(i64, usize)], ell_min: usize) -> Layout {
    assert!(!components.is_empty(), "empty component list");
    let mut ids: Vec<i64> = components.iter().map(|c| c.0).collect();
    ids.sort_unstable();
    assert!(
        ids.windows(2).all(|w| w[0] != w[1]),
        "duplicate component id"
    );
    for &(cid, _) in components {
        assert!((0..1 << 32).contains(&cid), "bad component");
    }
    let mut order: Vec<(i64, usize)> = components.to_vec();
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut placements = Vec::with_capacity(order.len());
    let mut cur = 0usize;
    for (cid, ls) in order {
        debug_assert_eq!(cur % (1usize << ls), 0);
        placements.push(Placement {
            comp_id: cid,
            log_size: ls,
            offset: cur,
            suffix: cur >> ls,
        });
        cur += 1usize << ls;
    }
    let total_log = if cur > 1 {
        (usize::BITS - (cur - 1).leading_zeros()) as usize
    } else {
        0
    };
    let max_ls = components.iter().map(|c| c.1).max().unwrap_or(0);
    Layout {
        ell: total_log.max(ell_min).max(max_ls),
        placements,
    }
}

/// Assemble `G`: the bit vector of length `2^ell` holding each
/// component table at its offset and zero elsewhere (spec 05 §5.2).
///
/// # Panics
/// Panics if a component's table is missing (Python `KeyError`) or is
/// not a bit table of `2^log_size` entries (Python `ValueError`; a
/// `bool` table is bit-valued by construction, so only the length can
/// fail).
pub fn build_g(layout: &Layout, tables: &BTreeMap<i64, Vec<bool>>) -> Vec<bool> {
    let mut g = vec![false; 1usize << layout.ell];
    for p in &layout.placements {
        let t = tables
            .get(&p.comp_id)
            .unwrap_or_else(|| panic!("missing table for component {}", p.comp_id));
        assert_eq!(
            t.len(),
            1usize << p.log_size,
            "component must be a bit table of 2^log_size entries"
        );
        g[p.offset..p.offset + t.len()].copy_from_slice(t);
    }
    g
}

/// Lift a claim point to `G`: `r ∥ s_i`, the suffix bits as `K` elements
/// 0/1 in the high variables (Eq. 4, spec 05 §5.3).
///
/// # Errors
/// Returns [`VerificationError`] if the component id is unknown or the
/// point length is not the component's log size.
pub fn lift_point(layout: &Layout, claim: &Claim) -> Result<Vec<K>, VerificationError> {
    let p = layout
        .placement(claim.comp_id)
        .ok_or_else(|| VerificationError::new("claim on unknown component"))?;
    if claim.point.len() != p.log_size {
        return Err(VerificationError::new("claim point has wrong length"));
    }
    let mut out = claim.point.clone();
    out.extend(layout.suffix_bits(p).iter().map(|&b| K::from_bit(b)));
    Ok(out)
}

/// A batching proof: the sum-check proof, then `v_G` (spec 05 §5.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchProof {
    pub sumcheck: SumcheckProof,
    pub v_g: K,
}

/// Write one component's packed bit table directly into its placement in `g`
/// (`g.len() == 2^layout.ell`). Avoids holding all component tables in memory
/// simultaneously before assembling `G`.
///
/// # Panics
/// Panics if `comp_id` is not in `layout` or `table.len() != 2^pl.log_size`.
pub fn place_in_g_packed(g: &mut PackedBits, layout: &Layout, comp_id: i64, table: &PackedBits) {
    assert_eq!(g.len(), 1usize << layout.ell);
    let pl = layout
        .placement(comp_id)
        .unwrap_or_else(|| panic!("unknown component {comp_id}"));
    assert_eq!(
        table.len(),
        1usize << pl.log_size,
        "component must be a bit table of 2^log_size entries"
    );
    let g_words = g.words_mut();
    if pl.offset.is_multiple_of(64) && table.len() >= 64 {
        let dst = pl.offset / 64;
        let src = table.words();
        g_words[dst..dst + src.len()].copy_from_slice(src);
    } else {
        for idx in table.iter_ones() {
            let bit = pl.offset + idx;
            g_words[bit / 64] |= 1u64 << (bit % 64);
        }
    }
}

/// [`build_g`] in packed form: component tables are packed bit
/// tables and the assembled `G` is returned packed, so neither is
/// ever expanded to a byte per bit. The bits are identical.
///
/// # Panics
/// As [`build_g`].
pub fn build_g_packed(layout: &Layout, tables: &BTreeMap<i64, PackedBits>) -> PackedBits {
    let mut g = PackedBits::zero(1usize << layout.ell);
    for pl in &layout.placements {
        let table = tables
            .get(&pl.comp_id)
            .unwrap_or_else(|| panic!("missing table for component {}", pl.comp_id));
        place_in_g_packed(&mut g, layout, pl.comp_id, table);
    }
    g
}

/// The table `E(x) = Σ_j α_j eq(r_j ∥ s_j, x)` (spec 05 §5.4). The
/// suffix factor is an indicator, so claim `j` only touches its own
/// subcube; several claims on one component accumulate there.
///
/// # Panics
/// Panics if `alphas` and `claims` differ in length (Python's `zip`
/// would silently truncate) or a claim names an unknown component
/// (Python `KeyError`).
pub fn eq_sum_table(layout: &Layout, claims: &[Claim], alphas: &[K]) -> Vec<K> {
    assert_eq!(
        alphas.len(),
        claims.len(),
        "eq_sum_table: alphas/claims length mismatch"
    );
    let mut t = vec![K::ZERO; 1usize << layout.ell];
    let pl = layout.by_id();
    for (a, c) in alphas.iter().zip(claims) {
        let p = pl
            .get(&c.comp_id)
            .unwrap_or_else(|| panic!("eq_sum_table: unknown component {}", c.comp_id));
        for (x, e) in eq_table(&c.point).iter().enumerate() {
            t[p.offset + x] += *a * *e;
        }
    }
    t
}

/// Prove the batching sum-check for `claims` against `g` (spec 05
/// §5.4), returning `(proof, r_G, v_G)`. The claims must be true for
/// `g`; their order is the caller's and is never sorted or
/// deduplicated.
///
/// # Panics
/// Panics where Python raises in the prover: more than [`MAX_CLAIMS`]
/// claims (`ValueError`, before any transcript interaction), a claim
/// on an unknown component or with a wrong-length point
/// (`VerificationError` from [`lift_point`]), or a false claim, which
/// the sum-check prover refuses. Also panics if `g` does not have
/// length `2^ell` (the sum-check's table-length check).
pub fn batch_prove(
    layout: &Layout,
    claims: &[Claim],
    g: &[bool],
    transcript: &mut Transcript,
) -> (BatchProof, Vec<K>, K) {
    assert!(
        claims.len() <= MAX_CLAIMS,
        "too many Mosaic claims for powers-of-gamma batching (05.4)"
    );
    for c in claims {
        lift_point(layout, c).unwrap_or_else(|e| panic!("batch_prove: {}", e.message()));
    }
    assert_eq!(
        g.len(),
        1usize << layout.ell,
        "batch_prove: g must have length 2^ell"
    );
    split_batch_prove(
        layout,
        claims,
        &PackedBits::from_bools(g),
        transcript,
        BIT_ROUNDS,
    )
}

/// [`batch_prove`] over a packed `G`: the same proof, without ever
/// expanding `G` (or the eq table) to field elements per bit.
///
/// # Panics
/// As [`batch_prove`], plus if the packed length is not `2^ell`.
pub fn batch_prove_packed(
    layout: &Layout,
    claims: &[Claim],
    g: &PackedBits,
    transcript: &mut Transcript,
) -> (BatchProof, Vec<K>, K) {
    batch_prove_packed_with(layout, claims, g, transcript, BIT_ROUNDS)
}

/// Rounds of the batching sum-check computed straight from the bits
/// of `G`, before the bound tables (`2^(ell - BIT_ROUNDS)` entries
/// each) are materialised for the remaining dense rounds.
pub const BIT_ROUNDS: usize = 10;

/// [`batch_prove_packed`] with an explicit number of bit-level rounds
/// (any value gives the same proof; exposed for tests).
///
/// # Panics
/// As [`batch_prove_packed`].
#[doc(hidden)]
pub fn batch_prove_packed_with(
    layout: &Layout,
    claims: &[Claim],
    g: &PackedBits,
    transcript: &mut Transcript,
    bit_rounds: usize,
) -> (BatchProof, Vec<K>, K) {
    assert!(
        claims.len() <= MAX_CLAIMS,
        "too many Mosaic claims for powers-of-gamma batching (05.4)"
    );
    for c in claims {
        lift_point(layout, c).unwrap_or_else(|e| panic!("batch_prove: {}", e.message()));
    }
    assert_eq!(
        g.len(),
        1usize << layout.ell,
        "batch_prove: g must have length 2^ell"
    );
    split_batch_prove(layout, claims, g, transcript, bit_rounds)
}

/// One claim as the batching prover sees it: its placement, its point
/// lifted to `ell` coordinates, and `alpha_j` times the `eq` factors
/// of the rounds bound so far.
struct BatchTerm<'a> {
    offset: usize,
    log_size: usize,
    point: &'a [K],
    lifted: Vec<K>,
    coef: K,
}

/// Call `f(i)` for each set bit `i` of `words` in `[start, start + len)`,
/// in increasing order, with `i` relative to `start`.
fn for_each_one(words: &[u64], start: usize, len: usize, mut f: impl FnMut(usize)) {
    let end = start + len;
    let mut wi = start / 64;
    while wi * 64 < end {
        let mut w = words[wi];
        let lo = wi * 64;
        if lo < start {
            w &= !0u64 << (start - lo);
        }
        if lo + 64 > end {
            w &= (1u64 << (end - lo)) - 1;
        }
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            w &= w - 1;
            f(lo + b - start);
        }
        wi += 1;
    }
}

/// Per-position sums for round `r` of the bit-level phase, for the
/// claims `group` that share one region of `G`: for each claim, the
/// vector over positions `p < 2^(r+1)` of
/// `sum_y eq(point[r+1..], y) G(base + p + 2^(r+1) y)`. The `eq`
/// factor is split into two square-root tables so the inner sum over
/// set bits is additions only.
fn region_sums(
    words: &[u64],
    base: usize,
    r: usize,
    nfree: usize,
    group: &[&BatchTerm<'_>],
) -> Vec<Vec<K>> {
    let np = 1usize << (r + 1);
    let h = nfree / 2;
    // With no free coordinates the point may be shorter than `r + 1`.
    let coords = |t: &BatchTerm<'_>, a: usize, b: usize| {
        if a == b {
            Vec::new()
        } else {
            t.point[r + 1 + a..r + 1 + b].to_vec()
        }
    };
    let lo: Vec<Vec<K>> = group.iter().map(|t| eq_table(&coords(t, 0, h))).collect();
    let hi: Vec<Vec<K>> = group
        .iter()
        .map(|t| eq_table(&coords(t, h, nfree)))
        .collect();
    let block = 1usize << (r + 1 + h);
    // Scratch is per Rayon worker, not per high block. Products remain
    // unreduced across high blocks and are reduced once per output slot.
    let zero = || {
        (
            vec![vec![U256::ZERO; np]; group.len()],
            vec![vec![K::ZERO; np]; group.len()],
        )
    };
    let rows = 1usize << h;
    // Rows of whole words, eight or more per block: scan eight rows at a time
    // and use subset-sum lookups (`om_field::subset`) where they are dense.
    let grouped = np >= 64 && rows >= 8;
    let words_per_row = np / 64;
    let one_block = |(mut tot, mut acc): (Vec<Vec<U256>>, Vec<Vec<K>>), yhi: usize| {
        for column in &mut acc {
            column.fill(K::ZERO);
        }
        if grouped {
            let mut tables = vec![[K::ZERO; 256]; lo.len()];
            let first = (base + yhi * block) / 64;
            for y0 in (0..rows).step_by(8) {
                let start = first + y0 * words_per_row;
                let group = &words[start..start + 8 * words_per_row];
                let ones: usize = group.iter().map(|w| w.count_ones() as usize).sum();
                if ones * DENSE_DIVISOR >= 8 * np {
                    for (t, e) in tables.iter_mut().zip(&lo) {
                        *t = om_field::subset::subset_sums8(&e[y0..y0 + 8]);
                    }
                    for c in 0..words_per_row {
                        let column_rows: [u64; 8] =
                            std::array::from_fn(|k| group[k * words_per_row + c]);
                        for (q, bytes) in om_field::subset::column_bytes(&column_rows)
                            .into_iter()
                            .enumerate()
                        {
                            for j in 0..8 {
                                let set = ((bytes >> (8 * j)) & 0xff) as usize;
                                if set != 0 {
                                    let p = 64 * c + 8 * q + j;
                                    for (a, t) in acc.iter_mut().zip(&tables) {
                                        a[p] += t[set];
                                    }
                                }
                            }
                        }
                    }
                } else if ones != 0 {
                    for k in 0..8 {
                        for_each_one(group, k * np, np, |p| {
                            for (a, e) in acc.iter_mut().zip(&lo) {
                                a[p] += e[y0 + k];
                            }
                        });
                    }
                }
            }
        } else {
            for_each_one(words, base + yhi * block, block, |l| {
                let (p, ylo) = (l & (np - 1), l >> (r + 1));
                for (a, e) in acc.iter_mut().zip(&lo) {
                    a[p] += e[ylo];
                }
            });
        }
        for ((t, a), e) in tot.iter_mut().zip(&acc).zip(&hi) {
            for (x, y) in t.chunks_exact_mut(2).zip(a.chunks_exact(2)) {
                if y[0].is_zero() {
                    if !y[1].is_zero() {
                        x[1] += e[yhi].clmul(y[1]);
                    }
                } else if y[1].is_zero() {
                    x[0] += e[yhi].clmul(y[0]);
                } else {
                    let [p0, p1] = clmul_pairs([e[yhi], e[yhi]], [y[0], y[1]]);
                    x[0] += p0;
                    x[1] += p1;
                }
            }
        }
        (tot, acc)
    };
    let nblocks = 1usize << (nfree - h);
    let merge = |mut a: Vec<Vec<U256>>, b: Vec<Vec<U256>>| {
        for (x, y) in a.iter_mut().zip(b) {
            for (x, y) in x.iter_mut().zip(y) {
                *x += y;
            }
        }
        a
    };
    let totals = if nblocks * block >= PAR_BITS {
        (0..nblocks)
            .into_par_iter()
            .fold(zero, one_block)
            .map(|(tot, _)| tot)
            .reduce(|| vec![vec![U256::ZERO; np]; group.len()], merge)
    } else {
        (0..nblocks).fold(zero(), one_block).0
    };
    totals
        .into_iter()
        .map(|column| column.into_iter().map(U256::reduce).collect())
        .collect()
}

/// Regions at least this many bits long are scanned in parallel.
const PAR_BITS: usize = 1 << 16;

/// In [`region_sums`], a group of eight rows uses subset-sum lookups when at
/// least `1 / DENSE_DIVISOR` of its bits are set, and adds bit by bit
/// otherwise. Both paths give the same sums.
const DENSE_DIVISOR: usize = 6;

/// The batching sum-check over packed `G`. Rounds `0..m` (`m =
/// min(bit_rounds, ell)`) are computed from the bits of `G` directly:
/// the round polynomial is `sum_j coef_j eq(P_jr, t) ((1+t) S_j(0) +
/// t S_j(1))` with `S_j(v) = sum_y eq(P_j[r+1..], y) G(chal, v, y)`,
/// and `G(chal, v, y)` expands over the `2^r` bound positions with an
/// `eq` table of the challenges. After `m` rounds the bound `G` and
/// `E` tables (`2^(ell - m)` entries) are built and the remaining
/// rounds run densely, binding in place. Transcript and proof bytes
/// match the dense definition of spec 05 §5.4.
fn split_batch_prove(
    layout: &Layout,
    claims: &[Claim],
    g: &PackedBits,
    transcript: &mut Transcript,
    bit_rounds: usize,
) -> (BatchProof, Vec<K>, K) {
    let ell = layout.ell;
    let words = g.words();
    let values: Vec<K> = claims.iter().map(|c| c.value).collect();
    let (alphas, claim) = batch_claims(&values, transcript, b"mos/batch");
    let by_id = layout.by_id();
    // Keep every public claim and alpha draw above. Equal internal basis
    // functions can share one term even when their claimed values disagree:
    // the original batched value remains checked by the first actual message.
    let mut indices = BTreeMap::<(usize, usize, Vec<u128>), usize>::new();
    let mut terms: Vec<BatchTerm<'_>> = Vec::new();
    for (c, &alpha) in claims.iter().zip(&alphas) {
        let p = by_id[&c.comp_id];
        let key = (
            p.offset,
            p.log_size,
            c.point.iter().map(|x| x.value()).collect(),
        );
        if let Some(&j) = indices.get(&key) {
            terms[j].coef += alpha;
        } else {
            indices.insert(key, terms.len());
            terms.push(BatchTerm {
                offset: p.offset,
                log_size: p.log_size,
                point: &c.point,
                lifted: lift_point(layout, c).expect("claims pre-checked by caller"),
                coef: alpha,
            });
        }
    }
    terms.retain(|t| !t.coef.is_zero());
    // eq factor of one coordinate: the multilinear `1 + p + v` (C2).
    let eqfac = |p: K, v: K| K::ONE + p + v;
    let label = b"mos/batch".as_slice();
    sumcheck_init(transcript, label, claim, ell, 2);
    let msg_label: Vec<u8> = [label, b"/msg"].concat();
    let chal_label: Vec<u8> = [label, b"/chal"].concat();
    let (mut msgs, mut rs): (Vec<Vec<K>>, Vec<K>) = (Vec::new(), Vec::new());
    let mut cur = claim;
    let mut round = |ev: [K; 3], cur: &mut K, rs: &mut Vec<K>, tr: &mut Transcript| -> K {
        assert_eq!(
            ev[0] + ev[1],
            *cur,
            "claim inconsistent with tables (g(0)+g(1) != claim)"
        );
        let msg = vec![ev[0], ev[2]];
        tr.absorb_ks(&msg_label, &msg);
        let chal = tr.challenge_k(&chal_label);
        msgs.push(msg);
        rs.push(chal);
        *cur = interpolate_eval(&ev, chal);
        chal
    };

    use om_sumcheck::{bind_low_and_round_evals_mul2_into, sumcheck_round_evals_mul2};

    // After fixing the low m variables, a term's remaining Eq basis depends
    // only on its aligned region and point[m..]. In particular, claims whose
    // low point coordinates differ can use the same unweighted M table below.
    // Short components in the same 2^m-bit block also share that table: their
    // different subcube indicators remain in the term-specific lifted prefix.
    let m = bit_rounds.min(ell);
    let mut cofactor_indices = BTreeMap::<(usize, usize, Vec<u128>), usize>::new();
    let mut cofactor_sources = Vec::new();
    let mut term_cofactors = Vec::with_capacity(terms.len());
    for (j, t) in terms.iter().enumerate() {
        let key = (
            t.offset >> m,
            t.log_size.saturating_sub(m),
            t.point
                .get(m..)
                .unwrap_or(&[])
                .iter()
                .map(|x| x.value())
                .collect(),
        );
        let id = *cofactor_indices.entry(key).or_insert_with(|| {
            cofactor_sources.push(j);
            cofactor_sources.len() - 1
        });
        term_cofactors.push(id);
    }

    // Phase 1: scan G once per region to build a 2^m-entry M table for each
    // distinct remaining Eq basis, then bind each shared table once per round.
    // Every original term keeps its own prefix, coefficient and cofactor claim.
    if m > 0 {
        let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (id, &j) in cofactor_sources.iter().enumerate() {
            let t = &terms[j];
            let nfree = t.log_size.saturating_sub(m);
            let base = t.offset & !((1usize << m) - 1);
            groups.entry((base, nfree)).or_default().push(id);
        }
        let mut m_tables = vec![Vec::new(); cofactor_sources.len()];
        for ((base, nfree), js) in &groups {
            let group: Vec<&BatchTerm<'_>> =
                js.iter().map(|&id| &terms[cofactor_sources[id]]).collect();
            let sums = region_sums(words, *base, m - 1, *nfree, &group);
            for (&id, sum) in js.iter().zip(sums) {
                m_tables[id] = sum;
            }
        }
        // Unscaled per-term cofactor claims survive zero equality prefixes.
        // The first actual dot checks the original batched public claim; later
        // rounds recover H(0) from C = H(0) + point[r] * H_delta.
        let mut cofactor_claims = vec![K::ZERO; terms.len()];
        let mut slopes = vec![K::ZERO; terms.len()];
        for r in 0..m {
            let mut products = [U256::ZERO; 3];
            for (j, t) in terms.iter().enumerate() {
                let mj = &m_tables[term_cofactors[j]];
                let eq_rem = eq_table(&t.lifted[r + 1..m]);
                let mut u0 = U256::ZERO;
                let mut ud = U256::ZERO;
                for (phi, e) in eq_rem.chunks_exact(2).enumerate() {
                    let at = 4 * phi;
                    if r == 0 {
                        let [p0, p1] = clmul_pairs([e[0], e[1]], [mj[at], mj[at + 2]]);
                        u0 += p0 + p1;
                    }
                    let [p0, p1] =
                        clmul_pairs([e[0], e[1]], [mj[at] + mj[at + 1], mj[at + 2] + mj[at + 3]]);
                    ud += p0 + p1;
                }
                if eq_rem.len() == 1 {
                    if r == 0 {
                        u0 += eq_rem[0].clmul(mj[0]);
                    }
                    ud += eq_rem[0].clmul(mj[0] + mj[1]);
                }
                let sd = ud.reduce();
                let s0 = if r == 0 {
                    u0.reduce()
                } else {
                    cofactor_claims[j] + t.lifted[r] * sd
                };
                cofactor_claims[j] = s0;
                slopes[j] = sd;
                let a0 = t.coef * (K::ONE + t.lifted[r]);
                let [p0, pd] = clmul_pairs([a0, t.coef], [s0, sd]);
                products[0] += p0;
                if r == 0 {
                    products[1] += (a0 + t.coef).clmul(s0 + sd);
                }
                products[2] += pd;
            }
            let g0 = products[0].reduce();
            let g1 = if r == 0 {
                products[1].reduce()
            } else {
                cur + g0
            };
            let gd = products[2].reduce();
            let ev = [g0, g1, g0 + (g0 + g1 + gd).mul_x() + gd.mul_x().mul_x()];
            let chal = round(ev, &mut cur, &mut rs, transcript);
            for (j, t) in terms.iter_mut().enumerate() {
                cofactor_claims[j] += chal * slopes[j];
                t.coef *= eqfac(t.lifted[r], chal);
            }
            for mj in &mut m_tables {
                bind_low_in_place(mj, chal);
            }
        }
    }

    // Bound tables: G(chal, z) by byte lookups over each block of 2^m
    // bits, E(chal, z) from the claims' remaining coordinates.
    let n_rest = 1usize << (ell - m);
    let gt = {
        let _span = om_transcript::profiling::Span::new("mosaic.bind_g");
        bind_bits(words, &eq_table(&rs[..m]), m, n_rest)
    };
    let mut et = vec![K::ZERO; n_rest];
    // Prefixes have now been absorbed into the coefficients, so equal
    // remaining Eq bases can be combined before constructing the dense E.
    let mut cofactor_coefficients = vec![K::ZERO; cofactor_sources.len()];
    for (t, &id) in terms.iter().zip(&term_cofactors) {
        cofactor_coefficients[id] += t.coef;
    }
    for (&j, coefficient) in cofactor_sources.iter().zip(cofactor_coefficients) {
        if coefficient.is_zero() {
            continue;
        }
        let t = &terms[j];
        let z0 = t.offset >> m;
        if t.log_size > m {
            let point = &t.point[m..];
            let h = (point.len() / 2).min(10);
            let lo = eq_table(&point[..h]);
            let hi = eq_table(&point[h..]);
            let dst = &mut et[z0..z0 + (1usize << point.len())];
            let add = |(chunk, &e): (&mut [K], &K)| {
                let coefficient = coefficient * e;
                let mut slots = chunk.chunks_exact_mut(2);
                let mut factors = lo.chunks_exact(2);
                for (d, l) in slots.by_ref().zip(factors.by_ref()) {
                    let [a, b] = mul_pairs([coefficient, coefficient], [l[0], l[1]]);
                    d[0] += a;
                    d[1] += b;
                }
                if let (Some(d), Some(&l)) = (
                    slots.into_remainder().first_mut(),
                    factors.remainder().first(),
                ) {
                    *d += coefficient * l;
                }
            };
            if dst.len() >= PAR_BITS {
                dst.par_chunks_mut(lo.len())
                    .zip(hi.par_iter())
                    .for_each(add);
            } else {
                dst.chunks_mut(lo.len()).zip(&hi).for_each(add);
            }
        } else {
            et[z0] += coefficient;
        }
    }
    drop(terms);

    // Phase 2: reuse buffers and compute the next message while binding.
    let dense_span = om_transcript::profiling::Span::new("mosaic.dense");
    let mut tables = vec![et, gt];
    let mut scratch = Vec::new();
    let mut ev = if m < ell {
        sumcheck_round_evals_mul2(&tables[0], &tables[1]).to_vec()
    } else {
        Vec::new()
    };
    for _ in m..ell {
        let chal = round([ev[0], ev[1], ev[2]], &mut cur, &mut rs, transcript);
        ev = bind_low_and_round_evals_mul2_into(&tables, &mut scratch, chal, Some(cur));
        std::mem::swap(&mut tables, &mut scratch);
    }
    let v_g = tables[1][0];
    drop(dense_span);
    transcript.absorb_k(b"mos/vg", v_g);
    (
        BatchProof {
            sumcheck: SumcheckProof { round_msgs: msgs },
            v_g,
        },
        rs,
        v_g,
    )
}

/// `out[z] = sum_b eqm[b] G(z 2^m + b)` for `z < n_out`: the bit table
/// bound in its low `m` variables. Blocks of at least a byte use one
/// 256-entry subset-sum table per byte position.
fn bind_bits(words: &[u64], eqm: &[K], m: usize, n_out: usize) -> Vec<K> {
    let bit = |i: usize| (words[i / 64] >> (i % 64)) & 1 == 1;
    if m < 3 {
        return (0..n_out)
            .map(|z| {
                (0..1usize << m).fold(
                    K::ZERO,
                    |a, b| {
                        if bit((z << m) + b) { a + eqm[b] } else { a }
                    },
                )
            })
            .collect();
    }
    let nbytes = 1usize << (m - 3);
    let tables: Vec<[K; 256]> = (0..nbytes)
        .map(|q| {
            let mut t = [K::ZERO; 256];
            for p in 1..256usize {
                let low = p.trailing_zeros() as usize;
                t[p] = t[p & (p - 1)] + eqm[8 * q + low];
            }
            t
        })
        .collect();
    if m >= 6 {
        let nwords = 1usize << (m - 6);
        let one_word_aligned = |z: usize| {
            let base_w = z * nwords;
            let mut acc = K::ZERO;
            for wi in 0..nwords {
                let w = words[base_w + wi];
                if w == 0 {
                    continue;
                }
                let tb = &tables[8 * wi..8 * wi + 8];
                acc += tb[0][(w & 0xff) as usize]
                    + tb[1][((w >> 8) & 0xff) as usize]
                    + tb[2][((w >> 16) & 0xff) as usize]
                    + tb[3][((w >> 24) & 0xff) as usize]
                    + tb[4][((w >> 32) & 0xff) as usize]
                    + tb[5][((w >> 40) & 0xff) as usize]
                    + tb[6][((w >> 48) & 0xff) as usize]
                    + tb[7][((w >> 56) & 0xff) as usize];
            }
            acc
        };
        return if n_out << m >= PAR_BITS {
            (0..n_out).into_par_iter().map(one_word_aligned).collect()
        } else {
            (0..n_out).map(one_word_aligned).collect()
        };
    }
    let byte_at = |i: usize| ((words[i / 8] >> (8 * (i % 8))) & 0xff) as usize;
    let one = |z: usize| (0..nbytes).fold(K::ZERO, |a, q| a + tables[q][byte_at(z * nbytes + q)]);
    if n_out << m >= PAR_BITS {
        (0..n_out).into_par_iter().map(one).collect()
    } else {
        (0..n_out).map(one).collect()
    }
}

/// Verify the batching sum-check (spec 05 §5.4), returning
/// `(r_G, v_G)`: the single claim `G̃(r_G) = v_G` handed to ring
/// switching (§5.5).
///
/// # Errors
/// Returns [`VerificationError`] if there are more than [`MAX_CLAIMS`]
/// claims (checked before any transcript interaction), a claim fails
/// [`lift_point`], the sum-check fails, or the final identity
/// `e = (Σ_j α_j eq(R_j, r_G)) · v_G` does not hold.
pub fn batch_verify(
    layout: &Layout,
    claims: &[Claim],
    proof: &BatchProof,
    transcript: &mut Transcript,
) -> Result<(Vec<K>, K), VerificationError> {
    if claims.len() > MAX_CLAIMS {
        return Err(VerificationError::new(
            "too many Mosaic claims for powers-of-gamma batching (05.4)",
        ));
    }
    let lifted: Vec<Vec<K>> = claims
        .iter()
        .map(|c| lift_point(layout, c))
        .collect::<Result<_, _>>()?;
    let values: Vec<K> = claims.iter().map(|c| c.value).collect();
    let (alphas, claim) = batch_claims(&values, transcript, b"mos/batch");
    let (r_g, e) = sumcheck_verify(
        claim,
        layout.ell,
        2,
        &proof.sumcheck,
        transcript,
        b"mos/batch",
    )?;
    transcript.absorb_k(b"mos/vg", proof.v_g);
    let mut ew = K::ZERO;
    for (a, pt) in alphas.iter().zip(&lifted) {
        ew += *a * eq_eval(pt, &r_g);
    }
    if ew * proof.v_g != e {
        return Err(VerificationError::new(
            "mosaic batching: final check failed",
        ));
    }
    Ok((r_g, proof.v_g))
}

/// Evaluate `G̃` directly (Python `direct_G_eval`): `mle_eval` of a bit
/// table.
pub fn direct_g_eval(g: &[bool], point: &[K]) -> K {
    mle_eval_bits(g, point)
}
