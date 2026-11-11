//! One pass that evaluates many committed bitplanes at several points.
//!
//! Each point's equality table is split into a low table over the first
//! `h <= 10` coordinates and a high table over the rest, so no full `2^m`
//! table is built. The pass walks blocks of `2^h` rows in parallel. For each
//! local 64-row position it builds, per point, sixteen 16-entry subset-sum
//! tables of the low weights. At large sizes the same tables serve every
//! high block; smaller calls prepare them in reusable scratch. Each requested
//! plane then costs one lookup per four rows. A block's sums are multiplied
//! by the point's high weight once.

use om_field::{K, clmul_pairs};
use om_sumcheck::{PackedBits, eq_table};
use rayon::prelude::*;

/// Planes to evaluate at `points[point]`.
pub struct Request<'a> {
    pub planes: Vec<&'a PackedBits>,
    pub point: usize,
}

/// `out[r][i] = sum_x eq(points[r.point], x) * requests[r].planes[i](x)`.
///
/// Every plane has `2^m` entries, where `m` is the length of every point.
pub fn evaluate(points: &[&[K]], requests: &[Request<'_>]) -> Vec<Vec<K>> {
    evaluate_with_prefixes(points, &vec![None; points.len()], requests)
}

/// Evaluate an optional arbitrary low-coordinate functional per point.
///
/// A prefix with `2^k` weights replaces Eq in the first `k` coordinates;
/// those coordinates of `points[p]` are ignored. The remaining coordinates
/// retain their usual Eq weights. Prefixes fit the at-most-ten-bit low table.
pub fn evaluate_with_prefixes(
    points: &[&[K]],
    prefixes: &[Option<&[K]>],
    requests: &[Request<'_>],
) -> Vec<Vec<K>> {
    let m = points.first().map_or(0, |p| p.len());
    assert!(points.iter().all(|p| p.len() == m), "planes: point lengths");
    assert_eq!(prefixes.len(), points.len(), "planes: prefix count");
    assert!(
        prefixes
            .iter()
            .flatten()
            .all(|w| w.len().is_power_of_two() && w.len() <= 1usize << m.min(10)),
        "planes: prefix shape"
    );
    assert!(
        requests
            .iter()
            .all(|r| r.point < points.len() && r.planes.iter().all(|p| p.len() == 1 << m)),
        "planes: plane lengths"
    );
    let flat = if m < 6 {
        small(points, prefixes, requests)
    } else {
        blocked(points, prefixes, requests, m)
    };
    let mut flat = flat.into_iter();
    requests
        .iter()
        .map(|r| flat.by_ref().take(r.planes.len()).collect())
        .collect()
}

fn low_weights(point: &[K], prefix: Option<&[K]>, h: usize) -> Vec<K> {
    match prefix {
        None => eq_table(&point[..h]),
        Some(prefix) => {
            let k = prefix.len().trailing_zeros() as usize;
            eq_table(&point[k..h])
                .into_iter()
                .flat_map(|high| prefix.iter().map(move |&low| high * low))
                .collect()
        }
    }
}

fn small(points: &[&[K]], prefixes: &[Option<&[K]>], requests: &[Request<'_>]) -> Vec<K> {
    let tables: Vec<Vec<K>> = points
        .iter()
        .zip(prefixes)
        .map(|(p, &prefix)| low_weights(p, prefix, p.len()))
        .collect();
    requests
        .iter()
        .flat_map(|r| {
            let eq = &tables[r.point];
            r.planes
                .iter()
                .map(move |plane| plane.iter_ones().fold(K::ZERO, |s, i| s + eq[i]))
        })
        .collect()
}

/// Sixteen subset-sum tables over consecutive groups of four weights.
fn nibble_tables(weights: &[K], out: &mut [[K; 16]; 16]) {
    for (table, group) in out.iter_mut().zip(weights.chunks_exact(4)) {
        table[0] = K::ZERO;
        for v in 1..16usize {
            table[v] = table[v & (v - 1)] + group[v.trailing_zeros() as usize];
        }
    }
}

#[inline(always)]
fn lookup(tables: &[[K; 16]; 16], word: u64) -> K {
    let mut sum = K::ZERO;
    for (q, table) in tables.iter().enumerate() {
        sum += table[((word >> (4 * q)) & 15) as usize];
    }
    sum
}

/// Large batches amortize a shared low-table bank over at least 256 high
/// blocks. Keep the original streaming kernel for smaller domains, where
/// preparing and retaining the whole bank is slower.
fn blocked(
    points: &[&[K]],
    prefixes: &[Option<&[K]>],
    requests: &[Request<'_>],
    m: usize,
) -> Vec<K> {
    if m >= 18 {
        blocked_cached(points, prefixes, requests, m)
    } else {
        blocked_streaming(points, prefixes, requests, m)
    }
}

fn blocked_streaming(
    points: &[&[K]],
    prefixes: &[Option<&[K]>],
    requests: &[Request<'_>],
    m: usize,
) -> Vec<K> {
    let h = m.min(10);
    let lo: Vec<Vec<K>> = points
        .iter()
        .zip(prefixes)
        .map(|(p, &prefix)| low_weights(p, prefix, h))
        .collect();
    let hi: Vec<Vec<K>> = points.iter().map(|p| eq_table(&p[h..])).collect();
    let mut used = vec![false; points.len()];
    for r in requests {
        used[r.point] = true;
    }
    let total: usize = requests.iter().map(|r| r.planes.len()).sum();
    let words_per_block = 1usize << (h - 6);
    let block = |index: usize| -> Vec<K> {
        let mut sums = vec![K::ZERO; total];
        let mut tables = vec![[[K::ZERO; 16]; 16]; points.len()];
        for local in 0..words_per_block {
            let word = index * words_per_block + local;
            for (p, table) in tables.iter_mut().enumerate() {
                if used[p] {
                    nibble_tables(&lo[p][64 * local..64 * local + 64], table);
                }
            }
            let mut slot = 0;
            for r in requests {
                let table = &tables[r.point];
                for plane in &r.planes {
                    let bits = plane.words()[word];
                    if bits != 0 {
                        sums[slot] += lookup(table, bits);
                    }
                    slot += 1;
                }
            }
        }
        let mut slot = 0;
        for r in requests {
            let weight = hi[r.point][index];
            for _ in &r.planes {
                sums[slot] *= weight;
                slot += 1;
            }
        }
        sums
    };
    let add = |mut a: Vec<K>, b: Vec<K>| {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
        a
    };
    let blocks = 1usize << (m - h);
    if blocks > 1 {
        (0..blocks)
            .into_par_iter()
            .map(block)
            .reduce(|| vec![K::ZERO; total], add)
    } else {
        block(0)
    }
}

fn blocked_cached(
    points: &[&[K]],
    prefixes: &[Option<&[K]>],
    requests: &[Request<'_>],
    m: usize,
) -> Vec<K> {
    let h = m.min(10);
    let lo: Vec<Vec<K>> = points
        .iter()
        .zip(prefixes)
        .map(|(p, &prefix)| low_weights(p, prefix, h))
        .collect();
    let hi: Vec<Vec<K>> = points.iter().map(|p| eq_table(&p[h..])).collect();
    let mut used = vec![false; points.len()];
    for r in requests {
        used[r.point] |= !r.planes.is_empty();
    }
    let total: usize = requests.iter().map(|r| r.planes.len()).sum();
    let words_per_block = 1usize << (h - 6);
    // Low Eq weights depend on the local word, not on the high block. At
    // m=20, rebuilding these in each block repeated the same preparation
    // 1024 times. Keep at most 64 KiB per used point, shared by all workers.
    let tables: Vec<Vec<[[K; 16]; 16]>> = lo
        .iter()
        .zip(&used)
        .map(|(lo, &used)| {
            if !used {
                return Vec::new();
            }
            lo.chunks_exact(64)
                .map(|weights| {
                    let mut table = [[K::ZERO; 16]; 16];
                    nibble_tables(weights, &mut table);
                    table
                })
                .collect()
        })
        .collect();
    // Reuse both accumulation buffers within each Rayon task rather than
    // allocating a result and a table bank for each high block.
    let zero = || (vec![K::ZERO; total], vec![K::ZERO; total]);
    let block = |(mut out, mut sums): (Vec<K>, Vec<K>), index: usize| {
        sums.fill(K::ZERO);
        for local in 0..words_per_block {
            let word = index * words_per_block + local;
            let mut slot = 0;
            for r in requests {
                if r.planes.is_empty() {
                    continue;
                }
                let table = &tables[r.point][local];
                for plane in &r.planes {
                    let bits = plane.words()[word];
                    if bits != 0 {
                        sums[slot] += lookup(table, bits);
                    }
                    slot += 1;
                }
            }
        }
        let mut slot = 0;
        for r in requests {
            let weight = hi[r.point][index];
            for _ in &r.planes {
                out[slot] += sums[slot] * weight;
                slot += 1;
            }
        }
        (out, sums)
    };
    let add = |mut a: Vec<K>, b: Vec<K>| {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
        a
    };
    let blocks = 1usize << (m - h);
    (0..blocks)
        .into_par_iter()
        .fold(zero, block)
        .map(|(out, _)| out)
        .reduce(|| vec![K::ZERO; total], add)
}

/// `[word_eval(v & !f0), word_eval(v & !f1)]` at `point` in one pass: the
/// masked-value claims of the two cycle-side fingerprint tuples. Each row's
/// polynomial-basis word embedding is weighted once and masked twice.
pub fn masked_word_evals(
    values: &[u64],
    flags: [&[bool]; 2],
    point: &[K],
    log_bits: usize,
) -> [K; 2] {
    masked_words(values, flags, point, log_bits)
}

/// One masked word claim without materializing a masked copy of the words.
pub fn masked_word_eval(values: &[u64], flags: &[bool], point: &[K], log_bits: usize) -> K {
    masked_words(values, [flags], point, log_bits)[0]
}

fn masked_words<const N: usize>(
    values: &[u64],
    flags: [&[bool]; N],
    point: &[K],
    log_bits: usize,
) -> [K; N] {
    let m = point.len();
    assert!(
        matches!(N, 1 | 2) && log_bits <= 6,
        "planes: masked word width"
    );
    assert!(
        values.len() == 1 << m && flags.iter().all(|f| f.len() == values.len()),
        "planes: masked word lengths"
    );
    let mask = if log_bits == 6 {
        u64::MAX
    } else {
        (1u64 << (1usize << log_bits)) - 1
    };
    // W[word] is the direct polynomial-basis embedding enc(word). The old
    // bit-Eq lookup followed by multiplication by its normalization factor
    // was exactly this value, so no word-bit table is needed here.
    let h = m.min(10);
    let lo = eq_table(&point[..h]);
    let hi = eq_table(&point[h..]);
    let block = |index: usize| -> [K; N] {
        let base = index << h;
        let mut sums = [om_field::U256::ZERO; N];
        for (x, &weight) in lo.iter().enumerate() {
            let row = base + x;
            let value = values[row] & mask;
            let masked = flags.map(|flag| {
                K::from_u128((value & 0u64.wrapping_sub(u64::from(!flag[row]))) as u128)
            });
            if N == 2 {
                let products = clmul_pairs([weight; 2], [masked[0], masked[1]]);
                sums[0] += products[0];
                sums[1] += products[1];
            } else {
                sums[0] += weight.clmul(masked[0]);
            }
        }
        sums.map(|sum| hi[index] * sum.reduce())
    };
    let add = |a: [K; N], b: [K; N]| std::array::from_fn(|i| a[i] + b[i]);
    let blocks = 1usize << (m - h);
    if blocks > 1 {
        (0..blocks)
            .into_par_iter()
            .map(block)
            .reduce(|| [K::ZERO; N], add)
    } else {
        block(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use om_sumcheck::mle_eval_packed;

    #[test]
    fn masked_word_evals_match_word_eval_of_masked_values() {
        for (m, log_bits) in (0..=6).flat_map(|log_bits| [0, 4, 11].map(|m| (m, log_bits))) {
            let width_mask = if log_bits == 6 {
                u64::MAX
            } else {
                (1u64 << (1 << log_bits)) - 1
            };
            let values: Vec<u64> = (0..1u64 << m)
                .map(|i| i.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            let flags: [Vec<bool>; 2] =
                [0, 1].map(|f| (0..1usize << m).map(|i| (i * 7 + f) % 3 == 0).collect());
            let point: Vec<K> = (0..m).map(|i| K::from_u128(3 + 5 * i as u128)).collect();
            let out = masked_word_evals(&values, [&flags[0], &flags[1]], &point, log_bits);
            for f in 0..2 {
                let masked: Vec<u64> = values
                    .iter()
                    .zip(&flags[f])
                    .map(|(&v, &w)| if w { 0 } else { v & width_mask })
                    .collect();
                assert_eq!(out[f], crate::word_eval(&masked, &point, log_bits));
                assert_eq!(
                    masked_word_eval(&values, &flags[f], &point, log_bits),
                    out[f]
                );
            }
        }
    }

    #[test]
    fn matches_direct_evaluation_for_all_sizes_and_points() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for m in [0, 3, 6, 7, 10, 12, 14, 18] {
            let planes: Vec<PackedBits> = (0..5)
                .map(|i| {
                    let words = (0..(1usize << m).div_ceil(64))
                        .map(|_| if i == 4 { 0 } else { next() })
                        .collect::<Vec<_>>();
                    let mut bits = PackedBits::from_words(words, 1 << m);
                    if m < 6 {
                        let keep = (1u64 << (1 << m)) - 1;
                        bits.words_mut()[0] &= keep;
                    }
                    bits
                })
                .collect();
            let points: Vec<Vec<K>> = (0..4)
                .map(|point| {
                    (0..m)
                        .map(|_| match point {
                            0 => K::ZERO,
                            1 => K::ONE,
                            _ => K::from_u128(next() as u128),
                        })
                        .collect()
                })
                .collect();
            let refs: Vec<&[K]> = points.iter().map(Vec::as_slice).collect();
            let requests = vec![
                Request {
                    planes: vec![&planes[0], &planes[1], &planes[4]],
                    point: 2,
                },
                Request {
                    planes: vec![&planes[2]],
                    point: 0,
                },
                Request {
                    planes: vec![&planes[1], &planes[3]],
                    point: 2,
                },
                Request {
                    planes: vec![&planes[0], &planes[4]],
                    point: 1,
                },
                // An unused point and an empty request do not need a table
                // bank or shift the following requested output positions.
                Request {
                    planes: Vec::new(),
                    point: 3,
                },
            ];
            let out = evaluate(&refs, &requests);
            for (r, values) in requests.iter().zip(&out) {
                for (plane, &value) in r.planes.iter().zip(values) {
                    assert_eq!(value, mle_eval_packed(plane, &points[r.point]), "m={m}");
                }
            }
        }
    }

    #[test]
    fn arbitrary_prefix_functionals_match_full_weighted_bit_sums() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (m, k) in [(0, 0), (3, 2), (6, 6), (8, 6), (12, 6), (18, 6)] {
            let len = 1usize << m;
            let planes: Vec<_> = (0..3)
                .map(|_| {
                    let mut plane = PackedBits::from_words(
                        (0..len.div_ceil(64)).map(|_| next()).collect(),
                        len,
                    );
                    if m < 6 {
                        plane.words_mut()[0] &= (1u64 << len) - 1;
                    }
                    plane
                })
                .collect();
            let mut points: [Vec<K>; 2] =
                std::array::from_fn(|_| (0..m).map(|_| K::from_u128(next() as u128)).collect());
            let prefix: Vec<K> = (0..1usize << k)
                .map(|_| K::from_u128(next() as u128))
                .collect();
            let requests = [
                Request {
                    planes: vec![&planes[0], &planes[2]],
                    point: 0,
                },
                Request {
                    planes: vec![&planes[1]],
                    point: 1,
                },
            ];
            let result = evaluate_with_prefixes(
                &[&points[0], &points[1]],
                &[Some(&prefix), None],
                &requests,
            );
            let high = eq_table(&points[0][k..]);
            let weights: Vec<K> = (0..len)
                .map(|i| prefix[i & (prefix.len() - 1)] * high[i >> k])
                .collect();
            for (plane, &actual) in requests[0].planes.iter().zip(&result[0]) {
                let expected = plane.iter_ones().fold(K::ZERO, |sum, i| sum + weights[i]);
                assert_eq!(actual, expected, "m={m}, k={k}");
            }
            assert_eq!(result[1][0], mle_eval_packed(&planes[1], &points[1]));
            // Placeholder coordinates really are ignored by the functional.
            points[0][..k].fill(K::ONE);
            assert_eq!(
                result,
                evaluate_with_prefixes(
                    &[&points[0], &points[1]],
                    &[Some(&prefix), None],
                    &requests,
                )
            );
        }
    }
}
