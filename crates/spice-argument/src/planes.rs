//! One pass that evaluates many committed bitplanes at several points.
//!
//! Each point's equality table is split into a low table over the first
//! `h <= 10` coordinates and a high table over the rest, so no full `2^m`
//! table is built. The pass walks blocks of `2^h` rows in parallel. For each
//! 64-row word it builds, per point, sixteen 16-entry subset-sum tables of the
//! low weights, and each requested plane then costs one table lookup per four
//! rows. A block's sums are multiplied by the point's high weight once.

use om_field::K;
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
    let m = points.first().map_or(0, |p| p.len());
    assert!(points.iter().all(|p| p.len() == m), "planes: point lengths");
    assert!(
        requests
            .iter()
            .all(|r| r.point < points.len() && r.planes.iter().all(|p| p.len() == 1 << m)),
        "planes: plane lengths"
    );
    let flat = if m < 6 {
        small(points, requests)
    } else {
        blocked(points, requests, m)
    };
    let mut flat = flat.into_iter();
    requests
        .iter()
        .map(|r| flat.by_ref().take(r.planes.len()).collect())
        .collect()
}

fn small(points: &[&[K]], requests: &[Request<'_>]) -> Vec<K> {
    let tables: Vec<Vec<K>> = points.iter().map(|p| eq_table(p)).collect();
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

fn blocked(points: &[&[K]], requests: &[Request<'_>], m: usize) -> Vec<K> {
    let h = m.min(10);
    let lo: Vec<Vec<K>> = points.iter().map(|p| eq_table(&p[..h])).collect();
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

/// `[word_eval(v & !f0), word_eval(v & !f1)]` at `point` in one pass: the
/// masked-value claims of the two cycle-side fingerprint tuples. Each row's
/// word is evaluated once through byte subset-sum tables and masked twice.
pub fn masked_word_evals(
    values: &[u64],
    flags: [&[bool]; 2],
    point: &[K],
    log_bits: usize,
) -> [K; 2] {
    let m = point.len();
    assert!(
        values.len() == 1 << m && flags.iter().all(|f| f.len() == values.len()),
        "planes: masked word lengths"
    );
    let (word_point, factor) = crate::word_encoding(log_bits);
    let weights = eq_table(&word_point);
    let bytes: Vec<[K; 256]> = weights
        .chunks(8)
        .map(|group| {
            let mut table = [K::ZERO; 256];
            for v in 1..256usize {
                let bit = v.trailing_zeros() as usize;
                table[v] = table[v & (v - 1)] + group.get(bit).copied().unwrap_or(K::ZERO);
            }
            table
        })
        .collect();
    let mask = if weights.len() >= 64 {
        u64::MAX
    } else {
        (1u64 << weights.len()) - 1
    };
    let word = |v: u64| {
        let v = v & mask;
        bytes.iter().enumerate().fold(K::ZERO, |sum, (i, table)| {
            sum + table[((v >> (8 * i)) & 0xff) as usize]
        })
    };
    let h = m.min(10);
    let lo = eq_table(&point[..h]);
    let hi = eq_table(&point[h..]);
    let block = |index: usize| -> [K; 2] {
        let base = index << h;
        let mut sums = [om_field::U256::ZERO; 2];
        for (x, &weight) in lo.iter().enumerate() {
            let row = base + x;
            let value = word(values[row]).value();
            for (sum, flag) in sums.iter_mut().zip(flags) {
                let masked = K::from_u128(value & 0u128.wrapping_sub(u128::from(!flag[row])));
                *sum += weight.clmul(masked);
            }
        }
        sums.map(|sum| hi[index] * sum.reduce())
    };
    let blocks = 1usize << (m - h);
    let sum = if blocks > 1 {
        (0..blocks)
            .into_par_iter()
            .map(block)
            .reduce(|| [K::ZERO; 2], |a, b| [a[0] + b[0], a[1] + b[1]])
    } else {
        block(0)
    };
    sum.map(|s| factor * s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use om_sumcheck::mle_eval_packed;

    #[test]
    fn masked_word_evals_match_word_eval_of_masked_values() {
        for (m, log_bits) in [(0, 6), (4, 3), (11, 6), (12, 0)] {
            let width_mask = if log_bits == 6 {
                u64::MAX
            } else {
                (1u64 << (1 << log_bits)) - 1
            };
            let values: Vec<u64> = (0..1u64 << m)
                .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) & width_mask)
                .collect();
            let flags: [Vec<bool>; 2] =
                [0, 1].map(|f| (0..1usize << m).map(|i| (i * 7 + f) % 3 == 0).collect());
            let point: Vec<K> = (0..m).map(|i| K::from_u128(3 + 5 * i as u128)).collect();
            let out = masked_word_evals(&values, [&flags[0], &flags[1]], &point, log_bits);
            for f in 0..2 {
                let masked: Vec<u64> = values
                    .iter()
                    .zip(&flags[f])
                    .map(|(&v, &w)| if w { 0 } else { v })
                    .collect();
                assert_eq!(out[f], crate::word_eval(&masked, &point, log_bits));
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
        for m in [0, 3, 6, 7, 10, 12] {
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
            let points: Vec<Vec<K>> = (0..3)
                .map(|_| (0..m).map(|_| K::from_u128(next() as u128)).collect())
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
            ];
            let out = evaluate(&refs, &requests);
            for (r, values) in requests.iter().zip(&out) {
                for (plane, &value) in r.planes.iter().zip(values) {
                    assert_eq!(value, mle_eval_packed(plane, &points[r.point]), "m={m}");
                }
            }
        }
    }
}
