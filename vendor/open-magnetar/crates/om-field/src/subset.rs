//! Helpers for summing field elements over the set bits of dense bit rows.
//!
//! Sums of the form `acc[p] += e[y]` over every set bit `(y, p)` of a bit
//! matrix cost one addition per set bit when done bit by bit. For dense
//! rows it is cheaper to take eight rows at a time: [`subset_sums8`] builds
//! the 256 sums of subsets of their eight elements, [`column_bytes`]
//! transposes the eight rows so that each column's eight bits form one
//! byte, and one table lookup per column replaces up to eight additions.

use crate::K;

/// Transpose an 8 × 8 bit matrix stored with row `k` in byte `k` and
/// column `j` in bit `j` of each byte.
#[inline]
pub fn transpose8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// For eight words `rows`, the 64 column bytes: byte `j` of the result's
/// entry `q` has bit `k` equal to bit `8q + j` of `rows[k]`. Column
/// `8q + j` of the eight rows is `out[q] >> (8 j) & 0xff`.
#[inline]
pub fn column_bytes(rows: &[u64; 8]) -> [u64; 8] {
    let mut out = [0u64; 8];
    for (q, o) in out.iter_mut().enumerate() {
        let mut x = 0u64;
        for (k, &w) in rows.iter().enumerate() {
            x |= ((w >> (8 * q)) & 0xff) << (8 * k);
        }
        *o = transpose8(x);
    }
    out
}

/// `t[s] = Σ_{k ∈ s} e[k]` for every subset `s` of `0..8`, with bit `k` of
/// `s` selecting `e[k]`.
#[inline]
pub fn subset_sums8(e: &[K]) -> [K; 256] {
    assert_eq!(e.len(), 8, "subset_sums8: eight elements");
    let mut t = [K::ZERO; 256];
    for (i, &x) in e.iter().enumerate() {
        let b = 1usize << i;
        for s in 0..b {
            t[b | s] = t[s] + x;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn column_bytes_transposes_eight_rows() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..64 {
            let rows: [u64; 8] = std::array::from_fn(|_| next(&mut s));
            let cols = column_bytes(&rows);
            for c in 0..64 {
                let byte = (cols[c / 8] >> (8 * (c % 8))) & 0xff;
                for (k, &w) in rows.iter().enumerate() {
                    assert_eq!((byte >> k) & 1, (w >> c) & 1, "column {c}, row {k}");
                }
            }
        }
    }

    #[test]
    fn subset_sums_match_direct_sums() {
        let mut s = 7u64;
        let e: Vec<K> = (0..8)
            .map(|_| K::from_u128(((next(&mut s) as u128) << 64) | next(&mut s) as u128))
            .collect();
        let t = subset_sums8(&e);
        for (set, &value) in t.iter().enumerate() {
            let direct = (0..8)
                .filter(|k| (set >> k) & 1 == 1)
                .fold(K::ZERO, |acc, k| acc + e[k]);
            assert_eq!(value, direct, "subset {set:#x}");
        }
    }
}
