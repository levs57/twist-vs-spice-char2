//! Research only: exact GF128 identities, not a production kernel or benchmark.
use om_field::K;
use om_sumcheck::{bind_low_in_place, eq_table, mle_eval};
use std::fmt::Debug;

#[derive(Default)]
struct Counts {
    fixtures: usize,
    nodes: usize,
    lazy: usize,
    paired: usize,
    assertions: usize,
}

fn check<T: PartialEq + Debug>(a: T, b: T, counts: &mut Counts) {
    counts.assertions += 1;
    assert_eq!(a, b);
}

fn random(seed: &mut u64) -> K {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    K::from_u128((*seed as u128) | ((seed.rotate_left(23) as u128) << 64))
}

fn digit(address: u32, shift: usize, bits: usize) -> usize {
    (address as usize >> shift) & ((1usize << bits) - 1)
}

fn bind_prefix(mut values: Vec<K>, point: &[K]) -> Vec<K> {
    for &r in point {
        bind_low_in_place(&mut values, r);
    }
    values
}

fn lazy_values(addresses: &[u32], shift: usize, bits: usize, base: &[K], point: &[K]) -> Vec<K> {
    let weights = eq_table(point);
    let branches = weights.len();
    let scaled: Vec<Vec<K>> = weights
        .iter()
        .map(|&w| base.iter().map(|&v| w * v).collect())
        .collect();
    (0..addresses.len() / branches)
        .map(|row| {
            (0..branches).fold(K::ZERO, |sum, branch| {
                sum + scaled[branch][digit(addresses[row * branches + branch], shift, bits)]
            })
        })
        .collect()
}

fn paired_values(addresses: &[u32], shift: usize, bits: usize, base: &[K], point: &[K]) -> Vec<K> {
    let weights = eq_table(point);
    let branches = weights.len();
    let radix = base.len();
    assert!(branches >= 2 && radix <= 16);
    // Two adjacent branches become one lookup indexed by both hot digits.
    let pairs: Vec<Vec<K>> = weights
        .chunks_exact(2)
        .map(|w| {
            let left: Vec<K> = base.iter().map(|&v| w[0] * v).collect();
            let right: Vec<K> = base.iter().map(|&v| w[1] * v).collect();
            left.iter()
                .flat_map(|&a| right.iter().map(move |&b| a + b))
                .collect()
        })
        .collect();
    (0..addresses.len() / branches)
        .map(|row| {
            pairs
                .iter()
                .enumerate()
                .fold(K::ZERO, |sum, (pair, table)| {
                    let at = row * branches + 2 * pair;
                    let lo = digit(addresses[at], shift, bits);
                    let hi = digit(addresses[at + 1], shift, bits);
                    sum + table[lo * radix + hi]
                })
        })
        .collect()
}

fn fixture(d: usize, n: usize, address_point_kind: usize, pattern: usize, counts: &mut Counts) {
    let a = 16;
    let ac = a / d;
    let t = 1usize << n;
    let mut seed = 0x243f6a8885a308d3u64 ^ (d + 17 * n + 71 * pattern) as u64;
    let address_point: Vec<K> = (0..a)
        .map(|_| match address_point_kind {
            0 => K::ZERO,
            1 => K::ONE,
            _ => random(&mut seed),
        })
        .collect();
    let addresses: Vec<u32> = (0..t)
        .map(|j| match pattern {
            0 => 0xabcd,
            1 => [0x0102, 0xabcd, 0, 0xffff][j % 4],
            _ => random(&mut seed).value() as u32 & 0xffff,
        })
        .collect();
    let leaves: Vec<Vec<K>> = (0..d)
        .map(|i| {
            let eq = eq_table(&address_point[i * ac..(i + 1) * ac]);
            addresses
                .iter()
                .map(|&address| eq[digit(address, i * ac, ac)])
                .collect()
        })
        .collect();
    let cycle_points = [
        vec![K::ZERO; n],
        vec![K::ONE; n],
        (0..n).map(|_| random(&mut seed)).collect::<Vec<K>>(),
    ];
    let mut subtree = 1;
    while subtree < d {
        let bits = subtree * ac;
        for first in (0..d).step_by(subtree) {
            let shift = first * ac;
            let cache = eq_table(&address_point[shift..shift + bits]);
            let oracle: Vec<K> = addresses
                .iter()
                .map(|&address| cache[digit(address, shift, bits)])
                .collect();
            let pointwise: Vec<K> = (0..t)
                .map(|j| (first..first + subtree).fold(K::ONE, |product, i| product * leaves[i][j]))
                .collect();
            for (&actual, &expected) in oracle.iter().zip(&pointwise) {
                check(actual, expected, counts);
            }
            for cycle_point in &cycle_points {
                // Compare every prefix-bound table, then the fully bound MLE.
                let (mut actual, mut expected) = (oracle.clone(), pointwise.clone());
                for b in 0..=n {
                    check(&actual, &expected, counts);
                    if b < n {
                        bind_low_in_place(&mut actual, cycle_point[b]);
                        bind_low_in_place(&mut expected, cycle_point[b]);
                    }
                }
                check(actual[0], mle_eval(&pointwise, cycle_point), counts);
                for b in 0..=n.min(4) {
                    let expected = bind_prefix(pointwise.clone(), &cycle_point[..b]);
                    check(
                        lazy_values(&addresses, shift, bits, &cache, &cycle_point[..b]),
                        expected.clone(),
                        counts,
                    );
                    counts.lazy += 1;
                    if b > 0 && cache.len() <= 16 {
                        check(
                            paired_values(&addresses, shift, bits, &cache, &cycle_point[..b]),
                            expected,
                            counts,
                        );
                        counts.paired += 1;
                    }
                }
            }
            counts.nodes += 1;
        }
        subtree *= 2;
    }
    counts.fixtures += 1;
}

fn off_cube_counterexample(counts: &mut Counts) {
    let (r0, r1, z) = (K::t(2), K::t(3), K::t(7));
    let left = [K::ONE + r0, r0];
    let right = [K::ONE + r1, r1];
    let pointwise = [left[0] * right[0], left[1] * right[1]];
    let product_of_mles = mle_eval(&left, &[z]) * mle_eval(&right, &[z]);
    let mle_of_product = mle_eval(&pointwise, &[z]);
    check(mle_of_product == product_of_mles, false, counts);
    check(mle_of_product + product_of_mles, z.square() + z, counts);
    println!(
        "off_cube: MLE(pointwise_product)={mle_of_product:?} != product(MLEs)={product_of_mles:?}"
    );
    check(K::ONE + K::ONE, K::ZERO, counts);
    check(K::t(2) == K::ZERO, false, counts);
    println!(
        "characteristic_two: 1+1=0, but polynomial-basis t_2={:?}",
        K::t(2)
    );
}

fn cache_entries(a: usize, d: usize) -> usize {
    let mut entries = 0;
    let mut subtree = 2;
    while subtree < d {
        entries += (d / subtree) * (1usize << (a / d * subtree));
        subtree *= 2;
    }
    entries
}

fn main() {
    let mut counts = Counts::default();
    for d in [2, 4, 8, 16] {
        for n in [0, 1, 4, 6] {
            for point in 0..3 {
                for pattern in 0..3 {
                    fixture(d, n, point, pattern, &mut counts);
                }
            }
        }
    }
    off_cube_counterexample(&mut counts);
    let t = 1usize << 18;
    for (d, expected) in [(4, 512), (8, 576), (16, 608)] {
        let cache = cache_entries(16, d);
        check(cache, expected, &mut counts);
        println!(
            "a=16 d={d}: nonroot_internal_cache={cache} field_elements ({} bytes); full_internal_tables_at_T={t}: {} field_elements",
            cache * 16,
            (d - 2) * t
        );
    }
    println!(
        "PASS fixtures={} subtree_nodes={} lazy_branch_checks={} paired_branch_checks={} exact_assertions={}",
        counts.fixtures, counts.nodes, counts.lazy, counts.paired, counts.assertions
    );
    println!("Exact algebra checks only; no performance measurements or production code changes.");
}
