//! Audit-only Stage 1.4 algebra/tuple probe; not a Spice implementation.
//! Run with scripts/check_spice_protocol.py. Uses the actual OM K backend.

use om_field::K;
use std::collections::{BTreeMap, BTreeSet};

const ORDER: u128 = (1u128 << 32) - 1;
const FACTORS: [u128; 5] = [3, 5, 17, 257, 65537];
const GENERATOR: u128 = 0x92e3_41fc_ad25_1f6b_0530_a488_3c2f_d05e;
const POLYNOMIAL: u64 = 0x1_3f4d_23d5;
const REDUCTION: u32 = POLYNOMIAL as u32;

fn check<T: PartialEq + std::fmt::Debug>(actual: T, expected: T, count: &mut usize) {
    assert_eq!(actual, expected);
    *count += 1;
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn advance(word: u32) -> u32 {
    (word << 1) ^ (REDUCTION & 0u32.wrapping_sub(word >> 31))
}

/// Independent 32-bit polynomial arithmetic, not truncated GF128 arithmetic.
fn multiply32(mut a: u32, mut b: u32) -> u32 {
    let mut result = 0;
    while b != 0 {
        if b & 1 != 0 {
            result ^= a;
        }
        b >>= 1;
        a = advance(a);
    }
    result
}

fn embed(word: u32, basis: &[K; 32]) -> K {
    let mut result = K::ZERO;
    for (i, &value) in basis.iter().enumerate() {
        if word >> i & 1 != 0 {
            result += value;
        }
    }
    result
}

fn eq(point: &[K], index: usize) -> K {
    point.iter().enumerate().fold(K::ONE, |value, (i, &r)| {
        value * if index >> i & 1 != 0 { r } else { K::ONE + r }
    })
}

fn field_probe(count: &mut usize) {
    let g = K::from_u128(GENERATOR);
    check(K::X.pow(u128::MAX / ORDER), g, count);
    check(g.pow(ORDER), K::ONE, count);
    for factor in FACTORS {
        check(g.pow(ORDER / factor) != K::ONE, true, count);
    }
    check(FACTORS.iter().product::<u128>(), ORDER, count);
    check(g.pow(1u128 << 32), g, count);
    check(K::X.pow(1u128 << 32) != K::X, true, count);

    // Independently recover the minimal polynomial from all Frobenius roots.
    let mut polynomial = vec![K::ONE];
    let mut conjugate = g;
    let mut roots = BTreeSet::new();
    for _ in 0..32 {
        roots.insert(conjugate.value());
        let mut product = vec![K::ZERO; polynomial.len() + 1];
        for (i, &coefficient) in polynomial.iter().enumerate() {
            product[i] += coefficient * conjugate;
            product[i + 1] += coefficient;
        }
        polynomial = product;
        conjugate = conjugate.square();
    }
    check(roots.len(), 32, count);
    check(conjugate, g, count);
    let mut encoded_polynomial = 0u64;
    for (i, &coefficient) in polynomial.iter().enumerate() {
        check(coefficient == K::ZERO || coefficient == K::ONE, true, count);
        encoded_polynomial |= ((coefficient == K::ONE) as u64) << i;
    }
    check(encoded_polynomial, POLYNOMIAL, count);
    check(
        polynomial.iter().rev().fold(K::ZERO, |v, &c| v * g + c),
        K::ZERO,
        count,
    );

    let basis: [K; 32] = std::array::from_fn(|i| g.pow(i as u128));
    for (i, &left) in basis.iter().enumerate() {
        check(left.pow(1u128 << 32), left, count);
        check(embed(1u32 << i, &basis), left, count);
        for (j, &right) in basis.iter().enumerate() {
            check(
                embed(multiply32(1u32 << i, 1u32 << j), &basis),
                left * right,
                count,
            );
        }
    }
    check(embed(0, &basis), K::ZERO, count);
    check(embed(1, &basis), K::ONE, count);
    check(embed(2, &basis), g, count);
    check(std::mem::size_of::<u32>(), 4, count);
    check((1u32 << 31).to_le_bytes(), [0, 0, 0, 128], count);

    let mut state = 0x9f6d_312e_6dc0_873b;
    for _ in 0..2048 {
        let a = next(&mut state) as u32;
        let b = next(&mut state) as u32;
        let ea = embed(a, &basis);
        check(embed(a ^ b, &basis), ea + embed(b, &basis), count);
        check(
            embed(multiply32(a, b), &basis),
            ea * embed(b, &basis),
            count,
        );
        check(embed(advance(a), &basis), g * ea, count);
        check(ea.pow(1u128 << 32), ea, count);
        // Two consecutive counters occupy one u64, with no per-word expansion.
        let packed = u64::from(a) | u64::from(b) << 32;
        check((packed as u32, (packed >> 32) as u32), (a, b), count);
    }

    // Exactly the existing scaled ordinary-MLE construction, with g replacing X.
    let mut point = Vec::new();
    let mut factor = K::ONE;
    let mut power = g;
    for _ in 0..5 {
        check(power != K::ONE, true, count);
        let denominator = K::ONE + power;
        factor *= denominator;
        point.push(power / denominator);
        power = power.square();
    }
    for (i, &value) in basis.iter().enumerate() {
        check(factor * eq(&point, i), value, count);
    }
    for log_rows in [0, 1, 3] {
        let words: Vec<_> = (0..1usize << log_rows)
            .map(|_| next(&mut state) as u32)
            .collect();
        let cycle: Vec<_> = (0..log_rows)
            .map(|_| {
                K::from_u128(u128::from(next(&mut state)) | u128::from(next(&mut state)) << 64)
            })
            .collect();
        let embedded_mle = words.iter().enumerate().fold(K::ZERO, |sum, (j, &word)| {
            sum + eq(&cycle, j) * embed(word, &basis)
        });
        let bit_mle = words.iter().enumerate().fold(K::ZERO, |sum, (j, &word)| {
            sum + eq(&cycle, j)
                * (0..32).fold(K::ZERO, |value, i| {
                    value
                        + if word >> i & 1 != 0 {
                            eq(&point, i)
                        } else {
                            K::ZERO
                        }
                })
        });
        check(factor * bit_mle, embedded_mle, count);
    }
    println!(
        "GF32: generator_integer=0x{GENERATOR:032x} polynomial=0x{POLYNOMIAL:09x} order={ORDER}"
    );
    println!("GF32: exact order/minimal polynomial/basis/random products/32 opening weights PASS; low32 polynomial injection is not a subfield");
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Quad {
    label: u32,
    address: u32,
    value: u64,
    counter: u128,
}

fn quad(label: u32, value: u64, counter: K) -> Quad {
    Quad {
        label,
        address: 0,
        value,
        counter: counter.value(),
    }
}

fn same_multiset(mut pushed: Vec<Quad>, mut pulled: Vec<Quad>) -> bool {
    pushed.sort();
    pulled.sort();
    pushed == pulled
}

fn reaches_initial(start: u32, previous: &BTreeMap<u32, u32>, initial: u32) -> bool {
    let mut label = start;
    let mut visited = BTreeSet::new();
    while label != initial {
        if !visited.insert(label) {
            return false;
        }
        let Some(&parent) = previous.get(&label) else {
            return false;
        };
        label = parent;
    }
    true
}

fn counterexamples(count: &mut usize) {
    let initial = 0;
    let t0 = 256;
    let t1 = 257;
    let g = K::from_u128(GENERATOR);
    let g2 = g.square();

    // Chronological access order is t0 then t1, but the permutation links them
    // as initial -> t1 -> t0 -> final. Every counter is nonzero and the graph
    // is acyclic. Exact multiset equality therefore does not enforce chronology.
    let pushed = vec![quad(initial, 0, K::ONE), quad(t0, 2, g2), quad(t1, 1, g)];
    let pulled = vec![quad(t1, 1, g), quad(initial, 0, K::ONE), quad(t0, 2, g2)];
    check(same_multiset(pushed.clone(), pulled.clone()), true, count);
    check(
        pushed.iter().chain(&pulled).all(|q| q.counter != 0),
        true,
        count,
    );
    let previous = BTreeMap::from([(t0, t1), (t1, initial)]);
    check(reaches_initial(t0, &previous, initial), true, count);
    check(reaches_initial(t1, &previous, initial), true, count);
    check(previous[&t0] > t0, true, count);
    let reads = [1, 0];
    let writes = [2, 1];
    let mut memory = 0;
    check(reads[0] != memory, true, count);
    for (&read, &write) in reads.iter().zip(&writes) {
        check(read != memory, true, count);
        memory = write;
    }
    check(memory != 2, true, count);
    println!("CHRONOLOGY: exact tuple permutation + nonzero GF32 counters + acyclic links PASS, but chronological reads/final memory FAIL as expected");

    // A separate defect if zero is admitted: initial/final memory stays 7,
    // while two access rows form a disconnected zero-counter cycle.
    let pushed = vec![
        quad(initial, 7, K::ONE),
        quad(t0, 7, K::ZERO),
        quad(t1, 7, K::ZERO),
    ];
    let pulled = vec![
        quad(t1, 7, K::ZERO),
        quad(t0, 7, K::ZERO),
        quad(initial, 7, K::ONE),
    ];
    check(same_multiset(pushed.clone(), pulled.clone()), true, count);
    check(g * K::ZERO, K::ZERO, count);
    check(pushed[1..].iter().all(|q| q.counter == 0), true, count);
    check(pulled[..2].iter().all(|q| q.counter == 0), true, count);
    check((pushed[0].value, pulled[2].value), (7, 7), count);
    let previous = BTreeMap::from([(t0, t1), (t1, t0)]);
    check(reaches_initial(t0, &previous, initial), false, count);
    check(reaches_initial(t1, &previous, initial), false, count);
    println!("ZERO CYCLE: exact tuple permutation with unchanged public boundary PASS, but access counters violate GF32* membership as expected");
}

fn main() {
    let mut checks = 0;
    field_probe(&mut checks);
    counterexamples(&mut checks);
    println!("PASS {checks} assertions; audit-only arithmetic/multiset probe, not a full proof or implementation");
}
