//! Standalone field/basis/order conformance oracle; run check_bitz_compatibility.py.
//! The donor word kernels below are compiled directly from the hash-checked
//! extraction, without changing or building the donor workspace/dependencies.
#![allow(dead_code)]

use om_field::{K, U256};
use om_sumcheck::{eq_eval, eq_table, mle_eval};

const REDUCTION: u64 = 0x87;
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[path = "../sources/bitz-reference/crates/field/src/gf128/x86.rs"]
mod native;
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[path = "../sources/bitz-reference/crates/field/src/gf128/aarch64.rs"]
mod native;
#[path = "../sources/bitz-reference/crates/field/src/gf128/portable.rs"]
mod portable;

fn words(value: u128) -> [u64; 2] {
    [value as u64, (value >> 64) as u64]
}
fn integer(value: [u64; 2]) -> u128 {
    u128::from(value[0]) | (u128::from(value[1]) << 64)
}
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn random(state: &mut u64) -> u128 {
    u128::from(next(state)) | (u128::from(next(state)) << 64)
}
fn check<T: PartialEq + core::fmt::Debug>(actual: T, expected: T, checks: &mut usize) {
    assert_eq!(actual, expected);
    *checks += 1;
}

/// Polynomial long division, independent of both closed-form reduction kernels.
fn long_division(mut value: [u64; 4]) -> [u64; 2] {
    for bit in (128..256).rev() {
        if (value[bit / 64] >> (bit % 64)) & 1 != 0 {
            value[bit / 64] ^= 1u64 << (bit % 64);
            for term in [0, 1, 2, 7] {
                let target = bit - 128 + term;
                value[target / 64] ^= 1u64 << (target % 64);
            }
        }
    }
    [value[0], value[1]]
}

fn check_product(a: u128, b: u128, checks: &mut usize) {
    let expected = long_division(portable::clmul128(words(a), words(b)));
    check(
        integer(portable::mul(words(a), words(b))),
        integer(expected),
        checks,
    );
    check(
        (K::from_u128(a) * K::from_u128(b)).value(),
        integer(expected),
        checks,
    );
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    ))]
    {
        check(native::mul(words(a), words(b)), expected, checks);
        check(
            native::wide_reduce(native::wide_mul(words(a), words(b))),
            expected,
            checks,
        );
    }
}

fn main() {
    let mut state = 0x2a7f_34ec_921d_f167;
    let mut checks = 0;
    for i in 0..128 {
        let a = 1u128 << i;
        let k = K::from_u128(a);
        let w = words(a);
        // The exact donor scalar encoding contract is lo-LE followed by hi-LE.
        let mut donor_bytes = [0u8; 16];
        donor_bytes[..8].copy_from_slice(&w[0].to_le_bytes());
        donor_bytes[8..].copy_from_slice(&w[1].to_le_bytes());
        check(k.to_bytes(), donor_bytes, &mut checks);
        check(K::from_bytes(&donor_bytes), k, &mut checks);
        check(donor_bytes[i / 8], 1u8 << (i % 8), &mut checks);
        check(k.mul_x().value(), integer(portable::mul_x(w)), &mut checks);
        for j in 0..128 {
            check_product(a, 1u128 << j, &mut checks);
        }
    }
    for i in 0..2304 {
        let raw = if i < 256 {
            let mut out = [0; 4];
            out[i / 64] = 1u64 << (i % 64);
            out
        } else {
            [
                next(&mut state),
                next(&mut state),
                next(&mut state),
                next(&mut state),
            ]
        };
        let expected = long_division(raw);
        check(portable::reduce(raw), expected, &mut checks);
        check(
            U256(integer([raw[0], raw[1]]), integer([raw[2], raw[3]]))
                .reduce()
                .value(),
            integer(expected),
            &mut checks,
        );
        #[cfg(any(
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
            all(target_arch = "aarch64", target_feature = "aes")
        ))]
        check(native::wide_reduce(raw_native(raw)), expected, &mut checks);
    }
    for _ in 0..2048 {
        let a = random(&mut state);
        let b = random(&mut state);
        check_product(a, b, &mut checks);
        check(
            K::from_u128(a).square().value(),
            integer(portable::square(words(a))),
            &mut checks,
        );
        check(
            (K::from_u128(a) + K::from_u128(b)).value(),
            a ^ b,
            &mut checks,
        );
        #[cfg(any(
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
            all(target_arch = "aarch64", target_feature = "aes")
        ))]
        check(
            native::square(words(a)),
            portable::square(words(a)),
            &mut checks,
        );
    }
    // Original donor eq.rs documents variable i as index bit i. Construct the
    // cube weights by the defining product, rather than reusing either table
    // doubling algorithm, then check ordinary low-first MLE folding as well.
    for n in 0..=10 {
        for case in 0..4 {
            let point: Vec<_> = (0..n)
                .map(|i| match case {
                    0 => K::ZERO,
                    1 => K::ONE,
                    2 => K::from_bit(i % 2 == 1),
                    _ => K::from_u128(random(&mut state)),
                })
                .collect();
            let table = eq_table(&point);
            let mut data = Vec::with_capacity(table.len());
            let mut dot = U256::ZERO;
            for (index, &weight) in table.iter().enumerate() {
                let expected = point.iter().enumerate().fold(K::ONE, |out, (bit, &r)| {
                    out * if index >> bit & 1 != 0 { r } else { K::ONE + r }
                });
                check(weight, expected, &mut checks);
                let boolean: Vec<_> = (0..n)
                    .map(|bit| K::from_bit(index >> bit & 1 != 0))
                    .collect();
                check(eq_eval(&point, &boolean), expected, &mut checks);
                let value = K::from_u128(random(&mut state));
                dot += value.clmul(expected);
                data.push(value);
            }
            check(mle_eval(&data, &point), dot.reduce(), &mut checks);
        }
    }
    println!("BitZ revision cdfb9b73b5857f4dc83761866e475186c6a75ab3");
    println!("Exact donor portable kernels: compiled unchanged by path.");
    println!(
        "Native donor field selected: {}",
        cfg!(any(
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
            all(target_arch = "aarch64", target_feature = "aes")
        ))
    );
    println!("PASS: {checks} exact field/basis/endian/equality/MLE conformance assertions.");
    println!(
        "No full donor crate build or protocol/security/performance claim is made by this probe."
    );
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
fn raw_native(raw: [u64; 4]) -> native::Wide {
    raw
}
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn raw_native(raw: [u64; 4]) -> native::Wide {
    // Exact native wide type is a pair of NEON vectors, used only in this test.
    unsafe { core::mem::transmute(raw) }
}
