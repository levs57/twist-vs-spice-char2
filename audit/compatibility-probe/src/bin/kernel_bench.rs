//! Small API-level timing probe; all field mathematics comes from the donors.
use flock_core::pcs::ligerito::SumcheckProver;
use flock_field::{F128, F256Unreduced, f128_slice};
use flock_multilinear::{IndexOrder, eq_table, fold_low};
use om_field::{K, U256};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn median_ms<S, T>(mut setup: impl FnMut() -> S, mut run: impl FnMut(&mut S) -> T) -> f64 {
    for _ in 0..5 {
        let mut warm = setup();
        black_box(run(&mut warm));
    }
    let mut times = Vec::new();
    for _ in 0..9 {
        let mut total = Duration::ZERO;
        let mut count = 0;
        while total < Duration::from_millis(20) {
            let mut state = setup();
            let start = Instant::now();
            let result = run(&mut state);
            black_box(&result);
            black_box(&state);
            total += start.elapsed();
            count += 1;
            drop(result);
            drop(state);
        }
        times.push(total.as_secs_f64() * 1000.0 / count as f64);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn main() {
    println!(
        "# RAYON_NUM_THREADS={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default()
    );
    println!(
        "# compile pclmulqdq={} avx512f={} vpclmulqdq={}",
        cfg!(target_feature = "pclmulqdq"),
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "vpclmulqdq")
    );
    println!(
        "# 5 warmups, 9 samples each >=20ms cumulative timed work; median mean-ms/call; setup/clones and destruction excluded; includes returned allocations; no CPU pinning"
    );
    println!("kernel,log2_input_len,om_ms,flock_ms");
    for log in [12, 16, 20] {
        let len = 1usize << log;
        let mut state = 0x243f_6a88_85a3_08d3;
        let a: Vec<F128> = (0..len)
            .map(|_| F128::new(next(&mut state), next(&mut state)))
            .collect();
        let b: Vec<F128> = (0..len)
            .map(|_| F128::new(next(&mut state), next(&mut state)))
            .collect();
        let ka: Vec<K> = a
            .iter()
            .map(|v| K::from_u128(v.lo as u128 | ((v.hi as u128) << 64)))
            .collect();
        let kb: Vec<K> = b
            .iter()
            .map(|v| K::from_u128(v.lo as u128 | ((v.hi as u128) << 64)))
            .collect();
        let r = F128::new(0xc0de_cafe_1234_5678, 0xabcd_abcd_abcd_abcd);
        let kr = K::from_u128(r.lo as u128 | ((r.hi as u128) << 64));

        let om = median_ms(
            || (),
            |_| {
                let mut accum = [K::ZERO; 4];
                for (i, (&x, &y)) in black_box(&ka).iter().zip(black_box(&kb)).enumerate() {
                    accum[i % 4] += x * y;
                }
                accum
            },
        );
        let flock = median_ms(
            || (),
            |_| {
                let mut accum = [F128::ZERO; 4];
                for (i, (&x, &y)) in black_box(&a).iter().zip(black_box(&b)).enumerate() {
                    accum[i % 4] += x * y;
                }
                accum
            },
        );
        println!("dot_reduced,{log},{om:.6},{flock:.6}");

        let om = median_ms(
            || (),
            |_| {
                let mut accum = [U256::ZERO; 4];
                for (i, (&x, &y)) in black_box(&ka).iter().zip(black_box(&kb)).enumerate() {
                    accum[i % 4] += x.clmul(y);
                }
                accum.map(U256::reduce)
            },
        );
        let flock = median_ms(
            || (),
            |_| {
                let mut accum = [F256Unreduced::default(); 4];
                for (i, (&x, &y)) in black_box(&a).iter().zip(black_box(&b)).enumerate() {
                    accum[i % 4] ^= x.mul_unreduced(y);
                }
                accum.map(F256Unreduced::reduce)
            },
        );
        println!("dot_delayed_reduce,{log},{om:.6},{flock:.6}");

        let om = median_ms(
            || ka.clone(),
            |v| om_sumcheck::bind_low_in_place(black_box(v), black_box(kr)),
        );
        let flock = median_ms(|| a.clone(), |v| fold_low(black_box(v), black_box(r)));
        println!("fold_in_place,{log},{om:.6},{flock:.6}");

        let om = median_ms(
            || (),
            |_| om_sumcheck::bind_low(black_box(&ka), black_box(kr)),
        );
        let flock = median_ms(
            || (),
            |_| {
                let mut dst = vec![F128::ZERO; len / 2];
                f128_slice::fold_pairs(black_box(&a), 0, &mut dst, black_box(r));
                dst
            },
        );
        println!("fold_out_allocating,{log},{om:.6},{flock:.6}");

        let om = median_ms(
            || (),
            |_| {
                let mut accum = [K::ZERO; 4];
                for (i, &x) in black_box(&ka).iter().enumerate() {
                    accum[i % 4] += x.square();
                }
                accum
            },
        );
        let flock = median_ms(
            || (),
            |_| {
                let mut accum = [F128::ZERO; 4];
                for (i, &x) in black_box(&a).iter().enumerate() {
                    accum[i % 4] += x.square();
                }
                accum
            },
        );
        println!("sum_squares,{log},{om:.6},{flock:.6}");

        let kp = &ka[..log];
        let fp = &a[..log];
        let om = median_ms(|| (), |_| om_sumcheck::eq_table(black_box(kp)));
        let flock = median_ms(
            || (),
            |_| eq_table(black_box(fp), F128::ONE, IndexOrder::LowToHigh),
        );
        println!("eq_table_allocating,{log},{om:.6},{flock:.6}");

        // Both APIs produce the same quadratic. Flock derives g(1) from
        // the existing claim; OM explicitly computes it. Flock's F128
        // class is a legacy kernel, not its current F256 PCS prover.
        let claim = a.iter().zip(&b).fold(F128::ZERO, |s, (&x, &y)| s + x * y);
        let to_k = |x: F128| K::from_u128(x.lo as u128 | ((x.hi as u128) << 64));
        let full_round = |msg: flock_core::pcs::ligerito::SumcheckMessage| {
            let g2 = msg.u_0
                + flock_field::mul_by_x(claim + msg.u_2)
                + flock_field::mul_by_x(flock_field::mul_by_x(msg.u_2));
            [to_k(msg.u_0), to_k(claim + msg.u_0), to_k(g2)]
        };
        let expected = om_sumcheck::sumcheck_round_evals_mul2(&ka, &kb);
        let (prover, msg) = SumcheckProver::new(a.clone(), b.clone(), claim);
        assert_eq!(expected, full_round(msg));
        drop(prover);
        let om = median_ms(
            || (ka.clone(), kb.clone()),
            |(x, y)| om_sumcheck::sumcheck_round_evals_mul2(black_box(x), black_box(y)),
        );
        let flock = median_ms(
            || (a.clone(), b.clone()),
            |(x, y)| {
                let (prover, msg) =
                    SumcheckProver::new(std::mem::take(x), std::mem::take(y), black_box(claim));
                (prover, full_round(msg))
            },
        );
        println!("quadratic_first_round_flock_legacy_f128,{log},{om:.6},{flock:.6}");
    }
}
