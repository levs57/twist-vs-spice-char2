use super::*;

fn element(i: usize, salt: usize) -> K {
    K::from_u128(((i as u128 + 3) << 79) ^ ((salt as u128 + 1) << 37) ^ (17 * i + salt + 1) as u128)
}

fn mle(values: &[K], point: &[K]) -> K {
    assert_eq!(values.len(), 1 << point.len());
    let mut values = values.to_vec();
    for &r in point {
        for j in 0..values.len() / 2 {
            values[j] = (K::ONE + r) * values[2 * j] + r * values[2 * j + 1];
        }
        values.truncate(values.len() / 2);
    }
    values[0]
}

fn product(values: &[K]) -> K {
    values.iter().copied().fold(K::ONE, |a, b| a * b)
}

/// Independent schoolbook coefficient convolution, without the ported quartic
/// identity, equality splitting, unreduced arithmetic, or identity skipping.
fn dense_message(values: &[K], suffix: &[K]) -> [K; 5] {
    assert_eq!(values.len(), 8 << suffix.len());
    let weights = eq_table(suffix);
    let mut message = [K::ZERO; 5];
    for (pair, &weight) in weights.iter().enumerate() {
        let mut coefficients = vec![K::ONE];
        for child in 0..4 {
            let a = values[8 * pair + child];
            let b = values[8 * pair + 4 + child];
            let mut next = vec![K::ZERO; coefficients.len() + 1];
            for (i, &coefficient) in coefficients.iter().enumerate() {
                next[i] += coefficient * a;
                next[i + 1] += coefficient * (a + b);
            }
            coefficients = next;
        }
        for (out, &coefficient) in message.iter_mut().zip(&coefficients) {
            *out += weight * coefficient;
        }
    }
    message
}

fn dense_fold(values: &[K], challenge: K) -> Vec<K> {
    values
        .chunks_exact(8)
        .flat_map(|pair| {
            (0..4).map(move |i| (K::ONE + challenge) * pair[i] + challenge * pair[4 + i])
        })
        .collect()
}

fn binary_layers(values: &[K], log_size: usize) -> Vec<Vec<K>> {
    let mut padded = values.to_vec();
    padded.resize(1 << log_size, K::ONE);
    let mut layers = vec![padded];
    for i in 0..log_size {
        layers.push(
            layers[i]
                .chunks_exact(2)
                .map(|pair| pair[0] * pair[1])
                .collect(),
        );
    }
    layers
}

/// Dense binary-tree definition, sharing only the public transport contract.
fn dense_prove(
    values: &[Vec<K>; 2],
    log_size: usize,
    tr: &mut Transcript,
) -> (GrandProductProof, LeafClaims) {
    let layers = values.each_ref().map(|v| binary_layers(v, log_size));
    let roots = layers.each_ref().map(|t| t[log_size][0]);
    bind_context(log_size, &roots, tr, b"gp");
    let mut proof = GrandProductProof {
        roots,
        layers: Vec::new(),
    };
    let mut point = Vec::new();
    let mut claims = roots;
    let mut layer = log_size;
    while layer > 0 {
        let ll = layer_label(b"gp", layer);
        let lambda = tr.challenge_k(&label(&ll, b"/batch"));
        let step = if layer % 2 == 1 { 1 } else { 2 };
        let mut current = layers.each_ref().map(|t| t[layer - step].clone());
        let mut claim = claims[0] + lambda * claims[1];
        let mut rounds = Vec::new();
        let mut round_point = Vec::new();
        for (round, &at) in point.iter().enumerate() {
            let coefficients = current
                .each_ref()
                .map(|v| dense_message(v, &point[round + 1..]));
            let coefficients: [K; 5] =
                std::array::from_fn(|i| coefficients[0][i] + lambda * coefficients[1][i]);
            assert_eq!(
                claim,
                coefficients[0]
                    + at * coefficients[1..]
                        .iter()
                        .copied()
                        .fold(K::ZERO, |a, b| a + b)
            );
            let tail: [K; 4] = coefficients[1..].try_into().unwrap();
            tr.absorb_ks(&label(&ll, b"/round"), &tail);
            let r = tr.challenge_k(&label(&ll, b"/challenge"));
            rounds.push(tail);
            round_point.push(r);
            claim = polynomial(&coefficients, r);
            current = current.each_ref().map(|v| dense_fold(v, r));
        }
        assert_eq!(claim, product(&current[0]) + lambda * product(&current[1]));
        let next = bind_children(&current, tr, &ll, step);
        claims = current.each_ref().map(|v| mle(v, &next));
        point = next;
        point.extend(round_point);
        proof.layers.push(LayerProof {
            rounds,
            children: current,
        });
        layer -= step;
    }
    (
        proof,
        LeafClaims {
            point,
            values: claims,
        },
    )
}

#[test]
fn complete_port_matches_dense_definition_and_verifier_even_odd_and_prefixes() {
    for log in 0..=10 {
        let size: usize = 1 << log;
        for lengths in [
            [size, size],
            [1, size],
            [size.max(2) / 2, (size.saturating_sub(3)).max(1)],
        ] {
            let lengths = lengths.map(|len| len.min(size));
            let values: [Vec<K>; 2] = [0, 1].map(|tree| {
                (0..lengths[tree])
                    .map(|j| {
                        if j % 29 == 7 {
                            K::ZERO
                        } else {
                            element(j, tree + log)
                        }
                    })
                    .collect()
            });
            let mut tr = Transcript::new_default(b"grandproduct-test");
            let (proof, claims) = prove(
                log,
                values.each_ref().map(|v| Leaves::new(v.clone())),
                &mut tr,
                b"gp",
            )
            .unwrap();
            let mut reference = Transcript::new_default(b"grandproduct-test");
            let (expected_proof, expected_claims) = dense_prove(&values, log, &mut reference);
            assert_eq!(proof, expected_proof, "log={log}, lengths={lengths:?}");
            assert_eq!(claims, expected_claims);
            let mut verifier = Transcript::new_default(b"grandproduct-test");
            assert_eq!(verify(log, &proof, &mut verifier, b"gp").unwrap(), claims);
            for tree in 0..2 {
                let mut padded = values[tree].clone();
                padded.resize(size, K::ONE);
                assert_eq!(claims.values[tree], mle(&padded, &claims.point));
                assert_eq!(proof.roots[tree], product(&padded));
            }
            let next = tr.challenge_k(b"after");
            assert_eq!(next, reference.challenge_k(b"after"));
            assert_eq!(next, verifier.challenge_k(b"after"));
        }
    }
}

#[test]
fn two_declared_padding_ranges_are_exact_and_do_not_merge_actual_data() {
    for log in 2..=10 {
        let size = 1 << log;
        let half = size / 2;
        let ranges = vec![half / 2..half, half + half / 2..size];
        let values = [0, 1].map(|tree| {
            (0..size)
                .map(|j| {
                    if ranges.iter().any(|r| r.contains(&j)) {
                        K::ONE
                    } else {
                        element(j, tree)
                    }
                })
                .collect::<Vec<_>>()
        });
        let mut sparse = Transcript::new_default(b"identity-padding");
        let sparse_result = prove(
            log,
            values
                .each_ref()
                .map(|v| Leaves::with_identity_ranges(v.clone(), ranges.clone()).unwrap()),
            &mut sparse,
            b"gp",
        )
        .unwrap();
        let mut dense = Transcript::new_default(b"identity-padding");
        let dense_result = prove(
            log,
            values.each_ref().map(|v| Leaves::new(v.clone())),
            &mut dense,
            b"gp",
        )
        .unwrap();
        assert_eq!(sparse_result, dense_result);
        assert_eq!(sparse.challenge_k(b"after"), dense.challenge_k(b"after"));
    }
    assert!(Leaves::with_identity_ranges(vec![K::ONE; 8], vec![0..4, 3..7]).is_err());
    assert!(Leaves::with_identity_ranges(vec![K::ONE; 8], vec![7..9]).is_err());
    assert!(Leaves::with_identity_ranges(vec![K::ZERO; 8], vec![0..4]).is_err());
    assert!(Leaves::with_identity_ranges(Vec::new(), Vec::new()).is_err());
}

#[test]
fn quartic_coefficients_and_fused_folds_match_dense_at_boolean_and_zero_eq_factors() {
    for log_rows in [1, 2, 4, 8, 15] {
        let rows = 1 << log_rows;
        let full: Vec<K> = (0..4 * rows).map(|j| element(j, rows)).collect();
        for n in [1, (4 * rows - 3).max(1), 4 * rows] {
            let mut explicit = full[..n].to_vec();
            let mut reference = explicit.clone();
            reference.resize(4 * rows, K::ONE);
            let mut state = LayerState::new(std::mem::take(&mut explicit), rows, Vec::new());
            let point: Vec<K> = (0..log_rows)
                .map(|j| match j % 3 {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => element(j, 17),
                })
                .collect();
            for round in 0..log_rows {
                let dense = dense_message(&reference, &point[round + 1..]);
                let actual = state.round_message(&SplitEq::new(&point[round + 1..]));
                assert_eq!(
                    actual,
                    [
                        dense[1] + dense[2] + dense[3] + dense[4],
                        dense[2],
                        dense[3],
                        dense[4]
                    ]
                );
                // Include the challenge making the removed equality factor zero.
                let challenge = if round % 2 == 0 {
                    K::ONE + point[round]
                } else {
                    point[round]
                };
                reference = dense_fold(&reference, challenge);
                if round + 1 < log_rows {
                    let next =
                        state.fold::<true>(challenge, Some(&SplitEq::new(&point[round + 2..])));
                    let dense_next = dense_message(&reference, &point[round + 2..]);
                    assert_eq!(
                        next,
                        [
                            dense_next[1] + dense_next[2] + dense_next[3] + dense_next[4],
                            dense_next[2],
                            dense_next[3],
                            dense_next[4]
                        ]
                    );
                } else {
                    state.fold::<false>(challenge, None);
                    assert_eq!(state.children().as_slice(), reference.as_slice());
                }
            }
        }
    }
}

#[test]
fn transcript_is_thread_count_independent_and_roots_are_not_silently_equalized() {
    let values = [0, 1].map(|tree| {
        (0..(1 << 12) - tree * 7)
            .map(|j| element(j, tree))
            .collect::<Vec<_>>()
    });
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut tr = Transcript::new_default(b"thread-count");
                let result = prove(
                    12,
                    values.each_ref().map(|v| Leaves::new(v.clone())),
                    &mut tr,
                    b"gp",
                )
                .unwrap();
                (result, tr.challenge_k(b"after"))
            })
    };
    let a = run(1);
    assert_eq!(a, run(4));
    assert_ne!(a.0.0.roots[0], a.0.0.roots[1]);
}

#[test]
fn generated_pair_prepares_exact_levels_without_visiting_padding() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for size in [1usize, 3, 5, 15, 16, 17, 63, 64, 65, 1024, (1 << 16) + 3] {
        let ranges = if size == 1 {
            vec![0..1]
        } else {
            vec![size / 4..size / 2, 3 * size / 4..size]
        };
        let ranges = ranges
            .into_iter()
            .filter(|r| !r.is_empty())
            .collect::<Vec<_>>();
        let visits = AtomicUsize::new(0);
        let prepared = Leaves::from_fn_pair(size, ranges.clone(), |i| {
            assert!(!ranges.iter().any(|r| r.contains(&i)));
            visits.fetch_add(1, Ordering::Relaxed);
            [element(i, 1), element(i, 2)]
        })
        .unwrap();
        assert_eq!(
            visits.load(Ordering::Relaxed),
            size - ranges.iter().map(|r| r.len()).sum::<usize>()
        );
        let reference = [1, 2].map(|salt| {
            Leaves::with_identity_ranges(
                (0..size)
                    .map(|i| {
                        if ranges.iter().any(|r| r.contains(&i)) {
                            K::ONE
                        } else {
                            element(i, salt)
                        }
                    })
                    .collect(),
                ranges.clone(),
            )
            .unwrap()
        });
        for tree in 0..2 {
            assert_eq!(prepared[tree].values, reference[tree].values);
            assert_eq!(prepared[tree].fours, reference[tree].fours);
            assert_eq!(prepared[tree].sixteens, reference[tree].sixteens);
            assert_eq!(prepared[tree].input_len, size);
        }
        if size <= 1024 {
            let log = size.next_power_of_two().ilog2() as usize;
            let a = prove(
                log,
                prepared,
                &mut Transcript::new_default(b"generated-pair"),
                b"gp",
            )
            .unwrap();
            let b = prove(
                log,
                reference,
                &mut Transcript::new_default(b"generated-pair"),
                b"gp",
            )
            .unwrap();
            assert_eq!(a, b);
        }
    }
    assert!(Leaves::from_fn_pair(0, Vec::new(), |_| panic!("empty generator")).is_err());
}

#[test]
fn malformed_proofs_and_every_message_child_and_root_tamper_reject() {
    let values = [0, 1].map(|tree| (0..(1 << 6)).map(|j| element(j, tree)).collect::<Vec<_>>());
    let mut tr = Transcript::new_default(b"tamper");
    let (proof, _) = prove(6, values.map(Leaves::new), &mut tr, b"gp").unwrap();
    let reject = |proof: &GrandProductProof| {
        assert!(verify(6, proof, &mut Transcript::new_default(b"tamper"), b"gp").is_err());
    };
    for tree in 0..2 {
        let mut bad = proof.clone();
        bad.roots[tree] += K::ONE;
        reject(&bad);
    }
    for layer in 0..proof.layers.len() {
        for tree in 0..2 {
            for child in 0..4 {
                let mut bad = proof.clone();
                bad.layers[layer].children[tree][child] += K::ONE;
                reject(&bad);
            }
            let mut bad = proof.clone();
            bad.layers[layer].children[tree].pop();
            reject(&bad);
        }
        for round in 0..proof.layers[layer].rounds.len() {
            for coefficient in 0..4 {
                let mut bad = proof.clone();
                bad.layers[layer].rounds[round][coefficient] += K::ONE;
                reject(&bad);
            }
        }
        let mut bad = proof.clone();
        bad.layers[layer].rounds.push([K::ZERO; 4]);
        reject(&bad);
    }
    let mut bad = proof.clone();
    bad.layers.pop();
    reject(&bad);
    assert!(verify(5, &proof, &mut Transcript::new_default(b"tamper"), b"gp").is_err());
    assert!(
        verify(
            6,
            &proof,
            &mut Transcript::new_default(b"other-statement"),
            b"gp"
        )
        .is_err()
    );
    assert!(
        verify(
            6,
            &proof,
            &mut Transcript::new_default(b"tamper"),
            b"other-label"
        )
        .is_err()
    );
    assert!(
        prove(
            0,
            [Leaves::new(vec![K::ONE; 2]), Leaves::new(vec![K::ONE])],
            &mut Transcript::new_default(b"bad-size"),
            b"gp"
        )
        .is_err()
    );
    assert!(
        verify(
            usize::BITS as usize,
            &proof,
            &mut Transcript::new_default(b"bad-size"),
            b"gp"
        )
        .is_err()
    );
}

#[test]
fn retained_upstream_files_match_provenance() {
    let manifest = include_str!("../UPSTREAM.json");
    assert!(manifest.contains("e5d75f858e91161921c0afe3d9f56300e9a3a03c"));
    for bytes in [
        include_bytes!("../reference/crates/leanvm_core/src/gkr.rs").as_slice(),
        include_bytes!("../reference/crates/verifier/src/bus.rs").as_slice(),
        include_bytes!("../reference/crates/verifier/src/transcript.rs").as_slice(),
    ] {
        let digest: String = om_transcript::Hasher::Sha256
            .digest(bytes)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        assert!(
            manifest.contains(&digest),
            "retained source hash differs from its manifest"
        );
    }
}

/// Reproducible slow definition baseline. Run alone with --ignored --nocapture.
#[test]
#[ignore]
fn benchmark_dense_definition_against_port() {
    use std::hint::black_box;
    use std::time::Instant;
    let env = |key: &str, default| {
        std::env::var(key)
            .ok()
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let (log, threads, samples) = (
        env("GP_LOG_SIZE", 14),
        env("GP_THREADS", 8),
        env("GP_SAMPLES", 5),
    );
    assert!((1..24).contains(&log) && threads > 0 && samples > 0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let input = [0, 1].map(|tree| (0..1 << log).map(|i| element(i, tree)).collect::<Vec<_>>());
        println!(
            "backend,field_bits,trees,log_size,threads,sample,total_ms,quartic_pairs_per_tree"
        );
        let pairs = (((1usize << log) - (1usize << (log % 2))) / 3).saturating_sub(log / 2);
        for sample in 0..samples + 1 {
            let values = input.clone();
            let mut tr = Transcript::new_default(b"dense-benchmark");
            let started = Instant::now();
            let port = prove(log, values.map(Leaves::new), &mut tr, b"gp").unwrap();
            let port_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut tr = Transcript::new_default(b"dense-benchmark");
            let started = Instant::now();
            let dense = dense_prove(&input, log, &mut tr);
            let dense_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(port, dense);
            black_box(&port);
            if sample > 0 {
                println!(
                    "ported-radix4,128,2,{log},{threads},{},{port_ms:.6},{pairs}",
                    sample - 1
                );
                println!(
                    "dense-schoolbook,128,2,{log},{threads},{},{dense_ms:.6},{pairs}",
                    sample - 1
                );
            }
        }
    });
}
