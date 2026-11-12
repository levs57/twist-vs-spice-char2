use super::*;
use om_sumcheck::eq_table;

fn rand_k(s: &mut u64) -> K {
    fn next(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }
    K::from_u128((u128::from(next(s)) << 64) | u128::from(next(s)))
}

fn scalar_bind(t: &[K], r: K) -> Vec<K> {
    t.chunks_exact(2)
        .map(|p| p[0] + r * (p[0] + p[1]))
        .collect()
}

#[test]
fn paired_dense_folds_match_scalar_at_boolean_points_and_parallel_threshold() {
    let mut seed = 0x517c_2450_9911;
    for half in [0, 1, 2, 3, 10, PAR - 1, PAR, PAR + 1, 2 * PAR] {
        let t: Vec<K> = (0..2 * half).map(|_| rand_k(&mut seed)).collect();
        for r in [K::ZERO, K::ONE, rand_k(&mut seed)] {
            assert_eq!(bind_dense(&t, r), scalar_bind(&t, r), "half={half}");
        }
    }
}

/// A scalar dense reference for the split-eq cycle rounds. Each round sums
/// reduced products directly, without the packed products or delayed sums.
fn scalar_dense_cycle(
    mut ra: Vec<K>,
    mut val: Vec<K>,
    r_c: &[K],
    mut cur: K,
    tr: &mut Transcript,
    msgs: &mut Vec<[K; 2]>,
    r_j: &mut Vec<K>,
) -> (K, K, K) {
    let mut scale = K::ONE;
    for i in 0..r_c.len() {
        let weights = eq_table(&r_c[i + 1..]);
        let mut q = [K::ZERO; 3];
        for (m, &weight) in weights.iter().enumerate() {
            let (a0, a1) = (ra[2 * m], ra[2 * m + 1]);
            let (v0, v1) = (val[2 * m], val[2 * m + 1]);
            q[0] += weight * a0 * v0;
            q[1] += weight * a1 * v1;
            q[2] += weight * (a0 + a1) * (v0 + v1);
        }
        q.iter_mut().for_each(|q| *q *= scale);
        assert_eq!((K::ONE + r_c[i]) * q[0] + r_c[i] * q[1], cur);
        let r = cycle_round(tr, [q[0], q[2]]);
        msgs.push([q[0], q[2]]);
        r_j.push(r);
        let e = eq_lin(r_c[i], r);
        cur = e * quad_at(q[0], q[1], q[2], r);
        scale *= e;
        ra = scalar_bind(&ra, r);
        val = scalar_bind(&val, r);
    }
    (cur, ra[0], val[0])
}

#[test]
fn paired_cycle_rounds_match_scalar_proof_and_transcript_across_thresholds() {
    let mut seed = 0x4e88_aab1_8112;
    for n in [0usize, 1, 2, 12, 13, 14] {
        let ra: Vec<K> = (0..1usize << n).map(|_| rand_k(&mut seed)).collect();
        let val: Vec<K> = (0..1usize << n)
            .map(|j| {
                if j % 5 == 0 {
                    K::ZERO
                } else {
                    rand_k(&mut seed)
                }
            })
            .collect();
        for boolean_point in [false, true] {
            let r_c: Vec<K> = (0..n)
                .map(|i| {
                    if boolean_point {
                        K::from_bit(i % 2 == 0)
                    } else {
                        rand_k(&mut seed)
                    }
                })
                .collect();
            let claim: K = eq_table(&r_c)
                .iter()
                .zip(&ra)
                .zip(&val)
                .fold(K::ZERO, |s, ((&e, &a), &v)| s + e * a * v);
            let mut fast_tr = Transcript::new_default(b"paired-read-cycle");
            let mut want_tr = fast_tr.clone();
            let (mut fast_msgs, mut fast_point) = (Vec::new(), Vec::new());
            let (mut want_msgs, mut want_point) = (Vec::new(), Vec::new());
            let fast = dense_cycle_rounds(
                ra.clone(),
                val.clone(),
                &r_c,
                claim,
                &mut fast_tr,
                &mut fast_msgs,
                &mut fast_point,
            );
            let want = scalar_dense_cycle(
                ra.clone(),
                val.clone(),
                &r_c,
                claim,
                &mut want_tr,
                &mut want_msgs,
                &mut want_point,
            );
            assert_eq!(
                (fast, fast_msgs, fast_point),
                (want, want_msgs, want_point),
                "n={n}"
            );
            assert_eq!(fast_tr.challenge_k(b"after"), want_tr.challenge_k(b"after"));
        }
    }
}

/// Dense two-dimensional oracle for either binding order. The memory table
/// holds the value before each cycle's write; the selector is an actual
/// Boolean address indicator. Only scalar field products and folds are used.
fn dense_read_oracle(
    w: &ReadWitness<'_>,
    r_c: &[K],
    claim: K,
    order: ReadOrder,
    tr: &mut Transcript,
) -> ReadResult {
    let (n, a, t) = (r_c.len(), w.a, w.addr.len());
    let cells = 1usize << a;
    let mut ra = vec![K::ZERO; cells * t];
    let mut val = ra.clone();
    let mut state = w.m0_e.to_vec();
    for j in 0..t {
        for k in 0..cells {
            let at = match order {
                ReadOrder::CycleFirst => j + t * k,
                ReadOrder::AddressFirst => k + cells * j,
            };
            ra[at] = K::from_bit(k == w.addr[j] as usize);
            val[at] = state[k];
        }
        state[w.addr[j] as usize] += w.inc_e[j];
    }
    init(tr, claim, n, a, order);
    let mut cur = claim;
    let (mut cycle, mut address, mut r_j, mut r_k) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let phases = match order {
        ReadOrder::CycleFirst => [true, false],
        ReadOrder::AddressFirst => [false, true],
    };
    let mut scale = K::ONE;
    for is_cycle in phases {
        for i in 0..if is_cycle { n } else { a } {
            let e = eq_table(if is_cycle { &r_c[i + 1..] } else { r_c });
            let mut q = [K::ZERO; 3];
            for m in 0..ra.len() / 2 {
                let weight = if is_cycle {
                    scale * e[m & (e.len() - 1)]
                } else if order == ReadOrder::AddressFirst {
                    e[m >> (a - i - 1)]
                } else {
                    scale
                };
                q[0] += weight * ra[2 * m] * val[2 * m];
                q[1] += weight * ra[2 * m + 1] * val[2 * m + 1];
                q[2] += weight * (ra[2 * m] + ra[2 * m + 1]) * (val[2 * m] + val[2 * m + 1]);
            }
            let r = if is_cycle {
                assert_eq!((K::ONE + r_c[i]) * q[0] + r_c[i] * q[1], cur);
                let r = cycle_round(tr, [q[0], q[2]]);
                cycle.push([q[0], q[2]]);
                r_j.push(r);
                let e = eq_lin(r_c[i], r);
                cur = e * quad_at(q[0], q[1], q[2], r);
                scale *= e;
                r
            } else {
                assert_eq!(q[0] + q[1], cur);
                let r = address_round(tr, [q[0], q[2]]);
                address.push([q[0], q[2]]);
                r_k.push(r);
                cur = quad_at(q[0], q[1], q[2], r);
                r
            };
            ra = scalar_bind(&ra, r);
            val = scalar_bind(&val, r);
        }
    }
    assert_eq!(cur, eq_eval(r_c, &r_j) * ra[0] * val[0]);
    finish(tr, ra[0], val[0]);
    ReadResult {
        proof: ReadProof {
            cycle,
            address,
            ra: ra[0],
            val: val[0],
        },
        r_k,
        r_j,
    }
}

#[test]
fn paired_read_provers_match_dense_grid_oracle_for_both_orders_and_merging() {
    let mut seed = 0x8a17_1234;
    for n in [1usize, 2, 5, 8] {
        let a = 4;
        let addr: Vec<u32> = (0..1usize << n)
            .map(|j| {
                if j % 3 == 0 {
                    0
                } else {
                    rand_k(&mut seed).value() as u32 % 16
                }
            })
            .collect();
        let inc: Vec<K> = (0..1usize << n)
            .map(|j| {
                if j % 5 == 0 {
                    rand_k(&mut seed)
                } else {
                    K::ZERO
                }
            })
            .collect();
        let store: Vec<bool> = inc.iter().map(|x| !x.is_zero()).collect();
        let m0: Vec<K> = (0..1usize << a).map(|_| rand_k(&mut seed)).collect();
        let mut state = m0.clone();
        let rv: Vec<K> = addr
            .iter()
            .zip(&inc)
            .map(|(&ad, &delta)| {
                let read = state[ad as usize];
                state[ad as usize] += delta;
                read
            })
            .collect();
        let r_c: Vec<K> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    K::ONE
                } else {
                    rand_k(&mut seed)
                }
            })
            .collect();
        let claim = eq_table(&r_c)
            .iter()
            .zip(&rv)
            .fold(K::ZERO, |s, (&e, &v)| s + e * v);
        let witness = ReadWitness {
            addr: &addr,
            rv_e: &rv,
            inc_e: &inc,
            store: &store,
            m0_e: &m0,
            a,
        };
        for order in [ReadOrder::CycleFirst, ReadOrder::AddressFirst] {
            let mut want_tr = Transcript::new_default(b"paired-read-grid");
            let want = dense_read_oracle(&witness, &r_c, claim, order, &mut want_tr);
            let tail = want_tr.challenge_k(b"after");
            for merge in [false, true] {
                let mut tr = Transcript::new_default(b"paired-read-grid");
                let fast = prove(&witness, &r_c, claim, order, merge, &mut tr);
                assert_eq!(
                    fast.proof, want.proof,
                    "n={n} order={order:?} merge={merge}"
                );
                assert_eq!(fast.r_k, want.r_k);
                assert_eq!(fast.r_j, want.r_j);
                assert_eq!(tr.challenge_k(b"after"), tail);
                verify(
                    claim,
                    &r_c,
                    a,
                    order,
                    &fast.proof,
                    &mut Transcript::new_default(b"paired-read-grid"),
                )
                .unwrap();
            }
        }
    }
}
