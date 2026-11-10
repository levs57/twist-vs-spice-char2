use super::*;
use crate::product::{prove_product_relation_into, verify_product_relation};

fn value(i: usize) -> K {
    K::from_u128((i as u128 + 7).wrapping_mul(0x9e3779b97f4a7c15_6a09e667f3bcc909))
}

fn eq_table(point: &[K]) -> Vec<K> {
    (0..1usize << point.len())
        .map(|row| {
            point.iter().enumerate().fold(K::ONE, |a, (bit, &r)| {
                a * if row >> bit & 1 == 0 { K::ONE + r } else { r }
            })
        })
        .collect()
}

fn mle(values: &[K], point: &[K]) -> K {
    assert_eq!(values.len(), 1 << point.len());
    values
        .iter()
        .zip(eq_table(point))
        .fold(K::ZERO, |a, (&v, w)| a + v * w)
}

// Dense algebra oracle: regenerate each remaining Eq table, use ordinary
// reduced products, and allocate fresh folded columns each round. It shares
// neither the donor suffix layout nor its optimized round reduction.
fn product_oracle(
    left: &[K],
    right: &[K],
    point: &[K],
    tr: &mut Transcript,
) -> (ProductRelationProof, Vec<K>, [K; 2]) {
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    let mut prefix = K::ONE;
    let mut challenges = Vec::new();
    let mut rounds = Vec::new();
    for round in (0..point.len()).rev() {
        let half = left.len() / 2;
        let eq = eq_table(&point[..round]);
        let mut endpoint = K::ZERO;
        let mut infinity = K::ZERO;
        for i in 0..half {
            let selected = if point[round] == K::ZERO { i + half } else { i };
            endpoint += eq[i] * left[selected] * right[selected];
            infinity += eq[i] * (left[i] + left[i + half]) * (right[i] + right[i + half]);
        }
        let message = [prefix * endpoint, prefix * infinity];
        rounds.push(message);
        write(tr, &message);
        let r = challenge(tr);
        challenges.push(r);
        left = (0..half)
            .map(|i| left[i] + r * (left[i] + left[i + half]))
            .collect();
        right = (0..half)
            .map(|i| right[i] + r * (right[i] + right[i + half]))
            .collect();
        prefix *= K::ONE + r + point[round];
    }
    let evaluations = [left[0], right[0]];
    write(tr, &evaluations);
    challenges.reverse();
    (
        ProductRelationProof {
            rounds,
            evaluations,
        },
        challenges,
        evaluations,
    )
}

struct DenseTree {
    nums: Vec<Vec<K>>,
    dens: Vec<Vec<K>>,
    ac: Vec<Vec<K>>,
}

impl DenseTree {
    fn new(nums: &[K], dens: &[K], n: usize, np: K, dp: K) -> Self {
        let mut a = vec![np; 1 << n];
        let mut b = vec![dp; 1 << n];
        a[..nums.len()].copy_from_slice(nums);
        b[..dens.len()].copy_from_slice(dens);
        let mut tree = Self {
            nums: vec![a],
            dens: vec![b],
            ac: Vec::new(),
        };
        for level in 0..n {
            let mut nums = Vec::new();
            let mut dens = Vec::new();
            let mut ac = Vec::new();
            for i in 0..tree.nums[level].len() / 2 {
                let [a, c] = [tree.nums[level][2 * i], tree.nums[level][2 * i + 1]];
                let [b, d] = [tree.dens[level][2 * i], tree.dens[level][2 * i + 1]];
                nums.push(a * d + b * c);
                dens.push(b * d);
                ac.push(a * c);
            }
            tree.nums.push(nums);
            tree.dens.push(dens);
            tree.ac.push(ac);
        }
        tree
    }

    fn prove(&self, tr: &mut Transcript) -> (FractionTreeProof, FractionClaim) {
        let n = self.ac.len();
        let mut point = Vec::new();
        let mut layers = Vec::new();
        for level in (0..n).rev() {
            let ac_eval = mle(&self.ac[level], &point);
            write(tr, &[ac_eval]);
            let gamma = challenge(tr);
            assert_ne!(gamma, K::ZERO);
            let left: Vec<K> = self.nums[level]
                .chunks_exact(2)
                .zip(self.dens[level].chunks_exact(2))
                .map(|(a, b)| b[0] + gamma * a[0])
                .collect();
            let right: Vec<K> = self.nums[level]
                .chunks_exact(2)
                .zip(self.dens[level].chunks_exact(2))
                .map(|(a, b)| b[1] + gamma * a[1])
                .collect();
            let (product, product_point, _) = product_oracle(&left, &right, &point, tr);
            let even: Vec<K> = self.nums[level].chunks_exact(2).map(|v| v[0]).collect();
            let odd: Vec<K> = self.nums[level].chunks_exact(2).map(|v| v[1]).collect();
            let numerators = [mle(&even, &product_point), mle(&odd, &product_point)];
            write(tr, &numerators);
            let selector = challenge(tr);
            point = vec![selector];
            point.extend_from_slice(&product_point);
            layers.push(FractionLayerProof {
                ac_eval,
                product,
                numerators,
            });
        }
        let num = mle(&self.nums[0], &point);
        let den = mle(&self.dens[0], &point);
        (
            FractionTreeProof { layers },
            FractionClaim { point, num, den },
        )
    }
}

fn rational_oracle(
    left: &DenseTree,
    right: &DenseTree,
    tr: &mut Transcript,
    label: &[u8],
) -> (RationalProof, RationalClaims) {
    let [ln, rn] = [left.ac.len(), right.ac.len()];
    bind(tr, label, b"bitz/logup-cut/rational/v1", &[ln, rn]);
    let roots = [
        left.nums[ln][0],
        left.dens[ln][0],
        right.nums[rn][0],
        right.dens[rn][0],
    ];
    write(tr, &roots);
    let (lp, left) = left.prove(tr);
    let (rp, right) = right.prove(tr);
    (
        RationalProof {
            roots,
            left: lp,
            right: rp,
        },
        RationalClaims { left, right },
    )
}

#[test]
fn product_relation_matches_dense_oracle_and_low_first_mles() {
    for n in [0, 1, 2, 3, 7, 13] {
        let left: Vec<K> = (0..1 << n).map(value).collect();
        let right: Vec<K> = (0..1 << n).map(|i| value(i + 33)).collect();
        for kind in 0..3 {
            let point: Vec<K> = (0..n)
                .map(|i| match kind {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => value(i + 71),
                })
                .collect();
            let mut a = Transcript::new_default(b"product-test");
            let mut b = a.clone();
            let mut witness = [left.clone(), right.clone()].concat();
            let mut storage = vec![K::ZERO; (1usize << n).max(1)];
            let actual = prove_product_relation_into(&mut a, &point, &mut witness, &mut storage);
            let expected = product_oracle(&left, &right, &point, &mut b);
            assert_eq!(actual, expected);
            assert_eq!(a, b);
            assert_eq!(actual.2, [mle(&left, &actual.1), mle(&right, &actual.1)]);
            let claim = left
                .iter()
                .zip(&right)
                .zip(eq_table(&point))
                .fold(K::ZERO, |s, ((&l, &r), w)| s + w * l * r);
            let mut v = Transcript::new_default(b"product-test");
            assert_eq!(
                verify_product_relation(&mut v, claim, &point, &actual.0).unwrap(),
                (actual.1, actual.2)
            );
            assert_eq!(v, a);
        }
    }
}

#[test]
fn product_relation_handles_zero_equality_prefix() {
    let left: Vec<K> = (0..8).map(value).collect();
    let right: Vec<K> = (0..8).map(|i| value(i + 17)).collect();
    let mut point = vec![value(41), K::ONE, K::ONE];
    let probe = product_oracle(
        &left,
        &right,
        &point,
        &mut Transcript::new_default(b"zero-prefix"),
    );
    // The first message is independent of the current nonzero Eq coordinate.
    point[2] = K::ONE + probe.1[2];
    assert_ne!(point[2], K::ZERO);
    let mut a = Transcript::new_default(b"zero-prefix");
    let mut b = a.clone();
    let mut witness = [left.clone(), right.clone()].concat();
    let actual = prove_product_relation_into(&mut a, &point, &mut witness, &mut [K::ZERO; 8]);
    assert_eq!(actual, product_oracle(&left, &right, &point, &mut b));
    assert_eq!(actual.0.rounds[1], [K::ZERO; 2]);
    assert_eq!(actual.0.rounds[2], [K::ZERO; 2]);
    let claim = left
        .iter()
        .zip(&right)
        .zip(eq_table(&point))
        .fold(K::ZERO, |s, ((&l, &r), w)| s + w * l * r);
    let mut v = Transcript::new_default(b"zero-prefix");
    assert_eq!(
        verify_product_relation(&mut v, claim, &point, &actual.0).unwrap(),
        (actual.1, actual.2)
    );
    assert_eq!(a, v);
}

#[test]
fn rational_matches_dense_oracle_padding_and_different_dimensions() {
    for (active, ln, rn) in [
        (1, 0, 0),
        (1, 0, 3),
        (1, 1, 2),
        (3, 2, 4),
        (5, 3, 3),
        (31, 5, 6),
        (8195, 14, 15),
    ] {
        for zero in [false, true] {
            let nums: Vec<K> = (0..active)
                .map(|i| if zero { K::ZERO } else { value(i) })
                .collect();
            let dens: Vec<K> = (0..active).map(|i| value(i + 91)).collect();
            let np = K::ZERO;
            let dp = value(881);
            let mut left = FractionTreeWitness::default();
            let mut right = FractionTreeWitness::default();
            left.rebuild(&nums, &dens, ln, np, dp).unwrap();
            right.rebuild(&nums, &dens, rn, np, dp).unwrap();
            let lo = DenseTree::new(&nums, &dens, ln, np, dp);
            let ro = DenseTree::new(&nums, &dens, rn, np, dp);
            assert_eq!(left.root(), (lo.nums[ln][0], lo.dens[ln][0]));
            assert_eq!(right.root(), (ro.nums[rn][0], ro.dens[rn][0]));
            let mut a = Transcript::new_default(b"rational-test");
            let mut b = a.clone();
            let actual = prove_rational(&mut left, &mut right, &mut a, b"rational").unwrap();
            let expected = rational_oracle(&lo, &ro, &mut b, b"rational");
            assert_eq!(actual, expected, "{active} {ln} {rn} zero={zero}");
            assert_eq!(a, b);
            let mut v = Transcript::new_default(b"rational-test");
            assert_eq!(
                verify_rational(ln, rn, &actual.0, &mut v, b"rational").unwrap(),
                actual.1
            );
            assert_eq!(a, v);
        }
    }
}

#[test]
fn weighted_repeats_have_characteristic_two_safe_pushforward() {
    // Both addresses repeat an even number of times. Unweighted integer
    // multiplicities would be zero in characteristic two; Eq weights do not.
    let addresses = [0usize, 0, 1, 1, 1, 1, 0, 0];
    let point = [value(87), value(88), value(89)];
    let weights = eq_table(&point);
    let mut histogram = [K::ZERO; 2];
    for (&a, &w) in addresses.iter().zip(&weights) {
        histogram[a] += w;
    }
    assert!(histogram.iter().any(|&m| m != K::ZERO));
    let tau = value(400);
    let left_den: Vec<K> = addresses
        .iter()
        .map(|&a| tau + K::from_u128(a as u128))
        .collect();
    let right_den = [tau, tau + K::ONE];
    let mut left = FractionTreeWitness::default();
    let mut right = FractionTreeWitness::default();
    left.rebuild(&weights, &left_den, 3, K::ZERO, tau).unwrap();
    right
        .rebuild(&histogram, &right_den, 1, K::ZERO, tau)
        .unwrap();
    let mut p = Transcript::new_default(b"weighted");
    let (proof, claims) = prove_rational(&mut left, &mut right, &mut p, b"ratio").unwrap();
    let mut v = Transcript::new_default(b"weighted");
    assert_eq!(
        verify_rational(3, 1, &proof, &mut v, b"ratio").unwrap(),
        claims
    );
    assert_eq!(mle(&weights, &claims.left.point), claims.left.num);
    assert_eq!(mle(&left_den, &claims.left.point), claims.left.den);
    assert_eq!(mle(&histogram, &claims.right.point), claims.right.num);
    assert_eq!(mle(&right_den, &claims.right.point), claims.right.den);
}

#[test]
fn invalid_inputs_roots_and_proof_shapes_reject() {
    let mut a = FractionTreeWitness::default();
    for (nums, dens, n, dp, expected) in [
        (vec![], vec![], 0, K::ONE, Error::InvalidShape),
        (
            vec![K::ONE; 2],
            vec![K::ONE],
            1,
            K::ONE,
            Error::InvalidShape,
        ),
        (
            vec![K::ONE; 2],
            vec![K::ONE; 2],
            0,
            K::ONE,
            Error::InvalidShape,
        ),
        (
            vec![K::ONE],
            vec![K::ZERO],
            0,
            K::ONE,
            Error::ZeroDenominator,
        ),
        (
            vec![K::ONE],
            vec![K::ONE],
            0,
            K::ZERO,
            Error::ZeroDenominator,
        ),
        (
            vec![K::ONE],
            vec![K::ONE],
            usize::MAX,
            K::ONE,
            Error::InvalidShape,
        ),
    ] {
        assert_eq!(a.rebuild(&nums, &dens, n, K::ZERO, dp), Err(expected));
    }
    let mut tr = Transcript::new_default(b"invalid");
    assert_eq!(
        prove_rational(&mut a, &mut FractionTreeWitness::default(), &mut tr, b"r"),
        Err(Error::InvalidShape)
    );
    a.rebuild(&[K::ONE], &[K::ONE], 0, K::ZERO, K::ONE).unwrap();
    let mut b = FractionTreeWitness::default();
    b.rebuild(&[K::ZERO], &[K::ONE], 0, K::ZERO, K::ONE)
        .unwrap();
    assert_eq!(
        prove_rational(&mut a, &mut b, &mut tr, b"r"),
        Err(Error::InvalidClaim)
    );
    let nums: Vec<K> = (0..8).map(value).collect();
    let dens: Vec<K> = (20..28).map(value).collect();
    a.rebuild(&nums, &dens, 3, K::ZERO, K::ONE).unwrap();
    b.rebuild(&nums, &dens, 3, K::ZERO, K::ONE).unwrap();
    let (proof, _) = prove_rational(
        &mut a,
        &mut b,
        &mut Transcript::new_default(b"mutations"),
        b"r",
    )
    .unwrap();
    let verify = |proof: &RationalProof| {
        verify_rational(
            3,
            3,
            proof,
            &mut Transcript::new_default(b"mutations"),
            b"r",
        )
    };
    let mut bad = proof.clone();
    bad.roots[1] = K::ZERO;
    assert_eq!(verify(&bad), Err(Error::ZeroDenominator));
    let mut bad = proof.clone();
    bad.roots[0] += K::ONE;
    assert!(verify(&bad).is_err());
    let mut bad = proof.clone();
    bad.left.layers.pop();
    assert_eq!(verify(&bad), Err(Error::MalformedProof));
    let mut bad = proof.clone();
    bad.right.layers[1].product.rounds.push([K::ZERO; 2]);
    assert_eq!(verify(&bad), Err(Error::MalformedProof));
    for side in 0..2 {
        for layer in 0..3 {
            for field in 0..3 {
                let mut bad = proof.clone();
                let l = if side == 0 {
                    &mut bad.left.layers[layer]
                } else {
                    &mut bad.right.layers[layer]
                };
                match field {
                    0 => l.ac_eval += K::ONE,
                    1 => l.product.evaluations[0] += K::ONE,
                    _ => l.numerators[0] += K::ONE,
                }
                // The final numerators are terminal claims, deliberately left for
                // the caller's PCS authentication, so only nonterminal edits must
                // reject inside the reduction itself.
                if side == 1 && layer == 2 && field == 2 {
                    let claims = verify(&bad).unwrap();
                    let claim = if side == 0 { claims.left } else { claims.right };
                    assert_ne!(mle(&nums, &claim.point), claim.num);
                } else {
                    assert!(
                        verify(&bad).is_err(),
                        "side={side} layer={layer} field={field}"
                    );
                }
            }
        }
    }
}

#[test]
fn inner_product_exact_dense_round_oracle_and_authentication() {
    for n in [0, 1, 2, 7, 13] {
        for zero in [false, true] {
            let a: Vec<K> = (0..1 << n)
                .map(|i| if zero { K::ZERO } else { value(i) })
                .collect();
            let b: Vec<K> = (0..1 << n).map(|i| value(i + 33)).collect();
            let claim = a.iter().zip(&b).fold(K::ZERO, |s, (&a, &b)| s + a * b);
            let mut tr = Transcript::new_default(b"inner");
            let (proof, out) = prove_inner_product(&a, &b, claim, &mut tr, b"ip").unwrap();
            let mut oracle = Transcript::new_default(b"inner");
            bind(&mut oracle, b"ip", b"bitz/logup-cut/inner-product/v1", &[n]);
            let mut aa = a.clone();
            let mut bb = b.clone();
            let mut point = Vec::new();
            let mut remaining = claim;
            for message in &proof.rounds {
                let mut c0 = K::ZERO;
                let mut c1 = K::ZERO;
                for (a, b) in aa.chunks_exact(2).zip(bb.chunks_exact(2)) {
                    c0 += a[0] * b[0];
                    c1 += a[0] * b[1] + a[1] * b[0];
                }
                assert_eq!(*message, [c0, c1]);
                write(&mut oracle, message);
                let r = challenge(&mut oracle);
                point.push(r);
                remaining = c0 + r * (c1 + r * (remaining + c1));
                aa = aa
                    .chunks_exact(2)
                    .map(|x| x[0] + r * (x[0] + x[1]))
                    .collect();
                bb = bb
                    .chunks_exact(2)
                    .map(|x| x[0] + r * (x[0] + x[1]))
                    .collect();
            }
            write(&mut oracle, &[aa[0]]);
            assert_eq!(remaining, aa[0] * bb[0]);
            assert_eq!(oracle, tr);
            assert_eq!(out.point, point);
            assert_eq!(out.value, mle(&a, &out.point));
            let mut v = Transcript::new_default(b"inner");
            assert_eq!(
                verify_inner_product(n, claim, |p| mle(&b, p), &proof, &mut v, b"ip").unwrap(),
                out
            );
            assert_eq!(v, tr);
            let mut bad = proof.clone();
            bad.value += K::ONE;
            assert!(
                verify_inner_product(
                    n,
                    claim,
                    |p| mle(&b, p),
                    &bad,
                    &mut Transcript::new_default(b"inner"),
                    b"ip"
                )
                .is_err()
            );
            let mut bad = proof.clone();
            bad.rounds.push([K::ZERO; 2]);
            assert_eq!(
                verify_inner_product(
                    n,
                    claim,
                    |p| mle(&b, p),
                    &bad,
                    &mut Transcript::new_default(b"inner"),
                    b"ip"
                ),
                Err(Error::MalformedProof)
            );
        }
    }
    assert_eq!(
        prove_inner_product(
            &[],
            &[],
            K::ZERO,
            &mut Transcript::new_default(b"inner"),
            b"ip"
        ),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        prove_inner_product(
            &[K::ONE],
            &[K::ONE],
            K::ZERO,
            &mut Transcript::new_default(b"inner"),
            b"ip"
        ),
        Err(Error::InvalidClaim)
    );
}
