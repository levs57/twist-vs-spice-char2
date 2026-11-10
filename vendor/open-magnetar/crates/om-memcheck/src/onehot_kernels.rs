//! Equality-factored cubic Frobenius cofactor for the quartic OneHot check.

use om_field::{K, U256};
use om_sumcheck::SumcheckProof;
use om_transcript::Transcript;

use crate::eq_factor::{self, Cofactor, Endpoint, scale_low_weights};

struct OneHot<'a> {
    weights: &'a [K],
    c: K,
    initial_zero: bool,
}

impl OneHot<'_> {
    #[inline(always)]
    fn endpoint_value(&self, a: K, b: K) -> K {
        let products = om_field::mul_pairs([a.square(), self.c], [a, b]);
        products[0] + products[1]
    }

    #[inline(always)]
    fn accumulate_mode<const BOTH: bool, const AT_ONE: bool, const NEITHER: bool>(
        &self,
        weights: &[K],
        left: &[K],
        right: &[K],
        out: &mut [U256],
    ) {
        for (node, &weight) in weights.iter().enumerate() {
            let (a0, a1) = (left[2 * node], right[2 * node]);
            let delta = a0 + a1;
            let delta_squared = delta.square();
            let weighted_delta_squared = if NEITHER {
                weight * delta_squared
            } else if BOTH {
                let endpoints = [
                    self.endpoint_value(a0, left[2 * node + 1]),
                    self.endpoint_value(a1, right[2 * node + 1]),
                ];
                let products = om_field::clmul_pairs([weight; 2], endpoints);
                out[0] += products[0];
                out[1] += products[1];
                weight * delta_squared
            } else {
                let value = if AT_ONE {
                    self.endpoint_value(a1, right[2 * node + 1])
                } else {
                    self.endpoint_value(a0, left[2 * node + 1])
                };
                let products = om_field::clmul_pairs([weight; 2], [value, delta_squared]);
                out[usize::from(AT_ONE)] += products[0];
                products[1].reduce()
            };
            // In characteristic two a(X)^3 has high coefficients a0*delta²
            // and delta³. Reuse their common scaled square in paired lanes.
            let high = om_field::clmul_pairs([weighted_delta_squared; 2], [a0, delta]);
            out[2] += high[0];
            out[3] += high[1];
        }
    }
}

impl Cofactor for OneHot<'_> {
    const DEGREE: usize = 3;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        2 * self.weights.len()
    }
    fn value(&self, row: &[K]) -> K {
        self.weights
            .iter()
            .enumerate()
            .fold(U256::ZERO, |sum, (node, &weight)| {
                let a = row[2 * node];
                sum + weight.clmul(a.square() * a + self.c * row[2 * node + 1])
            })
            .reduce()
    }
    fn initial_endpoints_zero(&self) -> bool {
        self.initial_zero
    }
    fn prepare(&self, low_eq: &[K]) -> Self::Prepared {
        scale_low_weights(self.weights, low_eq)
    }
    #[inline]
    fn accumulate(
        &self,
        prepared: &Self::Prepared,
        low: usize,
        left: &[K],
        right: &[K],
        endpoint: Endpoint,
        out: &mut [U256],
    ) {
        let count = self.weights.len();
        let weights = &prepared[low * count..(low + 1) * count];
        match endpoint {
            Endpoint::Both => self.accumulate_mode::<true, false, false>(weights, left, right, out),
            Endpoint::Zero => {
                self.accumulate_mode::<false, false, false>(weights, left, right, out)
            }
            Endpoint::One => self.accumulate_mode::<false, true, false>(weights, left, right, out),
            Endpoint::Neither => {
                self.accumulate_mode::<false, false, true>(weights, left, right, out)
            }
        }
    }
}

pub(crate) fn prove<F: Fn(usize, &mut [K]) + Sync>(
    point: &[K],
    row: F,
    weights: &[K],
    c: K,
    initial_zero: bool,
    tr: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    eq_factor::prove(
        point,
        &OneHot {
            weights,
            c,
            initial_zero,
        },
        row,
        tr,
        label,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq_factor::tests::{compare, compare_columns, random};

    #[test]
    fn cubic_cofactor_matches_dense_wire_for_arbitrary_fields_and_edge_challenges() {
        let weights = [K::ZERO, K::ONE, K::t(11), K::t(31)];
        for weights in [&weights[..], &weights[..0]] {
            for c in [K::ZERO, K::ONE, K::t(177)] {
                let cofactor = OneHot {
                    weights,
                    c,
                    initial_zero: false,
                };
                for n in [0, 1, 2, 5] {
                    let mut seed = 0x3f84d5b5b5470917;
                    let point: Vec<K> = (0..n)
                        .map(|i| match i % 3 {
                            0 => K::ONE,
                            1 => K::ZERO,
                            _ => random(&mut seed),
                        })
                        .collect();
                    compare(&cofactor, &point, None);
                    for value in [K::ZERO, K::ONE, K::t(2), K::t(3), K::t(4)] {
                        compare(&cofactor, &point, Some(&vec![value; n]));
                    }
                    if n > 0 {
                        let point: Vec<K> = (0..n).map(|_| random(&mut seed)).collect();
                        let mut forced: Vec<K> = (0..n).map(|_| random(&mut seed)).collect();
                        forced[0] = K::ONE + point[0];
                        compare(&cofactor, &point, Some(&forced));
                    }
                }
            }
        }
    }

    #[test]
    fn parallel_cubic_cofactor_matches_dense_wire() {
        let mut seed = 0x9216d5d98979fb1b;
        let point: Vec<K> = (0..14).map(|_| random(&mut seed)).collect();
        compare(
            &OneHot {
                weights: &[K::ONE, K::t(29)],
                c: K::t(83),
                initial_zero: false,
            },
            &point,
            None,
        );
    }

    #[test]
    fn structurally_zero_initial_endpoints_preserve_full_reference_messages() {
        use crate::onehot::onehot_aux;
        use om_sumcheck::eq_table;
        for n in [0, 1, 2, 5, 14] {
            let mut seed = 0x2ffd72dbd01adfb7;
            let ac = 3;
            let r: Vec<K> = (0..ac).map(|_| random(&mut seed)).collect();
            let (rp, c) = onehot_aux(&r).unwrap();
            let (er, erp) = (eq_table(&r), eq_table(&rp));
            let weights = [K::ZERO, K::ONE, random(&mut seed)];
            let mut columns = Vec::new();
            for _ in &weights {
                let mut a = Vec::new();
                let mut b = Vec::new();
                for _ in 0..1usize << n {
                    random(&mut seed);
                    let index = (seed as usize) & ((1 << ac) - 1);
                    assert_eq!(er[index].square() * er[index] + c * erp[index], K::ZERO);
                    a.push(er[index]);
                    b.push(erp[index]);
                }
                columns.push(a);
                columns.push(b);
            }
            let point: Vec<K> = (0..n)
                .map(|i| match i % 3 {
                    0 => K::ONE,
                    1 => K::ZERO,
                    _ => random(&mut seed),
                })
                .collect();
            let cofactor = OneHot {
                weights: &weights,
                c,
                initial_zero: true,
            };
            compare_columns(&cofactor, &point, &columns, None);
            if n < 14 {
                for challenge in [K::ZERO, K::ONE, K::t(2), K::t(4)] {
                    compare_columns(&cofactor, &point, &columns, Some(&vec![challenge; n]));
                }
            }
        }
    }
}
