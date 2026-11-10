//! Equality-factored weighted quadratic cofactor for GKR virtualization.

use om_field::{K, U256};
use om_sumcheck::SumcheckProof;
use om_transcript::Transcript;

use crate::eq_factor::{self, Cofactor, Endpoint, scale_low_weights};

struct Products<'a> {
    weights: &'a [K],
}

impl Products<'_> {
    #[inline(always)]
    fn accumulate_mode<const BOTH: bool, const AT_ONE: bool>(
        &self,
        weights: &[K],
        left: &[K],
        right: &[K],
        out: &mut [U256],
    ) {
        for (node, &weight) in weights.iter().enumerate() {
            let (l0, l1) = (left[2 * node], right[2 * node]);
            let (r0, r1) = (left[2 * node + 1], right[2 * node + 1]);
            if BOTH {
                let scaled = om_field::mul_pairs([weight; 2], [l0, l1]);
                let products = om_field::clmul_pairs(scaled, [r0, r1]);
                out[0] += products[0];
                out[1] += products[1];
                out[2] += (weight * (l0 + l1)).clmul(r0 + r1);
            } else {
                let endpoint = if AT_ONE { (l1, r1) } else { (l0, r0) };
                let scaled = om_field::mul_pairs([weight; 2], [endpoint.0, l0 + l1]);
                let products = om_field::clmul_pairs(scaled, [endpoint.1, r0 + r1]);
                out[usize::from(AT_ONE)] += products[0];
                out[2] += products[1];
            }
        }
    }
}

impl Cofactor for Products<'_> {
    const DEGREE: usize = 2;
    type Prepared = Vec<K>;
    fn width(&self) -> usize {
        2 * self.weights.len()
    }
    fn value(&self, row: &[K]) -> K {
        self.weights
            .iter()
            .enumerate()
            .fold(U256::ZERO, |sum, (node, &weight)| {
                sum + (weight * row[2 * node]).clmul(row[2 * node + 1])
            })
            .reduce()
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
        // Apply lambda*Eq-low to one child line. Endpoint and slope use the
        // host's two independent carry-less lanes; products stay unreduced.
        match endpoint {
            Endpoint::Both => self.accumulate_mode::<true, false>(weights, left, right, out),
            Endpoint::Zero => self.accumulate_mode::<false, false>(weights, left, right, out),
            Endpoint::One => self.accumulate_mode::<false, true>(weights, left, right, out),
            Endpoint::Neither => {
                for (node, &weight) in weights.iter().enumerate() {
                    out[2] += (weight * (left[2 * node] + right[2 * node]))
                        .clmul(left[2 * node + 1] + right[2 * node + 1]);
                }
            }
        }
    }
}

pub(crate) fn prove<F: Fn(usize, &mut [K]) + Sync>(
    point: &[K],
    row: F,
    weights: &[K],
    initial_claim: Option<K>,
    tr: &mut Transcript,
    label: &[u8],
) -> (K, SumcheckProof, Vec<K>, Vec<K>) {
    eq_factor::prove_with_initial_claim(point, &Products { weights }, row, initial_claim, tr, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq_factor::tests::{compare, compare_known, random};

    #[test]
    fn quadratic_cofactor_matches_dense_wire_for_boolean_points_and_forced_challenges() {
        let weights = [K::ZERO, K::ONE, K::t(19), K::t(37)];
        for weights in [&weights[..], &weights[..0]] {
            let cofactor = Products { weights };
            for n in [0, 1, 2, 5] {
                let mut seed = 0x082efa98ec4e6c89;
                let point: Vec<K> = (0..n)
                    .map(|i| match i % 3 {
                        0 => K::ONE,
                        1 => K::ZERO,
                        _ => random(&mut seed),
                    })
                    .collect();
                compare(&cofactor, &point, None);
                for value in [K::ZERO, K::ONE, K::t(2), K::t(3)] {
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

    #[test]
    fn parallel_quadratic_cofactor_matches_dense_wire() {
        let mut seed = 0xbe5466cf34e90c6c;
        let point: Vec<K> = (0..14).map(|_| random(&mut seed)).collect();
        compare(
            &Products {
                weights: &[K::ONE, K::t(13), K::t(137)],
            },
            &point,
            None,
        );
    }

    #[test]
    fn known_initial_claim_matches_dense_with_arbitrary_rows_and_zero_prefix() {
        let weights = [K::ZERO, K::ONE, K::t(7), K::t(113)];
        for n in [0, 1, 2, 5, 14] {
            let mut seed = 0xd1310ba698dfb5ac;
            let cofactor = Products { weights: &weights };
            let random_point: Vec<K> = (0..n).map(|_| random(&mut seed)).collect();
            let boolean_point: Vec<K> = (0..n).map(|i| K::from_bit(i % 2 == 0)).collect();
            compare_known(&cofactor, &random_point, None);
            compare_known(&cofactor, &boolean_point, None);
            if n < 14 {
                for value in [K::ZERO, K::ONE, K::t(2), K::t(3)] {
                    compare_known(&cofactor, &boolean_point, Some(&vec![value; n]));
                }
                if n > 0 {
                    let mut forced = vec![K::t(19); n];
                    forced[0] = K::ONE + random_point[0];
                    compare_known(&cofactor, &random_point, Some(&forced));
                }
            }
        }
    }
}
