//! Ratio of medians with a paired percentile-bootstrap interval.
//!
//! Samples are paired by measured round: round `i` ran every configuration
//! once, back to back, so its two values saw nearly the same machine state.
//! Each resample draws rounds with replacement and takes the same rounds for
//! numerator and denominator, which keeps that pairing and lets the interval
//! reflect drift that hits both methods alike only once.

use serde::Serialize;

pub const RESAMPLES: usize = 10_000;
pub const SEED: u64 = 0x5eed_7715_7a1c_e5e5;

/// SplitMix64, as in ram-payload: a fixed, dependency-free stream.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound` (rejection sampling; no modulo bias).
    fn below(&mut self, bound: usize) -> usize {
        let bound = bound as u64;
        let zone = u64::MAX - u64::MAX % bound;
        loop {
            let v = self.next();
            if v < zone {
                return (v % bound) as usize;
            }
        }
    }
}

pub fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let m = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[m - 1] + values[m]) / 2.0
    } else {
        values[m]
    }
}

/// Linear-interpolated quantile of sorted values.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len() - 1) as f64;
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RatioInterval {
    /// median(numerator) / median(denominator) over all rounds.
    pub ratio: f64,
    /// 2.5% and 97.5% percentiles of the bootstrap distribution.
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub rounds: usize,
    pub resamples: usize,
    pub seed: u64,
}

/// Ratio of medians of paired samples, with a 95% percentile bootstrap
/// interval over rounds (`numerator[i]` and `denominator[i]` are round `i`).
pub fn paired_ratio(numerator: &[f64], denominator: &[f64]) -> Option<RatioInterval> {
    paired_ratio_with(numerator, denominator, RESAMPLES, SEED)
}

pub fn paired_ratio_with(
    numerator: &[f64],
    denominator: &[f64],
    resamples: usize,
    seed: u64,
) -> Option<RatioInterval> {
    let rounds = numerator.len();
    if rounds == 0 || rounds != denominator.len() || resamples == 0 {
        return None;
    }
    let ratio = median(&mut numerator.to_vec()) / median(&mut denominator.to_vec());
    let mut rng = SplitMix64(seed);
    let (mut num, mut den) = (vec![0.0; rounds], vec![0.0; rounds]);
    let mut ratios: Vec<f64> = (0..resamples)
        .map(|_| {
            for slot in 0..rounds {
                let pick = rng.below(rounds);
                num[slot] = numerator[pick];
                den[slot] = denominator[pick];
            }
            median(&mut num) / median(&mut den)
        })
        .collect();
    ratios.sort_by(f64::total_cmp);
    Some(RatioInterval {
        ratio,
        ci95_low: quantile(&ratios, 0.025),
        ci95_high: quantile(&ratios, 0.975),
        rounds,
        resamples,
        seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_ratio_has_a_degenerate_interval() {
        let den = [10.0, 12.0, 9.0, 30.0, 11.0];
        let num: Vec<f64> = den.iter().map(|v| 2.0 * v).collect();
        let r = paired_ratio(&num, &den).unwrap();
        // Pairing by round keeps the ratio exact even under drift.
        assert_eq!((r.ratio, r.ci95_low, r.ci95_high), (2.0, 2.0, 2.0));
        assert_eq!((r.rounds, r.resamples), (5, RESAMPLES));
    }

    #[test]
    fn interval_brackets_the_ratio_and_is_reproducible() {
        let num = [
            100.0, 104.0, 98.0, 120.0, 101.0, 99.0, 103.0, 97.0, 110.0, 102.0,
        ];
        let den = [50.0, 49.0, 52.0, 51.0, 48.0, 50.0, 53.0, 47.0, 50.0, 49.0];
        let a = paired_ratio(&num, &den).unwrap();
        let b = paired_ratio(&num, &den).unwrap();
        assert_eq!(a, b);
        assert!(a.ci95_low <= a.ratio && a.ratio <= a.ci95_high);
        assert!(a.ci95_low > 1.8 && a.ci95_high < 2.3, "{a:?}");
        assert!(a.ci95_high > a.ci95_low);
        assert_eq!(paired_ratio(&num, &den[..3]), None);
        // scripts/bench_stats.py implements the same stream and percentiles;
        // scripts/test_scripts.py asserts the same three numbers.
        assert_eq!(a.ratio, 2.03);
        assert!((a.ci95_low - 1.97).abs() < 1e-12, "{}", a.ci95_low);
        assert!(
            (a.ci95_high - 2.122448979591837).abs() < 1e-12,
            "{}",
            a.ci95_high
        );
    }

    #[test]
    fn rng_below_is_in_range() {
        let mut rng = SplitMix64(1);
        assert!((0..1000).all(|_| rng.below(7) < 7));
    }
}
