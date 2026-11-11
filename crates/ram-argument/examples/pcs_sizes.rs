//! Commit, open and verify cost of the Ligerito bit PCS at 2^26, 2^27 and
//! 2^28 bits, for docs/commitment-padding.md. At 2^28 the top quarter is
//! zero, as in Twist v2's padded commitment at n = 20, a = 16, c = 4; pass
//! `full` to fill it.
//!
//! cargo run --release -p ram-argument --example pcs_sizes [full]

use om_pcs::BitPCS;
use om_pcs::ligerito::{make_params, soundness_bits};
use om_sumcheck::PackedBits;
use om_transcript::Transcript;
use std::time::Instant;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() {
    let full = std::env::args().nth(1).as_deref() == Some("full");
    for ell in [26usize, 27, 28] {
        let params = make_params(ell - 7, "provable-120", 2);
        let error: f64 = soundness_bits(&params, "provable-120")
            .iter()
            .map(|s| {
                [s.sumcheck, s.batch, s.gap, s.query]
                    .iter()
                    .map(|b| 2f64.powf(-b))
                    .sum::<f64>()
            })
            .sum();
        let pcs = BitPCS::new(params);
        let words = (1usize << ell) / 64;
        let populated = if ell == 28 && !full { words / 4 * 3 } else { words };
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let g = PackedBits::from_words(
            (0..words)
                .map(|i| {
                    if i < populated {
                        xorshift(&mut s) & xorshift(&mut s) & xorshift(&mut s)
                    } else {
                        0
                    }
                })
                .collect(),
            1usize << ell,
        );
        let (mut commit, mut open, mut verify, mut bytes) = (f64::MAX, f64::MAX, f64::MAX, 0);
        for _ in 0..4 {
            let mut tr = Transcript::new_default(b"pcs_sizes");
            let t = Instant::now();
            let (cm, state) = pcs.commit_packed(&g, &mut tr);
            commit = commit.min(t.elapsed().as_secs_f64() * 1e3);
            let r = tr.challenge_ks(b"point", ell);
            let t = Instant::now();
            let (y, proof) = pcs.open_packed(state, &g, &r, &mut tr);
            open = open.min(t.elapsed().as_secs_f64() * 1e3);
            let mut tv = Transcript::new_default(b"pcs_sizes");
            pcs.absorb_commitment(&cm, &mut tv);
            let r = tv.challenge_ks(b"point", ell);
            let t = Instant::now();
            pcs.verify(&cm, &r, y, &proof, &mut tv).unwrap();
            verify = verify.min(t.elapsed().as_secs_f64() * 1e3);
            bytes = proof.1.to_bytes().len();
        }
        println!(
            "2^{ell} bits ({populated} of {words} words populated): commit {commit:.1} ms, \
             open {open:.1} ms, verify {verify:.2} ms, Ligerito proof {bytes} B, \
             PCS soundness {:.3} bits (best of 4)",
            -error.log2()
        );
    }
}
