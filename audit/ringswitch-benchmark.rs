// Isolated RingSwitch benchmark, retained from the full-kernel-pass audit.
// Native release: RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS=4 cargo run --offline --release -- <workers> <log_cycles> <chunk_bits>
// Build this same file against either the before-full-kernel-pass or current
// om-field/om-sumcheck/om-transcript/om-mosaic/om-pcs path dependencies plus rayon.
// Set the Cargo [[bin]] path directly to this file (so its sibling module resolves).
// Each operation has 2 warmups and 11 samples; report the median. Setup/clones
// and output destruction are outside each timer. All outputs are black-boxed.
// Inputs reproduce random_ram(n,12,42), shared read/write addresses, one read,
// 64-bit words, and the exact component layout used by the RAM adapter. The
// opening point is deterministic and identical across both builds; it is not
// taken from a composed RAM proof. The scalar_sc comparison uses the retained
// pre-pass source with the linked build's field and transcript implementations.
// Production code is not modified by this driver.
use om_field::K;
use om_pcs::ring_switch as rs;
use om_sumcheck::{PackedBits, eq_table};
use om_transcript::Transcript;
use std::{collections::BTreeMap, hint::black_box, time::Instant};
#[allow(dead_code)]
mod mle {
    pub use om_sumcheck::mle::bind_low_in_place;
    pub const PAR_THRESHOLD: usize = 1 << 14;
}
#[allow(dead_code)]
#[path = "ringswitch-before-sumcheck.rs"]
mod scalar_sc;
fn next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn field(s: &mut u64) -> K {
    K::from_u128(next(s) as u128 | ((next(s) as u128) << 64))
}
fn packed(n: usize, c: usize) -> (PackedBits, Vec<K>) {
    let t = 1usize << n;
    let mut state = 42;
    let mut memory: Vec<u64> = (0..4096).map(|_| next(&mut state)).collect();
    let d = 12 / c;
    let mut addresses: Vec<PackedBits> = (0..d).map(|_| PackedBits::zero(t << c)).collect();
    let (mut rv, mut inc) = (Vec::with_capacity(t), Vec::with_capacity(t));
    for j in 0..t {
        let a = next(&mut state) as usize & 4095;
        let v = next(&mut state);
        rv.push(memory[a]);
        inc.push(memory[a] ^ v);
        memory[a] = v;
        for (lane, bits) in addresses.iter_mut().enumerate() {
            bits.set((j << c) | ((a >> (lane * c)) & ((1 << c) - 1)));
        }
    }
    let comps: Vec<(i64, usize)> = (0..d)
        .map(|i| (i as i64, n + c))
        .chain([(d as i64, n + 6), (d as i64 + 1, n + 6)])
        .collect();
    let layout = om_mosaic::compute_layout(&comps, 14);
    let mut tables: BTreeMap<i64, PackedBits> = addresses
        .into_iter()
        .enumerate()
        .map(|(i, b)| (i as i64, b))
        .collect();
    tables.insert(d as i64, PackedBits::from_words(inc, t * 64));
    tables.insert(d as i64 + 1, PackedBits::from_words(rv, t * 64));
    let g = om_mosaic::build_g_packed(&layout, &tables);
    let r = (0..layout.ell).map(|_| field(&mut state)).collect();
    (g, r)
}
fn median<S, T>(mut setup: impl FnMut() -> S, mut run: impl FnMut(S) -> T) -> f64 {
    for _ in 0..2 {
        black_box(run(setup()));
    }
    let mut times = Vec::new();
    for _ in 0..11 {
        let input = setup();
        let st = Instant::now();
        let out = run(input);
        times.push(st.elapsed().as_secs_f64() * 1e3);
        black_box(&out);
        drop(out);
    }
    times.sort_by(f64::total_cmp);
    times[5]
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workers = args.get(1).map(|v| v.parse().unwrap()).unwrap_or(1);
    let n = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(18);
    let c = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    pool.install(||{let(g,r)=packed(n,c);let s=rs::partial_evals_packed(&g,&r[7..]);let er=eq_table(&r[..7]);let y=er.iter().zip(&s).fold(K::ZERO,|a,(&x,&z)|a+x*z);let mut tr=Transcript::new_default(b"rs-isolated-same-input");tr.absorb_ks(b"rs/point",&r);tr.absorb_k(b"rs/value",y);tr.absorb_ks(b"rs/s",&s);let rho=tr.challenge_ks(b"rs/rho",7);let a=rs::a_table(&r[7..],&rho);let p=rs::pack_packed(&g);let sc=rs::transpose_bits(&s);let target=eq_table(&rho).iter().zip(sc).fold(K::ZERO,|a,(&x,z)|a+x*z);
let mut oldtr=tr.clone();let old=scalar_sc::sumcheck_prove_mul2(target,vec![p.clone(),a.clone()],&mut oldtr,b"rs/sumcheck");let mut newtr=tr.clone();let new=om_sumcheck::sumcheck_prove_mul2(target,vec![p.clone(),a.clone()],&mut newtr,b"rs/sumcheck");assert_eq!(old.0.round_msgs,new.0.round_msgs);assert_eq!(old.1,new.1);assert_eq!(old.2,new.2);assert_eq!(oldtr.challenge_k(b"after"),newtr.challenge_k(b"after"));
let part=median(||(),|_|rs::partial_evals_packed(&g,&r[7..]));let at=median(||(),|_|rs::a_table(&r[7..],&rho));let pack=median(||(),|_|rs::pack_packed(&g));let scalar=median(||(vec![p.clone(),a.clone()],tr.clone()),|(t,mut tr)|scalar_sc::sumcheck_prove_mul2(target,t,&mut tr,b"rs/sumcheck"));let dense=median(||(vec![p.clone(),a.clone()],tr.clone()),|(t,mut tr)|om_sumcheck::sumcheck_prove_mul2(target,t,&mut tr,b"rs/sumcheck"));let full=median(||Transcript::new_default(b"rs-isolated-same-input"),|mut tr|rs::rs_prove_packed_value(&g,&r,&mut tr));let mut finaltr=Transcript::new_default(b"rs-isolated-same-input");let out=rs::rs_prove_packed_value(&g,&r,&mut finaltr);println!("workers={workers} n={n} c={c} ell={} fingerprint={:?}/{:?}/{:?} partial_ms={part:.5} A_ms={at:.5} pack_ms={pack:.5} scalar_sc_ms={scalar:.5} current_sc_ms={dense:.5} full_rs_ms={full:.5}",r.len(),y,out.2,finaltr.challenge_k(b"after"));});
}
