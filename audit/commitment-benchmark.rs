use om_field::K;
use om_pcs::{ligerito, ntt};
use om_transcript::Hasher;
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};
fn next(s: &mut u64) -> K {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    K::from_u128((z ^ (z >> 31)) as u128 | ((*s as u128) << 64))
}
fn median<S, T>(mut setup: impl FnMut() -> S, mut run: impl FnMut(S) -> T) -> f64 {
    for _ in 0..2 {
        black_box(run(setup()));
    }
    let mut times = Vec::new();
    for _ in 0..7 {
        let state = setup();
        let start = Instant::now();
        let result = run(state);
        times.push(start.elapsed().as_secs_f64() * 1000.);
        black_box(&result);
        drop(result);
    }
    times.sort_by(f64::total_cmp);
    times[3]
}
fn main() {
    println!(
        "native sha={} avx2={} vpclmul={}",
        cfg!(target_feature = "sha"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "vpclmulqdq")
    );
    for workers in [1, 8] {
        rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap().install(||{
 let data:Vec<u8>=(0..1024*4096).map(|i|((i*193)^(i>>7))as u8).collect();let h=Hasher::Sha256;
 let hash=median(||(),|()|data.par_chunks(1024).map(|s|h.digest_slices(&[&[0],black_box(s)])).collect::<Vec<_>>());println!("workers={workers} sha4MiB_ms={hash:.4}");
 #[cfg(feature="sha4")]
 { let hash4=median(||(),|()|data.par_chunks(4096).map(|s|{let mut out=[0u8;128];h.hash_prefixed_four_into(0,core::array::from_fn(|i|&s[1024*i..1024*(i+1)]),&mut out);out}).collect::<Vec<_>>());println!("workers={workers} sha4MiB_batch4_ms={hash4:.4}"); }
 for n in [15,19]{let mut rng=42;let table:Vec<K>=(0..1usize<<n).map(|_|next(&mut rng)).collect();let p=ligerito::make_params(n,"provable-120",2);let rows=1usize<<p.ks[0];let cols=table.len()/rows;
 let encode=median(||table.clone(),|mut a|{ntt::encode_coset_col_major(&mut a,rows,cols,2,3);a});
 let commit=median(||table.clone(),|a|ligerito::commit_stripped(&p,h,a));
 let root=ligerito::commit_stripped(&p,h,table).0;let hex:String=root.iter().map(|b|format!("{b:02x}")).collect();
 println!("workers={workers} n={n} rows={rows} encode1coset_ms={encode:.4} commit_ms={commit:.4} root={hex}");
 }
});
    }
}
