//! SHA-256 per FIPS 180-4, from scratch.

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const IV256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[inline]
pub fn digest(data: &[u8]) -> [u8; 32] {
    digest_slices(&[data])
}

pub fn digest_slices(slices: &[&[u8]]) -> [u8; 32] {
    let mut h = IV256;
    let mut buf = [0u8; 64];
    let mut buf_len = 0usize;
    let mut total_len = 0u64;

    for &slice in slices {
        total_len = total_len.wrapping_add(slice.len() as u64);
        let mut s = slice;
        if buf_len > 0 {
            let take = (64 - buf_len).min(s.len());
            buf[buf_len..buf_len + take].copy_from_slice(&s[..take]);
            buf_len += take;
            s = &s[take..];
            if buf_len == 64 {
                compress_block(&mut h, &buf);
                buf_len = 0;
            }
        }
        let (blocks, rem) = s.as_chunks::<64>();
        for block in blocks {
            compress_block(&mut h, block);
        }
        if !rem.is_empty() {
            buf[..rem.len()].copy_from_slice(rem);
            buf_len = rem.len();
        }
    }

    // Padding: 0x80, zeros, 64-bit big-endian bit length, to a multiple of 64.
    let bit_len = total_len.wrapping_mul(8);
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 56 {
        buf[buf_len..64].fill(0);
        compress_block(&mut h, &buf);
        buf[..56].fill(0);
    } else {
        buf[buf_len..56].fill(0);
    }
    buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress_block(&mut h, &buf);

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Fast path for `SHA256(0x01 || left_32 || right_32)` (65 bytes = two 64-byte blocks).
#[inline]
pub fn hash_node_64(left: &[u8], right: &[u8], out: &mut [u8]) {
    debug_assert_eq!(left.len(), 32);
    debug_assert_eq!(right.len(), 32);
    debug_assert_eq!(out.len(), 32);
    let mut b0 = [0u8; 64];
    b0[0] = 0x01;
    b0[1..33].copy_from_slice(left);
    b0[33..64].copy_from_slice(&right[..31]);

    let mut b1 = [0u8; 64];
    b1[0] = right[31];
    b1[1] = 0x80;
    // 65 bytes = 520 bits = 0x0208 in big-endian u64 at b1[56..64].
    b1[62] = 0x02;
    b1[63] = 0x08;

    let mut h = IV256;
    compress_block(&mut h, &b0);
    compress_block(&mut h, &b1);
    for (i, word) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
}

#[inline(always)]
fn compress_block(h: &mut [u32; 8], block: &[u8; 64]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sha",
        target_feature = "ssse3",
        target_feature = "sse4.1"
    ))]
    {
        // SAFETY: all features used by the SHA-NI kernel are enabled.
        unsafe { compress_block_x86_64(h, block) };
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(
            target_feature = "sha",
            target_feature = "ssse3",
            target_feature = "sse4.1"
        ))
    ))]
    {
        if std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            // SAFETY: every required feature was detected above.
            unsafe { compress_block_x86_64(h, block) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
    {
        // SAFETY: `sha2` target feature is enabled at compile time.
        unsafe { compress_block_aarch64(h, block) };
        return;
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sha2")))]
    {
        if std::arch::is_aarch64_feature_detected!("sha2") {
            // SAFETY: `sha2` feature was just detected at runtime.
            unsafe { compress_block_aarch64(h, block) };
            return;
        }
    }
    #[allow(unreachable_code)]
    compress_block_portable(h, block);
}

pub(crate) fn compress_block_portable(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
        (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K256[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

// SHA-NI schedule and state layout adapted from pinned Flock's
// crates/flock-merkle/src/merkle/x86_64.rs (four streams reduced to one).
// Copyright (c) 2026 Succinct Labs, Benedikt Bunz, and William Wang.
// Licensed under MIT OR Apache-2.0; donor notice in LICENSE-FLOCK-MIT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sha,ssse3,sse4.1")]
unsafe fn compress_block_x86_64(h: &mut [u32; 8], block: &[u8; 64]) {
    use core::arch::x86_64::*;

    #[cfg(target_feature = "avx")]
    // SAFETY: AVX is enabled by the compiled target.
    unsafe {
        _mm256_zeroupper()
    };

    // SAFETY: the arrays expose eight state words, four input vectors, and
    // sixteen constant vectors. Unaligned loads/stores remain in bounds.
    unsafe {
        let dcba = _mm_loadu_si128(h.as_ptr().cast());
        let efgh = _mm_shuffle_epi32::<0x1b>(_mm_loadu_si128(h.as_ptr().add(4).cast()));
        let cdab = _mm_shuffle_epi32::<0xb1>(dcba);
        let mut abef = _mm_alignr_epi8::<8>(cdab, efgh);
        let mut cdgh = _mm_blend_epi16::<0xf0>(efgh, cdab);
        let (abef_saved, cdgh_saved) = (abef, cdgh);
        let endian = _mm_set_epi64x(
            0x0c0d_0e0f_0809_0a0bu64 as i64,
            0x0405_0607_0001_0203u64 as i64,
        );
        let mut w0 = _mm_shuffle_epi8(_mm_loadu_si128(block.as_ptr().cast()), endian);
        let mut w1 = _mm_shuffle_epi8(_mm_loadu_si128(block.as_ptr().add(16).cast()), endian);
        let mut w2 = _mm_shuffle_epi8(_mm_loadu_si128(block.as_ptr().add(32).cast()), endian);
        let mut w3 = _mm_shuffle_epi8(_mm_loadu_si128(block.as_ptr().add(48).cast()), endian);

        macro_rules! round4 {
            ($w:ident, $index:expr) => {{
                let wk = _mm_add_epi32($w, _mm_loadu_si128(K256.as_ptr().add($index).cast()));
                cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
                abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32::<0x0e>(wk));
            }};
            ($w0:ident, $w1:ident, $w2:ident, $w3:ident, $index:expr) => {{
                let schedule = _mm_sha256msg1_epu32($w0, $w1);
                let align = _mm_alignr_epi8::<4>($w3, $w2);
                $w0 = _mm_sha256msg2_epu32(_mm_add_epi32(schedule, align), $w3);
                round4!($w0, $index);
            }};
        }
        round4!(w0, 0);
        round4!(w1, 4);
        round4!(w2, 8);
        round4!(w3, 12);
        round4!(w0, w1, w2, w3, 16);
        round4!(w1, w2, w3, w0, 20);
        round4!(w2, w3, w0, w1, 24);
        round4!(w3, w0, w1, w2, 28);
        round4!(w0, w1, w2, w3, 32);
        round4!(w1, w2, w3, w0, 36);
        round4!(w2, w3, w0, w1, 40);
        round4!(w3, w0, w1, w2, 44);
        round4!(w0, w1, w2, w3, 48);
        round4!(w1, w2, w3, w0, 52);
        round4!(w2, w3, w0, w1, 56);
        round4!(w3, w0, w1, w2, 60);

        abef = _mm_add_epi32(abef, abef_saved);
        cdgh = _mm_add_epi32(cdgh, cdgh_saved);
        let feba = _mm_shuffle_epi32::<0x1b>(abef);
        let dchg = _mm_shuffle_epi32::<0xb1>(cdgh);
        _mm_storeu_si128(h.as_mut_ptr().cast(), _mm_blend_epi16::<0xf0>(feba, dchg));
        _mm_storeu_si128(
            h.as_mut_ptr().add(4).cast(),
            _mm_alignr_epi8::<8>(dchg, feba),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "sha2,neon")]
unsafe fn compress_block_aarch64(h: &mut [u32; 8], block: &[u8; 64]) {
    use core::arch::aarch64::{
        vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32,
        vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
    };

    // SAFETY: `h` has 8 `u32` elements and `block` has 64 `u8` elements;
    // `K256` has 64 `u32` elements. All pointer offsets are in-bounds.
    unsafe {
        let abcd_orig = vld1q_u32(h.as_ptr());
        let efgh_orig = vld1q_u32(h.as_ptr().add(4));
        let mut abcd = abcd_orig;
        let mut efgh = efgh_orig;

        let mut w0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr())));
        let mut w1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(16))));
        let mut w2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(32))));
        let mut w3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(48))));

        macro_rules! round4 {
            ($w:ident, $idx:expr) => {{
                let wk = vaddq_u32($w, vld1q_u32(K256.as_ptr().add($idx)));
                let prev_abcd = abcd;
                abcd = vsha256hq_u32(abcd, efgh, wk);
                efgh = vsha256h2q_u32(efgh, prev_abcd, wk);
            }};
            ($w0:ident, $w1:ident, $w2:ident, $w3:ident, $idx:expr) => {{
                round4!($w0, $idx);
                $w0 = vsha256su1q_u32(vsha256su0q_u32($w0, $w1), $w2, $w3);
            }};
        }

        round4!(w0, w1, w2, w3, 0);
        round4!(w1, w2, w3, w0, 4);
        round4!(w2, w3, w0, w1, 8);
        round4!(w3, w0, w1, w2, 12);

        round4!(w0, w1, w2, w3, 16);
        round4!(w1, w2, w3, w0, 20);
        round4!(w2, w3, w0, w1, 24);
        round4!(w3, w0, w1, w2, 28);

        round4!(w0, w1, w2, w3, 32);
        round4!(w1, w2, w3, w0, 36);
        round4!(w2, w3, w0, w1, 40);
        round4!(w3, w0, w1, w2, 44);

        round4!(w0, 48);
        round4!(w1, 52);
        round4!(w2, 56);
        round4!(w3, 60);

        abcd = vaddq_u32(abcd, abcd_orig);
        efgh = vaddq_u32(efgh, efgh_orig);

        vst1q_u32(h.as_mut_ptr(), abcd);
        vst1q_u32(h.as_mut_ptr().add(4), efgh);
    }
}

/// Four independent domain-prefixed digests. The scalar fallback also handles
/// unequal input lengths; Merkle batches normally have equal-length rows.
pub(crate) fn digest4_prefixed_into(prefix: u8, inputs: [&[u8]; 4], out: &mut [u8]) {
    assert_eq!(out.len(), 128);
    #[cfg(target_arch = "x86_64")]
    if inputs.iter().all(|input| input.len() == inputs[0].len())
        && std::arch::is_x86_feature_detected!("sha")
        && std::arch::is_x86_feature_detected!("ssse3")
        && std::arch::is_x86_feature_detected!("sse4.1")
    {
        // SAFETY: features and equal lengths checked; output has four digests.
        unsafe { x86_batch::digest4_prefixed(prefix, inputs, out) };
        return;
    }
    for (input, digest) in inputs.into_iter().zip(out.chunks_exact_mut(32)) {
        digest.copy_from_slice(&digest_slices(&[&[prefix], input]));
    }
}

// Adapted from pinned Flock's four-stream SHA-NI Merkle implementation.
// Copyright (c) 2026 Succinct Labs, Benedikt Bunz, and William Wang.
// Licensed under MIT OR Apache-2.0; see LICENSE-MIT and LICENSE-APACHE.
#[cfg(target_arch = "x86_64")]
mod x86_batch {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi32,
        _mm_set_epi64x, _mm_setzero_si128, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32,
        _mm_sha256rnds2_epu32, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
    };

    use super::{IV256 as SHA256_IV, K256 as SHA256_K};

    #[inline]
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    unsafe fn schedule(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
        let t1 = _mm_sha256msg1_epu32(v0, v1);
        let t2 = _mm_alignr_epi8(v3, v2, 4);
        _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), v3)
    }

    #[inline]
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    unsafe fn rounds4(
        abef: &mut [__m128i; 4],
        cdgh: &mut [__m128i; 4],
        words: &[__m128i; 4],
        group: usize,
    ) {
        let k = _mm_set_epi32(
            SHA256_K[4 * group + 3] as i32,
            SHA256_K[4 * group + 2] as i32,
            SHA256_K[4 * group + 1] as i32,
            SHA256_K[4 * group] as i32,
        );
        for stream in 0..4 {
            let wk = _mm_add_epi32(words[stream], k);
            cdgh[stream] = _mm_sha256rnds2_epu32(cdgh[stream], abef[stream], wk);
            let wk_hi = _mm_shuffle_epi32(wk, 0x0e);
            abef[stream] = _mm_sha256rnds2_epu32(abef[stream], cdgh[stream], wk_hi);
        }
    }

    /// Compress one block for each of four independent SHA-256 states.
    ///
    /// # Safety
    /// Every block pointer must expose 64 readable bytes. The caller must
    /// ensure SHA-NI, SSSE3 and SSE4.1 support.
    #[inline]
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    unsafe fn compress4(abef: &mut [__m128i; 4], cdgh: &mut [__m128i; 4], blocks: [*const u8; 4]) {
        unsafe {
            let endian = _mm_set_epi64x(
                0x0c0d_0e0f_0809_0a0bu64 as i64,
                0x0405_0607_0001_0203u64 as i64,
            );
            let mut w0 = [_mm_setzero_si128(); 4];
            let mut w1 = [_mm_setzero_si128(); 4];
            let mut w2 = [_mm_setzero_si128(); 4];
            let mut w3 = [_mm_setzero_si128(); 4];
            for stream in 0..4 {
                let data = blocks[stream].cast::<__m128i>();
                w0[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data), endian);
                w1[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(1)), endian);
                w2[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(2)), endian);
                w3[stream] = _mm_shuffle_epi8(_mm_loadu_si128(data.add(3)), endian);
            }

            let abef_saved = *abef;
            let cdgh_saved = *cdgh;
            #[cfg(target_feature = "avx")]
            core::arch::x86_64::_mm256_zeroupper();

            macro_rules! schedule_rounds4 {
                ($dst:ident, $v1:ident, $v2:ident, $v3:ident, $group:expr) => {{
                    for stream in 0..4 {
                        $dst[stream] =
                            schedule($dst[stream], $v1[stream], $v2[stream], $v3[stream]);
                    }
                    rounds4(abef, cdgh, &$dst, $group);
                }};
            }

            rounds4(abef, cdgh, &w0, 0);
            rounds4(abef, cdgh, &w1, 1);
            rounds4(abef, cdgh, &w2, 2);
            rounds4(abef, cdgh, &w3, 3);
            schedule_rounds4!(w0, w1, w2, w3, 4);
            schedule_rounds4!(w1, w2, w3, w0, 5);
            schedule_rounds4!(w2, w3, w0, w1, 6);
            schedule_rounds4!(w3, w0, w1, w2, 7);
            schedule_rounds4!(w0, w1, w2, w3, 8);
            schedule_rounds4!(w1, w2, w3, w0, 9);
            schedule_rounds4!(w2, w3, w0, w1, 10);
            schedule_rounds4!(w3, w0, w1, w2, 11);
            schedule_rounds4!(w0, w1, w2, w3, 12);
            schedule_rounds4!(w1, w2, w3, w0, 13);
            schedule_rounds4!(w2, w3, w0, w1, 14);
            schedule_rounds4!(w3, w0, w1, w2, 15);

            for stream in 0..4 {
                abef[stream] = _mm_add_epi32(abef[stream], abef_saved[stream]);
                cdgh[stream] = _mm_add_epi32(cdgh[stream], cdgh_saved[stream]);
            }
        }
    }

    /// SHA256(prefix || input), with four independent equal-length inputs.
    #[inline]
    #[target_feature(enable = "sha,ssse3,sse4.1")]
    pub(super) unsafe fn digest4_prefixed(prefix: u8, inputs: [&[u8]; 4], out: &mut [u8]) {
        // SAFETY: the wrapper checks features, equal input lengths and output size.
        // Full blocks below reference 64 in-bounds bytes; tails use fixed arrays.
        unsafe {
            let state = SHA256_IV.as_ptr().cast::<__m128i>();
            let dcba = _mm_loadu_si128(state);
            let efgh = _mm_shuffle_epi32(_mm_loadu_si128(state.add(1)), 0x1b);
            let cdab = _mm_shuffle_epi32(dcba, 0xb1);
            let mut abef = [_mm_alignr_epi8(cdab, efgh, 8); 4];
            let mut cdgh = [_mm_blend_epi16(efgh, cdab, 0xf0); 4];
            let len = inputs[0].len();
            let total_len = len + 1;
            let full_blocks = total_len / 64;
            if full_blocks > 0 {
                let mut first = [[0u8; 64]; 4];
                for stream in 0..4 {
                    first[stream][0] = prefix;
                    first[stream][1..].copy_from_slice(&inputs[stream][..63]);
                }
                compress4(
                    &mut abef,
                    &mut cdgh,
                    core::array::from_fn(|stream| first[stream].as_ptr()),
                );
                for block in 1..full_blocks {
                    compress4(
                        &mut abef,
                        &mut cdgh,
                        core::array::from_fn(|stream| inputs[stream].as_ptr().add(block * 64 - 1)),
                    );
                }
            }
            let rem = total_len % 64;
            let tail_blocks = if rem < 56 { 1 } else { 2 };
            let mut tails = [[0u8; 128]; 4];
            for stream in 0..4 {
                if full_blocks == 0 {
                    tails[stream][0] = prefix;
                    tails[stream][1..1 + len].copy_from_slice(inputs[stream]);
                } else {
                    tails[stream][..rem].copy_from_slice(&inputs[stream][len - rem..]);
                }
                tails[stream][rem] = 0x80;
                tails[stream][tail_blocks * 64 - 8..tail_blocks * 64]
                    .copy_from_slice(&(total_len as u64).wrapping_mul(8).to_be_bytes());
            }
            for block in 0..tail_blocks {
                compress4(
                    &mut abef,
                    &mut cdgh,
                    core::array::from_fn(|stream| tails[stream].as_ptr().add(block * 64)),
                );
            }
            let endian = _mm_set_epi64x(
                0x0c0d_0e0f_0809_0a0bu64 as i64,
                0x0405_0607_0001_0203u64 as i64,
            );
            for stream in 0..4 {
                let feba = _mm_shuffle_epi32(abef[stream], 0x1b);
                let dchg = _mm_shuffle_epi32(cdgh[stream], 0xb1);
                let dcba = _mm_blend_epi16(feba, dchg, 0xf0);
                let hgef = _mm_alignr_epi8(dchg, feba, 8);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(32 * stream).cast(),
                    _mm_shuffle_epi8(dcba, endian),
                );
                _mm_storeu_si128(
                    out.as_mut_ptr().add(32 * stream + 16).cast(),
                    _mm_shuffle_epi8(hgef, endian),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portable_digest(data: &[u8]) -> [u8; 32] {
        let mut padded = data.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
        let mut state = IV256;
        for block in padded.as_chunks::<64>().0 {
            compress_block_portable(&mut state, block);
        }
        let mut result = [0; 32];
        for (out, word) in result.chunks_mut(4).zip(state) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        result
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_sha_ni_compression_matches_portable_states() {
        if !(std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1"))
        {
            return;
        }
        let mut portable = IV256;
        let mut accelerated = portable;
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        for iteration in 0..128 {
            let mut block = [0; 64];
            for byte in &mut block {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *byte = seed as u8;
            }
            if iteration % 7 == 0 {
                portable = core::array::from_fn(|i| (seed.rotate_left(i as u32) >> 16) as u32);
                accelerated = portable;
            }
            compress_block_portable(&mut portable, &block);
            // SAFETY: all required features were detected above.
            unsafe { compress_block_x86_64(&mut accelerated, &block) };
            assert_eq!(accelerated, portable);
        }
    }

    #[test]
    fn dispatched_sha256_matches_portable_padding_slices_and_merkle_nodes() {
        let data: Vec<u8> = (0..4096usize)
            .map(|i| (i.wrapping_mul(193) ^ (i >> 3) ^ (i * i)) as u8)
            .collect();
        for len in (0..=260).chain([511, 512, 513, 1023, 1024, 2048, 4096]) {
            let input = &data[..len];
            let expected = portable_digest(input);
            assert_eq!(digest(input), expected, "length {len}");
            let mut slices = vec![&input[..0]];
            slices.extend(input.chunks(17));
            slices.push(&input[len..]);
            assert_eq!(digest_slices(&slices), expected, "sliced length {len}");
        }
        for offset in 0..64 {
            let (left, right) = (&data[offset..offset + 32], &data[offset + 32..offset + 64]);
            let mut message = vec![1];
            message.extend_from_slice(left);
            message.extend_from_slice(right);
            let mut out = [0; 32];
            hash_node_64(left, right, &mut out);
            assert_eq!(out, portable_digest(&message));
        }
    }

    #[test]
    fn four_stream_prefixed_hashes_match_portable_at_all_padding_boundaries() {
        let data: [Vec<u8>; 4] = core::array::from_fn(|stream| {
            (0..2048usize)
                .map(|i| ((i * 193) ^ (i >> 2) ^ (stream * 97)) as u8)
                .collect()
        });
        for prefix in [0, 1, 0x87] {
            for len in (0..=260).chain([511, 512, 513, 1023, 1024, 2048]) {
                let mut output = [0; 128];
                digest4_prefixed_into(
                    prefix,
                    core::array::from_fn(|i| &data[i][..len]),
                    &mut output,
                );
                for (stream, digest) in output.chunks(32).enumerate() {
                    let mut input = vec![prefix];
                    input.extend_from_slice(&data[stream][..len]);
                    assert_eq!(
                        digest,
                        portable_digest(&input),
                        "prefix={prefix}, len={len}, stream={stream}"
                    );
                }
            }
            let mut output = [0; 128];
            digest4_prefixed_into(
                prefix,
                core::array::from_fn(|i| &data[i][..17 * i]),
                &mut output,
            );
            for (stream, digest) in output.chunks(32).enumerate() {
                let mut input = vec![prefix];
                input.extend_from_slice(&data[stream][..17 * stream]);
                assert_eq!(digest, portable_digest(&input));
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_sha2_matches_portable() {
        if !std::arch::is_aarch64_feature_detected!("sha2") {
            return;
        }
        let mut h_port = IV256;
        let mut h_hw = IV256;
        for b in 0..32u8 {
            let mut block = [0u8; 64];
            for (i, byte) in block.iter_mut().enumerate() {
                *byte = b.wrapping_mul(37).wrapping_add(i as u8);
            }
            compress_block_portable(&mut h_port, &block);
            // SAFETY: `sha2` was detected just above.
            unsafe { compress_block_aarch64(&mut h_hw, &block) };
            assert_eq!(h_hw, h_port);
        }
    }

    #[test]
    fn digest_slices_matches_concatenated() {
        let a = b"hello ";
        let b = &[0u8; 73];
        let c = b"world of open magnetar";
        let mut full = Vec::new();
        full.extend_from_slice(a);
        full.extend_from_slice(b);
        full.extend_from_slice(c);
        assert_eq!(digest_slices(&[a, b, c]), digest(&full));
    }
}
