//! Small helpers shared by the Twist v2 provers: split equality tables, word
//! evaluation by byte lookup and sum-check message plumbing.

use om_field::{K, U256};
use om_sumcheck::{eq_table, interpolate_eval};
use om_transcript::Transcript;
use rayon::prelude::*;

/// Below this many items, loops stay sequential.
pub(crate) const PAR: usize = 1 << 12;

pub(crate) fn sublabel(label: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(label.len() + suffix.len());
    out.extend_from_slice(label);
    out.extend_from_slice(suffix);
    out
}

/// The sum-check node `t_i` (the field element with integer encoding `i`).
#[inline]
pub(crate) fn node(i: usize) -> K {
    K::from_u128(i as u128)
}

/// `eq(r, j)` for all `j < 2^len(r)` from two tables of about `2^(len/2)`.
pub(crate) struct SplitEq {
    pub(crate) lo: Vec<K>,
    pub(crate) hi: Vec<K>,
    pub(crate) h: usize,
}

impl SplitEq {
    pub(crate) fn new(r: &[K]) -> Self {
        let h = r.len() / 2;
        SplitEq {
            lo: eq_table(&r[..h]),
            hi: eq_table(&r[h..]),
            h,
        }
    }

    #[inline]
    pub(crate) fn at(&self, j: usize) -> K {
        self.lo[j & ((1usize << self.h) - 1)] * self.hi[j >> self.h]
    }

    /// The full table `eq(r, j)`, computed in parallel.
    pub(crate) fn table(&self) -> Vec<K> {
        let n = self.lo.len() * self.hi.len();
        let mut out = vec![K::ZERO; n];
        let lo = &self.lo;
        out.par_chunks_mut(lo.len())
            .zip(self.hi.par_iter())
            .for_each(|(chunk, &h)| {
                for (dst, &l) in chunk.iter_mut().zip(lo) {
                    *dst = l * h;
                }
            });
        out
    }
}

/// `LT~(bits(j), y)` for all `j` from split tables.
pub(crate) struct SplitLt {
    lt_lo: Vec<K>,
    lt_hi: Vec<K>,
    eq_hi: Vec<K>,
    h: usize,
}

impl SplitLt {
    pub(crate) fn new(y: &[K]) -> Self {
        let h = y.len() / 2;
        SplitLt {
            lt_lo: om_memcheck::twist::lt_table(&y[..h]),
            lt_hi: om_memcheck::twist::lt_table(&y[h..]),
            eq_hi: eq_table(&y[h..]),
            h,
        }
    }

    #[inline]
    pub(crate) fn at(&self, j: usize) -> K {
        let (lo, hi) = (j & ((1usize << self.h) - 1), j >> self.h);
        self.lt_hi[hi] + self.eq_hi[hi] * self.lt_lo[lo]
    }
}

/// `w ↦ Σ_o eq(r_o, o) · bit_o(w)`: one word's MLE in the bit variables.
pub(crate) struct WordEval {
    tabs: Vec<[K; 256]>,
    mask: u64,
}

impl WordEval {
    pub(crate) fn new(r_o: &[K]) -> Self {
        assert!(r_o.len() <= 6, "WordEval: ell must be <= 6");
        let eo = eq_table(r_o);
        let l = eo.len();
        let nbytes = l.div_ceil(8);
        let mut tabs = vec![[K::ZERO; 256]; nbytes];
        for (b, tab) in tabs.iter_mut().enumerate() {
            for v in 1..256usize {
                let low = v & (v - 1);
                let bit = v.trailing_zeros() as usize;
                let o = 8 * b + bit;
                tab[v] = tab[low] + if o < l { eo[o] } else { K::ZERO };
            }
        }
        let mask = if l >= 64 { u64::MAX } else { (1u64 << l) - 1 };
        WordEval { tabs, mask }
    }

    #[inline]
    pub(crate) fn eval(&self, w: u64) -> K {
        let w = w & self.mask;
        let mut acc = K::ZERO;
        for (b, tab) in self.tabs.iter().enumerate() {
            acc += tab[((w >> (8 * b)) & 0xff) as usize];
        }
        acc
    }

    pub(crate) fn eval_all(&self, words: &[u64]) -> Vec<K> {
        if words.len() >= PAR {
            words.par_iter().map(|&w| self.eval(w)).collect()
        } else {
            words.iter().map(|&w| self.eval(w)).collect()
        }
    }
}

/// `MLE(word table)(r_o ‖ r_c)` for `words.len() == 2^len(r_c)`, with bit
/// `o` of word `j` at index `o + 2^len(r_o) j`.
pub(crate) fn word_mle(words: &[u64], r_o: &[K], r_c: &[K]) -> K {
    assert_eq!(words.len(), 1usize << r_c.len(), "word_mle: length");
    let we = WordEval::new(r_o);
    let se = SplitEq::new(r_c);
    let block = |hi: usize| -> K {
        let base = hi << se.h;
        let mut acc = U256::ZERO;
        for (lo, &e) in se.lo.iter().enumerate() {
            acc += e.clmul(we.eval(words[base | lo]));
        }
        se.hi[hi] * acc.reduce()
    };
    if words.len() >= PAR {
        (0..se.hi.len())
            .into_par_iter()
            .map(block)
            .reduce(|| K::ZERO, |a, b| a + b)
    } else {
        (0..se.hi.len()).map(block).fold(K::ZERO, |a, b| a + b)
    }
}

/// Emit one round: absorb `[g(t_0), g(t_2), …]`, draw the challenge, and
/// return it together with `g(r)`.
pub(crate) fn emit_round(
    ys: &[K],
    transcript: &mut Transcript,
    label: &[u8],
    msgs: &mut Vec<Vec<K>>,
) -> (K, K) {
    let mut msg = Vec::with_capacity(ys.len() - 1);
    msg.push(ys[0]);
    msg.extend_from_slice(&ys[2..]);
    transcript.absorb_ks(&sublabel(label, b"/msg"), &msg);
    let r = transcript.challenge_k(&sublabel(label, b"/chal"));
    msgs.push(msg);
    (r, interpolate_eval(ys, r))
}

static LENIENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether provers assert their own round sums. On in debug builds, except
/// in negative tests, which let a cheating prover finish so that the
/// verifier's rejection is what the test observes.
#[inline]
pub(crate) fn checks_enabled() -> bool {
    cfg!(debug_assertions) && !LENIENT.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn set_lenient() {
    LENIENT.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// A zeroed vector, filled in parallel when large. `vec![K::ZERO; len]`
/// fills serially because `K` is not a primitive.
pub(crate) fn zeros(len: usize) -> Vec<K> {
    if len >= 1 << 16 {
        (0..len)
            .into_par_iter()
            .with_min_len(1 << 14)
            .map(|_| K::ZERO)
            .collect()
    } else {
        vec![K::ZERO; len]
    }
}

/// Build a vector of `len` elements in parallel chunks of `chunk` elements.
/// `fill(start, out)` must write every element of `out`, which starts at
/// index `start`. This avoids zero-filling memory that is overwritten.
pub(crate) fn par_build<T: Copy + Send + Sync>(
    len: usize,
    chunk: usize,
    fill: impl Fn(usize, &mut [std::mem::MaybeUninit<T>]) + Sync,
) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(len);
    v.spare_capacity_mut()[..len]
        .par_chunks_mut(chunk.max(1))
        .enumerate()
        .for_each(|(c, out)| fill(c * chunk.max(1), out));
    // SAFETY: every element in 0..len was written by `fill`.
    unsafe { v.set_len(len) };
    v
}
