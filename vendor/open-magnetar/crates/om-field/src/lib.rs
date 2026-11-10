//! The binary field `K = GF(2^128)` of Open Magnetar.
//!
//! Normative references: `docs/spec/CONVENTIONS.md` C1 and
//! `docs/spec/01-fields.md`. Behavioral reference: class `K` in
//! `reference/openmagnetar/field.py`.
//!
//! `K = GF(2)[X] / (X^128 + X^7 + X^2 + X + 1)`. An element is identified
//! with the 128-bit unsigned integer whose bit `i` is the coefficient of
//! `X^i` — the plain polynomial basis, **not** GHASH's bit-reflected
//! convention. Addition is XOR (so subtraction is addition and negation is
//! the identity); multiplication is carry-less polynomial multiplication
//! followed by reduction modulo `X^128 + X^7 + X^2 + X + 1`.
//!
//! Encoding (C1/C8, spec 01 §1.2): 16 bytes, little-endian — byte 0 holds
//! bits 0..7. Every 16-byte string is a valid canonical encoding. The
//! textual form (used by [`Display`](core::fmt::Display), [`K::to_hex`],
//! and test vectors) is the 32-character lowercase hex of those
//! little-endian bytes. [`Debug`](core::fmt::Debug) instead mirrors the
//! Python `repr`, `K(0x…)`, printed from the integer value.

// In characteristic 2, addition and subtraction genuinely are XOR and
// division genuinely is multiplication by the inverse.
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::suspicious_op_assign_impl)]

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

/// `X^128 mod m(X) = X^7 + X^2 + X + 1`, the low terms of the modulus
/// `m(X) = X^128 + X^7 + X^2 + X + 1` (as an integer, `(1 << 128) | 0x87`).
const REDUCTION: u128 = 0x87;

/// An element of `K = GF(2^128)`, stored as its integer value (bit `i` =
/// coefficient of `X^i`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct K(u128);

impl K {
    /// The zero element, `0`.
    pub const ZERO: K = K(0);
    /// The one element, `1` (also the embedding of `1 ∈ GF(2)`).
    pub const ONE: K = K(1);
    /// The element `X` (integer value 2).
    pub const X: K = K(2);

    /// The element with integer value `v`. Every `u128` is in range, so
    /// this never fails (the Python `K(v)` range check is vacuous here).
    #[inline(always)]
    pub const fn from_u128(v: u128) -> K {
        K(v)
    }

    /// The field point `t_k`: the element with integer value `k`
    /// (CONVENTIONS C1). So `t(0) = 0`, `t(1) = 1`, `t(2) = X`,
    /// `t(3) = X + 1`.
    #[inline(always)]
    pub const fn t(k: u128) -> K {
        K(k)
    }

    /// The integer value of this element (bit `i` = coefficient of `X^i`).
    #[inline(always)]
    pub const fn value(self) -> u128 {
        self.0
    }

    /// The embedding of a bit of `B = GF(2)`: `false ↦ 0`, `true ↦ 1`.
    #[inline(always)]
    pub const fn from_bit(b: bool) -> K {
        K(b as u128)
    }

    /// Whether this element is zero.
    #[inline(always)]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Whether this element is one.
    #[inline(always)]
    pub const fn is_one(self) -> bool {
        self.0 == 1
    }

    // ---- byte / hex encoding (spec 01 §1.2, CONVENTIONS C8) ----

    /// The canonical 16-byte little-endian encoding: byte `i` holds
    /// integer bits `8i..8i+7`, so byte 0 holds `X^0..X^7`.
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// Decode a canonical 16-byte little-endian encoding. Every 16-byte
    /// string is valid.
    #[inline(always)]
    pub fn from_bytes(b: &[u8; 16]) -> K {
        K(u128::from_le_bytes(*b))
    }

    /// Decode from a byte slice, rejecting any length other than 16
    /// (mirrors Python `K.from_bytes`, which raises on other lengths).
    #[inline]
    pub fn from_byte_slice(b: &[u8]) -> Option<K> {
        let arr: &[u8; 16] = b.try_into().ok()?;
        Some(K::from_bytes(arr))
    }

    /// The 32-character lowercase hex of the little-endian byte encoding
    /// (the test-vector form of CONVENTIONS C8; Python `K.hex`).
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(32);
        for byte in self.to_bytes() {
            s.push(char::from_digit(u32::from(byte >> 4), 16).expect("nibble < 16"));
            s.push(char::from_digit(u32::from(byte & 0xf), 16).expect("nibble < 16"));
        }
        s
    }

    /// Parse the 32-character hex form produced by [`K::to_hex`]
    /// (upper- or lowercase digits accepted, as in Python
    /// `bytes.fromhex`). Returns `None` unless the input is exactly 32
    /// hex characters.
    pub fn from_hex(s: &str) -> Option<K> {
        let bytes = s.as_bytes();
        if bytes.len() != 32 {
            return None;
        }
        let mut out = [0u8; 16];
        for (i, byte) in out.iter_mut().enumerate() {
            let hi = hex_val(bytes[2 * i])?;
            let lo = hex_val(bytes[2 * i + 1])?;
            *byte = (hi << 4) | lo;
        }
        Some(K::from_bytes(&out))
    }

    // ---- packing (spec 01 §1.3) ----

    /// Pack up to 128 bits with basis `β_j = X^j`: bit `j` of the result
    /// is `bits[j]`. Missing high bits are zero, as in Python `K.pack`.
    ///
    /// # Panics
    /// Panics if `bits.len() > 128`.
    pub fn pack(bits: &[bool]) -> K {
        assert!(bits.len() <= 128, "K::pack expects at most 128 bits");
        let mut v = 0u128;
        for (j, &bit) in bits.iter().enumerate() {
            v |= u128::from(bit) << j;
        }
        K(v)
    }

    /// The inverse of [`K::pack`]: entry `j` is the coefficient of `X^j`.
    pub fn unpack(self) -> [bool; 128] {
        let mut out = [false; 128];
        for (j, bit) in out.iter_mut().enumerate() {
            *bit = (self.0 >> j) & 1 == 1;
        }
        out
    }

    // ---- arithmetic ----

    /// The Frobenius map `x ↦ x^2` (Python `K.square`).
    #[inline(always)]
    pub fn square(self) -> K {
        let (lo, hi) = square128(self.0);
        K(reduce(lo, hi))
    }

    /// `self^e` for a `u128` exponent, by square-and-multiply. `0^0 = 1`,
    /// as in Python.
    pub fn pow(self, mut e: u128) -> K {
        let mut r = K::ONE;
        let mut b = self;
        while e != 0 {
            if e & 1 == 1 {
                r *= b;
            }
            b = b.square();
            e >>= 1;
        }
        r
    }

    /// `self^e` for an exponent given as little-endian 64-bit limbs, for
    /// exponents that do not fit in `u128` (e.g. the Fermat exponent
    /// `2^128`, limbs `[0, 0, 1]`). An empty or all-zero exponent gives
    /// `1` (so `0^0 = 1`, as in Python).
    pub fn pow_limbs(self, exp: &[u64]) -> K {
        let mut r = K::ONE;
        // Square-and-multiply from the most significant bit down.
        let mut started = false;
        for &limb in exp.iter().rev() {
            for i in (0..64).rev() {
                if started {
                    r = r.square();
                }
                if (limb >> i) & 1 == 1 {
                    if started {
                        r *= self;
                    } else {
                        r = self;
                        started = true;
                    }
                }
            }
        }
        r
    }

    /// The multiplicative inverse, or `None` for zero.
    pub fn inv(self) -> Option<K> {
        if self.is_zero() {
            None
        } else {
            // a^(2^128 - 2), as in Python `K.inv`.
            Some(self.pow(u128::MAX - 1))
        }
    }

    /// The multiplicative inverse, panicking on zero. This mirrors Python
    /// `K.inv`, which raises `ZeroDivisionError` on zero; prefer
    /// [`K::inv`] where zero is possible.
    ///
    /// # Panics
    /// Panics if `self` is zero.
    pub fn inverse(self) -> K {
        self.inv().expect("inverse of zero in K")
    }

    /// The product `b · self` for a bit `b ∈ B` (spec 01 §1.5): `self` if
    /// `b` is set, else zero.
    #[inline(always)]
    pub fn mul_bit(self, b: bool) -> K {
        if b { self } else { K::ZERO }
    }

    /// Multiply `self` by `X = K(2)`: `(v << 1) ^ (if v >> 127 != 0 { 0x87 } else { 0 })`.
    #[inline(always)]
    pub const fn mul_x(self) -> K {
        let hi = (self.0 as i128 >> 127) as u128;
        K((self.0 << 1) ^ (hi & REDUCTION))
    }

    /// Multiply `self` by the field point `K::t(k)`, using shift-and-XOR
    /// for small `k <= 63` (the sum-check evaluation points `t_0..t_63`).
    #[inline(always)]
    pub fn mul_t(self, k: u128) -> K {
        match k {
            0 => K::ZERO,
            1 => self,
            2 => self.mul_x(),
            3 => self.mul_x() + self,
            4 => self.mul_x().mul_x(),
            5 => self.mul_x().mul_x() + self,
            6 => {
                let x = self.mul_x();
                x.mul_x() + x
            }
            7 => {
                let x = self.mul_x();
                x.mul_x() + x + self
            }
            8..=15 => self.mul_t(k ^ 8) + self.mul_x().mul_x().mul_x(),
            16..=31 => self.mul_t(k ^ 16) + self.mul_x().mul_x().mul_x().mul_x(),
            32..=63 => self.mul_t(k ^ 32) + self.mul_x().mul_x().mul_x().mul_x().mul_x(),
            _ => self * K::t(k),
        }
    }

    /// Unreduced 128×128 → 256-bit carry-less product `self · rhs` in `GF(2)[X]`.
    /// By linearity of modular reduction over `GF(2)`, a sum of products
    /// `Σ_i a_i * b_i` can be accumulated as `U256` via XOR (`+` / `+=`) and
    /// reduced once with [`U256::reduce`].
    #[inline(always)]
    pub fn clmul(self, rhs: K) -> U256 {
        let (lo, hi) = clmul128(self.0, rhs.0);
        U256(lo, hi)
    }
}

/// Apply the additive-transform butterfly `left += factor*right; right += left`
/// elementwise. The slices must have equal lengths and cannot overlap.
/// Packed x86 lanes use the same polynomial basis and reduction as scalar `K`.
#[inline]
pub fn butterfly(left: &mut [K], right: &mut [K], factor: K) {
    assert_eq!(left.len(), right.len(), "butterfly slice lengths");
    if factor.is_zero() {
        for (&a, b) in left.iter().zip(right) {
            *b += a;
        }
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        // SAFETY: the compile-time features and equal disjoint slices satisfy
        // the packed kernel's preconditions.
        unsafe { butterfly_x86_64(left, right, factor) };
        return;
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "vpclmulqdq"))
    ))]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            // SAFETY: the required features were detected above.
            unsafe { butterfly_x86_64(left, right, factor) };
            return;
        }
    }
    #[allow(unreachable_code)]
    for (a, b) in left.iter_mut().zip(right) {
        *a += factor * *b;
        *b += *a;
    }
}

/// Two independent reduced products. Native builds with AVX2 and VPCLMULQDQ
/// use packed lanes; other builds use scalar products without runtime dispatch.
#[inline(always)]
pub fn mul_pairs(left: [K; 2], right: [K; 2]) -> [K; 2] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        // SAFETY: the target features are enabled and arrays have two K lanes.
        unsafe {
            use core::arch::x86_64::*;
            let a = _mm256_loadu_si256(left.as_ptr().cast());
            let b = _mm256_loadu_si256(right.as_ptr().cast());
            let mut out = [K::ZERO; 2];
            _mm256_storeu_si256(out.as_mut_ptr().cast(), mul_x2(a, b));
            return out;
        }
    }
    #[allow(unreachable_code)]
    [left[0] * right[0], left[1] * right[1]]
}

/// Two independent unreduced products for delayed XOR accumulation. Packed
/// lanes require a native AVX2/VPCLMULQDQ build; no reduction is performed.
#[inline(always)]
pub fn clmul_pairs(left: [K; 2], right: [K; 2]) -> [U256; 2] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        // SAFETY: all vector loads/stores cover complete two-element arrays.
        unsafe {
            use core::arch::x86_64::*;
            let (lo, hi) = clmul_x2(
                _mm256_loadu_si256(left.as_ptr().cast()),
                _mm256_loadu_si256(right.as_ptr().cast()),
            );
            let (mut low, mut high) = ([0u128; 2], [0u128; 2]);
            _mm256_storeu_si256(low.as_mut_ptr().cast(), lo);
            _mm256_storeu_si256(high.as_mut_ptr().cast(), hi);
            return [U256(low[0], high[0]), U256(low[1], high[1])];
        }
    }
    #[allow(unreachable_code)]
    [left[0].clmul(right[0]), left[1].clmul(right[1])]
}

/// Bind sibling slices: `left[i] += r * (left[i] + right[i])`.
#[inline]
pub fn fold_slices(left: &mut [K], right: &[K], r: K) {
    assert_eq!(left.len(), right.len(), "fold slice lengths");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        // SAFETY: the target features and disjoint equal-length slices suffice.
        unsafe { fold_slices_x86_64(left, right, r) };
        return;
    }
    #[allow(unreachable_code)]
    for (a, &b) in left.iter_mut().zip(right) {
        *a += r * (*a + b);
    }
}

/// Bind contiguous sibling pairs into `output`, whose length is half `source`.
#[inline]
pub fn fold_pairs(source: &[K], r: K, output: &mut [K]) {
    assert_eq!(source.len(), 2 * output.len(), "fold pair lengths");
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "vpclmulqdq"
    ))]
    {
        // SAFETY: the target features and exact disjoint slice lengths suffice.
        unsafe { fold_pairs_x86_64(source, r, output) };
        return;
    }
    #[allow(unreachable_code)]
    for (pair, out) in source.chunks_exact(2).zip(output) {
        *out = pair[0] + r * (pair[0] + pair[1]);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
#[inline]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn clmul_x2(
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    use core::arch::x86_64::*;
    let low = _mm256_clmulepi64_epi128::<0x00>(a, b);
    let high = _mm256_clmulepi64_epi128::<0x11>(a, b);
    let cross = _mm256_xor_si256(
        _mm256_clmulepi64_epi128::<0x01>(a, b),
        _mm256_clmulepi64_epi128::<0x10>(a, b),
    );
    (
        _mm256_xor_si256(low, _mm256_slli_si256::<8>(cross)),
        _mm256_xor_si256(high, _mm256_srli_si256::<8>(cross)),
    )
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn mul_x2(
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let polynomial = _mm256_set_epi64x(0, 0x87, 0, 0x87);
    let mut t0 = _mm256_clmulepi64_epi128::<0x00>(a, b);
    let mut t1 = _mm256_xor_si256(
        _mm256_clmulepi64_epi128::<0x01>(a, b),
        _mm256_clmulepi64_epi128::<0x10>(a, b),
    );
    let t2 = _mm256_clmulepi64_epi128::<0x11>(a, b);
    t1 = _mm256_xor_si256(t1, _mm256_slli_si256::<8>(t2));
    t1 = _mm256_xor_si256(t1, _mm256_clmulepi64_epi128::<0x01>(t2, polynomial));
    t0 = _mm256_xor_si256(t0, _mm256_slli_si256::<8>(t1));
    _mm256_xor_si256(t0, _mm256_clmulepi64_epi128::<0x01>(t1, polynomial))
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
#[inline]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn fold_slices_x86_64(left: &mut [K], right: &[K], r: K) {
    use core::arch::x86_64::*;
    // SAFETY: callers supply equal, disjoint slices and required features.
    unsafe {
        let factor = _mm256_broadcastsi128_si256(_mm_set_epi64x((r.0 >> 64) as i64, r.0 as i64));
        let count = left.len() & !1;
        for i in (0..count).step_by(2) {
            let a = _mm256_loadu_si256(left.as_ptr().add(i).cast());
            let b = _mm256_loadu_si256(right.as_ptr().add(i).cast());
            let folded = _mm256_xor_si256(a, mul_x2(_mm256_xor_si256(a, b), factor));
            _mm256_storeu_si256(left.as_mut_ptr().add(i).cast(), folded);
        }
        for (a, &b) in left[count..].iter_mut().zip(&right[count..]) {
            *a += r * (*a + b);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "vpclmulqdq"
))]
#[inline]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn fold_pairs_x86_64(source: &[K], r: K, output: &mut [K]) {
    use core::arch::x86_64::*;
    // SAFETY: callers provide twice as many source elements as outputs;
    // each iteration reads four whole K elements and writes two.
    unsafe {
        let factor = _mm256_broadcastsi128_si256(_mm_set_epi64x((r.0 >> 64) as i64, r.0 as i64));
        let count = output.len() & !1;
        for i in (0..count).step_by(2) {
            let x = _mm256_loadu_si256(source.as_ptr().add(2 * i).cast());
            let y = _mm256_loadu_si256(source.as_ptr().add(2 * i + 2).cast());
            let a = _mm256_permute2x128_si256::<0x20>(x, y);
            let b = _mm256_permute2x128_si256::<0x31>(x, y);
            let folded = _mm256_xor_si256(a, mul_x2(_mm256_xor_si256(a, b), factor));
            _mm256_storeu_si256(output.as_mut_ptr().add(i).cast(), folded);
        }
        if count < output.len() {
            output[count] = source[2 * count] + r * (source[2 * count] + source[2 * count + 1]);
        }
    }
}

// Two-lane AVX2 adaptation of Flock's ghash_mul_x4 and butterfly_row_pair.
// Copyright (c) 2026 Succinct Labs, Benedikt Bunz, and William Wang.
// Licensed under MIT OR Apache-2.0; donor notice in LICENSE-FLOCK-MIT.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,vpclmulqdq")]
unsafe fn butterfly_x86_64(left: &mut [K], right: &mut [K], factor: K) {
    use core::arch::x86_64::*;

    // SAFETY: K is transparent over u128; each vector covers exactly two
    // initialized elements. Mutable slices are disjoint and loads unaligned.
    unsafe {
        let twiddle =
            _mm256_broadcastsi128_si256(_mm_set_epi64x((factor.0 >> 64) as i64, factor.0 as i64));
        let count = left.len() & !1;
        for i in (0..count).step_by(2) {
            let a = _mm256_loadu_si256(left.as_ptr().add(i).cast());
            let b = _mm256_loadu_si256(right.as_ptr().add(i).cast());
            let next_left = _mm256_xor_si256(a, mul_x2(twiddle, b));
            _mm256_storeu_si256(left.as_mut_ptr().add(i).cast(), next_left);
            _mm256_storeu_si256(
                right.as_mut_ptr().add(i).cast(),
                _mm256_xor_si256(b, next_left),
            );
        }
        for (a, b) in left[count..].iter_mut().zip(&mut right[count..]) {
            *a += factor * *b;
            *b += *a;
        }
    }
}

/// An unreduced 256-bit polynomial in `GF(2)[X]` (`lo + hi · X^128`), used to
/// accumulate sums of products before a single modular reduction via [`U256::reduce`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct U256(pub u128, pub u128);

impl U256 {
    /// The zero polynomial.
    pub const ZERO: U256 = U256(0, 0);

    /// Embed a reduced field element `k` as `k + 0 · X^128`.
    #[inline(always)]
    pub const fn from_k(k: K) -> U256 {
        U256(k.0, 0)
    }

    /// Reduce modulo `m(X) = X^128 + X^7 + X^2 + X + 1` into [`K`].
    #[inline(always)]
    pub fn reduce(self) -> K {
        K(reduce(self.0, self.1))
    }
}

impl Add for U256 {
    type Output = U256;
    #[inline(always)]
    fn add(self, rhs: U256) -> U256 {
        U256(self.0 ^ rhs.0, self.1 ^ rhs.1)
    }
}

impl AddAssign for U256 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: U256) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl From<bool> for K {
    #[inline(always)]
    fn from(b: bool) -> K {
        K::from_bit(b)
    }
}

impl Add for K {
    type Output = K;
    #[inline(always)]
    fn add(self, rhs: K) -> K {
        K(self.0 ^ rhs.0)
    }
}

impl AddAssign for K {
    #[inline(always)]
    fn add_assign(&mut self, rhs: K) {
        self.0 ^= rhs.0;
    }
}

/// Subtraction is addition in characteristic 2.
impl Sub for K {
    type Output = K;
    #[inline(always)]
    fn sub(self, rhs: K) -> K {
        K(self.0 ^ rhs.0)
    }
}

impl SubAssign for K {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: K) {
        self.0 ^= rhs.0;
    }
}

/// Negation is the identity in characteristic 2.
impl Neg for K {
    type Output = K;
    #[inline(always)]
    fn neg(self) -> K {
        self
    }
}

impl Mul for K {
    type Output = K;
    #[inline(always)]
    fn mul(self, rhs: K) -> K {
        mul(self, rhs)
    }
}

impl MulAssign for K {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: K) {
        *self = *self * rhs;
    }
}

/// Division multiplies by the inverse (Python `__truediv__`).
///
/// # Panics
/// Panics if `rhs` is zero.
impl Div for K {
    type Output = K;
    fn div(self, rhs: K) -> K {
        self * rhs.inverse()
    }
}

impl DivAssign for K {
    fn div_assign(&mut self, rhs: K) {
        *self = *self / rhs;
    }
}

/// The [`Display`](fmt::Display) form is the vector encoding: 32-char
/// lowercase hex of the little-endian bytes (CONVENTIONS C8).
impl fmt::Display for K {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// The [`Debug`](fmt::Debug) form mirrors the Python `repr`:
/// `K(0x…)` with the integer value in hex.
impl fmt::Debug for K {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "K(0x{:032x})", self.0)
    }
}

/// Error returned when parsing a [`K`] from a string that is not exactly
/// 32 hex characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseKError;

impl fmt::Display for ParseKError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("K hex encoding must be exactly 32 hex characters")
    }
}

impl std::error::Error for ParseKError {}

impl FromStr for K {
    type Err = ParseKError;
    fn from_str(s: &str) -> Result<K, ParseKError> {
        K::from_hex(s).ok_or(ParseKError)
    }
}

// ---- multiplication: carry-less multiply + reduction (spec 01 §1.6) ----

#[inline(always)]
fn clmul128(a: u128, b: u128) -> (u128, u128) {
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: `aes` (FEAT_PMULL) is enabled at compile time.
        return unsafe { clmul128_aarch64(a, b) };
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
    {
        if std::arch::is_aarch64_feature_detected!("aes") {
            // SAFETY: the `aes` target feature (which includes PMULL) was
            // just detected at runtime on this CPU.
            return unsafe { clmul128_aarch64(a, b) };
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: PCLMULQDQ is enabled at compile time; SSE2 is part of the
        // x86-64 baseline.
        return unsafe { clmul128_x86_64(a, b) };
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "pclmulqdq")))]
    {
        if std::arch::is_x86_feature_detected!("pclmulqdq") {
            // SAFETY: PCLMULQDQ was just detected at runtime on this CPU;
            // SSE2 is part of the x86-64 baseline.
            return unsafe { clmul128_x86_64(a, b) };
        }
    }
    #[allow(unreachable_code)]
    clmul128_portable(a, b)
}

#[inline(always)]
fn mul(a: K, b: K) -> K {
    let (lo, hi) = clmul128(a.0, b.0);
    K(reduce(lo, hi))
}

/// Portable multiplication, always compiled; the reference that the
/// accelerated paths are tested against.
#[cfg(test)]
pub(crate) fn mul_portable(a: K, b: K) -> K {
    let (lo, hi) = clmul128_portable(a.0, b.0);
    K(reduce(lo, hi))
}

/// Carry-less product of two `u64`s as a `u128`, by shift-and-XOR.
fn clmul64_portable(a: u64, b: u64) -> u128 {
    let mut r = 0u128;
    let mut a = u128::from(a);
    let mut b = b;
    while b != 0 {
        if b & 1 == 1 {
            r ^= a;
        }
        a <<= 1;
        b >>= 1;
    }
    r
}

/// Carry-less 128×128 → 256 product as `(lo, hi)`, from four 64×64
/// products computed by `clmul64`.
#[inline(always)]
fn clmul128_with<Fn64: Fn(u64, u64) -> u128>(a: u128, b: u128, clmul64: Fn64) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let p00 = clmul64(a0, b0);
    let mid = clmul64(a0, b1) ^ clmul64(a1, b0);
    let p11 = clmul64(a1, b1);
    (p00 ^ (mid << 64), p11 ^ (mid >> 64))
}

fn clmul128_portable(a: u128, b: u128) -> (u128, u128) {
    clmul128_with(a, b, clmul64_portable)
}

/// Reduce the 256-bit product `(lo, hi)` modulo `m(X)` by folding `hi`
/// twice with `X^128 ≡ 0x87` (spec 01 §1.6): `lo ^= hi·0x87 (mod 2^128)`,
/// then the at-most-7-bit overflow of that product is folded once more.
#[inline(always)]
fn reduce(lo: u128, hi: u128) -> u128 {
    // hi · 0x87 = hi ^ hi<<1 ^ hi<<2 ^ hi<<7, up to 135 bits.
    let low = hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7);
    let overflow = (hi >> 127) ^ (hi >> 126) ^ (hi >> 121);
    debug_assert_eq!(REDUCTION, 0x87);
    lo ^ low ^ overflow ^ (overflow << 1) ^ (overflow << 2) ^ (overflow << 7)
}

/// In characteristic two the cross terms of a square cancel. Only the two
/// diagonal 64-bit products remain (the same specialization used by Flock).
#[inline(always)]
fn square128(value: u128) -> (u128, u128) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    {
        // SAFETY: PCLMULQDQ is enabled for the compiled target.
        return unsafe { square128_x86_64(value) };
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "pclmulqdq")))]
    if std::arch::is_x86_feature_detected!("pclmulqdq") {
        // SAFETY: the required instruction was detected above.
        return unsafe { square128_x86_64(value) };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    {
        // SAFETY: the AES/PMULL feature is enabled for the target.
        return unsafe { square128_aarch64(value) };
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
    if std::arch::is_aarch64_feature_detected!("aes") {
        // SAFETY: the AES/PMULL feature was detected above.
        return unsafe { square128_aarch64(value) };
    }
    #[allow(unreachable_code)]
    (spread64(value as u64), spread64((value >> 64) as u64))
}

#[inline(always)]
fn spread64(value: u64) -> u128 {
    let mut x = u128::from(value);
    x = (x | (x << 32)) & 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn square128_x86_64(value: u128) -> (u128, u128) {
    use core::arch::x86_64::*;
    let v = _mm_set_epi64x((value >> 64) as i64, value as i64);
    let low = _mm_clmulepi64_si128::<0x00>(v, v);
    let high = _mm_clmulepi64_si128::<0x11>(v, v);
    let mut out = [0u128; 2];
    // SAFETY: the two unaligned vector stores cover exactly the output array.
    unsafe {
        _mm_storeu_si128(out.as_mut_ptr().cast(), low);
        _mm_storeu_si128(out.as_mut_ptr().add(1).cast(), high);
    }
    (out[0], out[1])
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn square128_aarch64(value: u128) -> (u128, u128) {
    use core::arch::aarch64::vmull_p64;
    (
        vmull_p64(value as u64, value as u64),
        vmull_p64((value >> 64) as u64, (value >> 64) as u64),
    )
}

/// AArch64 128×128 → 256 carry-less product using `PMULL` (`vmull_p64`).
///
/// # Safety
/// The caller must ensure the CPU supports the `aes` target feature (FEAT_PMULL).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "aes")]
unsafe fn clmul128_aarch64(a: u128, b: u128) -> (u128, u128) {
    use core::arch::aarch64::vmull_p64;
    clmul128_with(a, b, |x, y| vmull_p64(x, y))
}

/// x86-64 128×128 → 256 carry-less product using `PCLMULQDQ`.
///
/// # Safety
/// The caller must ensure the CPU supports PCLMULQDQ and SSE2.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn clmul128_x86_64(a: u128, b: u128) -> (u128, u128) {
    use core::arch::x86_64::{__m128i, _mm_clmulepi64_si128, _mm_set_epi64x, _mm_storeu_si128};

    let clmul64 = |x: u64, y: u64| {
        let px = _mm_set_epi64x(0, x as i64);
        let py = _mm_set_epi64x(0, y as i64);
        let prod = _mm_clmulepi64_si128::<0x00>(px, py);
        let mut out = [0u64; 2];
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), prod) };
        u128::from(out[0]) | (u128::from(out[1]) << 64)
    };
    clmul128_with(a, b, clmul64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic splitmix64 stream of `u128`s (no `rand` crate).
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn next_k(&mut self) -> K {
            K(u128::from(self.next_u64()) | (u128::from(self.next_u64()) << 64))
        }
    }

    #[test]
    fn portable_matches_operator() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..512 {
            let (a, b) = (rng.next_k(), rng.next_k());
            assert_eq!(mul_portable(a, b), a * b);
        }
        // Edge operands, including the reduction-heavy high bits.
        for &v in &[0u128, 1, 2, 0x87, u128::MAX, 1 << 127, (1 << 127) | 3] {
            for &w in &[0u128, 1, 2, 0x87, u128::MAX, 1 << 127] {
                assert_eq!(mul_portable(K(v), K(w)), K(v) * K(w));
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn pmull_matches_portable() {
        if !std::arch::is_aarch64_feature_detected!("aes") {
            return;
        }
        let mut rng = Rng(0xdead_beef);
        for _ in 0..1024 {
            let (a, b) = (rng.next_k(), rng.next_k());
            // SAFETY: `aes` was detected just above.
            let (lo, hi) = unsafe { clmul128_aarch64(a.0, b.0) };
            assert_eq!(K(reduce(lo, hi)), mul_portable(a, b), "a={a:?} b={b:?}");
        }
        for &v in &[0u128, 1, u128::MAX, 1 << 127, 1 << 64, (1 << 64) - 1] {
            for &w in &[0u128, 1, u128::MAX, 1 << 127, 1 << 64] {
                // SAFETY: `aes` was detected just above.
                let (lo, hi) = unsafe { clmul128_aarch64(v, w) };
                assert_eq!(K(reduce(lo, hi)), mul_portable(K(v), K(w)));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pclmulqdq_matches_portable() {
        if !std::arch::is_x86_feature_detected!("pclmulqdq") {
            return;
        }
        let mut rng = Rng(0xdead_beef);
        for _ in 0..1024 {
            let (a, b) = (rng.next_k(), rng.next_k());
            // SAFETY: PCLMULQDQ was detected just above.
            let (lo, hi) = unsafe { clmul128_x86_64(a.0, b.0) };
            assert_eq!(K(reduce(lo, hi)), mul_portable(a, b), "a={a:?} b={b:?}");
        }
    }

    #[test]
    fn reduce_matches_python_fold() {
        // The reference reduce: while r >> 128: r = lo ^ h·0x87.
        fn py_reduce(lo: u128, hi: u128) -> u128 {
            // One fold may still overflow; emulate with a 256-bit pair.
            let (mut lo, mut hi) = (lo, hi);
            while hi != 0 {
                let (nlo, nhi) = clmul128_portable(hi, REDUCTION);
                lo ^= nlo;
                hi = nhi;
            }
            lo
        }
        let mut rng = Rng(7);
        for _ in 0..256 {
            let (a, b) = (rng.next_k(), rng.next_k());
            let (lo, hi) = clmul128_portable(a.0, b.0);
            assert_eq!(reduce(lo, hi), py_reduce(lo, hi));
        }
    }

    #[test]
    fn mul_x_and_mul_t_match_mul() {
        let mut rng = Rng(0x4242);
        for _ in 0..512 {
            let a = rng.next_k();
            assert_eq!(a.mul_x(), a * K::X);
            for k in 0..128 {
                assert_eq!(a.mul_t(k), a * K::t(k));
            }
        }
        for &v in &[0u128, 1, 2, 0x87, u128::MAX, 1 << 127, (1 << 127) | 3] {
            let a = K(v);
            assert_eq!(a.mul_x(), a * K::X);
            for k in 0..128 {
                assert_eq!(a.mul_t(k), a * K::t(k));
            }
        }
        for bit in 0..128 {
            let a = K(1u128 << bit);
            for k in 0..64 {
                assert_eq!(a.mul_t(k), mul_portable(a, K::t(k)));
            }
        }
    }

    #[test]
    fn packed_butterflies_match_portable_on_tails_and_unaligned_slices() {
        let mut rng = Rng(0x7a1b);
        for len in [0, 1, 2, 3, 4, 7, 8, 17, 64, 257, 1025] {
            for index in 0..8 {
                let factor = match index {
                    0 => K::ZERO,
                    1 => K::ONE,
                    2 => K::X,
                    3 => K(u128::MAX),
                    _ => rng.next_k(),
                };
                let mut left: Vec<K> = (0..len + 3).map(|_| rng.next_k()).collect();
                let mut right: Vec<K> = (0..len + 5).map(|_| rng.next_k()).collect();
                let (mut expected_left, mut expected_right) = (left.clone(), right.clone());
                for i in 0..len {
                    expected_left[i + 1] += mul_portable(factor, expected_right[i + 3]);
                    expected_right[i + 3] += expected_left[i + 1];
                }
                butterfly(&mut left[1..1 + len], &mut right[3..3 + len], factor);
                assert_eq!(left, expected_left, "left len={len}, factor={factor:?}");
                assert_eq!(right, expected_right, "right len={len}, factor={factor:?}");
            }
        }
    }

    #[test]
    fn packed_products_folds_and_specialized_squares_match_portable() {
        let mut rng = Rng(0x17cd);
        for iteration in 0..512 {
            let a = [rng.next_k(), rng.next_k()];
            let b = [rng.next_k(), rng.next_k()];
            let reduced = mul_pairs(a, b);
            let unreduced = clmul_pairs(a, b);
            for lane in 0..2 {
                assert_eq!(reduced[lane], mul_portable(a[lane], b[lane]));
                assert_eq!(unreduced[lane], a[lane].clmul(b[lane]));
                assert_eq!(a[lane].square(), mul_portable(a[lane], a[lane]));
                let (lo, hi) = (
                    spread64(a[lane].0 as u64),
                    spread64((a[lane].0 >> 64) as u64),
                );
                assert_eq!(K(reduce(lo, hi)), a[lane].square());
            }
            if iteration < 128 {
                let basis = K(1u128 << iteration);
                assert_eq!(basis.square(), mul_portable(basis, basis));
            }
        }
        for len in [0, 1, 2, 3, 7, 16, 65, 1025] {
            let source: Vec<K> = (0..2 * len + 3).map(|_| rng.next_k()).collect();
            for r in [K::ZERO, K::ONE, K::X, rng.next_k()] {
                let mut out = vec![K::ZERO; len];
                fold_pairs(&source[1..1 + 2 * len], r, &mut out);
                let expected: Vec<K> = source[1..1 + 2 * len]
                    .chunks_exact(2)
                    .map(|pair| pair[0] + mul_portable(r, pair[0] + pair[1]))
                    .collect();
                assert_eq!(out, expected);
                let mut left: Vec<K> = source[1..1 + 2 * len]
                    .chunks_exact(2)
                    .map(|pair| pair[0])
                    .collect();
                let right: Vec<K> = source[1..1 + 2 * len]
                    .chunks_exact(2)
                    .map(|pair| pair[1])
                    .collect();
                fold_slices(&mut left, &right, r);
                assert_eq!(left, expected);
            }
        }
    }

    #[test]
    fn clmul_unreduced_sum_matches_reduced() {
        let mut rng = Rng(0x9999);
        let mut sum_k = K::ZERO;
        let mut sum_u256 = U256::ZERO;
        for _ in 0..512 {
            let (a, b) = (rng.next_k(), rng.next_k());
            sum_k += a * b;
            sum_u256 += a.clmul(b);
        }
        assert_eq!(sum_u256.reduce(), sum_k);
    }
}
