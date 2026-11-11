//! The PCS trait and its Ligerito / bit-vector instantiations (spec 04 §4.1).
//! Mirrors `reference/openmagnetar/pcs.py`.

use om_field::K;
use om_sumcheck::{PackedBits, VerificationError, mle_eval, mle_eval_bits};
use om_transcript::{Hasher, Transcript};

use crate::ligerito::{self as lg, LevelData, LigeritoParams, LigeritoProof};
use crate::merkle::Digest;
use crate::ring_switch::{self as rsw, RingSwitchProof};

/// A commitment: one Merkle root plus the variable count (§4.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// The level-0 Merkle root.
    pub root: Digest,
    /// Variables of the committed (packed) polynomial.
    pub n: usize,
}

/// Single-point MLE opening interface over K (§4.1):
/// `commit` absorbs setup (`pcs/params`, `pcs/root`) into the transcript;
/// `open`/`verify` absorb `pcs/point` and `pcs/value` first (for
/// [`BitPCS`], after the `rs/*` ring-switching steps).
pub trait Pcs {
    /// What is committed: `[K]` for [`LigeritoPCS`], `[bool]` for [`BitPCS`].
    type Input: ?Sized;
    /// Commitment type.
    type Commitment;
    /// Prover state returned by `commit`.
    type State;
    /// Opening proof type.
    type Proof;

    /// Commit, absorbing setup into `tr`.
    fn commit(&self, input: &Self::Input, tr: &mut Transcript) -> (Self::Commitment, Self::State);

    /// Verifier-side setup absorption for `cm` (no commitment computed).
    fn absorb_commitment(&self, cm: &Self::Commitment, tr: &mut Transcript);

    /// Open at `r`, returning `(y, proof)` with `y` the MLE evaluation.
    fn open(&self, state: &Self::State, r: &[K], tr: &mut Transcript) -> (K, Self::Proof);

    /// Verify an opening claim.
    ///
    /// # Errors
    /// Returns [`VerificationError`] on failure.
    fn verify(
        &self,
        cm: &Self::Commitment,
        r: &[K],
        y: K,
        proof: &Self::Proof,
        tr: &mut Transcript,
    ) -> Result<(), VerificationError>;
}

/// Ligerito PCS for a K-valued table (§4.1, §4.4).
#[derive(Clone, Debug)]
pub struct LigeritoPCS {
    /// Parameters (Python `p`).
    pub p: LigeritoParams,
    /// Hash profile (Python `h`).
    pub h: Hasher,
}

impl LigeritoPCS {
    /// With the default SHA-256 profile.
    pub fn new(params: LigeritoParams) -> Self {
        LigeritoPCS {
            p: params,
            h: Hasher::Sha256,
        }
    }

    /// With an explicit hash profile.
    pub fn with_hasher(params: LigeritoParams, hasher: Hasher) -> Self {
        LigeritoPCS {
            p: params,
            h: hasher,
        }
    }

    /// `commit(table, tr) -> (cm, state)`, absorbing setup (§4.1).
    pub fn commit(&self, table: &[K], tr: &mut Transcript) -> (Commitment, LevelData) {
        let (root, state) = lg::commit(&self.p, self.h, table);
        lg::absorb_setup(&self.p, &root, tr);
        (Commitment { root, n: self.p.n }, state)
    }

    /// Verifier-side setup absorption for `cm`.
    pub fn absorb_commitment(&self, cm: &Commitment, tr: &mut Transcript) {
        lg::absorb_setup(&self.p, &cm.root, tr);
    }

    /// `open(state, r, tr) -> (y, proof)` with `y = f̃(r)`.
    pub fn open(&self, state: &LevelData, r: &[K], tr: &mut Transcript) -> (K, LigeritoProof) {
        let y = mle_eval(&state.f, r);
        (y, lg::open_(&self.p, self.h, state, r, y, tr))
    }

    /// `verify(cm, r, y, proof, tr)`.
    ///
    /// # Errors
    /// Returns [`VerificationError`] on a size mismatch or failed check.
    pub fn verify(
        &self,
        cm: &Commitment,
        r: &[K],
        y: K,
        proof: &LigeritoProof,
        tr: &mut Transcript,
    ) -> Result<(), VerificationError> {
        if cm.n != self.p.n {
            return Err(VerificationError::new("commitment size mismatch"));
        }
        lg::verify(&self.p, self.h, &cm.root, r, y, proof, tr)
    }
}

impl Pcs for LigeritoPCS {
    type Input = [K];
    type Commitment = Commitment;
    type State = LevelData;
    type Proof = LigeritoProof;

    fn commit(&self, input: &[K], tr: &mut Transcript) -> (Commitment, LevelData) {
        LigeritoPCS::commit(self, input, tr)
    }

    fn absorb_commitment(&self, cm: &Commitment, tr: &mut Transcript) {
        LigeritoPCS::absorb_commitment(self, cm, tr);
    }

    fn open(&self, state: &LevelData, r: &[K], tr: &mut Transcript) -> (K, LigeritoProof) {
        LigeritoPCS::open(self, state, r, tr)
    }

    fn verify(
        &self,
        cm: &Commitment,
        r: &[K],
        y: K,
        proof: &LigeritoProof,
        tr: &mut Transcript,
    ) -> Result<(), VerificationError> {
        LigeritoPCS::verify(self, cm, r, y, proof, tr)
    }
}

/// Prover state of [`BitPCS`]: the packed commitment state and the bits.
pub type BitState = (LevelData, Vec<bool>);

/// Prover state of the packed [`BitPCS::commit_packed`] path: the
/// level-0 Merkle tree only ([`LevelData::strip`]ped). The bits are
/// passed again to [`BitPCS::open_packed`], which packs the level table once,
/// after RingSwitch, then borrows it for Ligerito's initial sumcheck before
/// re-encoding queried columns.
pub type BitStatePacked = LevelData;
/// Proof of [`BitPCS`]: ring-switching proof and Ligerito proof.
pub type BitProof = (RingSwitchProof, LigeritoProof);

/// Bit-valued PCS (§4.1): commit to a bit vector of length `2^ℓ` (`ℓ ≥ 8`)
/// via its packing (packed `n = ℓ - 7`); open via ring switching (§4.5)
/// followed by a Ligerito opening of the packed polynomial.
#[derive(Clone, Debug)]
pub struct BitPCS {
    /// The underlying Ligerito PCS over the packed polynomial.
    pub pcs: LigeritoPCS,
}

impl BitPCS {
    /// With the default SHA-256 profile.
    pub fn new(params: LigeritoParams) -> Self {
        BitPCS {
            pcs: LigeritoPCS::new(params),
        }
    }

    /// With an explicit hash profile.
    pub fn with_hasher(params: LigeritoParams, hasher: Hasher) -> Self {
        BitPCS {
            pcs: LigeritoPCS::with_hasher(params, hasher),
        }
    }

    /// `commit(bits, tr) -> (cm, state)`: pack, then Ligerito-commit.
    pub fn commit(&self, bits: &[bool], tr: &mut Transcript) -> (Commitment, BitState) {
        let (root, st) = lg::commit_owned(&self.pcs.p, self.pcs.h, rsw::pack(bits));
        lg::absorb_setup(&self.pcs.p, &root, tr);
        (
            Commitment {
                root,
                n: self.pcs.p.n,
            },
            (st, bits.to_vec()),
        )
    }

    /// Verifier-side setup absorption (delegates to the inner PCS).
    pub fn absorb_commitment(&self, cm: &Commitment, tr: &mut Transcript) {
        self.pcs.absorb_commitment(cm, tr);
    }

    /// `open(state, r, tr) -> (y, (rs_proof, ligerito_proof))`.
    ///
    /// # Panics
    /// Panics if the packed opening value disagrees with ring switching
    /// (the Python `assert yp == v`).
    /// Commit to a packed bit vector: the same commitment as
    /// [`BitPCS::commit`] on the unpacked bits, without expanding
    /// them.
    ///
    /// # Panics
    /// As [`BitPCS::commit`].
    pub fn commit_packed(
        &self,
        bits: &PackedBits,
        tr: &mut Transcript,
    ) -> (Commitment, BitStatePacked) {
        let (root, st) = lg::commit_stripped(&self.pcs.p, self.pcs.h, rsw::pack_packed(bits));
        lg::absorb_setup(&self.pcs.p, &root, tr);
        (
            Commitment {
                root,
                n: self.pcs.p.n,
            },
            st,
        )
    }

    /// The packed counterpart of [`BitPCS::open`]: same transcript
    /// and proof bytes, computed from the commitment's tree and the
    /// packed `bits` it was made from. The value `y` comes from the
    /// ring-switching partial evaluations; no table over all `2^ell`
    /// bits is built.
    ///
    /// # Panics
    /// As [`BitPCS::open`] (a packed value inconsistent with ring
    /// switching fails the first Ligerito round's check), plus if
    /// `bits` has the wrong size for the tree.
    pub fn open_packed(
        &self,
        state: BitStatePacked,
        bits: &PackedBits,
        r: &[K],
        tr: &mut Transcript,
    ) -> (K, BitProof) {
        // RingSwitch reads its first table from packed bits without allocating
        // a full field table. Pack once afterwards, for Ligerito's initial
        // borrowed sumcheck and its subsequent queried encoding.
        let (rs_proof, r_s, v, y) = {
            let _span = om_transcript::profiling::Span::new("pcs.ringswitch");
            rsw::rs_prove_packed_value(bits, r, tr)
        };
        // `v` is the packed polynomial's value at `r_s` (what `open`
        // asserts against `mle_eval`); a wrong `v` would fail the first
        // Ligerito sum-check round's consistency check.
        let pf = {
            let _span = om_transcript::profiling::Span::new("pcs.ligerito_open");
            let packed = rsw::pack_packed(bits);
            lg::open_owned_with_table(&self.pcs.p, self.pcs.h, state, packed, &r_s, v, tr)
        };
        (y, (rs_proof, pf))
    }

    pub fn open(&self, state: &BitState, r: &[K], tr: &mut Transcript) -> (K, BitProof) {
        let (st, bits) = state;
        let (y, rs_proof, r_s, v) = {
            let _span = om_transcript::profiling::Span::new("pcs.ringswitch");
            let y = mle_eval_bits(bits, r);
            let (rs_proof, r_s, v) = rsw::rs_prove(bits, r, y, tr);
            (y, rs_proof, r_s, v)
        };
        let (yp, pf) = {
            let _span = om_transcript::profiling::Span::new("pcs.ligerito_open");
            self.pcs.open(st, &r_s, tr)
        };
        assert_eq!(yp, v, "packed opening must match the ring-switching output");
        (y, (rs_proof, pf))
    }

    /// `verify(cm, r, y, proof, tr)`: ring switching, then Ligerito.
    ///
    /// # Errors
    /// Returns [`VerificationError`] on any failed check.
    pub fn verify(
        &self,
        cm: &Commitment,
        r: &[K],
        y: K,
        proof: &BitProof,
        tr: &mut Transcript,
    ) -> Result<(), VerificationError> {
        let (rs_proof, pf) = proof;
        let (r_s, v) = rsw::rs_verify(r, y, rs_proof, tr)?;
        self.pcs.verify(cm, &r_s, v, pf, tr)
    }
}

impl Pcs for BitPCS {
    type Input = [bool];
    type Commitment = Commitment;
    type State = BitState;
    type Proof = BitProof;

    fn commit(&self, input: &[bool], tr: &mut Transcript) -> (Commitment, BitState) {
        BitPCS::commit(self, input, tr)
    }

    fn absorb_commitment(&self, cm: &Commitment, tr: &mut Transcript) {
        BitPCS::absorb_commitment(self, cm, tr);
    }

    fn open(&self, state: &BitState, r: &[K], tr: &mut Transcript) -> (K, BitProof) {
        BitPCS::open(self, state, r, tr)
    }

    fn verify(
        &self,
        cm: &Commitment,
        r: &[K],
        y: K,
        proof: &BitProof,
        tr: &mut Transcript,
    ) -> Result<(), VerificationError> {
        BitPCS::verify(self, cm, r, y, proof, tr)
    }
}

#[cfg(test)]
mod opening_copy_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn one_pack_opening_matches_repacking_path_and_preserves_stripped_state() {
        let mut seed = 0x19a7_25d3_4455u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (n, ks, rs, ts, gs) in [
            (4usize, vec![2], vec![2], vec![3], vec![1]),
            (8, vec![3, 3], vec![2, 1], vec![3, 4], vec![2, 1]),
            (14, vec![4, 4], vec![1, 2], vec![3, 4], vec![1, 1]),
        ] {
            let bits = PackedBits::from_words(
                (0..1usize << (n + 7 - 6)).map(|_| next()).collect(),
                1 << (n + 7),
            );
            let point: Vec<K> = (0..n + 7)
                .map(|i| match i % 3 {
                    0 => K::ZERO,
                    1 => K::ONE,
                    _ => K::from_u128(u128::from(next()) | (u128::from(next()) << 64)),
                })
                .collect();
            let pcs = BitPCS::new(LigeritoParams::with_gs(n, ks, rs, ts, gs));
            let mut commit_tr = Transcript::new_default(b"one-pack");
            let (cm, state) = pcs.commit_packed(&bits, &mut commit_tr);
            assert!(
                state.f.is_empty() && state.cw.is_empty(),
                "no table or codewords retained across the IOP"
            );
            let mut old_tr = commit_tr.clone();
            let mut new_tr = commit_tr;
            let old_state = state.clone();
            let (rs_proof, r_s, v, y) =
                rsw::rs_prove_packed_value_owned_reference(&bits, &point, &mut old_tr);
            let repacks = Cell::new(0usize);
            let old = lg::open_owned_repack(
                &pcs.pcs.p,
                pcs.pcs.h,
                old_state,
                &r_s,
                v,
                &mut old_tr,
                &|| {
                    repacks.set(repacks.get() + 1);
                    rsw::pack_packed(&bits)
                },
            );
            assert_eq!(
                repacks.get(),
                2,
                "old opening regenerates before and after level-zero sumcheck"
            );
            let new = pcs.open_packed(state, &bits, &point, &mut new_tr);
            assert_eq!(new, (y, (rs_proof, old)), "packed n={n}");
            assert_eq!(old_tr.challenge_k(b"after"), new_tr.challenge_k(b"after"));
            let mut verify_tr = Transcript::new_default(b"one-pack");
            pcs.absorb_commitment(&cm, &mut verify_tr);
            pcs.verify(&cm, &point, y, &new.1, &mut verify_tr).unwrap();
        }
    }
}
