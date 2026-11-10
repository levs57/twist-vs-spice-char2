//! Commitment backend equivalence across layouts, hash profiles and workers.
mod common;

use common::Rng;
use om_field::K;
use om_pcs::{
    ligerito::{self, LigeritoParams},
    ntt,
};
use om_transcript::Hasher;

#[test]
fn column_major_cosets_match_defining_generator_matrix() {
    for workers in [1, 4] {
        rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap().install(|| {
            let mut rng = Rng(0x1829);
            for (rows, cols, rate) in [(8, 1, 2), (1, 8, 0), (2, 8, 2), (32, 1024, 2), (64, 2048, 1)] {
                let source = rng.ks(rows * cols);
                let mu = (cols as usize).ilog2() as usize;
                for coset in 0..1usize << rate {
                    let mut encoded = source.clone();
                    ntt::encode_coset_col_major(&mut encoded, rows, cols, rate, coset);
                    for local in [0, cols / 2, cols - 1] {
                        let basis = ntt::basis_row(coset * cols + local, mu, mu + rate);
                        for row in [0, rows / 2, rows - 1] {
                            let expected = basis.iter().enumerate().fold(K::ZERO, |sum, (j, &weight)| {
                                sum + source[row + rows * j] * weight
                            });
                            assert_eq!(encoded[row + rows * local], expected,
                                "workers={workers} rows={rows} cols={cols} coset={coset} q={local} row={row}");
                        }
                    }
                }
            }
        });
    }
}

#[test]
fn stripped_commitment_matches_retained_columns_roots_and_authentication() {
    for workers in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                let mut rng = Rng(0x9a8b);
                for hash in [Hasher::Sha256, Hasher::Sha3_256, Hasher::Sha384] {
                    for (n, k, rate) in [(4, 1, 1), (10, 7, 2), (17, 5, 2)] {
                        if n == 17 && hash != Hasher::Sha256 {
                            continue;
                        }
                        let source = rng.ks(1 << n);
                        let params = LigeritoParams::new(n, vec![k], vec![rate], vec![1]);
                        let (retained_root, retained) =
                            ligerito::commit_owned(&params, hash, source.clone());
                        let (stripped_root, stripped) =
                            ligerito::commit_stripped(&params, hash, source.clone());
                        assert_eq!(stripped_root, retained_root);
                        assert!(stripped.f.is_empty() && stripped.cw.is_empty());
                        let leaves = retained.tree.num_leaves;
                        let mut queries = vec![0, 1, leaves / 2, leaves - 1];
                        queries.sort_unstable();
                        queries.dedup();
                        assert_eq!(
                            stripped.tree.open(&queries).unwrap(),
                            retained.tree.open(&queries).unwrap()
                        );
                        assert_eq!(
                            stripped.columns(&queries, &source),
                            retained.columns(&queries, &source)
                        );
                    }
                }
            });
    }
}

#[test]
fn four_stream_merkle_batches_preserve_mixed_lengths_short_tails_and_profiles() {
    use om_pcs::merkle::{MerkleTree, hash_leaf};
    for hash in [Hasher::Sha256, Hasher::Sha3_256, Hasher::Sha384] {
        for count in [1, 2, 3, 4, 5, 7, 8, 9, 1025] {
            let leaves: Vec<Vec<u8>> = (0..count)
                .map(|i| {
                    let len = if i % 9 < 5 { 64 } else { i % 129 };
                    (0..len).map(|j| (i * 13 + j * 193) as u8).collect()
                })
                .collect();
            let expected = MerkleTree::from_leaf_hashes(
                hash,
                leaves.iter().map(|leaf| hash_leaf(hash, leaf)).collect(),
            );
            let actual = MerkleTree::new(hash, &leaves);
            assert_eq!(actual.root(), expected.root());
            let mut queries = vec![0, count / 2, count - 1];
            queries.sort_unstable();
            queries.dedup();
            assert_eq!(
                actual.open(&queries).unwrap(),
                expected.open(&queries).unwrap()
            );
        }
    }
}
