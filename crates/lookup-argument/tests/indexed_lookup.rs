use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use lookup_argument::{
    ArgumentProof, Config, LookupProof, LookupTrace, Method, ProveResult, VirtualizationMode,
    prove, verify,
};
use om_field::K;

fn trace(a: usize, n: usize) -> LookupTrace {
    LookupTrace {
        table: (0..1usize << a)
            .map(|i| match i % 7 {
                0 => 0,
                1 => u64::MAX,
                _ => (i as u64).wrapping_mul(0xfedc_ba98_7654_3211),
            })
            .collect(),
        addresses: (0..1usize << n)
            .map(|j| {
                if j % 3 == 0 {
                    0
                } else {
                    ((j * 17 + j / 3) % (1 << a)) as u32
                }
            })
            .collect(),
        read_values: None,
    }
}
fn config(a: usize, c: usize) -> Config {
    Config {
        address_bits: a,
        chunk_bits: c,
        ..Config::default()
    }
}
fn fixture(method: Method) -> &'static ProveResult {
    static SHOUT: OnceLock<ProveResult> = OnceLock::new();
    static LOGUP: OnceLock<ProveResult> = OnceLock::new();
    match method {
        Method::Shout => SHOUT.get_or_init(|| prove(method, &config(4, 2), &trace(4, 4)).unwrap()),
        Method::LogupStar => {
            LOGUP.get_or_init(|| prove(method, &config(4, 2), &trace(4, 4)).unwrap())
        }
    }
}
fn rejects(base: &ProveResult, proof: &LookupProof) {
    let result = catch_unwind(AssertUnwindSafe(|| verify(&base.statement, proof)));
    assert!(result.is_ok(), "untrusted proof panicked");
    assert!(result.unwrap().is_err(), "tampered proof accepted");
}

#[test]
fn both_methods_prove_indexed_reads_with_full_width_duplicate_and_zero_values() {
    for (a, c, n) in [
        (2, 1, 0),
        (2, 2, 1),
        (4, 2, 4),
        (6, 2, 5),
        (8, 2, 8),
        (12, 4, 3),
    ] {
        let input = trace(a, n);
        for method in [Method::Shout, Method::LogupStar] {
            let result = prove(method, &config(a, c), &input).unwrap();
            verify(&result.statement, &result.proof).unwrap();
            assert_eq!(result.summary.core_bits, (a + 64) * (1 << n));
            assert_eq!(
                result.summary.auxiliary_bits,
                if method == Method::LogupStar {
                    128 * (1 << a)
                } else {
                    0
                }
            );
            assert_eq!(
                result.summary.argument_bits,
                if method == Method::Shout {
                    (a / c) * (1 << c) * (1 << n)
                } else {
                    0
                }
            );
            assert_eq!(
                result.proof.input.address_bits.len(),
                if method == Method::Shout { a } else { 0 }
            );
        }
    }
}

#[test]
fn all_public_parameters_and_input_claims_are_bound() {
    for method in [Method::Shout, Method::LogupStar] {
        let base = fixture(method);
        let mut statement = base.statement.clone();
        statement.table[3] ^= 1;
        assert!(verify(&statement, &base.proof).is_err());
        let mut statement = base.statement.clone();
        statement.config.log_rate = 3;
        assert!(verify(&statement, &base.proof).is_err());
        let mut statement = base.statement.clone();
        statement.method = if method == Method::Shout {
            Method::LogupStar
        } else {
            Method::Shout
        };
        assert!(verify(&statement, &base.proof).is_err());
        let mut proof = base.proof.clone();
        proof.input.read_value += K::ONE;
        rejects(base, &proof);
        for i in 0..proof.input.address_bits.len() {
            let mut proof = base.proof.clone();
            proof.input.address_bits[i] += K::ONE;
            rejects(base, &proof);
        }
        let mut proof = base.proof.clone();
        proof.commitment.root[0] ^= 1;
        rejects(base, &proof);
        let mut proof = base.proof.clone();
        proof.mosaic.v_g += K::ONE;
        rejects(base, &proof);
        let mut proof = base.proof.clone();
        proof.input.address_bits.push(K::ZERO);
        rejects(base, &proof);
    }
}

#[test]
fn all_rational_messages_and_auxiliary_openings_are_authenticated() {
    let base = fixture(Method::LogupStar);
    let ArgumentProof::LogupStar(original) = &base.proof.argument else {
        unreachable!()
    };
    for i in 0..4 {
        let mut proof = base.proof.clone();
        let ArgumentProof::LogupStar(p) = &mut proof.argument else {
            unreachable!()
        };
        p.rational.roots[i] += K::ONE;
        rejects(base, &proof);
    }
    for left in [true, false] {
        let tree = if left {
            &original.rational.left
        } else {
            &original.rational.right
        };
        for (level, layer) in tree.layers.iter().enumerate() {
            // The nonlinear split message, both child numerators, both
            // terminal product evaluations, and every sumcheck coefficient.
            let fields = 1 + 2 + 2 + layer.product.rounds.len() * 2;
            for field in 0..fields {
                let mut proof = base.proof.clone();
                let ArgumentProof::LogupStar(p) = &mut proof.argument else {
                    unreachable!()
                };
                let tree = if left {
                    &mut p.rational.left
                } else {
                    &mut p.rational.right
                };
                let layer = &mut tree.layers[level];
                match field {
                    0 => layer.ac_eval += K::ONE,
                    1..=2 => layer.numerators[field - 1] += K::ONE,
                    3..=4 => layer.product.evaluations[field - 3] += K::ONE,
                    _ => layer.product.rounds[(field - 5) / 2][(field - 5) % 2] += K::ONE,
                }
                rejects(base, &proof);
            }
        }
    }
    for i in 0..original.address_evals.len() {
        let mut proof = base.proof.clone();
        let ArgumentProof::LogupStar(p) = &mut proof.argument else {
            unreachable!()
        };
        p.address_evals[i] += K::ONE;
        rejects(base, &proof);
    }
    for round in 0..original.inner_product.rounds.len() {
        for field in 0..2 {
            let mut proof = base.proof.clone();
            let ArgumentProof::LogupStar(p) = &mut proof.argument else {
                unreachable!()
            };
            p.inner_product.rounds[round][field] += K::ONE;
            rejects(base, &proof);
        }
    }
    for what in 0..3 {
        let mut proof = base.proof.clone();
        let ArgumentProof::LogupStar(p) = &mut proof.argument else {
            unreachable!()
        };
        match what {
            0 => p.inner_product.value += K::ONE,
            1 => p.auxiliary.commitment.root[0] ^= 1,
            _ => p.auxiliary.mosaic.v_g += K::ONE,
        }
        rejects(base, &proof);
    }
}

#[test]
fn malformed_branches_shapes_and_zero_denominators_reject() {
    let shout = fixture(Method::Shout);
    let logup = fixture(Method::LogupStar);
    let mut mixed = shout.proof.clone();
    mixed.argument = logup.proof.argument.clone();
    rejects(shout, &mixed);
    for what in 0..7 {
        let mut proof = logup.proof.clone();
        let ArgumentProof::LogupStar(p) = &mut proof.argument else {
            unreachable!()
        };
        match what {
            0 => {
                p.address_evals.pop();
            }
            1 => {
                p.rational.left.layers.pop();
            }
            2 => {
                p.rational.right.layers[2].product.rounds.pop();
            }
            3 => p.rational.roots[1] = K::ZERO,
            4 => {
                p.inner_product.rounds.pop();
            }
            5 => {
                p.auxiliary.commitment.root.pop();
            }
            _ => p.auxiliary.commitment.n += 1,
        }
        rejects(logup, &proof);
    }
}

#[test]
fn shout_virtualization_choices_preserve_lookup_semantics() {
    for c in [1, 2, 8] {
        for virtualization in [
            VirtualizationMode::Flat,
            VirtualizationMode::FlatLegacy,
            VirtualizationMode::GkrTree,
        ] {
            let cfg = Config {
                virtualization,
                ..config(8, c)
            };
            let result = prove(Method::Shout, &cfg, &trace(8, 4)).unwrap();
            verify(&result.statement, &result.proof).unwrap();
            let mut other = result.statement.clone();
            other.config.virtualization = if virtualization == VirtualizationMode::Flat {
                VirtualizationMode::FlatLegacy
            } else {
                VirtualizationMode::Flat
            };
            assert!(verify(&other, &result.proof).is_err());
        }
    }
}

#[test]
fn invalid_trace_values_and_non_power_of_two_lengths_are_rejected() {
    for method in [Method::Shout, Method::LogupStar] {
        let mut input = trace(4, 4);
        input.read_values = Some(vec![17; 16]);
        assert!(prove(method, &config(4, 2), &input).is_err());
        let mut input = trace(4, 4);
        input.addresses[0] = 16;
        assert!(prove(method, &config(4, 2), &input).is_err());
        let mut input = trace(4, 4);
        input.addresses.pop();
        assert!(prove(method, &config(4, 2), &input).is_err());
    }
}

#[test]
fn main_and_auxiliary_pcs_openings_are_required_with_roots_and_iop_unchanged() {
    // Alter only PCS opening data: changing a root or an IOP message would
    // also change earlier challenges and would not isolate this check.
    for method in [Method::Shout, Method::LogupStar] {
        let base = fixture(method);
        let mut proof = base.proof.clone();
        proof.opening.0.s[0] += K::ONE;
        rejects(base, &proof);
        let mut proof = base.proof.clone();
        proof.opening.1.openings[0].columns[0][0] += K::ONE;
        rejects(base, &proof);
    }
    let base = fixture(Method::LogupStar);
    for ringswitch in [true, false] {
        let mut proof = base.proof.clone();
        let ArgumentProof::LogupStar(logup) = &mut proof.argument else {
            unreachable!()
        };
        if ringswitch {
            logup.auxiliary.opening.0.s[0] += K::ONE;
        } else {
            logup.auxiliary.opening.1.openings[0].columns[0][0] += K::ONE;
        }
        rejects(base, &proof);
    }
}
