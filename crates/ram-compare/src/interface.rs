//! The CPU stand-in: check the word-level claims a verifier returns at the
//! CPU point against the trace.
//!
//! A CPU proof would prove its own columns at `r_c` and compare them with
//! these claims. The harness has no CPU, so it recomputes the columns from
//! the trace after the verifier's timer stops. The check uses only
//! `eq_table` and `K::from_u128`, never the provers' helpers, so a bug in
//! prover code cannot hide a wrong claim. A mismatch fails the run.

use om_field::K;
use ram_argument::{AccessTrace, twist2};
use spice_argument::operations;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// `Σ_j eq(point, j) enc(x_j)`, with `enc(x) = K::from_u128(x)`.
fn word_mle(eq: &[K], xs: impl Iterator<Item = u64>) -> K {
    eq.iter()
        .zip(xs)
        .fold(K::ZERO, |acc, (&e, x)| acc + e * K::from_u128(x as u128))
}

fn eq_for(point: &[K], cycles: usize) -> Result<Vec<K>> {
    if cycles != 1usize << point.len() {
        return Err(format!(
            "interface point has {} coordinates for {cycles} cycles",
            point.len()
        )
        .into());
    }
    Ok(om_sumcheck::eq_table(point))
}

fn mismatch(method: &str, claim: &str) -> Box<dyn std::error::Error> {
    format!("{method}: the {claim} claim at the CPU point does not match the trace").into()
}

/// Twist v2: `address = W[addr]`, `read = W[rv]` and `write = W[wv]`, where
/// `addr_j = writes[j]`, `wv_j = new_values[j]` and `rv_j` is the cell's
/// word before cycle `j`, found by replaying the trace.
pub fn check_twist(trace: &AccessTrace, i: &twist2::Interface) -> Result<()> {
    let eq = eq_for(&i.point, trace.writes.len())?;
    let mut memory = trace.initial.clone();
    let mut read = Vec::with_capacity(trace.writes.len());
    for (&k, &v) in trace.writes.iter().zip(&trace.new_values) {
        let cell = memory
            .get_mut(k as usize)
            .ok_or("twist trace address out of range")?;
        read.push(*cell);
        *cell = v;
    }
    let claims = &i.claims;
    if claims.address != word_mle(&eq, trace.writes.iter().map(|&k| k as u64)) {
        return Err(mismatch("twist", "address"));
    }
    if claims.read != word_mle(&eq, read.into_iter()) {
        return Err(mismatch("twist", "read value"));
    }
    if claims.write != word_mle(&eq, trace.new_values.iter().copied()) {
        return Err(mismatch("twist", "write value"));
    }
    Ok(())
}

/// Typed Spice: `address = W[addresses]`, `value = W[values]` and
/// `is_write = Σ_j eq(r_c, j) is_write_j`.
pub fn check_spice(trace: &operations::AccessTrace, i: &operations::Interface) -> Result<()> {
    let eq = eq_for(&i.point, trace.addresses.len())?;
    if i.address != word_mle(&eq, trace.addresses.iter().map(|&k| k as u64)) {
        return Err(mismatch("spice-typed", "address"));
    }
    if i.value != word_mle(&eq, trace.values.iter().copied()) {
        return Err(mismatch("spice-typed", "value"));
    }
    if i.is_write != word_mle(&eq, trace.is_write.iter().map(|&w| u64::from(w))) {
        return Err(mismatch("spice-typed", "is_write"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ram_argument::twist2::{Config, prove, verify};

    fn twist_trace() -> AccessTrace {
        AccessTrace {
            initial: vec![5, 6, 7, 8],
            reads: vec![vec![0, 1, 1, 3, 2, 0, 3, 1]],
            writes: vec![0, 1, 1, 3, 2, 0, 3, 1],
            new_values: vec![5, 9, 9, 1, 7, 4, 2, 9],
            read_values: None,
        }
    }

    #[test]
    fn twist_claims_pass_and_perturbed_claims_fail() {
        let config = Config {
            address_bits: 2,
            chunk_bits: 1,
            ..Config::default()
        };
        let trace = twist_trace();
        let r = prove(&config, &trace).unwrap();
        let i = verify(&r.statement, &r.proof).unwrap();
        check_twist(&trace, &i).unwrap();
        for field in 0..4 {
            let mut bad = i.clone();
            match field {
                0 => bad.claims.address += K::ONE,
                1 => bad.claims.read += K::ONE,
                2 => bad.claims.write += K::ONE,
                _ => bad.point[0] += K::ONE,
            }
            assert!(check_twist(&trace, &bad).is_err(), "field {field}");
        }
        // The same claims against a trace with two cycles' addresses swapped.
        let mut other = trace.clone();
        other.writes.swap(1, 4);
        assert!(check_twist(&other, &i).is_err());
    }

    #[test]
    fn spice_claims_pass_and_perturbed_claims_fail() {
        let config = spice_argument::Config {
            address_bits: 2,
            chunk_bits: 1,
            ..spice_argument::Config::default()
        };
        let trace = operations::AccessTrace {
            initial: vec![7, 11, 19, 23],
            addresses: vec![0, 0, 0, 1, 1, 0, 0, 1],
            values: vec![9, 9, 9, 11, 13, 9, 0, 13],
            is_write: vec![true, false, true, false, true, false, true, false],
        };
        let r = operations::prove(&config, &trace).unwrap();
        let i = operations::verify(&r.statement, &r.proof).unwrap();
        check_spice(&trace, &i).unwrap();
        for field in 0..4 {
            let mut bad = i.clone();
            match field {
                0 => bad.address += K::ONE,
                1 => bad.value += K::ONE,
                2 => bad.is_write += K::ONE,
                _ => bad.point.pop().map(|_| ()).unwrap(),
            }
            assert!(check_spice(&trace, &bad).is_err(), "field {field}");
        }
    }
}
