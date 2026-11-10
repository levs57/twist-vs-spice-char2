//! Shared-Eq virtualization when the preceding Shout read already supplied one
//! virtual selector claim. The initial value is this prover's read-sumcheck
//! terminal evaluation; no additional claim-reduction protocol is needed.

use om_field::K;
use om_sumcheck::{VerificationError, eq_eval, sumcheck_verify};
use om_transcript::Transcript;

use crate::onehot::{AddrComps, EvalClaim, VirtGroup, VirtProof, split_point, virt_claims};
use crate::sublabel;
use crate::virt_selectors::{SelectorColumn, SelectorState};

fn validate(group: &VirtGroup, a: usize, d: usize, n: usize) -> Result<(), VerificationError> {
    if d <= 1
        || a < d
        || a > 32
        || !a.is_multiple_of(d)
        || n >= usize::BITS as usize
        || group.names.len() != d
        || group.r_k.len() != a
        || group.terms.len() != 1
        || group.terms[0].cyc.len() != n
    {
        return Err(VerificationError::new(
            "single-cycle virtualization: invalid shape",
        ));
    }
    Ok(())
}

fn bind_context(
    group: &VirtGroup,
    a: usize,
    d: usize,
    n: usize,
    tr: &mut Transcript,
    label: &[u8],
) {
    tr.absorb_bytes(&sublabel(label, b"/protocol"), b"flat/single-cycle/v1");
    let dimensions: Vec<u8> = [a, d, n]
        .into_iter()
        .flat_map(|v| (v as u64).to_le_bytes())
        .collect();
    tr.absorb_bytes(&sublabel(label, b"/dimensions"), &dimensions);
    for name in &group.names {
        tr.absorb_bytes(&sublabel(label, b"/name"), name.as_bytes());
    }
    tr.absorb_ks(&sublabel(label, b"/address"), &group.r_k);
    tr.absorb_ks(&sublabel(label, b"/cycle"), &group.terms[0].cyc);
    tr.absorb_k(&sublabel(label, b"/value"), group.terms[0].value);
}

/// Crate-private: Shout's preceding read proves the supplied initial value.
/// External arbitrary virtual claims must use the general claim-reduction API.
pub(crate) fn prove(
    group: &VirtGroup,
    component: &AddrComps,
    a: usize,
    d: usize,
    n: usize,
    tr: &mut Transcript,
    label: &[u8],
) -> (VirtProof, Vec<EvalClaim>) {
    validate(group, a, d, n).expect("single-cycle virtualization: group shape");
    assert!(crate::virt_tree::component_shape(component, a, d, n));
    bind_context(group, a, d, n, tr, label);
    let points = split_point(&group.r_k, a, d);
    let columns = points
        .iter()
        .enumerate()
        .map(|(chunk, point)| SelectorColumn {
            component,
            chunk,
            point,
        })
        .collect();
    let mut selectors = SelectorState::new(n, columns);
    let term = &group.terms[0];
    let (sc, cycle, evals) = crate::virt_product::prove(
        &term.cyc,
        &mut selectors,
        &[K::ONE],
        d,
        term.value,
        tr,
        &sublabel(label, b"/product"),
    );
    tr.absorb_ks(&sublabel(label, b"/evals"), &evals);
    let claims = virt_claims(std::slice::from_ref(group), d, &cycle, &evals, a);
    (VirtProof { sc, evals }, claims)
}

pub(crate) fn verify(
    group: &VirtGroup,
    a: usize,
    d: usize,
    n: usize,
    proof: &VirtProof,
    tr: &mut Transcript,
    label: &[u8],
) -> Result<Vec<EvalClaim>, VerificationError> {
    validate(group, a, d, n)?;
    if proof.evals.len() != d
        || proof.sc.round_msgs.len() != n
        || proof.sc.round_msgs.iter().any(|msg| msg.len() != d + 1)
    {
        return Err(VerificationError::new(
            "single-cycle virtualization: malformed proof",
        ));
    }
    bind_context(group, a, d, n, tr, label);
    let term = &group.terms[0];
    let (cycle, expected) = sumcheck_verify(
        term.value,
        n,
        d + 1,
        &proof.sc,
        tr,
        &sublabel(label, b"/product"),
    )?;
    let product = proof
        .evals
        .iter()
        .copied()
        .fold(K::ONE, |p, value| p * value);
    if eq_eval(&term.cyc, &cycle) * product != expected {
        return Err(VerificationError::new(
            "single-cycle virtualization: product check failed",
        ));
    }
    tr.absorb_ks(&sublabel(label, b"/evals"), &proof.evals);
    Ok(virt_claims(
        std::slice::from_ref(group),
        d,
        &cycle,
        &proof.evals,
        a,
    ))
}
