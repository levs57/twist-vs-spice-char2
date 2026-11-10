//! Exact cycle folding of address selectors, with compact early-round state.

use om_field::{K, mul_pairs};
use om_sumcheck::eq_table;
use rayon::prelude::*;

use crate::onehot::{AddrComps, PAR_THRESHOLD};

pub(crate) struct SelectorColumn<'a> {
    pub component: &'a AddrComps,
    pub chunk: usize,
    pub point: &'a [K],
}

#[derive(Clone, Copy)]
struct Policy {
    /// Bind four normally switches to T/16 dense rows.
    max_width: usize,
    branch_bytes: usize,
    paired_bytes: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            max_width: 8,
            branch_bytes: 1 << 20,
            paired_bytes: 1 << 20,
        }
    }
}

struct CompactColumn<'a> {
    addresses: &'a [u32],
    shift: usize,
    mask: u32,
    radix: usize,
}
impl CompactColumn<'_> {
    #[inline]
    fn index(&self, cycle: usize) -> usize {
        ((self.addresses[cycle] >> self.shift) & self.mask) as usize
    }
}

struct RawColumn<'a> {
    component: &'a AddrComps,
    chunk: usize,
    eq: Vec<K>,
}

struct Lazy<'a> {
    columns: Vec<CompactColumn<'a>>,
    /// Branch-major: tables[c][branch * radix_c + digit].
    tables: Vec<Vec<K>>,
    width: usize,
    /// Adjacent branch sums. Compact Addrs always select a real digit,
    /// including digit zero, so these tables have no artificial cold state.
    paired: Option<Vec<Vec<K>>>,
}
impl Lazy<'_> {
    fn fill(&self, row: usize, output: &mut [K]) {
        let start = row * self.width;
        if let Some(paired) = &self.paired {
            for ((column, table), value) in self.columns.iter().zip(paired).zip(output) {
                let square = column.radix * column.radix;
                let mut sum = K::ZERO;
                for pair in 0..self.width / 2 {
                    let lo = column.index(start + 2 * pair);
                    let hi = column.index(start + 2 * pair + 1);
                    sum += table[pair * square + lo * column.radix + hi];
                }
                *value = sum;
            }
        } else {
            for ((column, table), value) in self.columns.iter().zip(&self.tables).zip(output) {
                let mut sum = K::ZERO;
                for branch in 0..self.width {
                    sum += table[branch * column.radix + column.index(start + branch)];
                }
                *value = sum;
            }
        }
    }

    fn double(&mut self, r: K) {
        let scale = |table: &Vec<K>| {
            let mut next = vec![K::ZERO; 2 * table.len()];
            let (low, high) = next.split_at_mut(table.len());
            let mut input = table.chunks_exact(2);
            let mut lo = low.chunks_exact_mut(2);
            let mut hi = high.chunks_exact_mut(2);
            for ((input, lo), hi) in input.by_ref().zip(lo.by_ref()).zip(hi.by_ref()) {
                let scaled = mul_pairs([r; 2], [input[0], input[1]]);
                lo[0] = input[0] + scaled[0];
                lo[1] = input[1] + scaled[1];
                hi.copy_from_slice(&scaled);
            }
            if let Some(&input) = input.remainder().first() {
                let scaled = r * input;
                lo.into_remainder()[0] = input + scaled;
                hi.into_remainder()[0] = scaled;
            }
            next
        };
        self.tables = if self.tables.iter().map(Vec::len).sum::<usize>() >= PAR_THRESHOLD {
            self.tables.par_iter().map(scale).collect()
        } else {
            self.tables.iter().map(scale).collect()
        };
        self.width *= 2;
        self.paired = None;
    }

    fn cache_pairs(&mut self, len: usize, budget: usize) {
        const MIN_ROWS_PER_VALUE: usize = 16;
        if !matches!(self.width, 4 | 8)
            || self.columns.is_empty()
            || self.columns.iter().any(|column| column.radix > 16)
        {
            return;
        }
        let entries = |column: &CompactColumn<'_>| self.width / 2 * column.radix * column.radix;
        let total = self.columns.iter().map(entries).sum::<usize>();
        let largest = self.columns.iter().map(entries).max().unwrap_or(0);
        if total.saturating_mul(size_of::<K>()) > budget
            || len < largest.saturating_mul(MIN_ROWS_PER_VALUE)
        {
            return;
        }
        let build = |(column, table): (&CompactColumn<'_>, &Vec<K>)| {
            let mut output = Vec::with_capacity(entries(column));
            for pair in 0..self.width / 2 {
                let lo = &table[2 * pair * column.radix..(2 * pair + 1) * column.radix];
                let hi = &table[(2 * pair + 1) * column.radix..(2 * pair + 2) * column.radix];
                for &a in lo {
                    for &b in hi {
                        output.push(a + b);
                    }
                }
            }
            output
        };
        self.paired = Some(if total >= PAR_THRESHOLD {
            self.columns
                .par_iter()
                .zip(&self.tables)
                .map(build)
                .collect()
        } else {
            self.columns.iter().zip(&self.tables).map(build).collect()
        });
    }
}

enum State<'a> {
    Lazy(Lazy<'a>),
    /// Arbitrary Bits selectors are evaluated exactly, then become dense on
    /// their first bind. No one-hot identity is assumed on this path.
    Raw(Vec<RawColumn<'a>>),
    /// Row-major storage makes every round's product inputs contiguous.
    Dense(Vec<K>),
}
impl State<'_> {
    fn fill(&self, row: usize, columns: usize, output: &mut [K]) {
        match self {
            Self::Lazy(lazy) => lazy.fill(row, output),
            Self::Raw(raw) => {
                for (column, output) in raw.iter().zip(output) {
                    *output = column.component.row_eval(column.chunk, row, &column.eq);
                }
            }
            Self::Dense(values) => {
                output.copy_from_slice(&values[row * columns..(row + 1) * columns])
            }
        }
    }
}

fn fill_row(
    state: &State<'_>,
    row: usize,
    columns: usize,
    prefix: usize,
    tail: Option<&[K]>,
    output: &mut [K],
) {
    if row >= prefix {
        if let Some(tail) = tail {
            output.copy_from_slice(tail);
            return;
        }
    }
    state.fill(row, columns, output);
}

fn fold_row(left: &[K], right: &[K], r: K, output: &mut [K]) {
    let mut dst = output.chunks_exact_mut(2);
    let mut lo = left.chunks_exact(2);
    let mut hi = right.chunks_exact(2);
    for ((dst, lo), hi) in dst.by_ref().zip(lo.by_ref()).zip(hi.by_ref()) {
        let products = mul_pairs([r; 2], [lo[0] + hi[0], lo[1] + hi[1]]);
        dst[0] = lo[0] + products[0];
        dst[1] = lo[1] + products[1];
    }
    if let Some(dst) = dst.into_remainder().first_mut() {
        *dst = lo.remainder()[0] + r * (lo.remainder()[0] + hi.remainder()[0]);
    }
}

pub(crate) struct SelectorState<'a> {
    state: State<'a>,
    len: usize,
    columns: usize,
    prefix: usize,
    /// Actual hot selector values on a proven constant-address suffix.
    /// None for arbitrary Bits, whose entire domain stays explicit.
    tail: Option<Vec<K>>,
    scratch: Vec<K>,
    policy: Policy,
}
impl<'a> SelectorState<'a> {
    pub(crate) fn new(n: usize, columns: Vec<SelectorColumn<'a>>) -> Self {
        Self::with_policy(n, columns, Policy::default())
    }

    fn with_policy(n: usize, columns: Vec<SelectorColumn<'a>>, policy: Policy) -> Self {
        assert!(
            n < usize::BITS as usize,
            "selector cycle dimension too large"
        );
        let len = 1usize << n;
        for column in &columns {
            assert_eq!(
                column.component.cycles(),
                len,
                "selector source length mismatch"
            );
            assert_eq!(
                column.point.len(),
                column.component.ac(),
                "selector point length mismatch"
            );
            assert!(
                column.chunk < column.component.d(),
                "selector chunk out of range"
            );
        }
        let count = columns.len();
        let compact = columns
            .iter()
            .all(|column| matches!(column.component, AddrComps::Addrs { .. }));
        let (state, prefix, tail) = if compact {
            let mut sources = Vec::with_capacity(count);
            let mut tables = Vec::with_capacity(count);
            let mut seen = std::collections::BTreeSet::new();
            let mut prefix = 0;
            for column in columns {
                let AddrComps::Addrs { addrs, .. } = column.component else {
                    unreachable!()
                };
                // Scan each shared full address stream once. A constant zero
                // address remains hot; it is never treated as a cold selector.
                if seen.insert((addrs.as_ptr() as usize, addrs.len())) {
                    let last = addrs[len - 1];
                    prefix =
                        prefix.max(addrs.iter().rposition(|&a| a != last).map_or(0, |i| i + 1));
                }
                let bits = column.point.len();
                sources.push(CompactColumn {
                    addresses: addrs,
                    shift: bits * column.chunk,
                    mask: ((1u64 << bits) - 1) as u32,
                    radix: 1usize << bits,
                });
                tables.push(eq_table(column.point));
            }
            let tail: Vec<K> = sources
                .iter()
                .zip(&tables)
                .map(|(column, table)| table[column.index(len - 1)])
                .collect();
            let state = if prefix == 0 {
                State::Dense(Vec::new())
            } else {
                State::Lazy(Lazy {
                    columns: sources,
                    tables,
                    width: 1,
                    paired: None,
                })
            };
            (state, prefix, Some(tail))
        } else {
            (
                State::Raw(
                    columns
                        .into_iter()
                        .map(|column| RawColumn {
                            component: column.component,
                            chunk: column.chunk,
                            eq: eq_table(column.point),
                        })
                        .collect(),
                ),
                len,
                None,
            )
        };
        Self {
            state,
            len,
            columns: count,
            prefix,
            tail,
            scratch: Vec::new(),
            policy,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn columns(&self) -> usize {
        self.columns
    }
    pub(crate) fn prefix_len(&self) -> usize {
        self.prefix
    }
    pub(crate) fn tail_values(&self) -> Option<&[K]> {
        self.tail.as_deref()
    }

    pub(crate) fn pairs(&self, pair: usize, left: &mut [K], right: &mut [K]) {
        assert!(pair < self.len / 2, "selector pair out of range");
        assert_eq!(left.len(), self.columns);
        assert_eq!(right.len(), self.columns);
        fill_row(
            &self.state,
            2 * pair,
            self.columns,
            self.prefix,
            self.tail_values(),
            left,
        );
        fill_row(
            &self.state,
            2 * pair + 1,
            self.columns,
            self.prefix,
            self.tail_values(),
            right,
        );
    }

    pub(crate) fn final_values(&self) -> Vec<K> {
        assert_eq!(self.len, 1, "selector state is not fully bound");
        let mut values = vec![K::ZERO; self.columns];
        fill_row(
            &self.state,
            0,
            self.columns,
            self.prefix,
            self.tail_values(),
            &mut values,
        );
        values
    }

    /// Visit only pairs intersecting the explicit prefix, including its mixed
    /// boundary pair. The caller accounts for omitted constant pairs with Eq
    /// suffix mass. Each accumulator sees increasing global pair indices.
    pub(crate) fn visit_pairs<A, I, F, M>(&self, init: I, visit: F, merge: M) -> A
    where
        A: Send,
        I: Fn() -> A + Sync,
        F: Fn(&mut A, usize, &[K], &[K]) + Sync,
        M: Fn(A, A) -> A + Sync,
    {
        const PAIRS: usize = 256;
        let count = self.prefix.div_ceil(2).min(self.len / 2);
        if count == 0 {
            return init();
        }
        if let State::Dense(values) = &self.state {
            if self.columns != 0 {
                let process = |mut a: A, (chunk, rows): (usize, &[K])| {
                    for (pair, rows) in rows.chunks(2 * self.columns).enumerate() {
                        let (left, right) = if rows.len() == 2 * self.columns {
                            rows.split_at(self.columns)
                        } else {
                            (rows, self.tail_values().expect("constant boundary tail"))
                        };
                        visit(&mut a, chunk * PAIRS + pair, left, right);
                    }
                    a
                };
                return if values.len() >= PAR_THRESHOLD {
                    values
                        .par_chunks(2 * self.columns * PAIRS)
                        .enumerate()
                        .fold(&init, process)
                        .reduce(&init, &merge)
                } else {
                    values
                        .chunks(2 * self.columns * PAIRS)
                        .enumerate()
                        .fold(init(), process)
                };
            }
        }
        let fresh = || {
            (
                init(),
                vec![K::ZERO; self.columns],
                vec![K::ZERO; self.columns],
            )
        };
        let process = |(mut a, mut left, mut right): (A, Vec<K>, Vec<K>), chunk: usize| {
            for pair in chunk * PAIRS..((chunk + 1) * PAIRS).min(count) {
                self.pairs(pair, &mut left, &mut right);
                visit(&mut a, pair, &left, &right);
            }
            (a, left, right)
        };
        if count.saturating_mul(self.columns.max(1)) >= PAR_THRESHOLD {
            (0..count.div_ceil(PAIRS))
                .into_par_iter()
                .fold(fresh, process)
                .map(|(a, _, _)| a)
                .reduce(&init, &merge)
        } else {
            (0..count.div_ceil(PAIRS)).fold(fresh(), process).0
        }
    }

    pub(crate) fn bind(&mut self, r: K) {
        self.bind_impl::<false, _, _, _, _>(r, || (), |_, _, _, _| {}, |(), ()| ());
    }

    /// Bind and consume the next message's prefix pairs while the output tile
    /// remains hot. Prefix/tail accessors describe the new state on return.
    pub(crate) fn bind_and_visit_pairs<A, I, F, M>(
        &mut self,
        r: K,
        init: I,
        visit: F,
        merge: M,
    ) -> A
    where
        A: Send,
        I: Fn() -> A + Sync,
        F: Fn(&mut A, usize, &[K], &[K]) + Sync,
        M: Fn(A, A) -> A + Sync,
    {
        self.bind_impl::<true, _, _, _, _>(r, init, visit, merge)
    }

    fn bind_impl<const VISIT: bool, A, I, F, M>(&mut self, r: K, init: I, visit: F, merge: M) -> A
    where
        A: Send,
        I: Fn() -> A + Sync,
        F: Fn(&mut A, usize, &[K], &[K]) + Sync,
        M: Fn(A, A) -> A + Sync,
    {
        assert!(self.len >= 2, "cannot bind a fully bound selector");
        let next_len = self.len / 2;
        let next_prefix = self.prefix.div_ceil(2);
        let mut fold_source = true;
        if let State::Lazy(lazy) = &mut self.state {
            let branch_entries = lazy
                .tables
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                .saturating_mul(2);
            let dense_entries = next_prefix.saturating_mul(self.columns);
            let crossover = branch_entries >= dense_entries
                || branch_entries.saturating_mul(size_of::<K>()) > self.policy.branch_bytes;
            if !crossover {
                let stay_lazy = lazy.width < self.policy.max_width;
                lazy.double(r);
                if stay_lazy {
                    lazy.cache_pairs(next_prefix, self.policy.paired_bytes);
                    self.len = next_len;
                    self.prefix = next_prefix;
                    return if VISIT {
                        self.visit_pairs(init, visit, merge)
                    } else {
                        init()
                    };
                }
                fold_source = false;
            }
        }
        let fields = next_prefix.saturating_mul(self.columns);
        let mut output = std::mem::take(&mut self.scratch);
        if output.len() < fields && fields >= PAR_THRESHOLD {
            (0..fields)
                .into_par_iter()
                .map(|_| K::ZERO)
                .collect_into_vec(&mut output);
        } else {
            output.resize(fields, K::ZERO);
        }
        let state = &self.state;
        let columns = self.columns;
        let tail = self.tail.as_deref();
        let source_prefix = if fold_source {
            self.prefix
        } else {
            next_prefix
        };
        let fill_one = |row: usize, dst: &mut [K], left: &mut [K], right: &mut [K]| {
            if !fold_source {
                fill_row(state, row, columns, source_prefix, tail, dst);
            } else if let State::Dense(values) = state {
                let slice = |row: usize| {
                    if row < source_prefix {
                        &values[row * columns..(row + 1) * columns]
                    } else {
                        tail.expect("constant input tail")
                    }
                };
                fold_row(slice(2 * row), slice(2 * row + 1), r, dst);
            } else {
                fill_row(state, 2 * row, columns, source_prefix, tail, left);
                fill_row(state, 2 * row + 1, columns, source_prefix, tail, right);
                fold_row(left, right, r, dst);
            }
        };
        let result = if columns == 0 || next_prefix == 0 {
            init()
        } else if next_len == 1 {
            let (mut left, mut right) = (vec![K::ZERO; columns], vec![K::ZERO; columns]);
            fill_one(0, &mut output, &mut left, &mut right);
            init()
        } else {
            const PAIRS: usize = 256;
            let fresh = || (init(), vec![K::ZERO; columns], vec![K::ZERO; columns]);
            let process = |(mut a, mut left, mut right): (A, Vec<K>, Vec<K>),
                           (chunk, output): (usize, &mut [K])| {
                for (pair, rows) in output.chunks_mut(2 * columns).enumerate() {
                    let global = chunk * PAIRS + pair;
                    if rows.len() == 2 * columns {
                        let (lo, hi) = rows.split_at_mut(columns);
                        fill_one(2 * global, lo, &mut left, &mut right);
                        fill_one(2 * global + 1, hi, &mut left, &mut right);
                        if VISIT {
                            visit(&mut a, global, lo, hi);
                        }
                    } else {
                        fill_one(2 * global, rows, &mut left, &mut right);
                        if VISIT {
                            visit(&mut a, global, rows, tail.expect("constant boundary tail"));
                        }
                    }
                }
                (a, left, right)
            };
            if fields >= PAR_THRESHOLD {
                output
                    .par_chunks_mut(2 * columns * PAIRS)
                    .enumerate()
                    .fold(fresh, process)
                    .map(|(a, _, _)| a)
                    .reduce(&init, &merge)
            } else {
                output
                    .chunks_mut(2 * columns * PAIRS)
                    .enumerate()
                    .fold(fresh(), process)
                    .0
            }
        };
        let old = std::mem::replace(&mut self.state, State::Dense(output));
        if let State::Dense(values) = old {
            self.scratch = values;
        }
        self.len = next_len;
        self.prefix = next_prefix;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use om_sumcheck::bind_low;

    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn field(state: &mut u64) -> K {
        K::from_u128(next(state) as u128 | ((next(state) as u128) << 64))
    }
    fn dense(columns: &[SelectorColumn<'_>]) -> Vec<Vec<K>> {
        columns
            .iter()
            .map(|column| {
                let eq = eq_table(column.point);
                let bits = column.component.table(column.chunk);
                (0..column.component.cycles())
                    .map(|j| {
                        eq.iter().enumerate().fold(K::ZERO, |sum, (u, &e)| {
                            if bits[j * eq.len() + u] { sum + e } else { sum }
                        })
                    })
                    .collect()
            })
            .collect()
    }
    #[derive(Default)]
    struct Seen {
        last: Option<usize>,
        indices: Vec<usize>,
    }
    fn check_visit(seen: &mut Seen, pair: usize, lo: &[K], hi: &[K], expected: &[Vec<K>]) {
        assert!(
            seen.last.is_none_or(|last| last < pair),
            "nonascending worker indices"
        );
        seen.last = Some(pair);
        seen.indices.push(pair);
        for (i, column) in expected.iter().enumerate() {
            assert_eq!(lo[i], column[2 * pair]);
            assert_eq!(hi[i], column[2 * pair + 1]);
        }
    }
    fn merge(mut a: Seen, b: Seen) -> Seen {
        a.last = None;
        a.indices.extend(b.indices);
        a
    }
    fn coverage(mut seen: Seen, state: &SelectorState<'_>) {
        seen.indices.sort_unstable();
        assert_eq!(
            seen.indices,
            (0..state.prefix_len().div_ceil(2).min(state.len() / 2)).collect::<Vec<_>>()
        );
    }
    fn check_rows(state: &SelectorState<'_>, expected: &[Vec<K>]) {
        let (mut lo, mut hi) = (
            vec![K::ZERO; state.columns()],
            vec![K::ZERO; state.columns()],
        );
        for pair in 0..state.len() / 2 {
            state.pairs(pair, &mut lo, &mut hi);
            for (i, column) in expected.iter().enumerate() {
                assert_eq!(lo[i], column[2 * pair]);
                assert_eq!(hi[i], column[2 * pair + 1]);
            }
        }
        if let Some(tail) = state.tail_values() {
            for (i, column) in expected.iter().enumerate() {
                for &value in &column[state.prefix_len()..] {
                    assert_eq!(value, tail[i]);
                }
            }
        }
        coverage(
            state.visit_pairs(
                Seen::default,
                |s, p, lo, hi| check_visit(s, p, lo, hi, expected),
                merge,
            ),
            state,
        );
    }
    fn run(n: usize, sources: &[AddrComps], points: &[Vec<K>], policy: Policy, seed: u64) {
        let make = || {
            sources
                .iter()
                .enumerate()
                .flat_map(|(group, source)| {
                    (0..source.d()).map(move |chunk| SelectorColumn {
                        component: source,
                        chunk,
                        point: &points[group * source.d() + chunk],
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut expected = dense(&make());
        let mut state = SelectorState::with_policy(n, make(), policy);
        let mut random = seed;
        check_rows(&state, &expected);
        for round in 0..n {
            let r = match round % 4 {
                0 => K::ZERO,
                1 => K::ONE,
                2 => K::X,
                _ => field(&mut random),
            };
            expected = expected.iter().map(|column| bind_low(column, r)).collect();
            if round % 2 == 0 {
                let seen = state.bind_and_visit_pairs(
                    r,
                    Seen::default,
                    |s, p, lo, hi| check_visit(s, p, lo, hi, &expected),
                    merge,
                );
                coverage(seen, &state);
            } else {
                state.bind(r);
            }
            assert_eq!(state.len(), 1usize << (n - round - 1));
            check_rows(&state, &expected);
        }
        assert_eq!(
            state.final_values(),
            expected.iter().map(|column| column[0]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn lazy_dense_and_constant_tail_match_independent_bit_grid_folding() {
        let mut random = 0x7319;
        for n in [0usize, 1, 2, 5, 10] {
            let len = 1usize << n;
            for prefix in [
                0,
                1.min(len - 1),
                3.min(len - 1),
                len.saturating_sub(5),
                len - 1,
            ] {
                let sources: Vec<AddrComps> = (0..2)
                    .map(|group| {
                        let tail = if group == 0 { 0 } else { 0xa7 };
                        let addresses = (0..len)
                            .map(|j| {
                                if j < prefix {
                                    // Ensure the declared boundary is genuinely nonconstant.
                                    (next(&mut random) as u32 & 255)
                                        ^ if j + 1 == prefix { tail ^ 0x100 } else { 0 }
                                } else {
                                    tail
                                }
                            })
                            .map(|a| a & 255)
                            .collect();
                        AddrComps::from_addrs(addresses, 8, 2)
                    })
                    .collect();
                let points: Vec<Vec<K>> = (0..4)
                    .map(|i| {
                        (0..4)
                            .map(|_| match i {
                                0 => K::ZERO,
                                1 => K::ONE,
                                _ => field(&mut random),
                            })
                            .collect()
                    })
                    .collect();
                run(n, &sources, &points, Policy::default(), random);
                run(
                    n,
                    &sources,
                    &points,
                    Policy {
                        max_width: 1,
                        ..Policy::default()
                    },
                    random,
                );
                run(
                    n,
                    &sources,
                    &points,
                    Policy {
                        branch_bytes: 0,
                        ..Policy::default()
                    },
                    random,
                );
            }
        }
    }

    #[test]
    fn arbitrary_and_mixed_bits_keep_exact_dense_fallback() {
        let mut random = 0x1497;
        for n in [0usize, 1, 2, 6] {
            let len = 1usize << n;
            for kind in 0..3 {
                let tables = (0..2)
                    .map(|_| {
                        (0..16 * len)
                            .map(|i| match kind {
                                0 => false,
                                1 => i % 16 < 2,
                                _ => next(&mut random) & 1 != 0,
                            })
                            .collect()
                    })
                    .collect();
                let bits = AddrComps::Bits { tables, ac: 4 };
                let compact = AddrComps::from_addrs(
                    (0..len).map(|_| (next(&mut random) & 255) as u32).collect(),
                    8,
                    2,
                );
                let sources = [bits, compact];
                let points = (0..4)
                    .map(|_| (0..4).map(|_| field(&mut random)).collect())
                    .collect::<Vec<Vec<K>>>();
                run(n, &sources, &points, Policy::default(), random);
                let state = SelectorState::new(
                    n,
                    vec![SelectorColumn {
                        component: &sources[0],
                        chunk: 0,
                        point: &points[0],
                    }],
                );
                assert!(state.tail_values().is_none());
                assert_eq!(state.prefix_len(), len);
            }
        }
    }

    #[test]
    fn fourth_bind_transition_and_paired_caches_preserve_parallel_rows() {
        let n = 18;
        let len = 1usize << n;
        let mut random = 0xace3;
        let source = AddrComps::from_addrs(
            (0..len)
                .map(|j| {
                    if j + 13 < len {
                        next(&mut random) as u32 & 0xffff
                    } else {
                        0
                    }
                })
                .collect(),
            16,
            4,
        );
        let points = (0..4)
            .map(|_| (0..4).map(|_| field(&mut random)).collect())
            .collect::<Vec<Vec<K>>>();
        for workers in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    run(
                        n,
                        std::slice::from_ref(&source),
                        &points,
                        Policy::default(),
                        123,
                    );
                });
        }
        let mut state = SelectorState::new(
            n,
            (0..4)
                .map(|chunk| SelectorColumn {
                    component: &source,
                    chunk,
                    point: &points[chunk],
                })
                .collect(),
        );
        for round in 0..4 {
            state.bind(K::from_u128(19 + round as u128));
            if round < 3 {
                let State::Lazy(lazy) = &state.state else {
                    panic!("materialized before bind four")
                };
                assert_eq!(lazy.width, 1 << (round + 1));
                if round > 0 {
                    assert!(lazy.paired.is_some());
                }
            } else {
                let State::Dense(values) = &state.state else {
                    panic!("missing bind-four transition")
                };
                assert_eq!(values.len(), state.prefix_len() * 4);
                assert_eq!(state.len(), len / 16);
            }
        }
    }

    #[test]
    fn empty_columns_preserve_logical_lengths_without_selector_storage() {
        let mut state = SelectorState::new(5, Vec::new());
        assert_eq!(state.prefix_len(), 0);
        for _ in 0..5 {
            let count = state.bind_and_visit_pairs(
                K::X,
                || 0,
                |count, _, lo, hi| {
                    assert!(lo.is_empty() && hi.is_empty());
                    *count += 1;
                },
                |a, b| a + b,
            );
            assert_eq!(count, 0);
        }
        assert!(state.final_values().is_empty());
    }
}
