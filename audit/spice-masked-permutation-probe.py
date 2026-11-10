#!/usr/bin/env python3
"""Exhaustive small-domain audit of the Stage 1.5 masked tuple relation.

No proof implementation, hashes or field fingerprint collisions are modeled:
the probe checks exact tuple multisets with Boolean flags and values in {0,1}.
It enumerates all earlier predecessors and final pointers. Projecting tuples
onto (label,address) first only removes structurally impossible equalities.
Every accepted relation must match chronological typed RAM, and every valid
typed trace must admit exactly one assignment of successor flags.

Run: python3 audit/spice-masked-permutation-probe.py
"""

from collections import Counter
from itertools import product


def tuples(n, t, addresses, previous, last, values, writes, nexts,
           initial_next, initial, final, final_write=0):
    base = 1 << (max(n, t) - 1).bit_length()
    ids = [base + j for j in range(t)]
    push = [(k, k, initial[k] * (1 - initial_next[k]), initial_next[k])
            for k in range(n)]
    push += [(ids[j], addresses[j], values[j] * (1 - nexts[j]), nexts[j])
             for j in range(t)]
    pull = [(previous[j], addresses[j], values[j] * (1 - writes[j]), writes[j])
            for j in range(t)]
    pull += [(last[k], k, final[k] * (1 - final_write), final_write)
             for k in range(n)]
    return Counter(push) == Counter(pull)


def ram(initial, addresses, values, writes, final):
    memory = list(initial)
    for address, value, write in zip(addresses, values, writes):
        if write:
            memory[address] = value
        elif memory[address] != value:
            return False
    return memory == list(final)


def main():
    for n, t in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]:
        base = 1 << (max(n, t) - 1).bit_length()
        ids = [base + j for j in range(t)]
        domain = list(range(n)) + ids
        structural = checked = accepted = valid_traces = 0
        for addresses in product(range(n), repeat=t):
            pushed = Counter([(k, k) for k in range(n)] + list(zip(ids, addresses)))
            predecessors = [list(range(n)) + ids[:j] for j in range(t)]
            for previous in product(*predecessors):
                for last in product(domain, repeat=n):
                    pulled = Counter(list(zip(previous, addresses))
                                     + [(last[k], k) for k in range(n)])
                    if pushed != pulled:
                        continue
                    structural += 1
                    for values in product(range(2), repeat=t):
                        for writes in product(range(2), repeat=t):
                            for initial in product(range(2), repeat=n):
                                for final in product(range(2), repeat=n):
                                    correct = ram(initial, addresses, values, writes, final)
                                    count = 0
                                    for nexts in product(range(2), repeat=t):
                                        for initial_next in product(range(2), repeat=n):
                                            checked += 1
                                            if tuples(n, t, addresses, previous, last, values,
                                                      writes, nexts, initial_next, initial, final):
                                                assert correct, (n, t, addresses, previous, last,
                                                                 values, writes, nexts, initial_next,
                                                                 initial, final)
                                                accepted += 1
                                                count += 1
                                    assert count == int(correct), ("completeness", n, t, correct, count)
                                    valid_traces += int(correct)
        print(f"PASS N={n} T={t}: structural_chains={structural} "
              f"exact_flag/value_checks={checked} accepted={accepted} "
              f"valid_typed_traces={valid_traces}")

    # Without order, an all-write closed cycle disconnects from public memory.
    assert tuples(1, 2, [0, 0], [3, 2], [0], [1, 1], [1, 1], [1, 1], [0], [0], [0])
    assert not ram([0], [0, 0], [1, 1], [1, 1], [0])
    # A write-type final row masks an incorrect public final value.
    assert tuples(1, 1, [0], [0], [1], [1], [1], [1], [1], [0], [0], final_write=1)
    assert not ram([0], [0], [1], [1], [0])
    print("PASS counterexamples: missing order or a write-type final boundary "
          "permits an incorrect final state")


if __name__ == "__main__":
    main()
