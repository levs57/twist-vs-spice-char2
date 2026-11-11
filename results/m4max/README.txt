Benchmarks for issue #1 on an Apple M4 Max (12 performance and 4 efficiency
cores, macOS), shared with other jobs. Every binary was built with
RUSTFLAGS='-C target-cpu=native'. The baseline binary is ram-compare built at
632855e (upstream master).

final/
    The current results: Twist v2 with a virtual read value and the CPU
    interface claims in both methods, from ram-compare built at deb2bfb
    (environment.git_commit in each JSON; later commits change only docs and
    results). final/run.sh is the script that produced them:
    - interleave-n20: baseline (632855e) against the branch, 10 rounds;
    - interleave-pre-n20: PRE (0e9aa9c, the round-2 head with a committed
      read value and no interface claims) against the branch, 10 rounds.
      For twist it measures the virtual read value; for spice-typed it
      measures the cost of the interface claims alone;
    - branch-n20: all four methods in one process, 10 rounds;
    - private-n20, cycle-order-n20, chunk-sweep-n20, chunk-sweep-a20-n20,
      sweep-na and sweep-threads, as named in run.sh.
    Every sample of twist, spice-typed and spice-typed-2fp passed the
    harness check of the interface claims against the trace. The load before
    each run is in run.log and the .txt logs. All three binaries were built
    after `cargo clean --release` in one target directory.

final-committed-rv/
    The first-round results, when Twist v2 committed its read value, from
    ram-compare built at ed1c284. final-committed-rv/run.sh produced them.
    The one-minute load was 17 to 48 during this run, so the
    branch/baseline intervals in interleave-n20.txt are wide.

final-192c417/
    An earlier baseline/branch interleaved run (10 rounds, 3 samples per
    invocation) from ram-compare built at 192c417, at one-minute load 7 to 14.
    Between 192c417 and ed1c284 the prover code differs only in that typed
    Spice moves or copies the final memory into ProveResult::final_memory and
    Twist v2 marks an internal helper unsafe; the rest of the difference is
    harness labelling, provenance fields, tests and documentation.

round2/
    The second round: the Twist v2 cycle prover changes, from ram-compare
    built at a186534 (the BRANCH binary; environment.git_commit in each JSON).
    round2/run.sh is the script that produced them:
    - interleave-n20: baseline (632855e) against the branch, 10 rounds;
    - branch-n20: all four methods in one process, 10 rounds;
    - ablation-n20: Twist v2 only, 6 interleaved rounds of five binaries that
      add the cycle prover changes one at a time: first-round (ed1c284),
      split-eq (c7df2c5), split-lt (eec558c), lookups (acdf49a) and skip-inc
      (the branch binary; its prover code equals 1f73411).
    The load before each run is in the .txt logs and, for the ablation, in
    ablation-n20/summary.txt.
    Every binary was built after `cargo clean --release` in one target
    directory (see docs/methodology.md, "Building binaries to compare"). An
    earlier round-2 run used a binary whose ram-argument code came from a
    stale worktree build. Its results were discarded and are not here.
