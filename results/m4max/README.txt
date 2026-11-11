Benchmarks for issue #1 on an Apple M4 Max (12 performance and 4 efficiency
cores, macOS), shared with other jobs. Every binary was built with
RUSTFLAGS='-C target-cpu=native'. The baseline binary is ram-compare built at
632855e (upstream master).

final/
    All results for the pull request, from ram-compare built at ed1c284
    (environment.git_commit in each JSON). final/run.sh is the script that
    produced them. The one-minute load was 17 to 48 during this run, so the
    branch/baseline intervals in final/interleave-n20.txt are wide.

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
