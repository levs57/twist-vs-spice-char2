Benchmarks for issue #1 on an Apple M4 Max (12 performance and 4 efficiency
cores, macOS), shared with other jobs.

The data (JSON, CSV and logs, about 800k lines) is not on this branch, so that
the pull request shows only code. It is on the branch like-for-like-results
of https://github.com/quangvdao/twist-vs-spice-char2, commit 04ceb20, under
results/m4max/, with a README.txt that describes each directory:

    final/               Twist v2 with a virtual read value and the CPU
                         interface claims, from ram-compare built at e896b64.
    final-committed-rv/  The first round, when Twist v2 committed its read
                         value, from ram-compare built at ed1c284.
    final-192c417/       An earlier first-round interleaved run.
    round2/              The second round (cycle prover changes and their
                         ablation), from ram-compare built at a186534.

The run.sh script in each directory produced that directory's results. Run it
from the repository root with BASE and BRANCH set, plus PRE for final and L0
to L3 for round2, and OUT set to a new directory. The header of each
script says which commit each binary must be built at.
