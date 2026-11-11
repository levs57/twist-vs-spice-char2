#!/bin/bash
# The second-round benchmarks behind results/m4max/round2. Run from the
# repository root, with every binary built with RUSTFLAGS='-C target-cpu=native':
#   BASE=<ram-compare at 632855e> BRANCH=<ram-compare at this branch> \
#   L0=<at ed1c284> L1=<at c7df2c5> L2=<at eec558c> L3=<at acdf49a> \
#   OUT=results/x86/round2 results/m4max/round2/run.sh
# L0 is the first-round branch; L1..L3 add the cycle prover changes one at a
# time, and BRANCH adds the last one (Inc skipping, 1f73411).
set -u
: "${BASE:?set BASE}" "${BRANCH:?set BRANCH}" "${L0:?set L0}" "${L1:?set L1}" "${L2:?set L2}" "${L3:?set L3}"
R=${OUT:-results/m4max/round2}
mkdir -p $R
GUEST="--guest-program erc20-mixed-block,pointer-chase --guest-rows memory"
COMMON="--trace-format operations --address-bits 16 --log-cycles 20 --chunk-bits 4 --threads 8"
log() { echo "== $(date '+%H:%M:%S') $*"; uptime; }

log "A: baseline vs branch, interleaved binaries, 10 rounds"
scripts/interleave.py --baseline $BASE --branch $BRANCH --rounds 10 --warmup 1 --repeat 3 \
  --out-dir $R/interleave-n20 -- $COMMON --methods twist,spice-typed --patterns random,guest $GUEST \
  > $R/interleave-n20.txt 2>&1
log "B: branch methods in one process, 10 rounds"
$BRANCH $COMMON --methods twist,twist-legacy,spice-typed,spice-typed-2fp --patterns random,guest,locality $GUEST \
  --warmup 1 --repeat 10 --output $R/branch-n20.json --csv $R/branch-n20.csv > $R/branch-n20.txt 2>&1
log "C: cycle prover ablation, interleaved binaries, 6 rounds"
scripts/ablation.py --rounds 6 --out-dir $R/ablation-n20 \
  first-round=$L0 split-eq=$L1 split-lt=$L2 lookups=$L3 skip-inc=$BRANCH -- \
  $COMMON --methods twist --patterns random,guest,locality $GUEST --warmup 1 --repeat 3 \
  > $R/ablation-n20.txt 2>&1
log "done"
