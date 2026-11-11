#!/bin/bash
# The benchmarks behind results/m4max/final (Twist v2 with a virtual read
# value and the CPU interface claims). Run from the repository root:
#   BASE=/path/to/ram-compare-at-632855e PRE=/path/to/ram-compare-at-0e9aa9c \
#     BRANCH=/path/to/ram-compare OUT=results/x86/final results/m4max/final/run.sh
# BASE is ram-compare built at 632855e, PRE at 0e9aa9c (committed read value,
# before the CPU interface) and BRANCH at this branch, each with
# RUSTFLAGS='-C target-cpu=native' after `cargo clean --release`. If PREV is
# set to ram-compare built at e896b64 (before the dense bit scans in Mosaic
# and the ring switch), stage A3 also compares it with BRANCH.
set -u
: "${BASE:?set BASE to the baseline ram-compare binary}"
: "${PRE:?set PRE to the ram-compare binary built at 0e9aa9c}"
: "${BRANCH:?set BRANCH to the branch ram-compare binary}"
R=${OUT:-results/m4max/final}
mkdir -p $R
GUEST="--guest-program erc20-mixed-block,pointer-chase --guest-rows memory"
COMMON="--trace-format operations --address-bits 16 --log-cycles 20 --chunk-bits 4 --threads 8"
log() { echo "== $(date '+%H:%M:%S') $*"; uptime; }

log "A: baseline vs branch, interleaved binaries, 10 rounds"
scripts/interleave.py --baseline $BASE --branch $BRANCH --rounds 10 --warmup 1 --repeat 3 \
  --out-dir $R/interleave-n20 -- $COMMON --methods twist,spice-typed --patterns random,guest $GUEST \
  > $R/interleave-n20.txt 2>&1
log "A2: PRE vs branch, interleaved binaries, 10 rounds"
scripts/interleave.py --baseline $PRE --branch $BRANCH --rounds 10 --warmup 1 --repeat 3 \
  --out-dir $R/interleave-pre-n20 -- $COMMON --methods twist,spice-typed --patterns random,guest $GUEST \
  > $R/interleave-pre-n20.txt 2>&1
if [ -n "${PREV:-}" ]; then
  log "A3: e896b64 vs branch, interleaved binaries, 10 rounds"
  scripts/interleave.py --baseline $PREV --branch $BRANCH --rounds 10 --warmup 1 --repeat 3 \
    --out-dir $R/interleave-prev-n20 -- $COMMON --methods twist,spice-typed --patterns random,guest $GUEST \
    > $R/interleave-prev-n20.txt 2>&1
fi
log "B: branch methods in one process, 10 rounds"
$BRANCH $COMMON --methods twist,twist-legacy,spice-typed,spice-typed-2fp --patterns random,guest,locality $GUEST \
  --warmup 1 --repeat 10 --output $R/branch-n20.json --csv $R/branch-n20.csv > $R/branch-n20.txt 2>&1
log "C: private final memory, 10 rounds"
$BRANCH $COMMON --methods twist,spice-typed --patterns random,guest,locality $GUEST --final-memory private \
  --warmup 1 --repeat 10 --output $R/private-n20.json --csv $R/private-n20.csv > $R/private-n20.txt 2>&1
log "D: cycle-first read order, 10 rounds"
$BRANCH $COMMON --methods twist --patterns random,guest,locality $GUEST --read-order cycle \
  --warmup 1 --repeat 10 --output $R/cycle-order-n20.json --csv $R/cycle-order-n20.csv > $R/cycle-order-n20.txt 2>&1
log "E: chunk sweep"
$BRANCH --trace-format operations --address-bits 16 --log-cycles 20 --threads 8 --chunk-bits 2,4,8 \
  --methods twist,twist-legacy --patterns random,guest,locality $GUEST --warmup 1 --repeat 5 \
  --output $R/chunk-sweep-n20.json --csv $R/chunk-sweep-n20.csv > $R/chunk-sweep-n20.txt 2>&1
log "E2: chunk sweep at a = 20"
$BRANCH --trace-format operations --address-bits 20 --log-cycles 20 --threads 8 --chunk-bits 2,4 \
  --methods twist --patterns random,guest,locality $GUEST --warmup 1 --repeat 5 \
  --output $R/chunk-sweep-a20-n20.json --csv $R/chunk-sweep-a20-n20.csv > $R/chunk-sweep-a20-n20.txt 2>&1
log "F: (n, a) sweep"
scripts/sweep.py --binary $BRANCH --address-bits 12,16,20 --log-cycles 16,18,20 --threads 8 --out-dir $R/sweep-na -- \
  --trace-format operations --methods twist,spice-typed --patterns random,locality --warmup 1 --repeat 5 \
  > $R/sweep-na.txt 2>&1
log "G: thread sweep"
scripts/sweep.py --binary $BRANCH --address-bits 16 --log-cycles 20 --threads 1,8,16 --out-dir $R/sweep-threads -- \
  --trace-format operations --methods twist,spice-typed --patterns random,guest $GUEST --warmup 1 --repeat 5 \
  > $R/sweep-threads.txt 2>&1
log "done"
