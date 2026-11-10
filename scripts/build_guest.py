#!/usr/bin/env python3
"""Rebuild pinned upstream RV64I guests and their individual provenance manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GUEST = ROOT / "payloads" / "open-magnetar"
UPSTREAM = "febfc0e66f5809e034dd1afac78202bd4882de27"
PROGRAMS = {
    "binary-search": (8, "fixture.json"),
    "erc20-mixed-block": (11, "erc20-mixed-block.fixture.json"),
    "pointer-chase": (8, "pointer-chase.fixture.json"),
}


def find_lld():
    configured = os.environ.get("LD_LLD")
    if configured:
        return configured
    installed = shutil.which("ld.lld")
    if installed:
        return installed
    rustc = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    candidates = sorted(Path(rustc).glob("lib/rustlib/*/bin/gcc-ld/ld.lld"))
    if candidates:
        return str(candidates[0])
    raise SystemExit("ld.lld is required to rebuild; set LD_LLD to its path")


def build(program, clang, linker):
    # These constants are part of the embedded fixture's ABI in ram-payload.
    output_bits, manifest_name = PROGRAMS[program]
    address_bits, input_bits = 16, 12
    ram_base = 0x80000000
    in_word = (1 << address_bits) - (1 << (max(input_bits, output_bits) + 1))
    out_word = in_word + (1 << max(input_bits, output_bits))
    output = GUEST / f"{program}.elf"
    command = [
        clang, "--target=riscv64", "-march=rv64i", "-mabi=lp64",
        "-mcmodel=medany", "-mno-relax", "-O2", "-nostdlib", "-ffreestanding",
        "-fno-pic", "-fno-builtin", "-I.", f"--ld-path={linker}", "-static",
        "-Wl,-T,link.ld", "-Wl,-z,max-page-size=8",
        f"-Wl,--defsym=__stack_top={ram_base + 8 * in_word}",
        f"-Wl,--defsym=__input_base={ram_base + 8 * in_word}",
        f"-Wl,--defsym=__output_base={ram_base + 8 * out_word}",
        "-Wl,--defsym=__stack_size=65536",
        "rt/start.S", f"examples/{program}.c", "-o", output.name,
    ]
    subprocess.run(command, cwd=GUEST, check=True)
    sources = [f"examples/{program}.c", "rt/start.S", "rt/om.h", "link.ld"]
    manifest = {
        "upstream_repository": "https://github.com/Lulzx/open-magnetar",
        "upstream_commit": UPSTREAM,
        "guest_source": f"guest/examples/{program}.c",
        "source_sha256": {name: hashlib.sha256((GUEST / name).read_bytes()).hexdigest() for name in sources},
        "elf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "elf_bytes": output.stat().st_size,
        "ram_log_words": address_bits,
        "input_log_words": input_bits,
        "output_log_words": output_bits,
        "clang_version": subprocess.check_output([clang, "--version"], text=True).splitlines()[0],
        "linker_version": subprocess.check_output([linker, "--version"], text=True).strip(),
        "command": command,
    }
    (GUEST / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {output.relative_to(ROOT)} ({manifest['elf_bytes']} bytes)")
    print(f"SHA256 {manifest['elf_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", default=os.environ.get("CLANG", "clang"))
    parser.add_argument("--program", choices=[*PROGRAMS, "all"], default="binary-search")
    args = parser.parse_args()
    clang = shutil.which(args.clang) or args.clang
    linker = find_lld()
    for program in PROGRAMS if args.program == "all" else [args.program]:
        build(program, clang, linker)


if __name__ == "__main__":
    main()
