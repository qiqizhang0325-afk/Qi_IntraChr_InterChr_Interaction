"""Step 1: Detect PLINK and convert VCF to PLINK binary (BED/BIM/FAM).

This script:
- Detects a valid PLINK executable (prefers plink2, falls back to plink 1.9)
- Avoids PuTTY's SSH `plink.exe`
- Converts a VCF to PLINK binary format (.bed/.bim/.fam)

Usage examples:

  python src/step1_plink.py \
      --vcf data/test.vcf \
      --out-prefix results/plink/test

Optional override PLINK path:

  python src/step1_plink.py --vcf data/test.vcf \
      --out-prefix results/plink/test \
      --plink-path "C:/tools/plink/plink.exe"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple


def is_valid_plink(binary_path: str) -> Tuple[bool, Optional[str]]:
    """Return (is_valid, flavor) where flavor is 'plink2' or 'plink1'.

    Reject PuTTY's plink (SSH client). Detect genetics PLINK by version/help text.
    """
    try:
        # Try a benign flag that prints version/help and exits.
        proc = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False, None

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    text_lower = combined.lower()

    # Reject PuTTY's plink (SSH) which prints PuTTY related text.
    if "putty" in text_lower:
        return False, None

    # Detect PLINK 2 first.
    if "plink v2." in combined.lower() or "plink2" in combined.lower():
        return True, "plink2"

    # Detect PLINK 1.x.
    if "plink v1." in combined.lower() or "www.cog-genomics.org/plink/1.9" in combined.lower():
        return True, "plink1"

    # Some builds print version help on --help only
    proc2 = subprocess.run([binary_path, "--help"], capture_output=True, text=True, check=False)
    combined2 = (proc2.stdout or "") + "\n" + (proc2.stderr or "")
    l2 = combined2.lower()
    if "putty" in l2:
        return False, None
    if "plink v2." in l2 or "plink2" in l2:
        return True, "plink2"
    if "plink v1." in l2 or "www.cog-genomics.org/plink/1.9" in l2:
        return True, "plink1"

    return False, None


def find_plink(explicit_path: Optional[str]) -> Tuple[str, str]:
    """Find a valid PLINK binary. Returns (binary_path, flavor).

    Search order:
      1) --plink-path if provided
      2) plink2 / plink2.exe in PATH
      3) plink / plink.exe in PATH
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    # Prefer plink2
    for name in ("plink2", "plink2.exe"):
        p = shutil.which(name)
        if p:
            candidates.append(p)

    # Fallback to plink 1.9
    for name in ("plink", "plink.exe"):
        p = shutil.which(name)
        if p:
            candidates.append(p)

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        ok, flavor = is_valid_plink(path)
        if ok and flavor:
            return path, flavor

    raise FileNotFoundError(
        "No valid genetics PLINK binary found. Please install PLINK 1.9/2.0 and ensure it is in PATH, "
        "or pass --plink-path to this script. Note: PuTTY's plink.exe is not supported."
    )


def convert_vcf_to_bed(plink_bin: str, flavor: str, vcf_path: str, out_prefix: str, allow_extra_chr: bool) -> None:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    base_args = [plink_bin]
    if flavor == "plink2":
        # plink2 supports the same flags for this conversion use-case
        args = base_args + [
            "--vcf",
            vcf_path,
            "--make-bed",
            "--double-id",
        ]
    else:
        args = base_args + [
            "--vcf",
            vcf_path,
            "--make-bed",
            "--double-id",
        ]

    if allow_extra_chr:
        args.append("--allow-extra-chr")

    # Auto-generate variant IDs when missing: CHR:POS:REF:ALT
    # Pass as a single argument; no need to escape $ when not in a shell string.
    args += ["--set-missing-var-ids", "@:#:$1:$2"]

    args += ["--out", out_prefix]

    print("Running:", " ".join(args))
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"PLINK failed with exit code {proc.returncode}")
    else:
        # Print PLINK's summary lines
        sys.stdout.write(proc.stdout)
        sys.stdout.write(proc.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VCF to PLINK binary (BED/BIM/FAM)")
    parser.add_argument("--vcf", required=True, help="Path to input VCF file (bgzipped or plain)")
    parser.add_argument(
        "--out-prefix",
        default=os.path.join("results", "plink", "test"),
        help="Output prefix for PLINK files (default: results/plink/test)",
    )
    parser.add_argument(
        "--plink-path",
        default=None,
        help="Explicit path to plink/plink2 binary (optional)",
    )
    parser.add_argument(
        "--allow-extra-chr",
        action="store_true",
        help="Pass --allow-extra-chr to PLINK for nonstandard chromosome labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"VCF not found: {args.vcf}")

    plink_bin, flavor = find_plink(args.plink_path)
    print(f"Detected PLINK binary: {plink_bin} (flavor: {flavor})")

    convert_vcf_to_bed(
        plink_bin=plink_bin,
        flavor=flavor,
        vcf_path=args.vcf,
        out_prefix=args.out_prefix,
        allow_extra_chr=bool(args.allow_extra_chr),
    )

    out_dir = os.path.dirname(args.out_prefix) or "."
    bed = args.out_prefix + ".bed"
    bim = args.out_prefix + ".bim"
    fam = args.out_prefix + ".fam"
    print("\nDone. Outputs:")
    print(f"- {bed}")
    print(f"- {bim}")
    print(f"- {fam}")


if __name__ == "__main__":
    main()









