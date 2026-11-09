#!/usr/bin/env python3
"""Expand a Restormer-style file-pair list into an AICT per-view list.

Restormer's ``train_sim_proj.txt`` / ``valid_sim_proj.txt`` usually contain one
line per (input, label) raw-file pair::

    /path/to/proj_pbi_fs_dec_blur_phase.raw /path/to/proj_no_pbi.raw

AICT's ``proj_reader`` instead expects one line per *view*::

    /path/to/input.raw /path/to/label.raw 0

This script expands every pair into ``--views`` lines (default 100) and
optionally checks that each raw file has the expected byte size.

If the source file already contains a third integer column, it is treated as an
already-expanded per-view list and copied through instead of expanded again.

Examples::

    python scripts/make_nf0128_list.py \
        ~/restormer/train_sim_proj.txt nf0128_list/train_nf0128_views.txt

    python scripts/make_nf0128_list.py \
        ~/restormer/valid_sim_proj.txt nf0128_list/valid_nf0128_views.txt

    # Skip the byte-size check (e.g. when the files are on another mount)
    python scripts/make_nf0128_list.py --no-check in.txt out.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path

DEFAULT_VIEWS = 100
DEFAULT_EXPECTED_SIZE = 100 * 500 * 500 * 4  # 100,000,000 bytes


def _resolve(path: str, path_root: Path) -> str:
    """Expand ~ / env vars and optionally resolve relative paths."""
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"\'')))
    p = Path(path)
    if not p.is_absolute() and path_root is not None:
        p = path_root / p
    return str(p.resolve())


def _head_id(path: str):
    """Extract the two-digit head id from a path like .../07/dose_1e8/..."""
    m = re.search(r"/(\d{2})/", path.replace("\\", "/"))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Restormer-style pair list")
    ap.add_argument("dst", help="Output per-view list")
    ap.add_argument("--views", type=int, default=DEFAULT_VIEWS,
                    help=f"Views per file (default {DEFAULT_VIEWS})")
    ap.add_argument("--expected-size", type=int, default=DEFAULT_EXPECTED_SIZE,
                    help=f"Expected bytes per raw file (default {DEFAULT_EXPECTED_SIZE})")
    ap.add_argument("--no-check", action="store_true",
                    help="Do not check raw-file byte sizes")
    ap.add_argument("--path-root", default=None,
                    help="Base directory for relative paths in src "
                         "(default: directory containing src)")
    ap.add_argument("--keep-relative", action="store_true",
                    help="Write paths exactly as given instead of absolute paths")
    ap.add_argument("--report-heads", action="store_true",
                    help="Print the unique head ids found in the output list")
    ap.add_argument("--already-per-view", action="store_true",
                    help="Source already has a third integer view-index column; "
                         "copy it through instead of expanding")
    ap.add_argument("--drop-extra-columns", action="store_true",
                    help="When not --already-per-view, ignore columns after the "
                         "first two instead of aborting")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"source list not found: {src}")

    path_root = Path(args.path_root) if args.path_root else src.parent

    out_dir = Path(args.dst).parent
    if str(out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    n_views = 0
    heads = set()

    with open(src, encoding="utf-8") as f, open(args.dst, "w", encoding="utf-8") as g:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                sys.exit(f"{src}:{lineno}: expected at least 2 columns, got: {line!r}")

            inp = _resolve(parts[0], path_root)
            lab = _resolve(parts[1], path_root)

            # Decide whether to copy through or expand. Never guess: a third
            # column may be a key rather than a view index, and silently
            # expanding a per-view list would duplicate every view 100 times.
            if args.already_per_view:
                if len(parts) < 3:
                    sys.exit(f"{src}:{lineno}: --already-per-view requires a "
                             f"third column with the view index")
                try:
                    view_idx = int(parts[2])
                except ValueError:
                    sys.exit(f"{src}:{lineno}: third column is not an integer: "
                             f"{parts[2]!r}")
            elif len(parts) > 2 and not args.drop_extra_columns:
                sys.exit(
                    f"{src}:{lineno}: found {len(parts)} columns but this list is "
                    f"being treated as file pairs. If the third column is a view "
                    f"index, pass --already-per-view; if it is an irrelevant key, "
                    f"pass --drop-extra-columns."
                )

            if not args.no_check:
                for p in (inp, lab):
                    if not os.path.exists(p):
                        sys.exit(f"{src}:{lineno}: file not found: {p}")
                    size = os.path.getsize(p)
                    if size != args.expected_size:
                        sys.exit(
                            f"{src}:{lineno}: unexpected file size for {p}: "
                            f"{size} != {args.expected_size}. "
                            f"Pass --expected-size or --no-check."
                        )

            if not args.keep_relative:
                out_inp, out_lab = inp, lab
            else:
                out_inp, out_lab = parts[0], parts[1]

            if args.already_per_view:
                if not 0 <= view_idx < args.views:
                    sys.exit(f"{src}:{lineno}: view index {view_idx} out of range "
                             f"[0, {args.views})")
                g.write(f"{out_inp} {out_lab} {view_idx}\n")
                n_views += 1
            else:
                for v in range(args.views):
                    g.write(f"{out_inp} {out_lab} {v}\n")
                    n_views += 1
            h = _head_id(inp)
            if h:
                heads.add(h)

            n_lines += 1

    print(f"pairs: {n_lines}  ->  views: {n_views}  ->  {args.dst}")
    if heads:
        print(f"heads ({len(heads)}): {', '.join(sorted(heads))}")


if __name__ == "__main__":
    main()
