"""Run every example script and save its figures as PDF for the book.

Usage:  python book/build_figures.py [examples/<section>/plot_xx.py ...]

Figures are written to book/figures/<section>/<script>_<n>.pdf where n is
the 1-based figure number in the script, so that the LaTeX macro
``\\exfig{<section>/<script>_<n>}{caption}{label}`` can include them.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
OUT = ROOT / "book" / "figures"


def build(script: pathlib.Path) -> list[pathlib.Path]:
    section = script.parent.name
    out_dir = OUT / section
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.close("all")
    runpy.run_path(str(script), run_name="__main__")
    written = []
    for n, num in enumerate(plt.get_fignums(), start=1):
        target = out_dir / f"{script.stem}_{n}.pdf"
        plt.figure(num).savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close("all")
    return written


def main(argv: list[str]) -> int:
    scripts = [pathlib.Path(a).resolve() for a in argv] if argv else sorted(EXAMPLES.glob("*/plot_*.py"))
    for script in scripts:
        try:
            files = build(script)
            print(f"{script.relative_to(ROOT)}: {len(files)} figure(s)")
        except Exception as exc:  # keep going, report at the end
            print(f"FAILED {script}: {exc!r}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
