"""Command-line entry point. Run `python -m fresh_corn_wine --help`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fresh corn wine: processed-data analyses and figure reproduction")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("analyze", "Recompute summary statistics, scores and estimated aroma indices"),
        ("figures", "Draw publication figures from processed data and archived model outputs"),
        ("validate", "Check numerical agreement with the released supplementary workbook"),
        ("build", "Run analyses, draw figures and validate the release"),
    ]:
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory (default: ./outputs)")
        if name in {"figures", "build"}:
            sub.add_argument("--figures", nargs="+", choices=[f"Fig{i}" for i in range(1, 8)] + [f"FigS{i}" for i in range(1, 5)], help="Figure IDs; default: all 11")
            sub.add_argument("--dpi", type=int, default=600, help="PNG resolution; PDFs remain vector where applicable")
    args = parser.parse_args(argv)
    if hasattr(args, "dpi") and not 50 <= args.dpi <= 1200:
        parser.error("--dpi must be between 50 and 1200")
    try:
        if args.command in {"analyze", "build"}:
            from .analysis import analyze
            files = analyze(args.output / "tables")
            print(f"Wrote {len(files)} analysis files to {(args.output / 'tables').resolve()}")
        if args.command in {"figures", "build"}:
            from .plotting import generate_figures
            files = generate_figures(args.output / "figures", names=args.figures, dpi=args.dpi)
            print(f"Wrote {len(files)} figure files to {(args.output / 'figures').resolve()}")
        if args.command in {"validate", "build"}:
            from .validation import validate
            report = validate()
            directory = args.output / "checks"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Validation: {'PASS' if report['passed'] else 'FAIL'} ({report['checks_run']} checks)")
            if not report["passed"]:
                for error in report["errors"]:
                    print(f"  {error}")
                return 1
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
