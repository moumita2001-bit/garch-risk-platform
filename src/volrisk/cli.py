"""Command-line entry point: `volrisk run`."""

import argparse
import logging
import sys

from volrisk.pipeline import PipelineError, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="volrisk", description="VolRisk volatility and VaR platform")
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="download, clean, fit and compute tomorrow's VaR")
    run_cmd.add_argument("--ticker", default="SPY")
    run_cmd.add_argument("--start", default="2008-01-01")
    run_cmd.add_argument("--force", action="store_true", help="re-download even if done today")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        path = run(ticker=args.ticker, start=args.start, force=args.force)
    except PipelineError as exc:
        logging.error("pipeline failed: %s", exc)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
