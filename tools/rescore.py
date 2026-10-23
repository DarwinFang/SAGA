"""Check stored primary scores against the benchmark references."""

import argparse
from pathlib import Path

import numpy as np

from saga.data import DATASETS, Panel, load_quality, write_json
from saga.evaluation import score_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/rescore.json"))
    args = parser.parse_args()
    report = {}
    for dataset in args.datasets:
        folder = args.data / dataset
        panel = Panel.load(folder)
        saved, metric = load_quality(folder, panel)
        actual = score_candidates(folder, dataset, panel)
        np.testing.assert_allclose(
            actual[panel.valid], saved[panel.valid], atol=1e-10, rtol=0
        )
        report[dataset] = {"n": len(panel), "metric": metric, "scores_match": True}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
