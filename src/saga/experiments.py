"""Run the experiments reported in the paper."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .analysis import decision_ablations, primary_reversal
from .data import DATASETS, Panel, evaluation_texts, load_quality, read_jsonl, write_json
from .evaluation import selection_score
from .runner import GLOBAL, QUERY, fit_panel, selections
from .supplementary import supplementary_results
from .analysis import category_correlations

BASELINES = ("SMS", "RASA", "BAU", "AQER")


def main_fit(args, ds, shared=False):
    panel = Panel.load(args.data / ds)
    arrays = fit_panel(panel, args.output / "fits/main" / ds, args.config["estimator"], shared=shared)
    quality, metric = load_quality(args.data / ds, panel)
    return panel, quality, metric, arrays


def scores(quality, arrays, methods=None):
    return {
        m: selection_score(quality, p)
        for m, p in selections(arrays).items()
        if methods is None or m in methods
    }


def main_results(args):
    result = {}
    for ds in args.datasets:
        p, y, metric, a = main_fit(args, ds)
        result[ds] = {
            "n": len(p),
            "metric": metric,
            "scores": scores(y, a),
            "single_sources": {
                str(s): float(y[:, k].mean()) for k, s in enumerate(p.sources)
            },
        }
    return result


def shuffle_fit(task):
    ds, data, output, estimator, repeat, order = task
    panel = Panel.load(Path(data) / ds)
    shuffled = replace(panel, queries=panel.queries[np.asarray(order)])
    arrays = fit_panel(
        shuffled,
        Path(output) / "fits/shuffled" / ds / str(repeat),
        estimator,
        baselines=False,
    )
    y, _ = load_quality(Path(data) / ds, panel)
    return selection_score(y, selections(arrays)[QUERY])


def ablations(args):
    result = {}
    for ds in args.datasets:
        p, y, metric, a = main_fit(args, ds, shared=True)
        rules = decision_ablations(p, a[QUERY + "_alignment"])
        values = {name: selection_score(y, picks) for name, picks in rules.items()}
        values["SAGA w/o Query"] = selection_score(y, selections(a)[GLOBAL])
        rng = np.random.default_rng(args.config["query_control"]["seed"])
        tasks = [
            (
                ds,
                str(args.data),
                str(args.output),
                args.config["estimator"],
                k,
                rng.permutation(len(p)).tolist(),
            )
            for k in range(args.config["query_control"]["permutations"])
        ]
        if args.workers == 1:
            shuffled = list(map(shuffle_fit, tasks))
        else:
            with ProcessPoolExecutor(args.workers) as pool:
                shuffled = list(pool.map(shuffle_fit, tasks))
        values["SAGA-SQ"] = float(np.mean(shuffled))
        values["SAGA"] = selection_score(y, selections(a)[QUERY])
        result[ds] = {"metric": metric, "scores": values, "shuffle_scores": shuffled}
    return result


def source_composition(args):
    result = {}
    for ds in args.datasets:
        p, y, metric, _ = main_fit(args, ds)
        rows = []
        for cols in combinations(range(len(p.sources)), 3):
            cols = np.asarray(cols)
            sub = replace(
                p,
                sources=p.sources[cols],
                answers=p.answers[:, cols],
                texts=p.texts[:, cols],
                valid=p.valid[:, cols],
                source_indices=np.tile(np.arange(3), (len(p), 1)),
            )
            arrays = fit_panel(
                sub,
                args.output / "fits/composition" / ds / "".join(map(str, cols)),
                args.config["estimator"],
            )
            rows.append(
                {
                    "sources": sub.sources.tolist(),
                    "scores": scores(y[:, cols], arrays, (*BASELINES, QUERY)),
                }
            )
        result[ds] = {"metric": metric, "subsets": rows}
    return result


def source_selection(args):
    result = {}
    for ds in args.datasets:
        p, y, metric, a = main_fit(args, ds)
        weights = a[QUERY + "_weights"]
        spread = np.linalg.norm(weights - weights.mean(0), axis=1)
        result[ds] = {
            "sources": p.sources.tolist(),
            "n": len(p),
            "counts": {
                m: np.bincount(picks, minlength=len(p.sources)).tolist()
                for m, picks in selections(a).items()
                if m in (*BASELINES, QUERY)
            },
            "distances_to_mean_weight": spread.tolist(),
            "median_distance": float(np.median(spread)),
        }
        if ds in ("Mintaka", "PopQA"):
            instances = {r["id"]: r for r in read_jsonl(args.data / ds / "instances.jsonl")}
            labels = np.array([instances[str(i)]["label"] for i in p.ids])
            result[ds]["categories"] = category_correlations(p, y, selections(a), labels)
    return result


def supplementary(args):
    return supplementary_results(args, main_fit)


def reversal(args):
    result = {}
    for ds in args.datasets:
        if ds not in args.config["reversal"]["datasets"]:
            continue
        p = Panel.load(args.data / ds)
        y, metric = load_quality(args.data / ds, p)
        cfg = args.config["reversal"]
        batch = primary_reversal(
            p, y, cfg["regions"], cfg["minimum_per_direction"], cfg["seed"]
        )
        ix = np.asarray(batch["indices"])
        arrays = fit_panel(
            p.subset(ix), args.output / "fits/reversal" / ds, args.config["estimator"]
        )
        result[ds] = {
            **batch,
            "metric": metric,
            "scores": scores(y[ix], arrays, (*BASELINES, QUERY)),
        }
    return result


def case_studies(args):
    result = {}
    for ds in args.datasets:
        p, y, metric, a = main_fit(args, ds)
        instances = {r["id"]: r for r in read_jsonl(args.data / ds / "instances.jsonl")}
        references = {
            r["id"]: r for r in read_jsonl(args.data / ds / "references.jsonl")
        }
        cases = []
        for identity in args.config["cases"][ds]:
            matches = np.flatnonzero(p.ids == identity)
            if len(matches) != 1:
                raise ValueError("Case ID must identify exactly one instance")
            i = int(matches[0])
            cases.append(
                {
                    "id": identity,
                    "query": instances[identity]["query"],
                    "reference": references[identity],
                    "sources": p.sources.tolist(),
                    "candidates": p.texts[i].tolist(),
                    "evaluation_answers": evaluation_texts(args.data / ds, ds, p)[i].tolist(),
                    "quality": y[i].tolist(),
                    "alignment": a[QUERY + "_alignment"][i].tolist(),
                    "weights": a[QUERY + "_weights"][i].tolist(),
                    "choices": {m: int(v[i]) for m, v in selections(a).items()},
                }
            )
        result[ds] = {"metric": metric, "cases": cases}
    return result


def figures(args):
    from .plotting import plot_figures

    return plot_figures(args, main_fit)


COMMANDS = {
    "main": main_results,
    "supplementary": supplementary,
    "ablations": ablations,
    "source-composition": source_composition,
    "source-selection": source_selection,
    "reversal": reversal,
    "cases": case_studies,
    "figures": figures,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=COMMANDS)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.config = json.loads(args.config.read_text())
    result = COMMANDS[args.experiment](args)
    write_json(args.output / (args.experiment + ".json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
