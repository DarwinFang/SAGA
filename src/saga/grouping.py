"""Fixed query-embedding regions for expertise-reversal batches."""

import warnings

import numpy as np
from scipy.cluster.vq import kmeans2


def cluster(profiles, count=4):
    fits = []
    for start in range(10):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                centers, labels = kmeans2(
                    profiles,
                    count,
                    iter=100,
                    minit="++",
                    missing="raise",
                    rng=np.random.default_rng(42 + start),
                )
            if len(set(labels.tolist())) == count:
                inertia = float(((profiles - centers[labels]) ** 2).sum())
                fits.append((inertia, centers))
        except (Warning, ValueError):
            continue
    if not fits:
        raise ValueError("No nonempty query partition could be fitted.")
    _, centers = min(fits, key=lambda pair: pair[0])
    order = sorted(
        range(count),
        key=lambda i: (
            int(centers[i].argmax()),
            float(centers[i, 0]),
            float(centers[i, 1]),
        ),
    )
    return centers[order]


def assign(profiles, centers):
    return ((profiles[:, None] - centers[None]) ** 2).sum(2).argmin(1)
