"""Vector plots for weight variation and fixed query regions."""

import numpy as np

from .data import read_jsonl
from .grouping import assign, cluster
from .runner import QUERY


def plot_figures(args, load):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    output = args.output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    panels = {ds: load(args, ds) for ds in args.datasets}
    spread = [
        np.linalg.norm(a[QUERY + "_weights"] - a[QUERY + "_weights"].mean(0), axis=1)
        for p, y, metric, a in panels.values()
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    violin = ax.violinplot(spread, showmedians=True)
    for body in violin["bodies"]:
        body.set_facecolor("#338BC2")
        body.set_alpha(1)
        body.set_edgecolor("#1B5981")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        violin[key].set_color("black")
        violin[key].set_linewidth(0.7)
    ax.set_xticks(range(1, len(panels) + 1), list(panels), fontsize=8)
    ax.set_ylabel("Distance to mean weight vector", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for extension in ("pdf", "svg"):
        fig.savefig(output / ("four_datasets_weight_violin." + extension))
    plt.close(fig)

    palette = [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#626262",
        "#BCBD22",
        "#17BECF",
        "#263B8C",
        "#006B4F",
        "#A50F15",
        "#7B4173",
        "#B8860B",
        "#008C99",
    ]
    markers = ["o", "^", "s", "*"]
    paths = [
        MarkerStyle(m).get_path().transformed(MarkerStyle(m).get_transform())
        for m in markers
    ]
    sizes = np.asarray([2.8, 3.6, 2.8, 7.2]) * 1.35**2
    fig = plt.figure(figsize=(16 / 2.54, 4.8 / 2.54), facecolor="white")
    for j, (ds, (p, y, metric, a)) in enumerate(panels.items()):
        ax = fig.add_axes([0.04 + 0.25 * j, 1.05 / 4.8, 0.205, 2.6 / 4.8])
        queries = p.queries.astype(float)
        centered = queries - queries.mean(0)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        xy = u[:, :2] * singular[:2]
        regions = assign(queries, cluster(queries, 4))
        instances = {r["id"]: r for r in read_jsonl(args.data / ds / "instances.jsonl")}
        labels = [instances[str(i)].get("label", "") for i in p.ids]
        mapping = dict(zip(sorted(set(labels)), palette))
        colors = np.asarray([mapping[label] for label in labels])
        order = np.random.default_rng(42).permutation(len(p))
        collection = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=colors[order],
            s=sizes[regions[order]],
            alpha=1,
            edgecolors="white",
            linewidths=0.10,
            rasterized=False,
        )
        collection.set_paths([paths[k] for k in regions[order]])
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7, width=0.4, length=2, pad=2)
        ax.set_axisbelow(True)
        ax.grid(color="#E6E6E6", linewidth=0.30)
        fig.text(
            0.1425 + 0.25 * j,
            0.025,
            f"({chr(97 + j)}) {ds}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
    handles = [
        Line2D(
            [],
            [],
            color="#252525",
            marker=m,
            linestyle="None",
            markersize=4.5 if m == "*" else 3.5,
            label=f"Region {k + 1}",
        )
        for k, m in enumerate(markers)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.98),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=1.5,
        borderaxespad=0,
    )
    for extension in ("pdf", "svg"):
        fig.savefig(output / ("query_regions." + extension))
    plt.close(fig)
    correlation_files = plot_correlations(args, panels)
    return {
        "files": correlation_files + [
            "figures/" + name + "." + ext
            for name in ["four_datasets_weight_violin", "query_regions"]
            for ext in ["pdf", "svg"]
        ]
    }


def plot_correlations(args, panels):
    """Source accuracy and exact-answer fractional-count correlations."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from .analysis import category_correlations
    from .runner import selections

    names = [name for name in ("Mintaka", "PopQA") if name in panels]
    if not names:
        return []
    methods = ["SMS", "RASA", "BAU", "AQER", "SAGA"]
    cmap = LinearSegmentedColormap.from_list("pearson", ["#D96D54", "#FAF8F2", "#087FAB"])
    fig = plt.figure(figsize=(7.05, 6.2))
    locations = {"Mintaka": (.687, .236, .976), "PopQA": (.150, .446, .649)}
    for ds in names:
        p, quality, _, arrays = panels[ds]
        instances = {r["id"]: r for r in read_jsonl(args.data / ds / "instances.jsonl")}
        labels = np.array([instances[str(i)]["label"] for i in p.ids])
        stats = category_correlations(p, quality, selections(arrays), labels)
        accuracy = np.array(stats["source_em"])
        correlations = np.array([stats["methods"][m]["pearson"] for m in methods]).T
        bottom, height, header = locations[ds]
        labels = ["Video games" if s == "videogames" else s.capitalize() for s in stats["labels"]] + ["Mean"]
        for j, (values, columns) in enumerate([(accuracy, ["Qwen", "Llama", "DeepSeek", "Granite"]), (correlations, methods)]):
            values = np.vstack([values, values.mean(0)])
            ax = fig.add_axes([.170 if j == 0 else .540, bottom, .340 if j == 0 else .440, height])
            nr, nc = values.shape
            mesh = ax.pcolormesh(np.arange(nc + 1) - .5, np.arange(nr + 1) - .5, values,
                                cmap=cmap if j else "YlGnBu", vmin=-1 if j else 0,
                                vmax=1 if j else 100, edgecolors="white", linewidth=.9, rasterized=False)
            ax.set_ylim(nr - .5, -.5)
            ax.set_xticks(range(nc), columns, fontsize=8.8)
            ax.xaxis.tick_top()
            ax.tick_params(length=0, pad=5)
            ax.set_yticks(range(nr), labels if j == 0 else [""] * nr, fontsize=8.3)
            for row in range(nr):
                for col in range(nc):
                    value = values[row, col]
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8.4,
                            color="white" if value > (.69 if j else 63) else "#24333B")
            ax.axhline(nr - 1.5, color="#3F5058", linewidth=.85)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.text(.014, header, ds, fontsize=10.5)
        fig.text(.340, header, "Source accuracy (EM, %)", ha="center", fontsize=9)
        fig.text(.760, header, "Pearson correlation", ha="center", fontsize=9)
    bar = fig.colorbar(mesh, cax=fig.add_axes([.54, .06, .44, .018]), orientation="horizontal")
    bar.solids.set_rasterized(False)
    bar.ax.tick_params(labelsize=8)
    files = []
    for extension in ("pdf", "svg"):
        name = "figures/source_em_pearson_saga_style." + extension
        fig.savefig(args.output / name)
        files.append(name)
    plt.close(fig)
    return files
