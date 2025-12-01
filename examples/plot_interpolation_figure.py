"""
Create summary figures for interpolation experiments.

This script reads the `interpolation_summary.json` produced by
`prompt_interpolation_experiment.py` and creates nice matplotlib figures:

- One figure for CFG
- One figure for CFG++

Each figure shows all interpolation steps (different alphas) side by side,
with the alpha value annotated in the subplot title.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def load_summary(results_dir: Path):
    summary_path = results_dir / "interpolation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary JSON at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)
    return summary


def build_method_figure(
    entries,
    method_name: str,
    results_dir: Path,
    output_name: str,
    figsize_scale: float = 3.0,
):
    """
    Build and save a figure for one method (CFG or CFG++).

    Args:
        entries: list of dicts with keys including 'alpha' and 'path'
        method_name: "CFG" or "CFG++"
        results_dir: directory where images live
        output_name: filename for the saved figure (inside results_dir)
        figsize_scale: scaling factor for figure width
    """
    if not entries:
        return

    # Sort by alpha to keep order consistent
    entries = sorted(entries, key=lambda e: e.get("alpha", 0.0))

    n = len(entries)
    # Rough heuristic: each image gets figsize_scale units of width
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_scale * n, figsize_scale * 0.9),
        squeeze=False,
    )
    axes = axes[0]

    for ax, entry in zip(axes, entries):
        # Paths in the summary are stored workspace-relative (e.g.,
        # "examples/workdir/.../results/xxx.png"), referenced from the repo root.
        # We therefore treat them as-is relative to the current working directory.
        img_path = Path(entry["path"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            ax.set_title(f"Error\n{img_path.name}", fontsize=8)
            ax.axis("off")
            continue

        ax.imshow(img)
        ax.axis("off")
        alpha = entry.get("alpha", None)
        if alpha is not None:
            ax.set_title(rf"$\alpha={alpha:.2f}$", fontsize=10)
        else:
            ax.set_title(img_path.name, fontsize=8)

    prompts = []
    if entries and "prompt1" in entries[0] and "prompt2" in entries[0]:
        prompts = [entries[0]["prompt1"], entries[0]["prompt2"]]

    if prompts:
        plt.suptitle(
            f"{method_name} interpolation: '{prompts[0]}' → '{prompts[1]}'",
            fontsize=12,
        )
    else:
        plt.suptitle(f"{method_name} interpolation", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = results_dir / output_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {method_name} interpolation figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side interpolation figures for CFG and CFG++"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("examples/workdir/prompt_interpolation"),
        help="Workdir used in the interpolation experiment (the one passed to --workdir there).",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdirectory under workdir where results and JSON are stored.",
    )
    parser.add_argument(
        "--interpolation_method",
        type=str,
        default=None,
        help="If set, filter to this interpolation method (e.g. 'linear' or 'slerp').",
    )

    args = parser.parse_args()

    results_dir = args.workdir / args.results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    summary = load_summary(results_dir)
    all_results = summary.get("results", [])

    # Filter for linear/slerp entries with alpha (skip multi_blend for now)
    filtered = [
        r
        for r in all_results
        if "alpha" in r
        and (args.interpolation_method is None
             or r.get("interpolation_method") == args.interpolation_method)
    ]

    if not filtered:
        raise RuntimeError(
            "No interpolation entries with 'alpha' found in summary. "
            "Did you run a linear/slerp interpolation with --compare_both?"
        )

    # Group by method
    cfg_entries = [r for r in filtered if r.get("method") == "CFG"]
    cfgpp_entries = [r for r in filtered if r.get("method") == "CFG++"]

    if not cfg_entries and not cfgpp_entries:
        raise RuntimeError("No CFG or CFG++ entries found in summary.")

    # Determine method name from summary if not specified
    interpolation_method = (
        args.interpolation_method or summary.get("interpolation_method", "linear")
    )

    if cfg_entries:
        build_method_figure(
            cfg_entries,
            "CFG",
            results_dir,
            output_name=f"interpolation_figure_cfg_{interpolation_method}.png",
        )

    if cfgpp_entries:
        build_method_figure(
            cfgpp_entries,
            "CFG++",
            results_dir,
            output_name=f"interpolation_figure_cfgpp_{interpolation_method}.png",
        )


if __name__ == "__main__":
    main()


