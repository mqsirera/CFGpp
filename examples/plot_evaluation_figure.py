"""
Create summary figures for evaluation experiments.

This script reads the `evaluation_summary.json` produced by
`evaluate_cfg_comparison.py` and creates matplotlib figures:

- One figure per prompt
- Top row: CFG results (all guidance scales)
- Bottom row: CFG++ results (all guidance scales)

Each figure shows all guidance scales side by side, with the scale value
annotated in the subplot title.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def load_summary(results_dir: Path):
    """Load the evaluation summary JSON file."""
    summary_path = results_dir / "evaluation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary JSON at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)
    return summary


def build_prompt_figure(
    prompt: str,
    prompt_idx: int,
    cfg_entries: list,
    cfgpp_entries: list,
    results_dir: Path,
    output_name: str,
    figsize_scale: float = 3.0,
):
    """
    Build and save a figure for one prompt with CFG on top and CFG++ on bottom.

    Args:
        prompt: The prompt text
        prompt_idx: Index of the prompt
        cfg_entries: list of dicts with keys including 'guidance' and 'path' for CFG
        cfgpp_entries: list of dicts with keys including 'guidance' and 'path' for CFG++
        results_dir: directory where images live
        output_name: filename for the saved figure (inside results_dir)
        figsize_scale: scaling factor for figure width
    """
    # Sort by guidance scale to keep order consistent
    cfg_entries = sorted(cfg_entries, key=lambda e: e.get("guidance", 0.0))
    cfgpp_entries = sorted(cfgpp_entries, key=lambda e: e.get("guidance", 0.0))

    # Determine number of columns (use max of CFG and CFG++ scales)
    n_cfg = len(cfg_entries)
    n_cfgpp = len(cfgpp_entries)
    n_cols = max(n_cfg, n_cfgpp)

    if n_cols == 0:
        print(f"Warning: No images found for prompt {prompt_idx}")
        return

    # Create 2-row subplot: top row for CFG, bottom row for CFG++
    # Reduce spacing between subplots
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(figsize_scale * n_cols * 1.1, figsize_scale * 2.0),
        squeeze=False,
    )
    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # Top row: CFG images
    for col_idx, entry in enumerate(cfg_entries):
        ax = axes[0, col_idx]
        stored_path = Path(entry["path"])

        # Try multiple resolution strategies for robustness
        img_path = results_dir / stored_path.name
        if not img_path.exists():
            img_path = stored_path
            if not img_path.exists():
                if not img_path.is_absolute():
                    img_path = Path.cwd() / stored_path
                if not img_path.exists():
                    img_path = results_dir / stored_path.name

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            guidance = entry.get("guidance", None)
            if guidance is not None:
                ax.set_title(f"CFG\nscale={guidance:.2f}", fontsize=14)
            else:
                ax.set_title("CFG", fontsize=14)
        except Exception as e:
            ax.set_title(f"Error\n{img_path.name}", fontsize=8)
            ax.axis("off")
            print(f"Warning: Could not load CFG image {stored_path}: {e}")

    # Hide unused CFG subplots
    for col_idx in range(n_cfg, n_cols):
        axes[0, col_idx].axis("off")

    # Bottom row: CFG++ images
    for col_idx, entry in enumerate(cfgpp_entries):
        ax = axes[1, col_idx]
        stored_path = Path(entry["path"])

        # Try multiple resolution strategies for robustness
        img_path = results_dir / stored_path.name
        if not img_path.exists():
            img_path = stored_path
            if not img_path.exists():
                if not img_path.is_absolute():
                    img_path = Path.cwd() / stored_path
                if not img_path.exists():
                    img_path = results_dir / stored_path.name

        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            guidance = entry.get("guidance", None)
            if guidance is not None:
                ax.set_title(f"CFG++\nscale={guidance:.2f}", fontsize=14)
            else:
                ax.set_title("CFG++", fontsize=14)
        except Exception as e:
            ax.set_title(f"Error\n{img_path.name}", fontsize=8)
            ax.axis("off")
            print(f"Warning: Could not load CFG++ image {stored_path}: {e}")

    # Hide unused CFG++ subplots
    for col_idx in range(n_cfgpp, n_cols):
        axes[1, col_idx].axis("off")

    # Set overall title
    plt.suptitle(f"Prompt {prompt_idx}: '{prompt}'", fontsize=16, y=0.99)

    # Tight layout with minimal padding
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.0)
    out_path = results_dir / output_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side evaluation figures for CFG (top) and CFG++ (bottom)"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("examples/workdir/evaluation"),
        help="Workdir used in the evaluation experiment (the one passed to --workdir there).",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdirectory under workdir where results and JSON are stored.",
    )
    parser.add_argument(
        "--prompt_idx",
        type=int,
        default=None,
        help="If set, only plot figure for this prompt index. Otherwise, plot all prompts.",
    )
    parser.add_argument(
        "--figsize_scale",
        type=float,
        default=3.0,
        help="Scaling factor for figure size (default: 3.0)",
    )

    args = parser.parse_args()

    results_dir = args.workdir / args.results_subdir
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    summary = load_summary(results_dir)
    all_results = summary.get("results", [])
    prompts = summary.get("prompts", [])

    if not all_results:
        raise RuntimeError("No results found in summary.")

    # Group results by prompt and method
    prompt_data = {}
    for result in all_results:
        prompt = result.get("prompt", "")
        method = result.get("method", "")
        prompt_idx = prompts.index(prompt) if prompt in prompts else -1

        if prompt_idx not in prompt_data:
            prompt_data[prompt_idx] = {
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "cfg": [],
                "cfgpp": [],
            }

        if method == "CFG":
            prompt_data[prompt_idx]["cfg"].append(result)
        elif method == "CFG++":
            prompt_data[prompt_idx]["cfgpp"].append(result)

    # Generate figures
    prompt_indices = (
        [args.prompt_idx] if args.prompt_idx is not None else sorted(prompt_data.keys())
    )

    for prompt_idx in prompt_indices:
        if prompt_idx not in prompt_data:
            print(f"Warning: No data found for prompt index {prompt_idx}")
            continue

        data = prompt_data[prompt_idx]
        prompt = data["prompt"]
        cfg_entries = data["cfg"]
        cfgpp_entries = data["cfgpp"]

        if not cfg_entries and not cfgpp_entries:
            print(f"Warning: No images found for prompt {prompt_idx}: '{prompt}'")
            continue

        # Create filename-safe version of prompt for output
        prompt_safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in prompt)
        prompt_safe = prompt_safe.replace(" ", "_")[:50]  # Limit length

        output_name = f"evaluation_figure_prompt_{prompt_idx:02d}_{prompt_safe}.png"

        build_prompt_figure(
            prompt,
            prompt_idx,
            cfg_entries,
            cfgpp_entries,
            results_dir,
            output_name,
            figsize_scale=args.figsize_scale,
        )

    print(f"\nEvaluation figure generation complete!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

