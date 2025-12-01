"""
Create summary figures for semantic difference experiments.

This script reads the results from `semantic_difference_experiment.py` and creates
matplotlib figures, especially for comparison experiments (--compare_both):

- Baseline prompt (CFG and CFG++)
- Concept references (concept1 and concept2)
- Different strength scales (sorted by strength value)

Layout: 
- Row 1: CFG baseline, concept1, concept2, then strength images (CFG)
- Row 2: CFG++ baseline, concept1, concept2, then strength images (CFG++)
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from PIL import Image


def find_semantic_diff_files(results_dir: Path, base_prompt: str, concept1: str, concept2: str):
    """
    Find all relevant files for a semantic difference experiment.
    
    Returns:
        dict with keys: 'baseline_cfg', 'baseline_cfgpp', 'baseline' (for non-comparison),
                       'concept1', 'concept2', 
                       'strengths_cfg', 'strengths_cfgpp', 'strengths' (for non-comparison),
                       'is_comparison' (bool)
    """
    base_prompt_safe = base_prompt.replace(' ', '_')
    concept1_safe = concept1.replace(' ', '_')
    concept2_safe = concept2.replace(' ', '_')
    
    files = {
        'baseline_cfg': None,
        'baseline_cfgpp': None,
        'baseline': None,  # For non-comparison mode
        'concept1': None,
        'concept2': None,
        'strengths_cfg': [],
        'strengths_cfgpp': [],
        'strengths': [],  # For non-comparison mode
        'is_comparison': False
    }
    
    # Find baseline files
    baseline_cfg_path = results_dir / f"{base_prompt_safe}_baseline_cfg.png"
    baseline_cfgpp_path = results_dir / f"{base_prompt_safe}_baseline_cfgpp.png"
    baseline_path = results_dir / f"{base_prompt_safe}_baseline.png"
    
    if baseline_cfg_path.exists():
        files['baseline_cfg'] = baseline_cfg_path
        files['is_comparison'] = True
    if baseline_cfgpp_path.exists():
        files['baseline_cfgpp'] = baseline_cfgpp_path
        files['is_comparison'] = True
    if baseline_path.exists() and not files['is_comparison']:
        files['baseline'] = baseline_path
    
    # Find concept reference files
    concept1_path = results_dir / f"reference_concept1_{concept1_safe}.png"
    concept2_path = results_dir / f"reference_concept2_{concept2_safe}.png"
    
    if concept1_path.exists():
        files['concept1'] = concept1_path
    if concept2_path.exists():
        files['concept2'] = concept2_path
    
    # Find strength files
    # Pattern: {base_prompt}_strength_{strength}_{method}.png or {base_prompt}_strength_{strength}.png
    pattern_cfg = re.compile(rf"{re.escape(base_prompt_safe)}_strength_([+-]?[\d.]+)_cfg\.png")
    pattern_cfgpp = re.compile(rf"{re.escape(base_prompt_safe)}_strength_([+-]?[\d.]+)_cfg\+\+\.png")
    pattern_generic = re.compile(rf"{re.escape(base_prompt_safe)}_strength_([+-]?[\d.]+)\.png")
    
    for img_path in results_dir.glob(f"{base_prompt_safe}_strength_*.png"):
        filename = img_path.name
        
        # Try CFG pattern
        match = pattern_cfg.match(filename)
        if match:
            strength = float(match.group(1))
            files['strengths_cfg'].append((strength, img_path))
            continue
        
        # Try CFG++ pattern
        match = pattern_cfgpp.match(filename)
        if match:
            strength = float(match.group(1))
            files['strengths_cfgpp'].append((strength, img_path))
            continue
        
        # Try generic pattern (for non-comparison mode)
        match = pattern_generic.match(filename)
        if match:
            strength = float(match.group(1))
            # Only add to generic strengths if we haven't found method-specific ones
            if not files['strengths_cfg'] and not files['strengths_cfgpp']:
                files['strengths'].append((strength, img_path))
    
    # Sort strengths
    files['strengths_cfg'] = sorted(files['strengths_cfg'], key=lambda x: x[0])
    files['strengths_cfgpp'] = sorted(files['strengths_cfgpp'], key=lambda x: x[0])
    files['strengths'] = sorted(files['strengths'], key=lambda x: x[0])
    
    return files


def build_semantic_diff_figure(
    base_prompt: str,
    concept1: str,
    concept2: str,
    files: dict,
    results_dir: Path,
    output_name: str,
    figsize_scale: float = 3.0,
):
    """
    Build and save a figure for semantic difference experiment.
    
    Layout for comparison mode:
    - Row 1 (CFG): baseline, concept1, concept2, strength images
    - Row 2 (CFG++): baseline, concept1, concept2, strength images
    
    Layout for non-comparison mode:
    - Single row: baseline, concept1, concept2, strength images
    """
    is_comparison = files.get('is_comparison', False)
    
    if is_comparison:
        # Comparison mode: 2 rows
        n_strengths_cfg = len(files['strengths_cfg'])
        n_strengths_cfgpp = len(files['strengths_cfgpp'])
        n_base_cols = 3
        n_cols = n_base_cols + max(n_strengths_cfg, n_strengths_cfgpp)
        n_rows = 2
    else:
        # Non-comparison mode: 1 row
        n_strengths = len(files['strengths'])
        n_base_cols = 3
        n_cols = n_base_cols + n_strengths
        n_rows = 1
    
    if n_cols == 0:
        print(f"Warning: No images found for base prompt '{base_prompt}'")
        return
    
    # Create subplot grid
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_scale * n_cols * 1.3, figsize_scale * (2.2 if is_comparison else 1.1)),
        squeeze=False,
    )
    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=-0.7, hspace=0.1)
    
    if is_comparison:
        # COMPARISON MODE: 2 rows
        # Row 1: CFG
        col_idx = 0
        
        # CFG Baseline
        if files['baseline_cfg']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['baseline_cfg']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("CFG\nBaseline", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nBaseline", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load CFG baseline: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # Concept 1
        if files['concept1']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['concept1']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG\nConcept1", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept1", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load concept1: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # Concept 2
        if files['concept2']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['concept2']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG\nConcept2", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept2", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load concept2: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # CFG Strength images
        for strength, img_path in files['strengths_cfg']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG\nstrength={strength:+.2f}", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\n{img_path.name}", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load CFG strength image {img_path}: {e}")
            col_idx += 1
        
        # Hide unused CFG subplots
        for idx in range(col_idx, n_cols):
            axes[0, idx].axis("off")
        
        # Row 2: CFG++
        col_idx = 0
        
        # CFG++ Baseline
        if files['baseline_cfgpp']:
            ax = axes[1, col_idx]
            try:
                img = Image.open(files['baseline_cfgpp']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("CFG++\nBaseline", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nBaseline", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load CFG++ baseline: {e}")
        else:
            axes[1, col_idx].axis("off")
        col_idx += 1
        
        # Concept 1 (same for both methods, but show in CFG++ row too)
        if files['concept1']:
            ax = axes[1, col_idx]
            try:
                img = Image.open(files['concept1']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG++\nConcept1", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept1", fontsize=12)
                ax.axis("off")
        else:
            axes[1, col_idx].axis("off")
        col_idx += 1
        
        # Concept 2
        if files['concept2']:
            ax = axes[1, col_idx]
            try:
                img = Image.open(files['concept2']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG++\nConcept2", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept2", fontsize=12)
                ax.axis("off")
        else:
            axes[1, col_idx].axis("off")
        col_idx += 1
        
        # CFG++ Strength images
        for strength, img_path in files['strengths_cfgpp']:
            ax = axes[1, col_idx]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"CFG++\nstrength={strength:+.2f}", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\n{img_path.name}", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load CFG++ strength image {img_path}: {e}")
            col_idx += 1
        
        # Hide unused CFG++ subplots
        for idx in range(col_idx, n_cols):
            axes[1, idx].axis("off")
    else:
        # NON-COMPARISON MODE: 1 row
        col_idx = 0
        
        # Baseline
        if files['baseline']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['baseline']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Baseline", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nBaseline", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load baseline: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # Concept 1
        if files['concept1']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['concept1']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Concept1", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept1", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load concept1: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # Concept 2
        if files['concept2']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(files['concept2']).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Concept2", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\nConcept2", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load concept2: {e}")
        else:
            axes[0, col_idx].axis("off")
        col_idx += 1
        
        # Strength images
        for strength, img_path in files['strengths']:
            ax = axes[0, col_idx]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"strength={strength:+.2f}", fontsize=14)
            except Exception as e:
                ax.set_title(f"Error\n{img_path.name}", fontsize=12)
                ax.axis("off")
                print(f"Warning: Could not load strength image {img_path}: {e}")
            col_idx += 1
        
        # Hide unused subplots
        for idx in range(col_idx, n_cols):
            axes[0, idx].axis("off")
    
    # Set overall title
    title = f"Base: '{base_prompt}'\nConcepts: '{concept1}' → '{concept2}'"
    plt.suptitle(title, fontsize=16, y=0.99)
    
    # Tight layout with minimal padding
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.0)
    out_path = results_dir / output_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved semantic difference figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create figures for semantic difference experiments"
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("examples/workdir/semantic_diff"),
        help="Workdir used in the semantic difference experiment.",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdirectory under workdir where results are stored.",
    )
    parser.add_argument(
        "--base_prompt",
        type=str,
        default=None,
        help="Base prompt to plot. If not provided, will try to infer from files.",
    )
    parser.add_argument(
        "--concept1",
        type=str,
        default=None,
        help="First concept. If not provided, will try to infer from files.",
    )
    parser.add_argument(
        "--concept2",
        type=str,
        default=None,
        help="Second concept. If not provided, will try to infer from files.",
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

    # Try to infer base_prompt, concept1, concept2 from files if not provided
    if args.base_prompt is None or args.concept1 is None or args.concept2 is None:
        # Look for baseline files to infer base_prompt
        baseline_files = list(results_dir.glob("*_baseline*.png"))
        if baseline_files:
            # Extract base prompt from first baseline file
            baseline_name = baseline_files[0].stem
            # Remove _baseline_cfg or _baseline_cfgpp or _baseline
            base_prompt_safe = baseline_name.replace("_baseline_cfg", "").replace("_baseline_cfgpp", "").replace("_baseline", "")
            args.base_prompt = base_prompt_safe.replace("_", " ")
        
        # Look for concept files
        concept1_files = list(results_dir.glob("reference_concept1_*.png"))
        concept2_files = list(results_dir.glob("reference_concept2_*.png"))
        
        if concept1_files and args.concept1 is None:
            concept1_name = concept1_files[0].stem
            # Extract concept from reference_concept1_{concept}
            match = re.match(r"reference_concept1_(.+)", concept1_name)
            if match:
                args.concept1 = match.group(1).replace("_", " ")
        
        if concept2_files and args.concept2 is None:
            concept2_name = concept2_files[0].stem
            match = re.match(r"reference_concept2_(.+)", concept2_name)
            if match:
                args.concept2 = match.group(1).replace("_", " ")
    
    if args.base_prompt is None:
        raise ValueError("Could not infer base_prompt. Please provide --base_prompt.")
    if args.concept1 is None:
        raise ValueError("Could not infer concept1. Please provide --concept1.")
    if args.concept2 is None:
        raise ValueError("Could not infer concept2. Please provide --concept2.")

    # Find all relevant files
    files = find_semantic_diff_files(results_dir, args.base_prompt, args.concept1, args.concept2)
    
    # Create filename-safe version
    base_prompt_safe = args.base_prompt.replace(" ", "_")
    concept1_safe = args.concept1.replace(" ", "_")
    concept2_safe = args.concept2.replace(" ", "_")
    
    output_name = f"semantic_diff_figure_{base_prompt_safe}_{concept1_safe}_to_{concept2_safe}.png"
    
    build_semantic_diff_figure(
        args.base_prompt,
        args.concept1,
        args.concept2,
        files,
        results_dir,
        output_name,
        figsize_scale=args.figsize_scale,
    )
    
    print(f"\nSemantic difference figure generation complete!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

