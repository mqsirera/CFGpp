"""
Evaluation script to compare CFG vs CFG++ on multiple prompts and guidance scales.
This script generates images for both methods and saves them for comparison.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
from munch import munchify
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from latent_diffusion import get_solver
from utils.log_util import create_workdir, set_seed


def evaluate_method(solver, prompt_pair, cfg_guidance, method_name, seed, device):
    """Evaluate a single method with given parameters."""
    set_seed(seed)
    
    null_prompt, prompt = prompt_pair
    result = solver.sample(
        prompt=[null_prompt, prompt],
        cfg_guidance=cfg_guidance,
        callback_fn=None
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare CFG vs CFG++ on multiple prompts")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts_file", type=Path, default=None, 
                       help="JSON file with list of prompts. If not provided, uses default prompts.")
    parser.add_argument("--cfg_scales", type=float, nargs='+', 
                       default=[1.0, 2.5, 5.0, 7.5, 10.0],
                       help="CFG guidance scales to test for standard CFG")
    parser.add_argument("--cfgpp_scales", type=float, nargs='+',
                       default=[0.1, 0.3, 0.5, 0.7, 1.0],
                       help="CFG++ guidance scales to test")
    parser.add_argument("--null_prompt", type=str, 
                       default="low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,")
    
    args = parser.parse_args()
    
    # Create workdir
    create_workdir(args.workdir)
    results_dir = args.workdir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load prompts
    if args.prompts_file and args.prompts_file.exists():
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
            if isinstance(prompts_data, list):
                prompts = prompts_data
            elif isinstance(prompts_data, dict) and 'prompts' in prompts_data:
                prompts = prompts_data['prompts']
            else:
                prompts = [prompts_data]
    else:
        # Default test prompts
        prompts = [
            "a portrait of a dog",
            "a beautiful landscape with mountains and a lake",
            "a futuristic city at night",
            "a cat sitting on a windowsill",
            "a vintage car on a country road",
            "a portrait of a person",
            "a bowl of fruit on a table",
            "a sunset over the ocean",
        ]
    
    print(f"Testing {len(prompts)} prompts")
    print(f"CFG scales: {args.cfg_scales}")
    print(f"CFG++ scales: {args.cfgpp_scales}")
    
    # Initialize solvers
    solver_config = munchify({'num_sampling': args.NFE})
    
    if args.model == "sdxl":
        from latent_sdxl import get_solver as get_solver_sdxl
        solver_cfg = get_solver_sdxl("ddim", solver_config=solver_config, device=args.device)
        solver_cfgpp = get_solver_sdxl("ddim_cfg++", solver_config=solver_config, device=args.device)
    else:
        solver_cfg = get_solver("ddim", solver_config=solver_config, device=args.device)
        solver_cfgpp = get_solver("ddim_cfg++", solver_config=solver_config, device=args.device)
    
    # Store results
    all_results = []
    
    # Evaluate each prompt
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        prompt_results = {
            'prompt': prompt,
            'prompt_idx': prompt_idx,
            'cfg_results': {},
            'cfgpp_results': {}
        }
        
        # Test CFG with different scales
        for cfg_scale in args.cfg_scales:
            try:
                img = evaluate_method(
                    solver_cfg, 
                    [args.null_prompt, prompt],
                    cfg_scale,
                    "ddim",
                    args.seed + prompt_idx,
                    args.device
                )
                
                # Save individual image
                img_path = results_dir / f"prompt_{prompt_idx:02d}_cfg_{cfg_scale:.1f}.png"
                save_image(img, img_path, normalize=True)
                
                prompt_results['cfg_results'][f'{cfg_scale:.1f}'] = str(img_path)
                all_results.append({
                    'prompt': prompt,
                    'method': 'CFG',
                    'guidance': cfg_scale,
                    'path': str(img_path)
                })
            except Exception as e:
                print(f"Error with CFG scale {cfg_scale} for prompt '{prompt}': {e}")
        
        # Test CFG++ with different scales
        for cfgpp_scale in args.cfgpp_scales:
            try:
                img = evaluate_method(
                    solver_cfgpp,
                    [args.null_prompt, prompt],
                    cfgpp_scale,
                    "ddim_cfg++",
                    args.seed + prompt_idx,
                    args.device
                )
                
                # Save individual image
                img_path = results_dir / f"prompt_{prompt_idx:02d}_cfgpp_{cfgpp_scale:.1f}.png"
                save_image(img, img_path, normalize=True)
                
                prompt_results['cfgpp_results'][f'{cfgpp_scale:.1f}'] = str(img_path)
                all_results.append({
                    'prompt': prompt,
                    'method': 'CFG++',
                    'guidance': cfgpp_scale,
                    'path': str(img_path)
                })
            except Exception as e:
                print(f"Error with CFG++ scale {cfgpp_scale} for prompt '{prompt}': {e}")
        
        # Create comparison grid for this prompt
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Collect images for this prompt
            grid_images = []
            
            # CFG images
            for cfg_scale in args.cfg_scales:
                img_path = results_dir / f"prompt_{prompt_idx:02d}_cfg_{cfg_scale:.1f}.png"
                if img_path.exists():
                    try:
                        img = transforms.ToTensor()(Image.open(img_path))
                        grid_images.append(img)
                    except Exception as e:
                        print(f"  Warning: Could not load {img_path}: {e}")
            
            # CFG++ images
            for cfgpp_scale in args.cfgpp_scales:
                img_path = results_dir / f"prompt_{prompt_idx:02d}_cfgpp_{cfgpp_scale:.1f}.png"
                if img_path.exists():
                    try:
                        img = transforms.ToTensor()(Image.open(img_path))
                        grid_images.append(img)
                    except Exception as e:
                        print(f"  Warning: Could not load {img_path}: {e}")
            
            if grid_images:
                n_cols = len(args.cfg_scales) + len(args.cfgpp_scales)
                grid = make_grid(grid_images, nrow=n_cols)
                grid_path = results_dir / f"prompt_{prompt_idx:02d}_comparison_grid.png"
                save_image(grid, grid_path, normalize=True)
                print(f"  Saved comparison grid: {grid_path}")
        except Exception as e:
            print(f"  Error creating grid for prompt {prompt_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    summary = {
        'prompts': prompts,
        'cfg_scales': args.cfg_scales,
        'cfgpp_scales': args.cfgpp_scales,
        'results': all_results,
        'config': {
            'model': args.model,
            'NFE': args.NFE,
            'seed': args.seed,
            'null_prompt': args.null_prompt
        }
    }
    
    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nTotal images generated: {len(all_results)}")


if __name__ == "__main__":
    main()

