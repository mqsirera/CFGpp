"""
Timestep-Dependent Prompt Conditioning Experiment: Using different prompts at different timesteps.

This experiment:
1. Defines prompt schedules that vary over timesteps
2. Uses different prompts or prompt weights at different stages of sampling
3. Tests strategies like progressive refinement (coarse -> fine)
4. Compares CFG vs CFG++ responses to time-varying conditioning
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from munch import munchify
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from latent_diffusion import get_solver
from utils.log_util import create_workdir, set_seed


class TimestepConditioningSolver:
    """Wrapper around solver to enable timestep-dependent prompt conditioning."""
    
    def __init__(self, base_solver):
        self.solver = base_solver
        self.device = base_solver.device
    
    def get_embedding(self, prompt):
        """Get text embedding for a prompt."""
        null_prompt = ""
        uc, c = self.solver.get_text_embed(null_prompt, prompt)
        return c
    
    def sample_with_timestep_conditioning(self, prompt_schedule_fn, null_embedding, cfg_guidance, 
                                          use_cfgpp=False, seed=None):
        """Sample with timestep-dependent prompt conditioning.
        
        Args:
            prompt_schedule_fn: Function that takes (step, t, total_steps) and returns embedding
            null_embedding: The null/negative embedding
            cfg_guidance: Guidance scale
            use_cfgpp: If True, use CFG++ sampling
            seed: Random seed
        """
        if seed is not None:
            set_seed(seed)
        
        # Initialize zT
        zt = self.solver.initialize_latent()
        zt = zt.requires_grad_()
        
        # Get null embedding if not provided
        if null_embedding is None:
            null_prompt = "low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,"
            null_embedding, _ = self.solver.get_text_embed(null_prompt, "")
        
        # Ensure all tensors are in the correct dtype
        zt = zt.to(dtype=self.solver.dtype)
        null_embedding = null_embedding.to(dtype=self.solver.dtype)
        
        total_steps = len(self.solver.scheduler.timesteps)
        
        # Sampling loop
        pbar = tqdm(self.solver.scheduler.timesteps, desc="SD with timestep conditioning")
        for step, t in enumerate(pbar):
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            at = self.solver.alpha(t_int)
            at_prev = self.solver.alpha(t_int - self.solver.skip)
            
            if isinstance(t, torch.Tensor):
                t_float = t.float().to(self.device)
            else:
                t_float = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            # Get timestep-dependent embedding
            prompt_embedding = prompt_schedule_fn(step, t_int, total_steps)
            prompt_embedding = prompt_embedding.to(dtype=self.solver.dtype)
            
            with torch.no_grad():
                noise_uc, noise_c = self.solver.predict_noise(zt, t_float, null_embedding, prompt_embedding)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            
            # add noise - CFG++ uses null embedding, CFG uses guided prediction
            if use_cfgpp:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
            else:
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
        # Decode
        img = self.solver.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


def create_progressive_refinement_schedule(emb_coarse, emb_fine, transition_start=0.3, transition_end=0.7):
    """Create a schedule that transitions from coarse to fine prompt.
    
    Args:
        emb_coarse: Embedding for coarse/simple prompt
        emb_fine: Embedding for fine/detailed prompt
        transition_start: Fraction of steps to start transition (0-1)
        transition_end: Fraction of steps to end transition (0-1)
    """
    def schedule_fn(step, t, total_steps):
        progress = step / total_steps
        
        if progress < transition_start:
            return emb_coarse
        elif progress > transition_end:
            return emb_fine
        else:
            # Linear interpolation during transition
            alpha = (progress - transition_start) / (transition_end - transition_start)
            return (1 - alpha) * emb_coarse + alpha * emb_fine
    
    return schedule_fn


def create_style_content_schedule(emb_content, emb_style, content_weight_start=1.0, content_weight_end=0.3):
    """Create a schedule that transitions from content to style.
    
    Args:
        emb_content: Embedding for content prompt
        emb_style: Embedding for style prompt
        content_weight_start: Weight for content at start (0-1)
        content_weight_end: Weight for content at end (0-1)
    """
    def schedule_fn(step, t, total_steps):
        progress = step / total_steps
        content_weight = content_weight_start + (content_weight_end - content_weight_start) * progress
        style_weight = 1 - content_weight
        
        return content_weight * emb_content + style_weight * emb_style
    
    return schedule_fn


def create_multi_prompt_schedule(embeddings, weights_schedule):
    """Create a schedule that blends multiple prompts with time-varying weights.
    
    Args:
        embeddings: List of embeddings
        weights_schedule: Function that takes (step, t, total_steps) and returns list of weights
    """
    def schedule_fn(step, t, total_steps):
        weights = weights_schedule(step, t, total_steps)
        weights = torch.tensor(weights, device=embeddings[0].device, dtype=embeddings[0].dtype)
        weights = weights / weights.sum()  # Normalize
        
        result = torch.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            result += w * emb
        
        return result
    
    return schedule_fn


def create_negative_prompt_schedule(emb_positive, emb_negative, negative_strength_start=0.0, negative_strength_end=1.0):
    """Create a schedule that gradually increases negative prompt influence.
    
    Args:
        emb_positive: Positive prompt embedding
        emb_negative: Negative prompt embedding
        negative_strength_start: Negative strength at start
        negative_strength_end: Negative strength at end
    """
    def schedule_fn(step, t, total_steps):
        progress = step / total_steps
        negative_strength = negative_strength_start + (negative_strength_end - negative_strength_start) * progress
        
        # Blend: positive - negative_strength * negative
        return emb_positive - negative_strength * (emb_negative - emb_positive)
    
    return schedule_fn


def main():
    parser = argparse.ArgumentParser(description="Timestep-Dependent Prompt Conditioning Experiment")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/timestep_conditioning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--use_cfgpp", action="store_true",
                       help="Use CFG++ sampling instead of standard CFG")
    parser.add_argument("--compare_both", action="store_true",
                       help="Generate images with both CFG and CFG++ for comparison")
    
    # Schedule parameters
    parser.add_argument("--schedule_type", type=str,
                       choices=["progressive", "style_content", "multi_prompt", "negative"],
                       default="progressive",
                       help="Type of timestep schedule")
    
    # Progressive refinement
    parser.add_argument("--coarse_prompt", type=str, default="a person",
                       help="Coarse/simple prompt for progressive refinement")
    parser.add_argument("--fine_prompt", type=str, default="a person with blue eyes, wearing a red shirt, smiling",
                       help="Fine/detailed prompt for progressive refinement")
    parser.add_argument("--transition_start", type=float, default=0.3,
                       help="Fraction of steps to start transition (0-1)")
    parser.add_argument("--transition_end", type=float, default=0.7,
                       help="Fraction of steps to end transition (0-1)")
    
    # Style-content
    parser.add_argument("--content_prompt", type=str, default="a cat",
                       help="Content prompt")
    parser.add_argument("--style_prompt", type=str, default="in the style of Van Gogh",
                       help="Style prompt")
    parser.add_argument("--content_weight_start", type=float, default=1.0,
                       help="Content weight at start")
    parser.add_argument("--content_weight_end", type=float, default=0.3,
                       help="Content weight at end")
    
    # Multi-prompt (simplified - uses equal weights with linear transition)
    parser.add_argument("--prompts", type=str, nargs='+', default=None,
                       help="Multiple prompts for multi-prompt schedule")
    
    # Negative prompt
    parser.add_argument("--positive_prompt", type=str, default="a beautiful landscape",
                       help="Positive prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality",
                       help="Negative prompt")
    parser.add_argument("--negative_strength_start", type=float, default=0.0,
                       help="Negative strength at start")
    parser.add_argument("--negative_strength_end", type=float, default=1.0,
                       help="Negative strength at end")
    
    args = parser.parse_args()
    
    # Create workdir
    create_workdir(args.workdir)
    results_dir = args.workdir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Timestep-Dependent Prompt Conditioning Experiment")
    print(f"Schedule type: {args.schedule_type}")
    
    # Initialize solver
    solver_config = munchify({'num_sampling': args.NFE})
    
    if args.model == "sdxl":
        from latent_sdxl import get_solver as get_solver_sdxl
        base_solver = get_solver_sdxl("ddim", solver_config=solver_config, device=args.device)
        print("Warning: SDXL support for custom embeddings may need additional work")
    else:
        base_solver = get_solver("ddim", solver_config=solver_config, device=args.device)
    
    # Create timestep conditioning solver
    timestep_solver = TimestepConditioningSolver(base_solver)
    
    # Get null embedding
    null_prompt = "low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,"
    null_emb, _ = base_solver.get_text_embed(null_prompt, "")
    
    # Create schedule based on type
    if args.schedule_type == "progressive":
        print(f"Coarse prompt: '{args.coarse_prompt}'")
        print(f"Fine prompt: '{args.fine_prompt}'")
        emb_coarse = timestep_solver.get_embedding(args.coarse_prompt)
        emb_fine = timestep_solver.get_embedding(args.fine_prompt)
        schedule_fn = create_progressive_refinement_schedule(
            emb_coarse, emb_fine, 
            args.transition_start, args.transition_end
        )
        schedule_name = f"progressive_{args.coarse_prompt.replace(' ', '_')}_to_{args.fine_prompt.replace(' ', '_')}"
    
    elif args.schedule_type == "style_content":
        print(f"Content prompt: '{args.content_prompt}'")
        print(f"Style prompt: '{args.style_prompt}'")
        emb_content = timestep_solver.get_embedding(args.content_prompt)
        emb_style = timestep_solver.get_embedding(args.style_prompt)
        schedule_fn = create_style_content_schedule(
            emb_content, emb_style,
            args.content_weight_start, args.content_weight_end
        )
        schedule_name = f"style_content_{args.content_prompt.replace(' ', '_')}"
    
    elif args.schedule_type == "multi_prompt":
        if args.prompts is None or len(args.prompts) < 2:
            raise ValueError("Need at least 2 prompts for multi-prompt schedule")
        print(f"Prompts: {args.prompts}")
        embeddings = [timestep_solver.get_embedding(p) for p in args.prompts]
        
        # Simple schedule: transition from first to last prompt
        def weights_schedule(step, t, total_steps):
            progress = step / total_steps
            weights = [0.0] * len(embeddings)
            if len(embeddings) == 2:
                weights[0] = 1 - progress
                weights[1] = progress
            else:
                # For more prompts, use linear interpolation
                segment = progress * (len(embeddings) - 1)
                idx = int(segment)
                alpha = segment - idx
                if idx < len(embeddings) - 1:
                    weights[idx] = 1 - alpha
                    weights[idx + 1] = alpha
                else:
                    weights[-1] = 1.0
            return weights
        
        schedule_fn = create_multi_prompt_schedule(embeddings, weights_schedule)
        schedule_name = f"multi_prompt_{len(args.prompts)}prompts"
    
    elif args.schedule_type == "negative":
        print(f"Positive prompt: '{args.positive_prompt}'")
        print(f"Negative prompt: '{args.negative_prompt}'")
        emb_positive = timestep_solver.get_embedding(args.positive_prompt)
        emb_negative = timestep_solver.get_embedding(args.negative_prompt)
        schedule_fn = create_negative_prompt_schedule(
            emb_positive, emb_negative,
            args.negative_strength_start, args.negative_strength_end
        )
        schedule_name = f"negative_{args.positive_prompt.replace(' ', '_')}"
    
    # Generate images
    all_results = []
    
    try:
        methods_to_test = []
        if args.compare_both:
            methods_to_test = [("CFG", False), ("CFG++", True)]
        else:
            methods_to_test = [("CFG++" if args.use_cfgpp else "CFG", args.use_cfgpp)]
        
        for method_name, use_cfgpp in methods_to_test:
            print(f"\nGenerating with {method_name}...")
            img = timestep_solver.sample_with_timestep_conditioning(
                schedule_fn,
                null_emb,
                cfg_guidance=args.cfg_guidance,
                use_cfgpp=use_cfgpp,
                seed=args.seed
            )
            
            # Save image
            method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
            img_filename = f"{schedule_name}{method_suffix}.png"
            img_path = results_dir / img_filename
            save_image(img, img_path, normalize=True)
            
            all_results.append({
                'schedule_type': args.schedule_type,
                'method': method_name,
                'path': str(img_path),
                'config': vars(args)
            })
            
            print(f"  Saved ({method_name}): {img_path}")
    
    except Exception as e:
        print(f"  Error generating image: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate baseline (constant prompt) for comparison
    print("\nGenerating baseline (constant prompt) for comparison...")
    try:
        if args.schedule_type == "progressive":
            baseline_prompt = args.fine_prompt
        elif args.schedule_type == "style_content":
            baseline_prompt = args.content_prompt
        elif args.schedule_type == "multi_prompt":
            baseline_prompt = args.prompts[-1] if args.prompts else args.content_prompt
        else:
            baseline_prompt = args.positive_prompt
        
        baseline_emb = timestep_solver.get_embedding(baseline_prompt)
        
        # Constant schedule
        def constant_schedule(step, t, total_steps):
            return baseline_emb
        
        for method_name, use_cfgpp in methods_to_test:
            baseline_img = timestep_solver.sample_with_timestep_conditioning(
                constant_schedule,
                null_emb,
                cfg_guidance=args.cfg_guidance,
                use_cfgpp=use_cfgpp,
                seed=args.seed
            )
            
            method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
            baseline_path = results_dir / f"baseline_{baseline_prompt.replace(' ', '_')}{method_suffix}.png"
            save_image(baseline_img, baseline_path, normalize=True)
            print(f"  Saved baseline ({method_name}): {baseline_path}")
    except Exception as e:
        print(f"  Error generating baseline: {e}")
    
    # Save results summary
    summary = {
        'schedule_type': args.schedule_type,
        'results': all_results,
        'config': vars(args)
    }
    
    summary_path = results_dir / "timestep_conditioning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()


