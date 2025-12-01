"""
Prompt Interpolation Experiment: Interpolating between multiple prompts in embedding space.

This experiment:
1. Takes multiple prompts (e.g., "a cat", "a dog", "a bird")
2. Computes embeddings for each prompt
3. Interpolates between embeddings using various strategies
4. Generates images to visualize smooth transitions
5. Compares CFG vs CFG++ responses to interpolated embeddings
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


class PromptInterpolationSolver:
    """Wrapper around solver to enable prompt interpolation experiments."""
    
    def __init__(self, base_solver):
        self.solver = base_solver
        self.device = base_solver.device
    
    def get_embedding(self, prompt):
        """Get text embedding for a prompt."""
        null_prompt = ""
        uc, c = self.solver.get_text_embed(null_prompt, prompt)
        return c
    
    def linear_interpolate(self, emb1, emb2, alpha):
        """Linear interpolation between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            alpha: Interpolation factor (0.0 = emb1, 1.0 = emb2)
        """
        return (1 - alpha) * emb1 + alpha * emb2
    
    def slerp_interpolate(self, emb1, emb2, alpha):
        """Spherical linear interpolation (SLERP) between two embeddings.
        
        SLERP provides smoother interpolation on the hypersphere.
        
        Args:
            emb1: First embedding (normalized)
            emb2: Second embedding (normalized)
            alpha: Interpolation factor (0.0 = emb1, 1.0 = emb2)
        """
        # Normalize embeddings
        emb1_norm = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2_norm = emb2 / emb2.norm(dim=-1, keepdim=True)
        
        # Compute dot product (cosine similarity)
        dot = (emb1_norm * emb2_norm).sum(dim=-1, keepdim=True)
        
        # Clamp to avoid numerical issues
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Compute angle
        theta = torch.acos(dot)
        
        # Handle edge cases - check if embeddings are nearly parallel
        sin_theta = torch.sin(theta)
        min_sin_theta = sin_theta.abs().min().item()
        if min_sin_theta < 1e-6:
            # Embeddings are nearly parallel, use linear interpolation
            return self.linear_interpolate(emb1, emb2, alpha)
        
        # SLERP formula - add small epsilon to avoid division by zero
        epsilon = 1e-8
        sin_theta = sin_theta + epsilon
        w1 = torch.sin((1 - alpha) * theta) / sin_theta
        w2 = torch.sin(alpha * theta) / sin_theta
        
        # Scale back to original magnitude
        emb1_mag = emb1.norm(dim=-1, keepdim=True)
        emb2_mag = emb2.norm(dim=-1, keepdim=True)
        mag_interp = (1 - alpha) * emb1_mag + alpha * emb2_mag
        
        result = w1 * emb1 + w2 * emb2
        result = result / result.norm(dim=-1, keepdim=True) * mag_interp
        
        return result
    
    def multi_prompt_blend(self, embeddings, weights):
        """Blend multiple embeddings with given weights.
        
        Args:
            embeddings: List of embeddings
            weights: List of weights (will be normalized to sum to 1)
        """
        weights = torch.tensor(weights, device=embeddings[0].device, dtype=embeddings[0].dtype)
        weights = weights / weights.sum()  # Normalize
        
        result = torch.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            result += w * emb
        
        return result
    
    def sample_with_custom_embedding(self, custom_embedding, null_embedding, cfg_guidance, 
                                     use_cfgpp=False, seed=None):
        """Sample with a custom embedding instead of text prompt.
        
        Args:
            custom_embedding: The modified text embedding to use
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
        custom_embedding = custom_embedding.to(dtype=self.solver.dtype)
        
        # Sampling loop
        pbar = tqdm(self.solver.scheduler.timesteps, desc="SD with interpolated embedding")
        for step, t in enumerate(pbar):
            t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            at = self.solver.alpha(t_int)
            at_prev = self.solver.alpha(t_int - self.solver.skip)
            
            if isinstance(t, torch.Tensor):
                t_float = t.float().to(self.device)
            else:
                t_float = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                noise_uc, noise_c = self.solver.predict_noise(zt, t_float, null_embedding, custom_embedding)
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


def main():
    parser = argparse.ArgumentParser(description="Prompt Interpolation Experiment")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/prompt_interpolation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--use_cfgpp", action="store_true",
                       help="Use CFG++ sampling instead of standard CFG")
    parser.add_argument("--compare_both", action="store_true",
                       help="Generate images with both CFG and CFG++ for comparison")
    
    # Interpolation parameters
    parser.add_argument("--prompts", type=str, nargs='+',
                       default=["a cat", "a dog"],
                       help="Prompts to interpolate between")
    parser.add_argument("--interpolation_method", type=str, 
                       choices=["linear", "slerp", "multi_blend"],
                       default="linear",
                       help="Interpolation method to use")
    parser.add_argument("--interpolation_steps", type=int, default=10,
                       help="Number of interpolation steps (for linear/slerp)")
    parser.add_argument("--weights", type=float, nargs='+', default=None,
                       help="Weights for multi-prompt blending (must match number of prompts)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.interpolation_method == "multi_blend":
        if args.weights is None:
            args.weights = [1.0 / len(args.prompts)] * len(args.prompts)
        elif len(args.weights) != len(args.prompts):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of prompts ({len(args.prompts)})")
    
    # Create workdir
    create_workdir(args.workdir)
    results_dir = args.workdir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Prompt Interpolation Experiment")
    print(f"Prompts: {args.prompts}")
    print(f"Method: {args.interpolation_method}")
    print(f"Steps: {args.interpolation_steps}")
    
    # Initialize solver
    solver_config = munchify({'num_sampling': args.NFE})
    
    if args.model == "sdxl":
        from latent_sdxl import get_solver as get_solver_sdxl
        base_solver = get_solver_sdxl("ddim", solver_config=solver_config, device=args.device)
        print("Warning: SDXL support for custom embeddings may need additional work")
    else:
        base_solver = get_solver("ddim", solver_config=solver_config, device=args.device)
    
    # Create interpolation solver
    interp_solver = PromptInterpolationSolver(base_solver)
    
    # Get embeddings for all prompts
    print("\nComputing embeddings for prompts...")
    embeddings = []
    for prompt in args.prompts:
        emb = interp_solver.get_embedding(prompt)
        embeddings.append(emb)
        print(f"  '{prompt}': shape {emb.shape}, norm {emb.norm().item():.4f}")
    
    # Get null embedding
    null_prompt = "low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,"
    null_emb, _ = base_solver.get_text_embed(null_prompt, "")
    
    # Perform interpolation
    all_results = []
    
    if args.interpolation_method == "linear" or args.interpolation_method == "slerp":
        # Interpolate between first two prompts
        if len(args.prompts) < 2:
            raise ValueError("Need at least 2 prompts for linear/slerp interpolation")
        
        emb1, emb2 = embeddings[0], embeddings[1]
        
        print(f"\nInterpolating between '{args.prompts[0]}' and '{args.prompts[1]}'...")
        
        alphas = np.linspace(0, 1, args.interpolation_steps)
        
        for alpha in alphas:
            print(f"  Alpha: {alpha:.3f}")
            
            # Interpolate
            if args.interpolation_method == "linear":
                interp_emb = interp_solver.linear_interpolate(emb1, emb2, alpha)
            else:  # slerp
                interp_emb = interp_solver.slerp_interpolate(emb1, emb2, alpha)
            
            # Generate image
            try:
                methods_to_test = []
                if args.compare_both:
                    methods_to_test = [("CFG", False), ("CFG++", True)]
                else:
                    methods_to_test = [("CFG++" if args.use_cfgpp else "CFG", args.use_cfgpp)]
                
                for method_name, use_cfgpp in methods_to_test:
                    img = interp_solver.sample_with_custom_embedding(
                        interp_emb,
                        null_emb,
                        cfg_guidance=args.cfg_guidance,
                        use_cfgpp=use_cfgpp,
                        seed=args.seed
                    )
                    
                    # Save image
                    method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
                    img_filename = f"interp_{args.interpolation_method}_alpha_{alpha:.3f}{method_suffix}.png"
                    img_path = results_dir / img_filename
                    save_image(img, img_path, normalize=True)
                    
                    all_results.append({
                        'alpha': float(alpha),
                        'method': method_name,
                        'interpolation_method': args.interpolation_method,
                        'path': str(img_path),
                        'prompt1': args.prompts[0],
                        'prompt2': args.prompts[1]
                    })
                    
                    print(f"    Saved ({method_name}): {img_path}")
            
            except Exception as e:
                print(f"    Error generating image: {e}")
                import traceback
                traceback.print_exc()
    
    elif args.interpolation_method == "multi_blend":
        # Blend multiple prompts
        print(f"\nBlending {len(args.prompts)} prompts with weights {args.weights}...")
        
        blended_emb = interp_solver.multi_prompt_blend(embeddings, args.weights)
        
        try:
            methods_to_test = []
            if args.compare_both:
                methods_to_test = [("CFG", False), ("CFG++", True)]
            else:
                methods_to_test = [("CFG++" if args.use_cfgpp else "CFG", args.use_cfgpp)]
            
            for method_name, use_cfgpp in methods_to_test:
                img = interp_solver.sample_with_custom_embedding(
                    blended_emb,
                    null_emb,
                    cfg_guidance=args.cfg_guidance,
                    use_cfgpp=use_cfgpp,
                    seed=args.seed
                )
                
                # Save image
                method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
                weights_str = "_".join([f"{w:.2f}" for w in args.weights])
                img_filename = f"blend_weights_{weights_str}{method_suffix}.png"
                img_path = results_dir / img_filename
                save_image(img, img_path, normalize=True)
                
                all_results.append({
                    'weights': args.weights,
                    'method': method_name,
                    'interpolation_method': args.interpolation_method,
                    'path': str(img_path),
                    'prompts': args.prompts
                })
                
                print(f"  Saved ({method_name}): {img_path}")
        
        except Exception as e:
            print(f"  Error generating image: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate reference images for individual prompts
    print("\nGenerating reference images for individual prompts...")
    for prompt, emb in zip(args.prompts, embeddings):
        try:
            methods_to_test = []
            if args.compare_both:
                methods_to_test = [("CFG", False), ("CFG++", True)]
            else:
                methods_to_test = [("CFG++" if args.use_cfgpp else "CFG", args.use_cfgpp)]
            
            for method_name, use_cfgpp in methods_to_test:
                img = interp_solver.sample_with_custom_embedding(
                    emb,
                    null_emb,
                    cfg_guidance=args.cfg_guidance,
                    use_cfgpp=use_cfgpp,
                    seed=args.seed
                )
                
                method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
                ref_path = results_dir / f"reference_{prompt.replace(' ', '_')}{method_suffix}.png"
                save_image(img, ref_path, normalize=True)
                print(f"  Saved reference ({method_name}): {ref_path}")
        except Exception as e:
            print(f"  Error generating reference for '{prompt}': {e}")
    
    # Create comparison grid
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        grid_images = []
        
        # Add reference images
        for prompt in args.prompts:
            ref_path = results_dir / f"reference_{prompt.replace(' ', '_')}.png"
            if ref_path.exists():
                img = transforms.ToTensor()(Image.open(ref_path))
                grid_images.append(img)
        
        # Add interpolated images
        if args.interpolation_method in ["linear", "slerp"]:
            alphas = np.linspace(0, 1, args.interpolation_steps)
            for alpha in alphas:
                img_path = results_dir / f"interp_{args.interpolation_method}_alpha_{alpha:.3f}.png"
                if img_path.exists():
                    img = transforms.ToTensor()(Image.open(img_path))
                    grid_images.append(img)
        
        if grid_images:
            grid = make_grid(grid_images, nrow=len(grid_images))
            grid_path = results_dir / f"interpolation_comparison_{args.interpolation_method}.png"
            save_image(grid, grid_path, normalize=True)
            print(f"\nSaved comparison grid: {grid_path}")
    except Exception as e:
        print(f"Error creating grid: {e}")
    
    # Save results summary
    summary = {
        'prompts': args.prompts,
        'interpolation_method': args.interpolation_method,
        'interpolation_steps': args.interpolation_steps,
        'weights': args.weights,
        'results': all_results,
        'config': {
            'model': args.model,
            'NFE': args.NFE,
            'seed': args.seed,
            'cfg_guidance': args.cfg_guidance,
            'use_cfgpp': args.use_cfgpp,
            'compare_both': args.compare_both
        }
    }
    
    summary_path = results_dir / "interpolation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

