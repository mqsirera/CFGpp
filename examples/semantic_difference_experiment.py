"""
Semantic Difference Experiment: Manipulating embeddings to modify identity.

This experiment:
1. Computes embeddings for semantic concepts (e.g., "girl" vs "boy")
2. Calculates the difference vector between them
3. Applies this difference to base prompts to see if identity changes
4. Generates images to visualize the effect
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from munch import munchify
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from latent_diffusion import get_solver
from utils.log_util import create_workdir, set_seed


class SemanticDifferenceSolver:
    """Wrapper around solver to allow custom embedding manipulation."""
    
    def __init__(self, base_solver):
        self.solver = base_solver
        self.device = base_solver.device
    
    def get_embedding(self, prompt):
        """Get text embedding for a prompt."""
        null_prompt = ""
        uc, c = self.solver.get_text_embed(null_prompt, prompt)
        return c
    
    def compute_semantic_difference(self, concept1, concept2):
        """Compute the semantic difference vector between two concepts."""
        emb1 = self.get_embedding(concept1)
        emb2 = self.get_embedding(concept2)
        diff = emb2 - emb1
        return diff, emb1, emb2
    
    def apply_semantic_difference(self, base_prompt, difference_vector, strength=1.0):
        """Apply a semantic difference vector to a base prompt embedding."""
        base_emb = self.get_embedding(base_prompt)
        modified_emb = base_emb + strength * difference_vector
        return modified_emb
    
    def sample_with_custom_embedding(self, custom_embedding, null_embedding, cfg_guidance, 
                                     use_cfgpp=False, seed=None):
        """Sample with a custom embedding instead of text prompt.
        
        Args:
            custom_embedding: The modified text embedding to use
            null_embedding: The null/negative embedding
            cfg_guidance: Guidance scale
            use_cfgpp: If True, use CFG++ sampling (uses null embedding for noise addition)
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
        
        # Sampling loop (similar to BaseDDIM but with custom embeddings)
        pbar = tqdm(self.solver.scheduler.timesteps, desc="SD with custom embedding")
        for step, t in enumerate(pbar):
            at = self.solver.alpha(t)
            at_prev = self.solver.alpha(t - self.solver.skip)
            
            with torch.no_grad():
                noise_uc, noise_c = self.solver.predict_noise(zt, t, null_embedding, custom_embedding)
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
    parser = argparse.ArgumentParser(description="Semantic Difference Experiment")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/semantic_diff")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--use_cfgpp", action="store_true",
                       help="Use CFG++ sampling instead of standard CFG")
    parser.add_argument("--compare_both", action="store_true",
                       help="Generate images with both CFG and CFG++ for comparison")
    
    # Semantic difference parameters
    parser.add_argument("--concept1", type=str, default="a girl", 
                       help="First concept for computing difference")
    parser.add_argument("--concept2", type=str, default="a boy",
                       help="Second concept for computing difference")
    parser.add_argument("--base_prompts", type=str, nargs='+',
                       default=["a portrait of a person", "a person standing", "a person walking"],
                       help="Base prompts to apply the semantic difference to")
    parser.add_argument("--strengths", type=float, nargs='+',
                       default=[-1.0, -0.5, 0.0, 0.5, 1.0],
                       help="Strengths to apply the semantic difference (negative = concept1, positive = concept2)")
    
    args = parser.parse_args()
    
    # Create workdir
    create_workdir(args.workdir)
    results_dir = args.workdir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Semantic Difference Experiment")
    print(f"Concept 1: '{args.concept1}'")
    print(f"Concept 2: '{args.concept2}'")
    print(f"Base prompts: {args.base_prompts}")
    print(f"Strengths: {args.strengths}")
    
    # Initialize solver
    solver_config = munchify({'num_sampling': args.NFE})
    
    if args.model == "sdxl":
        from latent_sdxl import get_solver as get_solver_sdxl
        base_solver = get_solver_sdxl("ddim", solver_config=solver_config, device=args.device)
        print("Warning: SDXL support for custom embeddings may need additional work")
    else:
        base_solver = get_solver("ddim", solver_config=solver_config, device=args.device)
    
    # Create semantic difference solver
    semantic_solver = SemanticDifferenceSolver(base_solver)
    
    # Compute semantic difference
    print("\nComputing semantic difference vector...")
    diff_vector, emb1, emb2 = semantic_solver.compute_semantic_difference(
        args.concept1, 
        args.concept2
    )
    print(f"Difference vector shape: {diff_vector.shape}")
    print(f"Difference vector norm: {diff_vector.norm().item():.4f}")
    
    # Get null embedding
    null_prompt = "low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,"
    null_emb, _ = base_solver.get_text_embed(null_prompt, "")
    
    # Test each base prompt
    all_results = []
    
    for base_prompt in args.base_prompts:
        print(f"\nProcessing base prompt: '{base_prompt}'")
        prompt_results = []
        
        for strength in args.strengths:
            print(f"  Strength: {strength:.2f}")
            
            # Apply semantic difference
            modified_emb = semantic_solver.apply_semantic_difference(
                base_prompt, 
                diff_vector, 
                strength=strength
            )
            
            # Generate image with modified embedding
            try:
                methods_to_test = []
                if args.compare_both:
                    methods_to_test = [("CFG", False), ("CFG++", True)]
                else:
                    methods_to_test = [("CFG++" if args.use_cfgpp else "CFG", args.use_cfgpp)]
                
                for method_name, use_cfgpp in methods_to_test:
                    img = semantic_solver.sample_with_custom_embedding(
                        modified_emb,
                        null_emb,
                        cfg_guidance=args.cfg_guidance,
                        use_cfgpp=use_cfgpp,
                        seed=args.seed
                    )
                    
                    # Save image
                    method_suffix = f"_{method_name.lower()}" if args.compare_both else ""
                    img_filename = f"{base_prompt.replace(' ', '_')}_strength_{strength:+.2f}{method_suffix}.png"
                    img_path = results_dir / img_filename
                    save_image(img, img_path, normalize=True)
                    
                    prompt_results.append({
                        'strength': strength,
                        'method': method_name,
                        'path': str(img_path),
                        'concept': args.concept2 if strength > 0 else args.concept1
                    })
                    
                    print(f"    Saved ({method_name}): {img_path}")
                
            except Exception as e:
                print(f"    Error generating image: {e}")
                import traceback
                traceback.print_exc()
        
        # Also generate baseline (no modification)
        print(f"  Generating baseline (standard prompt)...")
        try:
            if args.compare_both:
                # Generate with both methods
                for method_name, use_cfgpp in [("CFG", False), ("CFG++", True)]:
                    if use_cfgpp:
                        from latent_diffusion import get_solver
                        cfgpp_solver = get_solver("ddim_cfg++", 
                                                 solver_config=solver_config, 
                                                 device=args.device)
                        baseline_img = cfgpp_solver.sample(
                            prompt=[null_prompt, base_prompt],
                            cfg_guidance=args.cfg_guidance,
                            callback_fn=None
                        )
                    else:
                        baseline_img = base_solver.sample(
                            prompt=[null_prompt, base_prompt],
                            cfg_guidance=args.cfg_guidance,
                            callback_fn=None
                        )
                    baseline_path = results_dir / f"{base_prompt.replace(' ', '_')}_baseline_{method_name.lower()}.png"
                    save_image(baseline_img, baseline_path, normalize=True)
                    print(f"    Saved baseline ({method_name}): {baseline_path}")
            else:
                if args.use_cfgpp:
                    from latent_diffusion import get_solver
                    cfgpp_solver = get_solver("ddim_cfg++", 
                                             solver_config=solver_config, 
                                             device=args.device)
                    baseline_img = cfgpp_solver.sample(
                        prompt=[null_prompt, base_prompt],
                        cfg_guidance=args.cfg_guidance,
                        callback_fn=None
                    )
                else:
                    baseline_img = base_solver.sample(
                        prompt=[null_prompt, base_prompt],
                        cfg_guidance=args.cfg_guidance,
                        callback_fn=None
                    )
                baseline_path = results_dir / f"{base_prompt.replace(' ', '_')}_baseline.png"
                save_image(baseline_img, baseline_path, normalize=True)
                print(f"    Saved baseline: {baseline_path}")
        except Exception as e:
            print(f"    Error generating baseline: {e}")
            import traceback
            traceback.print_exc()
        
        # Create comparison grid for this prompt
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            grid_images = []
            grid_labels = []
            
            # Baseline
            baseline_path = results_dir / f"{base_prompt.replace(' ', '_')}_baseline.png"
            if baseline_path.exists():
                img = transforms.ToTensor()(Image.open(baseline_path))
                grid_images.append(img)
                grid_labels.append("Baseline")
            
            # Modified images
            for result in prompt_results:
                img_path = Path(result['path'])
                if img_path.exists():
                    img = transforms.ToTensor()(Image.open(img_path))
                    grid_images.append(img)
                    grid_labels.append(f"Strength {result['strength']:+.2f}")
            
            if grid_images:
                grid = make_grid(grid_images, nrow=len(grid_images))
                grid_path = results_dir / f"{base_prompt.replace(' ', '_')}_comparison.png"
                save_image(grid, grid_path, normalize=True)
                print(f"  Saved comparison grid: {grid_path}")
        except Exception as e:
            print(f"  Error creating grid: {e}")
        
        all_results.append({
            'base_prompt': base_prompt,
            'results': prompt_results
        })
    
    # Generate reference images for the concepts themselves
    print("\nGenerating reference images for concepts...")
    for concept, label in [(args.concept1, "concept1"), (args.concept2, "concept2")]:
        try:
            ref_img = base_solver.sample(
                prompt=[null_prompt, concept],
                cfg_guidance=args.cfg_guidance,
                callback_fn=None
            )
            ref_path = results_dir / f"reference_{label}_{concept.replace(' ', '_')}.png"
            save_image(ref_img, ref_path, normalize=True)
            print(f"  Saved reference: {ref_path}")
        except Exception as e:
            print(f"  Error generating reference for {concept}: {e}")
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {results_dir}")
    print(f"\nSummary:")
    print(f"  Semantic difference: '{args.concept1}' -> '{args.concept2}'")
    print(f"  Tested {len(args.base_prompts)} base prompts")
    print(f"  Tested {len(args.strengths)} strength values")


if __name__ == "__main__":
    main()

