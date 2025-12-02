import argparse
from pathlib import Path
import json
from typing import Dict, Any, List

import torch
from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback # <-- Only import what exists
from utils.log_util import create_workdir, set_seed

# --- Global Target Timesteps for Montage ---
# We use this list to determine which steps to save for the visual comparison.
TARGET_TIMESTEPS = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0]

# --- Loss Calculation Utility ---
# (Omitted for brevity, assumed correct)

# --- Custom Callback for Data Collection ---
class TrajectoryCallback:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.log_data = []

        self.workdir.joinpath("trajectory").mkdir(parents=True, exist_ok=True)
        
        # 🎯 ADDED: Set of target times and a flag to track if they've been saved
        self.target_times = set(TARGET_TIMESTEPS)
        self.saved_times = set()

    def __call__(self, step: int, t: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        z0t = kwargs['z0t']
        # zt = kwargs['zt'] # Not used for saving, can be ignored here
        decode_fn = kwargs['decode']
        
        # 1. Trajectory Snapshot (save images)
        # Check if the current time t is close to any desired target time.
        # We check within 20 steps (the interval size for NFE=50) to ensure we catch it.
        
        should_save = False
        
        for target_t in self.target_times:
            if target_t in self.saved_times:
                continue
                
            # Check if current t is close to the target, and we haven't saved it yet
            if abs(t - target_t) <= 20 or t == 0:
                should_save = True
                self.saved_times.add(target_t)
                break
        
        if should_save: 
            img = decode_fn(z0t)
            # 🚨 CRITICAL: We save using the ACTUAL timestep t, 
            # so the plotting script can find the file using the t-value in the name.
            save_image(img.cpu(), self.workdir.joinpath(f'trajectory/t_{t:04d}_step_{step:03d}.png'), normalize=True)
            
        # 2. Score Matching Loss Plot Data Collection
        noise_diff = kwargs['noise_diff']
        
        # Calculate L2 norm, convert to a single item(), and ensure it's a float before squaring
        loss_metric_value = torch.linalg.norm(noise_diff).item()
        loss_metric_squared = loss_metric_value**2
        
        self.log_data.append({
            't': int(t), # Explicitly convert timestep to int just in case
            'step': step,
            'metric': float(loss_metric_squared) # Guarantee a standard Python float
        })
        
        return kwargs

def run_trajectory_experiment(args):
    # Setup
    set_seed(args.seed)
    # Ensure the trajectory folder exists for image saving
    # Path(args.workdir).joinpath("trajectory").mkdir(parents=True, exist_ok=True) 

    # --- Experiment Configuration ---
    # CFG: High guidance, extrapolation, expected artifacts/instability
    cfg_params = [
        {'solver_name': 'ddim', 'cfg_guidance': 12.5, 'label': 'CFG (w=7.5)'}
    ]
    # CFG++: Interpolation, expected smooth trajectory/stability
    cfgpp_params = [
        {'solver_name': 'ddim_cfg++', 'cfg_guidance': 1.0, 'label': 'CFG++ (l=0.6)'}
    ]
    
    experiment_configs = cfg_params + cfgpp_params
    
    all_loss_data = {}

    for config in experiment_configs:
        solver_name = config['solver_name']
        guidance = config['cfg_guidance']
        label = config['label']
        
        print(f"\nRunning experiment: {label}")
        
        # 1. Initialize Solver
        solver_config = munchify({'num_sampling': args.NFE })

        if args.model == "sdxl" or args.model == "sdxl_lightning":
            solver_fn = get_solver_sdxl
            # SDXL requires an additional empty prompt list if it's not a word swap edit
            prompt_args = {'prompt1': [args.null_prompt, args.prompt], 'prompt2': [args.null_prompt, args.prompt]}
        else:
            solver_fn = get_solver
            prompt_args = {'prompt': [args.null_prompt, args.prompt]}
        
        solver = solver_fn(solver_name, solver_config=solver_config, device=args.device)

        # 2. Setup Callback
        # Note: A dedicated folder is created for each run to separate image results
        run_workdir = Path(args.workdir).joinpath(label.replace(' ', '_').replace('=', ''))
        run_workdir.mkdir(parents=True, exist_ok=True)
        
        # Instantiate the custom callback with the run's specific workdir
        callback_instance = TrajectoryCallback(workdir=run_workdir)
        
        # 3. Run Sampling
        try:
            result = solver.sample(
                **prompt_args,
                cfg_guidance=guidance,
                callback_fn=callback_instance,
                # SDXL specific args
                target_size=(1024, 1024) if "sdxl" in args.model else None
            )
            
            save_image(result, run_workdir.joinpath(f'final_result.png'), normalize=True)
            
            # 4. Collect and Save Loss Data
            all_loss_data[label] = callback_instance.log_data
            
        except Exception as e:
            print(f"Error during {label} run: {e}")
            
    # 5. Final Loss Data Save
    with open(Path(args.workdir).joinpath('score_matching_loss_data.json'), 'w') as f:
        json.dump(all_loss_data, f, indent=4)
        
    print("\nExperiment complete. Check the output directory for images and JSON data.")
    print("Run the provided plotting script (not included here) to generate Figure 4 equivalent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFG++ Trajectory Evaluation")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/trajectory_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="low quality,jpeg artifacts,blurry,poorly drawn,ugly,worst quality,")
    parser.add_argument("--prompt", type=str, default="A beautiful portrait of a dog on a sunny day, high resolution, 4k")
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create the main work directory
    create_workdir(args.workdir)
    
    run_trajectory_experiment(args)