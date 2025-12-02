import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# --- Configuration ---
# 🚨 MODIFIED: TARGET_TIMESTEPS now includes your desired granular values.
TARGET_TIMESTEPS = [900, 800, 700, 600, 500, 400, 300, 200, 100, 0] 

# Define an acceptance range (the maximum difference allowed) since NFE=50 has steps 20 apart.
T_TOLERANCE = 10 


def create_loss_plot(loss_data, save_path):
    """Generates the Score Matching Loss plot (Figure 4 equivalent)."""
    plt.figure(figsize=(10, 6))
    
    for label, data_points in loss_data.items():
        # Extract time and metric values
        t_values = [d['t'] for d in data_points]
        metric_values = [d['metric'] for d in data_points]
        
        # Plotting the data
        plt.plot(t_values, metric_values, label=label, 
                 linestyle='-', marker='o', markersize=3)
        
    plt.gca().invert_xaxis() # Reverse X-axis to show diffusion time decreasing
    plt.xlabel('Diffusion Time $t$ (Steps)')
    plt.ylabel('Score Matching Loss ($\|\\epsilon_c - \\epsilon_\\emptyset\|^2$)')
    plt.title('Manifold Stability: Score Matching Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(save_path)
    print(f"\n✅ Saved Loss Plot to: {save_path}")


def create_trajectory_montage(workdir, loss_data, save_path):
    """
    Generates the visual trajectory montage (Figure 1/12 equivalent) 
    using the full size of the saved images.
    """
    
    # 1. Determine which subfolders (CFG/CFG++) we processed
    labels = list(loss_data.keys())
    
    # Cache the actual timesteps available in the loss data for easier searching
    available_t = {label: {d['t'] for d in loss_data[label]} for label in labels}
    
    image_rows = []
    
    for label in labels:
        row_images = []
        run_dir = workdir.joinpath(label.replace(' ', '_').replace('=', ''))
        img_dir = run_dir.joinpath('trajectory')
        
        for t_target in TARGET_TIMESTEPS:
            found = False
            img_to_use = None
            
            # --- MODIFIED SEARCH LOGIC ---
            # 1. Find the closest actual timestep 't_actual' available in the data
            t_actual = None
            
            # Search for the closest available t within tolerance
            closest_diff = float('inf')
            
            # We must iterate over all available t to find the closest one, 
            # especially since t_target may not exist exactly.
            for t_avail in available_t[label]:
                diff = abs(t_avail - t_target)
                if diff <= T_TOLERANCE and diff < closest_diff:
                    t_actual = t_avail
                    closest_diff = diff
            
            if t_actual is not None:
                # 2. Glob for the filename using the actual saved 't' value
                # Filename format: 't_{t:04d}_step_{step:03d}.png'
                matching_files = sorted(img_dir.glob(f't_{t_actual:04d}_step_*.png'))
                
                if matching_files:
                    img_path = matching_files[0]
                    img_to_use = Image.open(img_path)
                    found = True
            
            # -----------------------------
            
            if found:
                row_images.append(img_to_use)
            else:
                print(f"⚠️ Warning: Could not find image for {label} at t={t_target}. Using placeholder.")
                # Create a placeholder image (e.g., 512x512 for visual consistency)
                placeholder_size = (512, 512) 
                # If there are already images in the row, use their size
                if row_images:
                     placeholder_size = row_images[0].size
                     
                img_to_use = Image.new('RGB', placeholder_size, color = 'red')
                row_images.append(img_to_use)

        image_rows.append(row_images)
        
    if not image_rows or not image_rows[0]:
        print("❌ Error: No image rows or images found. Cannot create montage.")
        return

    # 3. Create the final stitched image using FULL IMAGE SIZE
    
    # Use the size of the first successfully loaded image as the standard cell size
    img_width, img_height = image_rows[0][0].size 
    
    num_rows = len(image_rows)
    num_cols = len(TARGET_TIMESTEPS)
    
    # Final canvas size (W x H)
    montage_width = img_width * num_cols
    montage_height = img_height * num_rows
    
    montage = Image.new('RGB', (montage_width, montage_height), color='white')
    
    # Paste images onto the canvas
    for r_idx, row in enumerate(image_rows):
        for c_idx, img in enumerate(row):
            # No resizing necessary here; we paste the original file size directly
            montage.paste(img, (c_idx * img_width, r_idx * img_height))

    # 4. Save the montage
    montage.save(save_path)
    print(f"✅ Saved Trajectory Montage to: {save_path}")
    print(f"Timesteps displayed (decreasing): {TARGET_TIMESTEPS}")


def main():
    parser = argparse.ArgumentParser(description="Plot CFG/CFG++ Trajectory Results")
    parser.add_argument("--workdir", type=Path, required=True, 
                        help="The path to the experiment work directory (e.g., examples/results/trajectory_landscape)")
    args = parser.parse_args()

    # --- Load Data ---
    loss_path = args.workdir.joinpath('score_matching_loss_data.json')
    if not loss_path.exists():
        print(f"❌ Error: Loss data not found at {loss_path}")
        print("Please ensure the 'evaluate_trajectory.py' script ran successfully.")
        return

    with open(loss_path, 'r') as f:
        loss_data = json.load(f)

    # --- Run Plotting Functions ---
    
    # 1. Create Loss Plot
    loss_save_path = args.workdir.joinpath('score_matching_loss_plot.png')
    create_loss_plot(loss_data, loss_save_path)
    
    # 2. Create Trajectory Montage
    montage_save_path = args.workdir.joinpath('trajectory_montage.png')
    create_trajectory_montage(args.workdir, loss_data, montage_save_path)


if __name__ == "__main__":
    main()