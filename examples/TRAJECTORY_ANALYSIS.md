# Trajectory Stability Analysis: CFG vs. CFG++

This experiment validates the core claim of the **CFG++ paper**:  
that **manifold-constrained guidance** produces a more numerically stable and smoother reverse diffusion trajectory than standard CFG, especially under high guidance force.

The experiment generates two key outputs for comparative analysis:

- **Quantitative Data**: Score Matching Loss plot  
  \( \\lVert \\hat{\\epsilon}_c - \\hat{\\epsilon}_{\\emptyset} \\rVert^2 \) over time.
- **Visual Data**: A Trajectory Montage grid showing the denoising evolution.

---

## 1. Prerequisites and Setup

Before running, ensure your solver files are correctly modified to expose the directional vector:

### Required Solver Modification

Inside DDIM and DDIM_CFG++ sampling loops, ensure this is passed:

```python
callback_kwargs = { ..., "noise_diff": noise_diff.detach() }
```

---

## 2. Running the Experiment (`evaluate_trajectory.py`)

This script executes the diffusion process for:

- **CFG** (ω = 7.5) — extrapolating, unstable  
- **CFG++** (λ = 0.6) — interpolating, stable

All intermediate steps and loss data are saved to `--workdir`.

### Execution Command Template

```bash
python -m examples.evaluate_trajectory     --model sd15     --prompt "[Your Complex Prompt Here]"     --NFE 50     --workdir examples/results/trajectory_EXP_NAME
```

### Example (Botanical Clock Prompt)

```bash
python -m examples.evaluate_trajectory     --model sd15     --prompt "An intricate, highly detailed render of a large, antique brass clock mechanism completely overgrown with vibrant, rare luminescent blue moss and tiny dew-covered orchid blossoms. The entire clockwork structure is suspended inside a perfect, transparent glass sphere filled with swirling, smoky white vapor. Include strong, directional volumetric light cutting through the vapor, casting sharp shadows on the brass. UHD, photorealistic, cinematic lighting."     --NFE 50     --workdir examples/results/trajectory_botanical_clock
```

### Key Parameters

| Parameter | Value | Description |
|----------|-------|-------------|
| `--NFE` | 50 | Number of diffusion steps |
| CFG Test | DDIM (ω = 7.5) | High-extrapolation guidance (unstable) |
| CFG++ Test | DDIM_CFG++ (λ = 0.6) | Manifold-constrained guidance (stable) |

---

## 3. Generating the Plots (`plot_results.py`)

Run the plotter to generate final visualizations.

### Command

```bash
python -m examples.plot_results     --workdir examples/results/trajectory_EXP_NAME
```

### Output Files

The following files will appear inside the workdir:

- `score_matching_loss_plot.png`
- `trajectory_montage.png`

---

## 4. Interpretation of Results

### A. Score Matching Loss Plot (Quantitative)

This plot shows:

\[
\| \hat{\epsilon}_c - \hat{\epsilon}_{\emptyset} \|^2
\]

over diffusion timestep **t**.

#### Expected Outcomes

- **CFG (High ω)**  
  - Sharp spikes  
  - High volatility  
  - Indicates off-manifold drift and unstable gradients

- **CFG++ (λ constrained)**  
  - Smooth trajectory  
  - Low variance  
  - Stable guidance direction

---

### B. Trajectory Montage (Visual)

This montage compares the denoised estimate \( \hat{x}_0 \) at specific timesteps.

#### Expected Visual Patterns

- **CFG**  
  - Color explosions  
  - Abrupt pattern jumps  
  - Pixelated artifacts  
  - Visual symptoms of off-manifold sampling

- **CFG++**  
  - Smooth transitions  
  - Gradual detail emergence  
  - Cleaner structure  
  - Matches the smooth loss curve

---

## Summary

This experiment empirically demonstrates that **CFG++ produces more stable guidance** than classical CFG—  
both **numerically** (loss curve) and **visually** (denoising trajectory).