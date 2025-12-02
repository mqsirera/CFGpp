## [ICLR2025] CFG++ : MANIFOLD-CONSTRAINED CLASSIFIER FREE GUIDANCE FOR DIFFUSION MODELS

This repository started as the official implementation of  
**[CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models](https://arxiv.org/abs/2406.08070)** by  
[Hyungjin Chung*](https://www.hj-chung.com/), [Jeongsol Kim*](https://jeongsol.dev/), [Geon Yeong Park*](https://geonyeong-park.github.io/), [Hyelin Nam*](https://www.linkedin.com/in/hyelin-nam-01ab631a3/), [Jong Chul Ye](https://bispl.weebly.com/professor.html).

We then **adapted the codebase** for the course  
**EECE 7398 – Special Topics: Machine Learning with Small Data**, to:

- Reproduce and evaluate CFG++ against standard classifier-free guidance (CFG)
- Design and run additional experiments (semantic differences, interpolation, timestep conditioning)
- Produce structured results and plots for analysis

![main figure](assets/main_test_v5.png)

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://cfgpp-diffusion.github.io/)  
[![arXiv](https://img.shields.io/badge/arXiv-2406.08070-b31b1b.svg)](https://arxiv.org/abs/2406.08070)

---

## Quick Summary: CFG, CFG++, and This Fork

- **Classifier-Free Guidance (CFG)** (Ho & Salimans, 2022) guides diffusion models by linearly combining unconditional and conditional scores. It works well, but typically needs **large guidance scales** (e.g. 7.5–12.5), which can cause mode collapse, off-manifold trajectories, and poor invertibility.

- **CFG++** is a simple modification that **keeps the reverse trajectory closer to the data manifold**, allowing:
  - **Small guidance scales** \(\lambda \in [0, 1]\) with effects comparable to large CFG scales
  - Better sample quality and text adherence
  - Smoother, straighter trajectories and improved invertibility

- **This repository**:
  - Keeps the original CFG++ text-to-image and inversion scripts
  - Adds new experimental scripts and plotting utilities for:
    - CFG vs CFG++ evaluation
    - Semantic difference in embedding space (original contribution)
    - Prompt interpolation (linear, SLERP, multi-blend)
    - Timestep-dependent prompt conditioning

---

## 🛠️ Setup

We use the original environment from the CFG++ paper.

git clone https://github.com/CFGpp-diffusion/CFGpp.git
cd CFGpp
conda env create -f environment.yaml
conda activate cfgpp   # or your chosen env nameDiffusers will automatically download checkpoints for SD v1.5 or SDXL. SDXL-Lightning is also supported for fast sampling.

---

## Original Usage: Text-to-Image and Inversion

### Text-to-Image

**CFG (baseline):**

python -m examples.text_to_img \
  --prompt "a portrait of a dog" \
  --method "ddim" \
  --cfg_guidance 7.5**CFG++ (recommended):**

python -m examples.text_to_img \
  --prompt "a portrait of a dog" \
  --method "ddim_cfg++" \
  --cfg_guidance 0.6### SDXL-Lightning + CFG++

1. Download `sdxl_lightning_4step_unet.safetensors` into `ckpt/` (from ByteDance SDXL-Lightning).
2. Run:

python -m examples.text_to_img \
  --prompt "stars, water, brilliantly, gorgeous large scale scene, a little girl, in the style of dreamy realism, light gold and amber, blue and pink, brilliantly illuminated in the background." \
  --method "ddim_cfg++_lightning" \
  --model "sdxl_lightning" \
  --cfg_guidance 1.0 \
  --NFE 4### Image Inversion (DDIM Inversion)

**CFG:**

python -m examples.inversion \
  --prompt "a photograph of baby fox" \
  --method "ddim_inversion" \
  --cfg_guidance 7.5**CFG++:**

python -m examples.inversion \
  --prompt "a photograph of baby fox" \
  --method "ddim_inversion_cfg++" \
  --cfg_guidance 0.6Add `--model sdxl` to switch to SDXL.

---

## New Experiments for EECE 7398

This section describes the additional experiments we implemented on top of the original CFG++ code.

### 1. CFG vs CFG++ Evaluation

**File:** `examples/evaluate_cfg_comparison.py`

**Purpose:**  
Systematically compare CFG vs CFG++ across prompts and guidance scales, producing both images and a JSON summary.

**Example: default prompts**

python -m examples.evaluate_cfg_comparison \
  --workdir examples/workdir/evaluation \
  --NFE 50 \
  --seed 42**Example: custom prompts**

python -m examples.evaluate_cfg_comparison \
  --workdir examples/workdir/evaluation \
  --prompts_file examples/sample_prompts.json \
  --cfg_scales 1.0 2.5 5.0 7.5 10.0 \
  --cfgpp_scales 0.1 0.3 0.5 0.7 1.0**Output:**

- Individual images for each (prompt, method, scale)
- Per-prompt comparison grids
- `evaluation_summary.json` with all metadata

**Plotting 2-row figures (CFG top, CFG++ bottom):**

python -m examples.plot_evaluation_figure \
  --workdir examples/workdir/evaluation---

### 2. Semantic Difference Experiment (Original Contribution)

**File:** `examples/semantic_difference_experiment.py`

**Purpose:**  
Test whether **semantic differences in text embeddings** (e.g. “a girl” → “a boy”) translate into controllable changes in generated identity when applied to a base prompt.

**Concept:**

1. Compute embeddings for concepts, e.g. `"a girl"` and `"a boy"`.
2. Difference:  
   \(\text{diff} = \text{emb}("a boy") - \text{emb}("a girl")\).
3. Apply to base prompt:  
   \(\text{emb}_\text{mod} = \text{emb}("a portrait of a person") + \alpha \cdot \text{diff}\) for multiple strengths \(\alpha\).
4. Generate images with CFG or CFG++ and compare.

**Example: basic experiment (CFG only)**

python -m examples.semantic_difference_experiment \
  --workdir examples/workdir/semantic_diff/emotion_portrait \
  --concept1 "a happy person" \
  --concept2 "a sad person" \
  --base_prompts "a portrait of a person" \
  --strengths -1.0 -0.5 0.0 0.5 1.0 \
  --cfg_guidance 7.5**Example: CFG vs CFG++ comparison**

python -m examples.semantic_difference_experiment \
  --workdir examples/workdir/semantic_diff/emotion_cfg_vs_cfgpp_comparison \
  --concept1 "a happy person" \
  --concept2 "a sad person" \
  --base_prompts "a portrait of a person" \
  --strengths -1.0 0.0 1.0 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6**Output:**

- Images for each base prompt and semantic strength
- Baseline (unmodified prompt) images
- Reference images for each concept
- Comparison grids per base prompt

**Plotting baseline + concepts + strengths (comparison or single-method):**

# Comparison experiment
python -m examples.plot_semantic_diff_figure \
  --workdir examples/workdir/semantic_diff/emotion_cfg_vs_cfgpp_comparison

# Non-comparison experiment
python -m examples.plot_semantic_diff_figure \
  --workdir examples/workdir/semantic_diff/emotion_portrait---

### 3. Prompt Interpolation Experiments

**File:** `examples/prompt_interpolation_experiment.py`

**Purpose:**  
Examine how the model behaves as we **move continuously between prompts** in embedding space, with linear interpolation, SLERP, and multi-prompt blending. Compare CFG vs CFG++ on the same interpolated embeddings.

**Concept:**

1. Get embeddings for prompts such as `"a cat"` and `"a dog"`.
2. Interpolate:
   - **Linear (LERP)**: straight line in embedding space.
   - **SLERP**: spherical interpolation on the unit hypersphere (better magnitude preservation, smoother semantics).
   - **Multi-blend**: weighted combination of 3+ prompts.
3. Generate images along the path for a grid of interpolation factors \(\alpha\).
4. Optionally generate both CFG and CFG++ images for each \(\alpha\).

**Examples:**

# Linear interpolation (CFG vs CFG++)
python -m examples.prompt_interpolation_experiment \
  --workdir examples/workdir/interpolation/interpolation_cat_to_dog_linear \
  --prompts "a cat" "a dog" \
  --interpolation_method linear \
  --interpolation_steps 10 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6

# SLERP interpolation (smoother transitions)
python -m examples.prompt_interpolation_experiment \
  --workdir examples/workdir/interpolation/interpolation_cat_to_dog_slerp \
  --prompts "a cat" "a dog" \
  --interpolation_method slerp \
  --interpolation_steps 15 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6

# Multi-prompt blending (e.g., cat, dog, bird)
python -m examples.prompt_interpolation_experiment \
  --workdir examples/workdir/interpolation/interpolation_animals_blend \
  --prompts "a cat" "a dog" "a bird" \
  --interpolation_method multi_blend \
  --weights 0.4 0.4 0.2 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6**Output:**

- Reference images for each prompt
- Interpolated images for each \(\alpha\)
- `interpolation_summary.json` with all prompts, methods, alphas, and paths

**Plotting 2-row interpolation figures:**

python -m examples.plot_interpolation_figure \
  --workdir examples/workdir/interpolation/interpolation_cat_to_dog_linearThis produces a figure with CFG in the top row and CFG++ in the bottom row, ordered by \(\alpha\).

---

### 4. Timestep-Dependent Prompt Conditioning

**File:** `examples/timestep_conditioning_experiment.py`

**Purpose:**  
Investigate what happens when **prompts or prompt weights change over diffusion timesteps**, and how CFG vs CFG++ respond to these dynamic conditioning schedules.

**Concept:**

- **Progressive refinement:**  
  Start with a simple prompt, gradually switch to a more detailed one (structure first, then details).
- **Style-content separation:**  
  Keep a content prompt fixed while increasing weight on a style prompt.
- **Multi-prompt schedules:**  
  Move between several prompts over time (e.g., cat → dog → bird).
- **Negative schedules:**  
  Gradually increase negative prompt strength to push out undesirable artifacts.

**Examples:**

# Progressive refinement (CFG vs CFG++)
python -m examples.timestep_conditioning_experiment \
  --workdir examples/workdir/timestep/person_progressive \
  --schedule_type progressive \
  --coarse_prompt "a person" \
  --fine_prompt "a person with blue eyes, wearing a red shirt, smiling, standing in a park" \
  --transition_start 0.3 \
  --transition_end 0.7 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6

# Style-content separation (content + Van Gogh style)
python -m examples.timestep_conditioning_experiment \
  --workdir examples/workdir/timestep/cat_van_gogh \
  --schedule_type style_content \
  --content_prompt "a cat" \
  --style_prompt "in the style of Van Gogh" \
  --content_weight_start 1.0 \
  --content_weight_end 0.3 \
  --compare_both \
  --cfg_guidance 7.5 \
  --cfgpp_guidance 0.6**Output:**

- Images generated with timestep-dependent conditioning
- Baseline (constant prompt) images for comparison
- `timestep_conditioning_summary.json` with schedule configuration

---

## Batch Runners

Several helper scripts run sets of experiments:

- `examples/run_semantic_experiments.sh`
- `examples/run_semantic_experiments_simple.sh`
- `examples/run_new_experiments.sh`
- `examples/run_new_experiments_quick.sh`

Example:

./examples/run_semantic_experiments.sh
./examples/run_new_experiments.shThese populate `examples/workdir/...` with consistent CFG vs CFG++ experiments ready to be plotted.

---

## Citations

If you use this repository or its experiments, please cite:

**CFG++ (base method):**
tex
@inproceedings{
  chung2025cfg,
  title={{CFG}++: Manifold-constrained Classifier Free Guidance for Diffusion Models},
  author={Hyungjin Chung and Jeongsol Kim and Geon Yeong Park and Hyelin Nam and Jong Chul Ye},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=E77uvbOTtp}
}**Original Classifier-Free Guidance (CFG):**
tex
@article{ho2022classifierfree,
  title        = {Classifier-Free Diffusion Guidance},
  author       = {Ho, Jonathan and Salimans, Tim},
  journal      = {arXiv preprint arXiv:2207.12598},
  year         = {2022}
}This fork builds directly on the official CFG++ implementation and extends it with course-specific experiments and plotting tools for **EECE 7398 – Special Topics: Machine Learning with Small Data**.