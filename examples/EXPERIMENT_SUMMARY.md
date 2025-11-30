# Experiment Summary and Recommendations

## Baseline Experiments Review

### Current Baseline

1. **CFG vs CFG++ Evaluation** (`evaluate_cfg_comparison.py`)
   - Systematic comparison across prompts and guidance scales
   - Shows CFG++ achieves similar results with lower guidance scales
   - Well-established baseline for reproduction

2. **Semantic Difference Experiment** (`semantic_difference_experiment.py`)
   - Original contribution: applying difference vectors between concepts
   - Tests how embedding manipulations affect generation
   - Good foundation for understanding embedding space

## New Experiments Created

### 1. Prompt Interpolation (`prompt_interpolation_experiment.py`)

**Why it's interesting:**
- Tests smooth transitions in embedding space
- Compares linear vs. SLERP interpolation
- Reveals how CFG vs CFG++ handle interpolated embeddings
- Visual results are immediately interpretable

**Key Research Questions:**
- Do interpolated embeddings produce smooth visual transitions?
- Is SLERP better than linear interpolation?
- How do CFG and CFG++ differ in handling interpolated embeddings?

**Recommended Usage:**
```bash
# Start with simple linear interpolation
python -m examples.prompt_interpolation_experiment \
    --prompts "a cat" "a dog" \
    --interpolation_method linear \
    --interpolation_steps 10 \
    --compare_both

# Then try SLERP for smoother transitions
python -m examples.prompt_interpolation_experiment \
    --prompts "a cat" "a dog" \
    --interpolation_method slerp \
    --interpolation_steps 15 \
    --compare_both
```

### 2. Timestep-Dependent Conditioning (`timestep_conditioning_experiment.py`)

**Why it's interesting:**
- Novel approach: different prompts at different timesteps
- Tests hypothesis that early vs. late conditioning matters
- Progressive refinement strategy (coarse → fine)
- Could reveal optimal conditioning strategies

**Key Research Questions:**
- Does early conditioning affect structure more than details?
- Can progressive refinement improve quality?
- How do CFG and CFG++ respond to time-varying conditioning?

**Recommended Usage:**
```bash
# Progressive refinement: most intuitive and likely to show differences
python -m examples.timestep_conditioning_experiment \
    --schedule_type progressive \
    --coarse_prompt "a person" \
    --fine_prompt "a person with blue eyes, wearing a red shirt, smiling" \
    --compare_both

# Style-content: interesting for artistic applications
python -m examples.timestep_conditioning_experiment \
    --schedule_type style_content \
    --content_prompt "a cat" \
    --style_prompt "in the style of Van Gogh" \
    --compare_both
```

## Recommended Experiment Order

### Phase 1: Quick Wins (Start Here)
1. **Prompt Interpolation (Linear)**
   - Easy to understand
   - Clear visual results
   - Builds on existing semantic difference work
   - Time: ~30 minutes per run

2. **Timestep Conditioning (Progressive)**
   - Novel contribution
   - Intuitive concept
   - Could reveal interesting dynamics
   - Time: ~30 minutes per run

### Phase 2: Deeper Analysis
3. **Prompt Interpolation (SLERP)**
   - Compare with linear interpolation
   - Test if spherical interpolation is better
   - Time: ~30 minutes per run

4. **Timestep Conditioning (Style-Content)**
   - Test different schedule types
   - Explore artistic applications
   - Time: ~30 minutes per run

### Phase 3: Advanced Experiments (If Time Permits)
5. **Embedding Perturbation Robustness**
   - Quantitative comparison
   - Important for understanding stability
   - See `EXPERIMENT_IDEAS.md` for details

6. **Learnable Prompt Tuning**
   - Most complex but very interesting
   - Could find optimal embeddings
   - See `EXPERIMENT_IDEAS.md` for details

## Key Insights to Look For

### When Comparing CFG vs CFG++

1. **Sensitivity to Embedding Changes**
   - Does CFG++ respond differently to interpolated embeddings?
   - Is one method more robust to perturbations?

2. **Timestep Dynamics**
   - Do CFG and CFG++ benefit differently from time-varying conditioning?
   - Which method is more sensitive to when conditioning is applied?

3. **Visual Quality**
   - Do interpolated embeddings produce smoother transitions with one method?
   - Does progressive refinement work better with CFG++?

4. **Guidance Scale Interactions**
   - How do guidance scales interact with embedding manipulations?
   - Does CFG++ need different guidance scales for different experiments?

## Presentation Recommendations

### For Your Final Project

1. **Start with Baseline** (2 min)
   - Show CFG vs CFG++ comparison
   - Establish that CFG++ works with lower guidance

2. **Semantic Difference** (3 min)
   - Your original contribution
   - Show how difference vectors affect generation
   - Compare CFG vs CFG++ responses

3. **Prompt Interpolation** (2 min)
   - New extension
   - Show smooth transitions
   - Discuss embedding space structure

4. **Timestep Conditioning** (2 min)
   - Novel contribution
   - Show progressive refinement results
   - Discuss optimal conditioning strategies

5. **Discussion** (1 min)
   - Key insights
   - Future directions
   - Limitations

## Quick Start Commands

### Option 1: Use the Bash Scripts (Recommended)

```bash
# Quick test run (4 curated experiments)
./examples/run_new_experiments_quick.sh

# Full comprehensive run (all experiments)
./examples/run_new_experiments.sh

# Custom configuration
DEVICE=cuda MODEL=sd15 NFE=50 ./examples/run_new_experiments.sh
```

### Option 2: Run Individual Experiments

```bash
# 1. Prompt Interpolation (Linear)
python -m examples.prompt_interpolation_experiment \
    --workdir examples/workdir/interpolation_linear \
    --prompts "a cat" "a dog" \
    --interpolation_method linear \
    --interpolation_steps 10 \
    --compare_both \
    --cfg_guidance 7.5

# 2. Prompt Interpolation (SLERP)
python -m examples.prompt_interpolation_experiment \
    --workdir examples/workdir/interpolation_slerp \
    --prompts "a cat" "a dog" \
    --interpolation_method slerp \
    --interpolation_steps 15 \
    --compare_both \
    --cfg_guidance 7.5

# 3. Timestep Conditioning (Progressive)
python -m examples.timestep_conditioning_experiment \
    --workdir examples/workdir/timestep_progressive \
    --schedule_type progressive \
    --coarse_prompt "a person" \
    --fine_prompt "a person with blue eyes, wearing a red shirt, smiling" \
    --compare_both \
    --cfg_guidance 7.5

# 4. Timestep Conditioning (Style-Content)
python -m examples.timestep_conditioning_experiment \
    --workdir examples/workdir/timestep_style \
    --schedule_type style_content \
    --content_prompt "a cat" \
    --style_prompt "in the style of Van Gogh" \
    --compare_both \
    --cfg_guidance 7.5
```

## Next Steps

1. **Run the new experiments** to get results
2. **Compare visual outputs** between CFG and CFG++
3. **Document findings** in your presentation
4. **Consider adding metrics** (CLIP scores, etc.) if time permits
5. **Explore additional ideas** from `EXPERIMENT_IDEAS.md` if interested

## Files Created

- `prompt_interpolation_experiment.py` - Prompt interpolation implementation
- `timestep_conditioning_experiment.py` - Timestep-dependent conditioning
- `run_new_experiments.sh` - Comprehensive bash script to run all experiments
- `run_new_experiments_quick.sh` - Quick test script (4 curated experiments)
- `EXPERIMENT_IDEAS.md` - Comprehensive list of additional experiment ideas
- `EXPERIMENT_SUMMARY.md` - This file

Good luck with your experiments!


