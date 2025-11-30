# Experimental Contribution for Final Project

This document describes the experimental scripts created for your final project on CFG++.

## Overview

Two main experimental scripts have been created:

1. **CFG vs CFG++ Evaluation Script** - Systematic comparison across multiple prompts and guidance scales
2. **Semantic Difference Experiment** - Original contribution exploring embedding manipulation

## 1. CFG vs CFG++ Evaluation Script

**File:** `examples/evaluate_cfg_comparison.py`

### Purpose
Reproduce and systematically compare CFG and CFG++ across:
- Multiple prompts
- Different guidance scales
- Visual and quantitative analysis

### Key Features
- Tests multiple prompts automatically
- Compares CFG (scales 1.0-10.0) vs CFG++ (scales 0.1-1.0)
- Generates comparison grids for easy visualization
- Saves structured JSON summary for analysis

### Quick Start

```bash
# Basic evaluation with default prompts
python -m examples.evaluate_cfg_comparison \
    --workdir examples/workdir/evaluation \
    --NFE 50 \
    --seed 42

# With custom prompts file
python -m examples.evaluate_cfg_comparison \
    --workdir examples/workdir/evaluation \
    --prompts_file examples/sample_prompts.json \
    --cfg_scales 1.0 2.5 5.0 7.5 10.0 \
    --cfgpp_scales 0.1 0.3 0.5 0.7 1.0
```

### Output
- Individual images for each combination
- Comparison grids showing all scales side-by-side
- JSON summary with all paths and metadata

## 2. Semantic Difference Experiment (Original Contribution)

**File:** `examples/semantic_difference_experiment.py`

### Purpose
Investigate whether semantic differences in the embedding space translate to visual differences in generated images.

### Concept
1. Compute embeddings for two concepts (e.g., "girl" vs "boy")
2. Calculate the difference vector: `diff = embedding("boy") - embedding("girl")`
3. Apply this difference to base prompts: `modified = embedding("person") + strength * diff`
4. Generate images to see if identity changes

### Research Question
**"Do semantic differences in text embeddings reflect in the diffusion process and affect generated identity?"**

### Key Features
- Computes semantic difference vectors between concepts
- Applies differences with controllable strength
- Supports both CFG and CFG++ for comparison
- Generates reference images for concepts

### Quick Start

```bash
# Basic experiment: girl vs boy
python -m examples.semantic_difference_experiment \
    --workdir examples/workdir/semantic_diff \
    --concept1 "a girl" \
    --concept2 "a boy" \
    --base_prompts "a portrait of a person" "a person standing" \
    --strengths -1.0 -0.5 0.0 0.5 1.0 \
    --cfg_guidance 7.5

# Compare CFG vs CFG++ with semantic differences
python -m examples.semantic_difference_experiment \
    --workdir examples/workdir/semantic_diff \
    --concept1 "a girl" \
    --concept2 "a boy" \
    --base_prompts "a portrait of a person" \
    --strengths -1.0 0.0 1.0 \
    --compare_both \
    --cfg_guidance 7.5

# Use CFG++ (lower guidance scale)
python -m examples.semantic_difference_experiment \
    --workdir examples/workdir/semantic_diff \
    --concept1 "a girl" \
    --concept2 "a boy" \
    --base_prompts "a portrait of a person" \
    --use_cfgpp \
    --cfg_guidance 0.6
```

### Example Experiments

1. **Gender Identity:**
   ```bash
   --concept1 "a girl" --concept2 "a boy"
   ```

2. **Age:**
   ```bash
   --concept1 "a young person" --concept2 "an old person"
   ```

3. **Emotion:**
   ```bash
   --concept1 "a happy person" --concept2 "a sad person"
   ```

4. **Art Style:**
   ```bash
   --concept1 "a realistic person" --concept2 "a cartoon person"
   ```

### Output
- Images for each base prompt with different semantic difference strengths
- Baseline images (standard generation)
- Reference images for the concepts
- Comparison grids showing the effect

## For Your Presentation

### 1. Review of Original Paper (2 minutes)
- Brief overview of CFG and its limitations
- Introduction to CFG++ and its improvements
- Key technical difference (manifold constraint)

### 2. Your Contribution (8 minutes)

#### A. Reproduction (2 minutes)
- Show results from evaluation script
- Demonstrate CFG vs CFG++ comparison
- Highlight that CFG++ achieves similar results with lower guidance scales

#### B. Original Contribution: Semantic Difference Experiment (4 minutes)
- **Motivation:** 
  - Question: "Do semantic differences in embeddings translate to visual differences?"
  - Why it matters: Understanding embedding space structure and how it affects generation
  
- **Methodology:**
  - Compute difference vectors between semantic concepts
  - Apply to base prompts with controllable strength
  - Generate images to visualize effects
  
- **Technical Implementation:**
  - Custom embedding manipulation
  - Integration with both CFG and CFG++ samplers
  - Systematic testing across concepts and strengths

#### C. Evaluation (2 minutes)
- Show visual results: how semantic differences affect generation
- Compare CFG vs CFG++ responses to embedding manipulation
- Discuss findings:
  - Do semantic differences translate to visual differences?
  - How does CFG++ compare to CFG in this context?
  - What does this tell us about the embedding space?

### 3. Results & Interpretation (2 minutes)
- Quantitative/qualitative analysis
- Key insights
- Limitations and future work

## Files Created

1. `examples/evaluate_cfg_comparison.py` - Evaluation script
2. `examples/semantic_difference_experiment.py` - Semantic difference experiment
3. `examples/README_EXPERIMENTS.md` - Detailed documentation
4. `examples/sample_prompts.json` - Sample prompts for evaluation
5. `EXPERIMENTAL_CONTRIBUTION.md` - This file

## Next Steps

1. **Run the evaluation script** to get baseline comparisons
2. **Run semantic difference experiments** with different concepts
3. **Analyze results** - look for patterns in how semantic differences affect generation
4. **Prepare visualizations** for your presentation
5. **Consider extensions:**
   - Add quantitative metrics (CLIP scores, etc.)
   - Test more semantic concepts
   - Analyze embedding space structure
   - Compare CFG vs CFG++ sensitivity to embedding changes

## Notes

- Both scripts work with SDv1.5, SDv2.0, and SDXL
- For faster testing, reduce `--NFE` to 20-30 steps
- Results directory structure makes it easy to organize findings
- The semantic difference experiment is your original contribution - focus on this in your presentation

Good luck with your final project!

