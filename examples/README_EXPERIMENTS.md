# Experimental Scripts

This directory contains experimental scripts for evaluating and extending the CFG++ work.

## 1. CFG vs CFG++ Evaluation Script

**File:** `evaluate_cfg_comparison.py`

This script systematically compares CFG and CFG++ across multiple prompts and guidance scales.

### Usage

```bash
# Basic usage with default prompts
python -m examples.evaluate_cfg_comparison --workdir examples/workdir/evaluation

# With custom prompts file (JSON format)
python -m examples.evaluate_cfg_comparison \
    --workdir examples/workdir/evaluation \
    --prompts_file examples/prompts.json \
    --cfg_scales 1.0 2.5 5.0 7.5 10.0 \
    --cfgpp_scales 0.1 0.3 0.5 0.7 1.0 \
    --NFE 50 \
    --seed 42
```

### Arguments

- `--workdir`: Directory to save results
- `--device`: Device to use (default: "cuda")
- `--model`: Model to use (sd15, sd20, sdxl)
- `--NFE`: Number of function evaluations (default: 50)
- `--seed`: Random seed (default: 42)
- `--prompts_file`: JSON file with list of prompts (optional)
- `--cfg_scales`: Guidance scales for standard CFG (default: [1.0, 2.5, 5.0, 7.5, 10.0])
- `--cfgpp_scales`: Guidance scales for CFG++ (default: [0.1, 0.3, 0.5, 0.7, 1.0])
- `--null_prompt`: Negative prompt (default: standard negative prompt)

### Output

The script generates:
- Individual images for each prompt/method/scale combination
- Comparison grids for each prompt
- A JSON summary file with all results

### Example Prompts File Format

Create a `prompts.json` file:

```json
[
  "a portrait of a dog",
  "a beautiful landscape with mountains and a lake",
  "a futuristic city at night"
]
```

## 2. Semantic Difference Experiment

**File:** `semantic_difference_experiment.py`

This script implements the original contribution: manipulating text embeddings using semantic differences to modify generated identity.

### Concept

The experiment:
1. Computes embeddings for two semantic concepts (e.g., "girl" vs "boy")
2. Calculates the difference vector between them
3. Applies this difference to base prompts (e.g., "a portrait of a person")
4. Generates images to see if the identity changes

### Usage

```bash
# Basic usage - test girl vs boy semantic difference
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

# Use CFG++ only
python -m examples.semantic_difference_experiment \
    --workdir examples/workdir/semantic_diff \
    --concept1 "a girl" \
    --concept2 "a boy" \
    --base_prompts "a portrait of a person" \
    --use_cfgpp \
    --cfg_guidance 0.6
```

### Arguments

- `--workdir`: Directory to save results
- `--device`: Device to use (default: "cuda")
- `--model`: Model to use (sd15, sd20, sdxl)
- `--NFE`: Number of function evaluations (default: 50)
- `--seed`: Random seed (default: 42)
- `--cfg_guidance`: Guidance scale (default: 7.5 for CFG, use 0.6 for CFG++)
- `--use_cfgpp`: Use CFG++ sampling instead of standard CFG
- `--compare_both`: Generate images with both CFG and CFG++ for comparison
- `--concept1`: First concept for computing difference (default: "a girl")
- `--concept2`: Second concept for computing difference (default: "a boy")
- `--base_prompts`: Base prompts to apply semantic difference to
- `--strengths`: Strengths to apply the semantic difference (negative = concept1, positive = concept2)

### Output

The script generates:
- Images for each base prompt with different semantic difference strengths
- Baseline images (standard generation without modification)
- Reference images for the concepts themselves
- Comparison grids showing the effect of different strengths

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

4. **Style:**
   ```bash
   --concept1 "a realistic person" --concept2 "a cartoon person"
   ```

## Notes

- Both scripts support SDv1.5, SDv2.0, and SDXL models
- For faster testing, reduce `--NFE` (e.g., 20-30 steps)
- The semantic difference experiment requires understanding of embedding spaces
- Results may vary based on the model and random seed

## For Your Final Project

These scripts provide:

1. **Reproduction:** The evaluation script allows systematic comparison of CFG vs CFG++
2. **Original Contribution:** The semantic difference experiment explores how embedding manipulations affect generation
3. **Quantitative Analysis:** Both scripts generate structured outputs for analysis
4. **Visual Results:** Grid comparisons make it easy to see differences

You can extend these scripts to:
- Add quantitative metrics (FID, CLIP scores, etc.)
- Test more semantic concepts
- Analyze the embedding space structure
- Compare how CFG vs CFG++ respond to embedding manipulations

