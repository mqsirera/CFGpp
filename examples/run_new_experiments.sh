#!/bin/bash

# ============================================================================
# Script to run interesting combinations of new experiments
# ============================================================================
# 
# This script runs a comprehensive set of experiments for:
# - Prompt Interpolation (linear, SLERP, multi-prompt blending)
# - Timestep-Dependent Conditioning (progressive, style-content, etc.)
#
# Usage:
#   ./run_new_experiments.sh
#
#   Or with custom settings:
#   DEVICE=cuda MODEL=sd15 NFE=50 ./run_new_experiments.sh
#
# For a quick test run, use:
#   ./run_new_experiments_quick.sh
#
# ============================================================================

set -e  # Exit on error

# Configuration
WORKDIR_BASE="examples/workdir"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-sd15}"
NFE="${NFE:-50}"
SEED="${SEED:-42}"
CFG_GUIDANCE="${CFG_GUIDANCE:-7.5}"
CFGPP_GUIDANCE="${CFGPP_GUIDANCE:-0.6}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running New Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run prompt interpolation
# Usage: run_interpolation name "prompt1" "prompt2" method steps compare_both
run_interpolation() {
    local name=$1
    local prompt1=$2
    local prompt2=$3
    local method=$4
    local steps=$5
    local compare_both=$6
    
    echo -e "${GREEN}[Prompt Interpolation]${NC} ${name}"
    echo "  Prompts: ${prompt1}, ${prompt2}"
    echo "  Method: ${method}"
    echo "  Steps: ${steps}"
    
    local workdir="${WORKDIR_BASE}/interpolation_${name}"
    
    if [ "$compare_both" = "true" ]; then
        python -m examples.prompt_interpolation_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --prompts "${prompt1}" "${prompt2}" \
            --interpolation_method "${method}" \
            --interpolation_steps "${steps}" \
            --compare_both
    else
        python -m examples.prompt_interpolation_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --prompts "${prompt1}" "${prompt2}" \
            --interpolation_method "${method}" \
            --interpolation_steps "${steps}" \
            --use_cfgpp
    fi
    
    echo -e "${GREEN}✓ Completed${NC}\n"
}

# Function to run multi-prompt blending with custom weights
# Usage: run_multi_blend name "prompt1" "prompt2" "prompt3" "weight1 weight2 weight3" compare_both
run_multi_blend() {
    local name=$1
    local prompt1=$2
    local prompt2=$3
    local prompt3=$4
    local weights=$5
    local compare_both=$6
    
    echo -e "${GREEN}[Prompt Interpolation]${NC} ${name} (Multi-blend)"
    echo "  Prompts: ${prompt1}, ${prompt2}, ${prompt3}"
    echo "  Weights: ${weights}"
    
    local workdir="${WORKDIR_BASE}/interpolation_${name}"
    
    if [ "$compare_both" = "true" ]; then
        python -m examples.prompt_interpolation_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --prompts "${prompt1}" "${prompt2}" "${prompt3}" \
            --interpolation_method multi_blend \
            --weights ${weights} \
            --compare_both
    else
        python -m examples.prompt_interpolation_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --prompts "${prompt1}" "${prompt2}" "${prompt3}" \
            --interpolation_method multi_blend \
            --weights ${weights} \
            --use_cfgpp
    fi
    
    echo -e "${GREEN}✓ Completed${NC}\n"
}

# Function to run timestep conditioning
# Usage: run_timestep_conditioning name schedule_type compare_both [extra_args...]
run_timestep_conditioning() {
    local name=$1
    local schedule_type=$2
    local compare_both=$3
    shift 3
    local extra_args=("$@")
    
    echo -e "${GREEN}[Timestep Conditioning]${NC} ${name}"
    echo "  Schedule: ${schedule_type}"
    
    local workdir="${WORKDIR_BASE}/timestep_${name}"
    
    if [ "$compare_both" = "true" ]; then
        python -m examples.timestep_conditioning_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --schedule_type "${schedule_type}" \
            "${extra_args[@]}" \
            --compare_both
    else
        python -m examples.timestep_conditioning_experiment \
            --workdir "${workdir}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --NFE "${NFE}" \
            --seed "${SEED}" \
            --cfg_guidance "${CFG_GUIDANCE}" \
            --cfgpp_guidance "${CFGPP_GUIDANCE}" \
            --schedule_type "${schedule_type}" \
            "${extra_args[@]}" \
            --use_cfgpp
    fi
    
    echo -e "${GREEN}✓ Completed${NC}\n"
}

# ============================================
# PROMPT INTERPOLATION EXPERIMENTS
# ============================================

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}1. Prompt Interpolation Experiments${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Animal transitions
run_interpolation "cat_to_dog_linear" "a cat" "a dog" "linear" 10 true
run_interpolation "cat_to_dog_slerp" "a cat" "a dog" "slerp" 15 true

# Person attributes
run_interpolation "young_to_old" "a young person" "an old person" "linear" 10 true
run_interpolation "happy_to_sad" "a happy person" "a sad person" "linear" 10 true

# Style transitions
run_interpolation "realistic_to_cartoon" "a realistic person" "a cartoon person" "linear" 10 true
run_interpolation "photo_to_painting" "a photograph of a landscape" "a painting of a landscape" "slerp" 12 true

# Object transitions
run_interpolation "cat_to_bird" "a cat" "a bird" "linear" 10 true
run_interpolation "car_to_bike" "a vintage car" "a bicycle" "slerp" 12 true

# Multi-prompt blending
run_multi_blend "animals_blend" "a cat" "a dog" "a bird" "0.4 0.4 0.2" true
run_multi_blend "emotions_blend" "a happy person" "a sad person" "an angry person" "0.4 0.3 0.3" true

# ============================================
# TIMESTEP CONDITIONING EXPERIMENTS
# ============================================

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}2. Timestep-Dependent Conditioning${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Progressive refinement - Person details
run_timestep_conditioning "person_progressive" "progressive" true \
    --coarse_prompt "a person" \
    --fine_prompt "a person with blue eyes, wearing a red shirt, smiling, standing in a park"

# Progressive refinement - Animal details
run_timestep_conditioning "cat_progressive" "progressive" true \
    --coarse_prompt "a cat" \
    --fine_prompt "a fluffy orange cat with green eyes, sitting on a windowsill, with sunlight streaming in"

# Progressive refinement - Landscape
run_timestep_conditioning "landscape_progressive" "progressive" true \
    --coarse_prompt "a landscape" \
    --fine_prompt "a beautiful mountain landscape with a lake, pine trees, and a sunset sky"

# Style-content separation
run_timestep_conditioning "cat_van_gogh" "style_content" true \
    --content_prompt "a cat" \
    --style_prompt "in the style of Van Gogh" \
    --content_weight_start 1.0 \
    --content_weight_end 0.3

run_timestep_conditioning "landscape_impressionist" "style_content" true \
    --content_prompt "a landscape with mountains" \
    --style_prompt "in the style of impressionist painting" \
    --content_weight_start 1.0 \
    --content_weight_end 0.4

# Multi-prompt schedule
run_timestep_conditioning "animals_multi" "multi_prompt" true \
    --prompts "a cat" "a dog" "a bird"

run_timestep_conditioning "emotions_multi" "multi_prompt" true \
    --prompts "a happy person" "a sad person" "an angry person"

# Negative prompt scheduling
run_timestep_conditioning "landscape_negative" "negative" true \
    --positive_prompt "a beautiful landscape" \
    --negative_prompt "blurry, low quality, distorted" \
    --negative_strength_start 0.0 \
    --negative_strength_end 1.0

run_timestep_conditioning "portrait_negative" "negative" true \
    --positive_prompt "a professional portrait" \
    --negative_prompt "blurry, artifacts, low quality" \
    --negative_strength_start 0.0 \
    --negative_strength_end 0.8

# ============================================
# SUMMARY
# ============================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Experiments Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Prompt Interpolation: ${WORKDIR_BASE}/interpolation_*/"
echo "  - Timestep Conditioning: ${WORKDIR_BASE}/timestep_*/"
echo ""
echo "To view results, check the 'results' subdirectories in each workdir."

