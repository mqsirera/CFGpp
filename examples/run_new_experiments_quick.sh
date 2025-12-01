#!/bin/bash

# Quick-start script: Runs a curated selection of the most interesting experiments
# This is faster than the full script and focuses on experiments likely to show clear differences

set -e

# Configuration
WORKDIR_BASE="examples/workdir"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-sd15}"
NFE="${NFE:-50}"
SEED="${SEED:-42}"
CFG_GUIDANCE="${CFG_GUIDANCE:-7.5}"
CFGPP_GUIDANCE="${CFGPP_GUIDANCE:-0.6}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Experiment Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run prompt interpolation
# Usage: run_interpolation name "prompt1" "prompt2" method steps
run_interpolation() {
    local name=$1
    local prompt1=$2
    local prompt2=$3
    local method=$4
    local steps=$5
    
    echo -e "${GREEN}[Interpolation]${NC} ${name}"
    local workdir="${WORKDIR_BASE}/interpolation_${name}"
    python -m examples.prompt_interpolation_experiment \
        --workdir ${workdir} \
        --device ${DEVICE} \
        --model ${MODEL} \
        --NFE ${NFE} \
        --seed ${SEED} \
        --cfg_guidance ${CFG_GUIDANCE} \
        --cfgpp_guidance ${CFGPP_GUIDANCE} \
        --prompts "${prompt1}" "${prompt2}" \
        --interpolation_method ${method} \
        --interpolation_steps ${steps} \
        --compare_both
    echo -e "${GREEN}✓${NC}\n"
}

# Function to run timestep conditioning
run_timestep() {
    local name=$1
    local schedule_type=$2
    shift 2
    local extra_args="$@"
    
    echo -e "${GREEN}[Timestep]${NC} ${name}"
    local workdir="${WORKDIR_BASE}/timestep_${name}"
    python -m examples.timestep_conditioning_experiment \
        --workdir ${workdir} \
        --device ${DEVICE} \
        --model ${MODEL} \
        --NFE ${NFE} \
        --seed ${SEED} \
        --cfg_guidance ${CFG_GUIDANCE} \
        --cfgpp_guidance ${CFGPP_GUIDANCE} \
        --schedule_type ${schedule_type} \
        --compare_both \
        ${extra_args}
    echo -e "${GREEN}✓${NC}\n"
}

# ============================================
# CURATED SELECTION - Most Interesting
# ============================================

# 1. Simple interpolation - easy to understand
run_interpolation "cat_to_dog" "a cat" "a dog" "linear" 10

# 2. SLERP for smoother transitions
run_interpolation "cat_to_dog_slerp" "a cat" "a dog" "slerp" 12

# 3. Progressive refinement - most intuitive timestep experiment
run_timestep "person_progressive" "progressive" \
    --coarse_prompt "a person" \
    --fine_prompt "a person with blue eyes, wearing a red shirt, smiling"

# 4. Style-content - interesting artistic application
run_timestep "cat_van_gogh" "style_content" \
    --content_prompt "a cat" \
    --style_prompt "in the style of Van Gogh" \
    --content_weight_start 1.0 \
    --content_weight_end 0.3

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Run Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

