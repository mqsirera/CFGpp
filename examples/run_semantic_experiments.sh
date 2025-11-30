#!/bin/bash

# Semantic Difference Experiments Runner
# This script runs multiple semantic difference experiments with various concept pairs
# and base prompts to explore how embedding manipulations affect generation.
#
# Usage:
#   ./examples/run_semantic_experiments.sh
#
#   Or with custom settings:
#   DEVICE=cuda MODEL=sd15 NFE=50 ./examples/run_semantic_experiments.sh
#
#   To run just a few experiments, edit this file and comment out the ones you don't need.
#
# For a simpler version with fewer experiments, see: run_semantic_experiments_simple.sh

set -e  # Exit on error

# Configuration
BASE_DIR="examples/workdir/semantic_diff"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-sd15}"
NFE="${NFE:-50}"
SEED="${SEED:-42}"
CFG_GUIDANCE="${CFG_GUIDANCE:-7.5}"
CFGPP_GUIDANCE="${CFGPP_GUIDANCE:-0.6}"

# Default strengths to test
STRENGTHS="-1.0 -0.5 0.0 0.5 1.0"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Semantic Difference Experiments Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run a single experiment
run_experiment() {
    local concept1="$1"
    local concept2="$2"
    local base_prompts="$3"
    local use_cfgpp="$4"
    local exp_name="$5"
    
    local workdir="${BASE_DIR}/${exp_name}"
    local guidance=$CFG_GUIDANCE
    local method_flag=""
    
    if [ "$use_cfgpp" = "true" ]; then
        guidance=$CFGPP_GUIDANCE
        method_flag="--use_cfgpp"
        exp_name="${exp_name}_cfgpp"
        workdir="${BASE_DIR}/${exp_name}"
    fi
    
    echo -e "${GREEN}Running: ${exp_name}${NC}"
    echo "  Concept 1: '$concept1'"
    echo "  Concept 2: '$concept2'"
    echo "  Base prompts: $base_prompts"
    echo "  Method: $([ "$use_cfgpp" = "true" ] && echo "CFG++" || echo "CFG")"
    echo "  Guidance: $guidance"
    echo ""
    
    python -m examples.semantic_difference_experiment \
        --workdir "$workdir" \
        --device "$DEVICE" \
        --model "$MODEL" \
        --NFE "$NFE" \
        --seed "$SEED" \
        --cfg_guidance "$guidance" \
        --concept1 "$concept1" \
        --concept2 "$concept2" \
        --base_prompts "$base_prompts" \
        --strengths $STRENGTHS \
        $method_flag
    
    echo -e "${YELLOW}Completed: ${exp_name}${NC}"
    echo ""
}

# Function to run comparison (both CFG and CFG++)
run_comparison() {
    local concept1="$1"
    local concept2="$2"
    local base_prompts="$3"
    local exp_name="$4"
    
    local workdir="${BASE_DIR}/${exp_name}_comparison"
    
    echo -e "${GREEN}Running comparison: ${exp_name}${NC}"
    echo "  Concept 1: '$concept1'"
    echo "  Concept 2: '$concept2'"
    echo "  Base prompts: $base_prompts"
    echo "  Comparing CFG vs CFG++"
    echo ""
    
    python -m examples.semantic_difference_experiment \
        --workdir "$workdir" \
        --device "$DEVICE" \
        --model "$MODEL" \
        --NFE "$NFE" \
        --seed "$SEED" \
        --cfg_guidance "$CFG_GUIDANCE" \
        --concept1 "$concept1" \
        --concept2 "$concept2" \
        --base_prompts "$base_prompts" \
        --strengths $STRENGTHS \
        --compare_both
    
    echo -e "${YELLOW}Completed comparison: ${exp_name}${NC}"
    echo ""
}

# ============================================
# Experiment 1: Gender Identity
# ============================================
echo -e "${BLUE}=== Experiment 1: Gender Identity ===${NC}"
run_experiment \
    "a girl" \
    "a boy" \
    "a portrait of a person" \
    false \
    "gender_portrait"

run_experiment \
    "a woman" \
    "a man" \
    "a person standing in a field" \
    false \
    "gender_standing"

# ============================================
# Experiment 2: Age
# ============================================
echo -e "${BLUE}=== Experiment 2: Age ===${NC}"
run_experiment \
    "a young person" \
    "an old person" \
    "a portrait of a person" \
    false \
    "age_portrait"

run_experiment \
    "a child" \
    "an elderly person" \
    "a person walking" \
    false \
    "age_walking"

# ============================================
# Experiment 3: Emotion
# ============================================
echo -e "${BLUE}=== Experiment 3: Emotion ===${NC}"
run_experiment \
    "a happy person" \
    "a sad person" \
    "a portrait of a person" \
    false \
    "emotion_portrait"

run_experiment \
    "a smiling person" \
    "a frowning person" \
    "a person standing" \
    false \
    "emotion_standing"

# ============================================
# Experiment 4: Art Style
# ============================================
echo -e "${BLUE}=== Experiment 4: Art Style ===${NC}"
run_experiment \
    "a realistic person" \
    "a cartoon person" \
    "a portrait of a person" \
    false \
    "style_realistic_cartoon"

run_experiment \
    "a photograph of a person" \
    "a painting of a person" \
    "a portrait of a person" \
    false \
    "style_photo_painting"

# ============================================
# Experiment 5: Hair Color
# ============================================
echo -e "${BLUE}=== Experiment 5: Hair Color ===${NC}"
run_experiment \
    "a person with blonde hair" \
    "a person with black hair" \
    "a portrait of a person" \
    false \
    "hair_blonde_black"

run_experiment \
    "a person with red hair" \
    "a person with brown hair" \
    "a portrait of a person" \
    false \
    "hair_red_brown"

# ============================================
# Experiment 6: Clothing Style
# ============================================
echo -e "${BLUE}=== Experiment 6: Clothing Style ===${NC}"
run_experiment \
    "a person in formal wear" \
    "a person in casual wear" \
    "a person standing" \
    false \
    "clothing_formal_casual"

# ============================================
# Experiment 7: Environment/Setting
# ============================================
echo -e "${BLUE}=== Experiment 7: Environment ===${NC}"
run_experiment \
    "a person in a city" \
    "a person in nature" \
    "a person standing" \
    false \
    "environment_city_nature"

# ============================================
# Experiment 8: Time of Day
# ============================================
echo -e "${BLUE}=== Experiment 8: Time of Day ===${NC}"
run_experiment \
    "a person in daylight" \
    "a person at night" \
    "a person standing" \
    false \
    "time_day_night"

# ============================================
# Experiment 9: CFG++ Comparisons
# ============================================
echo -e "${BLUE}=== Experiment 9: CFG++ Method ===${NC}"
run_experiment \
    "a girl" \
    "a boy" \
    "a portrait of a person" \
    true \
    "gender_portrait"

run_experiment \
    "a happy person" \
    "a sad person" \
    "a portrait of a person" \
    true \
    "emotion_portrait"

# ============================================
# Experiment 10: Side-by-Side Comparisons
# ============================================
echo -e "${BLUE}=== Experiment 10: CFG vs CFG++ Comparisons ===${NC}"
run_comparison \
    "a girl" \
    "a boy" \
    "a portrait of a person" \
    "gender_cfg_vs_cfgpp"

run_comparison \
    "a happy person" \
    "a sad person" \
    "a portrait of a person" \
    "emotion_cfg_vs_cfgpp"

run_comparison \
    "a young person" \
    "an old person" \
    "a portrait of a person" \
    "age_cfg_vs_cfgpp"

# ============================================
# Summary
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to: $BASE_DIR"
echo ""
echo "Experiment directories:"
ls -d ${BASE_DIR}/*/ 2>/dev/null | sed 's|$BASE_DIR/||' || echo "  (check $BASE_DIR for results)"
echo ""

