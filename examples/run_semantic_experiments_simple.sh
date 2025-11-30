#!/bin/bash

# Simplified Semantic Difference Experiments Runner
# Run specific experiments by uncommenting the ones you want

set -e

# Configuration (can be overridden with environment variables)
BASE_DIR="${BASE_DIR:-examples/workdir/semantic_diff}"
DEVICE="${DEVICE:-cuda}"
MODEL="${MODEL:-sd15}"
NFE="${NFE:-50}"
SEED="${SEED:-42}"
CFG_GUIDANCE="${CFG_GUIDANCE:-7.5}"
CFGPP_GUIDANCE="${CFGPP_GUIDANCE:-0.6}"
STRENGTHS="${STRENGTHS:--1.0 -0.5 0.0 0.5 1.0}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Semantic Difference Experiments${NC}"
echo ""

# Uncomment the experiments you want to run:

# Gender experiments
python -m examples.semantic_difference_experiment \
    --workdir "${BASE_DIR}/gender_girl_boy" \
    --device "$DEVICE" \
    --model "$MODEL" \
    --NFE "$NFE" \
    --seed "$SEED" \
    --cfg_guidance "$CFG_GUIDANCE" \
    --concept1 "a girl" \
    --concept2 "a boy" \
    --base_prompts "a portrait of a person" \
    --strengths $STRENGTHS

# Age experiments
# python -m examples.semantic_difference_experiment \
#     --workdir "${BASE_DIR}/age_young_old" \
#     --device "$DEVICE" \
#     --model "$MODEL" \
#     --NFE "$NFE" \
#     --seed "$SEED" \
#     --cfg_guidance "$CFG_GUIDANCE" \
#     --concept1 "a young person" \
#     --concept2 "an old person" \
#     --base_prompts "a portrait of a person" \
#     --strengths $STRENGTHS

# Emotion experiments
# python -m examples.semantic_difference_experiment \
#     --workdir "${BASE_DIR}/emotion_happy_sad" \
#     --device "$DEVICE" \
#     --model "$MODEL" \
#     --NFE "$NFE" \
#     --seed "$SEED" \
#     --cfg_guidance "$CFG_GUIDANCE" \
#     --concept1 "a happy person" \
#     --concept2 "a sad person" \
#     --base_prompts "a portrait of a person" \
#     --strengths $STRENGTHS

# Style experiments
# python -m examples.semantic_difference_experiment \
#     --workdir "${BASE_DIR}/style_realistic_cartoon" \
#     --device "$DEVICE" \
#     --model "$MODEL" \
#     --NFE "$NFE" \
#     --seed "$SEED" \
#     --cfg_guidance "$CFG_GUIDANCE" \
#     --concept1 "a realistic person" \
#     --concept2 "a cartoon person" \
#     --base_prompts "a portrait of a person" \
#     --strengths $STRENGTHS

# CFG++ version
# python -m examples.semantic_difference_experiment \
#     --workdir "${BASE_DIR}/gender_girl_boy_cfgpp" \
#     --device "$DEVICE" \
#     --model "$MODEL" \
#     --NFE "$NFE" \
#     --seed "$SEED" \
#     --cfg_guidance "$CFGPP_GUIDANCE" \
#     --concept1 "a girl" \
#     --concept2 "a boy" \
#     --base_prompts "a portrait of a person" \
#     --strengths $STRENGTHS \
#     --use_cfgpp

# Comparison (both CFG and CFG++)
# python -m examples.semantic_difference_experiment \
#     --workdir "${BASE_DIR}/gender_comparison" \
#     --device "$DEVICE" \
#     --model "$MODEL" \
#     --NFE "$NFE" \
#     --seed "$SEED" \
#     --cfg_guidance "$CFG_GUIDANCE" \
#     --concept1 "a girl" \
#     --concept2 "a boy" \
#     --base_prompts "a portrait of a person" \
#     --strengths $STRENGTHS \
#     --compare_both

echo -e "${GREEN}Done!${NC}"

