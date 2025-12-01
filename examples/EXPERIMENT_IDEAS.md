# Experiment Ideas for CFG++ Research

This document outlines interesting experiments to extend the baseline work, with a focus on text encoder conditioning and other novel directions.

## Current Baseline

1. **CFG vs CFG++ Evaluation**: Systematic comparison across prompts and guidance scales
2. **Semantic Difference Experiment**: Applying difference vectors between concepts (e.g., "girl" vs "boy") to base prompts

## Proposed Experiments

### 1. Prompt Interpolation and Blending

**Concept**: Interpolate between multiple prompts in embedding space to create smooth transitions.

**Research Question**: How do CFG and CFG++ respond to interpolated embeddings? Can we create smooth semantic transitions?

**Implementation**:
- Given prompts P1, P2, ..., Pn, compute embeddings E1, E2, ..., En
- Interpolate: `E_interp = α₁E₁ + α₂E₂ + ... + αₙEₙ` where Σαᵢ = 1
- Generate images with interpolated embeddings
- Compare CFG vs CFG++ sensitivity to interpolation

**Variants**:
- **Linear interpolation**: Between 2 prompts (α, 1-α)
- **Multi-prompt blending**: Weighted combination of 3+ prompts
- **Spherical interpolation**: Use SLERP for smoother transitions
- **Timestep-dependent interpolation**: Change interpolation weights during sampling

**Expected Insights**:
- How embedding space structure affects generation
- Whether CFG++ is more robust to embedding perturbations
- Visual quality of interpolated results

---

### 2. Learnable Prompt Tuning (Soft Prompts)

**Concept**: Optimize learnable embedding vectors directly instead of using text prompts.

**Research Question**: Can we find optimal embeddings that generate specific visual properties? How do CFG and CFG++ respond to optimized embeddings?

**Implementation**:
- Initialize learnable embedding vectors (soft prompts)
- Define loss function (e.g., CLIP similarity to target description, perceptual loss)
- Optimize embeddings using gradient descent
- Compare optimized embeddings with text-based embeddings
- Test with both CFG and CFG++

**Variants**:
- **Target-driven optimization**: Optimize to match a target image or description
- **Style transfer**: Optimize embeddings to achieve specific artistic styles
- **Negative prompt tuning**: Optimize negative embeddings to suppress unwanted features
- **Multi-objective optimization**: Balance multiple goals (e.g., realism + style)

**Expected Insights**:
- Whether learned embeddings outperform text prompts
- How CFG++ guidance affects optimization dynamics
- Embedding space regions that produce better results

---

### 3. Timestep-Dependent Prompt Conditioning

**Concept**: Use different prompts or prompt weights at different timesteps during sampling.

**Research Question**: Does early vs. late conditioning affect generation differently? How do CFG and CFG++ respond to time-varying conditioning?

**Implementation**:
- Define prompt schedule: `prompt(t)` or `weight(t)` for different prompts
- Early timesteps: Use coarse/structural prompts (e.g., "a person")
- Late timesteps: Use detailed prompts (e.g., "a person with blue eyes, wearing red shirt")
- Or blend multiple prompts with time-varying weights
- Compare with constant conditioning

**Variants**:
- **Progressive refinement**: Start with simple prompt, add details over time
- **Style-content separation**: Use content prompt early, style prompt late
- **Negative prompt scheduling**: Vary negative prompt strength over time
- **Multi-scale conditioning**: Different prompts for different noise levels

**Expected Insights**:
- Optimal conditioning strategies for different timesteps
- Whether CFG++ benefits more from time-varying conditioning
- Understanding of when semantic information matters most

---

### 4. Embedding Space Direction Discovery

**Concept**: Find meaningful directions in embedding space that control specific visual attributes.

**Research Question**: Can we discover interpretable directions that consistently modify specific attributes (e.g., age, emotion, style)?

**Implementation**:
- Collect pairs of prompts that differ in one attribute (e.g., "young person" vs "old person")
- Compute difference vectors for many pairs
- Use PCA or other methods to find principal directions
- Apply discovered directions to new prompts
- Compare with simple difference vectors (current baseline)

**Variants**:
- **PCA-based directions**: Find principal components of difference vectors
- **GAN-style directions**: Use techniques from StyleGAN (e.g., SeFa, GANSpace)
- **Attribute-specific directions**: Separate directions for different attributes
- **Orthogonal directions**: Ensure directions don't interfere with each other

**Expected Insights**:
- Better semantic directions than simple differences
- Whether certain directions work better with CFG++
- Understanding of embedding space structure

---

### 5. Cross-Attention Manipulation

**Concept**: Directly manipulate cross-attention maps between text and image features.

**Research Question**: How does modifying attention patterns affect generation? Do CFG and CFG++ respond differently?

**Implementation**:
- Extract cross-attention maps from UNet during sampling
- Modify attention weights (e.g., amplify certain tokens, suppress others)
- Re-inject modified attention into the sampling process
- Compare with standard attention

**Variants**:
- **Token importance**: Identify which tokens have most influence
- **Attention editing**: Selectively enhance/suppress specific concepts
- **Multi-head attention analysis**: Study different attention heads
- **Timestep-dependent attention**: Modify attention differently at different steps

**Expected Insights**:
- Which parts of prompts matter most
- Whether CFG++ uses attention differently
- Fine-grained control over generation

---

### 6. Negative Prompt Optimization

**Concept**: Systematically explore negative prompt space to understand what CFG++ suppresses.

**Research Question**: How do different negative prompts affect generation? Is CFG++ more or less sensitive to negative prompts?

**Implementation**:
- Test various negative prompts (empty, standard, concept-specific, etc.)
- Measure effect on generation (e.g., CLIP scores, visual quality)
- Compare CFG vs CFG++ sensitivity
- Find optimal negative prompts for specific tasks

**Variants**:
- **Concept-specific negatives**: Negative prompts targeting specific unwanted features
- **Negative prompt interpolation**: Blend multiple negative prompts
- **Adversarial negatives**: Find negatives that break generation
- **Positive-negative balance**: Study interaction between positive and negative prompts

**Expected Insights**:
- Optimal negative prompt strategies
- Whether CFG++ needs different negative prompts
- Understanding of what negative prompts actually suppress

---

### 7. Embedding Perturbation Robustness

**Concept**: Systematically add noise/perturbations to embeddings and measure robustness.

**Research Question**: How robust are CFG and CFG++ to embedding perturbations? Which is more stable?

**Implementation**:
- Add Gaussian noise to embeddings: `E_perturbed = E + ε * N(0,1)`
- Vary noise magnitude ε
- Measure quality degradation (e.g., CLIP score, FID, perceptual metrics)
- Compare CFG vs CFG++ robustness curves

**Variants**:
- **Directional perturbations**: Perturb along specific directions
- **Sparse perturbations**: Only modify certain embedding dimensions
- **Adversarial perturbations**: Find worst-case perturbations
- **Timestep-dependent perturbations**: Perturb at different stages

**Expected Insights**:
- Robustness comparison between CFG and CFG++
- Which embedding dimensions are most critical
- Failure modes of each method

---

### 8. Multi-Prompt Composition

**Concept**: Combine multiple prompts using various composition strategies.

**Research Question**: How can we effectively combine multiple concepts? Do CFG and CFG++ handle composition differently?

**Implementation**:
- **Averaging**: `E = (E₁ + E₂) / 2`
- **Weighted combination**: `E = αE₁ + (1-α)E₂`
- **Concatenation**: Use multiple prompts in sequence
- **Hierarchical**: Main prompt + detail prompts
- Compare composition strategies

**Variants**:
- **Object composition**: Combine multiple objects (e.g., "cat" + "dog")
- **Style-content**: Combine style and content prompts
- **Attribute addition**: Add attributes to base prompt
- **Conditional composition**: Compose based on compatibility

**Expected Insights**:
- Best practices for prompt composition
- Whether CFG++ handles complex compositions better
- Limits of embedding space composition

---

### 9. Prompt Inversion and Analysis

**Concept**: Given a generated image, find the prompt/embedding that would produce it.

**Research Question**: Can we invert the generation process? What does this tell us about embedding space?

**Implementation**:
- Start with a target image
- Optimize embedding to generate similar image
- Compare inverted embeddings with original prompts
- Analyze embedding space structure

**Variants**:
- **CLIP-based inversion**: Use CLIP to find matching text
- **Gradient-based inversion**: Optimize embeddings directly
- **Embedding space analysis**: Study distribution of effective embeddings
- **Canonical embeddings**: Find "canonical" embeddings for concepts

**Expected Insights**:
- Whether embeddings are unique for concepts
- Embedding space structure and redundancy
- How CFG++ affects invertibility

---

### 10. Guidance Scale vs Embedding Strength Interaction

**Concept**: Study interaction between guidance scale and embedding manipulation strength.

**Research Question**: How do guidance scale and embedding modifications interact? Is there an optimal combination?

**Implementation**:
- Vary both guidance scale and embedding manipulation strength
- Create 2D grid of results
- Find optimal combinations for different tasks
- Compare CFG vs CFG++ interaction patterns

**Variants**:
- **Semantic difference strength**: Vary both guidance and difference strength
- **Prompt interpolation weight**: Vary guidance and interpolation parameter
- **Multi-prompt weights**: Vary guidance and prompt blending weights
- **Task-specific optimization**: Find optimal settings for specific tasks

**Expected Insights**:
- Optimal hyperparameter combinations
- Whether CFG++ needs different settings
- Understanding of guidance mechanism

---

## Implementation Priority

### High Priority (Most Interesting + Feasible)

1. **Prompt Interpolation and Blending** - Easy to implement, clear visual results
2. **Timestep-Dependent Prompt Conditioning** - Novel and could reveal interesting dynamics
3. **Embedding Perturbation Robustness** - Quantitative comparison, important for understanding stability
4. **Multi-Prompt Composition** - Practical and interesting

### Medium Priority

5. **Learnable Prompt Tuning** - More complex but very interesting
6. **Embedding Space Direction Discovery** - Extends current semantic difference work
7. **Negative Prompt Optimization** - Practical and could improve results

### Lower Priority (More Complex)

8. **Cross-Attention Manipulation** - Requires UNet access, more complex
9. **Prompt Inversion** - Interesting but may be less directly useful
10. **Guidance Scale vs Embedding Strength** - More of an analysis than new experiment

---

## Recommended Next Steps

1. **Start with Prompt Interpolation** - Quick win, builds on existing code
2. **Implement Timestep-Dependent Conditioning** - Novel contribution
3. **Add Robustness Analysis** - Quantitative metrics to complement visual results
4. **Explore Learnable Prompts** - If time permits, very interesting direction

Each experiment should:
- Compare CFG vs CFG++
- Generate visual results for presentation
- Include quantitative metrics where possible
- Document findings and insights


