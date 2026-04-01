# Experimental Analysis Report

## Comprehensive Comparison of Pipeline Configurations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experimental Setup](#2-experimental-setup)
3. [Embedding Model Comparison](#3-embedding-model-comparison)
4. [Classifier Comparison](#4-classifier-comparison)
5. [Multi-Label Strategy Analysis](#5-multi-label-strategy-analysis)
6. [Feature Engineering Impact](#6-feature-engineering-impact)
7. [Best Configuration Analysis](#7-best-configuration-analysis)
8. [Key Findings and Trade-offs](#8-key-findings-and-trade-offs)
9. [Recommendations](#9-recommendations)
10. [Embedded/Bare-Metal Hardware Feature Expansion](#10-embeddedbare-metal-hardware-feature-expansion)

---

## 1. Executive Summary

### Key Findings

| Finding | Evidence | Impact |
|---------|----------|--------|
| **Qwen3-4B provides best embeddings** | +4.3% F1 vs CodeBERT | Worth the 13x slowdown for quality |
| **Logistic Regression dominates** | Best for 3/4 classifiers | Simple models outperform complex ones |
| **Random Forest severely overfits** | 0.379 F1 (worst) | Avoid for high-dimensional embeddings |
| **Hybrid features help marginally** | +0.5% F1 | Small but consistent improvement |
| **Purpose embeddings help error_handling** | +8.4% F1 on error task | Task-specific benefit |
| **All models overfit severely** | Train F1=1.0, Test F1~0.62 | Fundamental data limitation |
| **HW features help weaker embeddings** | CodeBERT Complexity +6.9pts, Error +5.7pts | Domain-specific features compensate for less expressive embeddings |

### Best Configurations by Task

| Task | Configuration | Test F1 |
|------|---------------|---------|
| **Side Effects** | Qwen3-4B + MLP + all features | **0.629** |
| **Complexity** | Nomic + LR | **0.704** |
| **Error Handling** | CodeBERT + LR + `--hardware-features` | **0.756** |
| **Overall Best** | Qwen3-4B + MLP + hybrid + AST + purpose + threshold | **0.629** |

---

## 2. Experimental Setup

### Dataset Characteristics

```
Total Functions: 780
Train/Test Split: 70/30 (546 train, 234 test)
Random Seed: 42

Side Effect Distribution:
├── hardware:     315 (40.4%)
├── global_state: 294 (37.7%)
├── memory:       281 (36.0%)
├── none:         110 (14.1%)
├── io:           105 (13.5%)
├── network:        9 (1.2%)
└── assertions:     1 (0.1%)
```

### Experimental Dimensions

| Dimension | Options Tested |
|-----------|----------------|
| Embedding Models | CodeBERT, Qwen3-0.6B, Qwen3-4B, Nomic |
| Classifiers | Logistic Regression, SVM, Random Forest, MLP |
| Multi-label Strategy | OneVsRest, Binary Classifiers |
| Class Imbalance | Balanced weights, SMOTE, Threshold Tuning |
| Features | Base, `--hybrid-features` (20), `--hardware-features` (40), +AST, +Purpose, All Combined |

### Evaluation Protocol

- **Primary Metric**: Test F1 (macro) - equal weight to all classes
- **Secondary Metrics**: Test Accuracy, Complexity F1, Error Handling F1
- **Overfitting Indicator**: Train F1 - Test F1 gap

### Feature Flag Reference

| Flag | Features | Description |
|------|----------|-------------|
| *(none)* | 0 | Pure embedding only |
| `--hybrid-features` | 20 | Base regex features: memory/IO/HW APIs, code metrics, keywords |
| `--hardware-features` | 40 | All 20 base + 20 embedded/bare-metal HW features (MMIO, regmap, IRQ, SPI, I2C, GPIO, etc). Implies `--hybrid-features` |
| `--ast-features` | 15 | AST-based: statement distribution, control-flow depth, cyclomatic complexity |
| `--purpose-embeddings` | +N | high_level_purpose text re-embedded (adds N dims matching embedder) |

---

## 3. Embedding Model Comparison

### Results Table

| Model | Dimensions | Test F1 | Test Acc | Complexity F1 | Error F1 | Embed Time |
|-------|------------|---------|----------|---------------|----------|------------|
| CodeBERT | 768 | 0.587 | 0.436 | 0.630 | 0.699 | 5s |
| Qwen3-0.6B | 1024 | 0.617 | 0.568 | 0.636 | 0.659 | 18s |
| **Qwen3-4B** | 2560 | **0.630** | 0.526 | 0.607 | 0.691 | 65s |
| Nomic | 3584 | 0.622 | 0.517 | **0.704** | **0.702** | 415s |

### Analysis

#### CodeBERT (Baseline)
- **Strengths**: Fast (16 it/s), low memory, well-documented
- **Weaknesses**: Lowest F1 (0.587), lowest accuracy (0.436)
- **Use case**: Quick prototyping, resource-constrained environments

#### Qwen3-0.6B
- **Strengths**: Good balance of speed (5.3 it/s) and quality
- **Weaknesses**: Moderate performance
- **Use case**: Default choice for most applications

#### Qwen3-4B (Best for Side Effects)
- **Strengths**: Highest side effects F1 (0.630), best similarity overlap (0.755)
- **Weaknesses**: 13x slower than CodeBERT, high memory
- **Use case**: When quality matters more than speed

#### Nomic
- **Strengths**: Best Complexity F1 (0.704), good Error Handling (0.702)
- **Weaknesses**: Slowest (4.2s/batch), highest dimensions (3584)
- **Use case**: Complexity/error analysis tasks

### Key Insight: Quality vs Speed Trade-off

```
                Quality (F1)
                    ↑
           0.63 ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ Qwen3-4B
                      │              ╱
           0.62 ─ ─ ─ ┼ ─ ─ ─ ─ ─ Nomic
                      │          ╱
           0.61 ─ ─ ─ ┼ ─ ─ ─ Qwen3-0.6B
                      │      ╱
           0.59 ─ ─ ─ ┼ ─ CodeBERT
                      │
                      └────────────────────────→ Speed
                        5s    18s   65s   415s
```

**Recommendation**: Qwen3-0.6B for development, Qwen3-4B for production.

---

## 4. Classifier Comparison

### Results Table (with Qwen3-0.6B embeddings)

| Classifier | Test F1 | Test Acc | Train F1 | Overfitting Gap | Complexity F1 | Error F1 |
|------------|---------|----------|----------|-----------------|---------------|----------|
| **Logistic Regression** | **0.617** | **0.568** | 1.000 | 0.383 | 0.636 | 0.659 |
| SVM (RBF) | 0.606 | 0.479 | 0.860 | 0.254 | **0.663** | 0.688 |
| MLP | 0.596 | 0.551 | 1.000 | 0.404 | 0.551 | **0.703** |
| Random Forest | 0.379 | 0.385 | 1.000 | **0.621** | 0.417 | 0.582 |

### Analysis

#### Logistic Regression (Best Overall)
```
Pros:
✓ Highest Test F1 (0.617) and Accuracy (0.568)
✓ Simple, interpretable, fast training
✓ Works well with high-dimensional embeddings
✓ Natural probability outputs for threshold tuning

Cons:
✗ Linear decision boundary may miss complex patterns
✗ Still overfits (Train=1.0, Test=0.617)
```

#### SVM (RBF Kernel)
```
Pros:
✓ Lowest overfitting gap (0.254)
✓ Best Complexity F1 (0.663)
✓ Handles high dimensions via kernel trick

Cons:
✗ Slow training on large datasets
✗ Lower Test F1 (0.606)
✗ No native probability estimates
```

#### MLP (Neural Network)
```
Pros:
✓ Best Error Handling F1 (0.703)
✓ Can learn non-linear patterns
✓ Flexible architecture

Cons:
✗ Severe overfitting (gap=0.404)
✗ Sensitive to hyperparameters
✗ Lower overall F1 (0.596)
```

#### Random Forest (Worst)
```
Pros:
✓ Feature importance analysis
✓ Handles non-linear relationships

Cons:
✗ Catastrophic overfitting (gap=0.621)
✗ Lowest Test F1 (0.379) and Accuracy (0.385)
✗ Poor for high-dimensional sparse data
```

### Why Random Forest Failed

**Hypothesis**: Random Forest creates deep trees that memorize training data.

**Evidence**:
- Train F1: 1.000 (perfect memorization)
- Test F1: 0.379 (complete failure to generalize)
- Gap: 0.621 (62.1% drop)

**Root Cause**: With 1024+ dimensional embeddings and only 546 training samples, each tree can easily find splits that perfectly separate training data but don't generalize.

**Solution**: Use `--rf-max-depth 5` or `--embed-dim 64` to constrain complexity.

---

## 5. Multi-Label Strategy Analysis

### Results Table (with Qwen3-0.6B + LR)

| Strategy | Test F1 | Test Acc | Per-Class F1 Distribution |
|----------|---------|----------|---------------------------|
| OneVsRest (standard) | 0.617 | 0.568 | Balanced across classes |
| Binary Classifiers | 0.617 | 0.568 | global=0.724, hardware=0.943, io=0.585, memory=0.726, network=0.500, none=0.842 |
| Binary + SMOTE | 0.616 | 0.556 | Slightly worse |
| Binary + Threshold Tuning | 0.617 | 0.568 | Same overall, better thresholds |

### Analysis

#### Standard OneVsRest
- Trains K binary classifiers, one per class
- Predicts all classes simultaneously
- **Result**: Baseline performance (0.617 F1)

#### Binary Classifiers (Separate Models)
- Allows per-class hyperparameter tuning
- Can skip rare classes (`assertions` with 1 sample)
- **Result**: Same F1, but provides per-class visibility

**Per-Class Performance**:
```
hardware:     F1=0.943  ████████████████████████ (Best - 230 samples)
none:         F1=0.842  █████████████████████    (Good - 71 samples)
memory:       F1=0.726  ██████████████████       (Good - 209 samples)
global_state: F1=0.724  ██████████████████       (Good - 193 samples)
io:           F1=0.585  ██████████████           (Moderate - 76 samples)
network:      F1=0.500  ████████████             (Poor - 5 samples)
assertions:   SKIPPED   (only 1 sample)
```

#### SMOTE Oversampling
- Creates synthetic samples for minority classes
- **Expected**: Improve minority class performance
- **Actual**: Slight degradation (0.616 vs 0.617)

**Why SMOTE Didn't Help**:
1. Synthetic samples in embedding space may not be semantically valid
2. SMOTE assumes Euclidean neighborhood, embeddings are cosine-similar
3. Oversampling increases training set size, potentially more overfitting

#### Threshold Tuning
- Optimizes decision threshold per class (not just 0.5)
- **Optimal Thresholds Found**:
  ```
  global_state: 0.10 (lower threshold = more predictions)
  hardware:     0.40
  io:           0.20
  memory:       0.70 (higher threshold = fewer predictions)
  network:      0.10
  none:         0.30
  ```

**Insight**: Threshold tuning helps calibrate predictions but doesn't improve overall F1 significantly because the underlying model limitations persist.

---

## 6. Feature Engineering Impact

### Results Table (with Qwen3-0.6B + LR)

| Features | Dims | Test F1 | Test Acc | Complexity F1 | Error F1 | Delta F1 |
|----------|------|---------|----------|---------------|----------|----------|
| **Baseline** | 1024 | 0.617 | 0.568 | 0.636 | 0.659 | - |
| +Hybrid | 1044 | 0.622 | 0.560 | 0.608 | 0.678 | **+0.005** |
| +AST | 1039 | 0.619 | 0.564 | **0.647** | 0.682 | +0.002 |
| +Purpose | 2048 | 0.621 | 0.560 | 0.634 | **0.743** | +0.004 |
| +Hybrid+AST | 1059 | 0.617 | 0.551 | 0.645 | 0.690 | +0.000 |

### Feature-by-Feature Analysis

#### Hybrid Features (+20 regex features)
```
Impact: +0.5% F1 (0.617 → 0.622)

Top Contributing Features:
├── memory_present:    Detects malloc/free calls
├── io_present:        Detects printf/fopen calls
├── hardware_present:  Detects volatile/ioctl
└── error_present:     Detects errno/assert

Why It Helps:
- Embeddings capture syntax patterns
- Regex explicitly detects API calls
- Direct signal for side effect classification
```

#### AST Features (+15 tree-sitter features)
```
Impact: +0.2% F1 (0.617 → 0.619)

Top Contributing Features:
├── stmt_call_expression_ratio:  Function call density
├── max_loop_depth:              Nesting complexity
├── cyclomatic_complexity:       Decision points
└── stmt_if_statement_ratio:     Branching patterns

Why Limited Impact:
- AST features more useful for complexity (0.647 vs 0.636)
- Side effects determined by API calls, not structure
```

#### Purpose Embeddings (+1024 purpose dims)
```
Impact: +0.4% F1 for side_effects, +8.4% for error_handling

Why Purpose Helps Error Handling:
- Purpose text: "Returns error code on failure"
- Directly mentions error handling approach
- Strong semantic signal

Why Limited for Side Effects:
- Purpose: "Allocates memory buffer"
- Embeddings already capture this from code
- Redundant information
```

#### Combined Features
```
Hybrid + AST (without purpose): 0.617 F1 (no improvement)

Observation: Features interfere when combined
- Hybrid + AST = 1059 dims
- More features = more overfitting potential
- Regularization needed for feature fusion
```

### Feature Engineering Verdict

| Feature Set | Worth Adding? | Reason |
|-------------|---------------|--------|
| Hybrid | **Yes** | +0.5% F1, minimal cost |
| AST | **Maybe** | Better for complexity task |
| Purpose | **Yes for error handling** | +8.4% F1 for that task |
| All Combined | **Needs tuning** | Risk of overfitting |

---

## 7. Best Configuration Analysis

### Overall Best: Qwen3-4B + MLP + All Features

```bash
python run_pipeline.py \
    --skip-labeling \
    --embedder qwen3-4b \
    --side-effects-clf mlp \
    --complexity-clf random_forest \
    --error-handling-clf logistic_regression \
    --hardware-features \
    --ast-features \
    --purpose-embeddings \
    --threshold-tuning \
    --smote
```

**Results**:
| Metric | Value |
|--------|-------|
| Side Effects F1 | **0.629** |
| Complexity F1 | 0.591 |
| Error Handling F1 | **0.749** |
| Test Accuracy | 0.551 |

**Per-Class Optimal Thresholds**:
```
assertions:   0.50 (default, only 1 sample)
global_state: 0.10 (lowered to catch more)
hardware:     0.40
io:           0.10 (lowered for minority class)
memory:       0.40
network:      0.10 (lowered for minority class)
none:         0.10
```

### Why This Configuration Works

1. **Qwen3-4B Embeddings**: Highest quality representations
2. **MLP for Side Effects**: Captures non-linear feature interactions
3. **Threshold Tuning**: Calibrates predictions per class
4. **All Features**: Maximum information (5155 dims)
5. **SMOTE**: Helps with class imbalance for complexity

### Performance Breakdown by Class

```
                    Test F1 Score
    0.0   0.2   0.4   0.6   0.8   1.0
     │     │     │     │     │     │
hardware     ███████████████████████████████████████ 0.944
none         ███████████████████████████████████ 0.911
memory       ██████████████████████████████ 0.782
global_state █████████████████████████████ 0.736
io           ████████████████████████ 0.627
network      ███████████████ 0.400
assertions   (skipped - 1 sample)
```

---

## 8. Key Findings and Trade-offs

### Finding 1: All Models Severely Overfit

**Evidence**:
| Model | Train F1 | Test F1 | Gap |
|-------|----------|---------|-----|
| Logistic Regression | 1.000 | 0.617 | 0.383 |
| Random Forest | 1.000 | 0.379 | 0.621 |
| MLP | 1.000 | 0.596 | 0.404 |

**Root Cause**:
- 780 functions is insufficient for 1024-5155 dimensional features
- High-dimensional embeddings have many degrees of freedom

**Trade-off**:
- More features → Better training fit → Worse generalization
- Fewer features → Worse training fit → Better generalization

**Recommendation**: Use PCA (`--embed-dim 128`) or stronger regularization (`--lr-c 0.1`)

### Finding 2: Embedding Quality > Classifier Complexity

**Evidence**:
```
CodeBERT + LR:  0.587 F1
Qwen3-4B + LR:  0.630 F1  (+7.3%)

Qwen3 + LR:     0.617 F1
Qwen3 + MLP:    0.596 F1  (-3.4%)
```

**Insight**: Upgrading embeddings provides more gain than upgrading classifier.

**Trade-off**:
- Better embeddings = More compute time
- Complex classifiers = More overfitting risk

### Finding 3: Simple Classifiers Win

**Evidence**:
| Classifier | Test F1 | Complexity |
|------------|---------|------------|
| Logistic Regression | 0.617 | O(n*d) |
| Random Forest | 0.379 | O(n*d*trees*depth) |

**Why**: With limited data, simple models generalize better.

**Trade-off**: Simple models may miss non-linear patterns, but on small datasets, non-linear models just memorize.

### Finding 4: Feature Engineering Has Diminishing Returns

**Evidence**:
```
Baseline:           0.617 F1
+Hybrid:            0.622 F1 (+0.8%)
+Hybrid+AST:        0.617 F1 (+0.0%)
+All Features:      0.617 F1 (+0.0%)
```

**Insight**: Adding more features doesn't always help; it can hurt by increasing overfitting.

**Trade-off**: Each additional feature dimension needs proportionally more training data.

### Finding 5: Class Imbalance Techniques Have Limited Effect

**Evidence**:
| Technique | Test F1 |
|-----------|---------|
| Baseline | 0.617 |
| +SMOTE | 0.616 |
| +Binary Classifiers | 0.617 |
| +Threshold Tuning | 0.617 |

**Why Limited Effect**:
- SMOTE creates synthetic points that may not be semantically valid
- The fundamental issue is label distribution in the dataset
- 9 `network` samples and 1 `assertion` sample cannot be learned

**Trade-off**: Aggressive oversampling can introduce noise.

### Finding 6: Task-Specific Configurations Matter

**Evidence**:
| Task | Best Classifier | Best Features |
|------|-----------------|---------------|
| side_effects | MLP | `--hardware-features` + AST + Purpose |
| complexity | LR or SVM | AST |
| error_handling | LR | `--hardware-features` or Purpose |

**Insight**: One-size-fits-all doesn't work; each task has different optimal settings.

---

## 9. Recommendations

### For Maximum Accuracy

```bash
python run_pipeline.py \
    --embedder qwen3-4b \
    --side-effects-clf mlp \
    --complexity-clf svm \
    --error-handling-clf logistic_regression \
    --hardware-features \
    --ast-features \
    --purpose-embeddings \
    --threshold-tuning \
    --skip-labeling
```
**Expected**: Side Effects F1 ~0.63, Error Handling F1 ~0.75

### For Best Error Handling & Complexity (Hardware-Heavy Code)

```bash
python run_pipeline.py \
    --embedder codebert \
    --classifier logistic_regression \
    --hardware-features \
    --skip-labeling
```
**Expected**: Complexity F1 ~0.70, Error Handling F1 ~0.76 (best observed for error handling)

### For Fast Development

```bash
python run_pipeline.py \
    --embedder codebert \
    --classifier logistic_regression \
    --hybrid-features \
    --skip-labeling
```
**Expected**: Side Effects F1 ~0.59, 10x faster

### For Reduced Overfitting

```bash
python run_pipeline.py \
    --embedder qwen3 \
    --classifier logistic_regression \
    --embed-dim 128 \
    --lr-c 0.1 \
    --skip-labeling
```
**Expected**: Smaller train-test gap, potentially better generalization

### To Improve Results Further

1. **Collect More Data**: Target 5000+ functions
2. **Data Augmentation**: Variable renaming, reformatting
3. **Contrastive Pre-training**: Fine-tune embeddings on side effect labels
4. **Cross-Validation**: Use 5-fold CV for more robust estimates
5. **Ensemble Methods**: Combine predictions from multiple models

---

## 10. Embedded/Bare-Metal Hardware Feature Expansion

### Motivation

The original `CodeFeatureExtractor` provided 20 hand-crafted regex features covering general-purpose API detection (malloc, printf, ioctl, etc.) and code metrics (line count, pointer ops, etc.). However, the dataset is ~60% Linux kernel driver code (GPIO, watchdog, SPI, I2C, PWM drivers), and the original features lacked specificity for embedded and bare-metal patterns. This experiment expanded the feature extractor with 20 additional hardware-targeted features to test whether domain-specific regex features improve classification on hardware-heavy code.

### New Features Added (20 features)

The `CodeFeatureExtractor` was expanded from 20 to 40 features. All new features are pure regex — no LLM or AST involvement. The expanded feature set is activated via the `--hardware-features` CLI flag.

| Group | Features | What they detect |
|-------|----------|-----------------|
| MMIO register access | `mmio_count_norm`, `mmio_present` | `readl`/`writel`/`ioread32`/`iowrite32`/`__iomem` |
| Regmap abstraction | `regmap_count_norm`, `regmap_present` | `regmap_read`/`regmap_write`/`regmap_update_bits` |
| Interrupt handling | `irq_count_norm`, `irq_present` | `request_irq`/`enable_irq`/`disable_irq` |
| Synchronization | `sync_count_norm`, `sync_present` | `spin_lock`/`mutex_lock`/`atomic_read`/`barrier`/`mb` |
| Device managed resources | `devm_count_norm`, `devm_present` | `devm_kzalloc`/`devm_ioremap`/`devm_clk_get` |
| SPI protocol | `spi_present` | `spi_sync`/`spi_transfer`/`spi_message` |
| I2C protocol | `i2c_present` | `i2c_transfer`/`i2c_smbus_read_byte` |
| GPIO subsystem | `gpio_present` | `gpiochip_get_data`/`gpio_set_value`/`gpiod_*` |
| Watchdog/timer | `wdt_timer_present` | `watchdog_register_device`/`mod_timer`/`hrtimer_*` |
| Bitwise density | `bitwise_density` | `<<`/`>>`/`& 0x`/`BIT()`/`GENMASK()` count |
| Driver lifecycle | `is_driver_lifecycle` | `probe`/`remove`/`suspend`/`resume` in function signature |
| Device structs | `has_device_struct` | `platform_device`/`i2c_client`/`spi_device` |
| Module macros | `has_module_macro` | `module_init`/`MODULE_LICENSE`/`module_platform_driver` |
| Inline assembly | `has_inline_asm` | `__asm__`/`asm volatile` |
| Compiler attrs | `has_compiler_attr` | `__attribute__`/`__packed`/`likely`/`unlikely` |

### CLI Usage

```bash
# 20 base regex features only (original behavior):
python run_pipeline.py --skip-labeling --embedder codebert \
    --classifier logistic_regression --hybrid-features

# All 40 features (20 base + 20 embedded/bare-metal HW):
python run_pipeline.py --skip-labeling --embedder codebert \
    --classifier logistic_regression --hardware-features

# Qwen3 with 40 HW features:
python run_pipeline.py --skip-labeling --embedder qwen3 \
    --classifier logistic_regression --hardware-features
```

`--hardware-features` automatically implies `--hybrid-features`, so you don't need both flags.

### Results

| # | Config | Embed Dim | Regex Features | Total Classifier Dims | Test F1 | Test Acc | Complexity F1 | Error F1 | Train F1 | Embed Time |
|---|--------|-----------|---------------|----------------------|---------|----------|---------------|----------|----------|------------|
| 14 | CodeBERT + `--hybrid-features` (baseline) | 768 | 20 | 788 | 0.587 | 0.436 | 0.630 | 0.699 | — | ~5s |
| 20 | **CodeBERT + `--hardware-features`** | 768 | 40 | 808 | 0.565 | 0.423 | **0.699** | **0.756** | 0.865 | ~38s |
| 14 | Qwen3 + `--hybrid-features` (baseline) | 1024 | 20 | 1044 | 0.622 | 0.560 | 0.608 | 0.678 | — | ~18s |
| 21 | **Qwen3 + `--hardware-features`** | 1024 | 40 | 1064 | 0.613 | 0.547 | 0.604 | 0.672 | 1.000 | ~94s |

### Analysis

#### CodeBERT: Hardware features significantly improve complexity and error handling

The 20 new hardware features produced clear gains on complexity (+6.9pts) and error handling (+5.7pts) for CodeBERT. Features like `is_driver_lifecycle`, `devm_present`, `bitwise_density`, and `sync_present` give the classifier direct signals to distinguish simple utility functions from full driver probe/init functions — which is exactly what complexity and error handling labels correlate with.

Side effects F1 dipped slightly (-2.2pts). The original 20 features already captured the primary side-effect-relevant APIs (malloc, printf, ioctl). The 20 new features are mostly orthogonal to side effect labels, so they add dimensionality without proportional signal for that specific task.

#### Qwen3: Hardware features provide diminishing returns

Qwen3's 1024-dim embeddings already capture more semantic information than CodeBERT's 768-dim, so the explicit regex features add proportionally less new signal. Side effects held roughly steady (-0.9pts), while complexity (-0.4pts) and error handling (-0.6pts) showed marginal drops. The train F1 of 1.000 across the board confirms the classifier memorizes perfectly — the bottleneck is generalization, not feature expressiveness.

#### Key takeaway: Feature injection helps weaker embeddings more

The hardware features act as a crutch for CodeBERT's less expressive embeddings. Qwen3 already knows what `regmap_update_bits` means from its pretraining; CodeBERT doesn't, so telling it explicitly via a binary feature helps. This pattern — explicit features helping weaker models more — is consistent with the broader finding in Section 6 that feature engineering has diminishing returns as embedding quality improves.

### Overfitting note

Both configurations overfit severely (CodeBERT train 0.865, Qwen3 train 1.000 vs test ~0.6). The 40 regex features on 780 samples still leaves the fundamental data limitation unresolved. The new features didn't worsen overfitting for CodeBERT (0.865 train is actually lower than the 1.000 seen with Qwen3), suggesting CodeBERT + LR + 40 features is in a healthier regime than Qwen3's perfect memorization.

---

## Appendix: Full Experimental Results

### Embedding Model Experiments

| # | Embedder | Classifier | Test F1 | Test Acc | Notes |
|---|----------|------------|---------|----------|-------|
| 1 | codebert | LR | 0.587 | 0.436 | Baseline |
| 2 | qwen3 | LR | 0.617 | 0.568 | +5.1% |
| 3 | qwen3-4b | LR | 0.630 | 0.526 | **Best** |
| 4 | nomic | LR | 0.622 | 0.517 | Good |

### Classifier Experiments (Qwen3)

| # | Classifier | Test F1 | Test Acc | Train F1 | Gap |
|---|------------|---------|----------|----------|-----|
| 5 | LR | 0.617 | 0.568 | 1.000 | 0.383 |
| 6 | SVM | 0.606 | 0.479 | 0.860 | 0.254 |
| 7 | RF | 0.379 | 0.385 | 1.000 | 0.621 |
| 8 | MLP | 0.596 | 0.551 | 1.000 | 0.404 |

### Multi-Label Strategy Experiments (Qwen3 + LR)

| # | Strategy | Test F1 | Test Acc | Notes |
|---|----------|---------|----------|-------|
| 9 | Standard | 0.617 | 0.568 | Baseline |
| 10 | Binary | 0.617 | 0.568 | Same |
| 11 | Binary+SMOTE | 0.616 | 0.556 | Slight degradation |
| 12 | Binary+Threshold | 0.617 | 0.568 | Same |

### Feature Engineering Experiments (Qwen3 + LR)

| # | Features | Dims | Test F1 | Test Acc | Delta |
|---|----------|------|---------|----------|-------|
| 13 | None | 1024 | 0.617 | 0.568 | - |
| 14 | `--hybrid-features` (20) | 1044 | 0.622 | 0.560 | +0.005 |
| 15 | +AST | 1039 | 0.619 | 0.564 | +0.002 |
| 16 | +Purpose | 2048 | 0.621 | 0.560 | +0.004 |
| 17 | +Hybrid+AST | 1059 | 0.617 | 0.551 | +0.000 |

### Best Combination Experiments

| # | Configuration | Test F1 | Error F1 | Notes |
|---|--------------|---------|----------|-------|
| 18 | Qwen3-4B+LR+All | 0.626 | 0.749 | Good |
| 19 | Qwen3-4B+MLP+All | **0.629** | 0.749 | **Best** |

### Embedded/Bare-Metal HW Feature Experiments (LR only)

| # | Config | Flag | Embed Dim | Regex Feats | Total Dims | Test F1 | Test Acc | Complexity F1 | Error F1 | Train F1 |
|---|--------|------|-----------|-------------|-----------|---------|----------|---------------|----------|----------|
| 20 | CodeBERT + HW | `--hardware-features` | 768 | 40 | 808 | 0.565 | 0.423 | **0.699** | **0.756** | 0.865 |
| 21 | Qwen3 + HW | `--hardware-features` | 1024 | 40 | 1064 | 0.613 | 0.547 | 0.604 | 0.672 | 1.000 |

---

*Report generated for C Function Understanding Pipeline*
*Experiments conducted: March–April 2025*
*Total experiments: 21*
