# Comprehensive Report & Experimental Configurations

## Overview

This repository implements a **C Function Understanding Pipeline** using embedding-based representation learning. The pipeline extracts functions from C source code, generates semantic labels, creates embeddings, and trains classifiers to predict function properties.

---

## Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. EXTRACTION  │───▶│   2. LABELING   │───▶│  3. EMBEDDING   │───▶│ 4. EVALUATION   │
│   (tree-sitter) │    │  (LLM/offline)  │    │  + CLASSIFICATION│    │   (metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step 1: Function Extraction (`src/extract.py`)
- **Method**: tree-sitter AST parsing (fallback: regex)
- **Output**: Function name, code, file path, line numbers

### Step 2: Semantic Labeling (`src/label.py`)
- **LLM Mode**: Claude API with `temperature=0` for deterministic output
- **Offline Mode**: Regex-based heuristics (no API needed)
- **Labels Generated**:
  | Label | Type | Values |
  |-------|------|--------|
  | `side_effects` | Multi-label | `io`, `memory`, `hardware`, `network`, `global_state`, `none` |
  | `complexity` | Single-label | `low`, `medium`, `high` |
  | `error_handling` | Single-label | `returns_code`, `uses_errno`, `assertions`, `none` |
  | `control_flow_elements` | Deterministic | `if`, `for`, `while`, `switch`, `goto`, `return` |
  | `high_level_purpose` | Free text | Natural language description |

### Step 3: Embedding & Classification (`src/embed.py`)

#### Available Embedding Models
| Model | Dimension | Notes |
|-------|-----------|-------|
| `codebert` | 768 | Default. Pre-trained on code, fast, CPU-friendly |
| `qwen3` | 1024 | Qwen3-Embedding-0.6B. Better quality |
| `qwen3-4b` | 2560 | Qwen3-Embedding-4B. Higher quality, slower |
| `qwen3-8b` | 4096 | Qwen3-Embedding-8B. Best quality, slowest |
| `nomic` | 768 | nomic-embed-text-v1.5 |
| `jina` | 1024 | jina-embeddings-v3 |
| `codesage` | 1024 | CodeSage-large-v2 (requires transformers==4.35.0, torch==2.1.0) |

#### Available Classifiers
| Classifier | Key Hyperparameters |
|------------|---------------------|
| `logistic_regression` | `--lr-c`, `--lr-max-iter`, `--lr-solver` |
| `random_forest` | `--rf-n-estimators`, `--rf-max-depth`, `--rf-min-samples-split` |
| `svm` | `--svm-c`, `--svm-kernel`, `--svm-gamma` |
| `mlp` | `--mlp-hidden-layers`, `--mlp-activation`, `--mlp-learning-rate` |

#### Feature Enhancement Options
| Option | Description | Features Added |
|--------|-------------|----------------|
| `--hybrid-features` | Regex-based API detection patterns | 20 features (memory/IO/hardware API counts, code metrics) |
| `--ast-features` | Tree-sitter AST analysis | 15 features (statement distribution, loop/branch depth, cyclomatic complexity) |
| `--purpose-embeddings` | Embed `high_level_purpose` text | +768/1024 dims (depends on model) |
| `--threshold-tuning` | Optimize per-class prediction thresholds | Improves F1 for imbalanced classes |

#### Class Imbalance Handling
| Option | Description |
|--------|-------------|
| `--smote` | SMOTE oversampling for minority classes |
| `--binary-classifiers` | Train separate binary classifier per side effect class |
| `--no-class-weight` | Disable balanced class weighting |

#### Dimensionality Reduction
| Option | Description |
|--------|-------------|
| `--embed-dim N` | Apply PCA to reduce embeddings to N dimensions |

### Step 4: Evaluation (`src/evaluate.py`)
- **Classification**: F1 (macro/micro), accuracy, per-class precision/recall
- **Similarity Search**: Side-effect overlap@K
- **Clustering**: Silhouette score, label purity
- **Failure Analysis**: Identifies misclassified functions and similar functions with different behaviors

---

## Experimental Configurations

Below are systematically designed experiments to compare different pipeline configurations. Run these commands and record the **Test F1 (macro)** and **Test Accuracy** metrics.

### Baseline Experiments (Embedding Models Only)

```bash
# Experiment 1: CodeBERT baseline
python run_pipeline.py --skip-labeling --embedder codebert --classifier logistic_regression

# Experiment 2: Qwen3-0.6B baseline
python run_pipeline.py --skip-labeling --embedder qwen3 --classifier logistic_regression

# Experiment 3: Qwen3-4B baseline (if GPU available)
python run_pipeline.py --skip-labeling --embedder qwen3-4b --classifier logistic_regression

# Experiment 4: Nomic baseline
python run_pipeline.py --skip-labeling --embedder nomic --classifier logistic_regression

# Experiment 5: Jina baseline
python run_pipeline.py --skip-labeling --embedder jina --classifier logistic_regression
```

### Classifier Comparison (with CodeBERT)

```bash
# Experiment 6: Logistic Regression
python run_pipeline.py --skip-labeling --classifier logistic_regression

# Experiment 7: Random Forest (default)
python run_pipeline.py --skip-labeling --classifier random_forest

# Experiment 8: SVM with RBF kernel
python run_pipeline.py --skip-labeling --classifier svm

# Experiment 9: SVM with linear kernel
python run_pipeline.py --skip-labeling --classifier svm --svm-kernel linear

# Experiment 10: MLP Neural Network
python run_pipeline.py --skip-labeling --classifier mlp
```

### Multi-label Strategy Comparison

```bash
# Experiment 11: Standard multi-label (OneVsRest)
python run_pipeline.py --skip-labeling --classifier logistic_regression

# Experiment 12: Binary classifiers per class
python run_pipeline.py --skip-labeling --classifier logistic_regression --binary-classifiers

# Experiment 13: Binary classifiers + SMOTE
python run_pipeline.py --skip-labeling --classifier logistic_regression --binary-classifiers --smote

# Experiment 14: Binary classifiers + threshold tuning
python run_pipeline.py --skip-labeling --classifier logistic_regression --binary-classifiers --threshold-tuning
```

### Feature Engineering Experiments

```bash
# Experiment 15: Hybrid features only
python run_pipeline.py --skip-labeling --classifier logistic_regression --hybrid-features

# Experiment 16: AST features only
python run_pipeline.py --skip-labeling --classifier logistic_regression --ast-features

# Experiment 17: Purpose embeddings only
python run_pipeline.py --skip-labeling --classifier logistic_regression --purpose-embeddings

# Experiment 18: Hybrid + AST features
python run_pipeline.py --skip-labeling --classifier logistic_regression --hybrid-features --ast-features

# Experiment 19: All features combined
python run_pipeline.py --skip-labeling --classifier logistic_regression --hybrid-features --ast-features --purpose-embeddings

# Experiment 20: All features + threshold tuning
python run_pipeline.py --skip-labeling --classifier logistic_regression --hybrid-features --ast-features --purpose-embeddings --threshold-tuning
```

### Dimensionality Reduction Experiments

```bash
# Experiment 21: PCA to 256 dimensions
python run_pipeline.py --skip-labeling --classifier logistic_regression --embed-dim 256

# Experiment 22: PCA to 128 dimensions
python run_pipeline.py --skip-labeling --classifier logistic_regression --embed-dim 128

# Experiment 23: PCA to 64 dimensions
python run_pipeline.py --skip-labeling --classifier logistic_regression --embed-dim 64

# Experiment 24: PCA + hybrid features
python run_pipeline.py --skip-labeling --classifier logistic_regression --embed-dim 128 --hybrid-features
```

### Class Imbalance Handling

```bash
# Experiment 25: SMOTE oversampling
python run_pipeline.py --skip-labeling --classifier logistic_regression --smote

# Experiment 26: No class weighting
python run_pipeline.py --skip-labeling --classifier logistic_regression --no-class-weight

# Experiment 27: SMOTE + Binary classifiers
python run_pipeline.py --skip-labeling --classifier random_forest --smote --binary-classifiers
```

### Hyperparameter Tuning Experiments

```bash
# Experiment 28: Logistic Regression with higher regularization
python run_pipeline.py --skip-labeling --classifier logistic_regression --lr-c 0.1

# Experiment 29: Logistic Regression with lower regularization
python run_pipeline.py --skip-labeling --classifier logistic_regression --lr-c 10.0

# Experiment 30: Random Forest with more trees
python run_pipeline.py --skip-labeling --classifier random_forest --rf-n-estimators 200

# Experiment 31: Random Forest with limited depth (prevent overfitting)
python run_pipeline.py --skip-labeling --classifier random_forest --rf-max-depth 10

# Experiment 32: MLP with larger hidden layers
python run_pipeline.py --skip-labeling --classifier mlp --mlp-hidden-layers "512,256,128"

# Experiment 33: MLP with smaller learning rate
python run_pipeline.py --skip-labeling --classifier mlp --mlp-learning-rate 0.0001
```

### Best Combination Experiments

```bash
# Experiment 34: Best embedding + best classifier + all features
python run_pipeline.py --skip-labeling --embedder qwen3 --classifier random_forest --hybrid-features --ast-features --threshold-tuning

# Experiment 35: CodeBERT + RF + all features + PCA
python run_pipeline.py --skip-labeling --embedder codebert --classifier random_forest --hybrid-features --ast-features --embed-dim 256 --threshold-tuning

# Experiment 36: Qwen3 + MLP + all features
python run_pipeline.py --skip-labeling --embedder qwen3 --classifier mlp --hybrid-features --ast-features --purpose-embeddings --threshold-tuning

# Experiment 37: Full pipeline with tuning
python run_pipeline.py --skip-labeling --embedder qwen3 --classifier random_forest --hybrid-features --ast-features --purpose-embeddings --threshold-tuning --binary-classifiers
```

---

## Results Recording Template

| Exp# | Embedder | Classifier | Features | Options | Train F1 | Test F1 | Test Acc | Notes |
|------|----------|------------|----------|---------|----------|---------|----------|-------|
| 1 | codebert | lr | - | - | | | | baseline |
| 2 | qwen3 | lr | - | - | | | | |
| 3 | qwen3-4b | lr | - | - | | | | |
| ... | ... | ... | ... | ... | | | | |

### Key Metrics to Compare:
1. **Test F1 (macro)** - Main metric for imbalanced multi-label classification
2. **Test Accuracy** - Overall accuracy
3. **Train F1 vs Test F1 gap** - Indicates overfitting (large gap = overfitting)
4. **Per-class F1** - Check `evaluation_report.json` for detailed breakdown

---

## Feature Details

### Hybrid Features (20 features from `CodeFeatureExtractor`)
| Feature | Description |
|---------|-------------|
| `memory_count_norm` | Normalized count of memory API calls |
| `memory_present` | Binary: any memory API present |
| `io_count_norm` | Normalized count of I/O API calls |
| `io_present` | Binary: any I/O API present |
| `hardware_count_norm` | Normalized count of hardware API calls |
| `hardware_present` | Binary: any hardware API present |
| `global_count_norm` | Normalized count of global state patterns |
| `global_present` | Binary: any global state patterns |
| `error_count_norm` | Normalized count of error handling patterns |
| `error_present` | Binary: any error handling patterns |
| `lines_norm` | Normalized line count |
| `func_calls_norm` | Normalized function call count |
| `control_flow_norm` | Normalized control flow statement count |
| `pointer_ops_norm` | Normalized pointer operation count |
| `param_count_norm` | Normalized parameter count |
| `return_count_norm` | Normalized return statement count |
| `has_null` | Binary: contains NULL |
| `has_sizeof` | Binary: contains sizeof |
| `has_struct_access` | Binary: contains struct access (`->` or `.`) |
| `has_array_access` | Binary: contains array access (`[]`) |

### AST Features (15 features from `ASTFeatureExtractor`)
| Feature | Description |
|---------|-------------|
| `stmt_assignment_expression_ratio` | Proportion of assignment statements |
| `stmt_declaration_ratio` | Proportion of declaration statements |
| `stmt_call_expression_ratio` | Proportion of function calls |
| `stmt_return_statement_ratio` | Proportion of return statements |
| `stmt_if_statement_ratio` | Proportion of if statements |
| `stmt_for_statement_ratio` | Proportion of for loops |
| `stmt_while_statement_ratio` | Proportion of while loops |
| `stmt_do_statement_ratio` | Proportion of do-while loops |
| `stmt_switch_statement_ratio` | Proportion of switch statements |
| `stmt_goto_statement_ratio` | Proportion of goto statements |
| `stmt_break_statement_ratio` | Proportion of break statements |
| `stmt_continue_statement_ratio` | Proportion of continue statements |
| `max_loop_depth` | Maximum nesting depth of loops |
| `max_branch_depth` | Maximum nesting depth of branches |
| `cyclomatic_complexity` | McCabe's cyclomatic complexity |

---

## Expected Insights

Based on the architecture, here are hypotheses to test:

1. **Embedding Models**: Qwen3 models should outperform CodeBERT due to larger training data and better architecture, but at the cost of speed.

2. **Classifiers**: Random Forest typically handles multi-label well; MLP may capture non-linear patterns but risks overfitting on small datasets.

3. **Hybrid Features**: Should significantly boost performance for `memory` and `io` classes since these have explicit API patterns (e.g., `malloc`, `printf`).

4. **AST Features**: Should help distinguish functions with similar code patterns but different control flow (e.g., simple getter vs loop-based search).

5. **Purpose Embeddings**: May help when function names/descriptions are informative, but could add noise if purposes are generic.

6. **Threshold Tuning**: Should improve macro F1 by finding better decision boundaries for minority classes.

7. **Binary Classifiers**: Often better than OneVsRest for highly imbalanced multi-label data.

8. **PCA**: May help prevent overfitting when embedding dimensions >> training samples.

---

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline experiment
python run_pipeline.py --skip-labeling --classifier logistic_regression

# Run with all enhancements
python run_pipeline.py --skip-labeling --embedder qwen3 --classifier random_forest \
    --hybrid-features --ast-features --threshold-tuning --binary-classifiers

# View detailed evaluation
cat data/processed/evaluation_report.json | python -m json.tool
```
