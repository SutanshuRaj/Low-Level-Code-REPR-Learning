# Technical Design Report

## C Function Understanding Pipeline Using Embedding-Based Representation Learning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Algorithm Design](#4-algorithm-design)
5. [Feature Engineering](#5-feature-engineering)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Optimal Configuration](#7-optimal-configuration)
8. [Limitations and Future Work](#8-limitations-and-future-work)

---

## 1. Executive Summary

This project implements an **embedding-based function understanding system** for C source code. The system extracts functions from C files, generates semantic labels, creates vector representations using pre-trained language models, and trains classifiers to predict function properties.

### Key Achievements

| Metric | Best Result | Configuration |
|--------|-------------|---------------|
| **Side Effects F1** | 0.629 | Qwen3-4B + MLP + All Features |
| **Complexity F1** | 0.704 | Nomic + Logistic Regression |
| **Error Handling F1** | 0.749 | Qwen3-4B + LR + Purpose Embeddings |
| **Similarity Overlap@5** | 0.755 | Qwen3-4B embeddings |

### Dataset Statistics
- **Total Functions**: 780
- **Side Effect Distribution**: hardware (315), global_state (294), memory (281), none (110), io (105), network (9), assertions (1)
- **Class Imbalance Ratio**: 315:1 (hardware vs assertions)

---

## 2. Problem Statement

### 2.1 Objective

Given C source code, automatically classify functions by their **semantic properties**:

1. **Side Effects** (multi-label): What external state does the function modify?
   - `io`: File/console operations (printf, fopen, fread)
   - `memory`: Heap allocation (malloc, free, memcpy)
   - `hardware`: Register/port access, interrupts, DMA
   - `network`: Socket operations (socket, send, recv)
   - `global_state`: Static/global variable modifications
   - `none`: Pure computation

2. **Complexity** (single-label): How complex is the control flow?
   - `low`, `medium`, `high`

3. **Error Handling** (single-label): How are errors handled?
   - `returns_code`, `uses_errno`, `assertions`, `none`

### 2.2 Challenges

1. **Multi-label Classification**: Functions can have multiple side effects simultaneously (e.g., `memory` + `io`)
2. **Severe Class Imbalance**: `hardware` has 315 samples, `assertions` has only 1
3. **Semantic Gap**: Code structure doesn't always reflect runtime behavior
4. **Limited Training Data**: Only 780 functions available

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     C Function Understanding Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ 1. EXTRACT   │───▶│  2. LABEL    │───▶│  3. EMBED    │───▶│4. TRAIN  │  │
│  │  tree-sitter │    │  LLM/Offline │    │  + FEATURES  │    │ Classify │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│         │                   │                   │                  │        │
│         ▼                   ▼                   ▼                  ▼        │
│    CFunction           Labels JSON        Embeddings +        Classifier   │
│    Objects             (structured)       Feature Matrix       Models      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### Stage 1: Function Extraction (`src/extract.py`)

**Algorithm**: Tree-sitter AST parsing with regex fallback

```python
class CFunctionExtractor:
    def extract_from_file(file_path):
        if tree_sitter_available:
            return _extract_with_tree_sitter(code)  # Preferred
        else:
            return _extract_with_regex(code)        # Fallback
```

**Why Tree-sitter?**
- Handles complex C syntax (nested functions, macros)
- Accurate line number tracking
- Robust to malformed code

**Output per function**:
- `function_name`: Identifier
- `function_code`: Complete source
- `file_path`: Source location
- `start_line`, `end_line`: Position markers

#### Stage 2: Semantic Labeling (`src/label.py`)

**Two modes available**:

| Mode | Method | Pros | Cons |
|------|--------|------|------|
| **LLM** | Claude API (temperature=0) | High quality, semantic understanding | API cost, latency |
| **Offline** | Regex pattern matching | Fast, free, reproducible | May miss edge cases |

**Label Generation Strategy**:
- `control_flow_elements`: **Deterministic** (regex-extracted, 100% accurate)
- `side_effects`: **LLM/Heuristic** (requires semantic understanding)
- `complexity`: **Rule-based** (lines + control flow + nesting depth)
- `error_handling`: **Pattern matching** (return -1, errno, assert)
- `high_level_purpose`: **LLM** (free-form text description)

#### Stage 3: Embedding Generation (`src/embed.py`)

**Available Embedders**:

| Model | Architecture | Dimensions | Speed | Quality |
|-------|--------------|------------|-------|---------|
| CodeBERT | BERT-base | 768 | Fast | Baseline |
| Qwen3-0.6B | Transformer | 1024 | Medium | Good |
| Qwen3-4B | Transformer | 2560 | Slow | Better |
| Qwen3-8B | Transformer | 4096 | Slowest | Best |
| Nomic | BERT-variant | 768 | Medium | Good |
| Jina | Custom | 1024 | Medium | Good |

**Embedding Process**:
```python
def embed(code: str) -> np.ndarray:
    tokens = tokenizer(code, truncation=True, max_length=512)
    outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :]  # CLS token pooling
```

#### Stage 4: Classification (`src/embed.py`)

**Multi-label Architecture**:
```
                          ┌──────────────────┐
                          │   Side Effects   │
                          │ (OneVsRest/Binary)│
                          └────────┬─────────┘
                                   │
┌──────────────┐    ┌─────────────┴─────────────┐
│  Embedding   │───▶│     Feature Fusion        │───▶ Predictions
│  (768-4096d) │    │ [embed | hybrid | AST |   │
└──────────────┘    │        purpose]           │
                    └───────────────────────────┘
```

---

## 4. Algorithm Design

### 4.1 Multi-Label Classification

**Problem**: Functions can have multiple side effects (e.g., `[memory, io]`)

**Solution 1: OneVsRest (Default)**
```python
from sklearn.multioutput import MultiOutputClassifier
clf = MultiOutputClassifier(LogisticRegression())
clf.fit(X, Y_multilabel)  # Y shape: (n_samples, n_classes)
```

**Solution 2: Binary Classifiers per Class**
```python
classifiers = {}
for class_name in classes:
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, Y[:, class_idx])  # Binary target
    classifiers[class_name] = clf
```

**Trade-offs**:
| Approach | Pros | Cons |
|----------|------|------|
| OneVsRest | Simple, captures correlations | All-or-nothing prediction |
| Binary | Class-specific tuning | Ignores label dependencies |

### 4.2 Class Imbalance Handling

**Strategies Implemented**:

1. **Class Weighting** (default):
   ```python
   clf = LogisticRegression(class_weight='balanced')
   # Weight_i = n_samples / (n_classes * n_samples_i)
   ```

2. **SMOTE Oversampling**:
   ```python
   from imblearn.over_sampling import SMOTE
   X_resampled, y_resampled = SMOTE().fit_resample(X, y)
   ```

3. **Threshold Tuning**:
   ```python
   for threshold in [0.1, 0.2, ..., 0.9]:
       y_pred = (y_proba >= threshold).astype(int)
       f1 = f1_score(y_true, y_pred)
       if f1 > best_f1:
           best_threshold = threshold
   ```

### 4.3 Dimensionality Reduction

**PCA for High-Dimensional Embeddings**:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
X_reduced = pca.fit_transform(embeddings)  # 2560 → 256 dims
```

**When to use**:
- Embedding dims >> training samples (curse of dimensionality)
- Combat overfitting (observed: Train F1=1.0, Test F1=0.6)

---

## 5. Feature Engineering

### 5.1 Hybrid Features (Regex-Based)

**Rationale**: Embeddings capture syntax, not explicit API calls. Regex features directly detect side effect indicators.

**20 Features Extracted**:

| Category | Features | Description |
|----------|----------|-------------|
| **Memory APIs** | count + binary | malloc, free, memcpy, etc. |
| **I/O APIs** | count + binary | printf, fopen, fread, etc. |
| **Hardware APIs** | count + binary | ioctl, mmap, volatile, etc. |
| **Global State** | count + binary | static, extern, g_prefix |
| **Error Handling** | count + binary | errno, assert, return -1 |
| **Code Metrics** | 6 features | lines, calls, control flow, pointers, params, returns |
| **Keywords** | 4 binary | NULL, sizeof, struct access, array access |

**Implementation**:
```python
class CodeFeatureExtractor:
    MEMORY_APIS = [r'\bmalloc\s*\(', r'\bfree\s*\(', ...]

    def extract_features(code: str) -> np.ndarray:
        features = []
        memory_count = sum(1 for p in self.memory_patterns if p.search(code))
        features.append(min(memory_count / 3.0, 1.0))  # Normalized
        features.append(1.0 if memory_count > 0 else 0.0)  # Binary
        # ... 20 total features
        return np.array(features)
```

### 5.2 AST Features (Tree-Sitter Based)

**Rationale**: Structural code analysis provides control-flow insights that embeddings may miss.

**15 Features Extracted**:

| Category | Features | Description |
|----------|----------|-------------|
| **Statement Distribution** | 12 ratios | Proportion of each statement type |
| **Control Flow Depth** | 2 metrics | Max loop depth, max branch depth |
| **Complexity** | 1 metric | McCabe's cyclomatic complexity |

**Statement Types Tracked**:
- `assignment_expression`, `declaration`, `call_expression`
- `return_statement`, `if_statement`, `for_statement`
- `while_statement`, `do_statement`, `switch_statement`
- `goto_statement`, `break_statement`, `continue_statement`

**Cyclomatic Complexity Calculation**:
```python
CC = 1 + decision_points
# decision_points = if + for + while + case + && + || + ?:
```

### 5.3 Purpose Embeddings

**Rationale**: The `high_level_purpose` text (e.g., "Allocates and initializes a buffer") contains semantic information about function intent.

**Process**:
```python
# During training:
purpose_text = func.labels.get("high_level_purpose", "")
purpose_embedding = embedder.embed(purpose_text)  # Same model as code

# Feature fusion:
X = np.hstack([code_embedding, hybrid_features, ast_features, purpose_embedding])
```

**Dimension Impact**:
| Base Model | + Hybrid | + AST | + Purpose | Total |
|------------|----------|-------|-----------|-------|
| Qwen3-4B (2560) | +20 | +15 | +2560 | 5155 |

---

## 6. Evaluation Framework

### 6.1 Metrics Definitions

#### Classification Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **F1 Macro** | $\frac{1}{n}\sum_{i=1}^{n} F1_i$ | Balanced across all classes (primary metric) |
| **F1 Micro** | $\frac{2 \cdot P \cdot R}{P + R}$ (global) | Weighted by class frequency |
| **Accuracy** | $\frac{TP + TN}{Total}$ | Overall correctness (misleading for imbalanced data) |
| **Precision** | $\frac{TP}{TP + FP}$ | "Of predicted positives, how many are correct?" |
| **Recall** | $\frac{TP}{TP + FN}$ | "Of actual positives, how many did we find?" |

**Why F1 Macro as Primary Metric?**
- Treats all classes equally, regardless of frequency
- Avoids bias toward majority classes (hardware: 315 vs network: 9)
- Standard for multi-label classification

#### Similarity Search Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Side Effect Overlap@K** | $\frac{1}{K}\sum_{i=1}^{K} \frac{|Q \cap R_i|}{|Q \cup R_i|}$ | Do similar embeddings have similar side effects? |

**Process**:
1. For query function Q, find K nearest neighbors by cosine similarity
2. Calculate Jaccard overlap between Q's side effects and each neighbor's
3. Average across all queries

#### Clustering Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Silhouette Score** | $\frac{b - a}{\max(a, b)}$ | Cluster cohesion (-1 to 1, higher = better) |
| **Label Purity** | $\frac{\text{most common label count}}{\text{cluster size}}$ | Do clusters align with labels? |

### 6.2 Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 70% train (546 samples), 30% test (234 samples)
```

### 6.3 Overfitting Detection

**Indicator**: Large gap between Train F1 and Test F1

| Gap | Interpretation | Action |
|-----|----------------|--------|
| < 0.1 | Healthy | None needed |
| 0.1-0.3 | Moderate overfitting | Add regularization, reduce model complexity |
| > 0.3 | Severe overfitting | PCA, simpler model, more data |

**Observed in Experiments**:
- Train F1: 1.000 (perfect)
- Test F1: 0.617
- Gap: **0.383 (severe overfitting)**

---

## 7. Optimal Configuration

### 7.1 Best Overall Configuration

Based on experimental results, the optimal configuration is:

```bash
python run_pipeline.py \
    --skip-labeling \
    --embedder qwen3-4b \
    --side-effects-clf mlp \
    --complexity-clf random_forest \
    --error-handling-clf logistic_regression \
    --hybrid-features \
    --ast-features \
    --purpose-embeddings \
    --threshold-tuning \
    --smote
```

**Results**:
| Task | F1 Score |
|------|----------|
| Side Effects | **0.629** |
| Complexity | 0.591 |
| Error Handling | **0.749** |

### 7.2 Per-Task Recommendations

| Task | Best Classifier | Features | Reason |
|------|-----------------|----------|--------|
| **side_effects** | MLP | All | Non-linear patterns, multi-label |
| **complexity** | Random Forest | AST | Structural features matter |
| **error_handling** | Logistic Regression | Purpose | Text descriptions help |

### 7.3 Configuration Guidelines

| Scenario | Recommendation |
|----------|----------------|
| **Fast inference needed** | CodeBERT + LR (5s embedding) |
| **Best quality needed** | Qwen3-4B + MLP + all features (65s embedding) |
| **Limited GPU memory** | Qwen3-0.6B (1024 dims) |
| **Severe imbalance** | Binary classifiers + threshold tuning |
| **Overfitting concerns** | PCA (--embed-dim 256) + higher regularization (--lr-c 0.1) |

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

1. **Severe Overfitting**: Train F1=1.0, Test F1~0.62 indicates memorization
2. **Class Imbalance**: `assertions` (1 sample), `network` (9 samples) cannot be learned
3. **Semantic Gap**: Embeddings capture syntax, not runtime behavior
4. **Small Dataset**: 780 functions insufficient for deep learning

### 8.2 Recommended Improvements

1. **Data Augmentation**:
   - Variable renaming
   - Comment addition/removal
   - Code reformatting
   - Synthetic function generation

2. **Contrastive Learning**:
   ```python
   # Pull together functions with same side effects
   # Push apart functions with different side effects
   loss = contrastive_loss(anchor, positive, negative)
   ```

3. **Graph Neural Networks**:
   - Use control flow graphs (CFG) as input
   - Better capture of program structure

4. **Transfer Learning**:
   - Pre-train on large C corpus (Linux kernel, FreeBSD)
   - Fine-tune on labeled functions

5. **Active Learning**:
   - Identify uncertain predictions
   - Request human labels for hard cases

### 8.3 Scalability Considerations

For millions of functions:
- **Streaming**: Use HuggingFace datasets with `streaming=True`
- **Distributed Training**: FSDP/DeepSpeed
- **Efficient Indexing**: FAISS for similarity search
- **Incremental Learning**: Update models without full retraining

---

## Appendix A: File Structure

```
low-level-code-repr-learning/
├── data/
│   ├── raw/                        # Input C source files
│   └── processed/
│       ├── labeled_functions.json  # LLM/offline labels
│       ├── embedded_functions.json # Embeddings + labels
│       ├── embeddings.npy          # Raw embedding matrix
│       ├── classifier.joblib       # Trained sklearn models
│       └── evaluation_report.json  # Metrics and analysis
├── src/
│   ├── extract.py                  # Tree-sitter extraction
│   ├── label.py                    # LLM/offline labeling
│   ├── embed.py                    # Embedding + classification
│   ├── evaluate.py                 # Metrics computation
│   └── infer.py                    # CLI inference
├── run_pipeline.py                 # Main orchestration
├── clean_labels.py                 # Data preprocessing
└── requirements.txt                # Dependencies
```

## Appendix B: CLI Reference

```bash
# Embedding models
--embedder {codebert,qwen3,qwen3-4b,qwen3-8b,nomic,jina}

# Classifiers
--classifier {logistic_regression,random_forest,svm,mlp}
--side-effects-clf, --complexity-clf, --error-handling-clf

# Feature engineering
--hybrid-features      # +20 regex features
--ast-features         # +15 AST features
--purpose-embeddings   # +N purpose embedding dims

# Imbalance handling
--smote               # SMOTE oversampling
--binary-classifiers  # Per-class binary models
--threshold-tuning    # Optimize prediction thresholds
--no-class-weight     # Disable balanced weighting

# Regularization
--embed-dim N         # PCA dimensionality reduction
--lr-c FLOAT          # Logistic regression regularization
--rf-max-depth INT    # Random forest depth limit
```

---

*Report generated for C Function Understanding Pipeline*
*Last updated: March 2025*
