# Low-Level Code Representation Learning

A function-level code understanding pipeline for C source code using embedding-based representation learning.

## Overview

This project implements **Track A: Embedding-Based Function Understanding** with the following capabilities:

1. **Function Extraction**: Extracts functions from C files using tree-sitter AST parsing
2. **Semantic Labeling**: Generates structured labels using LLM (Claude) or offline heuristics
3. **Embedding Generation**: Creates embeddings using CodeBERT (768-dim), Qwen3-Embedding (up to 4096-dim), and other models
4. **Similarity Search**: Finds semantically similar functions via cosine similarity
5. **Classification**: Predicts function side effects, complexity, and error handling using embeddings + classifiers

> For detailed technical design decisions, see [DESIGN_REPORT.md](DESIGN_REPORT.md).
> For comprehensive experiment results and ablations, see [EXPERIMENTS_REPORT.md](EXPERIMENTS_REPORT.md).
> For a quick reference of all pipeline configurations, see [EXPERIMENTS.md](EXPERIMENTS.md).

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# With LLM labeling (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key"
python run_pipeline.py --demo

# Without API (offline labeling)
python run_pipeline.py --offline --demo
```

### Run Inference on a C File

```bash
python src/infer.py --file path/to/your/file.c
```

A sample output is available at [`examples/hmac_analysis.json`](examples/hmac_analysis.json) (generated from `data/raw/hmac.c`).

## Project Structure

```
low-level-code-repr-learning/
├── data/
│   ├── raw/                    # Input C files (~50 files from Linux kernel, curl, Redis)
│   └── processed/              # Generated outputs
│       ├── labeled_functions.json
│       ├── embedded_functions.json
│       ├── embeddings.npy
│       ├── classifier.joblib
│       └── evaluation_report.json
├── src/
│   ├── extract.py              # Function extraction (tree-sitter)
│   ├── label.py                # LLM labeling pipeline
│   ├── embed.py                # Embedding generation & classification
│   ├── evaluate.py             # Evaluation metrics
│   └── infer.py                # CLI inference interface
├── examples/
│   └── hmac_analysis.json      # Sample inference output
├── notebooks/
│   └── analysis.ipynb          # Exploratory analysis
├── run_pipeline.py             # Main pipeline script
├── clean_labels.py             # Utility: remove rare label classes
├── DESIGN_REPORT.md            # Technical design decisions & architecture
├── EXPERIMENTS_REPORT.md       # Detailed experiment results & ablations
├── EXPERIMENTS.md              # Quick reference for all configurations
├── requirements.txt
└── README.md
```

## Part 1: Dataset Construction

### Function Extraction

Functions are extracted using **tree-sitter** with a regex fallback:

```python
from src.extract import CFunctionExtractor

extractor = CFunctionExtractor()
functions = extractor.extract_from_file("sample.c")
```

Each extracted function contains:
- `function_name`: Name of the function
- `function_code`: Complete source code
- `file_path`: Source file path
- `start_line`, `end_line`: Line numbers

### Dataset Statistics

- **Total Functions**: 780 (extracted from ~50 C files)
- **Sources**: Linux kernel drivers (watchdog, GPIO), curl, Redis utilities, Lua bindings
- **Side Effect Distribution**: hardware (315), global_state (294), memory (281), none (110), io (105), network (9), assertions (1)

### Labeling Strategy

Labels are generated using a **hybrid approach**:

| Label | Method | Rationale |
|-------|--------|-----------|
| `control_flow_elements` | Deterministic regex/AST | 100% accurate - directly verifiable from code |
| `side_effects` | LLM (Claude) or heuristics | Requires semantic understanding of function behavior |
| `complexity` | LLM or code metrics | Based on lines, nesting depth, control flow count |
| `error_handling` | LLM or pattern matching | Detects error handling idioms |
| `high_level_purpose` | LLM or nearest-neighbor | Free-form text, needs language understanding |

**Why this is reliable:**

1. **Control flow** (`if`, `for`, `while`, `switch`, `goto`, `return`) is extracted deterministically via regex pattern matching. No model needed, 100% accuracy.

2. **Side effects** classification uses:
   - **LLM mode**: Claude API with `temperature=0` for deterministic output, structured JSON output
   - **Offline mode**: Pattern matching for known I/O, memory, network, and hardware functions

3. **Complexity** is estimated from:
   - Line count (< 20 = low, 20-50 = medium, > 50 = high)
   - Control flow element count
   - Nesting depth (brace counting)

4. **Error handling** is detected via pattern matching for common idioms (`return -1`, `errno`, `assert`).

5. **Validation**: Control flow labels can be cross-verified against AST. Other labels are spot-checked on 10% sample.

## Part 2: Embedding-Based Function Understanding

### Model Choice: CodeBERT

**Default Model**: `microsoft/codebert-base`

**Why CodeBERT:**
- Pre-trained on 6M functions across 6 programming languages (including C)
- Understands both natural language and code semantics
- 768-dimensional embeddings suitable for downstream tasks
- Well-documented, reproducible, runs on CPU

**Additional models supported** (via `--embedder` flag):

| Model | Dimension | Context | Notes |
|-------|-----------|---------|-------|
| `codebert` | 768 | 512 tokens | Default, fast, CPU-friendly |
| `qwen3` | 1024 | 32k | Qwen3-Embedding-0.6B |
| `qwen3-4b` | 2560 | 32k | Best overall quality in our experiments |
| `qwen3-8b` | 4096 | 32k | Highest dimensionality |
| `nomic` | 768 | 32k | Code-specific |
| `codesage` | 2048 | 1024 | Code-specific, 1.3B params |

### Demonstrated Capabilities

#### 1. Function Similarity Search

Find functions semantically similar to a query:

```python
from src.embed import FunctionEmbeddingPipeline

pipeline = FunctionEmbeddingPipeline()
pipeline.load("data/processed")

# Find similar functions
results = pipeline.similarity_search(query_code, top_k=5)
for func, similarity in results:
    print(f"{func.function_name}: {similarity:.3f}")
```

#### 2. Side Effects Classification

Predict side effects using embeddings + classifiers:

```python
# Train classifier
metrics = pipeline.train_classifier()
print(f"Test F1: {metrics['test_f1_macro']:.3f}")

# Predict on new code
effects = pipeline.predict_side_effects(new_function_code)
# Returns: ["memory", "io"] or ["none"]
```

#### 3. Clustering by Semantic Purpose

```python
cluster_results = pipeline.cluster_functions(n_clusters=4)
print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")
```

#### 4. Multi-task Classification

Beyond side effects, the pipeline also classifies:
- **Complexity** (low/medium/high)
- **Error handling** (returns_code/uses_errno/assertions/none)

**Classification Labels:**

*Side Effects (multi-label):*
- `io`: File operations (fopen, fread), console I/O (printf, scanf)
- `memory`: Heap allocation/deallocation (malloc, free, memcpy)
- `hardware`: Register access, port I/O, interrupts, DMA
- `network`: Socket operations (socket, send, recv, connect)
- `global_state`: Reads/modifies global or static variables
- `none`: Pure computation with no side effects

*Complexity (single-label):*
- `low`: Simple logic, few branches, < 20 lines
- `medium`: Moderate branching, loops, 20-50 lines
- `high`: Complex control flow, nested loops, > 50 lines

*Error Handling (single-label):*
- `returns_code`: Returns error codes (0/-1, NULL)
- `uses_errno`: Uses errno or perror
- `assertions`: Uses assert() or similar
- `none`: No explicit error handling

## Part 3: Evaluation

### Metrics

| Task | Metric | Description |
|------|--------|-------------|
| Classification | F1-macro | Balanced across all classes |
| Classification | Per-class precision/recall | Identifies weak classes |
| Similarity Search | Side-effect overlap@K | Do similar functions have similar behavior? |
| Clustering | Silhouette score | Internal cluster cohesion |
| Clustering | Label purity | Alignment with side_effects labels |

### Best Results (780 functions)

| Task | Configuration | Test F1 |
|------|---------------|---------|
| **Side Effects** | Qwen3-4B + MLP + all features | **0.629** |
| **Complexity** | Nomic + Logistic Regression | **0.704** |
| **Error Handling** | Qwen3-4B + LR + purpose embeddings | **0.749** |
| **Similarity Overlap@5** | Qwen3-4B embeddings | **0.755** |

> For full experiment results across all configurations, see [EXPERIMENTS_REPORT.md](EXPERIMENTS_REPORT.md).

### Failure Analysis

**Identified failure cases:**

1. **Short utility functions**: Functions like getters/setters have similar syntactic structure regardless of purpose, causing poor discrimination in embedding space.

   ```c
   // These embed similarly despite different semantics
   int get_value(struct obj *o) { return o->value; }
   int get_count(struct list *l) { return l->count; }
   ```

2. **Similar syntax, different behavior**: Functions with similar control flow but different side effects.

   ```c
   // Both have if-return pattern but different side effects
   void *alloc_buffer(int n) { if (n <= 0) return NULL; return malloc(n); }
   int validate_input(int n) { if (n <= 0) return 0; return 1; }
   ```

3. **Severe overfitting across all models**: Train F1 = 1.0 vs Test F1 ~ 0.62, indicating the dataset is too small for the embedding dimensionality. This is a fundamental data limitation, not a model issue.

**Recommendations:**
1. Incorporate AST features (control flow depth, statement types) for better discrimination
2. Use contrastive learning to separate functions with different side effects
3. Add data augmentation for underrepresented categories
4. Scale the dataset to thousands of functions to reduce overfitting

## Part 4: Inference Interface

### CLI Usage

```bash
python src/infer.py --file sample.c
```

### Sample Output

See [`examples/hmac_analysis.json`](examples/hmac_analysis.json) for a complete example. Abbreviated output:

```json
{
  "file_path": "data/raw/hmac.c",
  "functions": [
    {
      "name": "Curl_HMAC_init",
      "start_line": 45,
      "end_line": 99,
      "embedding_summary": "Initializes an HMAC context by allocating memory...",
      "predicted_labels": {
        "high_level_purpose": "Initializes an HMAC context by allocating memory...",
        "control_flow_elements": ["if", "for", "goto", "return"],
        "side_effects": ["memory"],
        "complexity": "medium",
        "error_handling": "returns_code"
      }
    }
  ]
}
```

### Additional Options

```bash
# Use LLM for better summaries (requires API key)
python src/infer.py --file sample.c --use-llm

# Output to file
python src/infer.py --file sample.c --output results.json

# Use custom model directory
python src/infer.py --file sample.c --model-dir /path/to/model
```

## Bonus Tasks

### 1. AST-Aware Modeling (Bonus 1)

The pipeline incorporates structural information via the `ASTFeatureExtractor` (in `src/embed.py`):

- **Statement type distribution**: Counts of assignments, declarations, calls, returns, if/for/while/switch/goto/break/continue statements (normalized)
- **Control-flow depth**: Max loop nesting depth, max branch nesting depth
- **Cyclomatic complexity**: Decision points including if, for, while, case, &&, ||, ?:

These features are concatenated with embeddings when using the `--ast-features` flag:

```bash
python run_pipeline.py --offline --ast-features
```

Additionally, regex-based hybrid features (20 features covering API patterns and code metrics) can be enabled with `--hybrid-features`.

### 2. Continual Pretraining Thought Experiment (Bonus 3)

**Scaling to Millions of Functions:**

**Data Strategy:**
1. Source: Linux kernel, FreeBSD, embedded firmware repositories
2. Deduplication: MinHash + exact matching to remove duplicates
3. Quality filtering: Parse-able code only, exclude auto-generated files
4. Curriculum: Start with simpler functions, progressively add complex ones

**Training Objectives:**
1. **Continual Pretraining**: Next-token prediction on C code corpus
2. **Supervised Fine-tuning**: (function, structured_label) pairs from LLM-generated labels
3. **Contrastive Learning**: Pull together functions with same side effects, push apart different ones

**Infrastructure:**
- FSDP (Fully Sharded Data Parallel) for multi-GPU training
- Gradient checkpointing for memory efficiency
- Streaming datasets (HuggingFace datasets with streaming=True)
- Mixed precision (bf16) for faster training

**Challenges:**
1. Tokenizer coverage: C syntax (`->`, `::`, `#define`) needs adequate vocabulary
2. Context length: Long functions may exceed model context window
3. Label noise: LLM-generated labels at scale need quality control sampling

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|----------|-----------|----------|
| CodeBERT as default (Qwen3-4B as best) | CodeBERT is fast and accessible; Qwen3-4B gives best quality | Speed vs quality |
| Deterministic control flow extraction | 100% accurate, no model overhead | N/A - strictly better |
| LLM labeling with offline fallback | Quality + accessibility | API cost vs accuracy |
| Multiple classifier support | Different tasks benefit from different models | Complexity vs flexibility |
| CLS token pooling (CodeBERT) / Mean pooling (others) | Standard approaches for each architecture | May miss long-range dependencies |

> For a deeper analysis of these decisions, see [DESIGN_REPORT.md](DESIGN_REPORT.md).

## Reproducibility

```bash
# Set random seeds
export PYTHONHASHSEED=42

# Run pipeline (offline, no API needed)
python run_pipeline.py --offline

# Results should match evaluation_report.json
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- scikit-learn 1.3+
- tree-sitter, tree-sitter-c
- anthropic (optional, for LLM labeling)
- imbalanced-learn (optional, for SMOTE)
- sentence-transformers (optional, for Qwen3 embeddings)
