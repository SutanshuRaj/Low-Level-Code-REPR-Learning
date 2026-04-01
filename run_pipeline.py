#!/usr/bin/env python3
"""
Main pipeline script for C function understanding.

Runs the complete pipeline:
1. Extract functions from C files
2. Generate labels (LLM or offline)
3. Generate embeddings with CodeBERT
4. Train classifier
5. Evaluate and generate report

Usage:
    python run_pipeline.py                    # Full pipeline with LLM labeling
    python run_pipeline.py --offline          # Use offline labeling (no API needed)
    python run_pipeline.py --skip-labeling    # Use existing labeled dataset
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_extraction(input_dir: str) -> list:
    """Step 1: Extract functions from C files."""
    from src.extract import extract_functions_from_directory

    print("\n" + "=" * 60)
    print("STEP 1: Function Extraction")
    print("=" * 60)

    functions = extract_functions_from_directory(input_dir)
    print(f"Extracted {len(functions)} functions from {input_dir}")

    # Show summary
    files = set(f.file_path for f in functions)
    print(f"Files processed: {len(files)}")
    for file in sorted(files):
        count = sum(1 for f in functions if f.file_path == file)
        print(f"  - {Path(file).name}: {count} functions")

    return functions


def run_labeling(functions: list, output_path: str, use_offline: bool = False) -> list:
    """Step 2: Generate labels for functions."""
    from src.extract import CFunction
    from src.label import FunctionLabeler, OfflineLabeler, save_labeled_dataset

    print("\n" + "=" * 60)
    print("STEP 2: Label Generation")
    print("=" * 60)

    if use_offline:
        print("Using offline labeling (heuristic-based, no API needed)")
        labeler = OfflineLabeler()
    else:
        print("Using LLM labeling (requires ANTHROPIC_API_KEY)")
        try:
            labeler = FunctionLabeler()
        except Exception as e:
            print(f"Warning: LLM labeling failed ({e}). Falling back to offline mode.")
            labeler = OfflineLabeler()

    # Convert to CFunction objects if needed
    if functions and not isinstance(functions[0], CFunction):
        functions = [CFunction(**f) if isinstance(f, dict) else f for f in functions]

    labeled = labeler.label_functions(functions)
    save_labeled_dataset(labeled, output_path)

    # Show label distribution
    side_effect_counts = {}
    for lf in labeled:
        for se in lf.labels.get("side_effects", ["none"]):
            side_effect_counts[se] = side_effect_counts.get(se, 0) + 1

    print(f"\nSide effect distribution:")
    for se, count in sorted(side_effect_counts.items()):
        print(f"  - {se}: {count}")

    return [lf.to_dict() for lf in labeled]


def run_embedding(labeled_data: list, output_dir: str,
                  embedder_type: str = "codebert",
                  classifier_type: str = "random_forest",
                  side_effects_clf: str = None,
                  complexity_clf: str = None,
                  error_handling_clf: str = None,
                  tune_hyperparams: bool = False,
                  # Logistic Regression params
                  lr_c: float = 1.0,
                  lr_max_iter: int = 1000,
                  lr_solver: str = "lbfgs",
                  # Random Forest params
                  rf_n_estimators: int = 100,
                  rf_max_depth: int = 20,
                  rf_min_samples_split: int = 5,
                  rf_min_samples_leaf: int = 1,
                  # SVM params
                  svm_c: float = 1.0,
                  svm_kernel: str = "rbf",
                  svm_gamma: str = "scale",
                  # MLP params
                  mlp_hidden_layers: tuple = (256, 128),
                  mlp_activation: str = "relu",
                  mlp_learning_rate: float = 0.001,
                  mlp_max_iter: int = 500,
                  mlp_early_stopping: bool = True,
                  use_class_weight: bool = True,
                  embed_dim: int = None,
                  use_binary_classifiers: bool = False,
                  use_smote: bool = False,
                  use_hybrid_features: bool = False,
                  use_hardware_features: bool = False,
                  use_ast_features: bool = False,
                  use_purpose_embeddings: bool = False,
                  use_threshold_tuning: bool = False) -> "FunctionEmbeddingPipeline":
    """Step 3: Generate embeddings and train classifier."""
    from src.embed import FunctionEmbeddingPipeline, get_embedder

    print("\n" + "=" * 60)
    print("STEP 3: Embedding Generation & Classification")
    print("=" * 60)
    print(f"Embedder: {embedder_type}")

    # Resolve per-label classifiers (use default if not specified)
    se_clf = side_effects_clf or classifier_type
    cx_clf = complexity_clf or classifier_type
    eh_clf = error_handling_clf or classifier_type

    print(f"Classifiers:")
    print(f"  side_effects:   {se_clf}")
    print(f"  complexity:     {cx_clf}")
    print(f"  error_handling: {eh_clf}")
    if use_hybrid_features:
        if use_hardware_features:
            print(f"  Hybrid features: ENABLED (40 features: 20 base + 20 embedded/bare-metal HW)")
        else:
            print(f"  Hybrid features: ENABLED (20 base regex features)")
    if use_ast_features:
        print(f"  AST features: ENABLED (statement distribution, control-flow depth)")
    if use_purpose_embeddings:
        print(f"  Purpose embeddings: ENABLED (high_level_purpose text)")
    if use_threshold_tuning:
        print(f"  Threshold tuning: ENABLED (per-class optimization)")

    # Create embedder based on type
    embedder = get_embedder(embedder_type)

    pipeline = FunctionEmbeddingPipeline(embedder=embedder)
    pipeline.use_hardware_features = use_hardware_features
    pipeline.embed_labeled_functions(labeled_data)

    # Train classifier
    print(f"\nTraining classifiers...")
    metrics = pipeline.train_classifier(
        test_size=0.3,
        side_effects_clf=se_clf,
        complexity_clf=cx_clf,
        error_handling_clf=eh_clf,
        tune_hyperparams=tune_hyperparams,
        lr_c=lr_c,
        lr_max_iter=lr_max_iter,
        lr_solver=lr_solver,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_split=rf_min_samples_split,
        rf_min_samples_leaf=rf_min_samples_leaf,
        svm_c=svm_c,
        svm_kernel=svm_kernel,
        svm_gamma=svm_gamma,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_activation=mlp_activation,
        mlp_learning_rate=mlp_learning_rate,
        mlp_max_iter=mlp_max_iter,
        mlp_early_stopping=mlp_early_stopping,
        use_class_weight=use_class_weight,
        embed_dim=embed_dim,
        use_binary_classifiers=use_binary_classifiers,
        use_smote=use_smote,
        use_hybrid_features=use_hybrid_features,
        use_ast_features=use_ast_features,
        use_purpose_embeddings=use_purpose_embeddings,
        use_threshold_tuning=use_threshold_tuning
    )
    print(f"  Train F1 (macro): {metrics['train_f1_macro']:.3f}")
    print(f"  Test F1 (macro): {metrics['test_f1_macro']:.3f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")

    # Show per-task metrics
    if "complexity" in metrics:
        print(f"  Complexity F1: {metrics['complexity']['test_f1_macro']:.3f}")
    if "error_handling" in metrics:
        print(f"  Error Handling F1: {metrics['error_handling']['test_f1_macro']:.3f}")

    # Cluster functions
    print("\nClustering functions by semantic similarity...")
    n_clusters = min(4, len(labeled_data) // 3)  # Adaptive cluster count
    if n_clusters >= 2:
        cluster_results = pipeline.cluster_functions(n_clusters=n_clusters)
        print(f"  Silhouette score: {cluster_results['silhouette_score']:.3f}")
        print(f"  Cluster sizes: {cluster_results['cluster_sizes']}")

    # Save pipeline
    pipeline.save(output_dir)

    return pipeline


def run_evaluation(pipeline, output_path: str):
    """Step 4: Comprehensive evaluation."""
    from src.evaluate import generate_evaluation_report

    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

    report = generate_evaluation_report(pipeline, output_path)
    return report


def demonstrate_similarity_search(pipeline):
    """Demonstrate similarity search capability."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Similarity Search")
    print("=" * 60)

    if not pipeline.embedded_functions:
        print("No functions available for demonstration")
        return

    # Pick a sample function
    sample = pipeline.embedded_functions[0]
    print(f"\nQuery function: {sample.function_name}")
    print(f"  Side effects: {sample.labels.get('side_effects', [])}")

    similar = pipeline.find_similar_to_function(sample.function_name, top_k=3)

    print("\nMost similar functions:")
    for func, score in similar:
        print(f"  - {func.function_name} (similarity: {score:.3f})")
        print(f"    Side effects: {func.labels.get('side_effects', [])}")


def demonstrate_classification(pipeline):
    """Demonstrate classification capability."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Classification")
    print("=" * 60)

    if pipeline.classifier is None:
        print("Classifier not trained")
        return

    # Test on a few functions
    test_cases = [
        ("malloc test", """void* create_buffer(int size) {
            void *buf = malloc(size);
            if (buf) memset(buf, 0, size);
            return buf;
        }"""),
        ("IO test", """void log_error(const char *msg) {
            fprintf(stderr, "ERROR: %s\\n", msg);
        }"""),
        ("Pure computation", """int add_numbers(int a, int b) {
            return a + b;
        }"""),
    ]

    print("\nPredicting side effects for test functions:")
    for name, code in test_cases:
        predicted = pipeline.predict_side_effects(code)
        print(f"\n  {name}:")
        print(f"    Predicted: {predicted}")


def main():
    parser = argparse.ArgumentParser(
        description="ML Pipeline for C Function Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--input-dir", "-i",
        default="data/raw",
        help="Directory containing C source files (default: data/raw)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed",
        help="Output directory for processed data (default: data/processed)"
    )

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline labeling (no API needed)"
    )

    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip labeling step (use existing labeled dataset)"
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding step (use existing embeddings)"
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstrations after pipeline"
    )

    parser.add_argument(
        "--classifier", "-c",
        default="random_forest",
        choices=["logistic_regression", "random_forest", "svm", "mlp"],
        help="Default classifier type for all labels (default: random_forest)"
    )

    parser.add_argument(
        "--side-effects-clf",
        default=None,
        choices=["logistic_regression", "random_forest", "svm", "mlp"],
        help="Classifier for side_effects (overrides --classifier)"
    )

    parser.add_argument(
        "--complexity-clf",
        default=None,
        choices=["logistic_regression", "random_forest", "svm", "mlp"],
        help="Classifier for complexity (overrides --classifier)"
    )

    parser.add_argument(
        "--error-handling-clf",
        default=None,
        choices=["logistic_regression", "random_forest", "svm", "mlp"],
        help="Classifier for error_handling (overrides --classifier)"
    )

    parser.add_argument(
        "--embedder", "-e",
        default="codebert",
        choices=["codebert", "qwen3", "qwen3-4b", "nomic", "jina", "codesage"],
        help="Embedding model: codebert, qwen3 (0.6B), qwen3-4b, nomic, jina, or codesage (default: codebert)"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with GridSearchCV (slower but better)"
    )

    # Logistic Regression hyperparameters
    parser.add_argument(
        "--lr-c",
        type=float, default=1.0,
        help="Logistic Regression: inverse regularization strength (default: 1.0, higher = less regularization)"
    )

    parser.add_argument(
        "--lr-max-iter",
        type=int, default=1000,
        help="Logistic Regression: maximum iterations (default: 1000)"
    )

    parser.add_argument(
        "--lr-solver",
        default="lbfgs",
        choices=["lbfgs", "saga", "liblinear", "newton-cg"],
        help="Logistic Regression: optimization solver (default: lbfgs)"
    )

    # Random Forest hyperparameters
    parser.add_argument(
        "--rf-n-estimators",
        type=int, default=100,
        help="Random Forest: number of trees (default: 100)"
    )

    parser.add_argument(
        "--rf-max-depth",
        type=int, default=20,
        help="Random Forest: max tree depth (default: 20, use 0 for unlimited)"
    )

    parser.add_argument(
        "--rf-min-samples-split",
        type=int, default=5,
        help="Random Forest: min samples to split a node (default: 5)"
    )

    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int, default=1,
        help="Random Forest: min samples at leaf node (default: 1)"
    )

    # SVM hyperparameters
    parser.add_argument(
        "--svm-c",
        type=float, default=1.0,
        help="SVM: regularization parameter (default: 1.0, higher = less regularization)"
    )

    parser.add_argument(
        "--svm-kernel",
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="SVM: kernel type (default: rbf)"
    )

    parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="SVM: kernel coefficient - 'scale', 'auto', or float (default: scale)"
    )

    # MLP hyperparameters
    parser.add_argument(
        "--mlp-hidden-layers",
        default="256,128",
        help="MLP: hidden layer sizes as comma-separated values (default: 256,128)"
    )

    parser.add_argument(
        "--mlp-activation",
        default="relu",
        choices=["relu", "tanh", "logistic"],
        help="MLP: activation function (default: relu)"
    )

    parser.add_argument(
        "--mlp-learning-rate",
        type=float, default=0.001,
        help="MLP: initial learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--mlp-max-iter",
        type=int, default=500,
        help="MLP: max training epochs (default: 500)"
    )

    parser.add_argument(
        "--mlp-no-early-stopping",
        action="store_true",
        help="MLP: disable early stopping (not recommended)"
    )

    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class_weight='balanced' (use if data is already balanced)"
    )

    parser.add_argument(
        "--embed-dim",
        type=int, default=None,
        help="Reduce embedding dimensions using PCA (e.g., 128, 256). Helps prevent overfitting."
    )

    parser.add_argument(
        "--binary-classifiers",
        action="store_true",
        help="Use separate binary classifier per side effect (better for imbalanced multi-label)"
    )

    parser.add_argument(
        "--smote",
        action="store_true",
        help="Use SMOTE upsampling to balance classes (requires imbalanced-learn)"
    )

    parser.add_argument(
        "--hybrid-features",
        action="store_true",
        help="Concatenate hand-crafted code features (API calls, metrics) with embeddings (20 base features)"
    )

    parser.add_argument(
        "--hardware-features",
        action="store_true",
        help="Add 20 embedded/bare-metal HW features (MMIO, regmap, IRQ, SPI, I2C, GPIO, etc). Implies --hybrid-features"
    )

    parser.add_argument(
        "--threshold-tuning",
        action="store_true",
        help="Optimize prediction threshold per class using validation data (improves F1)"
    )

    parser.add_argument(
        "--ast-features",
        action="store_true",
        help="Add AST-based features (statement distribution, control-flow depth, cyclomatic complexity)"
    )

    parser.add_argument(
        "--purpose-embeddings",
        action="store_true",
        help="Add high_level_purpose text embeddings as features"
    )

    args = parser.parse_args()

    # --hardware-features implies --hybrid-features
    if args.hardware_features:
        args.hybrid_features = True

    # Ensure directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    labeled_path = Path(args.output_dir) / "labeled_functions.json"
    report_path = Path(args.output_dir) / "evaluation_report.json"

    # Step 1 & 2: Extract and label
    if args.skip_labeling:
        print(f"Loading existing labeled dataset from {labeled_path}")
        with open(labeled_path, 'r') as f:
            labeled_data = json.load(f)
    else:
        functions = run_extraction(args.input_dir)
        if not functions:
            print("Error: No functions extracted. Check input directory.")
            sys.exit(1)

        labeled_data = run_labeling(
            functions,
            str(labeled_path),
            use_offline=args.offline
        )

    # Step 3: Embedding and classification
    if args.skip_embedding:
        print(f"\nLoading existing pipeline from {args.output_dir}")
        from src.embed import FunctionEmbeddingPipeline
        pipeline = FunctionEmbeddingPipeline()
        pipeline.load(args.output_dir)
    else:
        # Handle rf_max_depth: 0 means unlimited (None)
        rf_max_depth = args.rf_max_depth if args.rf_max_depth > 0 else None

        # Parse MLP hidden layers from comma-separated string to tuple
        mlp_hidden_layers = tuple(int(x.strip()) for x in args.mlp_hidden_layers.split(","))

        pipeline = run_embedding(
            labeled_data,
            args.output_dir,
            embedder_type=args.embedder,
            classifier_type=args.classifier,
            side_effects_clf=args.side_effects_clf,
            complexity_clf=args.complexity_clf,
            error_handling_clf=args.error_handling_clf,
            tune_hyperparams=args.tune,
            lr_c=args.lr_c,
            lr_max_iter=args.lr_max_iter,
            lr_solver=args.lr_solver,
            rf_n_estimators=args.rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_split=args.rf_min_samples_split,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
            svm_c=args.svm_c,
            svm_kernel=args.svm_kernel,
            svm_gamma=args.svm_gamma,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_activation=args.mlp_activation,
            mlp_learning_rate=args.mlp_learning_rate,
            mlp_max_iter=args.mlp_max_iter,
            mlp_early_stopping=not args.mlp_no_early_stopping,
            use_class_weight=not args.no_class_weight,
            embed_dim=args.embed_dim,
            use_binary_classifiers=args.binary_classifiers,
            use_smote=args.smote,
            use_hybrid_features=args.hybrid_features,
            use_hardware_features=args.hardware_features,
            use_ast_features=args.ast_features,
            use_purpose_embeddings=args.purpose_embeddings,
            use_threshold_tuning=args.threshold_tuning
        )

    # Step 4: Evaluation
    run_evaluation(pipeline, str(report_path))

    # Demonstrations
    if args.demo:
        demonstrate_similarity_search(pipeline)
        demonstrate_classification(pipeline)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {args.output_dir}/")
    print(f"  - labeled_functions.json")
    print(f"  - embedded_functions.json")
    print(f"  - embeddings.npy")
    print(f"  - classifier.joblib")
    print(f"  - evaluation_report.json")
    print(f"\nTo run inference on a new file:")
    print(f"  python src/infer.py --file <path_to_c_file>")


if __name__ == "__main__":
    main()
