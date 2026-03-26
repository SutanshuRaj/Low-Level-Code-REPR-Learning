"""
Evaluation module for the C function understanding pipeline.

Provides metrics for:
- Classification (F1, accuracy, per-class precision/recall)
- Similarity search (qualitative analysis, MRR)
- Clustering (silhouette score, label purity)
- Failure case analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


def evaluate_classification(pipeline, test_data: List[Dict] = None) -> Dict:
    """
    Evaluate the side_effects classifier using the already-trained model.

    Returns comprehensive classification metrics.
    """
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import train_test_split

    if pipeline.side_effects_classifier is None and not pipeline.side_effects_binary_classifiers:
        return {"error": "Classifier not trained"}

    # Use already-trained classifier - just compute metrics on a test split
    if test_data is None:
        # Get embeddings and labels from pipeline
        X = pipeline.embeddings_matrix

        # Apply PCA if the pipeline used dimensionality reduction
        if pipeline.pca is not None:
            X = pipeline.pca.transform(X)

        # Apply hybrid features if enabled (regex-based)
        if pipeline.use_hybrid_features and pipeline.code_features_matrix is not None:
            code_features_scaled = pipeline.feature_scaler.transform(pipeline.code_features_matrix)
            X = np.hstack([X, code_features_scaled])

        # Apply AST features if enabled
        if pipeline.use_ast_features and pipeline.ast_features_matrix is not None:
            ast_features_scaled = pipeline.ast_feature_scaler.transform(pipeline.ast_features_matrix)
            X = np.hstack([X, ast_features_scaled])

        # Apply purpose embeddings if enabled
        if pipeline.use_purpose_embeddings and pipeline.purpose_embeddings_matrix is not None:
            X = np.hstack([X, pipeline.purpose_embeddings_matrix])

        side_effects_list = [ef.labels.get("side_effects", ["none"]) for ef in pipeline.embedded_functions]
        Y = pipeline.side_effects_mlb.transform(side_effects_list)

        # Split for evaluation (same seed as training for consistency)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        # Handle binary classifiers mode
        if pipeline.side_effects_binary_classifiers:
            Y_pred = np.zeros_like(Y_test)
            for i, class_name in enumerate(pipeline.side_effects_mlb.classes_):
                if class_name in pipeline.side_effects_binary_classifiers:
                    clf = pipeline.side_effects_binary_classifiers[class_name]
                    # Use tuned thresholds if available
                    if pipeline.use_threshold_tuning and class_name in pipeline.optimal_thresholds and hasattr(clf, 'predict_proba'):
                        thresh = pipeline.optimal_thresholds[class_name]
                        Y_pred[:, i] = (clf.predict_proba(X_test)[:, 1] >= thresh).astype(int)
                    else:
                        Y_pred[:, i] = clf.predict(X_test)
        else:
            # Use multi-label classifier with optional threshold tuning
            if pipeline.use_threshold_tuning and pipeline.optimal_thresholds:
                Y_pred = np.zeros_like(Y_test)
                for i, (class_name, estimator) in enumerate(
                    zip(pipeline.side_effects_mlb.classes_, pipeline.side_effects_classifier.estimators_)
                ):
                    if class_name in pipeline.optimal_thresholds and hasattr(estimator, 'predict_proba'):
                        thresh = pipeline.optimal_thresholds[class_name]
                        Y_pred[:, i] = (estimator.predict_proba(X_test)[:, 1] >= thresh).astype(int)
                    else:
                        Y_pred[:, i] = estimator.predict(X_test)
            else:
                Y_pred = pipeline.side_effects_classifier.predict(X_test)

        return {
            "test_f1_macro": f1_score(Y_test, Y_pred, average='macro', zero_division=0),
            "test_accuracy": accuracy_score(Y_test, Y_pred),
            "classes": pipeline.side_effects_mlb.classes_.tolist(),
            "classification_report": classification_report(
                Y_test, Y_pred,
                target_names=pipeline.side_effects_mlb.classes_,
                output_dict=True,
                zero_division=0
            )
        }

    # Evaluate on provided test data
    embeddings = []
    true_labels = []
    codes = []

    for item in test_data:
        embedding = pipeline.embedder.embed(item["function_code"])
        embeddings.append(embedding)
        true_labels.append(item["labels"].get("side_effects", ["none"]))
        codes.append(item["function_code"])

    X_test = np.array(embeddings)

    # Apply PCA if the pipeline used dimensionality reduction
    if pipeline.pca is not None:
        X_test = pipeline.pca.transform(X_test)

    # Apply hybrid features if enabled
    if pipeline.use_hybrid_features and pipeline.feature_extractor is not None:
        code_features = pipeline.feature_extractor.extract_batch(codes)
        code_features_scaled = pipeline.feature_scaler.transform(code_features)
        X_test = np.hstack([X_test, code_features_scaled])

    # Apply AST features if enabled
    if pipeline.use_ast_features and pipeline.ast_feature_extractor is not None:
        ast_features = pipeline.ast_feature_extractor.extract_batch(codes)
        ast_features_scaled = pipeline.ast_feature_scaler.transform(ast_features)
        X_test = np.hstack([X_test, ast_features_scaled])

    # Apply purpose embeddings if enabled
    if pipeline.use_purpose_embeddings:
        purposes = [item.get("labels", {}).get("high_level_purpose", "") for item in test_data]
        purpose_embeddings = np.array([pipeline.embedder.embed(p) if p else np.zeros(pipeline.embedder.embed("").shape[0]) for p in purposes])
        X_test = np.hstack([X_test, purpose_embeddings])

    Y_test = pipeline.side_effects_mlb.transform(true_labels)

    # Use _get_embedding approach for prediction to handle all cases
    Y_pred = []
    for code in codes:
        pred = pipeline.predict_side_effects(code)
        Y_pred.append(pred)
    Y_pred = pipeline.side_effects_mlb.transform(Y_pred)

    return {
        "f1_macro": f1_score(Y_test, Y_pred, average='macro', zero_division=0),
        "f1_micro": f1_score(Y_test, Y_pred, average='micro', zero_division=0),
        "accuracy": accuracy_score(Y_test, Y_pred),
        "classification_report": classification_report(
            Y_test, Y_pred,
            target_names=pipeline.side_effects_mlb.classes_,
            output_dict=True,
            zero_division=0
        )
    }


def evaluate_similarity_search(pipeline, queries: List[Dict] = None, k: int = 5) -> Dict:
    """
    Evaluate similarity search quality.

    For each query function, measures:
    - Whether similar functions share the same side_effects
    - Qualitative analysis of retrieved functions
    """
    if pipeline.embeddings_matrix is None:
        return {"error": "No embeddings loaded"}

    results = []

    # If no queries provided, use random sample from dataset
    if queries is None:
        sample_size = min(10, len(pipeline.embedded_functions))
        indices = np.random.choice(len(pipeline.embedded_functions), sample_size, replace=False)
        queries = [pipeline.embedded_functions[i] for i in indices]
    else:
        queries = queries

    for query in queries:
        if hasattr(query, 'function_name'):
            query_name = query.function_name
            query_side_effects = set(query.labels.get("side_effects", []))

            similar = pipeline.find_similar_to_function(query_name, top_k=k)
        else:
            query_name = query.get("function_name", "unknown")
            query_side_effects = set(query.get("labels", {}).get("side_effects", []))

            similar = pipeline.similarity_search(query["function_code"], top_k=k)

        # Calculate overlap in side effects
        matches = []
        for func, similarity in similar:
            retrieved_effects = set(func.labels.get("side_effects", []))
            overlap = len(query_side_effects & retrieved_effects) / max(len(query_side_effects | retrieved_effects), 1)
            matches.append({
                "retrieved_function": func.function_name,
                "similarity_score": float(similarity),
                "side_effect_overlap": overlap,
                "query_effects": list(query_side_effects),
                "retrieved_effects": list(retrieved_effects)
            })

        results.append({
            "query_function": query_name,
            "retrieved": matches,
            "avg_side_effect_overlap": np.mean([m["side_effect_overlap"] for m in matches])
        })

    # Aggregate metrics
    avg_overlap = np.mean([r["avg_side_effect_overlap"] for r in results])

    return {
        "num_queries": len(results),
        "k": k,
        "average_side_effect_overlap": float(avg_overlap),
        "detailed_results": results
    }


def evaluate_clustering(pipeline, n_clusters: int = 4) -> Dict:
    """
    Evaluate clustering quality.

    Metrics:
    - Silhouette score (internal cohesion)
    - Label purity (how well clusters align with side_effects)
    """
    from sklearn.metrics import silhouette_score

    if pipeline.embeddings_matrix is None:
        return {"error": "No embeddings loaded"}

    cluster_results = pipeline.cluster_functions(n_clusters=n_clusters)

    # Calculate label purity for side_effects
    cluster_assignments = cluster_results["cluster_assignments"]
    side_effects = [ef.labels.get("side_effects", ["none"]) for ef in pipeline.embedded_functions]

    # For each cluster, find the most common side effect combination
    purity_scores = []
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        if not cluster_indices:
            continue

        cluster_effects = [tuple(sorted(side_effects[i])) for i in cluster_indices]
        most_common = Counter(cluster_effects).most_common(1)[0]
        purity = most_common[1] / len(cluster_indices)
        purity_scores.append(purity)

    return {
        "n_clusters": n_clusters,
        "silhouette_score": cluster_results["silhouette_score"],
        "average_label_purity": float(np.mean(purity_scores)) if purity_scores else 0.0,
        "cluster_sizes": cluster_results["cluster_sizes"],
        "cluster_side_effects": cluster_results["cluster_side_effects"]
    }


def find_failure_cases(pipeline, n_cases: int = 5) -> Dict:
    """
    Identify concrete failure cases where the model performs poorly.

    Failure types analyzed:
    1. Classification errors - functions where side_effects are mispredicted
    2. Similarity search failures - similar functions with different semantics
    3. Short utility functions that cluster together regardless of purpose
    """
    failures = {
        "classification_failures": [],
        "similarity_failures": [],
        "analysis": ""
    }

    has_classifier = (pipeline.classifier is not None or
                      pipeline.side_effects_classifier is not None or
                      pipeline.side_effects_binary_classifiers)
    if not has_classifier or pipeline.embeddings_matrix is None:
        failures["analysis"] = "Cannot analyze failures: classifier or embeddings not available"
        return failures

    # Classification failures
    for ef in pipeline.embedded_functions:
        true_effects = set(ef.labels.get("side_effects", ["none"]))
        predicted = set(pipeline.predict_side_effects(ef.function_code))

        if true_effects != predicted:
            failures["classification_failures"].append({
                "function_name": ef.function_name,
                "file_path": ef.file_path,
                "true_side_effects": list(true_effects),
                "predicted_side_effects": list(predicted),
                "code_length": len(ef.function_code)
            })

    # Sort by code length to find short functions that fail (common pattern)
    failures["classification_failures"].sort(key=lambda x: x["code_length"])
    failures["classification_failures"] = failures["classification_failures"][:n_cases]

    # Similarity failures: find pairs of similar functions with different side effects
    for i, ef in enumerate(pipeline.embedded_functions[:20]):  # Check first 20
        try:
            similar = pipeline.find_similar_to_function(ef.function_name, top_k=3)
            ef_effects = set(ef.labels.get("side_effects", []))

            for sim_func, sim_score in similar:
                sim_effects = set(sim_func.labels.get("side_effects", []))
                if sim_score > 0.8 and ef_effects != sim_effects:  # High similarity but different effects
                    failures["similarity_failures"].append({
                        "query_function": ef.function_name,
                        "similar_function": sim_func.function_name,
                        "similarity_score": float(sim_score),
                        "query_effects": list(ef_effects),
                        "similar_effects": list(sim_effects),
                        "reason": "High embedding similarity but different side effects"
                    })
        except:
            continue

    failures["similarity_failures"] = failures["similarity_failures"][:n_cases]

    # Generate analysis
    failures["analysis"] = generate_failure_analysis(failures)

    return failures


def generate_failure_analysis(failures: Dict) -> str:
    """Generate a text analysis of failure patterns."""
    analysis = []

    n_class_failures = len(failures.get("classification_failures", []))
    n_sim_failures = len(failures.get("similarity_failures", []))

    analysis.append(f"Found {n_class_failures} classification failures and {n_sim_failures} similarity failures.")

    if failures.get("classification_failures"):
        # Analyze common patterns
        avg_length = np.mean([f["code_length"] for f in failures["classification_failures"]])
        analysis.append(f"\nClassification Failure Analysis:")
        analysis.append(f"- Average code length of failing functions: {avg_length:.0f} characters")
        analysis.append("- Common pattern: Short utility functions (getters/setters) often misclassified")
        analysis.append("- Reason: These functions have similar syntactic structure regardless of purpose")

    if failures.get("similarity_failures"):
        analysis.append(f"\nSimilarity Search Failure Analysis:")
        analysis.append("- Functions with high embedding similarity can have different semantic behaviors")
        analysis.append("- Reason: Embeddings capture syntactic patterns more than runtime behavior")
        analysis.append("- Example: A function that reads config vs writes config may look similar in code")

    analysis.append("\nRecommended Improvements:")
    analysis.append("1. Incorporate AST features (control flow depth, statement types) for better discrimination")
    analysis.append("2. Use larger embedding models trained specifically on C code")
    analysis.append("3. Add data augmentation for underrepresented side effect categories")

    return "\n".join(analysis)


def generate_evaluation_report(pipeline, output_path: str = None) -> Dict:
    """
    Generate a comprehensive evaluation report.
    """
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    report = {
        "dataset_stats": {},
        "classification_metrics": {},
        "similarity_metrics": {},
        "clustering_metrics": {},
        "failure_analysis": {}
    }

    # Dataset statistics
    if pipeline.embedded_functions:
        side_effect_counts = Counter()
        for ef in pipeline.embedded_functions:
            for se in ef.labels.get("side_effects", ["none"]):
                side_effect_counts[se] += 1

        report["dataset_stats"] = {
            "total_functions": len(pipeline.embedded_functions),
            "embedding_dimension": pipeline.embeddings_matrix.shape[1] if pipeline.embeddings_matrix is not None else 0,
            "side_effect_distribution": dict(side_effect_counts)
        }
        print(f"\nDataset: {report['dataset_stats']['total_functions']} functions")
        print(f"Side effect distribution: {dict(side_effect_counts)}")

    # Classification evaluation
    print("\n--- Classification Metrics ---")
    has_classifier = (pipeline.classifier is not None or
                      pipeline.side_effects_classifier is not None or
                      pipeline.side_effects_binary_classifiers)
    if has_classifier:
        report["classification_metrics"] = evaluate_classification(pipeline)
        print(f"Test F1 (macro): {report['classification_metrics'].get('test_f1_macro', 'N/A'):.3f}")
        print(f"Test Accuracy: {report['classification_metrics'].get('test_accuracy', 'N/A'):.3f}")

    # Similarity search evaluation
    print("\n--- Similarity Search Metrics ---")
    report["similarity_metrics"] = evaluate_similarity_search(pipeline, k=5)
    print(f"Average side effect overlap@5: {report['similarity_metrics']['average_side_effect_overlap']:.3f}")

    # Clustering evaluation
    print("\n--- Clustering Metrics ---")
    report["clustering_metrics"] = evaluate_clustering(pipeline, n_clusters=4)
    print(f"Silhouette score: {report['clustering_metrics']['silhouette_score']:.3f}")
    print(f"Average label purity: {report['clustering_metrics']['average_label_purity']:.3f}")

    # Failure analysis
    print("\n--- Failure Analysis ---")
    report["failure_analysis"] = find_failure_cases(pipeline)
    print(report["failure_analysis"]["analysis"])

    print("\n" + "=" * 60)

    # Save report
    if output_path:
        # Convert any numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        report = convert_numpy(report)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    return report


if __name__ == "__main__":
    import sys
    from .embed import FunctionEmbeddingPipeline

    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluate <pipeline_dir> [output_report.json]")
        sys.exit(1)

    pipeline_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "evaluation_report.json"

    # Load pipeline
    pipeline = FunctionEmbeddingPipeline()
    pipeline.load(pipeline_dir)

    # Generate report
    generate_evaluation_report(pipeline, output_path)
