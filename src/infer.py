#!/usr/bin/env python3
"""
CLI inference interface for C function analysis.

Usage:
    python infer.py --file sample.c
    python infer.py --file sample.c --model-dir data/processed
    python infer.py --file sample.c --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract import CFunctionExtractor, extract_control_flow
from src.embed import FunctionEmbeddingPipeline, get_embedding_summary


def analyze_file(
    file_path: str,
    pipeline: Optional[FunctionEmbeddingPipeline] = None,
    use_llm_summary: bool = False
) -> Dict:
    """
    Analyze a C file and return structured JSON output.

    Output format:
    {
        "file_path": "...",
        "functions": [
            {
                "name": "init_driver",
                "embedding_summary": "...",
                "predicted_labels": {
                    "high_level_purpose": "...",
                    "control_flow_elements": [...],
                    "side_effects": [...]
                }
            }
        ]
    }
    """
    # Extract functions
    extractor = CFunctionExtractor()
    functions = extractor.extract_from_file(file_path)

    if not functions:
        return {
            "file_path": file_path,
            "functions": [],
            "warning": "No functions extracted from file"
        }

    results = {
        "file_path": file_path,
        "functions": []
    }

    for func in functions:
        func_result = {
            "name": func.function_name,
            "start_line": func.start_line,
            "end_line": func.end_line,
            "embedding_summary": "",
            "predicted_labels": {
                "high_level_purpose": "",
                "control_flow_elements": [],
                "side_effects": [],
                "complexity": "",
                "error_handling": ""
            }
        }

        # Extract control flow deterministically (no model needed)
        control_flow = extract_control_flow(func.function_code)
        func_result["predicted_labels"]["control_flow_elements"] = control_flow

        # If pipeline is loaded, use embeddings for predictions
        if pipeline is not None and pipeline.side_effects_classifier is not None:
            # Predict all labels using trained classifiers
            func_result["predicted_labels"]["side_effects"] = pipeline.predict_side_effects(func.function_code)

            if pipeline.complexity_classifier is not None:
                func_result["predicted_labels"]["complexity"] = pipeline.predict_complexity(func.function_code)

            if pipeline.error_handling_classifier is not None:
                func_result["predicted_labels"]["error_handling"] = pipeline.predict_error_handling(func.function_code)

            # Get embedding summary from nearest neighbor
            func_result["embedding_summary"] = get_embedding_summary(func.function_code, pipeline)
            func_result["predicted_labels"]["high_level_purpose"] = func_result["embedding_summary"]
        else:
            # Fallback: use heuristic-based labeling
            from src.label import OfflineLabeler
            labeler = OfflineLabeler()
            labeled = labeler.label_function(func)

            func_result["predicted_labels"]["side_effects"] = labeled.labels["side_effects"]
            func_result["predicted_labels"]["complexity"] = labeled.labels["complexity"]
            func_result["predicted_labels"]["error_handling"] = labeled.labels["error_handling"]
            func_result["predicted_labels"]["high_level_purpose"] = labeled.labels["high_level_purpose"]
            func_result["embedding_summary"] = labeled.labels["high_level_purpose"]

        # Optionally use LLM for better summaries
        if use_llm_summary:
            try:
                from src.label import FunctionLabeler
                llm_labeler = FunctionLabeler()
                llm_labels = llm_labeler._get_llm_labels(func.function_code)
                func_result["embedding_summary"] = llm_labels.get("high_level_purpose", func_result["embedding_summary"])
                func_result["predicted_labels"]["high_level_purpose"] = llm_labels.get("high_level_purpose", "")
            except Exception as e:
                print(f"Warning: LLM summary failed for {func.function_name}: {e}", file=sys.stderr)

        results["functions"].append(func_result)

    return results


def load_pipeline(model_dir: str) -> Optional[FunctionEmbeddingPipeline]:
    """Load a trained pipeline from disk."""
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"Warning: Model directory {model_dir} not found. Using fallback mode.", file=sys.stderr)
        return None

    required_files = ["embeddings.npy", "embedded_functions.json"]
    for f in required_files:
        if not (model_path / f).exists():
            print(f"Warning: {f} not found in {model_dir}. Using fallback mode.", file=sys.stderr)
            return None

    try:
        pipeline = FunctionEmbeddingPipeline()
        pipeline.load(model_dir)
        return pipeline
    except Exception as e:
        print(f"Warning: Failed to load pipeline: {e}. Using fallback mode.", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze C source files and generate structured semantic outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer.py --file sample.c
  python infer.py --file sample.c --model-dir data/processed
  python infer.py --file sample.c --output results.json --use-llm
        """
    )

    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to C source file to analyze"
    )

    parser.add_argument(
        "--model-dir", "-m",
        default="data/processed",
        help="Directory containing trained model (default: data/processed)"
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for generating function summaries (requires ANTHROPIC_API_KEY)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)"
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    if not args.file.endswith('.c'):
        print(f"Warning: File does not have .c extension: {args.file}", file=sys.stderr)

    # Load pipeline (optional)
    pipeline = load_pipeline(args.model_dir)

    # Analyze file
    results = analyze_file(
        args.file,
        pipeline=pipeline,
        use_llm_summary=args.use_llm
    )

    # Output results
    indent = 2 if args.pretty else None
    output_json = json.dumps(results, indent=indent)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
