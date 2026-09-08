import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List

# Ensure core_api and src are in python search path
CURRENT_DIR = Path(__file__).resolve().parent
CHATBOT_API_DIR = CURRENT_DIR.parent
if str(CHATBOT_API_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_API_DIR))

DEFAULT_DATASET_PATH = CURRENT_DIR / "cypher_eval_dataset.json"
DEFAULT_OUTPUT_PATH = Path("eval_results.json")

logger = logging.getLogger("eval_cypher")


def normalize_cypher(query: str) -> str:
    """Normalizes Cypher query for comparison by removing formatting differences."""
    if not query:
        return ""
    # Strip markdown code blocks
    q = re.sub(r"^```(?:cypher)?\s*", "", query.strip(), flags=re.IGNORECASE)
    q = re.sub(r"\s*```$", "", q.strip())
    # Strip trailing semicolon and collapse whitespace
    q = q.rstrip(";").strip()
    return " ".join(q.split()).lower()


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads evaluation dataset from JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of test case objects")
    return data


def run_evaluation(
    dataset: List[Dict[str, Any]],
    mock_run: bool = False,
) -> Dict[str, Any]:
    """Runs evaluation over the dataset using enterprise_cypher_chain."""
    chain = None
    if not mock_run:
        from src.chains.enterprise_cypher_chain import enterprise_cypher_chain
        chain = enterprise_cypher_chain
        chain.return_intermediate_steps = True

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_cypher = item["expected_cypher"]
        category = item.get("category", "unknown")

        start_time = time.perf_counter()
        generated_cypher = ""
        execution_success = False
        execution_error = None

        if mock_run:
            # Mock mode for testing evaluation pipeline offline
            time.sleep(0.01)
            generated_cypher = expected_cypher
            execution_success = True
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        else:
            try:
                chain_response = chain.invoke(question)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

                steps = chain_response.get("intermediate_steps", [])
                if steps and isinstance(steps[0], dict):
                    generated_cypher = steps[0].get("query", "")

                # If chain succeeded without raising, Cypher generation & execution succeeded
                execution_success = True
            except Exception as e:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                execution_error = str(e)
                execution_success = False

        norm_gen = normalize_cypher(generated_cypher)
        norm_exp = normalize_cypher(expected_cypher)
        exact_match = bool(norm_gen and norm_exp and norm_gen == norm_exp)

        results.append({
            "id": idx,
            "question": question,
            "category": category,
            "expected_cypher": expected_cypher,
            "generated_cypher": generated_cypher,
            "exact_match": exact_match,
            "execution_success": execution_success,
            "execution_error": execution_error,
            "latency_ms": duration_ms,
        })

    return summarize_results(results)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates metrics overall and broken down by category."""
    total_cases = len(results)
    if total_cases == 0:
        return {"total_cases": 0, "results": []}

    categories = sorted(list({r["category"] for r in results}))
    category_summary: Dict[str, Any] = {}

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_count = len(cat_results)
        cat_exact = sum(1 for r in cat_results if r["exact_match"])
        cat_exec_success = sum(1 for r in cat_results if r["execution_success"])
        cat_latencies = [r["latency_ms"] for r in cat_results]

        category_summary[cat] = {
            "count": cat_count,
            "exact_match_count": cat_exact,
            "exact_match_pct": round((cat_exact / cat_count) * 100, 2),
            "execution_success_count": cat_exec_success,
            "execution_success_pct": round((cat_exec_success / cat_count) * 100, 2),
            "avg_latency_ms": round(sum(cat_latencies) / cat_count, 2),
            "min_latency_ms": min(cat_latencies),
            "max_latency_ms": max(cat_latencies),
        }

    total_exact = sum(1 for r in results if r["exact_match"])
    total_exec = sum(1 for r in results if r["execution_success"])
    all_latencies = [r["latency_ms"] for r in results]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "overall_exact_match_count": total_exact,
        "overall_exact_match_pct": round((total_exact / total_cases) * 100, 2),
        "overall_execution_success_count": total_exec,
        "overall_execution_success_pct": round((total_exec / total_cases) * 100, 2),
        "overall_avg_latency_ms": round(sum(all_latencies) / total_cases, 2),
        "overall_min_latency_ms": min(all_latencies),
        "overall_max_latency_ms": max(all_latencies),
        "category_breakdown": category_summary,
        "results": results,
    }
    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Prints a formatted ASCII summary table to stdout."""
    cat_data = summary.get("category_breakdown", {})

    print("\n" + "=" * 78)
    print("                      CYPHER EVALUATION SUMMARY")
    print("=" * 78)
    headers = f"{'Category':<16} | {'Count':<7} | {'Exact Match %':<15} | {'Exec Success %':<16} | {'Avg Latency (ms)':<15}"
    print(headers)
    print("-" * 78)

    for cat, stats in cat_data.items():
        print(
            f"{cat:<16} | "
            f"{stats['count']:<7} | "
            f"{stats['exact_match_pct']:>13.2f}% | "
            f"{stats['execution_success_pct']:>14.2f}% | "
            f"{stats['avg_latency_ms']:>16.2f}"
        )

    print("-" * 78)
    print(
        f"{'OVERALL':<16} | "
        f"{summary['total_cases']:<7} | "
        f"{summary['overall_exact_match_pct']:>13.2f}% | "
        f"{summary['overall_execution_success_pct']:>14.2f}% | "
        f"{summary['overall_avg_latency_ms']:>16.2f}"
    )
    print("=" * 78 + "\n")


def save_results(summary: Dict[str, Any], output_path: Path) -> None:
    """Saves raw evaluation results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results successfully saved to: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cypher Generation and Execution")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output eval_results.json file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode without live LLM or Neo4j connections",
    )
    args = parser.parse_args()

    print(f"Loading evaluation dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} test cases across categories.")

    summary = run_evaluation(dataset, mock_run=args.mock)
    print_summary_table(summary)
    save_results(summary, args.output)


if __name__ == "__main__":
    main()
