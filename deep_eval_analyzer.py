import json
import sys
from pathlib import Path
from statistics import mean

def summarize_run(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    test_run = data["testRunData"]
    test_file = test_run.get("testFile")
    test_cases = test_run.get("testCases", [])
    metrics_scores = test_run.get("metricsScores", [])
    test_passed = test_run.get("testPassed")
    test_failed = test_run.get("testFailed")
    run_duration = test_run.get("runDuration")
    evaluation_cost = test_run.get("evaluationCost")

    print(f"=== DeepEval test run summary ===")
    print(f"File:          {test_file}")
    print(f"Test cases:    {len(test_cases)}")
    print(f"Passed:        {test_passed}")
    print(f"Failed:        {test_failed}")
    print(f"Run duration:  {run_duration:.2f} s")
    print(f"Eval cost:     {evaluation_cost:.6f}\n")

    # 1) High-level metric summary from metricsScores
    if metrics_scores:
        print("=== Metrics overview (from metricsScores) ===")
        for m in metrics_scores:
            metric = m["metric"]
            scores = m.get("scores", [])
            if not scores:
                continue

            passes = m.get("passes", 0)
            fails = m.get("fails", 0)
            errors = m.get("errors", 0)

            print(
                f"- {metric}: "
                f"avg={mean(scores):.3f}, "
                f"min={min(scores):.3f}, "
                f"max={max(scores):.3f}, "
                f"n={len(scores)}, "
                f"passes={passes}, fails={fails}, errors={errors}"
            )
        print()

    # 2) Build per-metric mapping from testCases.metricsData to locate worst examples
    metric_to_scores: dict[str, list[tuple[int, float]]] = {}

    for idx, tc in enumerate(test_cases):
        for m in tc.get("metricsData", []):
            name = m.get("name")
            score = m.get("score")
            if name is None or score is None:
                continue
            metric_to_scores.setdefault(name, []).append((idx, score))

    if not metric_to_scores:
        print("No metricsData found in testCases; nothing more to show.")
        return

    print("=== Worst examples per metric (from testCases.metricsData) ===")
    for metric_name, idx_scores in metric_to_scores.items():
        # Compute basic stats from per-test metric scores
        scores_only = [s for _, s in idx_scores]
        avg_score = mean(scores_only)
        min_score = min(scores_only)
        max_score = max(scores_only)

        print(f"\nMetric: {metric_name}")
        print(f"  avg={avg_score:.3f}, min={min_score:.3f}, max={max_score:.3f}, n={len(scores_only)}")

        # Find worst-scoring test case (first occurrence of min)
        worst_idx, worst_score = min(idx_scores, key=lambda pair: pair[1])
        worst_tc = test_cases[worst_idx]

        print(f"  Worst case index: {worst_idx} (score={worst_score:.3f})")
        print(f"  Test name:   {worst_tc.get('name')}")
        print(f"  Success:     {worst_tc.get('success')}")
        print(f"  Input:       {worst_tc.get('input')}")
        print(f"  Expected:    {worst_tc.get('expectedOutput')}")
        print(f"  Actual:      {worst_tc.get('actualOutput')}")

        # Context: show length and first snippet to avoid spam
        ctx = worst_tc.get("retrievalContext") or []
        if isinstance(ctx, list):
            print(f"  Context size: {len(ctx)} chunks")
            if ctx:
                # print just the first chunk trimmed
                first_ctx = str(ctx[0])
                if len(first_ctx) > 200:
                    first_ctx = first_ctx[:200] + "... [truncated]"
                print(f"  First context chunk: {first_ctx}")
        else:
            print("  Context: (not a list, unexpected structure)")

        # Try to show metric-specific reason if present
        metric_reason = None
        for m in worst_tc.get("metricsData", []):
            if m.get("name") == metric_name:
                metric_reason = m.get("reason")
                break

        if metric_reason:
            trimmed_reason = metric_reason
            if len(trimmed_reason) > 300:
                trimmed_reason = trimmed_reason[:300] + "... [truncated]"
            print(f"  Metric reason: {trimmed_reason}")


if __name__ == "__main__":
    # Default to .deepeval/.latest_test_run.json, but allow overriding via CLI arg
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        file_path = Path(".deepeval/.latest_test_run.json")

    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    summarize_run(file_path)