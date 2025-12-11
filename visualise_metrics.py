import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics_scores(json_path: str | Path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both:
    # - {"testRunData": { "metricsScores": [...] }}
    # - {"metricsScores": [...]}
    test_run_data = data.get("testRunData", data)
    metrics_scores = test_run_data.get("metricsScores")

    if metrics_scores is None:
        raise KeyError(
            "Could not find 'metricsScores'. "
            "Make sure the JSON has either testRunData.metricsScores or metricsScores at the top level."
        )

    return metrics_scores


def plot_bar_charts(metrics_scores:list[dict]):
    threshold = 0.7

    for metric_obj in metrics_scores:
        metric_name = metric_obj.get("metric", "Unnamed metric")
        scores = metric_obj.get("scores", [])
        passes = metric_obj.get("passes")
        fails = metric_obj.get("fails")
        errors = metric_obj.get("errors")

        if not scores:
            print(f"No scores for {metric_name}, skipping.")
            continue

        x = list(range(len(scores)))  # 0, 1, 2, ...

        # Color map: green for >= threshold, red-ish for below
        colors = ["green" if s >= threshold else "salmon" for s in scores]

        plt.figure(figsize=(10, 4))
        plt.bar(x, scores, color=colors)
        
        plt.axhline(threshold, color="black", linestyle="--", linewidth=1)
        plt.text(
            len(scores) - 0.5, threshold + 0.02,
            f"Threshold = {threshold}",
            ha="right", va="bottom", fontsize=9, color="black"
        )


        plt.title(
            f"{metric_name}\npasses={passes}, fails={fails}, errors={errors}"
        )
        plt.xlabel("Test Case Index")
        plt.ylabel("Score (0–1)")
        plt.ylim(0, 1)
        plt.xticks(x)
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()

    plt.show()

def main():
    json_path = Path(".deepeval",".latest_test_run.json")  # change if needed
    metrics_scores = load_metrics_scores(json_path)
    plot_bar_charts(metrics_scores)


if __name__ == "__main__":
    main()
