import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_metrics_scores(json_path: str | Path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    test_run_data = data.get("testRunData", data)
    metrics_scores = test_run_data.get("metricsScores")

    if metrics_scores is None:
        raise KeyError("Could not find 'metricsScores' in the JSON file.")

    # Convert to dict keyed by metric name for easy lookup
    metrics_by_name = {}
    for m in metrics_scores:
        name = m.get("metric", "Unnamed metric")
        metrics_by_name[name] = {
            "scores": m.get("scores", []),
            "passes": m.get("passes"),
            "fails": m.get("fails"),
            "errors": m.get("errors"),
        }

    return metrics_by_name


def plot_comparison(
    metrics_run1: dict,
    metrics_run2: dict,
    label_run1: str = "Run 1",
    label_run2: str = "Run 2",
):
    threshold = 0.7

    # Only compare metrics present in BOTH runs
    common_metrics = set(metrics_run1.keys()) & set(metrics_run2.keys())
    if not common_metrics:
        print("No common metrics between runs, nothing to plot.")
        return

    for metric_name in sorted(common_metrics):
        data1 = metrics_run1[metric_name]
        data2 = metrics_run2[metric_name]

        scores1 = data1["scores"]
        scores2 = data2["scores"]

        if not scores1 or not scores2:
            print(f"Skipping metric '{metric_name}' due to missing scores.")
            continue

        # Use the minimum length to ensure alignment
        n = min(len(scores1), len(scores2))
        if len(scores1) != len(scores2):
            print(
                f"Warning: metric '{metric_name}' has different number of scores "
                f"({len(scores1)} vs {len(scores2)}). "
                f"Only comparing first {n} entries."
            )

        scores1 = scores1[:n]
        scores2 = scores2[:n]

        x = list(range(n))  # test case indices 0..n-1

        # Bar positions for side-by-side comparison
        width = 0.4
        x1 = [i - width / 2 for i in x]
        x2 = [i + width / 2 for i in x]

        plt.figure(figsize=(10, 4))

        # One color per run for comparability
        plt.bar(x1, scores1, width=width, label=label_run1)
        plt.bar(x2, scores2, width=width, label=label_run2)

        # Threshold line
        plt.axhline(threshold, linestyle="--", linewidth=1)
        plt.text(
            n - 0.5,
            threshold + 0.02,
            f"Threshold = {threshold}",
            ha="right",
            va="bottom",
            fontsize=9,
        )

        title = (
            f"{metric_name}\n"
            f"{label_run1}: passes={data1['passes']}, fails={data1['fails']} | "
            f"{label_run2}: passes={data2['passes']}, fails={data2['fails']}"
        )
        plt.title(title)
        plt.xlabel("Test Case Index")
        plt.ylabel("Score (0–1)")
        plt.ylim(0, 1)
        plt.xticks(x)
        plt.grid(axis="y", alpha=0.4)
        plt.legend()
        plt.tight_layout()

    plt.show()


def main():
    # 🔧 CHANGE THESE to your actual file names / labels
    json_path_run1 = Path(".deepeval", ".latest_test_run.json")
    json_path_run2 = Path(".deepeval", ".latest_test_run-dummy.json")
    label_run1 = "Current Run"
    label_run2 = "Previous Run"

    metrics_run1 = load_metrics_scores(json_path_run1)
    metrics_run2 = load_metrics_scores(json_path_run2)

    plot_comparison(metrics_run1, metrics_run2, label_run1, label_run2)


if __name__ == "__main__":
    main()


# def main():
#     json_path = Path(".deepeval",".latest_test_run.json")  # change if needed
#     metrics_scores = load_metrics_scores(json_path)
#     plot_bar_charts(metrics_scores)


# if __name__ == "__main__":
#     main()
