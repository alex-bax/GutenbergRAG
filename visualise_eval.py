import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def load_metrics_scores(json_path: str | Path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    test_run_data = data.get("testRunData", data)
    metrics_scores = test_run_data.get("metricsScores")

    if metrics_scores is None:
        raise KeyError("Could not find 'metricsScores' in the JSON file.")

    return metrics_scores


def save_plot(metric_name:str, run_name:str) -> None:
    safe_name = metric_name.replace(" ", "_")
    outp_p = Path("evals", "plots", run_name)
    outp_p.mkdir(parents=True, exist_ok=True)
    output_file = outp_p / Path(f"{safe_name}.png")
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Saved: {output_file}")


def format_hyperparams(hyperparams: dict | None) -> str:
    if not hyperparams:
        return ""
    lines = ["Hyperparameters:"]
    for k, v in hyperparams.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def plot_bar_charts(folder_name:str, 
                    metrics_scores:list[dict], 
                    hyperparams: dict):
    threshold = 0.7
    hp_text = format_hyperparams(hyperparams)

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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x, scores, color=colors)

        ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.text(
            len(scores) - 0.5,
            threshold + 0.02,
            f"Threshold = {threshold}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="black",
        )

        ax.set_title(
            f"{metric_name}\npasses={passes}, fails={fails}, errors={errors}"
        )
        ax.set_xlabel("Test Case Index")
        ax.set_ylabel("Score (0–1)")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.4)

        # Figure coordinates: (0,0) bottom-left, (1,1) top-right
        fig.text(
            0.01,
            0.99,
            hp_text,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        now = datetime.now().strftime("%d-%m-%Y_%H%M")
        save_plot(metric_name, run_name=f"{folder_name}_{now}")


def main():
    sub_dir = "1013_1212-2025"
    json_path = Path("evals", sub_dir, "latest_test_run.json")  # change if needed
    metrics_scores = load_metrics_scores(json_path)
    plot_bar_charts(metrics_scores=metrics_scores, 
                    folder_name=sub_dir,
                    hyperparams={"Hyperparameter config":"hp-ch400", 
                                "rerank_model":"gpt-5-nano"})


if __name__ == "__main__":
    main()
