import json
from pathlib import Path
from datetime import datetime

from matplotlib.ticker import MaxNLocator

def load_metrics_scores(json_path: str | Path):
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    test_run_data = data.get("testRunData", data)
    metrics_scores = test_run_data.get("metricsScores")

    if metrics_scores is None:
        raise KeyError("Could not find 'metricsScores' in the JSON file.")

    return metrics_scores


def save_plot(file_name:str, 
              save_parent_p:Path,
            ) -> None:
    import matplotlib.pyplot as plt
    
    safe_name = file_name.replace(" ", "_").replace('.','').replace(":","_")
    
    output_file = save_parent_p / Path(f"{safe_name}.png")
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

def plot_token_counts_bar(*, token_counts: list[int], 
                          title:str,
                          save_folder_name:Path):
    import matplotlib.pyplot as plt
    
    if not token_counts:
        raise ValueError("token_counts is empty")

    x = list(range(len(token_counts)))

    fig, ax = plt.subplots()  
    ax.bar(x, token_counts)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Token count")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)

    save_plot(file_name=title.replace(" ", "_").replace('.','').replace(":","_"), 
              save_parent_p=save_folder_name)


def plot_bar_charts(save_folder_name:Path, 
                    metrics_scores:list[dict], 
                    hyperparams: dict):
    import matplotlib.pyplot as plt
    
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
        
        save_plot(file_name=metric_name, 
                  save_parent_p=save_folder_name)


def main():
    sub_dir = "2912-2025_2339"
    json_path = Path("evals", sub_dir, ".latest_test_run.json")  # change if needed
    metrics_scores = load_metrics_scores(json_path)

    now = datetime.now().strftime("%d-%m-%Y_%H%M")
    eval_parent_p = Path("evals", "plots", f"{sub_dir}_{now}")
    eval_parent_p.mkdir(exist_ok=True, parents=True)

    plot_bar_charts(metrics_scores=metrics_scores, 
                    save_folder_name=eval_parent_p,
                    hyperparams={"Config name":"hp-ch500", 
                                })
    
    token_counts = [233, 150, 1406]  # replace with your real array
    now = datetime.now().strftime("%d-%m-%Y_%H%M")
    # token_count_parent_p = Path("stats", "index_stats", now)
    # token_count_parent_p.mkdir(exist_ok=True, parents=True)

    # plot_token_counts_bar(token_counts=token_counts,
    #                       save_folder_name=token_count_parent_p,
    #                       title="Book token_counts")


if __name__ == "__main__":
    main()
