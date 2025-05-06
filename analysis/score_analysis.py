import os
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path


def extract_scores_from_file(file_path):
    """
    Extract scores from trial file. Returns two sets of scores:
    1. Detailed scores from the episode logs (with episode numbers)
    2. Raw scores from the second part of the file
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Extract detailed scores from episode logs
    episode_pattern = r"Episode: (\d+) -----Score: (\d+\.\d+) -----Epsilon:"
    episode_matches = re.findall(episode_pattern, content)

    detailed_episodes = []
    detailed_scores = []

    if episode_matches:
        for match in episode_matches:
            episode = int(match[0])
            score = float(match[1])
            detailed_episodes.append(episode)
            detailed_scores.append(score)

    # Extract raw scores (just numbers) from the second part
    # Find where the episode logs end and raw scores begin
    # We'll look for a stretch of lines that are just numbers
    lines = content.split("\n")
    raw_scores = []

    # Start from the end and go backwards until we find a non-numeric line
    for line in reversed(lines):
        line = line.strip()
        if line and line.replace(".", "", 1).isdigit():
            raw_scores.append(float(line))
        elif line and not line.replace(".", "", 1).isdigit() and raw_scores:
            # We've hit a non-numeric line after collecting raw scores, so break
            break

    # Reverse back to original order
    raw_scores.reverse()

    return {"detailed": (detailed_episodes, detailed_scores), "raw": raw_scores}


def plot_trial_scores(trial_data, output_path=None):
    """
    Plot scores from all trials with fill-between to show variance
    """
    plt.figure(figsize=(14, 8))

    # Plot detailed scores (with episode numbers)
    plt.subplot(2, 1, 1)

    max_episodes = 0
    all_detailed_scores = []
    for trial, data in trial_data.items():
        episodes, scores = data["detailed"]
        if episodes:
            max_episodes = max(max_episodes, max(episodes))
            plt.plot(episodes, scores, alpha=0.5, label=f"{trial} (detailed)")
            all_detailed_scores.append((episodes, scores))

    # Calculate mean and std for detailed scores
    if all_detailed_scores:
        # Create a common x-axis (episode numbers)
        common_x = np.arange(1, max_episodes + 1)
        resampled_scores = []

        for episodes, scores in all_detailed_scores:
            # Convert to numpy arrays for easier manipulation
            ep_array = np.array(episodes)
            score_array = np.array(scores)

            # Resample scores to the common x-axis
            resampled = np.zeros(max_episodes)
            for i, ep in enumerate(common_x):
                if ep in ep_array:
                    resampled[i] = score_array[np.where(ep_array == ep)[0][0]]
                elif i > 0:
                    # Use last known value if episode not found
                    resampled[i] = resampled[i - 1]

            resampled_scores.append(resampled)

        # Calculate mean and std
        if resampled_scores:
            score_array = np.array(resampled_scores)
            mean_scores = np.mean(score_array, axis=0)
            std_scores = np.std(score_array, axis=0)

            plt.plot(common_x, mean_scores, "k-", linewidth=2, label="Mean")
            plt.fill_between(
                common_x,
                mean_scores - std_scores,
                mean_scores + std_scores,
                alpha=0.2,
                color="gray",
                label="±1 Std Dev",
            )

    plt.title("Detailed Scores by Episode")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot raw scores
    plt.subplot(2, 1, 2)

    all_raw_scores = []
    max_length = 0

    for trial, data in trial_data.items():
        scores = data["raw"]
        if scores:
            max_length = max(max_length, len(scores))
            x = np.arange(1, len(scores) + 1)
            plt.plot(x, scores, alpha=0.5, label=f"{trial} (raw)")
            all_raw_scores.append(scores)

    # Calculate mean and std for raw scores
    if all_raw_scores:
        # Pad all arrays to the same length
        padded_scores = []
        for scores in all_raw_scores:
            padded = np.zeros(max_length)
            padded[: len(scores)] = scores
            for i in range(len(scores), max_length):
                padded[i] = padded[len(scores) - 1] if len(scores) > 0 else 0
            padded_scores.append(padded)

        # Calculate mean and std
        score_array = np.array(padded_scores)
        mean_scores = np.mean(score_array, axis=0)
        std_scores = np.std(score_array, axis=0)

        x = np.arange(1, max_length + 1)
        plt.plot(x, mean_scores, "k-", linewidth=2, label="Mean")
        plt.fill_between(
            x,
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2,
            color="gray",
            label="±1 Std Dev",
        )

    plt.title("Raw Scores")
    plt.xlabel("Index")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def calculate_statistics(trial_data):
    """
    Calculate combined statistics for all trials in an experiment
    """
    # Collect all raw scores from all trials
    all_scores = []
    for trial, data in trial_data.items():
        all_scores.extend(data["raw"])

    # Calculate statistics
    if all_scores:
        avg_score = np.mean(all_scores)
        max_score = np.max(all_scores)
        std_dev = np.std(all_scores)

        # Also calculate statistics for each individual trial
        trial_stats = {}
        for trial, data in trial_data.items():
            scores = data["raw"]
            if scores:
                trial_stats[trial] = {
                    "avg": np.mean(scores),
                    "max": np.max(scores),
                    "std": np.std(scores),
                    "count": len(scores),
                }

        return {
            "combined": {
                "avg": avg_score,
                "max": max_score,
                "std": std_dev,
                "count": len(all_scores),
            },
            "trials": trial_stats,
        }

    return None


def print_statistics(exp_name, stats):
    """
    Print formatted statistics for an experiment
    """
    if not stats:
        print(f"No statistics available for {exp_name}")
        return

    print("\n" + "=" * 60)
    print(f" EXPERIMENT: {exp_name} ".center(60, "="))
    print("=" * 60)

    combined = stats["combined"]
    print(f"\nCOMBINED STATISTICS (All Trials):")
    print(f"  Average Score: {combined['avg']:.2f}")
    print(f"  Maximum Score: {combined['max']:.2f}")
    print(f"  Standard Deviation: {combined['std']:.2f}")
    print(f"  Total Samples: {combined['count']}")

    print("\nINDIVIDUAL TRIAL STATISTICS:")
    for trial, trial_stat in stats["trials"].items():
        print(f"\n  {trial}:")
        print(f"    Average Score: {trial_stat['avg']:.2f}")
        print(f"    Maximum Score: {trial_stat['max']:.2f}")
        print(f"    Standard Deviation: {trial_stat['std']:.2f}")
        print(f"    Sample Count: {trial_stat['count']}")

    print("\n" + "=" * 60)


def analyze_all_experiments():
    """
    Analyze all experiments (exp_1 and exp_2)
    """
    base_dir = Path("/Users/nicholas/Documents/GitHub/neuro140")
    experiments = ["exp_1", "exp_2"]

    all_experiments_stats = {}

    for exp in experiments:
        results_dir = base_dir / exp / "results"
        trial_data = {}

        # Process each trial file
        for i in range(1, 7):  # trials 1-6
            trial_file = results_dir / f"trial_{i}.txt"
            if trial_file.exists():
                print(f"Processing {trial_file}")
                trial_data[f"Trial {i}"] = extract_scores_from_file(trial_file)
            else:
                print(f"Warning: {trial_file} not found")

        # Calculate statistics for this experiment
        stats = calculate_statistics(trial_data)
        all_experiments_stats[exp] = stats

        # Print statistics
        print_statistics(exp, stats)

        # Create plot for this experiment
        output_path = base_dir / exp / "images" / "score_analysis.png"

        # Make sure the images directory exists
        os.makedirs(base_dir / exp / "images", exist_ok=True)

        plot_trial_scores(trial_data, output_path)
        print(f"Plot saved to {output_path}")

    # Compare experiments
    if len(all_experiments_stats) >= 2:
        print("\n" + "=" * 60)
        print(" EXPERIMENT COMPARISON ".center(60, "="))
        print("=" * 60)

        for exp, stats in all_experiments_stats.items():
            if stats:
                print(f"\n{exp}:")
                print(f"  Average Score: {stats['combined']['avg']:.2f}")
                print(f"  Maximum Score: {stats['combined']['max']:.2f}")
                print(f"  Standard Deviation: {stats['combined']['std']:.2f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_all_experiments()
