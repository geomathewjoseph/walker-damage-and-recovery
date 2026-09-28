"""
Evolutionary Resilience Benchmark
Compares the standard trained agent vs. the evolved damage-resistant agent
under acute neural lesions without retraining.

Generates comparative retention curves and failure-rate metrics.
Author: Geo Mathew Joseph
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO

from config import (
    ENV_NAME, DEVICE, SEED, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, HEALTHY_CHECKPOINT
)
from damage import apply_damage


def evaluate_policy(model: PPO, n_episodes: int = 10, seed: int = 42) -> dict:
    """Evaluates a policy on BipedalWalker and returns reward stats."""
    env = gym.make(ENV_NAME)
    rewards = []
    lengths = []
    falls = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_rew = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, truncated, _ = env.step(action)
            ep_rew += rew
            ep_len += 1

        rewards.append(ep_rew)
        lengths.append(ep_len)
        if ep_rew < 0:
            falls += 1

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "fall_rate": float(falls / n_episodes),
        "mean_length": float(np.mean(lengths)),
    }


def run_benchmark(
    standard_model_path: str = HEALTHY_CHECKPOINT,
    evolved_model_path: str = os.path.join(CHECKPOINT_DIR, "evolved_resistant.zip"),
    severities: list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    n_episodes: int = 10,
    output_json: str = os.path.join(LOG_DIR, "evolution_benchmark.json"),
    output_plot: str = os.path.join(PLOT_DIR, "evolution_comparison.png"),
):
    print("=" * 65)
    print("  EVOLUTIONARY RESILIENCE BENCHMARK")
    print(f"  Standard Model: {standard_model_path}")
    print(f"  Evolved Model:  {evolved_model_path}")
    print(f"  Severities:     {severities}")
    print("=" * 65)

    standard_model = PPO.load(standard_model_path, device=DEVICE)
    evolved_model = PPO.load(evolved_model_path, device=DEVICE)

    results = {
        "severities": severities,
        "standard": [],
        "evolved": [],
    }

    for sev in severities:
        print(f"\nEvaluating Severity {sev:.1f} (Neuron Kill)...")

        # 1. Standard model with damage
        std_damaged = apply_damage(standard_model, severity=sev, mode="neuron_kill", seed=42)
        std_stats = evaluate_policy(std_damaged, n_episodes=n_episodes)
        results["standard"].append(std_stats)
        print(f"  Standard Agent: Reward = {std_stats['mean_reward']:+6.1f} | Falls = {std_stats['fall_rate']*100:.0f}%")

        # 2. Evolved model with damage
        evo_damaged = apply_damage(evolved_model, severity=sev, mode="neuron_kill", seed=42)
        evo_stats = evaluate_policy(evo_damaged, n_episodes=n_episodes)
        results["evolved"].append(evo_stats)
        print(f"  Evolved Agent:  Reward = {evo_stats['mean_reward']:+6.1f} | Falls = {evo_stats['fall_rate']*100:.0f}%")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to: {output_json}")

    # Generate Cyber-Dark Comparative Plot
    plot_evolution_comparison(results, output_plot)


def plot_evolution_comparison(results: dict, output_plot: str):
    severities = np.array(results["severities"]) * 100
    std_means = [r["mean_reward"] for r in results["standard"]]
    std_stds = [r["std_reward"] for r in results["standard"]]
    evo_means = [r["mean_reward"] for r in results["evolved"]]
    evo_stds = [r["std_reward"] for r in results["evolved"]]

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="#0b0f19")

    # Panel 1: Reward Retention vs Severity
    ax1.set_facecolor("#0f172a")
    ax1.grid(color="#1e293b", linestyle="--", linewidth=0.8, alpha=0.7)

    # Standard agent curve
    ax1.plot(severities, std_means, "o-", color="#ff3366", linewidth=2.5, markersize=8,
             label="Standard Agent (Vulnerable)", zorder=4)
    ax1.fill_between(severities, np.array(std_means) - np.array(std_stds),
                     np.array(std_means) + np.array(std_stds), color="#ff3366", alpha=0.18)

    # Evolved agent curve
    ax1.plot(severities, evo_means, "s-", color="#00e5ff", linewidth=2.5, markersize=8,
             label="Evolved Agent (Neuro-Resistant)", zorder=5)
    ax1.fill_between(severities, np.array(evo_means) - np.array(evo_stds),
                     np.array(evo_means) + np.array(evo_stds), color="#00e5ff", alpha=0.22)

    ax1.axhline(300, color="#00ff88", linestyle=":", linewidth=1.5, label="Gymnasium Solved (+300)")
    ax1.axhline(0, color="#64748b", linestyle="-", linewidth=1.0, alpha=0.8)

    ax1.set_xlabel("Lesion Severity (% Ablation)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.set_ylabel("Evaluation Reward (Zero-Shot)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax1.set_title("Zero-Shot Motor Resilience Under Neural Trauma", fontsize=14, fontweight="bold", color="#38bdf8", pad=12)
    ax1.legend(loc="lower left", facecolor="#1e293b", edgecolor="#334155", fontsize=10)

    # Panel 2: Catastrophic Fall Rates
    ax2.set_facecolor("#0f172a")
    ax2.grid(color="#1e293b", linestyle="--", linewidth=0.8, alpha=0.7)

    width = 3.5
    std_falls = [r["fall_rate"] * 100 for r in results["standard"]]
    evo_falls = [r["fall_rate"] * 100 for r in results["evolved"]]

    ax2.bar(severities - width/2, std_falls, width=width, color="#ff3366", alpha=0.85, label="Standard Agent Fall Rate")
    ax2.bar(severities + width/2, evo_falls, width=width, color="#00e5ff", alpha=0.85, label="Evolved Agent Fall Rate")

    ax2.set_xlabel("Lesion Severity (% Ablation)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.set_ylabel("Fall / Failure Rate (%)", fontsize=12, fontweight="bold", color="#e2e8f0")
    ax2.set_title("Catastrophic Locomotive Failure Comparison", fontsize=14, fontweight="bold", color="#38bdf8", pad=12)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", fontsize=10)

    # Add subtitle & author
    fig.suptitle("BIOMECHANICAL EVOLUTION: STANDARD VS. EVOLVED DAMAGE-RESISTANT WALKER\nAuthor: Geo Mathew Joseph",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=1.03)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved evolutionary comparison plot to: {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Evolved vs Standard Policy")
    parser.add_argument("--standard", default=HEALTHY_CHECKPOINT)
    parser.add_argument("--evolved", default=os.path.join(CHECKPOINT_DIR, "evolved_resistant.zip"))
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()

    run_benchmark(
        standard_model_path=args.standard,
        evolved_model_path=args.evolved,
        n_episodes=args.episodes,
    )
