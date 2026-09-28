"""
3D Bipedal Robot Mech — Evolutionary Resilience Benchmark Suite.
Compares the standard trained agent vs. the evolved damage-resistant agent
under acute neural lesions without retraining (Zero-Shot Resilience).

Generates comparative retention curves, failure rate analysis, and speed retention metrics.

Author: Geo Mathew Joseph
"""
import os
import json
import argparse
import numpy as np

from stable_baselines3 import PPO

from config import (
    CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, SEED, DEVICE,
    WALKER_3D_CHECKPOINT,
)
from damage import apply_damage
from walker3d_env import Walker3DEnv


def evaluate_3d_lesion_resilience(
    model: PPO,
    n_episodes: int = 8,
    seed: int = 42,
) -> dict:
    """
    Evaluates a 3D policy across multiple episodes and collects
    fine-grained biomechanical and stability metrics.
    """
    env = Walker3DEnv(training_mode=False)
    rewards = []
    lengths = []
    forward_vels = []
    ctrl_efforts = []
    falls = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep * 10)
        done = False
        ep_rew = 0.0
        ep_len = 0
        ep_vys = []
        ep_ctrl = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            ep_rew += rew
            ep_len += 1
            ep_vys.append(info.get("vy", 0.0))
            ep_ctrl.append(np.sum(np.square(action)))
            done = term or trunc

        rewards.append(ep_rew)
        lengths.append(ep_len)
        forward_vels.append(np.mean(ep_vys) if ep_vys else 0.0)
        ctrl_efforts.append(np.mean(ep_ctrl) if ep_ctrl else 0.0)

        # Premature collapse defined as falling early with compromised gait
        if term and (ep_len < 45 or ep_rew < 150):
            falls += 1

    env.close()

    mean_rew = float(np.mean(rewards))
    mean_spd = float(np.mean(forward_vels))
    fall_prob = float(falls / n_episodes)
    mean_ctrl = float(np.mean(ctrl_efforts))

    return {
        "mean_reward": mean_rew,
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_forward_vel": mean_spd,
        "fall_rate": fall_prob,
        "mean_ctrl_cost": mean_ctrl,
    }


def run_evolution_benchmark(
    standard_model_path: str = WALKER_3D_CHECKPOINT,
    evolved_model_path: str = os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"),
    severities: list = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
    n_episodes: int = 8,
    output_json: str = os.path.join(LOG_DIR, "evolution_3d_benchmark.json"),
    mode: str = "neuron_kill",
):
    print("=" * 75)
    print("   AEGIS 3D BIPEDAL MECH — EVOLUTIONARY RESILIENCE AUDIT")
    print("   Zero-Shot Trauma Retention Benchmark (Standard vs Evolved)")
    print(f"   Standard Model: {standard_model_path}")
    print(f"   Evolved Model:  {evolved_model_path}")
    print(f"   Severities:     {severities}")
    print(f"   Lesion Mode:    {mode}")
    print("=" * 75)

    assert os.path.exists(standard_model_path), f"Missing standard model: {standard_model_path}"
    assert os.path.exists(evolved_model_path), f"Missing evolved model: {evolved_model_path}"

    standard_model = PPO.load(standard_model_path, device="cpu")
    evolved_model = PPO.load(evolved_model_path, device="cpu")

    results = {
        "severities": severities,
        "mode": mode,
        "standard": [],
        "evolved": [],
        "retention_standard": [],
        "retention_evolved": [],
    }

    std_baseline_reward = None
    evo_baseline_reward = None

    for sev in severities:
        print(f"\n[AUDIT] Evaluating Lesion Severity {sev*100:4.1f}% ({mode})...")

        # 1. Standard Model Evaluation
        std_damaged = apply_damage(standard_model, severity=sev, mode=mode, seed=42, device="cpu")
        std_stats = evaluate_3d_lesion_resilience(std_damaged, n_episodes=n_episodes)
        results["standard"].append(std_stats)

        if sev == 0.0 or std_baseline_reward is None:
            std_baseline_reward = max(1.0, std_stats["mean_reward"])

        base_std = std_baseline_reward if std_baseline_reward is not None else max(1.0, std_stats["mean_reward"])
        std_retention = (std_stats["mean_reward"] / base_std) * 100.0
        results["retention_standard"].append(std_retention)

        print(
            f"  Standard Agent: Rew = {std_stats['mean_reward']:+6.1f} | "
            f"Speed = {std_stats['mean_forward_vel']:+4.2f} m/s | "
            f"Falls = {std_stats['fall_rate']*100:3.0f}% | "
            f"Retention = {std_retention:5.1f}%"
        )

        # 2. Evolved Model Evaluation
        evo_damaged = apply_damage(evolved_model, severity=sev, mode=mode, seed=42, device="cpu")
        evo_stats = evaluate_3d_lesion_resilience(evo_damaged, n_episodes=n_episodes)
        results["evolved"].append(evo_stats)

        if sev == 0.0 or evo_baseline_reward is None:
            evo_baseline_reward = max(1.0, evo_stats["mean_reward"])

        base_evo = evo_baseline_reward if evo_baseline_reward is not None else max(1.0, evo_stats["mean_reward"])
        evo_retention = (evo_stats["mean_reward"] / base_evo) * 100.0
        results["retention_evolved"].append(evo_retention)

        print(
            f"  Evolved  Agent: Rew = {evo_stats['mean_reward']:+6.1f} | "
            f"Speed = {evo_stats['mean_forward_vel']:+4.2f} m/s | "
            f"Falls = {evo_stats['fall_rate']*100:3.0f}% | "
            f"Retention = {evo_retention:5.1f}%"
        )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Benchmark metrics stored to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Evolutionary Resilience Benchmark")
    parser.add_argument("--standard", default=WALKER_3D_CHECKPOINT)
    parser.add_argument("--evolved", default=os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"))
    parser.add_argument("--output", default=os.path.join(LOG_DIR, "evolution_3d_benchmark.json"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--mode", default="neuron_kill")
    args = parser.parse_args()

    run_evolution_benchmark(
        standard_model_path=args.standard,
        evolved_model_path=args.evolved,
        n_episodes=args.episodes,
        output_json=args.output,
        mode=args.mode,
    )
