"""
3D Bipedal Robot Mech — Evolutionary Resilience & Neuro-Trauma Curriculum Pipeline.
Evolves distributed, redundant synaptic representations resilient to catastrophic neural trauma.

Curriculum:
  - Gen 1: Micro-Focal Ablation (10% Neuron Kill + Exploratory Adaptation)
  - Gen 2: Diffuse Synaptic Zeroing (15% Weight Zero to enforce redundancy)
  - Gen 3: Stochastic Neuro-Perturbation (20% Gaussian Synaptic Noise Injection)
  - Gen 4: Evolutionary Consolidation & Synaptic Hardening (High-stability asymptotic gait)

Author: Geo Mathew Joseph
"""
import os
import time
import json
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from config import (
    CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, SEED, DEVICE,
    WALKER_3D_CHECKPOINT, WALKER_3D_N_ENVS,
)
from damage import apply_damage_inplace
from walker3d_env import Walker3DEnv
from train_3d import linear_schedule, evaluate_3d_policy
from visual_training_callback import VisualTrainingTelemetryCallback


def make_walker3d_train_env(rank: int, seed: int = 42):
    """Factory creating randomized Walker3D training environments for SubprocVecEnv."""
    def _init():
        env = Walker3DEnv(training_mode=True)
        env.reset(seed=seed + rank * 100)
        return env
    return _init


def train_evolved_3d(
    resume_path: str = WALKER_3D_CHECKPOINT,
    output_path: str = os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"),
    log_path: str = os.path.join(LOG_DIR, "evolved_3d_training.csv"),
    summary_path: str = os.path.join(LOG_DIR, "evolution_3d_history.json"),
    n_envs: int = WALKER_3D_N_ENVS,
    steps_per_gen: int = 60_000,
    seed: int = SEED,
):
    print("=" * 75)
    print("   AEGIS 3D BIPEDAL MECH — EVOLUTIONARY RESILIENCE TRAINING ENGINE")
    print("   Author: Geo Mathew Joseph")
    print(f"   Base Ancestor:  {resume_path}")
    print(f"   Champion Agent: {output_path}")
    print(f"   Parallel Envs:  {n_envs}")
    print(f"   Device:         {DEVICE}")
    print("=" * 75)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    if os.path.exists(log_path):
        os.remove(log_path)

    # 1. Initialize Vectorized Environments
    print(f"\n[INIT] Spawning {n_envs} parallel 3D physics environments...")
    env_fns = [make_walker3d_train_env(i, seed) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # 2. Load Ancestor Baseline Model
    print(f"[INIT] Loading ancestor healthy baseline: {resume_path}...")
    model = PPO.load(resume_path, env=vec_env, device=DEVICE)

    # Evaluate ancestor baseline performance
    ancestor_stats = evaluate_3d_policy(model, n_episodes=4)
    print(f"       Ancestor Baseline Reward: {ancestor_stats['mean_reward']:+.1f} | Speed: {ancestor_stats['mean_forward_vel']:+.2f} m/s\n")

    # 3. Define 4-Generation Evolutionary Curriculum
    generations = [
        {
            "gen": 1,
            "name": "Gen 1: Micro-Focal Ablation (10% Neuron Kill)",
            "severity": 0.10,
            "mode": "neuron_kill",
            "steps": steps_per_gen,
            "lr": 2.5e-4,
        },
        {
            "gen": 2,
            "name": "Gen 2: Diffuse Synaptic Zeroing (15% Weight Zero)",
            "severity": 0.15,
            "mode": "weight_zero",
            "steps": steps_per_gen,
            "lr": 2.0e-4,
        },
        {
            "gen": 3,
            "name": "Gen 3: Stochastic Synaptic Noise (20% Gaussian Perturbation)",
            "severity": 0.20,
            "mode": "noise_injection",
            "steps": steps_per_gen,
            "lr": 1.5e-4,
        },
        {
            "gen": 4,
            "name": "Gen 4: Evolutionary Consolidation & Synaptic Hardening",
            "severity": 0.0,
            "mode": "none",
            "steps": steps_per_gen,
            "lr": 8.0e-5,
        },
    ]

    total_training_steps = sum(g["steps"] for g in generations)
    total_start = time.time()
    evolution_history = []

    for g in generations:
        gen_idx = g["gen"]
        gen_name = g["name"]
        sev = g["severity"]
        mode = g["mode"]
        steps = g["steps"]
        gen_lr = g["lr"]

        print("-" * 75)
        print(f">>> [EVOLUTIONARY GENERATION {gen_idx}/4] {gen_name}")
        print(f"    Target Steps: {steps:,} | Mode: {mode} | Severity: {sev:.2f} | Base LR: {gen_lr}")

        # Apply evolutionary neural trauma
        if sev > 0.0:
            print(f"    [TRAUMA] Injecting in-place neural lesion ({sev*100:.0f}% {mode})...")
            apply_damage_inplace(model, severity=sev, mode=mode, seed=seed + gen_idx * 10)
            shock_stats = evaluate_3d_policy(model, n_episodes=3)
            post_shock_reward = shock_stats["mean_reward"]
            post_shock_speed = shock_stats["mean_forward_vel"]
            print(f"    [POST-SHOCK] Immediate Reward: {post_shock_reward:+.1f} | Speed: {post_shock_speed:+.2f} m/s")
        else:
            print("    [CONSOLIDATION] Zero lesion. Hardening synaptic weights and asymptotic limit cycles...")
            shock_stats = evaluate_3d_policy(model, n_episodes=3)
            post_shock_reward = shock_stats["mean_reward"]
            post_shock_speed = shock_stats["mean_forward_vel"]

        # Configure dynamic learning rate annealing for this generation
        model.learning_rate = linear_schedule(initial_lr=gen_lr, final_lr=gen_lr * 0.2)

        # Telemetry & visual limit-cycle callback
        telemetry_cb = VisualTrainingTelemetryCallback(
            log_path=log_path,
            total_timesteps=steps,
            generation_name=gen_name,
            gen_idx=gen_idx,
            total_gens=4,
            portrait_freq=max(10_000, steps // 3),
        )

        gen_start = time.time()
        model.learn(
            total_timesteps=steps,
            callback=[telemetry_cb],
            reset_num_timesteps=False,
            progress_bar=False,
        )
        gen_elapsed = time.time() - gen_start
        gen_fps = steps / max(0.01, gen_elapsed)

        # Evaluate recovered champion after generation
        eval_stats = evaluate_3d_policy(model, n_episodes=4)
        recovered_reward = eval_stats["mean_reward"]
        recovered_speed = eval_stats["mean_forward_vel"]

        print(f"\n    [EVOLVED] Gen {gen_idx} Completed in {gen_elapsed:.1f}s ({gen_fps:.0f} FPS)")
        print(f"    [METRIC] Post-Gen {gen_idx} Reward: {recovered_reward:+.1f} | Speed: {recovered_speed:+.2f} m/s")

        # Save inter-generation checkpoint to prevent progress loss on crash
        gen_checkpoint = os.path.join(CHECKPOINT_DIR, f"walker3d_evolved_gen{gen_idx}.zip")
        model.save(gen_checkpoint)
        print(f"    [CHECKPOINT] Saved gen {gen_idx} checkpoint: {gen_checkpoint}")

        evolution_history.append({
            "generation": gen_idx,
            "name": gen_name,
            "severity": sev,
            "mode": mode,
            "steps": steps,
            "post_shock_reward": post_shock_reward,
            "post_shock_speed": post_shock_speed,
            "recovered_reward": recovered_reward,
            "recovered_speed": recovered_speed,
            "elapsed_seconds": gen_elapsed,
            "fps": gen_fps,
        })

    total_elapsed = time.time() - total_start
    overall_fps = total_training_steps / max(0.01, total_elapsed)

    # Save Champion Evolved Model
    model.save(output_path)

    # Save Evolution Log Summary
    summary_data = {
        "ancestor_baseline": ancestor_stats,
        "total_timesteps": total_training_steps,
        "total_elapsed_seconds": total_elapsed,
        "average_fps": overall_fps,
        "final_evolved_stats": evolution_history[-1],
        "generations": evolution_history,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    vec_env.close()

    print("\n" + "=" * 75)
    print("   EVOLUTIONARY RESILIENCE TRAINING COMPLETE")
    print(f"   Model Saved To:    {output_path}")
    print(f"   Summary Log:       {summary_path}")
    print(f"   Total Timesteps:   {total_training_steps:,}")
    print(f"   Total Time:        {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print(f"   Average FPS:       {overall_fps:.0f} steps/second")
    print(f"   Final Reward:      {evolution_history[-1]['recovered_reward']:+.1f}")
    print(f"   Final Speed:       {evolution_history[-1]['recovered_speed']:+.2f} m/s")
    print("=" * 75)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Walker Evolutionary Resilience Training")
    parser.add_argument("--resume", default=WALKER_3D_CHECKPOINT)
    parser.add_argument("--output", default=os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"))
    parser.add_argument("--log", default=os.path.join(LOG_DIR, "evolved_3d_training.csv"))
    parser.add_argument("--summary", default=os.path.join(LOG_DIR, "evolution_3d_history.json"))
    parser.add_argument("--steps-per-gen", type=int, default=60_000)
    parser.add_argument("--n-envs", type=int, default=WALKER_3D_N_ENVS)
    args = parser.parse_args()

    train_evolved_3d(
        resume_path=args.resume,
        output_path=args.output,
        log_path=args.log,
        summary_path=args.summary,
        steps_per_gen=args.steps_per_gen,
        n_envs=args.n_envs,
    )
