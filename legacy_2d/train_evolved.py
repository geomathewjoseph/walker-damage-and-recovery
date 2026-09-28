"""
Intensive Evolutionary Curriculum Training Pipeline
Trains a biologically-inspired damage-resistant walker agent using stochastic neuro-lesion curriculum.
Evolves distributed, redundant motor representations resilient to sudden neural trauma.

Author: Geo Mathew Joseph
"""
import os
import time
import json
import argparse
import csv
import torch
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    ENV_NAME, N_ENVS, POLICY_KWARGS, PPO_PARAMS,
    SEED, DEVICE, CHECKPOINT_DIR, LOG_DIR, HEALTHY_CHECKPOINT
)
from damage import apply_damage_inplace


class EvolutionaryRewardLoggerCallback(BaseCallback):
    """Logs episode rewards to CSV during evolutionary training."""
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None
        self._episode_count = 0

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.csv_writer.writerow(["timestep", "episode", "reward", "length"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.csv_writer.writerow([
                    self.num_timesteps,
                    self._episode_count,
                    f"{ep_reward:.2f}",
                    ep_length,
                ])
                self.csv_file.flush()
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()


def quick_eval(model: PPO, n_episodes: int = 5) -> float:
    """Evaluates the model on BipedalWalker-v3 and returns mean reward."""
    env = gym.make(ENV_NAME)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(action)
            r += rew
            done = term or trunc
        rewards.append(r)
    env.close()
    return float(np.mean(rewards))


def train_evolved(
    resume_path: str = HEALTHY_CHECKPOINT,
    output_path: str = os.path.join(CHECKPOINT_DIR, "evolved_resistant.zip"),
    log_path: str = os.path.join(LOG_DIR, "evolved_training.csv"),
    n_envs: int = N_ENVS,
    seed: int = SEED,
):
    print("=" * 70)
    print("   INTENSIVE BIOLOGICAL EVOLUTION & RESILIENCE TRAINING PIPELINE")
    print("   Author: Geo Mathew Joseph")
    print(f"   Base Agent:    {resume_path}")
    print(f"   Output Agent:  {output_path}")
    print(f"   Parallel Envs: {n_envs}")
    print(f"   Compute Device:{DEVICE}")
    print("=" * 70)

    # Clean existing log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)

    env = make_vec_env(
        ENV_NAME,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    print(f"\n[INIT] Loading healthy ancestor checkpoint: {resume_path}...")
    model = PPO.load(resume_path, env=env, device=DEVICE)
    baseline_score = quick_eval(model)
    print(f"       Ancestor Baseline Reward: {baseline_score:+.1f}\n")

    # Evolutionary Curriculum Plan:
    # 4 Generations of traumatic shock followed by deep neuroplastic adaptation
    generations = [
        {
            "gen": 1,
            "name": "Generation 1: Focal Axon Ablation (10% Neuron Kill)",
            "severity": 0.10,
            "mode": "neuron_kill",
            "steps": 75_000,
        },
        {
            "gen": 2,
            "name": "Generation 2: Diffuse Synaptic Zeroing (15% Weight Zero)",
            "severity": 0.15,
            "mode": "weight_zero",
            "steps": 75_000,
        },
        {
            "gen": 3,
            "name": "Generation 3: Stochastic Neuro-Perturbation (20% Gaussian Noise)",
            "severity": 0.20,
            "mode": "noise_injection",
            "steps": 90_000,
        },
        {
            "gen": 4,
            "name": "Generation 4: Evolutionary Consolidation & Gait Synthesis",
            "severity": 0.0,
            "mode": "none",
            "steps": 120_000,
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

        print("-" * 70)
        print(f">>> [EVOLUTIONARY GENERATION {gen_idx}/4] {gen_name}")
        print(f"    Target Steps: {steps:,} | Mode: {mode} | Severity: {sev:.2f}")

        # Apply evolutionary trauma before adaptation phase
        if sev > 0:
            print(f"    [TRAUMA] Applying in-place neural lesion ({sev*100:.0f}% {mode})...")
            apply_damage_inplace(model, severity=sev, mode=mode)
            post_shock_reward = quick_eval(model)
            print(f"    [POST-SHOCK] Immediate Reward: {post_shock_reward:+.1f}")
        else:
            print("    [CONSOLIDATION] Zero damage applied. Optimizing redundant synaptic pathways...")

        logger = EvolutionaryRewardLoggerCallback(log_path)
        gen_start = time.time()

        model.learn(
            total_timesteps=steps,
            callback=[logger],
            reset_num_timesteps=False,
            progress_bar=True,
        )

        gen_elapsed = time.time() - gen_start
        gen_fps = steps / gen_elapsed
        recovered_reward = quick_eval(model)

        print(f"    [EVOLVED] Gen {gen_idx} Complete in {gen_elapsed:.1f}s ({gen_fps:.0f} FPS)")
        print(f"    [METRIC] Post-Gen {gen_idx} Mean Reward: {recovered_reward:+.1f}")

        evolution_history.append({
            "generation": gen_idx,
            "name": gen_name,
            "severity": sev,
            "mode": mode,
            "steps": steps,
            "reward": recovered_reward,
            "elapsed_seconds": gen_elapsed,
        })

    total_elapsed = time.time() - total_start
    overall_fps = total_training_steps / total_elapsed

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)

    # Save evolution log summary
    evo_summary_path = os.path.join(LOG_DIR, "evolution_history.json")
    with open(evo_summary_path, "w") as f:
        json.dump(evolution_history, f, indent=2)

    print("\n" + "=" * 70)
    print("   EVOLUTIONARY CURRICULUM TRAINING COMPLETE")
    print(f"   Model Saved To:    {output_path}")
    print(f"   Summary Log:       {evo_summary_path}")
    print(f"   Total Timesteps:   {total_training_steps:,}")
    print(f"   Total Time:        {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print(f"   Average Throughput:{overall_fps:.0f} steps/second")
    print(f"   Final Evolved Reward: {evolution_history[-1]['reward']:+.1f}")
    print("=" * 70)

    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intensive Evolutionary Training")
    parser.add_argument("--resume", default=HEALTHY_CHECKPOINT)
    parser.add_argument("--output", default=os.path.join(CHECKPOINT_DIR, "evolved_resistant.zip"))
    parser.add_argument("--log", default=os.path.join(LOG_DIR, "evolved_training.csv"))
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    args = parser.parse_args()

    train_evolved(
        resume_path=args.resume,
        output_path=args.output,
        log_path=args.log,
        n_envs=args.n_envs,
    )
