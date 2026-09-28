"""
Experiment orchestrator — automates the full severity matrix.
Only run after Steps 2-4 (train, damage, recovery) work individually.

For each severity level:
  1. Apply damage to the healthy checkpoint
  2. Record video of damaged walker
  3. Run recovery training
  4. Record video of recovered walker
  5. Save reward logs for visualization
"""
import argparse
import os
import time

import numpy as np
import torch
import random

from stable_baselines3 import PPO

from config import (
    SEVERITIES, RECOVERY_TIMESTEPS, SEED,
    CHECKPOINT_DIR, VIDEO_DIR, LOG_DIR, HEALTHY_CHECKPOINT,
    N_ENVS, DEVICE,
)
from damage import apply_damage
from record_video import record_episode
from train import train


def run_experiment(
    severities: list = None,
    recovery_steps: int = None,
    healthy_path: str = None,
    seed: int = None,
):
    """
    Run the full damage/recovery experiment across severity levels.
    """
    if severities is None:
        severities = SEVERITIES
    if recovery_steps is None:
        recovery_steps = RECOVERY_TIMESTEPS
    if healthy_path is None:
        healthy_path = HEALTHY_CHECKPOINT
    if seed is None:
        seed = SEED

    # Set global seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"{'='*60}")
    print(f"Walker Damage & Recovery Experiment")
    print(f"{'='*60}")
    print(f"  Healthy checkpoint: {healthy_path}")
    print(f"  Severity levels:   {severities}")
    print(f"  Recovery steps:    {recovery_steps:,}")
    print(f"  Seed:              {seed}")
    print(f"{'='*60}\n")

    # Verify healthy checkpoint exists
    if not os.path.exists(healthy_path):
        raise FileNotFoundError(
            f"Healthy checkpoint not found: {healthy_path}\n"
            f"Run 'python train.py' first to produce it."
        )

    # Record healthy baseline video
    print("Recording healthy baseline video...")
    healthy_reward = record_episode(
        healthy_path,
        os.path.join(VIDEO_DIR, "healthy.mp4"),
        n_episodes=1,
    )
    print(f"Healthy baseline reward: {healthy_reward:.1f}\n")

    results = []

    for severity in severities:
        print(f"\n{'='*60}")
        print(f"  Severity: {severity}")
        print(f"{'='*60}")

        damaged_path = os.path.join(CHECKPOINT_DIR, f"damaged_{severity}.zip")
        recovered_path = os.path.join(CHECKPOINT_DIR, f"recovered_{severity}.zip")
        damaged_video = os.path.join(VIDEO_DIR, f"damaged_{severity}.mp4")
        recovered_video = os.path.join(VIDEO_DIR, f"recovered_{severity}.mp4")
        recovery_log = os.path.join(LOG_DIR, f"recovery_{severity}.csv")

        # ── 1. Apply damage ──
        if severity == 0.0:
            # No damage — just copy the healthy model
            print("Severity 0.0: skipping damage, copying healthy checkpoint")
            healthy_model = PPO.load(healthy_path, device=DEVICE)
            healthy_model.save(damaged_path)
        else:
            print(f"Applying neuron_kill damage at severity={severity}...")
            healthy_model = PPO.load(healthy_path, device=DEVICE)
            damaged_model = apply_damage(healthy_model, severity, mode="neuron_kill",
                                         seed=seed + int(severity * 1000))
            damaged_model.save(damaged_path)
            del damaged_model

        # ── 2. Record damaged video ──
        print(f"Recording damaged video...")
        damaged_reward = record_episode(damaged_path, damaged_video)
        print(f"Damaged reward: {damaged_reward:.1f}")

        # ── 3. Recovery training ──
        if severity == 0.0:
            # No recovery needed for undamaged model
            print("Severity 0.0: skipping recovery training")
            PPO.load(damaged_path, device=DEVICE).save(recovered_path)
            # Create a minimal log for the plot
            with open(recovery_log, "w") as f:
                f.write("timestep,episode,reward,length\n")
                f.write(f"0,1,{healthy_reward:.2f},1600\n")
        else:
            print(f"Starting recovery training for {recovery_steps:,} steps...")
            start = time.time()
            recovered = train(
                total_steps=recovery_steps,
                resume_path=damaged_path,
                log_path=recovery_log,
                checkpoint_dir=os.path.join(CHECKPOINT_DIR, f"recovery_{severity}"),
                n_envs=N_ENVS,
                seed=seed,
            )
            elapsed = time.time() - start
            recovered.save(recovered_path)
            print(f"Recovery training done in {elapsed:.1f}s")

        # ── 4. Record recovered video ──
        print(f"Recording recovered video...")
        recovered_reward = record_episode(recovered_path, recovered_video)
        print(f"Recovered reward: {recovered_reward:.1f}")

        results.append({
            "severity": float(severity),
            "healthy_reward": float(healthy_reward),
            "damaged_reward": float(damaged_reward),
            "recovered_reward": float(recovered_reward),
        })

        print(f"\n  Summary: healthy={healthy_reward:.1f} -> "
              f"damaged={damaged_reward:.1f} -> recovered={recovered_reward:.1f}")

    # Print final summary table
    print(f"\n\n{'='*60}")
    print(f"{'FINAL RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Severity':>10} {'Healthy':>10} {'Damaged':>10} {'Recovered':>10} {'Recovery%':>10}")
    print(f"{'-'*50}")
    for r in results:
        if r["healthy_reward"] != 0:
            recovery_pct = r["recovered_reward"] / r["healthy_reward"] * 100
        else:
            recovery_pct = 0
        print(f"{r['severity']:>10.1f} {r['healthy_reward']:>10.1f} "
              f"{r['damaged_reward']:>10.1f} {r['recovered_reward']:>10.1f} "
              f"{recovery_pct:>9.1f}%")

    # Save results to JSON
    import json
    results_path = os.path.join(LOG_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved experiment summary to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run full damage/recovery experiment")
    parser.add_argument("--severities", nargs="+", type=float, default=SEVERITIES,
                        help="Severity levels to test")
    parser.add_argument("--recovery-steps", type=int, default=RECOVERY_TIMESTEPS)
    parser.add_argument("--healthy", default=HEALTHY_CHECKPOINT,
                        help="Path to healthy checkpoint")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    run_experiment(
        severities=args.severities,
        recovery_steps=args.recovery_steps,
        healthy_path=args.healthy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
