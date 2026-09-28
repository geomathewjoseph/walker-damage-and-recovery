"""
PPO training script with resume support, CSV logging, and periodic checkpointing.

Usage:
    # Fresh training
    python train.py --total-steps 2000000

    # Throughput smoke test
    python train.py --total-steps 100000

    # Resume from checkpoint (used for recovery)
    python train.py --resume checkpoints/damaged_0.3.zip --total-steps 500000 --log logs/recovery_0.3.csv

    # Damage-resistant training variant (stretch goal)
    python train.py --total-steps 2000000 --damage-resistant --log logs/damage_resistant_train.csv
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    ENV_NAME, N_ENVS, POLICY_KWARGS, PPO_PARAMS,
    TOTAL_TIMESTEPS, CHECKPOINT_FREQ, SEED, DEVICE,
    CHECKPOINT_DIR, LOG_DIR, HEALTHY_CHECKPOINT,
)


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Logs episode rewards to a CSV file.
    Reads from the VecEnv monitor to capture completed episode rewards.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None
        self._episode_count = 0
        self._last_report_step = 0

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Write header only if file is empty
        if os.path.getsize(self.log_path) == 0:
            self.csv_writer.writerow(["timestep", "episode", "reward", "length"])

    def _on_step(self) -> bool:
        # SB3 stores completed episode info in the info dict
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

        # Periodic stdout report
        if self.num_timesteps - self._last_report_step >= 10_000:
            self._last_report_step = self.num_timesteps
            if self._episode_count > 0:
                # Read recent rewards from the log
                pass  # SB3 already prints FPS and loss info
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()
            print(f"Reward log saved to {self.log_path} ({self._episode_count} episodes)")


class CheckpointCallback(BaseCallback):
    """Saves the model every `save_freq` timesteps."""
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            path = os.path.join(self.save_dir, f"step_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  Checkpoint saved: {path}.zip")
        return True


class DamageInjectionCallback(BaseCallback):
    """
    (Stretch goal) Periodically applies random damage during training
    to build damage-resistant representations.

    Fires on rollout boundaries (_on_rollout_start), NOT mid-rollout,
    to keep each PPO batch internally consistent.
    """
    def __init__(self, damage_interval: int = 50_000, max_severity: float = 0.3,
                 verbose: int = 0):
        super().__init__(verbose)
        self.damage_interval = damage_interval
        self.max_severity = max_severity
        self._last_damage_step = 0

    def _on_rollout_start(self):
        elapsed = self.num_timesteps - self._last_damage_step
        if elapsed >= self.damage_interval and self.num_timesteps > 0:
            from damage import apply_damage_inplace
            severity = random.uniform(0.05, self.max_severity)
            print(f"  [DamageInjection] Applying neuron_kill at severity={severity:.3f} "
                  f"(step {self.num_timesteps})")
            apply_damage_inplace(self.model, severity, mode="neuron_kill")
            self._last_damage_step = self.num_timesteps

    def _on_step(self) -> bool:
        return True


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train(
    total_steps: int,
    resume_path: str = None,
    log_path: str = None,
    checkpoint_dir: str = None,
    damage_resistant: bool = False,
    n_envs: int = None,
    seed: int = None,
    reset_num_timesteps: bool = True,
) -> PPO:
    """
    Train or resume PPO on BipedalWalker.

    Args:
        total_steps: Total timesteps to train for.
        resume_path: Path to checkpoint to resume from (None = train from scratch).
        log_path: Path for CSV reward log.
        checkpoint_dir: Directory for periodic checkpoints.
        damage_resistant: If True, apply periodic damage during training.
        n_envs: Number of parallel environments.
        seed: Random seed.

    Returns:
        Trained PPO model.
    """
    if n_envs is None:
        n_envs = N_ENVS
    if seed is None:
        seed = SEED
    if log_path is None:
        log_path = os.path.join(LOG_DIR, "train.csv")
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR

    print(f"{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"  Environment:     {ENV_NAME}")
    print(f"  Parallel envs:   {n_envs}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Resume from:     {resume_path or 'scratch'}")
    print(f"  Log path:        {log_path}")
    print(f"  Damage-resistant: {damage_resistant}")
    print(f"  Seed:            {seed}")
    print(f"{'='*60}")

    # Create vectorized environment
    env = make_vec_env(
        ENV_NAME,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    # Build callbacks
    callbacks = [
        RewardLoggerCallback(log_path),
        CheckpointCallback(CHECKPOINT_FREQ, checkpoint_dir, verbose=1),
    ]
    if damage_resistant:
        callbacks.append(DamageInjectionCallback(
            damage_interval=50_000,
            max_severity=0.3,
            verbose=1,
        ))

    if resume_path:
        # Resume from checkpoint
        print(f"Loading checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=env, device=DEVICE)
        print(f"Resuming training for {total_steps:,} additional steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )
    else:
        # Fresh training
        print("Initializing fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=POLICY_KWARGS,
            seed=seed,
            verbose=1,
            device=DEVICE,
            **PPO_PARAMS,
        )
        print(f"Policy architecture:")
        print(f"  {model.policy.mlp_extractor.policy_net}")
        print(f"  {model.policy.action_net}")
        print(f"\nStarting training for {total_steps:,} steps...")

        start_time = time.time()
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            progress_bar=True,
        )
        elapsed = time.time() - start_time
        throughput = total_steps / elapsed
        print(f"\nTraining complete in {elapsed:.1f}s ({throughput:.0f} steps/sec)")

    env.close()
    return model


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train PPO on BipedalWalker")
    parser.add_argument("--total-steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--log", default=None, help="CSV reward log path")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--save-as", default=None,
                        help="Final model save path (default: checkpoints/healthy.zip)")
    parser.add_argument("--damage-resistant", action="store_true",
                        help="Apply periodic damage during training (stretch goal)")
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    model = train(
        total_steps=args.total_steps,
        resume_path=args.resume,
        log_path=args.log or os.path.join(LOG_DIR, "train.csv"),
        checkpoint_dir=args.checkpoint_dir,
        damage_resistant=args.damage_resistant,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    # Save final model
    save_path = args.save_as or HEALTHY_CHECKPOINT
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nFinal model saved to {save_path}")


if __name__ == "__main__":
    main()
