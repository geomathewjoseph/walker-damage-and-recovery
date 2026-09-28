"""
Reinforcement Learning Training Pipeline for the 3D Bipedal Robot Mech.
Trains PPO on Walker3DEnv (MuJoCo 3.13) obeying physical laws and robotics dynamics.

Supports:
  - Fresh training from scratch
  - Parallel vectorized environment rollouts (SubprocVecEnv)
  - Resumption from checkpoint (used for neuroplastic recovery retraining)
  - High-frequency CSV episode logging and periodic checkpointing
  - Model evaluation with real biomechanical metrics

Author: Geo Mathew Joseph
"""
import os
import csv
import time
import atexit
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    CHECKPOINT_DIR, LOG_DIR, SEED, DEVICE,
    WALKER_3D_TOTAL_TIMESTEPS, WALKER_3D_CHECKPOINT,
    WALKER_3D_LOG, WALKER_3D_N_ENVS,
    WALKER_3D_POLICY_KWARGS, WALKER_3D_PPO_PARAMS,
)
from walker3d_env import Walker3DEnv


def linear_schedule(initial_lr: float = 3e-4, final_lr: float = 3e-5):
    """Linear learning rate annealing schedule for stable convergence across deep horizons."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ──────────────────────────────────────────────
# Logging & Checkpointing Callbacks
# ──────────────────────────────────────────────

class RewardLoggerCallback3D(BaseCallback):
    """Logs episode rewards, lengths, and timesteps to a CSV file."""
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None
        self._episode_count = 0

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        is_empty = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if is_empty:
            self.csv_writer.writerow(["timestep", "episode", "reward", "length"])
        # Register cleanup in case training crashes without calling _on_training_end
        atexit.register(self._close_csv)

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

    def _close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def _on_training_end(self):
        self._close_csv()
        print(f"Reward log saved to {self.log_path} ({self._episode_count} episodes)")


class CheckpointCallback3D(BaseCallback):
    """Saves periodic checkpoints every save_freq steps."""
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq < self.training_env.num_envs:
            path = os.path.join(self.save_dir, f"walker3d_step_{self.num_timesteps}")
            self.model.save(path)
            if self.verbose:
                print(f"  [Checkpoint] Saved 3D model checkpoint: {path}.zip")
        return True


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_3d_policy(model: PPO, n_episodes: int = 5, deterministic: bool = True):
    """Evaluates the 3D walker model and returns mean reward, mean length, and forward velocity."""
    env = Walker3DEnv(training_mode=False)
    rewards = []
    lengths = []
    forward_vels = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0
        ep_vy = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, term, trunc, info = env.step(action)
            ep_r += rew
            ep_len += 1
            ep_vy.append(info.get("vy", 0.0))
            done = term or trunc

        rewards.append(ep_r)
        lengths.append(ep_len)
        forward_vels.append(np.mean(ep_vy) if ep_vy else 0.0)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_forward_vel": float(np.mean(forward_vels)),
    }


# ──────────────────────────────────────────────
# Main Training Function
# ──────────────────────────────────────────────

def train_3d(
    total_steps: int = WALKER_3D_TOTAL_TIMESTEPS,
    resume_path: str = None,
    log_path: str = WALKER_3D_LOG,
    checkpoint_dir: str = CHECKPOINT_DIR,
    save_as: str = WALKER_3D_CHECKPOINT,
    n_envs: int = WALKER_3D_N_ENVS,
    seed: int = SEED,
    save_freq: int = 100_000,
    device: str = DEVICE,
    normalize_obs: bool = False,
) -> PPO:
    """
    Trains or resumes PPO for the 3D Bipedal Mech environment with GPU acceleration.
    """
    print("=" * 70)
    print("   3D BIPEDAL MECH REINFORCEMENT LEARNING PIPELINE")
    print("   Environment:      Walker3DEnv (MuJoCo 3.13 3D Physics)")
    print(f"   Target Steps:     {total_steps:,}")
    print(f"   Resume From:      {resume_path or 'Scratch (Random Initialization)'}")
    print(f"   Parallel Envs:    {n_envs}")
    print(f"   Log File:         {log_path}")
    print(f"   Final Checkpoint: {save_as}")
    print(f"   Compute Device:   {device}")
    print("=" * 70)

    # Build parallel vectorized environments
    env = make_vec_env(
        Walker3DEnv,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
    )

    if normalize_obs:
        print("   [NORMALIZATION] VecNormalize active (norm_obs=True, norm_reward=False, clip=10.0)")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    callbacks = [
        RewardLoggerCallback3D(log_path),
        CheckpointCallback3D(save_freq, checkpoint_dir, verbose=1),
    ]

    policy_kwargs = WALKER_3D_POLICY_KWARGS
    ppo_params = dict(WALKER_3D_PPO_PARAMS)
    base_lr = ppo_params.pop("learning_rate", 3e-4)
    ppo_params["learning_rate"] = linear_schedule(initial_lr=base_lr, final_lr=base_lr * 0.1)

    if resume_path and os.path.exists(resume_path):
        print(f"\n[RESUME] Loading existing checkpoint: {resume_path}...")
        model = PPO.load(resume_path, env=env, device=device)
        # Apply fresh LR schedule to avoid stalled learning from decayed checkpoint LR
        model.learning_rate = linear_schedule(initial_lr=base_lr, final_lr=base_lr * 0.1)
        print(f"Resuming training for {total_steps:,} timesteps...")
        start_time = time.time()
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        elapsed = time.time() - start_time
    else:
        print("\n[INIT] Constructing fresh PPO Actor-Critic architecture...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=1,
            device=device,
            **ppo_params,
        )
        print(f"Policy Actor Network: {model.policy.mlp_extractor.policy_net}")
        print(f"Action Head:          {model.policy.action_net}")

        print(f"\nStarting fresh 3D PPO training for {total_steps:,} steps...")
        start_time = time.time()
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            progress_bar=True,
        )
        elapsed = time.time() - start_time

    throughput = total_steps / elapsed if elapsed > 0 else 0
    print(f"\n[DONE] Training complete in {elapsed:.1f}s ({elapsed/60:.2f} min). Throughput: {throughput:.0f} steps/s")

    # Save final model
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    print(f"\n[SAVE] Writing final trained weights -> {save_as}...")
    model.save(save_as)
    if normalize_obs and hasattr(env, "save"):
        vec_norm_path = save_as.replace(".zip", "_vecnormalize.pkl")
        env.save(vec_norm_path)
        print(f"[SAVE] VecNormalize statistics saved -> {vec_norm_path}")
    env.close()

    # Evaluate final policy
    eval_stats = evaluate_3d_policy(model, n_episodes=5)
    print("\n" + "=" * 70)
    print("   FINAL EVALUATION METRICS (5 Episodes)")
    print(f"   Mean Reward:        {eval_stats['mean_reward']:+.2f} +/- {eval_stats['std_reward']:.2f}")
    print(f"   Mean Episode Length:{eval_stats['mean_length']:.1f} steps")
    print(f"   Mean Forward Speed: {eval_stats['mean_forward_vel']:+.2f} m/s")
    print("=" * 70)

    return model


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 3D Bipedal Mech PPO Model")
    parser.add_argument("--total-steps", type=int, default=WALKER_3D_TOTAL_TIMESTEPS)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--log", default=WALKER_3D_LOG, help="Reward CSV log path")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--save-as", default=WALKER_3D_CHECKPOINT)
    parser.add_argument("--n-envs", type=int, default=WALKER_3D_N_ENVS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--normalize-obs", action="store_true", help="Wrap vectorized environment with VecNormalize")
    args = parser.parse_args()

    train_3d(
        total_steps=args.total_steps,
        resume_path=args.resume,
        log_path=args.log,
        checkpoint_dir=args.checkpoint_dir,
        save_as=args.save_as,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        normalize_obs=args.normalize_obs,
    )


if __name__ == "__main__":
    main()
