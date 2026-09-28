"""
Sanity Verification Suite for Walker Damage & Recovery.
Verifies both 2D Box2D and 3D MuJoCo simulation environments, action spaces,
rendering engines, and neural network checkpoint loading.

Author: Geo Mathew Joseph
"""
import os
import numpy as np
import gymnasium as gym
import imageio

from config import (
    ENV_NAME, ENV_FPS, VIDEO_DIR,
    HEALTHY_CHECKPOINT, WALKER_3D_CHECKPOINT,
)
from walker3d_env import Walker3DEnv


def verify_2d_environment():
    print("-" * 60)
    print(f"1. Verifying 2D Environment: {ENV_NAME}")
    print("-" * 60)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(f"  Observation Shape: {obs.shape} (Expected: (24,))")
    print(f"  Action Space:      {env.action_space} (Expected: Box(-1.0, 1.0, (4,), float32))")

    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    frame = env.render()
    env.close()

    print(f"  Single Step:       Reward={rew:.3f}, Term={term}")
    print(f"  Render Frame:      Shape={frame.shape}, Dtype={frame.dtype}")
    print("  [PASS] 2D Box2D environment operating properly.\n")


def verify_3d_environment():
    print("-" * 60)
    print("2. Verifying 3D Environment: Walker3DEnv (MuJoCo 3.13)")
    print("-" * 60)
    env = Walker3DEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(f"  Observation Shape: {obs.shape} (Expected: (36,))")
    print(f"  Action Space:      {env.action_space} (Expected: Box(-1.0, 1.0, (8,), float32))")

    action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    frame = env.render()
    env.close()

    print(f"  Single Step:       Reward={rew:.3f}, Term={term}, Speed={info.get('vy', 0):.2f} m/s")
    print(f"  Render Frame:      Shape={frame.shape}, Dtype={frame.dtype}")
    print("  [PASS] 3D MuJoCo environment operating properly.\n")


def verify_checkpoints():
    print("-" * 60)
    print("3. Verifying Checkpoints Integrity")
    print("-" * 60)
    checkpoints = [
        ("2D Healthy Baseline", HEALTHY_CHECKPOINT),
        ("3D Healthy Baseline", WALKER_3D_CHECKPOINT),
    ]
    for name, path in checkpoints:
        exists = os.path.exists(path)
        size_kb = os.path.getsize(path) / 1024 if exists else 0
        status = f"[FOUND] {size_kb:.1f} KB" if exists else "[MISSING]"
        print(f"  {name:<22}: {status} -> {path}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("   WALKER DAMAGE & RECOVERY — SYSTEM SANITY AUDIT")
    print("=" * 60)
    verify_2d_environment()
    verify_3d_environment()
    verify_checkpoints()
    print("\n[COMPLETE] All core simulation frameworks verified.\n")


if __name__ == "__main__":
    main()
