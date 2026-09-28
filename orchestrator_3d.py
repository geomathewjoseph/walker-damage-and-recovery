"""
3D Walker Experiment Orchestrator — Automates 3D Damage and Recovery Experiments.
Tests neural lesion damage and subsequent neuroplastic recovery on the 3D bipedal mech.

Steps for each severity level:
  1. Evaluate healthy 3D baseline
  2. Inject neural lesion into the 3D policy (neuron_kill / weight_zero / noise)
  3. Evaluate damaged 3D policy and record HD telemetry video
  4. Train neuroplastic recovery on damaged model
  5. Evaluate recovered 3D policy and record HD telemetry video
  6. Save complete metrics to logs/experiment_3d_results.json

Author: Geo Mathew Joseph
"""
import os
import json
import time
import argparse
import numpy as np

from stable_baselines3 import PPO

from config import (
    CHECKPOINT_DIR, VIDEO_DIR, LOG_DIR,
    WALKER_3D_CHECKPOINT, WALKER_3D_N_ENVS,
    WALKER_3D_RECOVERY_TIMESTEPS, DEVICE,
)
from damage import apply_damage
from train_3d import train_3d, evaluate_3d_policy
from walker3d_env import Walker3DEnv


def record_3d_episode_video(
    model_or_path,
    output_path: str,
    status_label: str = None,
    status_type: str = "healthy",
    n_frames: int = 400,
    camera_name: str = "cinematic_3q",
):
    """
    Renders an episode of the 3D bipedal mech executing the REAL trained RL policy,
    saving a 720p HD MP4 video with an authentic telemetry HUD.
    """
    import imageio
    from mujoco_walker3d import draw_telemetry_hud

    if isinstance(model_or_path, str):
        model = PPO.load(model_or_path, device="cpu")
    else:
        model = model_or_path

    env = Walker3DEnv(render_mode="rgb_array", camera_name=camera_name)
    obs, _ = env.reset()

    frames = []
    total_reward = 0.0
    fps = 50

    print(f"  [Render] Generating 3D video with real policy inference -> {output_path}...")

    for step in range(n_frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew

        raw_frame = env.render()

        # Pass the full 36-dim observation directly to the HUD renderer
        # (draw_telemetry_hud already handles both 36-dim and 14-dim layouts)
        action_telemetry = action

        # Render 720p HD HUD
        hud_frame = draw_telemetry_hud(
            raw_frame, step, n_frames, total_reward,
            obs, action_telemetry,
            status_label=status_label,
            status_type=status_type,
        )
        frames.append(hud_frame)

        if term:
            # If the robot falls, re-initialize to continue recording the rest of the clip
            obs, _ = env.reset()

    env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, format="FFMPEG", codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"  [Saved] Video saved: {output_path} ({len(frames)} frames)")
    return total_reward


def run_3d_experiment(
    healthy_path: str = WALKER_3D_CHECKPOINT,
    severities: list = None,
    recovery_steps: int = WALKER_3D_RECOVERY_TIMESTEPS,
    damage_mode: str = "neuron_kill",
    seed: int = 42,
    device: str = DEVICE,
):
    """
    Executes the full 3D Damage & Recovery matrix across severities.
    """
    if severities is None:
        severities = [0.0, 0.1, 0.3, 0.5]

    print("=" * 70)
    print("   3D BIPEDAL MECH DAMAGE & RECOVERY EXPERIMENT MATRIX")
    print(f"   Healthy Base:    {healthy_path}")
    print(f"   Severities:      {severities}")
    print(f"   Damage Mode:     {damage_mode}")
    print(f"   Recovery Steps:  {recovery_steps:,}")
    print(f"   Device:          {device}")
    print(f"   Seed:            {seed}")
    print("=" * 70)

    if not os.path.exists(healthy_path):
        raise FileNotFoundError(
            f"Healthy 3D checkpoint not found: {healthy_path}\n"
            f"Please run 'python train_3d.py' first."
        )

    # 1. Baseline Evaluation
    print("\n[STEP 1] Evaluating Healthy 3D Baseline...")
    healthy_model = PPO.load(healthy_path, device="cpu")
    healthy_eval = evaluate_3d_policy(healthy_model, n_episodes=5)
    print(f"  Healthy Reward: {healthy_eval['mean_reward']:+.1f} | Speed: {healthy_eval['mean_forward_vel']:+.2f} m/s")

    # Record healthy baseline video
    healthy_video = os.path.join(VIDEO_DIR, "walker3d_healthy.mp4")
    record_3d_episode_video(
        healthy_model, healthy_video,
        status_label="[3D MODEL: HEALTHY ASYMPTOTIC / 100% INTEGRITY]",
        status_type="healthy",
    )

    results = []

    for sev in severities:
        print("\n" + "-" * 70)
        print(f">>> [3D EXPERIMENT SEVERITY {sev:.1f}] Mode: {damage_mode}")
        print("-" * 70)

        damaged_path = os.path.join(CHECKPOINT_DIR, f"walker3d_damaged_{sev:.1f}.zip")
        recovered_path = os.path.join(CHECKPOINT_DIR, f"walker3d_recovered_{sev:.1f}.zip")
        damaged_video = os.path.join(VIDEO_DIR, f"walker3d_damaged_{sev:.1f}.mp4")
        recovered_video = os.path.join(VIDEO_DIR, f"walker3d_recovered_{sev:.1f}.mp4")
        recovery_log = os.path.join(LOG_DIR, f"recovery_3d_{sev:.1f}.csv")

        # ── 1. Apply Damage ──
        if sev == 0.0:
            print("  Severity 0.0: Skipping damage injection...")
            healthy_model.save(damaged_path)
            healthy_model.save(recovered_path)
            damaged_eval = healthy_eval
            recovered_eval = healthy_eval
        else:
            print(f"  Injecting {sev*100:.0f}% {damage_mode} into 3D policy...")
            damaged_model = apply_damage(healthy_model, severity=sev, mode=damage_mode, seed=seed)
            damaged_model.save(damaged_path)

            damaged_eval = evaluate_3d_policy(damaged_model, n_episodes=5)
            print(f"  Post-Damage Reward: {damaged_eval['mean_reward']:+.1f} | Speed: {damaged_eval['mean_forward_vel']:+.2f} m/s")

            # Record damaged video
            record_3d_episode_video(
                damaged_model, damaged_video,
                status_label=f"[3D ACUTE LESION: {int(sev*100)}% {damage_mode.upper()}]",
                status_type="damaged",
            )

            # ── 2. Neuroplastic Recovery Training ──
            print(f"\n  Starting 3D Recovery Retraining ({recovery_steps:,} steps)...")
            recovered_model = train_3d(
                total_steps=recovery_steps,
                resume_path=damaged_path,
                log_path=recovery_log,
                save_as=recovered_path,
                n_envs=WALKER_3D_N_ENVS,
                seed=seed,
                device=device,
            )

            recovered_eval = evaluate_3d_policy(recovered_model, n_episodes=5)
            print(f"  Post-Recovery Reward: {recovered_eval['mean_reward']:+.1f} | Speed: {recovered_eval['mean_forward_vel']:+.2f} m/s")

            # Record recovered video
            record_3d_episode_video(
                recovered_model, recovered_video,
                status_label=f"[3D NEURO-RECOVERY: {int(sev*100)}% REHABILITATED]",
                status_type="recovered",
            )

        recovery_pct = (
            np.clip((recovered_eval["mean_reward"] / healthy_eval["mean_reward"] * 100), 0.0, 100.0)
            if healthy_eval["mean_reward"] > 0 else 0.0
        )

        results.append({
            "severity": float(sev),
            "damage_mode": damage_mode,
            "healthy_reward": float(healthy_eval["mean_reward"]),
            "damaged_reward": float(damaged_eval["mean_reward"]),
            "recovered_reward": float(recovered_eval["mean_reward"]),
            "healthy_speed": float(healthy_eval["mean_forward_vel"]),
            "damaged_speed": float(damaged_eval["mean_forward_vel"]),
            "recovered_speed": float(recovered_eval["mean_forward_vel"]),
            "recovery_percentage": float(recovery_pct),
        })

    # Save summary JSON
    results_path = os.path.join(LOG_DIR, "experiment_3d_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n\n" + "=" * 75)
    print(f"{'3D WALKER DAMAGE & RECOVERY MATRIX SUMMARY':^75}")
    print("=" * 75)
    print(f"{'Severity':>10} {'Healthy':>10} {'Damaged':>10} {'Recovered':>10} {'Recovery%':>12} {'Rec Speed':>12}")
    print("-" * 75)
    for r in results:
        print(f"{r['severity']:>10.1f} {r['healthy_reward']:>10.1f} {r['damaged_reward']:>10.1f} {r['recovered_reward']:>10.1f} {r['recovery_percentage']:>11.1f}% {r['recovered_speed']:>10.2f} m/s")
    print("=" * 75)
    print(f"Results successfully saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="3D Walker Damage and Recovery Experiment Orchestrator")
    parser.add_argument("--healthy", default=WALKER_3D_CHECKPOINT, help="Path to healthy 3D checkpoint")
    parser.add_argument("--severities", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--recovery-steps", type=int, default=150_000)
    parser.add_argument("--mode", default="neuron_kill", choices=["neuron_kill", "weight_zero", "noise_injection"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    run_3d_experiment(
        healthy_path=args.healthy,
        severities=args.severities,
        recovery_steps=args.recovery_steps,
        damage_mode=args.mode,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
