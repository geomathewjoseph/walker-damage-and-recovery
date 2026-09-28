"""
Reusable video recording utility with Sci-Fi / Robotics Telemetry HUD.
Loads a checkpoint, runs episodes, renders telemetry overlay, saves MP4 video.
Author: Geo Mathew Joseph
"""
import argparse
import os
import cv2
import numpy as np
import imageio
import gymnasium as gym

from stable_baselines3 import PPO
from config import ENV_NAME, ENV_FPS, VIDEO_DIR, DEVICE


def draw_telemetry_hud(
    frame: np.ndarray,
    step: int,
    max_steps: int,
    reward: float,
    obs: np.ndarray,
    action: np.ndarray,
    status_label: str = None,
    status_type: str = "healthy",
) -> np.ndarray:
    """
    Renders a high-tech Sci-Fi Robotics Telemetry HUD onto the frame.
    Upscales to 1280x720 (720p HD) and adds real-time diagnostics.
    """
    h, w = 720, 1280
    frame_hd = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = frame_hd.copy()

    # 1. Top Glass Banner
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 30), -1)
    cv2.line(overlay, (0, 55), (w, 55), (0, 220, 255), 2)

    # 2. Right Telemetry Card
    cv2.rectangle(overlay, (970, 70), (1260, 220), (15, 20, 30), -1)
    cv2.rectangle(overlay, (970, 70), (1260, 220), (80, 90, 110), 1)

    # 3. Left Actuator Card
    cv2.rectangle(overlay, (20, 510), (350, 695), (15, 20, 30), -1)
    cv2.rectangle(overlay, (20, 510), (350, 695), (80, 90, 110), 1)

    # Alpha blend overlay with background
    cv2.addWeighted(overlay, 0.82, frame_hd, 0.18, 0, frame_hd)

    # Top Bar Titles
    cv2.putText(frame_hd, "WALKER NEURO-RECOVERY LAB", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_hd, "BIPEDAL KINEMATICS TELEMETRY", (360, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    # Status Badge
    if status_type == "healthy":
        badge_bg = (40, 140, 40)
        badge_border = (80, 255, 80)
        text = status_label or "[STATE: NORMAL / 100% HEALTHY]"
    elif status_type == "damaged":
        badge_bg = (30, 30, 150)
        badge_border = (60, 70, 255)
        text = status_label or "[ACUTE NEURAL LESION / IMPAIRED]"
    elif status_type == "recovered":
        badge_bg = (120, 100, 20)
        badge_border = (255, 220, 0)
        text = status_label or "[REHABILITATED / COMPENSATORY GAIT]"
    elif status_type == "evolved":
        badge_bg = (120, 20, 120)
        badge_border = (255, 80, 255)
        text = status_label or "[EVOLVED / NEURO-RESISTANT AGENT]"
    else:
        badge_bg = (60, 60, 70)
        badge_border = (160, 160, 170)
        text = status_label or "[SYSTEM EVALUATION]"

    badge_w = min(420, max(280, len(text) * 11 + 25))
    badge_x1 = w - badge_w - 20
    badge_x2 = w - 20
    cv2.rectangle(frame_hd, (badge_x1, 12), (badge_x2, 44), badge_bg, -1)
    cv2.rectangle(frame_hd, (badge_x1, 12), (badge_x2, 44), badge_border, 1)
    cv2.putText(frame_hd, text, (badge_x1 + 12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # Right Telemetry Text
    vx = obs[2] * 5.0 if len(obs) > 2 else 0.0
    tilt = obs[0] * 57.3 if len(obs) > 0 else 0.0
    reward_color = (50, 255, 100) if reward >= 0 else (50, 80, 255)

    cv2.putText(frame_hd, "SYSTEM TELEMETRY", (990, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"STEP:   {step:04d} / {max_steps}", (990, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"REWARD: {reward:+6.1f}", (990, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, reward_color, 2, cv2.LINE_AA)
    cv2.putText(frame_hd, f"SPEED:  {vx:+5.2f} m/s", (990, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"TILT:   {tilt:+5.1f} deg", (990, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Left Actuator Text & Dynamic Torque Gauges
    cv2.putText(frame_hd, "ACTUATOR JOINT TORQUES", (35, 535),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

    joint_names = ["HIP 1", "KNEE 1", "HIP 2", "KNEE 2"]
    for i, name in enumerate(joint_names):
        val = float(action[i]) if action is not None and i < len(action) else 0.0
        y = 565 + i * 25
        cv2.putText(frame_hd, name, (35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        # Bar track
        cv2.rectangle(frame_hd, (110, y - 10), (330, y + 2), (40, 45, 55), -1)
        # Center zero line
        cv2.line(frame_hd, (220, y - 12), (220, y + 4), (100, 110, 120), 1)
        # Dynamic deflection bar
        bar_w = int(np.clip(val, -1.0, 1.0) * 100)
        color = (255, 200, 0) if val >= 0 else (50, 80, 255)
        if bar_w >= 0:
            cv2.rectangle(frame_hd, (220, y - 9), (220 + bar_w, y + 1), color, -1)
        else:
            cv2.rectangle(frame_hd, (220 + bar_w, y - 9), (220, y + 1), color, -1)

    # Leg Contact Flags
    c1 = "[ON]" if (len(obs) > 8 and obs[8] > 0.5) else "[AIR]"
    c2 = "[ON]" if (len(obs) > 13 and obs[13] > 0.5) else "[AIR]"
    cv2.putText(frame_hd, f"CONTACT:  LEG 1 {c1}   LEG 2 {c2}", (35, 680),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 100), 1, cv2.LINE_AA)

    # Bottom Progress Bar
    progress = np.clip(step / float(max_steps), 0.0, 1.0)
    cv2.rectangle(frame_hd, (0, 712), (w, 720), (30, 35, 45), -1)
    cv2.rectangle(frame_hd, (0, 712), (int(w * progress), 720), (0, 220, 255), -1)

    return frame_hd


def record_episode(
    model_or_path,
    output_path: str,
    n_episodes: int = 1,
    deterministic: bool = True,
    max_steps: int = 1600,
    hud: bool = True,
    status_label: str = None,
    status_type: str = None,
) -> float:
    """
    Record video of a model playing in the environment with optional Sci-Fi HUD.
    """
    if isinstance(model_or_path, str):
        model = PPO.load(model_or_path, device=DEVICE)
    else:
        model = model_or_path

    # Infer status type from output path if not specified
    if status_type is None:
        low_path = output_path.lower()
        if "damaged" in low_path:
            status_type = "damaged"
            if status_label is None:
                status_label = "[ACUTE NEURAL LESION / DAMAGED]"
        elif "recovered" in low_path:
            status_type = "recovered"
            if status_label is None:
                status_label = "[REHABILITATED / COMPENSATORY GAIT]"
        elif "healthy" in low_path:
            status_type = "healthy"
            if status_label is None:
                status_label = "[STATE: NORMAL / 100% HEALTHY]"
        else:
            status_type = "info"
            status_label = "[SYSTEM EVALUATION]"

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    all_frames = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        frames = []
        total_reward = 0.0
        done = False
        step_count = 0
        current_action = np.zeros(4)

        while not done and step_count < max_steps:
            raw_frame = env.render()
            if hud:
                proc_frame = draw_telemetry_hud(
                    raw_frame, step_count, max_steps, total_reward,
                    obs, current_action, status_label, status_type
                )
            else:
                proc_frame = raw_frame
            frames.append(proc_frame)

            action, _ = model.predict(obs, deterministic=deterministic)
            current_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step_count += 1

        # Capture final frame
        final_raw = env.render()
        if hud:
            final_proc = draw_telemetry_hud(
                final_raw, step_count, max_steps, total_reward,
                obs, current_action, status_label, status_type
            )
        else:
            final_proc = final_raw
        frames.append(final_proc)

        all_frames.extend(frames)
        episode_rewards.append(total_reward)
        print(f"  Episode {ep + 1}/{n_episodes}: reward={total_reward:.1f}, steps={step_count}")

    env.close()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    writer = imageio.get_writer(output_path, fps=ENV_FPS, format="FFMPEG",
                                codec="libx264", quality=8)
    for f in all_frames:
        writer.append_data(f)
    writer.close()

    mean_reward = float(np.mean(episode_rewards))
    print(f"  Saved video to {output_path} ({len(all_frames)} frames, mean reward={mean_reward:.1f})")
    return mean_reward


def main():
    parser = argparse.ArgumentParser(description="Record video with Sci-Fi Telemetry HUD")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.zip)")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions")
    parser.add_argument("--no-hud", action="store_true", help="Disable telemetry HUD")
    parser.add_argument("--label", default=None, help="Custom status badge text")
    parser.add_argument("--type", default=None, choices=["healthy", "damaged", "recovered"],
                        help="Status badge color theme")
    args = parser.parse_args()

    output = args.output or os.path.join(VIDEO_DIR, "recorded.mp4")
    record_episode(
        args.model, output, n_episodes=args.episodes,
        deterministic=not args.stochastic, hud=not args.no_hud,
        status_label=args.label, status_type=args.type
    )


if __name__ == "__main__":
    main()
