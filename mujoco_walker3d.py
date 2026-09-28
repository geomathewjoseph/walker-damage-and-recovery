"""
3D Bipedal Robot Mech Simulation & Video Renderer using MuJoCo 3D Physics Engine.
Supports 3D camera rendering, real PPO policy inference, neural lesion injection,
and 720p HD video generation with authentic telemetry HUD.

Author: Geo Mathew Joseph
"""
import os
import shutil
import argparse
import numpy as np
import imageio
import cv2
import mujoco

from config import VIDEO_DIR, CHECKPOINT_DIR, DOCS_DIR, WALKER_3D_CHECKPOINT
from walker3d_env import Walker3DEnv, MECH_3D_ROBOT_XML


def draw_telemetry_hud(
    frame: np.ndarray,
    step: int,
    max_steps: int,
    reward: float,
    obs: np.ndarray,
    action: np.ndarray,
    status_label: str = None,
    status_type: str = "healthy",
    inset_frame: np.ndarray = None,
) -> np.ndarray:
    """
    Renders an authentic, broadcast-quality 720p HD scientific HUD telemetry overlay with:
      - Multi-camera Picture-in-Picture (PiP) inset
      - True forward speed (vy), lateral velocity (vx), and pitch/roll tilt
      - All 8 active joint actuator deflection bars with zero-center lines
      - Biomechanical gait phase identification (Single stance L/R, Double support, Flight)
    """
    h, w = 720, 1280
    frame_hd = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = frame_hd.copy()

    # 1. Top Glass Banner
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 30), -1)
    cv2.line(overlay, (0, 55), (w, 55), (0, 220, 255), 2)

    # 2. Right Telemetry Card
    cv2.rectangle(overlay, (950, 68), (1260, 248), (15, 20, 30), -1)
    cv2.rectangle(overlay, (950, 68), (1260, 248), (80, 90, 110), 1)

    # 3. PiP Secondary Camera Card (if provided)
    if inset_frame is not None:
        cv2.rectangle(overlay, (950, 260), (1260, 480), (15, 20, 30), -1)
        cv2.rectangle(overlay, (950, 260), (1260, 480), (0, 220, 255), 1)

    # 4. Left Actuator Card (Enclosing 8 DOFs + Contact Sensors + Gait Phase)
    cv2.rectangle(overlay, (20, 460), (370, 702), (15, 20, 30), -1)
    cv2.rectangle(overlay, (20, 460), (370, 702), (80, 90, 110), 1)

    # Alpha blend overlay with background
    cv2.addWeighted(overlay, 0.82, frame_hd, 0.18, 0, frame_hd)

    # PiP Secondary Camera Video Inset
    if inset_frame is not None:
        pip_w, pip_h = 296, 180
        pip_resized = cv2.resize(inset_frame, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
        frame_hd[290:290 + pip_h, 957:957 + pip_w] = pip_resized
        # Inset title
        cv2.putText(frame_hd, "CAM 2: SAGITTAL PROFILE", (965, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        # Reticle corner accents
        cv2.drawMarker(frame_hd, (957, 290), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.drawMarker(frame_hd, (957 + pip_w, 290), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.drawMarker(frame_hd, (957, 290 + pip_h), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
        cv2.drawMarker(frame_hd, (957 + pip_w, 290 + pip_h), (0, 220, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

    # Top Bar Titles
    cv2.putText(frame_hd, "WALKER NEURO-RECOVERY LAB", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_hd, "3D BIPEDAL MECH TELEMETRY", (360, 35),
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

    # True State Decoding from Obs (36-dim standard)
    if len(obs) >= 28:
        z = float(obs[0])
        roll, pitch = float(obs[1]), float(obs[2])
        tilt = float(np.degrees(np.hypot(roll, pitch)))
        vx = float(obs[4]) # lateral drift
        vy = float(obs[5]) # forward velocity
        contact_l = bool(obs[26] > 0.5)
        contact_r = bool(obs[27] > 0.5)
    else:
        z = 1.48
        tilt = 0.0
        vx = 0.0
        vy = float(obs[2]) if len(obs) > 2 else 0.0
        contact_l = True
        contact_r = True

    reward_color = (50, 255, 100) if reward >= 0 else (50, 80, 255)

    # Right Telemetry Text
    cv2.putText(frame_hd, "SYSTEM TELEMETRY", (970, 93),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"STEP:   {step:04d} / {max_steps}", (970, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"REWARD: {reward:+6.1f}", (970, 147),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, reward_color, 2, cv2.LINE_AA)
    cv2.putText(frame_hd, f"FORWARD: {vy:+5.2f} m/s", (970, 174),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"LATERAL: {vx:+5.2f} m/s", (970, 201),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, f"ELEV/TILT: {z:.2f}m | {tilt:4.1f} deg", (970, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Left Actuator Text & Dynamic Torque Gauges (All 8 Active DOFs)
    cv2.putText(frame_hd, "ACTUATOR TORQUES (8-DOF)", (30, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    joint_names = [
        "L HIP ROLL", "L HIP PITCH", "L KNEE", "L ANKLE",
        "R HIP ROLL", "R HIP PITCH", "R KNEE", "R ANKLE",
    ]
    for i, name in enumerate(joint_names):
        val = float(action[i]) if action is not None and i < len(action) else 0.0
        y = 502 + i * 19
        cv2.putText(frame_hd, name, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        # Bar track
        cv2.rectangle(frame_hd, (130, y - 8), (350, y + 2), (40, 45, 55), -1)
        # Center zero line
        cv2.line(frame_hd, (240, y - 10), (240, y + 4), (100, 110, 120), 1)
        # Dynamic deflection bar
        bar_w = int(np.clip(val, -1.0, 1.0) * 105)
        color = (255, 200, 0) if val >= 0 else (50, 80, 255)
        if bar_w >= 0:
            cv2.rectangle(frame_hd, (240, y - 7), (240 + bar_w, y + 1), color, -1)
        else:
            cv2.rectangle(frame_hd, (240 + bar_w, y - 7), (240, y + 1), color, -1)

    # Leg Contact Flags & Gait Phase Badge
    c1_txt = "[ON]" if contact_l else "[AIR]"
    c2_txt = "[ON]" if contact_r else "[AIR]"
    c1_col = (50, 255, 100) if contact_l else (120, 120, 120)
    c2_col = (50, 255, 100) if contact_r else (120, 120, 120)
    cv2.putText(frame_hd, "CONTACT: L", (30, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, c1_txt, (110, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c1_col, 1, cv2.LINE_AA)
    cv2.putText(frame_hd, "R", (165, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame_hd, c2_txt, (185, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c2_col, 1, cv2.LINE_AA)

    # Dynamic Gait Phase Badge
    if contact_l and not contact_r:
        phase_str = "GAIT: SINGLE STANCE (LEFT)"
        phase_col = (0, 220, 255)
    elif contact_r and not contact_l:
        phase_str = "GAIT: SINGLE STANCE (RIGHT)"
        phase_col = (255, 180, 0)
    elif contact_l and contact_r:
        phase_str = "GAIT: DOUBLE SUPPORT"
        phase_col = (50, 255, 100)
    else:
        phase_str = "GAIT: AIRBORNE / FLIGHT"
        phase_col = (50, 80, 255)

    cv2.putText(frame_hd, phase_str, (30, 692),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, phase_col, 2, cv2.LINE_AA)

    # Bottom Progress Bar
    progress = np.clip(step / float(max_steps), 0.0, 1.0)
    cv2.rectangle(frame_hd, (0, 712), (w, 720), (30, 35, 45), -1)
    cv2.rectangle(frame_hd, (0, 712), (int(w * progress), 720), (0, 220, 255), -1)

    return frame_hd


def render_3d_robot_video(
    output_path: str,
    model_path: str = None,
    severity: float = 0.0,
    status_label: str = None,
    status_type: str = "healthy",
    n_frames: int = 400,
    camera_name: str = "cinematic_3q",
    width: int = 640,
    height: int = 480,
    save_preview: bool = True,
    enable_pip: bool = True,
):
    """
    Renders 3D physics-based simulation of the bipedal mech with MuJoCo.
    If model_path exists, runs inference using the REAL trained PPO policy.
    Includes secondary camera Picture-in-Picture inset.
    """
    if model_path is None:
        if status_type == "recovered":
            rec_candidate = os.path.join(CHECKPOINT_DIR, "walker3d_recovered_0.5.zip")
            if os.path.exists(rec_candidate):
                model_path = rec_candidate
            else:
                model_path = WALKER_3D_CHECKPOINT
        elif status_type == "evolved":
            evo_candidate = os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip")
            if os.path.exists(evo_candidate):
                model_path = evo_candidate
            else:
                model_path = WALKER_3D_CHECKPOINT
        else:
            model_path = WALKER_3D_CHECKPOINT

    has_model = model_path and os.path.exists(model_path)

    if has_model:
        from stable_baselines3 import PPO
        print(f"Loading trained 3D policy from {model_path} for realistic physics inference...")
        model = PPO.load(model_path, device="cpu")

        # Optionally apply damage if severity > 0 or status is damaged
        if status_type == "damaged":
            effective_sev = severity if severity > 0.0 else 0.5
            from damage import apply_damage
            print(f"Injecting acute {effective_sev*100:.0f}% neuron_kill lesion into 3D policy...")
            model = apply_damage(model, severity=effective_sev, mode="neuron_kill", seed=42)

        env = Walker3DEnv(
            render_mode="rgb_array",
            camera_name=camera_name,
            render_width=width,
            render_height=height,
        )
        obs, _ = env.reset()

        frames = []
        fps = 50
        total_reward = 0.0

        if status_label is None:
            if status_type == "healthy":
                status_label = "[3D MODEL: HEALTHY / 100% ASYMPTOTIC]"
            elif status_type == "damaged":
                status_label = f"[ACUTE 3D NEURAL LESION: {int(severity*100 if severity > 0 else 50)}% ABLATED]"
            elif status_type == "recovered":
                status_label = "[NEUROPLASTIC REHABILITATION: RESTORED]"
            elif status_type == "evolved":
                status_label = "[EVOLVED NEURO-RESISTANT 3D AGENT]"

        print(f"Rendering 3D MuJoCo Mech Video ({n_frames} frames, severity={severity:.1f}, camera={camera_name})...")

        secondary_cam = "side_profile" if camera_name != "side_profile" else "front_action"

        for step in range(n_frames):
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            total_reward += rew

            raw_pixels = env.render()
            inset_pixels = None
            if enable_pip and env._renderer is not None:
                env._renderer.update_scene(env.data, camera=secondary_cam)
                inset_pixels = env._renderer.render()

            hud_frame = draw_telemetry_hud(
                raw_pixels, step, n_frames, total_reward,
                obs, action,
                status_label=status_label,
                status_type=status_type,
                inset_frame=inset_pixels,
            )
            frames.append(hud_frame)

            if term:
                obs, _ = env.reset()

        env.close()

    else:
        # Fallback procedural preview if model not yet trained
        print("No trained checkpoint found. Running standalone physics preview...")
        model = mujoco.MjModel.from_xml_string(MECH_3D_ROBOT_XML)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height, width)

        frames = []
        fps = 50
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 1.48

        total_reward = 0.0

        for step in range(n_frames):
            t = step * 0.004 * 5
            data.ctrl[:] = np.sin(t * 3.5) * 0.4
            mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=camera_name)
            raw_pixels = renderer.render()

            inset_pixels = None
            if enable_pip:
                renderer.update_scene(data, camera="side_profile")
                inset_pixels = renderer.render()

            obs_mock = np.zeros(36, dtype=np.float32)
            obs_mock[0] = data.qpos[2]
            obs_mock[5] = float(data.qvel[1])
            obs_mock[26] = 1.0
            obs_mock[27] = 1.0
            action_mock = data.ctrl[:8].copy()

            hud_frame = draw_telemetry_hud(
                raw_pixels, step, n_frames, total_reward,
                obs_mock, action_mock,
                status_label=status_label or "[STANDALONE PREVIEW]",
                status_type=status_type,
                inset_frame=inset_pixels,
            )
            frames.append(hud_frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, format="FFMPEG", codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

    print(f"Saved 3D MuJoCo Robot Video to {output_path} ({len(frames)} frames)")

    # Save high-resolution preview frame to docs/assets
    if save_preview and len(frames) > 0:
        preview_idx = min(120, len(frames) - 1)
        preview_frame = frames[preview_idx]
        fname = os.path.basename(output_path).replace(".mp4", "_preview.png")
        docs_dest = os.path.join(DOCS_DIR, "assets", fname)
        if os.path.exists(os.path.dirname(docs_dest)):
            try:
                imageio.imwrite(docs_dest, preview_frame)
                print(f"Saved Preview Snapshot to {docs_dest}")
            except Exception:
                pass


def run_batch_showcase():
    """Render all benchmark 3D showcase videos across multiple camera angles."""
    print("=" * 70)
    print("   BATCH RENDERING ALL 3D BIPEDAL MECH SHOWCASE VIDEOS")
    print("=" * 70)

    video_tasks = [
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_healthy.mp4"),
            "model": WALKER_3D_CHECKPOINT,
            "severity": 0.0,
            "type": "healthy",
            "cam": "cinematic_3q",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_damaged_0.5.mp4"),
            "model": WALKER_3D_CHECKPOINT,
            "severity": 0.5,
            "type": "damaged",
            "cam": "cinematic_3q",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_recovered_0.5.mp4"),
            "model": os.path.join(CHECKPOINT_DIR, "walker3d_recovered_0.5.zip"),
            "severity": 0.0,
            "type": "recovered",
            "cam": "cinematic_3q",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_evolved.mp4"),
            "model": os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"),
            "severity": 0.0,
            "type": "evolved",
            "cam": "cinematic_3q",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "mujoco_3d_evolved.mp4"),
            "model": os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip"),
            "severity": 0.0,
            "type": "evolved",
            "cam": "cinematic_3q",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_side.mp4"),
            "model": WALKER_3D_CHECKPOINT,
            "severity": 0.0,
            "type": "healthy",
            "cam": "side_profile",
            "frames": 350,
        },
        {
            "output": os.path.join(VIDEO_DIR, "walker3d_front.mp4"),
            "model": WALKER_3D_CHECKPOINT,
            "severity": 0.0,
            "type": "healthy",
            "cam": "front_action",
            "frames": 350,
        },
    ]

    for t in video_tasks:
        print("\n" + "-" * 60)
        print(f"Rendering {os.path.basename(t['output'])} ({t['cam']}, type={t['type']})...")
        render_3d_robot_video(
            output_path=t["output"],
            model_path=t["model"],
            severity=t["severity"],
            status_type=t["type"],
            camera_name=t["cam"],
            n_frames=t["frames"],
            save_preview=True,
        )

    print("\n" + "=" * 70)
    print("   ALL 3D VIDEOS SUCCESSFULLY RENDERED & UPDATED")
    print("=" * 70)


# Alias for backwards compatibility with CLI --all flag
render_all_3d_videos = run_batch_showcase


def main():
    parser = argparse.ArgumentParser(description="Render 3D MuJoCo Bipedal Robot Mech Video")
    parser.add_argument("--output", default=os.path.join(VIDEO_DIR, "walker3d_healthy.mp4"))
    parser.add_argument("--model", default=None, help="Path to trained PPO checkpoint")
    parser.add_argument("--severity", type=float, default=0.0)
    parser.add_argument("--type", default="healthy", choices=["healthy", "damaged", "recovered", "evolved"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--frames", type=int, default=350)
    parser.add_argument("--cam", default="cinematic_3q")
    parser.add_argument("--all", action="store_true", help="Render all showcase 3D videos")
    args = parser.parse_args()

    if args.all:
        render_all_3d_videos()
    else:
        render_3d_robot_video(
            output_path=args.output,
            model_path=args.model,
            severity=args.severity,
            status_label=args.label,
            status_type=args.type,
            n_frames=args.frames,
            camera_name=args.cam,
            save_preview=True,
        )


if __name__ == "__main__":
    main()
