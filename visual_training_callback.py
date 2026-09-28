"""
Real-Time Visual Training Telemetry HUD & Biomechanical Phase-Portrait Callback.
Monitors 3D Walker training dynamics with:
  1. Live cyber-telemetry console progress displays
  2. Phase-space orbital limit cycles (joint angle vs joint velocity)
  3. Gait symmetry and stability tracking
  4. CSV telemetry logging

Author: Geo Mathew Joseph
"""
import os
import csv
import time
import atexit
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from config import PLOT_DIR, LOG_DIR

THEME = {
    "bg_canvas": "#0B0F19",
    "bg_panel": "#111827",
    "border": "#1F2937",
    "grid": "#1F2937",
    "text_main": "#F3F4F6",
    "text_muted": "#9CA3AF",
    "accent_cyan": "#00E5FF",
    "accent_green": "#00FF88",
    "accent_amber": "#FFB300",
    "accent_red": "#FF3366",
    "accent_purple": "#BD00FF",
}


class VisualTrainingTelemetryCallback(BaseCallback):
    """
    High-performance telemetry callback for 3D RL Walker training.
    Provides live terminal HUD updates and periodic phase-portrait visual dumps.
    """
    def __init__(
        self,
        log_path: str,
        total_timesteps: int,
        generation_name: str = "Training",
        gen_idx: int = 1,
        total_gens: int = 4,
        portrait_freq: int = 25_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_path = log_path
        self.total_timesteps = total_timesteps
        self.generation_name = generation_name
        self.gen_idx = gen_idx
        self.total_gens = total_gens
        self.portrait_freq = portrait_freq

        self.csv_file = None
        self.csv_writer = None
        self._episode_count = 0
        self._last_print_time = 0.0
        self._start_time = 0.0

        # Phase orbital trajectories buffer for limit cycle portrait
        self._hip_angles = []
        self._hip_vels = []
        self._knee_angles = []
        self._knee_vels = []
        self._recent_rewards = []
        self._recent_speeds = []

    def _on_training_start(self):
        self._start_time = time.time()
        self._last_print_time = self._start_time
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(PLOT_DIR, exist_ok=True)

        is_empty = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        self.csv_file = open(self.log_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if is_empty:
            self.csv_writer.writerow([
                "timestep", "episode", "reward", "length", "forward_vel", "generation"
            ])
        # Register cleanup in case training crashes without calling _on_training_end
        atexit.register(self._close_csv)

    def _on_step(self) -> bool:
        # Buffer joint states for limit cycle phase portrait from training env
        try:
            obs = self.locals.get("new_obs")
            if obs is not None and len(obs) > 0:
                first_obs = obs[0]
                if len(first_obs) >= 26:
                    # joint angles: [10..17], joint vels: [18..25]
                    # Left hip pitch is index 11, left knee is index 12
                    self._hip_angles.append(float(first_obs[11]))
                    self._hip_vels.append(float(first_obs[19]))
                    self._knee_angles.append(float(first_obs[12]))
                    self._knee_vels.append(float(first_obs[20]))

                    if len(self._hip_angles) > 1500:
                        self._hip_angles.pop(0)
                        self._hip_vels.pop(0)
                        self._knee_angles.pop(0)
                        self._knee_vels.pop(0)
        except Exception:
            pass

        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                # Use episode mean forward velocity from info if available,
                # otherwise fall back to last-step vy
                ep_vy = info.get("vy", 0.0)

                self._recent_rewards.append(ep_reward)
                self._recent_speeds.append(ep_vy)
                if len(self._recent_rewards) > 30:
                    self._recent_rewards.pop(0)
                    self._recent_speeds.pop(0)

                self.csv_writer.writerow([
                    self.num_timesteps,
                    self._episode_count,
                    f"{ep_reward:.2f}",
                    ep_length,
                    f"{ep_vy:.3f}",
                    self.gen_idx,
                ])
                self.csv_file.flush()

        # Terminal Cyber-HUD update at ~1.5 Hz interval
        now = time.time()
        if now - self._last_print_time >= 1.2:
            elapsed = now - self._start_time
            fps = self.num_timesteps / max(0.01, elapsed)
            pct = (self.num_timesteps / max(1, self.total_timesteps)) * 100.0
            mean_rew = np.mean(self._recent_rewards) if self._recent_rewards else 0.0
            mean_spd = np.mean(self._recent_speeds) if self._recent_speeds else 0.0

            # ASCII progress bar
            bar_len = 16
            filled = int(bar_len * (pct / 100.0))
            bar = "=" * filled + ">" + " " * (bar_len - filled - 1) if filled < bar_len else "=" * bar_len

            print(
                f"\r [HUD] Gen {self.gen_idx}/{self.total_gens} [{bar}] {pct:5.1f}% | "
                f"Step: {self.num_timesteps:6d}/{self.total_timesteps:6d} | "
                f"Rew: {mean_rew:+6.1f} | Spd: {mean_spd:+4.2f}m/s | "
                f"Rate: {fps:4.0f} fps",
                end="",
                flush=True,
            )
            self._last_print_time = now

        # Periodic Phase-Portrait Dump
        if self.portrait_freq > 0 and (self.num_timesteps % self.portrait_freq < self.training_env.num_envs):
            self._render_live_phase_portrait()

        return True

    def _render_live_phase_portrait(self):
        """Generates a dynamic 2-panel phase portrait limit cycle plot during training."""
        if len(self._hip_angles) < 50:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=THEME["bg_canvas"])

        # Panel 1: Hip Pitch Phase Portrait
        ax1.set_facecolor(THEME["bg_panel"])
        ax1.grid(True, color=THEME["grid"], linestyle="--", alpha=0.6)
        h_pts = np.array(self._hip_angles)
        h_vls = np.array(self._hip_vels)
        ax1.plot(h_pts, h_vls, color=THEME["accent_cyan"], linewidth=1.2, alpha=0.85, label="Hip Orbit")
        ax1.scatter(h_pts[-1], h_vls[-1], color=THEME["accent_green"], s=70, zorder=5, label="Current State")
        ax1.set_title("Hip Joint Limit Cycle Phase Portrait (θ vs. dθ/dt)", fontsize=11, fontweight="bold", color=THEME["text_main"])
        ax1.set_xlabel("Hip Angle (rad)", fontsize=10, color=THEME["text_muted"])
        ax1.set_ylabel("Hip Angular Velocity (rad/s)", fontsize=10, color=THEME["text_muted"])
        ax1.tick_params(colors=THEME["text_muted"])
        ax1.legend(facecolor=THEME["bg_canvas"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], loc="upper right")

        # Panel 2: Knee Joint Phase Portrait
        ax2.set_facecolor(THEME["bg_panel"])
        ax2.grid(True, color=THEME["grid"], linestyle="--", alpha=0.6)
        k_pts = np.array(self._knee_angles)
        k_vls = np.array(self._knee_vels)
        ax2.plot(k_pts, k_vls, color=THEME["accent_amber"], linewidth=1.2, alpha=0.85, label="Knee Orbit")
        ax2.scatter(k_pts[-1], k_vls[-1], color=THEME["accent_green"], s=70, zorder=5, label="Current State")
        ax2.set_title("Knee Joint Limit Cycle Phase Portrait (θ vs. dθ/dt)", fontsize=11, fontweight="bold", color=THEME["text_main"])
        ax2.set_xlabel("Knee Angle (rad)", fontsize=10, color=THEME["text_muted"])
        ax2.set_ylabel("Knee Angular Velocity (rad/s)", fontsize=10, color=THEME["text_muted"])
        ax2.tick_params(colors=THEME["text_muted"])
        ax2.legend(facecolor=THEME["bg_canvas"], edgecolor=THEME["border"], labelcolor=THEME["text_main"], loc="upper right")

        fig.suptitle(f"AEGIS 3D — REAL-TIME GAIT LIMIT CYCLES [{self.generation_name}]",
                     fontsize=13, fontweight="bold", color=THEME["accent_cyan"], y=0.98)

        portrait_path = os.path.join(PLOT_DIR, "live_gait_phase_portrait.png")
        fig.savefig(portrait_path, dpi=140, facecolor=THEME["bg_canvas"], bbox_inches="tight")
        plt.close(fig)

    def _close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def _on_training_end(self):
        self._close_csv()
        self._render_live_phase_portrait()
        print()
