"""
Central Configuration for the Walker Damage & Recovery Research Suite.
Defines all hyperparameters, environment settings, directories, and paths
for both the 2D (BipedalWalker-v3) and 3D (Walker3DEnv MuJoCo) pipelines.

Author: Geo Mathew Joseph
"""
import os

# ──────────────────────────────────────────────
# 2D Environment (BipedalWalker-v3)
# ──────────────────────────────────────────────
ENV_NAME = "BipedalWalker-v3"
ENV_FPS = 50  # BipedalWalker physics runs at 50 Hz

# Policy network — deliberately compact so damage effects are biomechanically visible
POLICY_KWARGS = dict(
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)

import torch

# Parallelism for Box2D rollouts
N_ENVS = 16

# Compute Device (Automatic CUDA GPU Acceleration with CPU fallback)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PPO Hyperparameters (2D Agent)
PPO_PARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,        # per env; total batch = n_steps × N_ENVS
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,      # slight entropy bonus for exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Training Horizons (2D Agent)
TOTAL_TIMESTEPS = 2_000_000
CHECKPOINT_FREQ = 100_000
SEED = 42

# Recovery Horizons & Severities (2D Agent)
RECOVERY_TIMESTEPS = 500_000
SEVERITIES = [0.0, 0.1, 0.3, 0.5, 0.7]

# ──────────────────────────────────────────────
# 3D Environment (Walker3DEnv MuJoCo 3.13)
# ──────────────────────────────────────────────
WALKER_3D_ENV_NAME = "Walker3DEnv"
WALKER_3D_FPS = 50
WALKER_3D_N_ENVS = 16  # High-throughput parallelized physics rollouts
WALKER_3D_TOTAL_TIMESTEPS = 1_200_000  # Intensive high-data training horizon
WALKER_3D_RECOVERY_TIMESTEPS = 200_000
WALKER_3D_SEVERITIES = [0.0, 0.1, 0.3, 0.5]

# Policy network — expanded capacity [256, 256] for high-fidelity multi-joint coordination
WALKER_3D_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)

# PPO Hyperparameters (3D Agent — Optimized for CUDA + large rollout buffers)
WALKER_3D_PPO_PARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,        # Total rollout buffer per update = 2048 * 16 = 32,768 steps
    batch_size=128,      # Mini-batch size for GPU tensor operations
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.003,      # Elevated entropy bonus for robust gait exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Realistic Physics & Biomechanics Hyperparameters
WALKER_3D_ACTUATOR_FILTER_ALPHA = 0.70  # First-order low-pass motor lag (35ms bandwidth)
WALKER_3D_ACTION_RATE_COEF = 0.02       # Action jerk penalty to suppress motor twitching
WALKER_3D_PERTURBATION_PROB = 0.015     # Probability per step of external force impulse
WALKER_3D_PERTURBATION_MAG = 25.0       # Push disturbance magnitude in Newtons

# ──────────────────────────────────────────────
# Filesystem Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# 2D Checkpoints & Logs
HEALTHY_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "healthy.zip")
EVOLVED_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "evolved_resistant.zip")
EXPERIMENT_RESULTS_PATH = os.path.join(LOG_DIR, "experiment_results.json")

# 3D Checkpoints & Logs
WALKER_3D_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "walker3d_healthy.zip")
WALKER_3D_LOG = os.path.join(LOG_DIR, "train_3d.csv")
EXPERIMENT_3D_RESULTS_PATH = os.path.join(LOG_DIR, "experiment_3d_results.json")

# Ensure fundamental directories exist
for d in [CHECKPOINT_DIR, VIDEO_DIR, LOG_DIR, PLOT_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)
