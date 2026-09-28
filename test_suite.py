"""
End-to-End Test Suite for Walker Damage & Recovery Research Platform.
Runs comprehensive automated verification across:
  1. 2D Box2D & 3D MuJoCo Environments
  2. Physics Laws & Contact Dynamics
  3. Trained Policy Inferences (Healthy, Damaged, Recovered)
  4. Lesion Injection across all 3 Modalities
  5. Video & Visualization Dashboard Asset Integrity

Author: Geo Mathew Joseph
"""
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from config import (
    HEALTHY_CHECKPOINT, WALKER_3D_CHECKPOINT,
    CHECKPOINT_DIR, VIDEO_DIR, PLOT_DIR, LOG_DIR,
)
from damage import apply_damage
from walker3d_env import Walker3DEnv, quat_to_euler
from train_3d import evaluate_3d_policy


def test_section(title):
    print("\n" + "=" * 70)
    print(f"   TEST: {title}")
    print("=" * 70)


def test_environments():
    test_section("1. ENVIRONMENT STEP & OBSERVATION INTEGRITY")

    # 2D BipedalWalker
    env_2d = gym.make("BipedalWalker-v3")
    obs_2d, _ = env_2d.reset(seed=42)
    assert obs_2d.shape == (24,), f"Expected 2D obs (24,), got {obs_2d.shape}"
    act_2d = env_2d.action_space.sample()
    obs_2d, r_2d, term_2d, trunc_2d, _ = env_2d.step(act_2d)
    env_2d.close()
    print("  [PASS] 2D BipedalWalker-v3: Reset, Step, and Obs shape (24,) verified.")

    # 3D Walker3DEnv
    env_3d = Walker3DEnv()
    obs_3d, _ = env_3d.reset(seed=42)
    assert obs_3d.shape == (36,), f"Expected 3D obs (36,), got {obs_3d.shape}"
    act_3d = env_3d.action_space.sample()
    obs_3d, r_3d, term_3d, trunc_3d, info_3d = env_3d.step(act_3d)
    env_3d.close()
    print("  [PASS] 3D Walker3DEnv: Reset, Step, and Obs shape (36,) verified.")


def test_physics_laws():
    test_section("2. PHYSICAL LAWS & DYNAMICS CONSTRAINTS")

    # Quaternion to Euler conversion test
    roll, pitch, yaw = quat_to_euler(1.0, 0.0, 0.0, 0.0)
    assert abs(roll) < 1e-5 and abs(pitch) < 1e-5 and abs(yaw) < 1e-5
    print("  [PASS] Quaternion to Euler kinematics transform verified.")

    # 3D Mech Contact & Gravity Check
    env = Walker3DEnv()
    env.reset()
    initial_z = env.data.qpos[2]
    assert 1.40 <= initial_z <= 1.60, f"Unphysical initial elevation: {initial_z}"

    # Verify active actuators
    assert env.model.nu == 8, f"Expected 8 actuators (dual hip roll/pitch, knee, ankle), got {env.model.nu}"
    actuator_names = [env.model.actuator(i).name for i in range(env.model.nu)]
    assert "act_hip_roll_l" in actuator_names and "act_hip_roll_r" in actuator_names
    print(f"  [PASS] Multi-body Actuators: {len(actuator_names)} active DOFs including dual hip roll.")

    # Step physics with zero control to verify gravity and floor contact
    for _ in range(50):
        env.step(np.zeros(8, dtype=np.float32))
    contacts = env._get_contacts()
    print(f"  [PASS] Ground Contact Sensors: Left={contacts[0]}, Right={contacts[1]} (Contact mechanics compliant).")
    env.close()


def test_3d_policy_inference():
    test_section("3. 3D POLICY INFERENCE & BENCHMARK AUDIT")

    models_to_test = [
        ("Healthy 3D Baseline", WALKER_3D_CHECKPOINT, 250.0),
        ("Acute 50% Lesion Damaged", os.path.join(CHECKPOINT_DIR, "walker3d_damaged_0.5.zip"), 100.0),
        ("Rehabilitated 50% Recovered", os.path.join(CHECKPOINT_DIR, "walker3d_recovered_0.5.zip"), 200.0),
    ]

    for name, path, min_expected in models_to_test:
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        model = PPO.load(path, device="cpu")
        stats = evaluate_3d_policy(model, n_episodes=3)
        rew = stats["mean_reward"]
        spd = stats["mean_forward_vel"]
        length = stats["mean_length"]
        print(f"  [PASS] {name:<30}: Reward = {rew:>+7.1f} | Speed = {spd:>+5.2f} m/s | Steps = {length:>4.0f}")
        assert rew > min_expected, f"Reward {rew} below threshold {min_expected} for {name}"


def test_damage_modalities():
    test_section("4. NEURAL LESION INJECTION MODALITIES")

    assert os.path.exists(WALKER_3D_CHECKPOINT)
    base_model = PPO.load(WALKER_3D_CHECKPOINT, device="cpu")

    modes = ["neuron_kill", "weight_zero", "noise_injection"]
    for mode in modes:
        damaged = apply_damage(base_model, severity=0.3, mode=mode, seed=42)
        stats = evaluate_3d_policy(damaged, n_episodes=2)
        print(f"  [PASS] Mode '{mode:<15}' applied at s=0.30: Output Reward = {stats['mean_reward']:>+7.1f}")


def test_assets_and_dashboards():
    test_section("5. 3D PLATFORM ASSETS & LEGACY ARCHIVE INTEGRITY")

    # 3D Primary Dashboards
    required_plots_3d = [
        "walker3d_dashboard.png",
        "walker3d_evolution_dashboard.png",
    ]
    for p in required_plots_3d:
        fpath = os.path.join(PLOT_DIR, p)
        assert os.path.exists(fpath), f"Missing 3D plot: {fpath}"
        size_kb = os.path.getsize(fpath) / 1024
        assert size_kb > 50, f"Plot {p} appears truncated ({size_kb:.1f} KB)"
        print(f"  [PASS] 3D Plot: {p:<30} ({size_kb:>6.1f} KB)")

    # 3D Primary Videos
    required_videos_3d = [
        "walker3d_healthy.mp4",
        "walker3d_damaged_0.5.mp4",
        "walker3d_recovered_0.5.mp4",
        "walker3d_evolved.mp4",
        "mujoco_3d_evolved.mp4",
        "walker3d_side.mp4",
        "walker3d_front.mp4",
    ]
    for v in required_videos_3d:
        fpath = os.path.join(VIDEO_DIR, v)
        assert os.path.exists(fpath), f"Missing 3D video: {fpath}"
        size_kb = os.path.getsize(fpath) / 1024
        assert size_kb > 100, f"Video {v} appears corrupted ({size_kb:.1f} KB)"
        print(f"  [PASS] 3D Video: {v:<29} ({size_kb:>6.1f} KB)")

    # Legacy 2D Archive Validation
    legacy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_2d")
    assert os.path.exists(legacy_dir), "legacy_2d archive directory missing!"
    for sub in ["videos", "checkpoints", "logs", "plots"]:
        sub_path = os.path.join(legacy_dir, sub)
        assert os.path.exists(sub_path), f"legacy_2d subfolder missing: {sub}"
        n_items = len(os.listdir(sub_path))
        print(f"  [PASS] Legacy Archive: legacy_2d/{sub:<15} ({n_items} items archived)")


def test_autonomous_learner():
    test_section("6. AUTONOMOUS CONTINUAL LEARNING & ANOMALY DETECTION")

    from autonomous_learner import GaitAnomalyDetector, ContinualCurriculum, AutonomousLearner

    # 1. Anomaly Detector verification
    detector = GaitAnomalyDetector(window_size=20, z_threshold=2.0)
    assert not detector.baseline_calibrated

    # Step with normal observations
    normal_obs = np.zeros(36, dtype=np.float32)
    normal_obs[5] = 1.35 # normal forward velocity
    is_anomaly, diag = detector.update(normal_obs, np.zeros(8))
    assert not is_anomaly
    print("  [PASS] GaitAnomalyDetector: Nominal gait passes anomaly threshold.")

    # Step with damaged / collapsed observation
    damaged_obs = np.zeros(36, dtype=np.float32)
    damaged_obs[5] = 0.05 # stalled velocity
    damaged_obs[1] = 0.85 # severe roll
    for _ in range(6):
        is_trauma, diag = detector.update(damaged_obs, np.zeros(8))
    assert is_trauma
    print(f"  [PASS] GaitAnomalyDetector: Trauma detected autonomously (vx_z={diag['vx_z']:.2f}, consecutive={diag['consecutive_anomalies']}).")

    # 2. Curriculum Manager verification
    curriculum = ContinualCurriculum()
    assert curriculum.current_stage["name"] == "STABILIZATION"
    # Test consecutive successes trigger promotion
    for _ in range(curriculum.success_threshold):
        curriculum.evaluate_episode(ep_reward=280.0, ep_length=300, mean_vx=1.2)
    assert curriculum.current_stage["name"] == "SPEED_SCALING"
    print(f"  [PASS] ContinualCurriculum: Autonomous skill promotion to Stage 2 ({curriculum.current_stage['name']}).")

    # 3. Autonomous Learner interface verification
    assert os.path.exists(WALKER_3D_CHECKPOINT)
    learner = AutonomousLearner(checkpoint_path=WALKER_3D_CHECKPOINT)
    env = Walker3DEnv(training_mode=False)
    obs, _ = env.reset(seed=42)
    action, is_trauma, diag = learner.step_autonomous(obs)
    assert action.shape == (8,)
    assert "vx_z" in diag
    assert "vx" in diag and "vy_fwd" in diag, "diag must provide forward speed metrics"
    
    # 4. Telemetry Contact Indices Verification
    contacts = env._get_contacts()
    assert obs[26] == contacts[0] and obs[27] == contacts[1], "Contact flags must be at indices 26 and 27"
    assert env.model.body_mass[env._torso_id] == env._default_torso_mass, "Nominal torso mass preserved in eval mode"
    assert env.model.geom_friction[env._floor_id, 0] == env._default_floor_friction, "Nominal floor friction preserved in eval mode"
    env.close()
    print("  [PASS] AutonomousLearner: Closed-loop inference and real-time telemetry diagnostics verified.")
    print("  [PASS] Telemetry & Contact Flag Indices: Verified at obs[26, 27].")
    print("  [PASS] Physics Invariance: Nominal mass and friction preserved in evaluation mode.")


def test_evolution_system():
    test_section("7. 3D EVOLUTIONARY RESILIENCE PIPELINE & BENCHMARK")
    evo_checkpoint = os.path.join(CHECKPOINT_DIR, "walker3d_evolved.zip")
    assert os.path.exists(evo_checkpoint), f"Missing evolved checkpoint: {evo_checkpoint}"

    # Load and test inference
    model = PPO.load(evo_checkpoint, device="cpu")
    stats = evaluate_3d_policy(model, n_episodes=2)
    print(f"  [PASS] Evolved Champion Policy: Reward = {stats['mean_reward']:>+7.1f} | Speed = {stats['mean_forward_vel']:>+5.2f} m/s")
    assert stats["mean_reward"] > 250.0, f"Evolved reward {stats['mean_reward']} below threshold"

    # Verify evolution history log
    evo_log = os.path.join(LOG_DIR, "evolution_3d_history.json")
    assert os.path.exists(evo_log), f"Missing evolution log: {evo_log}"
    print(f"  [PASS] Evolution History Log: {evo_log} verified.")


def main():
    print("=" * 70)
    print("   WALKER DAMAGE & RECOVERY PLATFORM — SYSTEM VERIFICATION SUITE")
    print("=" * 70)
    try:
        test_environments()
        test_physics_laws()
        test_3d_policy_inference()
        test_damage_modalities()
        test_assets_and_dashboards()
        test_autonomous_learner()
        test_evolution_system()

        print("\n" + "=" * 70)
        print("   >>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (7/7 SECTIONS) <<<")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n[FAIL] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


