"""
3D Bipedal Robot Mech Simulation & Gymnasium Reinforcement Learning Environment.
Built on MuJoCo 3.13 Physics Engine.

Enforces physical laws:
  - Newtonian multi-body rigid-body dynamics
  - Coulomb friction cones and compliant floor contacts
  - Actuator gear torque saturation limits and energy dissipation
  - Dynamic balance and center-of-mass stabilization
  - Explicit termination upon balance collapse (falling)

Author: Geo Mathew Joseph
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ──────────────────────────────────────────────
# 3D Bipedal Mech MJCF XML Model Definition
# ──────────────────────────────────────────────
MECH_3D_ROBOT_XML = """
<mujoco model="aegis_bipedal_mech_3d">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" integrator="RK4" timestep="0.004"/>

  <visual>
    <headlight ambient="0.35 0.38 0.45" diffuse="0.7 0.75 0.8" specular="0.3 0.3 0.3"/>
    <rgba haze="0.05 0.08 0.12 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.05 0.08 0.15" rgb2="0.02 0.03 0.06" width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.08 0.12 0.18" rgb2="0.05 0.07 0.12" width="512" height="512"/>
    <material name="floor_mat" texture="grid_tex" texrepeat="20 20" reflectance="0.3" roughness="0.5"/>
    <material name="armor_dark" rgba="0.12 0.15 0.22 1" reflectance="0.4" roughness="0.4"/>
    <material name="armor_blue" rgba="0.18 0.35 0.55 1" reflectance="0.6" roughness="0.3"/>
    <material name="chrome" rgba="0.85 0.9 0.95 1" reflectance="0.85" roughness="0.15"/>
    <material name="cyan_glow" rgba="0.0 0.9 1.0 1" emission="0.8"/>
    <material name="crimson_glow" rgba="1.0 0.2 0.4 1" emission="0.9"/>
  </asset>

  <worldbody>
    <light pos="2 3 6" dir="-0.3 -0.4 -1" diffuse="0.9 0.95 1.0" castshadow="true"/>
    <light pos="-3 -2 4" dir="0.5 0.3 -0.8" diffuse="0.0 0.6 0.9" castshadow="false"/>
    
    <geom name="floor" type="plane" size="80 80 0.1" material="floor_mat" friction="1.4 0.01 0.001" solref="0.005 1" solimp="0.9 0.99 0.001"/>

    <!-- 3D Bipedal Mech Chassis -->
    <body name="torso" pos="0 0 1.55">
      <freejoint name="root"/>
      <geom name="cockpit" type="box" size="0.26 0.20 0.28" material="armor_blue" mass="16.0"/>
      <geom name="eye" type="sphere" size="0.11" pos="0 0.21 0.04" material="cyan_glow" mass="0.5"/>
      <geom name="sensor_l" type="box" size="0.07 0.14 0.16" pos="-0.29 0 0" material="armor_dark" mass="1.0"/>
      <geom name="sensor_r" type="box" size="0.07 0.14 0.16" pos="0.29 0 0" material="armor_dark" mass="1.0"/>
      <geom name="pelvis" type="cylinder" size="0.18 0.12" pos="0 0 -0.32" material="armor_dark" mass="3.5"/>

      <!-- LEFT LEG WITH ACTIVE ROLL & PITCH -->
      <body name="hip_mount_l" pos="-0.28 0 -0.35">
        <joint name="hip_roll_l" type="hinge" axis="0 1 0" range="-25 25" damping="3.0" armature="0.05"/>
        <geom name="hip_motor_l" type="cylinder" size="0.12 0.08" material="chrome" mass="1.5"/>

        <body name="thigh_mount_l" pos="0 0 0">
          <joint name="hip_pitch_l" type="hinge" axis="1 0 0" range="-55 55" damping="3.5" armature="0.05"/>
          <geom name="thigh_l" type="capsule" fromto="0 0 0 0 0 -0.50" size="0.075" material="armor_dark" mass="3.5"/>
          <geom name="piston_l" type="cylinder" size="0.025 0.35" pos="0 0.08 -0.22" material="chrome" mass="0.8"/>

          <body name="knee_mount_l" pos="0 0 -0.50">
            <joint name="knee_l" type="hinge" axis="-1 0 0" range="0 115" damping="3.5" armature="0.05"/>
            <geom name="knee_pivot_l" type="cylinder" size="0.10 0.08" material="chrome" mass="1.2"/>
            <geom name="knee_accent_l" type="sphere" size="0.06" material="cyan_glow" mass="0.1"/>
            <geom name="shin_l" type="capsule" fromto="0 0 0 0 0 -0.50" size="0.065" material="armor_blue" mass="2.8"/>

            <body name="foot_mount_l" pos="0 0 -0.50">
              <joint name="ankle_l" type="hinge" axis="1 0 0" range="-35 35" damping="2.5" armature="0.03"/>
              <geom name="foot_plate_l" type="box" size="0.13 0.24 0.035" pos="0 0.04 -0.035" material="armor_dark" mass="1.8" friction="1.4 0.01 0.001" solref="0.005 1" solimp="0.9 0.99 0.001"/>
              <geom name="toe_claw_l" type="box" size="0.11 0.08 0.025" pos="0 0.25 -0.035" material="chrome" mass="0.4" friction="1.4 0.01 0.001"/>
              <geom name="heel_spur_l" type="box" size="0.11 0.06 0.025" pos="0 -0.15 -0.035" material="chrome" mass="0.4" friction="1.4 0.01 0.001"/>
            </body>
          </body>
        </body>
      </body>

      <!-- RIGHT LEG WITH ACTIVE ROLL & PITCH -->
      <body name="hip_mount_r" pos="0.28 0 -0.35">
        <joint name="hip_roll_r" type="hinge" axis="0 1 0" range="-25 25" damping="3.0" armature="0.05"/>
        <geom name="hip_motor_r" type="cylinder" size="0.12 0.08" material="chrome" mass="1.5"/>

        <body name="thigh_mount_r" pos="0 0 0">
          <joint name="hip_pitch_r" type="hinge" axis="1 0 0" range="-55 55" damping="3.5" armature="0.05"/>
          <geom name="thigh_r" type="capsule" fromto="0 0 0 0 0 -0.50" size="0.075" material="armor_dark" mass="3.5"/>
          <geom name="piston_r" type="cylinder" size="0.025 0.35" pos="0 0.08 -0.22" material="chrome" mass="0.8"/>

          <body name="knee_mount_r" pos="0 0 -0.50">
            <joint name="knee_r" type="hinge" axis="-1 0 0" range="0 115" damping="3.5" armature="0.05"/>
            <geom name="knee_pivot_r" type="cylinder" size="0.10 0.08" material="chrome" mass="1.2"/>
            <geom name="knee_accent_r" type="sphere" size="0.06" material="cyan_glow" mass="0.1"/>
            <geom name="shin_r" type="capsule" fromto="0 0 0 0 0 -0.50" size="0.065" material="armor_blue" mass="2.8"/>

            <body name="foot_mount_r" pos="0 0 -0.50">
              <joint name="ankle_r" type="hinge" axis="1 0 0" range="-35 35" damping="2.5" armature="0.03"/>
              <geom name="foot_plate_r" type="box" size="0.13 0.24 0.035" pos="0 0.04 -0.035" material="armor_dark" mass="1.8" friction="1.4 0.01 0.001" solref="0.005 1" solimp="0.9 0.99 0.001"/>
              <geom name="toe_claw_r" type="box" size="0.11 0.08 0.025" pos="0 0.25 -0.035" material="chrome" mass="0.4" friction="1.4 0.01 0.001"/>
              <geom name="heel_spur_r" type="box" size="0.11 0.06 0.025" pos="0 -0.15 -0.035" material="chrome" mass="0.4" friction="1.4 0.01 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Multi-Angle Studio Cameras -->
    <camera name="cinematic_3q" pos="3.8 -4.5 2.6" xyaxes="0.8 0.6 0 -0.2 0.3 0.93"/>
    <camera name="side_profile" pos="4.5 0 1.8" xyaxes="0 1 0 0 0 1"/>
    <camera name="front_action" pos="0 -4.8 1.6" xyaxes="1 0 0 0 0.25 0.96"/>
  </worldbody>

  <actuator>
    <motor name="act_hip_roll_l" joint="hip_roll_l" gear="140"/>
    <motor name="act_hip_pitch_l" joint="hip_pitch_l" gear="160"/>
    <motor name="act_knee_l" joint="knee_l" gear="160"/>
    <motor name="act_ankle_l" joint="ankle_l" gear="80"/>
    <motor name="act_hip_roll_r" joint="hip_roll_r" gear="140"/>
    <motor name="act_hip_pitch_r" joint="hip_pitch_r" gear="160"/>
    <motor name="act_knee_r" joint="knee_r" gear="160"/>
    <motor name="act_ankle_r" joint="ankle_r" gear="80"/>
  </actuator>
</mujoco>
"""


# ──────────────────────────────────────────────
# Named Observation Indices (36-dim vector)
# ──────────────────────────────────────────────
class Obs:
    """Named constants for the 36-dim observation vector layout."""
    Z = 0
    ROLL, PITCH, YAW = 1, 2, 3
    VX, VY, VZ = 4, 5, 6
    WX, WY, WZ = 7, 8, 9
    JOINT_ANGLES = slice(10, 18)   # 8 joint positions
    JOINT_VELS = slice(18, 26)     # 8 joint velocities
    CONTACT_L = 26
    CONTACT_R = 27
    PREV_ACTIONS = slice(28, 36)   # 8 previous action commands
    # Individual joint angle indices
    HIP_ROLL_L = 10
    HIP_PITCH_L = 11
    KNEE_L = 12
    ANKLE_L = 13
    HIP_ROLL_R = 14
    HIP_PITCH_R = 15
    KNEE_R = 16
    ANKLE_R = 17


def quat_to_euler(w, x, y, z):
    """Converts a quaternion to Euler roll, pitch, yaw in radians."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class Walker3DEnv(gym.Env):
    """
    MuJoCo 3D Bipedal Mech Locomotion Environment.
    Follows Gymnasium API standard.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        frame_skip: int = 5,
        max_episode_steps: int = 1000,
        render_mode: str = None,
        camera_name: str = "cinematic_3q",
        render_width: int = 640,
        render_height: int = 480,
        training_mode: bool = True,
        actuator_alpha: float = 0.70,
        action_rate_coef: float = 0.02,
        perturbation_prob: float = 0.015,
        perturbation_mag: float = 25.0,
    ):
        super().__init__()
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.render_width = render_width
        self.render_height = render_height
        self.training_mode = training_mode
        self.actuator_alpha = actuator_alpha
        self.action_rate_coef = action_rate_coef
        self.perturbation_prob = perturbation_prob
        self.perturbation_mag = perturbation_mag

        self.model = mujoco.MjModel.from_xml_string(MECH_3D_ROBOT_XML)
        self.data = mujoco.MjData(self.model)

        self.n_actuators = self.model.nu  # 8
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32
        )

        # Observation space dimension:
        #  - Torso z pos (1)
        #  - Torso roll, pitch, yaw (3)
        #  - Torso linear velocities vx, vy, vz (3)
        #  - Torso angular velocities wx, wy, wz (3)
        #  - Joint angles q (8)
        #  - Joint velocities qdot (8)
        #  - Left/right foot contact flags (2)
        #  - Previous actions (8)
        # Total = 1 + 3 + 3 + 3 + 8 + 8 + 2 + 8 = 36
        obs_dim = 36
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._applied_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._renderer = None

        # Geom and Body IDs for ground contact and perturbation tracking
        self._foot_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_plate_l")
        self._foot_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_plate_r")
        self._toe_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "toe_claw_l")
        self._toe_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "toe_claw_r")
        self._heel_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "heel_spur_l")
        self._heel_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "heel_spur_r")
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._default_floor_friction = float(self.model.geom_friction[self._floor_id, 0]) if self._floor_id >= 0 else 1.4
        self._default_torso_mass = float(self.model.body_mass[self._torso_id]) if self._torso_id >= 0 else 16.0
        # All left/right foot geom IDs for contact detection
        self._left_foot_geoms = {self._foot_l_id, self._toe_l_id, self._heel_l_id}
        self._right_foot_geoms = {self._foot_r_id, self._toe_r_id, self._heel_r_id}

    def _get_contacts(self):
        """Detects if left/right feet (plate, toe, heel) are contacting the floor."""
        contact_l = False
        contact_r = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            pair = {g1, g2}
            if self._floor_id in pair:
                other = g1 if g2 == self._floor_id else g2
                if other in self._left_foot_geoms:
                    contact_l = True
                elif other in self._right_foot_geoms:
                    contact_r = True
        return float(contact_l), float(contact_r)

    def _get_obs(self) -> np.ndarray:
        # Freejoint root qpos: pos (3), quat (4: w, x, y, z)
        z = self.data.qpos[2]
        quat_w, quat_x, quat_y, quat_z = self.data.qpos[3:7]
        roll, pitch, yaw = quat_to_euler(quat_w, quat_x, quat_y, quat_z)

        # Freejoint root qvel: linear vel (3), angular vel (3)
        vx, vy, vz = self.data.qvel[0:3]
        wx, wy, wz = self.data.qvel[3:6]

        # 8 hinge joint positions & velocities
        joint_angles = self.data.qpos[7:15].copy()
        joint_vels = self.data.qvel[6:14].copy()

        contact_l, contact_r = self._get_contacts()

        obs = np.concatenate([
            [z],
            [roll, pitch, yaw],
            [vx, vy, vz],
            [wx, wy, wz],
            joint_angles,
            joint_vels,
            [contact_l, contact_r],
            self._prev_action,
        ]).astype(np.float32)

        # Apply realistic IMU and joint encoder sensor noise during training
        if self.training_mode:
            sensor_noise = np.random.normal(0, 0.003, size=obs.shape).astype(np.float32)
            # Do not corrupt contact flags or previous action memory
            sensor_noise[Obs.CONTACT_L:] = 0.0  # Zero indices 26..35 (contacts + prev_actions)
            obs = obs + sensor_noise

        return obs

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Stance initialization with slight noise for exploration
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = 1.48 # z height above ground
        self.data.qpos[3] = 1.0  # quat w
        self.data.qpos[4:7] = 0.0 # quat x, y, z

        # Add slight random perturbation to joint positions for exploration diversity
        if self.training_mode:
            noise = np.random.uniform(-0.02, 0.02, size=8)
            self.data.qpos[7:15] += noise

        # Domain randomization of physical floor friction and torso mass
        if self.training_mode:
            if self._floor_id >= 0:
                self.model.geom_friction[self._floor_id, 0] = np.random.uniform(1.2, 1.6)
            if self._torso_id >= 0:
                self.model.body_mass[self._torso_id] = self._default_torso_mass * np.random.uniform(0.95, 1.05)
        else:
            if self._floor_id >= 0:
                self.model.geom_friction[self._floor_id, 0] = self._default_floor_friction
            if self._torso_id >= 0:
                self.model.body_mass[self._torso_id] = self._default_torso_mass

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._applied_action = np.zeros(self.n_actuators, dtype=np.float32)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

        # ──────────────────────────────────────────
        # Realistic Actuator Low-Pass Filter Dynamics
        # Models 35-50ms mechanical delay & back-EMF smoothing
        # ──────────────────────────────────────────
        applied_action = (1.0 - self.actuator_alpha) * self._applied_action + self.actuator_alpha * action
        self.data.ctrl[:] = applied_action
        self._applied_action = applied_action.copy()

        # ──────────────────────────────────────────
        # Stochastic Push Perturbation (External Disturbance)
        # ──────────────────────────────────────────
        if self.training_mode and (np.random.rand() < self.perturbation_prob):
            fx = np.random.uniform(-self.perturbation_mag, self.perturbation_mag)
            fy = np.random.uniform(-self.perturbation_mag, self.perturbation_mag)
            fz = np.random.uniform(-0.25 * self.perturbation_mag, 0.25 * self.perturbation_mag)
            tau_z = np.random.uniform(-0.08 * self.perturbation_mag, 0.08 * self.perturbation_mag)
            if self._torso_id >= 0:
                self.data.xfrc_applied[self._torso_id, 0] = fx
                self.data.xfrc_applied[self._torso_id, 1] = fy
                self.data.xfrc_applied[self._torso_id, 2] = fz
                self.data.xfrc_applied[self._torso_id, 5] = tau_z

        # Step physics forward through frame skip
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Clear external perturbation forces
        if self._torso_id >= 0:
            self.data.xfrc_applied[self._torso_id, :] = 0.0

        obs = self._get_obs()

        # Unpack state variables for biomechanical reward calculation
        z = obs[0]
        roll, pitch, yaw = obs[1], obs[2], obs[3]
        vx, vy, vz = obs[4], obs[5], obs[6]
        wx, wy, wz = obs[7], obs[8], obs[9]
        joint_vels = self.data.qvel[6:14]

        # ──────────────────────────────────────────
        # Physical Biomechanical Reward Engineering
        # ──────────────────────────────────────────
        # 1. Forward velocity reward (forward axis is +Y in robot coordinate frame)
        r_forward = 2.5 * vy

        # 2. Healthy survival bonus
        r_alive = 1.0

        # 3. Energy / Control cost penalty (conservation of work)
        r_ctrl = -0.005 * np.sum(np.square(action))

        # 4. Action Rate / Jerk Penalty (suppresses unphysical motor twitching & chatter)
        delta_action = action - self._prev_action
        r_action_rate = -self.action_rate_coef * np.sum(np.square(delta_action))

        # 5. Joint velocity smoothing penalty
        r_joint_vel = -0.0005 * np.sum(np.square(joint_vels))

        # 6. Posture stability & Heading Lock (aligns robot along corridor)
        r_pitch = -0.6 * (pitch ** 2)
        r_roll = -1.2 * (roll ** 2)       # strongly penalize lateral falling
        r_lateral = -0.8 * (vx ** 2)      # penalize drifting sideways off path
        r_yaw = -0.3 * (yaw ** 2) - 0.05 * (wz ** 2)  # align forward heading
        # Asymmetric height penalty: tolerate natural knee flexion squat, penalize jumping (IMP-4)
        dz = z - 1.48
        r_height = -1.2 * (dz ** 2) if dz < 0 else -2.4 * (dz ** 2)

        # 7. Alternating Bipedal Gait Coordination Reward
        # Reuse contacts already computed in obs to avoid redundant physics query
        contact_l = obs[Obs.CONTACT_L]
        contact_r = obs[Obs.CONTACT_R]
        if (contact_l > 0.5 and contact_r < 0.5) or (contact_r > 0.5 and contact_l < 0.5):
            # Single support stance with leg progression
            r_gait = 0.35 * np.clip(vy, 0.0, 1.5)
        elif contact_l > 0.5 and contact_r > 0.5:
            # Double support phase
            r_gait = 0.10
        else:
            # Both feet in air (airborne jump / fall penalty)
            r_gait = -0.30

        # 8. Mechanical Cost of Transport (Power dissipation = torque * speed)
        r_power = -0.0006 * np.sum(np.abs(self.data.ctrl * joint_vels))

        reward = (
            r_forward
            + r_alive
            + r_ctrl
            + r_action_rate
            + r_joint_vel
            + r_pitch
            + r_roll
            + r_lateral
            + r_yaw
            + r_height
            + r_gait
            + r_power
        )

        # ──────────────────────────────────────────
        # Physical Termination Constraints
        # ──────────────────────────────────────────
        # Collapse: Torso height drops below threshold (0.75m indicates fallen)
        fallen = z < 0.75 or z > 2.2
        # Excessive tilt: Robot tipped past recoverable threshold (~45 deg)
        tilted = abs(pitch) > 0.80 or abs(roll) > 0.65

        terminated = bool(fallen or tilted)
        if terminated:
            # Opportunity penalty for premature collapse
            reward -= 5.0

        truncated = bool(self._step_count >= self.max_episode_steps)

        self._prev_action = action.copy()

        info = {
            "vy": float(vy),
            "vx": float(vx),
            "z": float(z),
            "pitch_deg": float(np.degrees(pitch)),
            "roll_deg": float(np.degrees(roll)),
            "step": self._step_count,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )

        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
