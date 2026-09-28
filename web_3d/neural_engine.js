/**
 * NEURAL PLASTICITY & ONLINE REINFORCEMENT LEARNING ENGINE
 * Mathematical Multilayer Perceptron (Actor-Critic) running directly in JavaScript.
 *
 * Features:
 *   - Real 12-input -> [32, 32] -> 4-torque MLP Policy Network + 1-unit Critic Head
 *   - Real dynamic forward pass & analytical backpropagation
 *   - Real Temporal Difference (TD) error & Policy Gradient learning:
 *       delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
 *       theta <- theta + alpha * delta_t * grad_theta log pi(a_t|s_t)
 *   - True synaptic ablation (neuron kill / weight zeroing / noise)
 *   - Autonomous closed-loop trauma detector & online neuroplastic adaptation
 *   - Dynamic multi-joint physical integration (Euler solver)
 *
 * Author: Geo Mathew Joseph
 */

(function (window) {
  "use strict";

  // Utility math helpers
  function randn(std = 1.0) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
  }

  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  // ──────────────────────────────────────────────
  // 1. Multilayer Perceptron Actor-Critic Policy
  // ──────────────────────────────────────────────
  class NeuralPolicy {
    constructor() {
      this.inputDim = 12;
      this.h1Dim = 32;
      this.h2Dim = 32;
      this.actionDim = 4; // hipL, kneeL, hipR, kneeR

      // Weight matrices & biases
      this.W1 = this.createMatrix(this.h1Dim, this.inputDim, Math.sqrt(2.0 / this.inputDim));
      this.b1 = new Float32Array(this.h1Dim);

      this.W2 = this.createMatrix(this.h2Dim, this.h1Dim, Math.sqrt(2.0 / this.h1Dim));
      this.b2 = new Float32Array(this.h2Dim);

      // Actor policy head
      this.Wpi = this.createMatrix(this.actionDim, this.h2Dim, 0.1);
      this.bpi = new Float32Array(this.actionDim);

      // Critic value head
      this.Wv = new Float32Array(this.h2Dim);
      for (let i = 0; i < this.h2Dim; i++) this.Wv[i] = randn(0.1);
      this.bv = 0.0;

      // Lesion masks (1.0 = functional, 0.0 = ablated)
      this.maskH1 = new Float32Array(this.h1Dim).fill(1.0);
      this.maskH2 = new Float32Array(this.h2Dim).fill(1.0);
      this.maskActuators = new Float32Array(this.actionDim).fill(1.0);

      // Seed calibrated locomotion gait weights
      this.seedCalibratedWeights();
    }

    createMatrix(rows, cols, std) {
      const mat = new Array(rows);
      for (let r = 0; r < rows; r++) {
        mat[r] = new Float32Array(cols);
        for (let c = 0; c < cols; c++) {
          mat[r][c] = randn(std);
        }
      }
      return mat;
    }

    seedCalibratedWeights() {
      // Initializes rhythmic anti-phase bipedal walking synergies
      for (let r = 0; r < this.h1Dim; r++) {
        for (let c = 0; c < this.inputDim; c++) {
          const phase = (r / this.h1Dim) * Math.PI * 2;
          this.W1[r][c] += Math.sin(phase + c) * 0.25;
        }
      }
      for (let a = 0; a < this.actionDim; a++) {
        for (let c = 0; c < this.h2Dim; c++) {
          const sign = (a < 2) ? 1.0 : -1.0;
          this.Wpi[a][c] = Math.sin((c / this.h2Dim) * Math.PI * 2 + (a % 2) * 1.5) * 0.45 * sign;
        }
      }
    }

    applyLesion(mode, severity) {
      // Reset all masks
      this.maskH1.fill(1.0);
      this.maskH2.fill(1.0);
      this.maskActuators.fill(1.0);

      const sev = clamp(severity, 0.0, 1.0);
      if (sev === 0.0) return;

      if (mode === "neuron_kill") {
        // Targeted structural ablation of left motor control pathways
        const h1Kill = Math.floor(this.h1Dim * sev);
        const h2Kill = Math.floor(this.h2Dim * sev);
        for (let i = 0; i < h1Kill; i++) this.maskH1[i] = 0.0;
        for (let i = 0; i < h2Kill; i++) this.maskH2[i] = 0.0;
        if (sev >= 0.5) {
          this.maskActuators[1] = Math.max(0.05, 1.0 - sev * 1.3); // Severe left knee paralysis
        }
      } else if (mode === "weight_zero") {
        // Diffuse synaptic disconnection
        for (let r = 0; r < this.h1Dim; r++) {
          if (Math.random() < sev) this.maskH1[r] = 0.0;
        }
        for (let r = 0; r < this.h2Dim; r++) {
          if (Math.random() < sev) this.maskH2[r] = 0.0;
        }
      } else if (mode === "noise") {
        // Gaussian synaptic degradation
        const noiseStd = sev * 0.45;
        for (let r = 0; r < this.h2Dim; r++) {
          for (let c = 0; c < this.h1Dim; c++) {
            this.W2[r][c] += randn(noiseStd);
          }
        }
      }
    }

    forward(stateVec) {
      // 1. Layer 1 (Affine + Tanh with Lesion Mask)
      const h1 = new Float32Array(this.h1Dim);
      for (let r = 0; r < this.h1Dim; r++) {
        if (this.maskH1[r] === 0.0) {
          h1[r] = 0.0;
          continue;
        }
        let sum = this.b1[r];
        const row = this.W1[r];
        for (let c = 0; c < this.inputDim; c++) {
          sum += row[c] * stateVec[c];
        }
        h1[r] = Math.tanh(sum);
      }

      // 2. Layer 2 (Affine + Tanh with Lesion Mask)
      const h2 = new Float32Array(this.h2Dim);
      for (let r = 0; r < this.h2Dim; r++) {
        if (this.maskH2[r] === 0.0) {
          h2[r] = 0.0;
          continue;
        }
        let sum = this.b2[r];
        const row = this.W2[r];
        for (let c = 0; c < this.h1Dim; c++) {
          sum += row[c] * h1[c];
        }
        h2[r] = Math.tanh(sum);
      }

      // 3. Policy Head (Torques bounded [-1, +1])
      const actions = new Float32Array(this.actionDim);
      for (let a = 0; a < this.actionDim; a++) {
        let sum = this.bpi[a];
        const row = this.Wpi[a];
        for (let c = 0; c < this.h2Dim; c++) {
          sum += row[c] * h2[c];
        }
        // Apply actuator health mask
        actions[a] = Math.tanh(sum) * this.maskActuators[a];
      }

      // 4. Value Head (Critic Scalar)
      let value = this.bv;
      for (let c = 0; c < this.h2Dim; c++) {
        value += this.Wv[c] * h2[c];
      }

      return { actions, value, h1, h2 };
    }
  }

  // ──────────────────────────────────────────────
  // 2. Dynamic Physical Integrator (Biped Kinetics)
  // ──────────────────────────────────────────────
  class DynamicIntegrator {
    constructor() {
      this.reset();
    }

    reset() {
      // Joint angles (rad)
      this.q = [0.0, 0.1, 0.0, 0.1]; // hipL, kneeL, hipR, kneeR
      this.dq = [0.0, 0.0, 0.0, 0.0];

      // Torso pose
      this.torsoY = 2.4;
      this.pitch = 0.0; // rad
      this.roll = 0.0;  // rad
      this.dpitch = 0.0;
      this.droll = 0.0;

      // Ground velocity
      this.vx = 1.35;
      this.vy = 0.0;

      // Contacts
      this.contacts = [true, false]; // L, R
      this.walkPhase = 0.0;
      this.strideCount = 0;
    }

    step(torques, dt = 0.02, targetVx = 1.4) {
      // Joint dynamics: acceleration = (torque - damping * dq - spring_return) / inertia
      const jointInertia = 0.45;
      const damping = 0.25;

      for (let i = 0; i < 4; i++) {
        const netTorque = torques[i] - damping * this.dq[i] - 0.15 * this.q[i];
        const ddq = netTorque / jointInertia;
        this.dq[i] += ddq * dt;
        this.q[i] += this.dq[i] * dt;

        // Kinematic joint bounds
        if (i % 2 === 1) {
          // Knees (cannot hyperextend backwards)
          this.q[i] = clamp(this.q[i], 0.05, 1.45);
        } else {
          // Hips
          this.q[i] = clamp(this.q[i], -0.9, 0.9);
        }
      }

      // Gait phase progression driven by forward movement
      const cadence = 3.2; // rad/s
      this.walkPhase += cadence * dt * (this.vx / Math.max(0.2, targetVx));

      // Foot contacts: anti-phase stance and swing
      this.contacts[0] = Math.sin(this.walkPhase) >= -0.15;
      this.contacts[1] = Math.sin(this.walkPhase + Math.PI) >= -0.15;

      // Forward velocity derived from leg torque power & stance traction
      const leftStancePower = this.contacts[0] ? (this.q[0] * 0.8 + this.q[1] * 0.5) : 0.0;
      const rightStancePower = this.contacts[1] ? (this.q[2] * 0.8 + this.q[3] * 0.5) : 0.0;
      const propulsion = Math.abs(leftStancePower) + Math.abs(rightStancePower);

      // Smooth forward acceleration
      const targetSpeed = clamp(propulsion * 0.95 + 0.35, 0.1, targetVx * 1.3);
      this.vx += (targetSpeed - this.vx) * 0.12;

      // Torso attitude dynamics: torque asymmetry induces chassis roll and pitch
      const torqueAsymmetry = (torques[0] + torques[1]) - (torques[2] + torques[3]);
      this.dpitch += (-torques[0] * 0.15 - torques[2] * 0.15 - 4.5 * this.pitch - 0.8 * this.dpitch) * dt;
      this.pitch += this.dpitch * dt;

      this.droll += (torqueAsymmetry * 0.22 - 3.8 * this.roll - 0.7 * this.droll) * dt;
      this.roll += this.droll * dt;

      this.torsoY = 2.4 - Math.abs(this.pitch) * 0.35 + Math.sin(this.walkPhase * 2) * 0.04;

      return {
        vx: this.vx,
        pitch: this.pitch,
        roll: this.roll,
        contacts: this.contacts,
        q: this.q,
      };
    }

    getStateVector(targetVx = 1.4) {
      return new Float32Array([
        this.vx,
        this.vy,
        this.pitch,
        this.roll,
        this.dpitch,
        this.droll,
        this.q[0],
        this.q[1],
        this.q[2],
        this.q[3],
        this.contacts[0] ? 1.0 : 0.0,
        this.contacts[1] ? 1.0 : 0.0,
      ]);
    }
  }

  // ──────────────────────────────────────────────
  // 3. Online Reinforcement Learning Plasticity
  // ──────────────────────────────────────────────
  class OnlinePlasticityEngine {
    constructor() {
      this.policy = new NeuralPolicy();
      this.physics = new DynamicIntegrator();

      // Learning hyperparameters
      this.lrActor = 0.0035;
      this.lrCritic = 0.008;
      this.gamma = 0.96;
      this.sigma = 0.18; // Exploration noise

      // Telemetry & Learning History
      this.totalUpdates = 0;
      this.recentLosses = [];
      this.recentRewards = [];
      this.weightDeltaNorm = 0.0;
      this.rollingSymmetry = 98.0;

      // Autonomous Anomaly Detector State
      this.isAutoLearningActive = true;
      this.autoHealEnabled = true;
      this.continuousImproveEnabled = true;
      this.curriculumStage = "BALANCED_WALK"; // "BALANCED_WALK", "SPEED_SCALING", "ENERGY_ECONOMY"
      this.targetVelocity = 1.48;

      this.traumaCounter = 0;
      this.isAdapting = false;
      this.adaptationStep = 0;
    }

    step(dt = 0.02) {
      // 1. Get current physical state vector
      const s0 = this.physics.getStateVector(this.targetVelocity);

      // 2. Forward pass through policy
      const fwd0 = this.policy.forward(s0);

      // 3. Add exploration variance during online learning
      const actions = new Float32Array(4);
      for (let i = 0; i < 4; i++) {
        const noise = this.isAutoLearningActive ? randn(this.sigma) : 0.0;
        actions[i] = clamp(fwd0.actions[i] + noise, -1.0, 1.0);
      }

      // 4. Physical environment integration
      const physState = this.physics.step(actions, dt, this.targetVelocity);

      // 5. Next state vector
      const s1 = this.physics.getStateVector(this.targetVelocity);
      const fwd1 = this.policy.forward(s1);

      // 6. Compute instantaneous reward
      const torqueCost = 0.04 * (actions[0] * actions[0] + actions[1] * actions[1] + actions[2] * actions[2] + actions[3] * actions[3]);
      const attitudeCost = 1.2 * Math.abs(physState.pitch) + 1.6 * Math.abs(physState.roll);
      const velocityReward = 2.4 * Math.min(physState.vx, this.targetVelocity);
      const uprightBonus = 0.8;
      const stepReward = velocityReward - torqueCost - attitudeCost + uprightBonus;

      // 7. Temporal Difference Error: delta = r + gamma * V(s1) - V(s0)
      const tdError = stepReward + this.gamma * fwd1.value - fwd0.value;

      // 8. Rolling Gait Symmetry
      const leftAmp = Math.abs(physState.q[0]) + Math.abs(physState.q[1]);
      const rightAmp = Math.abs(physState.q[2]) + Math.abs(physState.q[3]);
      const symRatio = Math.min(leftAmp, rightAmp) / (Math.max(leftAmp, rightAmp) + 1e-4);
      this.rollingSymmetry = this.rollingSymmetry * 0.95 + (symRatio * 100) * 0.05;

      // 9. Autonomous Anomaly Detection & Self-Healing Trigger
      if (this.autoHealEnabled) {
        const isDegraded = (physState.vx < 0.85) || (this.rollingSymmetry < 75.0) || (Math.abs(physState.pitch) > 0.35);
        if (isDegraded) {
          this.traumaCounter++;
          if (this.traumaCounter > 12 && !this.isAdapting) {
            this.isAdapting = true;
            this.adaptationStep = 0;
            console.log("[NeuralEngine] Autonomous Anomaly Detected -> Initiating Neuroplastic Adaptation");
          }
        } else {
          this.traumaCounter = Math.max(0, this.traumaCounter - 1);
          if (this.isAdapting && this.rollingSymmetry > 88.0 && physState.vx > 1.2) {
            this.isAdapting = false;
            console.log("[NeuralEngine] Neuroplastic Adaptation Complete -> Gait Restabilized");
          }
        }
      }

      // 10. Perform Analytical Online Gradient Update (if learning active)
      if (this.isAutoLearningActive || this.isAdapting) {
        this.performGradientStep(s0, fwd0, actions, tdError);
      }

      // Telemetry updates
      this.recentLosses.push(Math.abs(tdError));
      if (this.recentLosses.length > 50) this.recentLosses.shift();

      this.recentRewards.push(stepReward);
      if (this.recentRewards.length > 50) this.recentRewards.shift();

      return {
        actions,
        vx: physState.vx,
        reward: stepReward,
        pitch: physState.pitch,
        roll: physState.roll,
        symmetry: this.rollingSymmetry,
        tdError,
        isAdapting: this.isAdapting,
        weightDeltaNorm: this.weightDeltaNorm,
        h1: fwd0.h1,
        h2: fwd0.h2,
      };
    }

    performGradientStep(s0, fwd0, actions, tdError) {
      this.totalUpdates++;
      let deltaNormSum = 0.0;

      // Adaptive learning rate scaling when adapting
      const effectiveLr = this.isAdapting ? this.lrActor * 1.8 : this.lrActor;

      // Policy gradient: grad_theta log pi(a|s) = (a - mu) / sigma^2
      const gradLogPi = new Float32Array(4);
      for (let a = 0; a < 4; a++) {
        gradLogPi[a] = ((actions[a] - fwd0.actions[a]) / (this.sigma * this.sigma)) * this.policy.maskActuators[a];
      }

      // Update Policy Head: Delta W_pi[a][c] = lr * tdError * gradLogPi[a] * h2[c]
      for (let a = 0; a < 4; a++) {
        const factor = effectiveLr * tdError * gradLogPi[a];
        for (let c = 0; c < this.policy.h2Dim; c++) {
          if (this.policy.maskH2[c] > 0.0) {
            const dW = factor * fwd0.h2[c];
            this.policy.Wpi[a][c] += dW;
            deltaNormSum += dW * dW;
          }
        }
        this.policy.bpi[a] += factor;
      }

      // Update Critic Head: Delta W_v[c] = lrCritic * tdError * h2[c]
      for (let c = 0; c < this.policy.h2Dim; c++) {
        if (this.policy.maskH2[c] > 0.0) {
          const dW = this.lrCritic * tdError * fwd0.h2[c];
          this.policy.Wv[c] += dW;
        }
      }
      this.policy.bv += this.lrCritic * tdError;

      // Backprop into Hidden Layer 2: Compensate intact neurons
      const backH2 = new Float32Array(this.policy.h2Dim);
      for (let c = 0; c < this.policy.h2Dim; c++) {
        if (this.policy.maskH2[c] === 0.0) continue;
        let grad = 0.0;
        for (let a = 0; a < 4; a++) {
          grad += gradLogPi[a] * this.policy.Wpi[a][c];
        }
        const dtanh = 1.0 - fwd0.h2[c] * fwd0.h2[c];
        backH2[c] = grad * dtanh;
      }

      // Update Layer 2 Weights
      for (let r = 0; r < this.policy.h2Dim; r++) {
        if (this.policy.maskH2[r] === 0.0) continue;
        const factor = effectiveLr * tdError * backH2[r] * 0.35;
        for (let c = 0; c < this.policy.h1Dim; c++) {
          if (this.policy.maskH1[c] > 0.0) {
            const dW = factor * fwd0.h1[c];
            this.policy.W2[r][c] += dW;
            deltaNormSum += dW * dW;
          }
        }
      }

      this.weightDeltaNorm = Math.sqrt(deltaNormSum);
    }

    setLesion(mode, severity) {
      this.policy.applyLesion(mode, severity);
    }
  }

  // Export engine to global window
  window.NeuralPlasticityEngine = OnlinePlasticityEngine;
})(window);
