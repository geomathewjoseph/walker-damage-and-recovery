# Experimental Protocol & Benchmark Guide

**Author**: **Geo Mathew Joseph**  
This document specifies the experimental designs, testing protocols, quantitative metrics, and reproducibility guidelines for both the **2D Planar Locomotive Suite** (`BipedalWalker-v3`) and the **3D Multi-Body Robotics Suite** (`Walker3DEnv` in MuJoCo 3.13).

---

## 1. 2D Experimental Matrix Protocol (`BipedalWalker-v3`)

The 2D experimental suite evaluates the agent across five damage severity tiers under the **Neuron Kill** structural ablation mode:

| Severity ID | Severity ($s$) | Mathematical Description | Hypothesized Motor Pathology | Expected Recovery Difficulty |
| :--- | :---: | :--- | :--- | :--- |
| **Control** | $0.0$ | Identity copy (no modification) | Pristine baseline walking | None required ($100\%$ baseline) |
| **Mild** | $0.1$ | $10\%$ neurons killed ($6/64$ per layer) | Minor limp, asymmetric stride | Rapid recovery ($< 50\text{k}$ steps) |
| **Moderate** | $0.3$ | $30\%$ neurons killed ($19/64$ per layer) | Significant stride degradation, stumble risk | Steady recovery ($100\text{k} - 250\text{k}$ steps) |
| **Severe** | $0.5$ | $50\%$ neurons killed ($32/64$ per layer) | Knee buckling, acute loss of balance, collapse | Intensive rehabilitation ($250\text{k} - 500\text{k}$ steps) |
| **Critical** | $0.7$ | $70\%$ neurons killed ($44/64$ per layer) | Total motor failure, immediate fall | Partial compensation; capacity ceiling |

### 1.1 Empirical 2D Benchmark Results

From `logs/experiment_results.json`:

| Severity ($s$) | Healthy Baseline ($R_{\text{healthy}}$) | Acute Damaged ($R_{\text{damaged}}$) | Post-Recovery ($R_{\text{recovered}}$) | Recovery % | Functional Restoration Index ($\text{FRI}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | +308.4 | +308.4 | +308.4 | 100.0% | 1.000 |
| **0.1** | +308.4 | +226.6 | +312.8 | 101.4% | 1.054 |
| **0.3** | +308.4 | +107.8 | +313.3 | 101.6% | 1.024 |
| **0.5** | +308.4 | -108.3 | +314.1 | 101.9% | 1.014 |
| **0.7** | +308.4 | -110.1 | +249.6 | 80.9% | 0.860 |

---

## 2. 3D Multi-Body Robotics Matrix Protocol (`Walker3DEnv`)

The 3D bipedal mech is evaluated in high-fidelity 3D MuJoCo multi-body physics, subjected to structural neuro-trauma on its 128-unit MLP policy network.

### 2.1 3D Experimental Matrix Design

| Severity ($s$) | Ablated Neurons | Mathematical Operation | Biomechanical Pathology | Recovery Training Horizon |
|:---:|:---:|:---|:---|:---:|
| **0.0** | $0 / 128$ | Control (pristine policy) | Normal dynamic bipedal gait (+283.7) | N/A |
| **0.1** | $12 / 128$ | Row/column zeroing | Slight lateral yaw oscillation, stride compensation | 100,000 steps |
| **0.3** | $38 / 128$ | Row/column zeroing | Stumbling gait, reduced stance time on left leg | 100,000 steps |
| **0.5** | $64 / 128$ | Row/column zeroing | Acute knee buckling, forward deceleration, collapse | 100,000 steps |

### 2.2 Empirical 3D Benchmark Results

From `logs/experiment_3d_results.json`:

| Severity ($s$) | Healthy Baseline ($R$) | Acute Post-Lesion ($R$) | Post-Recovery ($R$) | Recovery % | Healthy Speed | Recovered Speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | +283.7 | +283.7 | +283.7 | **100.0%** | 1.01 m/s | 1.01 m/s |
| **0.1** | +283.7 | +292.9 | +301.9 | **106.4%** | 1.01 m/s | 1.33 m/s |
| **0.3** | +283.7 | +298.1 | +297.0 | **104.7%** | 1.01 m/s | 1.33 m/s |
| **0.5** | +283.7 | +161.9 | +311.9 | **109.9%** | 1.01 m/s | 1.26 m/s |

**Key Finding**:
Under severe 50% neuron ablation, the damaged agent suffers a catastrophic performance drop to $+161.9$. When retrained with an intact critic head, the policy reorganizes its remaining 64 functional neurons to achieve $+311.9$ reward (**109.9% restoration**), with forward velocity increasing to $1.26\text{ m/s}$.

---

## 3. Evaluation Metrics

### 3.1 Deterministic Evaluation Reward ($R_{\text{eval}}$)
Measured using greedy actions $a_t = \mu(s_t)$ without Gaussian exploration noise:
$$R_{\text{eval}} = \sum_{t=0}^{T} R_t$$

### 3.2 Recovery Percentage ($\text{Rec}\%$)
$$\text{Rec}\%(s) = \frac{R_{\text{recovered}}(s)}{R_{\text{healthy}}} \times 100\%$$

### 3.3 Functional Restoration Index ($\text{FRI}$)
$$\text{FRI}(s) = \frac{R_{\text{recovered}}(s) - R_{\text{damaged}}(s)}{R_{\text{healthy}} - R_{\text{damaged}}(s)}$$
An $\text{FRI}$ of $1.0$ represents complete restoration back to pre-injury competence.

---

## 4. How to Execute Experiments

### 4.1 Run 2D Automated Experiment Matrix
```bash
python orchestrator.py --severities 0.0 0.1 0.3 0.5 0.7 --recovery-steps 500000
```

### 4.2 Run 3D Automated Experiment Matrix
```bash
python orchestrator_3d.py --healthy checkpoints/walker3d_healthy.zip --severities 0.0 0.1 0.3 0.5 --recovery-steps 100000 --mode neuron_kill
```

### 4.3 Alternative Damage Modalities
You can test alternative damage modalities (`weight_zero` or `noise_injection`) on either model:
```bash
# Test 40% diffuse synaptic disconnection on 2D model
python damage.py --model checkpoints/healthy.zip --severity 0.4 --mode weight_zero

# Test 30% Gaussian noise perturbation on 3D model
python orchestrator_3d.py --healthy checkpoints/walker3d_healthy.zip --severities 0.3 --mode noise_injection
```

### 4.4 Generate Publication-Grade Analytics
```bash
python visualize.py --log-dir logs --plot-dir plots
```

---

## 5. Evolutionary Curriculum & Zero-Shot Neuro-Resilience

Recorded in `logs/evolution_history.json`:

| Generation | Perturbation Mode | Severity | Training Steps | Post-Generation Reward |
| :---: | :---: | :---: | :---: | :---: |
| **Gen 1** | `neuron_kill` | 10% | 75,000 | **+220.2** |
| **Gen 2** | `weight_zero` | 15% | 75,000 | **+111.8** |
| **Gen 3** | `noise_injection` | 20% | 90,000 | **+174.0** |
| **Gen 4** | Consolidation (`none`) | 0% | 120,000 | **+317.2** |

### Zero-Shot Resilience Benchmark
- Under immediate 10% structural ablation without fine-tuning:
  - Standard Agent: Drops from $+307.5 \to +226.6$ ($-80.9$ points).
  - Evolved Agent: Retains **$+317.0$ ($0.0$ drop)** with $0\%$ fall rate.
