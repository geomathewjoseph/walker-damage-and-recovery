"""
Damage module — degrades a trained PPO policy network in controllable ways.

Three modes:
  - neuron_kill: Zero out a fraction of hidden neurons (row+column zeroing)
  - weight_zero: Randomly zero individual scalar weights
  - noise_injection: Add Gaussian noise scaled by layer weight std

All modes operate on the ACTOR network only. The critic (value network) is
left intact — it still provides useful gradient signal during recovery training.
"""
import copy
import random
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from config import DEVICE


def apply_damage(
    model: PPO,
    severity: float,
    mode: str = "neuron_kill",
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> PPO:
    """
    Apply damage to a PPO model's actor network.

    Args:
        model: A trained PPO model (or path to .zip checkpoint).
        severity: Float in [0, 1], fraction of the network to damage.
        mode: One of "neuron_kill", "weight_zero", "noise_injection".
        seed: Random seed for reproducible damage patterns.
        device: Device to load model onto (defaults to model's existing device or global DEVICE).

    Returns:
        A NEW damaged copy of the model. The original is NOT mutated.
    """
    target_device = device or getattr(model, "device", DEVICE)
    if isinstance(model, str):
        model = PPO.load(model, device=target_device)
        target_device = device or getattr(model, "device", DEVICE)

    # Handle accidental argument swapping gracefully
    if isinstance(severity, str) and isinstance(mode, (int, float)):
        mode, severity = severity, float(mode)

    severity = float(severity)
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")

    import io

    if severity == 0.0:
        buffer = io.BytesIO()
        model.save(buffer)
        buffer.seek(0)
        return PPO.load(buffer, env=getattr(model, "env", None), device=target_device)

    # In-memory buffer clone to avoid PyTorch 2.11 deepcopy non-leaf tensor errors
    buffer = io.BytesIO()
    model.save(buffer)
    buffer.seek(0)
    damaged = PPO.load(buffer, env=getattr(model, "env", None), device=target_device)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if mode == "neuron_kill":
        _neuron_kill(damaged, severity)
    elif mode == "weight_zero":
        _weight_zero(damaged, severity)
    elif mode == "noise_injection":
        _noise_injection(damaged, severity)
    else:
        raise ValueError(f"Unknown damage mode: {mode}")

    return damaged


def apply_damage_inplace(
    model: PPO,
    severity: float,
    mode: str = "neuron_kill",
    seed: Optional[int] = None,
) -> None:
    """
    Apply damage directly to a model's weights (no copy).
    Used by the damage-resistant training callback where we want
    to modify the live model during training.
    """
    if isinstance(severity, str) and isinstance(mode, (int, float)):
        mode, severity = severity, float(mode)

    severity = float(severity)
    if severity == 0.0:
        return

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if mode == "neuron_kill":
        _neuron_kill(model, severity)
    elif mode == "weight_zero":
        _weight_zero(model, severity)
    elif mode == "noise_injection":
        _noise_injection(model, severity)
    else:
        raise ValueError(f"Unknown damage mode: {mode}")


def _get_actor_layer_params(model: PPO):
    """
    Extract the actor (policy) network layer parameters.

    SB3 PPO ActorCriticPolicy structure:
      mlp_extractor.policy_net.0  -> Linear(obs_dim, 64)  (hidden layer 1)
      mlp_extractor.policy_net.2  -> Linear(64, 64)        (hidden layer 2)
      action_net                  -> Linear(64, act_dim)    (output layer)

    Returns list of (layer_name, weight_param, bias_param) tuples for hidden layers,
    plus the action_net weight for column-zeroing the final layer.
    """
    policy = model.policy
    layers = []

    # Hidden layers in the policy (actor) network
    policy_net = policy.mlp_extractor.policy_net
    for i, module in enumerate(policy_net):
        if hasattr(module, "weight"):
            layers.append((f"policy_net.{i}", module.weight, module.bias))

    return layers


def _neuron_kill(model: PPO, severity: float):
    """
    Zero out `severity` fraction of hidden neurons.

    For each hidden layer:
      - Select severity% of neurons randomly
      - Zero their output weight row + bias
      - Zero the corresponding input column in the next layer

    This makes the neuron truly dead — it neither outputs nor is read from.
    """
    layers = _get_actor_layer_params(model)
    action_net = model.policy.action_net

    for layer_idx, (name, weight, bias) in enumerate(layers):
        n_neurons = weight.shape[0]
        n_kill = int(round(severity * n_neurons))
        if n_kill <= 0:
            continue
        kill_indices = random.sample(range(n_neurons), min(n_kill, n_neurons))

        with torch.no_grad():
            # Zero output row (this neuron outputs nothing)
            weight.data[kill_indices, :] = 0.0
            if bias is not None:
                bias.data[kill_indices] = 0.0

            # Zero input column of the NEXT layer
            if layer_idx + 1 < len(layers):
                next_weight = layers[layer_idx + 1][1]
                next_weight.data[:, kill_indices] = 0.0
            else:
                # Last hidden layer -> zero input column of action_net
                action_net.weight.data[:, kill_indices] = 0.0

    print(f"  [neuron_kill] Killed ~{severity*100:.0f}% neurons across {len(layers)} hidden layers")


def _weight_zero(model: PPO, severity: float):
    """
    Randomly zero `severity` fraction of individual weights across all actor layers.
    Distributed damage — like diffuse microlesions rather than focal damage.
    """
    params_to_damage = []

    # Collect all actor parameters
    for module in model.policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            params_to_damage.append(module.weight)
            if module.bias is not None:
                params_to_damage.append(module.bias)

    params_to_damage.append(model.policy.action_net.weight)
    if model.policy.action_net.bias is not None:
        params_to_damage.append(model.policy.action_net.bias)

    total_zeroed = 0
    total_params = 0

    with torch.no_grad():
        for param in params_to_damage:
            mask = torch.rand_like(param.data) > severity  # True = keep
            param.data *= mask.float()
            total_zeroed += (~mask).sum().item()
            total_params += param.numel()

    print(f"  [weight_zero] Zeroed {total_zeroed}/{total_params} weights "
          f"({total_zeroed/total_params*100:.1f}%)")


def _noise_injection(model: PPO, severity: float):
    """
    Add Gaussian noise N(0, severity * std(layer_weights)) to each weight.
    At severity=1.0, noise magnitude equals weight magnitude → near-total corruption.
    """
    params_to_damage = []

    for module in model.policy.mlp_extractor.policy_net:
        if hasattr(module, "weight"):
            params_to_damage.append(module.weight)
            if module.bias is not None:
                params_to_damage.append(module.bias)

    params_to_damage.append(model.policy.action_net.weight)
    if model.policy.action_net.bias is not None:
        params_to_damage.append(model.policy.action_net.bias)

    with torch.no_grad():
        for param in params_to_damage:
            std = param.data.std()
            noise = torch.randn_like(param.data) * severity * std
            param.data += noise

    print(f"  [noise_injection] Added noise at severity={severity:.2f}")


# ──────────────────────────────────────────────
# Quick test / CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply damage to a trained model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.zip)")
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--mode", default="neuron_kill",
                        choices=["neuron_kill", "weight_zero", "noise_injection"])
    parser.add_argument("--output", default=None, help="Path to save damaged model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model, device=DEVICE)
    print(f"Applying {args.mode} damage at severity={args.severity}")

    damaged = apply_damage(model, args.severity, mode=args.mode, seed=args.seed)

    output_path = args.output or args.model.replace(".zip", f"_damaged_{args.severity}.zip")
    damaged.save(output_path)
    print(f"Saved damaged model to {output_path}")
