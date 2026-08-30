"""World-model training loop (proposal Day 12), single-step MSE loss with
gradient clipping, AdamW, and a constant-state baseline comparison."""
import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from f2s.world_model.dataset import TransitionDataset
from f2s.world_model.model import ResidualDynamicsMLP, WorldModelEnsemble


def constant_state_mse(states: np.ndarray, next_states: np.ndarray) -> float:
    """MSE_constant = ||x_t - x_{t+1}||^2, the proposal's Day 12.3 sanity
    baseline: the learned model must beat "predict no change"."""
    return float(np.mean(np.sum((states - next_states) ** 2, axis=-1)))


def train_world_model(
    train_states: np.ndarray,
    train_actions: np.ndarray,
    train_next_states: np.ndarray,
    val_states: np.ndarray,
    val_actions: np.ndarray,
    val_next_states: np.ndarray,
    output_dir: str,
    hidden_dim: int = 256,
    ensemble_size: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    device: Optional[str] = None,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]

    model = WorldModelEnsemble(state_dim, action_dim, hidden_dim=hidden_dim, ensemble_size=ensemble_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TransitionDataset(train_states, train_actions, train_next_states),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_ds = TransitionDataset(val_states, val_actions, val_next_states)

    log_rows = []
    best_val_mse = float("inf")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for state, action, next_state in train_loader:
            state, action, next_state = state.to(device), action.to(device), next_state.to(device)
            optimizer.zero_grad()
            if ensemble_size == 1:
                pred = model(state, action)
                loss = nn.functional.mse_loss(pred, next_state)
            else:
                preds = model.forward_members(state, action)  # (E, B, D)
                loss = nn.functional.mse_loss(preds, next_state.unsqueeze(0).expand_as(preds))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vs = val_ds.states.to(device)
            va = val_ds.actions.to(device)
            vns = val_ds.next_states.to(device)
            val_pred = model(vs, va) if len(vs) > 0 else None
            val_mse = float(nn.functional.mse_loss(val_pred, vns).item()) if val_pred is not None else float("nan")

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")
        log_rows.append(dict(epoch=epoch, train_mse=train_mse, val_mse=val_mse))
        print(f"epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pt"))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    with open(os.path.join(output_dir, "train_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        writer.writerows(log_rows)

    const_mse = constant_state_mse(val_states, val_next_states) if len(val_states) > 0 else float("nan")
    result = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        ensemble_size=ensemble_size,
        epochs=epochs,
        best_val_mse=best_val_mse,
        constant_state_val_mse=const_mse,
        beats_constant_baseline=bool(best_val_mse < const_mse) if len(val_states) > 0 else None,
        training_time_seconds=time.time() - t_start,
    )
    return model, result
