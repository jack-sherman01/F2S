"""Value-function training: binary cross-entropy on (state, eventual
success) pairs, same optimizer/schedule conventions as
f2s.world_model.train.train_world_model for consistency (AdamW, gradient
clipping, best-checkpoint-by-val-loss)."""
import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from f2s.value.dataset import compute_normalization_stats
from f2s.value.model import ValueFunction


class ValueDataset(Dataset):
    def __init__(self, states: np.ndarray, labels: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        states_norm = (states - mu) / sigma
        self.states = torch.from_numpy(states_norm).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]


def train_value_function(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    val_states: np.ndarray,
    val_labels: np.ndarray,
    output_dir: str,
    hidden_dim: int = 256,
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
    # normalization stats from TRAIN only, per f2s's established discipline
    # (f2s.failure.features.standardize / f2s.world_model.dataset) of never
    # fitting normalization on data the model will be evaluated against.
    mu, sigma = compute_normalization_stats(train_states)
    np.savez(os.path.join(output_dir, "normalization_stats.npz"), mu=mu, sigma=sigma)

    model = ValueFunction(state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        ValueDataset(train_states, train_labels, mu, sigma),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_ds = ValueDataset(val_states, val_labels, mu, sigma)

    log_rows = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for state, label in train_loader:
            state, label = state.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(state)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            vs = val_ds.states.to(device)
            vl = val_ds.labels.to(device)
            if len(vs) > 0:
                val_logits = model(vs)
                val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, vl).item())
                val_pred = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = float((val_pred == vl).float().mean().item())
            else:
                val_loss, val_acc = float("nan"), float("nan")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        log_rows.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc))
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    with open(os.path.join(output_dir, "train_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)

    # sanity baseline: predicting the train-set base rate for everyone
    base_rate = float(train_labels.mean()) if len(train_labels) > 0 else float("nan")
    baseline_val_loss = float(nn.functional.binary_cross_entropy(
        torch.full_like(val_ds.labels, base_rate).clamp(1e-6, 1 - 1e-6), val_ds.labels
    ).item()) if len(val_states) > 0 else float("nan")

    result = dict(
        state_dim=state_dim, hidden_dim=hidden_dim, epochs=epochs,
        n_train=int(train_states.shape[0]), n_val=int(val_states.shape[0]),
        train_base_rate=base_rate,
        best_val_loss=best_val_loss,
        constant_base_rate_val_loss=baseline_val_loss,
        beats_base_rate_baseline=bool(best_val_loss < baseline_val_loss) if len(val_states) > 0 else None,
        training_time_seconds=time.time() - t_start,
    )
    return model, result
