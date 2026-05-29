import math
import numpy as np
import torch
from simulation.simulation import simulate_outer_market_paths
from cva_nn_estimator import compute_cva_labels
from nn_regressor import Regressor


def box_muller(rng, sqrt_dt):
    u = max(rng.random(dtype=np.float32), 1e-10)
    v = rng.random(dtype=np.float32)
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v) * sqrt_dt


def vec_box_muller(rng, sqrt_dt, size):
    """a vectorized Box-Muller that generates a 'size' i.i.d. N(0, dt) samples"""
    u = np.maximum(rng.random(size, dtype=np.float32), 1e-10)
    v = rng.random(size, dtype=np.float32)
    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v) * sqrt_dt


def simulate_and_label(
    num_paths, seed, formulation,
    num_steps_total, num_substeps, dt, dT,
    fixing_window_size,
    r_0, gamma_0, diff_params, rho,
    irs,
):
    rng = np.random.default_rng(seed)
    X, default_step, rate_int, gamma_int = simulate_outer_market_paths(
        num_outer_paths=num_paths,
        num_steps_total=num_steps_total,
        num_substeps=num_substeps,
        dt=dt,
        fixing_window_size=fixing_window_size,
        r_0=r_0, gamma_0=gamma_0,
        diff_params=diff_params,
        rho=rho, rng=rng,
    )
    labels = compute_cva_labels(
        X, rate_int, gamma_int, default_step,
        irs, diff_params, dt, dT,
        fixing_window_size, num_substeps,
        num_steps_total, num_paths,
        formulation=formulation,
    )
    features_by_t = np.zeros((num_steps_total + 1, num_paths, 2), dtype=np.float32)
    fws = fixing_window_size
    for step in range(num_steps_total + 1):
        idx = (fws - 1) if step == 0 else (fws + step - 1)
        features_by_t[step, :, 0] = X[idx, 0, :]
        features_by_t[step, :, 1] = X[idx, 1, :]
    return features_by_t, labels


def train_with_oos_tracking(
    X_tr, y_tr, X_oos, y_oos,
    hidden_units, num_hidden_layers,
    num_epochs=128, batch_size=512, lr=1e-3,
    device='cuda', seed=0,
):
    """Train un NN sans early stopping, track MSE IS et OOS (standardisées) à chaque epoch."""
    reg = Regressor(
        input_dim=X_tr.shape[1],
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        lr=lr, num_epochs=1, batch_size=batch_size,
        val_frac=0.0, early_stop=False,
        device=device, seed=seed, verbose=False,
    )

    X_tr_t  = reg.to_tensor(X_tr)
    y_tr_t  = reg.to_tensor(y_tr).view(-1, 1)
    X_oos_t = reg.to_tensor(X_oos)
    y_oos_t = reg.to_tensor(y_oos).view(-1, 1)

    reg.standardize_fit(X_tr_t, y_tr_t)
    X_tr_s  = (X_tr_t  - reg.x_mean) / reg.x_std
    y_tr_s  = y_tr_t / reg.y_std
    X_oos_s = (X_oos_t - reg.x_mean) / reg.x_std

    var_y_tr  = float(y_tr_t.var())
    var_y_oos = float(y_oos_t.var())

    n_tr = X_tr_s.shape[0]
    is_hist, oos_hist = [], []

    for epoch in range(num_epochs):
        reg.model.train()
        perm = torch.randperm(n_tr, device=reg.device)
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            reg.opt.zero_grad()
            loss = reg.loss_fn(reg.model(X_tr_s[idx]), y_tr_s[idx])
            loss.backward()
            reg.opt.step()

        reg.model.eval()
        with torch.no_grad():
            pred_tr  = reg.model(X_tr_s)  * reg.y_std
            pred_oos = reg.model(X_oos_s) * reg.y_std
            mse_tr  = ((pred_tr  - y_tr_t ) ** 2).mean().item()
            mse_oos = ((pred_oos - y_oos_t) ** 2).mean().item()

        is_hist.append(mse_tr / var_y_tr)
        oos_hist.append(mse_oos / var_y_oos)

    return np.array(is_hist), np.array(oos_hist)