import time
import numpy as np
from simulation.simulation import simulate_nested_cva, simulate_nested_cva_swaptions_gpu


def cva_stats_swaps(t_i_idx, n_inner, indicator, X, default_step, irs,
                   diff_params, rho, dt, num_substeps, num_steps_total,
                   num_outer_paths, fixing_window_size, dT, seed=42):
    t0 = time.time()
    nested_cva, _, defaulted = simulate_nested_cva(
        t_i_idx, n_inner, indicator, X, default_step, irs,
        diff_params, rho, dt, num_substeps, num_steps_total,
        num_outer_paths, fixing_window_size, dT, seed=seed,
    )
    elapsed = time.time() - t0
    active = ~defaulted
    cva_k = nested_cva[active]
    cva = cva_k.mean()
    ic = 1.96 * cva_k.std(ddof=1) / np.sqrt(active.sum())
    return cva, ic, 100 * ic / cva, elapsed


def cva_stats_swaptions(n_inner, indicator, swaptions, X, default_step,
                       rate_integral_path, gamma_integral_path,
                       diff_params, rho, dt, dT, num_substeps, num_steps_total,
                       num_outer_paths, fixing_window_size, seed=42):
    t0 = time.time()
    nested_cva, _ = simulate_nested_cva_swaptions_gpu(
        swaptions, X, default_step, rate_integral_path, gamma_integral_path,
        diff_params, rho, dt, dT, num_substeps, num_steps_total,
        num_outer_paths, n_inner, fixing_window_size,
        indicator_in_cva=indicator, seed=seed,
    )
    elapsed = time.time() - t0
    cva = nested_cva.mean()
    ic = 1.96 * nested_cva.std(ddof=1) / np.sqrt(num_outer_paths)
    rel = 100 * ic / cva if cva > 0 else float('nan')
    return cva, ic, rel, elapsed