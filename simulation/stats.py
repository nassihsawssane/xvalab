import time
import numpy as np
from simulation.simulation import simulate_nested_cva, simulate_nested_cva_swaptions_gpu, simulate_outer_market_paths
import matplotlib.pyplot as plt


def cva_stats_swaps(t_i_idx, n_inner, indicator, X, default_step, irs,
                   diff_params, rho, dt, num_substeps, num_steps_total,
                   num_outer_paths, fixing_window_size, dT, seed=42):
    """run nested CVA on IRS book.
    returns : CVA, 95% half-width, relative IC %, elapsed time."""
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
    """run nested CVA on bermudan swaption book.
    returns : CVA, 95% half-width, relative IC %, elapsed time."""
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


def get_contributions(indicator, label, M_cva_star, M_mtm_star, seed,
                       num_steps_total, num_substeps, dt, fixing_window_size,
                       r_0, gamma_0, diff_params, rho, irs, dT, t_i_idx=0):
    """returns per-path CVA contributions for a given formulation."""
    M_cva = M_cva_star[label]
    M_mtm = M_mtm_star[label]
    
    rng = np.random.default_rng(seed)
    X, default_step, _, _ = simulate_outer_market_paths(
        num_outer_paths=M_cva, num_steps_total=num_steps_total,
        num_substeps=num_substeps, dt=dt,
        fixing_window_size=fixing_window_size,
        r_0=r_0, gamma_0=gamma_0,
        diff_params=diff_params, rho=rho, rng=rng,
    )
    nested_cva, _, defaulted = simulate_nested_cva(
        t_i_idx, M_mtm, indicator, X, default_step, irs,
        diff_params, rho, dt, num_substeps, num_steps_total,
        M_cva, fixing_window_size, dT, seed=seed,
    )
    return nested_cva[~defaulted], M_cva, M_mtm


def plot_contributions(M_cva_star, M_mtm_star, seed,
                       num_steps_total, num_substeps, dt, fixing_window_size,
                       r_0, gamma_0, diff_params, rho, irs, dT,
                       t_i_idx=0, plot_type="hist"):
    """
    plot per-path CVA contributions for both formulations indicator + intensity.
    plot_type: "hist dist" and "running_mean"
    """
    cases = [(False, "intensity", "#2E86AB"), (True, "indicator", "#E63946")]
    
    if plot_type == "hist":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (indicator, label, color) in zip(axes, cases):
            contributions, M_cva, M_mtm = get_contributions(
                indicator, label, M_cva_star, M_mtm_star, seed,
                num_steps_total, num_substeps, dt, fixing_window_size,
                r_0, gamma_0, diff_params, rho, irs, dT, t_i_idx,
            )
            ax.hist(contributions, bins=40, color=color, alpha=0.75, edgecolor='white')
            ax.axvline(contributions.mean(), color='black', linestyle='--', linewidth=1.5,
                       label=f'mean = {contributions.mean():.0f}')
            ax.set_title(f"{label.capitalize()} (M_cva={M_cva}, M_mtm={M_mtm})", fontsize=11)
            ax.set_xlabel(r"per-path contribution $\xi_j$")
            ax.set_ylabel("Count")
            ax.legend(frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
    
    elif plot_type == "running_mean":
        fig, ax = plt.subplots(figsize=(8, 5))
        for indicator, label, color in cases:
            contributions, _, _ = get_contributions(
                indicator, label, M_cva_star, M_mtm_star, seed,
                num_steps_total, num_substeps, dt, fixing_window_size,
                r_0, gamma_0, diff_params, rho, irs, dT, t_i_idx,
            )
            n = len(contributions)
            running = np.cumsum(contributions) / np.arange(1, n + 1)
            ax.plot(np.arange(1, n + 1), running, label=label, color=color, lw=1.2)
        ax.set_xscale('log')
        ax.set_xlabel("num of outer paths")
        ax.set_ylabel("CVA value")
        ax.set_title("convergence of CVA estimate vs number of outer paths")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    plt.tight_layout()
    plt.show()