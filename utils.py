import math
import numpy as np
import torch
from simulation.simulation import simulate_outer_market_paths
from cva_nn_estimator import compute_cva_labels
from nn_regressor import Regressor
import gc
import numpy as np
from simulation.simulation import simulate_outer_market_paths
from simulation.stats import cva_stats_swaptions 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
    """helper fucntion to simulate market paths + defaults and return (features + CVA labels)"""
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

    features_by_t = np.zeros((num_steps_total + 1, num_paths, 3), dtype=np.float32)
    fws = fixing_window_size
    for step in range(num_steps_total + 1):
        idx = (fws - 1) if step == 0 else (fws + step - 1)
        features_by_t[step, :, 0] = X[idx, 0, :]  # r_t
        features_by_t[step, :, 1] = X[idx, 1, :]  # gamma_t
        # default indicator
        features_by_t[step, :, 2] = ((default_step != -1) & (default_step <= step)).astype(np.float32)
    return features_by_t, labels


def plot_train_val_curves(
    histories,
    titles=None,
    figsize=(14, 4.5),
    dpi=110,
    ylim=None,
    inset=None,
    suptitle=None,
):
    """
    Plot train and val MSE curves epoch-by-epoch, side by side.
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, hist, title in zip(axes, histories, titles or [''] * n):
        epochs = np.arange(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], label='train', color='tab:blue', lw=1.5)
        ax.plot(epochs, hist['val_loss'], label='val (OOS)', color='tab:red', lw=1.5)
        ax.set_xlabel('# epochs')
        ax.set_ylabel('MSE loss (standardized)')
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
        if inset is not None:
            val_arr = np.asarray(hist['val_loss'])
            tail = val_arr[len(val_arr) // 5:]  
            relative_range = (tail.max() - tail.min()) / (tail.mean() + 1e-12)
            if relative_range > 0.05:  # >5% 
                axins = inset_axes(ax, width="40%", height="35%", loc='upper right',
                                   bbox_to_anchor=(0., 0., 0.95, 0.95),
                                   bbox_transform=ax.transAxes)
                axins.plot(epochs, hist['train_loss'], color='tab:blue', lw=1.2)
                axins.plot(epochs, hist['val_loss'], color='tab:red', lw=1.2)
                axins.set_xlim(inset['xlim'])
                axins.set_ylim(inset['ylim'])
                axins.tick_params(labelsize=7)
                axins.grid(alpha=0.3)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.7)

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_cva_density(cva_learned, nmc_dict, dates_steps, dates_years,
                     formulation=None, figsize_per=(5, 4.5)):
    """
    helper function to plot the empirical densities of the learned and NMC cva estimaties at selected pricing dates.
    Args:
        cva_learned: 2D array (num_steps, num_paths)learned CVA predictions.
        nmc_dict:    2D array (num_steps, num_paths) NMC reference CVA.
        dates_steps: list[int] step indices to plot.
        dates_years: list[float] same dates in years (for titles).
        formulation: str, optional label for suptitle.
        figsize_per: tuple, (width, height) per subplot.
    """
    n = len(dates_steps)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), dpi=110)
    if n == 1:
        axes = [axes]

    for ax, t_step, t_year in zip(axes, dates_steps, dates_years):
        learned = cva_learned[t_step]
        nmc = nmc_dict[t_step]
        all_vals = np.concatenate([learned, nmc])
        bins = np.linspace(np.percentile(all_vals, 1),
                           np.percentile(all_vals, 99), 60)
        ax.hist(learned, bins=bins, alpha=0.5, label='Learned',
                color='tab:blue', density=True)
        ax.hist(nmc, bins=bins, alpha=0.5, label='NMC',
                color='tab:orange', density=True)
        ax.set_title(f't = {t_year:.0f}y')
        ax.set_xlabel('CVA')
        ax.set_ylabel('density')
        ax.legend(frameon=False, fontsize=8)

    suptitle = 'Learned vs NMC'
    if formulation:
        suptitle += f' — {formulation} formulation'
    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cva_qq(cva_learned, nmc_dict, dates_steps, dates_years,
                formulation=None, figsize_per=(5, 4.5)):
    """
    QQ-plot of learned vs NMC CVA at selected pricing dates.
    Args:
        cva_learned: 2D array (num_steps, num_paths)learned CVA predictions.
        nmc_dict:    2D array (num_steps, num_paths) NMC reference CVA.
        dates_steps: list[int] step indices to plot.
        dates_years: list[float] same dates in years (for titles).
        formulation: str, optional label for suptitle.
        figsize_per: tuple, (width, height) per subplot.
    """
    n = len(dates_steps)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), dpi=110)
    if n == 1:
        axes = [axes]

    for ax, t_step, t_year in zip(axes, dates_steps, dates_years):
        learned = np.sort(cva_learned[t_step])
        nmc = np.sort(nmc_dict[t_step])
        lo = min(nmc.min(), learned.min())
        hi = max(nmc.max(), learned.max())
        ax.plot([lo, hi], [lo, hi], color='black', lw=0.8)
        ax.scatter(nmc, learned, s=8, alpha=0.4, color='tab:purple')
        ax.set_xlabel('Nested MC CVA')
        ax.set_ylabel('Learned CVA')
        ax.set_title(f't = {t_year:.0f}y')

    suptitle = 'QQ-plot Learned vs NMC'
    if formulation:
        suptitle += f' — {formulation} formulation'
    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cva_profile(cva_learned, nmc_dict=None, dates_steps=None,
                     title=None, figsize=(11, 5), ax=None):
    """
    Plot CVA profile (mean and quantile bands) over time, optionally overlaid with NMC reference points.

    Args:
        cva_learned: 2D array (num_steps+1, num_paths), learned CVA predictions.
        nmc_dict:    dict[int, np.ndarray], optional, {step: (num_paths,)} NMC reference CVA.
        dates_steps: list[int], optional, step indices where NMC points are plotted.
        title:       str, optional, plot title.
        figsize:     tuple, (width, height) of the figure if created standalone.
        ax:          matplotlib Axes, optional; if None, a new figure/axes is created.
    """
    # (label, percentile, color, linestyle, marker)
    stats = [
        ('Mean',  None, '#4C2A85', '-',  'o'),
        ('99%',   99,   '#1F4E79', '--', '^'),
        ('97.5%', 97.5, '#5B9BD5', '--', 's'),
        ('2.5%',  2.5,  '#C97B4A', ':',  'v'),
        ('1%',    1,    '#8B3A3A', ':',  'D'),
    ]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=110)

    # Learned curves
    for label, p, color, ls, _ in stats:
        y = cva_learned.mean(axis=1) if p is None else np.percentile(cva_learned, p, axis=1)
        prefix = 'Learned ' if nmc_dict is not None else ''
        ax.plot(y, color=color, linestyle=ls, linewidth=1.4,
                label=f'{prefix}{label}', alpha=0.9)

    # NMC markers
    if nmc_dict is not None and dates_steps is not None:
        for label, p, color, _, marker in stats:
            xs, ys = [], []
            for t_step in dates_steps:
                nmc = nmc_dict[t_step]
                xs.append(t_step)
                ys.append(nmc.mean() if p is None else np.percentile(nmc, p))
            ax.scatter(xs, ys, color=color, marker=marker, s=22,
                       edgecolor='white', linewidth=0.4, zorder=5,
                       label=f'NMC {label}')

    ax.set_xlabel('pricing time step', fontsize=10)
    ax.set_ylabel('CVA', fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)

    ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ncol = 2 if nmc_dict is not None else 1
    ax.legend(loc='upper right', frameon=False, fontsize=7.5,
              ncol=ncol, columnspacing=0.8, handlelength=2)

    if standalone:
        plt.tight_layout()
        plt.show()


def probe_M_cva(M_cva_candidates, M_LS_probe, indicator, swaptions,
                num_steps_total, num_substeps, dt, dT, fixing_window_size,
                r_0, gamma_0, diff_params, rho, seed=0):
    """Probe the largest M_cva that fits in GPU memory for a fixed M_LS, by trying increasing candidates until one fails"""
    M_cva_max = None
    for M_cva_try in M_cva_candidates:
        try:
            gc.collect()
            rng = np.random.default_rng(seed)
            X, d, ri, gi = simulate_outer_market_paths(
                num_outer_paths=M_cva_try, num_steps_total=num_steps_total,
                num_substeps=num_substeps, dt=dt,
                fixing_window_size=fixing_window_size,
                r_0=r_0, gamma_0=gamma_0, diff_params=diff_params,
                rho=rho, rng=rng,
            )
            cva, ic, rel_err, sec = cva_stats_swaptions(
                M_LS_probe, indicator, swaptions, X, d, ri, gi,
                diff_params, rho, dt, dT, num_substeps, num_steps_total,
                M_cva_try, fixing_window_size, seed=seed,
            )
            print(f"  M_cva={M_cva_try:>5} | OK   | CVA={cva:.2f} | CI=±{ic:.2f} | rel.err={rel_err:.2f}% | time={sec:.2f}s")
            M_cva_max = M_cva_try
            del X, d, ri, gi; gc.collect()
        except Exception as e:
            print(f"  M_cva={M_cva_try:>5} | FAIL | {type(e).__name__}")
            break
    return M_cva_max