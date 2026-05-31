import numpy as np
from numba import cuda
from nn_regressor import Regressor
from products.irs.gpu import price_irs
from simulation.simulation import irs_to_arrays
from tqdm.auto import tqdm


@cuda.jit
def compute_mtm_kernel(
    X,
    first_resets, reset_freqs, n_resets, notionals, swap_rates, n_trades,
    a, b, sigma,
    dt, dT, num_substeps, fixing_window_size,
    num_steps_total, num_outer_paths,
    mtm_out,
):
    o = cuda.grid(1)
    if o >= num_outer_paths:
        return

    for step in range(num_steps_total + 1):
        if step == 0:
            idx = fixing_window_size - 1
        else:
            idx = fixing_window_size + step - 1
        curr_t = step * dT
        r_o = X[idx, 0, o]
        m_total = 0.0

        for trade in range(n_trades):
            first_reset = first_resets[trade]
            reset_freq = reset_freqs[trade]
            n_reset = n_resets[trade]
            notional = notionals[trade]
            swap_rate = swap_rates[trade]
            maturity = first_reset + (n_reset - 1) * reset_freq
            if maturity + 0.1 * dt < curr_t:
                continue
            if curr_t > first_reset - 0.1 * dt:
                m_idx = int((curr_t - first_reset - (num_substeps - 1) * dt) / reset_freq)
                m_idx = int((curr_t - first_reset - m_idx * reset_freq + dt) / (num_substeps * dt))
                m_idx = fixing_window_size - m_idx
            else:
                m_idx = fixing_window_size - 1
            row_fixing = idx - (fixing_window_size - 1) + m_idx
            r_last_fixing = X[row_fixing, 0, o]
            price = price_irs(
                curr_t, r_o, r_last_fixing,
                first_reset, reset_freq, n_reset,
                swap_rate, a, b, sigma, False, 1e-6,
            )
            m_total += notional * price

        mtm_out[step, o] = m_total


def compute_cva_labels(
    X,
    rate_integral_path,
    gamma_integral_path,
    default_step,
    irs,
    diff_params,
    dt, dT,
    fixing_window_size,
    num_substeps,
    num_steps_total,
    num_outer_paths,
    formulation='intensity',
):
    """Compute pathwise CVA labels for NN regression, for all time steps and paths using GPU-accelerated MtM computation."""
    a, b, sigma, _, _, _ = diff_params

    # prep + device transfer for GPU
    first_resets, reset_freqs, n_resets, notionals, swap_rates = irs_to_arrays(irs)
    X_gpu = cuda.to_device(X)
    first_resets_gpu = cuda.to_device(first_resets)
    reset_freqs_gpu = cuda.to_device(reset_freqs)
    n_resets_gpu = cuda.to_device(n_resets)
    notionals_gpu = cuda.to_device(notionals)
    swap_rates_gpu = cuda.to_device(swap_rates)
    mtm_gpu = cuda.device_array((num_steps_total + 1, num_outer_paths), dtype=np.float32)

    threads = 128
    # ceil (paths/threads)
    blocks = (num_outer_paths + threads - 1) // threads

    compute_mtm_kernel[blocks, threads](
        X_gpu,
        first_resets_gpu, reset_freqs_gpu, n_resets_gpu, notionals_gpu, swap_rates_gpu, len(irs),
        a, b, sigma,
        dt, dT, num_substeps, fixing_window_size,
        num_steps_total, num_outer_paths,
        mtm_gpu,
    )
    cuda.synchronize()
    mtm = mtm_gpu.copy_to_host()  # device to host once MtM cube is built

    # computing labels for all t
    labels = np.zeros((num_steps_total + 1, num_outer_paths), dtype=np.float32)
    cumulative_cva = np.zeros(num_outer_paths, dtype=np.float32)

    if formulation == 'intensity':
        # left-Euler scheme where MtM evaluated at t_l
        for step in range(num_steps_total - 1, -1, -1):
            idx_now = (fixing_window_size - 1) if step == 0 else (fixing_window_size + step - 1)
            idx_next = fixing_window_size + step
            rate_integral_diff = rate_integral_path[idx_next] - rate_integral_path[idx_now]
            gamma_integral_diff = gamma_integral_path[idx_next] - gamma_integral_path[idx_now]
            df_r_d = np.exp(-(rate_integral_diff + gamma_integral_diff)) 
            mtm_left = np.maximum(mtm[step], 0.0)  # pos exposure 
            incr = mtm_left * (1.0 - np.exp(-gamma_integral_diff))
            already_def = (default_step != -1) & (default_step < step) # survival mask
            cumulative_cva = np.where(already_def, 0.0, cumulative_cva * df_r_d + incr).astype(np.float32)
            labels[step] = cumulative_cva

    else:  # indicator based formulation
        for step in range(num_steps_total - 1, -1, -1):
            idx_now = (fixing_window_size - 1) if step == 0 else (fixing_window_size + step - 1)
            idx_next = fixing_window_size + step
            rate_integral_diff = rate_integral_path[idx_next] - rate_integral_path[idx_now]
            df_r = np.exp(-rate_integral_diff) # beta_{t_{l+1}} / beta_{t_l} 
            m_next = np.maximum(mtm[step + 1], 0.0)
            is_default_here = (default_step == step) & (default_step != -1)
            incr = np.where(is_default_here, df_r * m_next, 0.0).astype(np.float32)
            already_def = (default_step != -1) & (default_step < step)
            cumulative_cva = np.where(already_def, 0.0, cumulative_cva * df_r + incr).astype(np.float32)
            labels[step] = cumulative_cva

    return labels


class LearnedCVA:
    def __init__(
        self,
        num_steps_total,
        input_dim,
        hidden_units=32,
        num_hidden_layers=2,
        lr=1e-3,
        num_epochs=100,
        batch_size=256,
        val_frac=0.1,
        early_stop=True,
        patience=20,
        device='cuda',
        seed=0,
        verbose=False,
        min_label_var=1e-7,  
    ):
        self.num_steps_total = num_steps_total
        self.min_label_var = min_label_var

        self.regressor = Regressor(
            input_dim=input_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            val_frac=val_frac,
            early_stop=early_stop,
            patience=patience,
            device=device,
            seed=seed,
            verbose=verbose,
        )

        self.states_by_t = [None] * (num_steps_total + 1)

    def fit(self, features_by_t, labels, formulation=None):
        title = 'Run backward NN training (CVA'
        if formulation is not None:
            title += f', {formulation}'
            title += ')'
        time_bar = tqdm(range(self.num_steps_total, -1, -1), desc=title)
        for t in time_bar:
            X_t = features_by_t[t]
            y_t = labels[t]  # cva labels at date t
            std_t = float(y_t.std())

            if std_t < self.min_label_var:  # below this label variance we skip nn regression and use mean
                self.states_by_t[t] = ('const', float(y_t.mean()))
                time_bar.set_postfix(t=t, mode='const')
                continue

            if t == 0:
                self.states_by_t[t] = ('const', float(y_t.mean()))
                time_bar.set_postfix(t=t, mode='const')
                continue

            time_bar.set_postfix(t=t, mode='NN')
            self.regressor.train(X_t, y_t)
            self.states_by_t[t] = ('nn', self.regressor.get_state())

    def predict(self, features_by_t):
        num_t = self.num_steps_total + 1
        num_outer = features_by_t.shape[1]
        out = np.zeros((num_t, num_outer), dtype=np.float32)

        for t in range(num_t):
            state = self.states_by_t[t]
            if state is None:
                continue
            kind, payload = state
            if kind == 'const':
                out[t, :] = payload
            else:
                self.regressor.set_state(payload)
                out[t, :] = self.regressor.predict(features_by_t[t])
        return out
