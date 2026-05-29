import math
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
from numba.cuda.random import create_xoroshiro128p_states
from products.irs.gpu import price_irs
from products.swaption.gpu import (
    swaptions_to_arrays,
    price_bermudan_swaption_gpu
)


def compute_fixing_window_size(irs, dt, dT):
    """sliding-window size needed to store past short rates at reset dates
    cf 'Market and Credit Model in Continuous Time' Appendix B in [2] in references.
    """
    max_reset_freq = max(t['reset_freq'] for t in irs)
    return max(int((max_reset_freq + dt) / dT), 1)


def simulate_outer_market_paths(
    num_outer_paths,
    num_steps_total,
    num_substeps,
    dt,
    fixing_window_size,
    r_0,
    gamma_0,
    diff_params,
    rho,
    rng
):
    """
    Forward MC simulation/diffusion of market factors per outer path (r, def intensity gamma...)
    Path arrays are stored on the pricing grid of step dT.
    """
    a, b, sigma, k, theta, xi = diff_params  # Vasicek(a,b,sigma) / CIR(k,theta,xi)
    rho_compl = math.sqrt(1 - rho * rho)
    sqrt_dt = math.sqrt(dt)

    # X[t, factor, outer]/ factor 0 = r/ factor 1 = gamma
    X = np.empty((num_steps_total + fixing_window_size, 2, num_outer_paths), dtype=np.float32)
    X[:fixing_window_size, 0, :] = r_0
    X[:fixing_window_size, 1, :] = gamma_0

    # arrays of int_0^t(r)ds and int_0^t(gamma)du stored on the pricing grid
    rate_integral_path = np.zeros((num_steps_total + fixing_window_size, num_outer_paths), dtype=np.float32)
    gamma_integral_path = np.zeros((num_steps_total + fixing_window_size, num_outer_paths), dtype=np.float32)

    # current running value per outer path
    r = np.full(num_outer_paths, r_0, dtype=np.float32)
    gamma = np.full(num_outer_paths, gamma_0, dtype=np.float32)
    rate_integral = np.zeros(num_outer_paths, dtype=np.float32)   
    gamma_integral = np.zeros(num_outer_paths, dtype=np.float32)

    # Cox
    E = -np.log(rng.random(num_outer_paths).astype(np.float32))
    default_step = np.full(num_outer_paths, -1, dtype=np.int32)

    for step in range(num_steps_total):  # pricing step dT
        for substep in range(num_substeps):  # diffusion fine sub_step dt
            u1 = rng.random(num_outer_paths).astype(np.float32)
            v1 = rng.random(num_outer_paths).astype(np.float32)
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * v1) * sqrt_dt
            u2 = rng.random(num_outer_paths).astype(np.float32)
            v2 = rng.random(num_outer_paths).astype(np.float32)
            z2 = np.sqrt(-2 * np.log(u2)) * np.cos(2 * np.pi * v2) * sqrt_dt
            dW_r = z1
            dW_gamma = rho * z1 + rho_compl * z2  # corr/uncorr brownian increments

            # Vasiceck step for r with trapezoidal accumulation of int(r)ds
            rate_integral += 0.5 * r * dt
            r += a * (b - r) * dt + sigma * dW_r  # Euler-Maruyama scheme
            rate_integral += 0.5 * r * dt

            # CIR step for gamma + trapez accumulation
            gamma_pos = np.maximum(gamma, 0.0)  # full truncat to avoid sqrt of negative values
            gamma += k * (theta - gamma_pos) * dt + xi * np.sqrt(gamma_pos) * dW_gamma  # Euler
            gamma_integral += 0.5 * gamma_pos * dt
            gamma_integral += 0.5 * np.maximum(gamma, 0.0) * dt

        # t = (step+1) * dT
        # snapshot the current state into the path arrays
        X[fixing_window_size + step, 0, :] = r
        X[fixing_window_size + step, 1, :] = gamma
        rate_integral_path[fixing_window_size + step] = rate_integral
        gamma_integral_path[fixing_window_size + step] = gamma_integral

        is_defaulted = (gamma_integral > E) & (default_step == -1)  # Cox
        default_step[is_defaulted] = step

    return X, default_step, rate_integral_path, gamma_integral_path


def irs_to_arrays(irs):
    """
    convert list of irs' dicts to arrays as upcoming gpu kernel can't read py dicts
    """
    n = len(irs)
    first_reset = np.empty(n, dtype=np.float32)
    reset_freq = np.empty(n, dtype=np.float32)
    num_resets = np.empty(n, dtype=np.int32)
    notional = np.empty(n, dtype=np.float32)
    swap_rate = np.empty(n, dtype=np.float32)
    for i, t in enumerate(irs):
        first_reset[i] = t['first_reset']
        reset_freq[i] = t['reset_freq']
        num_resets[i] = t['num_resets']
        notional[i] = t['notional']
        swap_rate[i] = t['swap_rate']
    return first_reset, reset_freq, num_resets, notional, swap_rate


# fixing window param fixed per-thread in local memory as its size must be fixed at compile time
# val 8 is enough for reset_freq = 0.5 (=5)
# to be increased increase for larger reset frequencies
MAX_FIXING_WINDOW = 8


@cuda.jit
def nested_cva_kernel(
    t_i_idx, num_steps, t_i,
    X, defaulted,
    a, b, sigma, k, theta, xi, rho, rho_compl,
    fr, rf, nr, notio, sr, n_trades,
    dt, num_substeps, num_outer_paths, num_inner_paths,
    fixing_window_size,
    rng_states,
    nested_cva, squared_nested_cva,
    indicator_in_cva,
):
    outer_path = cuda.grid(1)  # thread = 1 outer path
    if outer_path >= num_outer_paths:
        return

    if defaulted[outer_path]:
        nested_cva[outer_path] = 0.0
        squared_nested_cva[outer_path] = 0.0
        return

    sqrt_dt = math.sqrt(dt)
    fixing_rates = cuda.local.array(MAX_FIXING_WINDOW, dtype=np.float32)

    sum_payoff = 0.0
    squared_payoff_sum = 0.0

    for inner_path in range(num_inner_paths):

        # state at t_i + init
        r = X[t_i_idx + fixing_window_size - 1, 0, outer_path]
        gamma = X[t_i_idx + fixing_window_size - 1, 1, outer_path]
        for j in range(fixing_window_size):
            fixing_rates[fixing_window_size - j - 1] = X[t_i_idx - 1 - j, 0, outer_path]
        rate_integral = 0.0
        gamma_integral = 0.0
        inner_payoff = 0.0
        curr_t = t_i

        # indicator: exponential threshold re-simulated per inner path
        if indicator_in_cva:
            uE = xoroshiro128p_uniform_float32(rng_states, outer_path)
            if uE < 1e-10:
                uE = 1e-10
            E_inner = -math.log(uE)
            defaulted_inner = False

        for step in range(t_i_idx, t_i_idx + num_steps):

            # intensity based formulation/ pricing at left node t_l
            if not indicator_in_cva:
                rate_integral_left = rate_integral      
                gamma_integral_prev = gamma_integral    

                mtm_left = 0.0
                for trade in range(n_trades):
                    first_reset = fr[trade]
                    reset_freq = rf[trade]
                    num_resets = nr[trade]
                    notional = notio[trade]
                    swap_rate = sr[trade]

                    maturity = first_reset + (num_resets - 1) * reset_freq
                    # curr_t = t_l
                    if maturity + 0.1 * dt < curr_t:
                        continue

                    if curr_t > first_reset - 0.1 * dt:
                        m = int((curr_t - first_reset - (num_substeps - 1) * dt) / reset_freq)
                        m = int((curr_t - first_reset - m * reset_freq + dt) / (num_substeps * dt))
                        m = fixing_window_size - m
                    else:
                        m = fixing_window_size - 1

                    price = price_irs(
                        curr_t, r, fixing_rates[m],
                        first_reset, reset_freq, num_resets,
                        swap_rate, a, b, sigma, False, 1e-6)
                    mtm_left += notional * price

                discount_left = math.exp(-rate_integral_left)

            # diffusion
            curr_t += dt * num_substeps             

            for substep in range(num_substeps):
                # Box-Muller GPU
                u1 = xoroshiro128p_uniform_float32(rng_states, outer_path)
                if u1 < 1e-10:
                    u1 = 1e-10
                v1 = xoroshiro128p_uniform_float32(rng_states, outer_path)
                z1 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * v1) * sqrt_dt
                u2 = xoroshiro128p_uniform_float32(rng_states, outer_path)
                if u2 < 1e-10:
                    u2 = 1e-10
                v2 = xoroshiro128p_uniform_float32(rng_states, outer_path)
                z2 = math.sqrt(-2.0 * math.log(u2)) * math.cos(2.0 * math.pi * v2) * sqrt_dt

                dW_r = z1
                dW_gamma = rho * z1 + rho_compl * z2

                rate_integral += 0.5 * r * dt
                r += a * (b - r) * dt + sigma * dW_r
                rate_integral += 0.5 * r * dt

                gamma_pos = max(gamma, 0.0)
                gamma += k * (theta - gamma_pos) * dt + xi * math.sqrt(gamma_pos) * dW_gamma
                gamma_integral += 0.5 * gamma_pos * dt
                if gamma > 0.0:
                    gamma_integral += 0.5 * gamma * dt

            # shift fixing window left, add new r (state at t_l+1)
            for j in range(fixing_window_size - 1):
                fixing_rates[j] = fixing_rates[j + 1]
            fixing_rates[fixing_window_size - 1] = r

            if indicator_in_cva:
                # Indicator based formualtion pricing at right node t_l+1
                mtm = 0.0
                for trade in range(n_trades):
                    first_reset = fr[trade]
                    reset_freq = rf[trade]
                    num_resets = nr[trade]
                    notional = notio[trade]
                    swap_rate = sr[trade]

                    maturity = first_reset + (num_resets - 1) * reset_freq
                    if maturity + 0.1 * dt < curr_t:
                        continue

                    if curr_t > first_reset - 0.1 * dt:
                        m = int((curr_t - first_reset - (num_substeps - 1) * dt) / reset_freq)
                        m = int((curr_t - first_reset - m * reset_freq + dt) / (num_substeps * dt))
                        m = fixing_window_size - m
                    else:
                        m = fixing_window_size - 1

                    price = price_irs(
                        curr_t, r, fixing_rates[m],
                        first_reset, reset_freq, num_resets,
                        swap_rate, a, b, sigma, False, 1e-6)
                    mtm += notional * price

                discount = math.exp(-rate_integral)
                if (not defaulted_inner) and (gamma_integral > E_inner):
                    defaulted_inner = True
                    if mtm > 0.0:
                        inner_payoff = discount * mtm
            else:
                default_prob = math.exp(-gamma_integral_prev) - math.exp(-gamma_integral)
                increment = discount_left * mtm_left * default_prob
                if increment > 0.0:
                    inner_payoff += increment

        sum_payoff += inner_payoff
        squared_payoff_sum += inner_payoff * inner_payoff

    nested_cva[outer_path] = sum_payoff / num_inner_paths
    squared_nested_cva[outer_path] = squared_payoff_sum / num_inner_paths


def simulate_nested_cva(t_i_idx, n_inner, indicator, X, default_step, irs,
                   diff_params, rho, dt, num_substeps, num_steps_total,
                   num_outer_paths, fixing_window_size, dT, seed=42):
    
    # prep data for gpu
    a, b, sigma, k, theta, xi = diff_params
    rho_compl = math.sqrt(1 - rho*rho)
    t_i = t_i_idx * dT
    num_steps_nested = num_steps_total - t_i_idx
    defaulted = (default_step != -1) & (default_step < t_i_idx)
    fr, rf, nr, no, sr = irs_to_arrays(irs)

    threads = 128
    blocks = (num_outer_paths + threads - 1) // threads

    nested_cva = cuda.device_array(num_outer_paths, dtype=np.float32)
    squared = cuda.device_array(num_outer_paths, dtype=np.float32)
    rng_states = create_xoroshiro128p_states(num_outer_paths, seed=seed)

    nested_cva_kernel[blocks, threads](
        t_i_idx, num_steps_nested, t_i,
        cuda.to_device(X), cuda.to_device(defaulted),
        a, b, sigma, k, theta, xi, rho, rho_compl,
        cuda.to_device(fr), cuda.to_device(rf), cuda.to_device(nr),
        cuda.to_device(no), cuda.to_device(sr), len(irs),
        dt, num_substeps, num_outer_paths, n_inner,
        fixing_window_size, rng_states,
        nested_cva, squared, indicator,
    )
    cuda.synchronize()
    return nested_cva.copy_to_host(), squared.copy_to_host(), defaulted  # gpu to cpu


MAX_INNER_LS = 64
N_B = 3
MAX_CALL_DATES = 16


@cuda.jit
def nested_cva_swaptions_kernel(
    X, default_step, rate_integral_path, gamma_integral_path,
    a, b, sigma, k, theta, xi, rho, rho_compl,
    fr, rf, nr, notio, sr, st, n_swaptions,
    dt, dT, num_substeps, num_outer_paths, num_steps_total,
    num_inner_ls_paths, fixing_window_size,
    rng_states,
    nested_cva, squared_nested_cva,
    indicator_in_cva,
):
    outer_path = cuda.grid(1)
    if outer_path >= num_outer_paths:
        return

    # local arrays for LS
    r_at_call = cuda.local.array(1024, dtype=np.float32)
    integrated_r_at_call = cuda.local.array(1024, dtype=np.float32)
    V = cuda.local.array(MAX_INNER_LS, dtype=np.float32)
    payoff_buf = cuda.local.array(MAX_INNER_LS, dtype=np.float32)
    BtB = cuda.local.array((N_B, N_B), dtype=np.float32)
    Btv = cuda.local.array(N_B, dtype=np.float32)
    lambda_coef = cuda.local.array(N_B, dtype=np.float32)

    payoff = 0.0

    if indicator_in_cva:
        if default_step[outer_path] == -1:
            nested_cva[outer_path] = 0.0
            squared_nested_cva[outer_path] = 0.0
            return

        idx_default = default_step[outer_path]
        t_lp1 = (idx_default + 1) * dT
        r_at_default = X[fixing_window_size + idx_default, 0, outer_path]
        integrated_r_at_default = rate_integral_path[fixing_window_size + idx_default, outer_path]

        mtm = 0.0
        for swpt in range(n_swaptions):
            price = price_bermudan_swaption_gpu(
                t_lp1, r_at_default,
                fr[swpt], rf[swpt], nr[swpt], sr[swpt], st[swpt],
                a, b, sigma, dt, num_inner_ls_paths,
                rng_states, outer_path,
                r_at_call, integrated_r_at_call, V, payoff_buf,
                BtB, Btv, lambda_coef,
            )
            mtm += notio[swpt] * st[swpt] * price

        if mtm > 0.0:
            payoff = math.exp(-integrated_r_at_default) * mtm

    else:
        for step in range(num_steps_total):
            t_l = step * dT
            if step > 0:
                r_at_l = X[fixing_window_size + step - 1, 0, outer_path]
                integrated_r_at_l = rate_integral_path[fixing_window_size + step - 1, outer_path]
                gamma_int_at_l = gamma_integral_path[fixing_window_size + step - 1, outer_path]
            else:
                r_at_l = X[fixing_window_size - 1, 0, outer_path]
                integrated_r_at_l = 0.0
                gamma_int_at_l = 0.0
            gamma_int_at_lp1 = gamma_integral_path[fixing_window_size + step, outer_path]
            default_prob = math.exp(-gamma_int_at_l) - math.exp(-gamma_int_at_lp1)
            if default_prob <= 0.0:
                continue

            mtm = 0.0
            for swpt in range(n_swaptions):
                price = price_bermudan_swaption_gpu(
                    t_l, r_at_l,
                    fr[swpt], rf[swpt], nr[swpt], sr[swpt], st[swpt],
                    a, b, sigma, dt, num_inner_ls_paths,
                    rng_states, outer_path,
                    r_at_call, integrated_r_at_call, V, payoff_buf,
                    BtB, Btv, lambda_coef,
                )
                mtm += notio[swpt] * st[swpt] * price

            if mtm > 0.0:
                payoff += math.exp(-integrated_r_at_l) * mtm * default_prob

    nested_cva[outer_path] = payoff
    squared_nested_cva[outer_path] = payoff * payoff


def simulate_nested_cva_swaptions_gpu(
    swaptions, X, default_step, rate_integral_path, gamma_integral_path,
    diff_params, rho, dt, dT, num_substeps, num_steps_total,
    num_outer_paths, num_inner_ls_paths, fixing_window_size,
    indicator_in_cva=False, seed=42,
):
    a, b, sigma, k, theta, xi = diff_params
    rho_compl = math.sqrt(1 - rho * rho)
    fr, rf, nr, no, sr, st = swaptions_to_arrays(swaptions)

    threads = 128
    blocks = (num_outer_paths + threads - 1) // threads

    nested_cva = cuda.device_array(num_outer_paths, dtype=np.float32)
    squared = cuda.device_array(num_outer_paths, dtype=np.float32)
    rng_states = create_xoroshiro128p_states(num_outer_paths, seed=seed)

    nested_cva_swaptions_kernel[blocks, threads](
        cuda.to_device(X), cuda.to_device(default_step),
        cuda.to_device(rate_integral_path), cuda.to_device(gamma_integral_path),
        a, b, sigma, k, theta, xi, rho, rho_compl,
        cuda.to_device(fr), cuda.to_device(rf), cuda.to_device(nr),
        cuda.to_device(no), cuda.to_device(sr), cuda.to_device(st), len(swaptions),
        dt, dT, num_substeps, num_outer_paths, num_steps_total,
        num_inner_ls_paths, fixing_window_size,
        rng_states,
        nested_cva, squared, indicator_in_cva,
    )
    cuda.synchronize()
    return nested_cva.copy_to_host(), squared.copy_to_host()