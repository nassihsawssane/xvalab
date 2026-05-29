import math
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
from products.irs.gpu import price_irs


MAX_INNER_LS = 64
N_B = 3
MAX_CALL_DATES = 16


def swaptions_to_arrays(swaptions):
    n = len(swaptions)
    first_reset = np.empty(n, dtype=np.float32)
    reset_freq = np.empty(n, dtype=np.float32)
    num_resets = np.empty(n, dtype=np.int32)
    notional = np.empty(n, dtype=np.float32)
    swap_rate = np.empty(n, dtype=np.float32)
    swap_type = np.empty(n, dtype=np.int32)
    for i, s in enumerate(swaptions):
        first_reset[i] = s['first_reset']
        reset_freq[i] = s['reset_freq']
        num_resets[i] = s['num_resets']
        notional[i] = s['notional']
        swap_rate[i] = s['swap_rate']
        swap_type[i] = s['swap_type']
    return first_reset, reset_freq, num_resets, notional, swap_rate, swap_type


@cuda.jit(device=True, inline=True)
def cholesky_solve_Nb(BtB, Btv, x):
    # Cholesky inline for N_B x N_B SPD system (cf. Appendix D.1 Abbas-Turki et al.)
    L = cuda.local.array((N_B, N_B), dtype=np.float32)
    for i in range(N_B):
        for j in range(i + 1):
            s = BtB[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s < 1e-12:
                    s = 1e-12
                L[i, j] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    # forward solve L y = Btv
    y = cuda.local.array(N_B, dtype=np.float32)
    for i in range(N_B):
        s = Btv[i]
        for k in range(i):
            s -= L[i, k] * y[k]
        y[i] = s / L[i, i]
    # backward solve L^T x = y
    for i in range(N_B - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, N_B):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]


@cuda.jit(device=True, inline=True)
def price_bermudan_swaption_gpu(
    t_node, r_node,
    first_reset, reset_freq, num_resets, swap_rate, swap_type,
    a, b, sigma, dt, M_ls,
    rng_states, thread_id,
    # local arrays passed from caller
    r_at_call, integrated_r_at_call, V, payoff_buf,
    BtB, Btv, lambda_coef,
):
    # build call dates locally: T_n, T_n + dt_call, ..., T_N - dt_call
    # (residual after t_node)
    n_calls = 0
    call_dates = cuda.local.array(MAX_CALL_DATES, dtype=np.float32)
    for i in range(num_resets - 1):
        cd = first_reset + i * reset_freq
        if cd > t_node + 1e-6:
            call_dates[n_calls] = cd
            n_calls += 1
    if n_calls == 0:
        return 0.0

    last_call = call_dates[n_calls - 1]
    sqrt_dt = math.sqrt(dt)

    # simulate M_ls paths of r from (t_node, r_node) to last_call
    # store r and integrated_r at each call date
    for m in range(M_ls):
        r = r_node
        integrated_r = 0.0
        curr_t = t_node
        next_call_idx = 0

        while curr_t < last_call - 1e-6 and next_call_idx < n_calls:
            # advance one dt
            u1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            if u1 < 1e-10:
                u1 = 1e-10
            v1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * v1) * sqrt_dt

            integrated_r += 0.5 * r * dt
            r += a * (b - r) * dt + sigma * z
            integrated_r += 0.5 * r * dt
            curr_t += dt

            # snapshot when we cross a call date
            if next_call_idx < n_calls and curr_t >= call_dates[next_call_idx] - 1e-6:
                r_at_call[next_call_idx * M_ls + m] = r
                integrated_r_at_call[next_call_idx * M_ls + m] = integrated_r
                next_call_idx += 1

    # backward induction
    # init V at last call date: payoff
    k_last = n_calls - 1
    for m in range(M_ls):
        r_m = r_at_call[k_last * M_ls + m]
        swap_val = price_irs(call_dates[k_last], r_m, r_m,
                             first_reset, reset_freq, num_resets,
                             swap_rate, a, b, sigma, False, 1e-6)
        p = swap_type * swap_val
        if p < 0.0:
            p = 0.0
        V[m] = p

    # iterate backward
    for k in range(n_calls - 2, -1, -1):
        # discount V from s_{k+1} to s_k along each path
        for m in range(M_ls):
            ratio = math.exp(-(integrated_r_at_call[(k + 1) * M_ls + m]
                               - integrated_r_at_call[k * M_ls + m]))
            V[m] = ratio * V[m]

        # intrinsic payoff at s_k
        for m in range(M_ls):
            r_m = r_at_call[k * M_ls + m]
            swap_val = price_irs(call_dates[k], r_m, r_m,
                                 first_reset, reset_freq, num_resets,
                                 swap_rate, a, b, sigma, False, 1e-6)
            p = swap_type * swap_val
            if p < 0.0:
                p = 0.0
            payoff_buf[m] = p

        # regression LS on ITM paths only
        # build B^T B (N_B x N_B) and B^T V (N_B), with B columns = [1, x, x^2]
        # but only over ITM mask
        for ii in range(N_B):
            Btv[ii] = 0.0
            for jj in range(N_B):
                BtB[ii, jj] = 0.0

        n_itm = 0
        for m in range(M_ls):
            if payoff_buf[m] > 0.0:
                x = r_at_call[k * M_ls + m]
                # build [1, x, x^2]
                phi0 = 1.0
                phi1 = x
                phi2 = x * x
                # B^T B
                BtB[0, 0] += phi0 * phi0
                BtB[0, 1] += phi0 * phi1
                BtB[0, 2] += phi0 * phi2
                BtB[1, 1] += phi1 * phi1
                BtB[1, 2] += phi1 * phi2
                BtB[2, 2] += phi2 * phi2
                # B^T V
                Btv[0] += phi0 * V[m]
                Btv[1] += phi1 * V[m]
                Btv[2] += phi2 * V[m]
                n_itm += 1
        # symmetrize
        BtB[1, 0] = BtB[0, 1]
        BtB[2, 0] = BtB[0, 2]
        BtB[2, 1] = BtB[1, 2]

        if n_itm > N_B:
            cholesky_solve_Nb(BtB, Btv, lambda_coef)
            # exercise rule on ITM paths
            for m in range(M_ls):
                if payoff_buf[m] > 0.0:
                    x = r_at_call[k * M_ls + m]
                    c_hat = lambda_coef[0] + lambda_coef[1] * x + lambda_coef[2] * x * x
                    if payoff_buf[m] > c_hat:
                        V[m] = payoff_buf[m]

    # final discount from s_0 to t_node
    total = 0.0
    for m in range(M_ls):
        discount_to_t = math.exp(-integrated_r_at_call[m])  # k=0 column
        total += discount_to_t * V[m]
    return total / M_ls

