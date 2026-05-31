"""
Bermudan swaption pricer (CPU reference implementation)
"""
import numpy as np
from utils import vec_box_muller
from products.irs.cpu import price_irs


def get_call_dates(swaption, curr_t=0.0, eps=1e-6):
    """build the exercise grid of the bermudan swaption residual to curr_t
    by convention in our context a sawption = T_N nc T_n ie non-callable/lock-out before T_n
    and exercisable at every reset of the underlying swap except T_N (swap maturity)"""
    first_reset = swaption['first_reset']
    reset_freq = swaption['reset_freq']
    num_resets = swaption['num_resets']
    all_calls = [first_reset + i * reset_freq for i in range(num_resets - 1)]
    # keeping only dates strictly in the future wrt curr_t
    return [c for c in all_calls if c > curr_t + eps]


def simulate_short_rate_paths(M, T_final, dt, r0, a, b, sigma, rng):
    """simulate M paths of Vasicek short rate from t=0 to last date of exercise
    returns:
        r_paths : r_t at each diffusion grid point
        integrated_r_paths : int_0^t r_s ds at each diffusion grid point
    """
    n_steps = round(T_final / dt)
    sqrt_dt = np.sqrt(dt)

    r_paths = np.empty((n_steps + 1, M), dtype=np.float32)
    integrated_r_paths = np.empty((n_steps + 1, M), dtype=np.float32)
    r = np.full(M, r0, dtype=np.float32)
    rate_integral = np.zeros(M, dtype=np.float32)
    r_paths[0] = r
    integrated_r_paths[0] = rate_integral

    for step in range(n_steps):
        dW = vec_box_muller(rng, sqrt_dt, M)
        rate_integral += 0.5 * r * dt
        r += a * (b - r) * dt + sigma * dW
        rate_integral += 0.5 * r * dt
        r_paths[step + 1] = r
        integrated_r_paths[step + 1] = rate_integral

    return r_paths, integrated_r_paths


def basis(x, N_b):
    """polynomial basis functions setup"""
    n = len(x)
    B = np.ones((n, N_b), dtype=np.float32)
    for k in range(1, N_b):
        B[:, k] = B[:, k - 1] * x
    return B


def payoff_swpt_vec(t, r_array, swaption, a, b, sigma):
    """returns intrinsic payoff of the swaption at time t for a vector of short rates
    """
    M = len(r_array)
    swap_vals = np.empty(M, dtype=np.float32)
    for j in range(M):
        swap_vals[j] = price_irs(
            t=t,
            r=float(r_array[j]),
            r_last_fixing= float(r_array[j]),   
            first_reset=swaption['first_reset'],
            reset_freq=swaption['reset_freq'],
            num_resets=swaption['num_resets'],
            swap_rate=swaption['swap_rate'],
            a=a, b=b, sigma=sigma,
        )
    return np.maximum(swaption['swap_type'] * swap_vals, 0.0).astype(np.float32)


def price_bermudan_swaption(t, r_0, swaption, M, dt, a, b, sigma, N_b, rng):

    # exercise grid
    call_dates = get_call_dates(swaption, curr_t=t)
    if len(call_dates) == 0:
        return 0.0

    # diffuse r from (t, r_0) to to the last exercise date
    # inner re-simulated paths of r
    time_to_last_exercise = call_dates[-1] - t
    r_paths, integrated_r_paths = simulate_short_rate_paths(
        M, time_to_last_exercise, dt, r_0, a, b, sigma, rng,
    )

    # below funct is used as r_paths[step] is eval at time t + step * dt
    def idx_of(date):
        """grid idx of a given date"""
        return round((date - t) / dt)

    # backward induction
    idx_last = idx_of(call_dates[-1])
    V_swpt = payoff_swpt_vec(call_dates[-1], r_paths[idx_last], swaption, a, b, sigma)
    for k in range(len(call_dates) - 2, -1, -1):
        s_k = call_dates[k]
        next_s_k = call_dates[k + 1] # s_k+1
        idx_k = idx_of(s_k)
        idx_next_k = idx_of(next_s_k)
        # discounting V_swpt from s_k+1 back to s_k by each inner path
        # as we've had in the formule : beta(s_k+1) / beta(s_k) = exp(-int_{s_k}^{s_k+1}(r_u)du)
        discount_ratio = np.exp(-(integrated_r_paths[idx_next_k] - integrated_r_paths[idx_k]))
        V_swpt = discount_ratio * V_swpt
        # intrinsic payoff at s_k
        payoff_k = payoff_swpt_vec(s_k, r_paths[idx_k], swaption, a, b, sigma)
        ITM_mask = payoff_k > 0    # LS regression on ITM paths only to improve accuracy
        if ITM_mask.sum() > N_b:   # need enough points to fit
            r_ITM = r_paths[idx_k, ITM_mask]
            V_ITM = V_swpt[ITM_mask]
            payoff_ITM = payoff_k[ITM_mask]
            B = basis(r_ITM, N_b)
            lambda_coef, *_ = np.linalg.lstsq(B, V_ITM, rcond=None)
            C_hat = B @ lambda_coef
            # exercise if intrinsic > continuation value
            exercise = payoff_ITM > C_hat
            V_ITM = np.where(exercise, payoff_ITM, V_ITM)
            V_swpt[ITM_mask] = V_ITM

    # final discount at s_0
    idx_s_0 = idx_of(call_dates[0])
    discount_to_t = np.exp(-integrated_r_paths[idx_s_0])   
    V = discount_to_t * V_swpt

    return float(V.mean())


def price_book_swaptions(t, r_0, swaptions, M, dt, a, b, sigma, N_b, rng):
    """returns total MtM of the book. """
    mtm = 0.0
    for swaption in swaptions:
        price = price_bermudan_swaption(t, r_0, swaption, M, dt, a, b, sigma, N_b, rng)
        mtm += swaption['swap_type'] * swaption['notional'] * price
    return mtm