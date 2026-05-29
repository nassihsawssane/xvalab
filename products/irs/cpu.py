import math


def vasicek_B(t, T, a):
    return (1 - math.exp(-a * (T - t))) / a


def vasicek_A(t, T, a, b, sigma):
    B = vasicek_B(t, T, a)
    return math.exp((b - 0.5 * sigma**2 / a**2) * (B - (T - t))
                    - 0.25 * sigma**2 / a * B**2)


def price_zcb(t, T, r, a, b, sigma):
    if T <= t:
        return 1.0
    return vasicek_A(t, T, a, b, sigma) * math.exp(-vasicek_B(t, T, a) * r)


def price_irs(t, r, r_last_fixing, first_reset, reset_freq, num_resets,
              swap_rate, a, b, sigma, only_fixed_leg=False, eps=1e-6):
    maturity = first_reset + (num_resets - 1) * reset_freq

    if t > maturity + eps:
        return 0.0

    k = max(int((t - first_reset + eps) / reset_freq), 0) if t >= first_reset - eps else -1

    fixed_leg = 0.0
    zcb_last = 1.0
    for i in range(k + 1, num_resets):
        T_i = first_reset + i * reset_freq
        zcb_last = price_zcb(t, T_i, r, a, b, sigma)
        fixed_leg += zcb_last
    fixed_leg *= reset_freq * swap_rate

    if only_fixed_leg:
        return fixed_leg

    if k == -1:
        floating_leg = price_zcb(t, first_reset, r, a, b, sigma) - zcb_last

    elif abs(t - (first_reset + k * reset_freq)) < eps:
        if k == 0:
            floating_leg = 1.0 - zcb_last
        else:
            inv_zcb_reset = 1.0 / price_zcb(0, reset_freq, r_last_fixing, a, b, sigma)
            floating_leg = inv_zcb_reset - zcb_last
            fixed_leg += reset_freq * swap_rate   
    else:
        # entre deux resets : T_k < t < T_{k+1}
        T_next = first_reset + (k + 1) * reset_freq
        inv_zcb_reset = 1.0 / price_zcb(0, reset_freq, r_last_fixing, a, b, sigma)
        floating_leg = price_zcb(t, T_next, r, a, b, sigma) * inv_zcb_reset - zcb_last

    return floating_leg - fixed_leg


def calibrate_swap_rate_at_par(trade, r0, a, b, sigma):
    annuity = price_irs(
        t=0.0, r=r0, r_last_fixing=r0,
        first_reset=trade['first_reset'],
        reset_freq=trade['reset_freq'],
        num_resets=trade['num_resets'],
        swap_rate=1.0,
        a=a, b=b, sigma=sigma,
        only_fixed_leg=True,
    )
    floating = price_irs(
        t=0.0, r=r0, r_last_fixing=r0,
        first_reset=trade['first_reset'],
        reset_freq=trade['reset_freq'],
        num_resets=trade['num_resets'],
        swap_rate=0.0,
        a=a, b=b, sigma=sigma,
        only_fixed_leg=False,
    )
    return floating / annuity