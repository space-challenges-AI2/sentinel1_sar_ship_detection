
"""
fabf.py - Fast Adaptive Bilateral Filter (reference Python implementation)

Public API:
    adaptive_bilateral_filter(f, sigma_map=None, theta_map=None, rho=5.0, N=5, out_path=None, clip=True)
"""

import numpy as np
from scipy import ndimage, special

def _ensure_float01(img):
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def adaptive_bilateral_filter(f, sigma_map=None, theta_map=None, rho=5.0, N=5, out_path=None, clip=True):
    is_color = (f.ndim == 3 and f.shape[2] == 3)
    orig_dtype = f.dtype
    f_arr = _ensure_float01(f.copy())

    if is_color:
        chans = []
        for c in range(3):
            chans.append(_fabf_channel(f_arr[...,c], sigma_map, theta_map, rho, N))
        out = np.stack(chans, axis=-1)
    else:
        out = _fabf_channel(f_arr, sigma_map, theta_map, rho, N)

    if clip:
        out = np.clip(out, 0.0, 1.0)
    if np.issubdtype(orig_dtype, np.integer):
        out = (out * 255.0).round().astype(orig_dtype)
    return out

def _fabf_channel(f, sigma_map, theta_map, rho, N):
    eps = 1e-12
    H, W = f.shape

    # theta_map default
    if theta_map is None:
        theta = f.copy()
    else:
        if np.isscalar(theta_map):
            theta = float(theta_map) * np.ones_like(f)
        else:
            theta = theta_map.astype(np.float64)
            if theta.shape != f.shape:
                raise ValueError("theta_map must be scalar or same shape as image channel")

    # sigma_map default
    if sigma_map is None:
        sigma = 0.1 * np.ones_like(f)
    else:
        if np.isscalar(sigma_map):
            sigma = float(sigma_map) * np.ones_like(f)
        else:
            sigma = sigma_map.astype(np.float64)
            if sigma.shape != f.shape:
                raise ValueError("sigma_map must be scalar or same shape as image channel")

    # spatial window radius and min/max using box of size (2R+1)
    R = int(np.ceil(3.0 * rho))
    size = 2 * R + 1

    alpha = ndimage.minimum_filter(f, size=(size, size), mode='reflect')
    beta  = ndimage.maximum_filter(f, size=(size, size), mode='reflect')

    # Precompute m_k = omega * f^k as gaussian convolutions (k = 0..N)
    mks = np.empty((N+1, H, W), dtype=np.float64)
    for k in range(N+1):
        if k == 0:
            mks[k] = np.ones_like(f)
        else:
            mks[k] = ndimage.gaussian_filter(f**k, sigma=rho, mode='reflect')

    diff = beta - alpha
    small_diff = diff < 1e-12

    from math import comb
    combs = [[comb(k, r) for r in range(k+1)] for k in range(N+1)]

    mu = np.empty_like(mks)
    for k in range(N+1):
        acc = np.zeros_like(f)
        for r in range(k+1):
            acc += combs[k][r] * ((-alpha) ** (k - r)) * mks[r]
        with np.errstate(divide='ignore', invalid='ignore'):
            mu[k] = np.where(~small_diff, acc / (diff ** k + 0.0), 0.0)

    for k in range(N+1):
        mu[k][small_diff] = f[small_diff] ** k

    A = np.fromfunction(lambda i, j: 1.0 / (i + j + 1.0), (N+1, N+1), dtype=int)
    A_inv = np.linalg.inv(A)

    hw = H * W
    mu_flat = mu.reshape((N+1, hw))
    c_flat = A_inv.dot(mu_flat)
    c = c_flat.reshape((N+1, H, W))

    t0 = np.zeros_like(f)
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = np.where(~small_diff, (theta - alpha) / diff, 0.0)
    t0 = np.clip(t0, 0.0, 1.0)

    lam = 0.5 * (diff ** 2) / (sigma ** 2 + 1e-15)

    Ik = np.zeros((N+2, H, W), dtype=np.float64)

    small_lam_mask = lam < 1e-8
    if np.any(small_lam_mask):
        for k in range(N+2):
            Ik[k][small_lam_mask] = 1.0 / (k + 1.0)

    normal_mask = ~small_lam_mask
    if np.any(normal_mask):
        idxs = np.where(normal_mask)
        lam_n = lam[normal_mask]
        t0_n  = t0[normal_mask]
        u1 = -np.sqrt(lam_n) * t0_n
        u2 = np.sqrt(lam_n) * (1.0 - t0_n)
        max_m = N + 1
        J = np.zeros((max_m + 1, u1.size), dtype=np.float64)
        J[0] = 0.5 * np.sqrt(np.pi) * (special.erf(u2) - special.erf(u1))
        if max_m >= 1:
            J[1] = -0.5 * (np.exp(-u2**2) - np.exp(-u1**2))
        for m in range(2, max_m + 1):
            J[m] = -0.5 * ( (u2**(m-1)) * np.exp(-u2**2) - (u1**(m-1)) * np.exp(-u1**2) ) + ((m-1)/2.0) * J[m-2]

        from math import comb as _comb
        for k in range(N+2):
            acc = np.zeros(u1.size, dtype=np.float64)
            for m in range(k+1):
                coeff = _comb(k, m)
                tpow = (t0_n ** (k - m))
                lam_pow = (lam_n ** (-0.5 * m)) if m > 0 else 1.0
                acc += coeff * tpow * lam_pow * J[m]
            Ik[k][normal_mask] = acc

    T1 = np.zeros_like(f)
    T2 = np.zeros_like(f)
    for k in range(N+1):
        T1 += c[k] * Ik[k+1]
        T2 += c[k] * Ik[k]

    denom_small = np.abs(T2) < 1e-14
    out = np.where(denom_small, f, alpha + diff * (T1 / (T2 + 1e-300)))
    out[diff < 1e-12] = f[diff < 1e-12]
    return out
