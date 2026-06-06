#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Generalized Pareto distribution, in pure PyTensor.

Intended for upstreaming to pymc-devs/pytensor-distributions and kept here in that
project's module shape (``logpdf``/``logcdf``/``logsf``/``cdf``/``pdf``/``sf``/
``ppf``/``isf``/``rvs``, value argument ``x`` / ``q``) so pymc-extras can swap to a
``from pytensor_distributions import genpareto`` import once it depends on it.
"""

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable

from pymc_extras.distributions._pytensor_distributions_helper import ppf_bounds_cont


def _series_cutoff(dtype) -> float:
    """``|u|`` below which the divided-difference helpers switch to their series.

    The exact forms (``log1p(u)/u``, ``expm1(u)/u``) are accurate in *value* for
    every ``u != 0``, but autodiff builds their gradient as a difference that
    cancels with relative error ``~ eps / |u|``. Balancing that against the
    series' ``~ u**4`` gradient-truncation error puts the crossover at
    ``eps ** (1/5)`` -- so the cutoff tracks the dtype's ``eps`` (a constant
    tuned for float64 is ~50x too small for float32).
    """
    return float(np.finfo(dtype).eps) ** 0.2


def _log1p_div(u: TensorVariable) -> TensorVariable:
    """``log1p(u) / u`` continued to its removable ``u = 0`` limit (= 1), C1-smooth.

    ``safe_u`` keeps the unused exact branch from evaluating ``0/0`` so the
    gradient graph stays NaN-free.
    """
    cutoff = _series_cutoff(u.dtype)
    use_series = pt.lt(pt.abs(u), cutoff)
    series = 1.0 - u / 2.0 + u**2 / 3.0 - u**3 / 4.0 + u**4 / 5.0
    safe_u = pt.switch(use_series, np.asarray(1.0, dtype=u.dtype), u)
    return pt.switch(use_series, series, pt.log1p(safe_u) / safe_u)


def _expm1_div(u: TensorVariable) -> TensorVariable:
    """``expm1(u) / u`` continued to its removable ``u = 0`` limit (= 1), C1-smooth."""
    cutoff = _series_cutoff(u.dtype)
    use_series = pt.lt(pt.abs(u), cutoff)
    series = 1.0 + u / 2.0 + u**2 / 6.0 + u**3 / 24.0 + u**4 / 120.0
    safe_u = pt.switch(use_series, np.asarray(1.0, dtype=u.dtype), u)
    return pt.switch(use_series, series, pt.expm1(safe_u) / safe_u)


def _gpd_tail(z, xi):
    """``(t, log_s) = (xi * z, log(1 + xi * z))``, formed once per call.

    ``s = 1 + xi * z`` is the only place the observation enters the family; it
    cancels catastrophically as ``value`` nears the ``xi < 0`` upper wall
    ``mu - sigma/xi`` -- a precision loss in the input that no rearrangement
    here recovers.
    """
    t = xi * z
    return t, pt.log1p(t)


def _gpd_log_h(z, sigma, t, log_s):
    """GPD log-density, in-support expression (no support masking)."""
    return -pt.log(sigma) - log_s - z * _log1p_div(t)


def _gpd_log_S(z, t):
    """GPD log survival ``log(1 - H) = -m``, in-support expression.

    Computed directly from the survival exponent ``m = log1p(xi z) / xi`` rather
    than as ``log1mexp(log H)``, so it stays exact arbitrarily deep in the upper
    tail (where ``log H -> 0`` and any ``H``-based route underflows).
    """
    return -(z * _log1p_div(t))


def _gpd_log_H(z, t):
    """GPD log-CDF, in-support expression (no support masking / no saturation)."""
    return pt.log1mexp(_gpd_log_S(z, t))


def _gpd_quantile_from_excess(excess, mu, sigma, xi):
    """Invert the GPD given ``excess = -log(survival) = m >= 0``.

    ``z = expm1(xi m) / xi = m * expm1_div(xi m)`` -> ``value = mu + sigma z``.
    Reused by the ``icdf`` method (``excess`` from a CDF probability) and by the
    random Op (``excess`` from a uniform survival draw).
    """
    return mu + sigma * excess * _expm1_div(xi * excess)


def _gpd_upper_bound(mu, sigma, xi):
    """Right endpoint of the GPD support: ``mu - sigma / xi`` for xi < 0, else +inf."""
    return pt.switch(pt.lt(xi, 0), mu - sigma / xi, np.inf)


def _in_gpd_support(z, t):
    """GPD support mask: ``z >= 0`` and (for ``xi < 0``) ``s = 1 + t > 0``."""
    return pt.and_(z >= 0, 1 + t > 0)


def logpdf(x, mu, sigma, xi):
    z = (x - mu) / sigma
    t, log_s = _gpd_tail(z, xi)
    logp = pt.switch(_in_gpd_support(z, t), _gpd_log_h(z, sigma, t, log_s), -np.inf)
    # The density vanishes at the +inf tail for every xi; for xi > 0 the
    # in-support branch would evaluate log1p(inf)/inf -> nan there, so pin
    # z = +inf to -inf explicitly.
    logp = pt.switch(pt.eq(z, np.inf), -np.inf, logp)
    return logp


def logcdf(x, mu, sigma, xi):
    z = (x - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    # Three regions: below mu -> 0 (log -inf); for xi < 0 past the finite upper
    # endpoint mu - sigma/xi -> 1 (log 0); else log1mexp(-m).
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logcdf = pt.switch(above_upper, 0.0, _gpd_log_H(z, t))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    # CDF -> 1 (logcdf 0) at the +inf tail; for xi > 0 _gpd_log_H(inf) is nan.
    logcdf = pt.switch(pt.eq(z, np.inf), 0.0, logcdf)
    return logcdf


def logsf(x, mu, sigma, xi):
    z = (x - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    # m directly (log S = -m): exact in the heavy tail, where the generic
    # log1mexp(logcdf) survival fallback collapses (logcdf -> 0 there).
    logsf = _gpd_log_S(z, t)
    # For xi < 0 past the finite upper endpoint, and at the +inf tail, S = 0.
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logsf = pt.switch(pt.or_(above_upper, pt.eq(z, np.inf)), -np.inf, logsf)
    # Below mu the survival is 1 (logsf 0).
    logsf = pt.switch(z < 0, 0.0, logsf)
    return logsf


def ppf(q, mu, sigma, xi):
    q = pt.as_tensor_variable(q)
    x = _gpd_quantile_from_excess(-pt.log1p(-q), mu, sigma, xi)  # excess = -log(1 - q)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi):
    return pt.exp(logcdf(x, mu, sigma, xi))


def pdf(x, mu, sigma, xi):
    return pt.exp(logpdf(x, mu, sigma, xi))


def sf(x, mu, sigma, xi):
    return pt.exp(logsf(x, mu, sigma, xi))


def isf(x, mu, sigma, xi):
    x = pt.as_tensor_variable(x)
    # m = -log(x) directly; ppf(1 - x) would lose a tiny x to the 1 - x rounding.
    quantile = _gpd_quantile_from_excess(-pt.log(x), mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def rvs(mu, sigma, xi, size=None, random_state=None):
    # Inverse-CDF on a survival draw: excess = -log(v) avoids the 1 - v cancellation.
    v = pt.random.uniform(size=size, rng=random_state)
    return _gpd_quantile_from_excess(-pt.log(v), mu, sigma, xi)
