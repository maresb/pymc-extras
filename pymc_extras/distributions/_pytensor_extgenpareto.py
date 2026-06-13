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

"""Extended Generalized Pareto distribution (Naveau et al. 2016 Type 1), in pure
PyTensor.

Intended for upstreaming to pymc-devs/pytensor-distributions and kept here in that
project's module shape (``logpdf``/``logcdf``/``logsf``/``cdf``/``pdf``/``sf``/
``ppf``/``isf``/``rvs``, value argument ``x`` / ``q``) so pymc-extras can swap to a
``from pytensor_distributions import extgenpareto`` import once it depends on it.
"""

import numpy as np
import pytensor.tensor as pt

from pymc_extras.distributions._pytensor_distributions_helper import ppf_bounds_cont
from pymc_extras.distributions._pytensor_genpareto import (
    _gpd_log_H,
    _gpd_log_h,
    _gpd_log_S,
    _gpd_quantile_from_excess,
    _gpd_tail,
    _gpd_upper_bound,
    _in_gpd_support,
)

# Extended Generalized Pareto core. Naveau et al. (2016) extended GPD with
# carrier G(v) = v ** kappa: the CDF is F = H ** kappa with H the GPD CDF, so
# kappa > 0 reshapes the lower tail while xi keeps controlling the upper tail.
# kappa = 1 recovers the plain GPD.


def logpdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    t, log_s = _gpd_tail(z, xi)
    # log g = log kappa + (kappa - 1) log H + log h. The carrier term vanishes
    # at kappa = 1; guarding it keeps the GPD reduction exact at the lower
    # endpoint (z = 0, log H = -inf), where (kappa - 1) * log H would be 0 * -inf.
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * _gpd_log_H(z, t))
    logp = pt.log(kappa) + carrier + _gpd_log_h(z, sigma, t, log_s)
    logp = pt.switch(_in_gpd_support(z, t), logp, -np.inf)
    logp = pt.switch(pt.eq(z, np.inf), -np.inf, logp)
    return logp


def logcdf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logcdf = pt.switch(above_upper, 0.0, kappa * _gpd_log_H(z, t))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    logcdf = pt.switch(pt.eq(z, np.inf), 0.0, logcdf)
    return logcdf


def logsf(x, mu, sigma, xi, kappa):
    z = (x - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    a = _gpd_log_S(z, t)  # log(1 - H), exact in the tail
    log_H = pt.log1mexp(a)
    generic = pt.log1mexp(kappa * log_H)
    s = pt.exp(a)  # S_gpd
    # r = kappa * S_gpd, formed via exp(log kappa + a) and capped at 1 so the
    # unused (generic-branch) tail expression cannot overflow; in the tail branch
    # below r < e^-30.
    r = pt.exp(pt.minimum(pt.log(kappa) + a, 0.0))
    # 1 - H**kappa = kappa S [1 + (s - r)/2 + (r**2 - 3 r s + 2 s**2)/6 + ...].
    # This is a Taylor expansion in *both* S_gpd and kappa*S_gpd, so it is only
    # valid where both are small; writing it in r = kappa S and s = S keeps every
    # term bounded (no kappa**k powers, which overflow for huge kappa).
    series_m1 = (s - r) / 2.0 + (r * r - 3.0 * r * s + 2.0 * s * s) / 6.0
    tail = pt.log(kappa) + a + pt.log1p(series_m1)
    # Use the tail expansion only where the generic log1mexp(kappa log H) underflows
    # (S_gpd itself tiny, a very negative) AND the expansion is valid (kappa*S_gpd
    # also small). Gating on kappa*S_gpd alone is wrong: for tiny kappa it holds
    # even when S_gpd ~ 1, i.e. in the body, where the generic branch is correct
    # (log(1 - H**kappa) ~ log(kappa) + log(-log H), not log(kappa) + log(1 - H)).
    logsf = pt.switch(pt.and_(a < -30.0, pt.log(kappa) + a < -30.0), tail, generic)
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logsf = pt.switch(pt.or_(above_upper, pt.eq(z, np.inf)), -np.inf, logsf)
    logsf = pt.switch(z < 0, 0.0, logsf)
    return logsf


def _ext_gpd_excess_from_log_prob(log_q, kappa):
    """GPD excess ``m = -log(1 - F ** (1/kappa))`` from ``log_q = log F``.

    For the carrier ``F = H ** kappa``, the GPD CDF is ``H = exp(log_q / kappa)``
    and its survival ``1 - H``, so ``m = -log(1 - H) = -log1mexp(log_q / kappa)``
    (``pt.log1mexp(a) = log(1 - exp(a))`` for ``a <= 0``). Using ``log1mexp`` -- its
    ``log1p`` branch -- instead of ``-log(-expm1(.))`` keeps ``m`` exact when ``H``
    rounds to ``1``: for small ``kappa`` the survival ``1 - H`` is tiny and the
    naive form collapses the excess to ``0`` (and the quantile to ``mu``). Shared
    by the quantile, the sampler, ``support_point`` and the default transform so
    all four inverses agree. ``log_q`` must be ``<= 0`` (a log-probability).
    """
    return -pt.log1mexp(log_q / kappa)


def ppf(q, mu, sigma, xi, kappa):
    q = pt.as_tensor_variable(q)
    # F = H ** kappa = q  ->  H = q ** (1/kappa); excess m = -log(1 - H), built
    # with log1mexp so a tiny 1 - H (small kappa) is not rounded away to 0.
    excess = _ext_gpd_excess_from_log_prob(pt.log(q), kappa)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(x, q, mu, _gpd_upper_bound(mu, sigma, xi))


def cdf(x, mu, sigma, xi, kappa):
    return pt.exp(logcdf(x, mu, sigma, xi, kappa))


def pdf(x, mu, sigma, xi, kappa):
    return pt.exp(logpdf(x, mu, sigma, xi, kappa))


def sf(x, mu, sigma, xi, kappa):
    return pt.exp(logsf(x, mu, sigma, xi, kappa))


def isf(x, mu, sigma, xi, kappa):
    x = pt.as_tensor_variable(x)
    # log F = log1p(-x), accurate for tiny x; ppf(1 - x) forms 1 - x first and loses it.
    excess = _ext_gpd_excess_from_log_prob(pt.log1p(-x), kappa)
    quantile = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    return ppf_bounds_cont(quantile, x, _gpd_upper_bound(mu, sigma, xi), mu)


def rvs(mu, sigma, xi, kappa, size=None, random_state=None):
    # Inverse-CDF on a carrier draw u = F; excess = -log(1 - u ** (1/kappa)).
    u = pt.random.uniform(size=size, rng=random_state, return_next_rng=True)[1]
    excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
    return _gpd_quantile_from_excess(excess, mu, sigma, xi)
