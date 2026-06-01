"""Margin-aware GPD / ExtGPD primitives for boundary-critical (xi < 0) callers.

EXPERIMENTAL -- prototype for the boundary-robustness design discussion. Not wired
into the package ``__init__`` and not part of the public API yet.

Background. The (Ext)GPD family depends on the data only through the margin
``s = 1 + xi * z`` (``z = (value - mu) / sigma``); see ``continuous.py``. For
``xi < 0`` the support is bounded above by ``mu - sigma/xi`` and ``s -> 0`` at that
wall. Given ``value`` as a float64, ``s = 1 + xi*z`` is a subtraction of near-equal
quantities, so once ``value`` is within ~1e-13 of the wall the low bits of ``s``
are gone: the log-density *value* survives but its ``sigma``/``xi`` gradient (which
carries ``~z/s`` terms) is accurate only to ``~ulp/s``.

These entry points take the margin ``s`` directly, so they never re-form
``1 + xi*z`` and stay exact to full float64 precision at the wall -- *provided the
caller supplies an accurate, correctly-differentiable* ``s``. A model that knows
the wall (a peaks-over-threshold model parameterized by the wall margin) can build
``s`` without cancellation; see the proposal write-up. ``xi != 0`` is required (the
exponential limit has no wall, so use the value-based ``gen_pareto_logp`` there).
"""

import numpy as np
import pytensor.tensor as pt

__all__ = [
    "ext_gen_pareto_logp_from_s",
    "ext_gen_pareto_logsf_from_s",
    "gen_pareto_logp_from_s",
    "gen_pareto_logsf_from_s",
    "genpareto_sigma_from_upper",
]


def gen_pareto_logp_from_s(s, sigma, xi):
    """GPD log-density from the margin ``s = 1 + xi*z`` (``s > 0``, ``xi != 0``).

    ``log h = -log sigma - (1 + 1/xi) * log s``. Out-of-support (``s <= 0``) -> -inf.
    """
    log_s = pt.log(s)
    logp = -pt.log(sigma) - log_s * (1 + 1 / xi)
    return pt.switch(s > 0, logp, -np.inf)


def gen_pareto_logsf_from_s(s, sigma, xi):
    """GPD log-survival ``-m = -log(s)/xi`` from the margin ``s`` (the POT primitive)."""
    logsf = -pt.log(s) / xi
    return pt.switch(s > 0, logsf, -np.inf)


def ext_gen_pareto_logp_from_s(s, sigma, xi, kappa):
    """ExtGPD log-density from the margin ``s``; ``kappa = 1`` recovers the GPD."""
    log_s = pt.log(s)
    m = log_s / xi  # = -log(GPD survival) >= 0 inside the support
    log_H = pt.log1mexp(-m)  # log(1 - exp(-m))
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * log_H)
    logp = pt.log(kappa) + carrier - pt.log(sigma) - log_s - m
    return pt.switch(s > 0, logp, -np.inf)


def ext_gen_pareto_logsf_from_s(s, sigma, xi, kappa):
    """ExtGPD log-survival ``log(1 - H**kappa)`` from the margin ``s``."""
    m = pt.log(s) / xi
    log_H = pt.log1mexp(-m)
    logsf = pt.log1mexp(kappa * log_H)  # log(1 - exp(kappa * log H))
    return pt.switch(s > 0, logsf, -np.inf)


def genpareto_sigma_from_upper(mu, upper, xi):
    """``sigma`` for a GPD whose upper wall is at ``upper`` (``xi < 0``).

    The bounded GPD has wall ``mu - sigma/xi``; pinning it to ``upper`` gives
    ``sigma = (upper - mu) * (-xi)``. Reparameterizing a peaks-over-threshold model
    by ``upper`` (or by the wall margin ``upper - max(data)``) turns the stiff,
    data-coupled support constraint into a plain positivity bound -- which is what
    makes standard ``GenPareto`` / ``ExtGenPareto`` sample without divergences. This
    is the *recommended* fix; the ``*_from_s`` primitives above are only needed when
    the wall is pinned within ~1e-6 of the data.
    """
    return (upper - mu) * (-xi)
