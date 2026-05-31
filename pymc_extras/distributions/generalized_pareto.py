#   Copyright 2024 The PyMC Developers
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
r"""Generalized Pareto and Extended Generalized Pareto distributions.

The module is organised in three layers, top to bottom:

1. **Pure PyTensor math** -- ``gen_pareto_logp`` / ``_logcdf`` / ``_icdf`` and
   the extended-family variants. These are plain graph builders that depend only
   on ``pytensor`` (no PyMC machinery), so they can be reused directly or moved
   to ``pytensor-distributions`` unchanged.
2. **SymbolicRandomVariable** Ops for forward sampling, implemented by inverse-CDF
   sampling on a uniform draw -- so the random methods work on every backend
   (C, Numba, JAX) with no SciPy object-mode fallback.
3. Thin PyMC ``Continuous`` distribution classes that wire the two together.

The ``xi -> 0`` limit (where the heavy-tailed GPD collapses to the exponential)
is handled with C1-smooth ``log1p(u)/u`` and ``expm1(u)/u`` helpers rather than a
``switch(isclose(xi, 0), ...)`` branch. The naive branch leaves a gradient kink
at ``xi = 0`` that drives NUTS divergences; routing the whole family through
``m = log1p(xi z) / xi`` keeps both the value and its gradient continuous through
the limit.
"""

import numpy as np
import pytensor.tensor as pt

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.basic import uniform
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.variable import TensorVariable

__all__ = ["ExtGenPareto", "GenPareto"]


# ---------------------------------------------------------------------------
# Smooth divided-difference building blocks
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pure PyTensor: Generalized Pareto core
# ---------------------------------------------------------------------------
# Standardize to z = (value - mu) / sigma >= 0. With m = log1p(xi z) / xi the
# whole family is analytic in xi:
#     survival   exp(-m) = (1 + xi z) ** (-1/xi)
#     log-pdf    -log sigma - log1p(xi z) - m
#     log-cdf    log(1 - exp(-m)) = log1mexp(-m)
# All three reduce to the exponential law as xi -> 0 (m -> z).


def _gpd_log_h(z, sigma, xi):
    """GPD log-density, in-support expression (no support masking)."""
    return -pt.log(sigma) - pt.log1p(xi * z) - z * _log1p_div(xi * z)


def _gpd_log_H(z, xi):
    """GPD log-CDF, in-support expression (no support masking / no saturation)."""
    return pt.log1mexp(-(z * _log1p_div(xi * z)))


def _gpd_quantile_from_excess(excess, mu, sigma, xi):
    """Invert the GPD given ``excess = -log(survival) = m >= 0``.

    ``z = expm1(xi m) / xi = m * expm1_div(xi m)`` -> ``value = mu + sigma z``.
    Reused by the ``icdf`` method (``excess`` from a CDF probability) and by the
    random Op (``excess`` from a uniform survival draw).
    """
    return mu + sigma * excess * _expm1_div(xi * excess)


def _in_gpd_support(z, xi):
    """Boolean mask of the GPD support: z >= 0 and (for xi < 0) z <= -1/xi."""
    return pt.and_(z >= 0, 1 + xi * z > 0)


def gen_pareto_logp(value, mu, sigma, xi):
    """Log-density of the Generalized Pareto distribution."""
    z = (value - mu) / sigma
    # Out-of-support values get -inf via the switch; check_parameters only guards
    # the scalar parameter (it raises, which is the wrong response for a value
    # being out of support).
    logp = pt.switch(_in_gpd_support(z, xi), _gpd_log_h(z, sigma, xi), -np.inf)
    return check_parameters(logp, sigma > 0, msg="sigma > 0")


def gen_pareto_logcdf(value, mu, sigma, xi):
    """Log-CDF of the Generalized Pareto distribution."""
    z = (value - mu) / sigma
    # Three regions: below mu the CDF is 0 (-inf); for xi < 0 past the finite
    # upper endpoint mu - sigma/xi it saturates at 1 (log-CDF 0); else log1mexp(-m).
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logcdf = pt.switch(above_upper, 0.0, _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    return check_parameters(logcdf, sigma > 0, msg="sigma > 0")


def gen_pareto_icdf(value, mu, sigma, xi):
    """Inverse CDF (quantile function) of the Generalized Pareto distribution."""
    excess = -pt.log1p(-value)  # = -log(1 - q) = m
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    x = pt.switch(pt.and_(value >= 0, value <= 1), x, np.nan)
    return check_parameters(x, sigma > 0, msg="sigma > 0")


# ---------------------------------------------------------------------------
# Pure PyTensor: Extended Generalized Pareto core
# ---------------------------------------------------------------------------
# Naveau et al. (2016) extended GPD with carrier G(v) = v ** kappa: the CDF is
# F = H ** kappa with H the GPD CDF, so kappa > 0 reshapes the lower tail while
# xi keeps controlling the upper tail. kappa = 1 recovers the plain GPD.


def ext_gen_pareto_logp(value, mu, sigma, xi, kappa):
    """Log-density of the Extended Generalized Pareto distribution."""
    z = (value - mu) / sigma
    # log g = log kappa + (kappa - 1) log H + log h. The carrier term vanishes
    # at kappa = 1; guarding it keeps the GPD reduction exact at the lower
    # endpoint (z = 0, log H = -inf), where (kappa - 1) * log H would be 0 * -inf.
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * _gpd_log_H(z, xi))
    logp = pt.log(kappa) + carrier + _gpd_log_h(z, sigma, xi)
    logp = pt.switch(_in_gpd_support(z, xi), logp, -np.inf)
    return check_parameters(logp, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")


def ext_gen_pareto_logcdf(value, mu, sigma, xi, kappa):
    """Log-CDF of the Extended Generalized Pareto distribution."""
    z = (value - mu) / sigma
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + xi * z, 0))
    logcdf = pt.switch(above_upper, 0.0, kappa * _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    return check_parameters(logcdf, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")


def ext_gen_pareto_icdf(value, mu, sigma, xi, kappa):
    """Inverse CDF of the Extended Generalized Pareto distribution."""
    # F = H ** kappa = q  ->  H = q ** (1/kappa); the GPD excess is -log(1 - H),
    # and 1 - H = -expm1(log(q)/kappa) (stable as q ** (1/kappa) -> 1).
    gpd_survival = -pt.expm1(pt.log(value) / kappa)
    excess = -pt.log(gpd_survival)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    x = pt.switch(pt.and_(value >= 0, value <= 1), x, np.nan)
    return check_parameters(x, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")


# ---------------------------------------------------------------------------
# Random variables (symbolic, inverse-CDF sampling)
# ---------------------------------------------------------------------------
def _uniform_draw(size, rng):
    """``(next_rng, draw)`` from a uniform RV (the pytensor random idiom)."""
    return uniform(size=size, rng=rng, return_next_rng=True)


class GenParetoRV(SymbolicRandomVariable):
    name = "genpareto"
    extended_signature = "[rng],[size],(),(),()->[rng],()"
    _print_name = ("GenPareto", "\\operatorname{GenPareto}")

    @classmethod
    def rv_op(cls, mu, sigma, xi, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        sigma = pt.as_tensor(sigma)
        xi = pt.as_tensor(xi)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, sigma, xi, ndims_params=cls.ndims_params)
        # Draw the survival probability directly so excess = -log(v) avoids the
        # 1 - u cancellation that hurts the heavy upper tail.
        next_rng, v = _uniform_draw(size, rng)
        draws = _gpd_quantile_from_excess(-pt.log(v), mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi
        )


class ExtGenParetoRV(SymbolicRandomVariable):
    name = "extgenpareto"
    extended_signature = "[rng],[size],(),(),(),()->[rng],()"
    _print_name = ("ExtGenPareto", "\\operatorname{ExtGenPareto}")

    @classmethod
    def rv_op(cls, mu, sigma, xi, kappa, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        sigma = pt.as_tensor(sigma)
        xi = pt.as_tensor(xi)
        kappa = pt.as_tensor(kappa)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)
        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, sigma, xi, kappa, ndims_params=cls.ndims_params)
        next_rng, u = _uniform_draw(size, rng)
        # GPD survival of the carrier draw: 1 - u ** (1/kappa) = -expm1(log u / kappa).
        excess = -pt.log(-pt.expm1(pt.log(u) / kappa))
        draws = _gpd_quantile_from_excess(excess, mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi, kappa], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi, kappa
        )


# ---------------------------------------------------------------------------
# PyMC distribution classes
# ---------------------------------------------------------------------------
class GenPareto(Continuous):
    r"""Univariate Generalized Pareto log-likelihood.

    The Generalized Pareto distribution (GPD) is the canonical model for
    exceedances over a threshold (peaks-over-threshold), the counterpart of the
    GEV used for block maxima. Its CDF is

    .. math::

       G(x \mid \mu, \sigma, \xi) =
           1 - \left(1 + \xi \frac{x - \mu}{\sigma}\right)^{-1/\xi}

    for :math:`\xi \neq 0`, continued to :math:`1 - \exp(-(x - \mu)/\sigma)` at
    :math:`\xi = 0`. The shape :math:`\xi` is parametrized as in Coles (2001)
    [1]_, matching SciPy's ``genpareto`` ``c`` parameter.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 10, 200)
        mus = [0., 0., 0.]
        sigmas = [1., 1., 2.]
        xis = [0.0, 0.4, -0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genpareto.pdf(x, c=xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$ = {xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   * :math:`x \geq \mu`, when :math:`\xi \geq 0`
              * :math:`\mu \leq x \leq \mu - \sigma/\xi`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma / (1 - \xi)`, when :math:`\xi < 1`
              * :math:`\infty`, when :math:`\xi \geq 1`
    Variance  * :math:`\sigma^2 / ((1 - \xi)^2 (1 - 2\xi))`, when :math:`\xi < 1/2`
              * :math:`\infty`, when :math:`\xi \geq 1/2`
    ========  =========================================================================

    Parameters
    ----------
    mu : tensor_like of float
        Location parameter (the threshold).
    sigma : tensor_like of float
        Scale parameter (sigma > 0).
    xi : tensor_like of float
        Shape parameter. :math:`\xi > 0` gives a heavy (Pareto) tail,
        :math:`\xi = 0` an exponential tail, and :math:`\xi < 0` a bounded
        upper tail.

    References
    ----------
    .. [1] Coles, S. G. (2001). An Introduction to the Statistical Modeling of
        Extreme Values. Springer-Verlag, London.
    .. [2] Pickands, J. (1975). Statistical Inference Using Extreme Order
        Statistics. Annals of Statistics, 3(1), 119-131.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import GenPareto

        with pm.Model():
            sigma = pm.HalfNormal("sigma", 1.0)
            xi = pm.Normal("xi", 0.0, 0.5)
            obs = GenPareto("obs", mu=0.0, sigma=sigma, xi=xi, observed=exceedances)
    """

    rv_type = GenParetoRV
    rv_op = GenParetoRV.rv_op

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))
        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        return gen_pareto_logp(value, mu, sigma, xi)

    def logcdf(value, mu, sigma, xi):
        return gen_pareto_logcdf(value, mu, sigma, xi)

    def icdf(value, mu, sigma, xi):
        return gen_pareto_icdf(value, mu, sigma, xi)

    def support_point(rv, size, mu, sigma, xi):
        # Median: mean is infinite for xi >= 1, so the median is the safe point.
        excess = np.log(2.0)  # -log(1 - 0.5)
        median = _gpd_quantile_from_excess(excess, mu, sigma, xi)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


class ExtGenPareto(Continuous):
    r"""Univariate Extended Generalized Pareto log-likelihood.

    The extended GPD of Naveau et al. (2016) [1]_ transforms the GPD CDF
    :math:`H` through the carrier :math:`G(v) = v^\kappa`:

    .. math::

       F(x \mid \mu, \sigma, \xi, \kappa) =
           \left[H\!\left(x \mid \mu, \sigma, \xi\right)\right]^{\kappa}.

    The shape :math:`\xi` governs the upper tail exactly as in the GPD, while
    :math:`\kappa > 0` reshapes the lower tail (a single, threshold-free model
    for low, moderate and heavy values). :math:`\kappa = 1` recovers the GPD.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 8, 200)
        xi = 0.1
        for kappa in [0.5, 1.0, 2.0, 5.0]:
            H = st.genpareto.cdf(x, c=xi)
            h = st.genpareto.pdf(x, c=xi)
            pdf = kappa * np.power(H, kappa - 1) * h
            plt.plot(x, pdf, label=rf'$\kappa$ = {kappa}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   * :math:`x \geq \mu`, when :math:`\xi \geq 0`
              * :math:`\mu \leq x \leq \mu - \sigma/\xi`, when :math:`\xi < 0`
    ========  =========================================================================

    Parameters
    ----------
    mu : tensor_like of float
        Location parameter (the threshold).
    sigma : tensor_like of float
        Scale parameter (sigma > 0).
    xi : tensor_like of float
        Upper-tail shape parameter (as in :class:`GenPareto`).
    kappa : tensor_like of float
        Lower-tail shape parameter (kappa > 0). ``kappa = 1`` reduces the
        distribution to :class:`GenPareto`.

    References
    ----------
    .. [1] Naveau, P., Huser, R., Ribereau, P., & Hannart, A. (2016). Modeling
        jointly low, moderate, and heavy rainfall intensities without a threshold
        selection. Water Resources Research, 52(4), 2753-2769.
    .. [2] Papastathopoulos, I., & Tawn, J. A. (2013). Extended generalised
        Pareto models for tail estimation. Journal of Statistical Planning and
        Inference, 143(1), 131-143.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import ExtGenPareto

        with pm.Model():
            sigma = pm.HalfNormal("sigma", 1.0)
            xi = pm.Normal("xi", 0.0, 0.5)
            kappa = pm.HalfNormal("kappa", 2.0)
            obs = ExtGenPareto("obs", mu=0.0, sigma=sigma, xi=xi, kappa=kappa, observed=data)
    """

    rv_type = ExtGenParetoRV
    rv_op = ExtGenParetoRV.rv_op

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, kappa=1, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))
        kappa = pt.as_tensor_variable(floatX(kappa))
        return super().dist([mu, sigma, xi, kappa], **kwargs)

    def logp(value, mu, sigma, xi, kappa):
        return ext_gen_pareto_logp(value, mu, sigma, xi, kappa)

    def logcdf(value, mu, sigma, xi, kappa):
        return ext_gen_pareto_logcdf(value, mu, sigma, xi, kappa)

    def icdf(value, mu, sigma, xi, kappa):
        return ext_gen_pareto_icdf(value, mu, sigma, xi, kappa)

    def support_point(rv, size, mu, sigma, xi, kappa):
        # Median solves H(m) ** kappa = 1/2, i.e. the ExtGPD quantile at 1/2.
        gpd_survival = -pt.expm1(np.log(0.5) / kappa)
        median = _gpd_quantile_from_excess(-pt.log(gpd_survival), mu, sigma, xi)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median
