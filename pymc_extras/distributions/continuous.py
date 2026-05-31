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

# coding: utf-8
"""
Experimental probability distributions for stochastic nodes in PyMC.

The imports from pymc are not fully replicated here: add imports as necessary.
"""

import numpy as np
import pytensor.tensor as pt

from pymc import ChiSquared, CustomDist
from pymc.distributions import transforms
from pymc.distributions.dist_math import (
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
)
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.basic import uniform
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.variable import TensorVariable
from scipy import stats


class GenExtremeRV(RandomVariable):
    name: str = "Generalized Extreme Value"
    signature = "(),(),()->()"
    dtype: str = "floatX"
    _print_name: tuple[str, str] = ("Generalized Extreme Value", "\\operatorname{GEV}")

    def __call__(self, mu=0.0, sigma=1.0, xi=0.0, size=None, **kwargs) -> TensorVariable:
        return super().__call__(mu, sigma, xi, size=size, **kwargs)

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState | np.random.Generator,
        mu: np.ndarray,
        sigma: np.ndarray,
        xi: np.ndarray,
        size: tuple[int, ...],
    ) -> np.ndarray:
        # Notice negative here, since remainder of GenExtreme is based on Coles parametrization
        return stats.genextreme.rvs(c=-xi, loc=mu, scale=sigma, random_state=rng, size=size)


gev = GenExtremeRV()


class GenExtreme(Continuous):
    r"""
    Univariate Generalized Extreme Value log-likelihood

    The cdf of this distribution is

    .. math::

       G(x \mid \mu, \sigma, \xi) = \exp\left[ -\left(1 + \xi z\right)^{-\frac{1}{\xi}} \right]

    where

    .. math::

        z = \frac{x - \mu}{\sigma}

    and is defined on the set:

    .. math::

        \left\{x: 1 + \xi\left(\frac{x-\mu}{\sigma}\right) > 0 \right\}.

    Note that this parametrization is per Coles (2001) [1]_, and differs from that of
    Scipy in the sign of the shape parameter, :math:`\xi`.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        sigmas = [2., 2., 4.]
        xis = [-0.3, 0.0, 0.3]
        for mu, sigma, xi in zip(mus, sigmas, xis):
            pdf = st.genextreme.pdf(x, c=-xi, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=rf'$\mu$ = {mu}, $\sigma$ = {sigma}, $\xi$={xi}')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  =========================================================================
    Support   * :math:`x \in [\mu - \sigma/\xi, +\infty]`, when :math:`\xi > 0`
              * :math:`x \in \mathbb{R}` when :math:`\xi = 0`
              * :math:`x \in [-\infty, \mu - \sigma/\xi]`, when :math:`\xi < 0`
    Mean      * :math:`\mu + \sigma(g_1 - 1)/\xi`, when :math:`\xi \neq 0, \xi < 1`
              * :math:`\mu + \sigma \gamma`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 1`
                where :math:`\gamma` is the Euler-Mascheroni constant, and
                :math:`g_k = \Gamma (1-k\xi)`
    Variance  * :math:`\sigma^2 (g_2 - g_1^2)/\xi^2`, when :math:`\xi \neq 0, \xi < 0.5`
              * :math:`\frac{\pi^2}{6} \sigma^2`, when :math:`\xi = 0`
              * :math:`\infty`, when :math:`\xi \geq 0.5`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    xi : float
        Shape parameter
    scipy : bool
        Whether or not to use the Scipy interpretation of the shape parameter
        (defaults to `False`).

    References
    ----------
    .. [1] Coles, S.G. (2001).
        An Introduction to the Statistical Modeling of Extreme Values
        Springer-Verlag, London

    """

    rv_op = gev

    @classmethod
    def dist(cls, mu=0, sigma=1, xi=0, scipy=False, **kwargs):
        # If SciPy, use its parametrization, otherwise convert to standard
        if scipy:
            xi = -xi
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        xi = pt.as_tensor_variable(floatX(xi))

        return super().dist([mu, sigma, xi], **kwargs)

    def logp(value, mu, sigma, xi):
        """
        Calculate log-probability of Generalized Extreme Value distribution
        at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Pytensor tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma

        logp_expression = pt.switch(
            pt.isclose(xi, 0),
            -pt.log(sigma) - scaled - pt.exp(-scaled),
            -pt.log(sigma)
            - ((xi + 1) / xi) * pt.log1p(xi * scaled)
            - pt.pow(1 + xi * scaled, -1 / xi),
        )

        logp = pt.switch(pt.gt(1 + xi * scaled, 0.0), logp_expression, -np.inf)

        return check_parameters(
            logp, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def logcdf(value, mu, sigma, xi):
        """
        Compute the log of the cumulative distribution function for Generalized Extreme Value
        distribution at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or `TensorVariable`
            Value(s) for which log CDF is calculated. If the log CDF for
            multiple values are desired the values must be provided in a numpy
            array or `TensorVariable`.

        Returns
        -------
        TensorVariable
        """
        scaled = (value - mu) / sigma
        logc_expression = pt.switch(
            pt.isclose(xi, 0), -pt.exp(-scaled), -pt.pow(1 + xi * scaled, -1 / xi)
        )

        logc = pt.switch(1 + xi * (value - mu) / sigma > 0, logc_expression, -np.inf)

        return check_parameters(
            logc, sigma > 0, pt.and_(xi > -1, xi < 1), msg="sigma > 0 or -1 < xi < 1"
        )

    def support_point(rv, size, mu, sigma, xi):
        r"""
        Using the mode, as the mean can be infinite when :math:`\xi > 1`
        """
        mode = pt.switch(pt.isclose(xi, 0), mu, mu + sigma * (pt.pow(1 + xi, -xi) - 1) / xi)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode


class Chi:
    r"""
    :math:`\chi` log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{\nu - 1}e^{-x^2/2}}{2^{\nu/2 - 1}\Gamma(\nu/2)}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 10, 200)
        for df in [1, 2, 3, 6, 9]:
            pdf = st.chi.pdf(x, df)
            plt.plot(x, pdf, label=r'$\nu$ = {}'.format(df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{2}\frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)}`
    Variance  :math:`\nu - 2\left(\frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)}\right)^2`
    ========  =========================================================================

    Parameters
    ----------
    nu : tensor_like of float
        Degrees of freedom (nu > 0).

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import Chi

        with pm.Model():
            x = Chi("x", nu=1)
    """

    @staticmethod
    def chi_dist(nu: TensorVariable, size: TensorVariable) -> TensorVariable:
        return pt.math.sqrt(ChiSquared.dist(nu=nu, size=size))

    def __new__(cls, name, nu, **kwargs):
        if "observed" not in kwargs:
            kwargs.setdefault("default_transform", transforms.log)
        return CustomDist(name, nu, dist=cls.chi_dist, class_name="Chi", **kwargs)

    @classmethod
    def dist(cls, nu, **kwargs):
        return CustomDist.dist(nu, dist=cls.chi_dist, class_name="Chi", **kwargs)


class Maxwell:
    R"""
    The Maxwell-Boltzmann distribution

    The pdf of this distribution is

    .. math::

       f(x \mid a) = {\displaystyle {\sqrt {\frac {2}{\pi }}}\,{\frac {x^{2}}{a^{3}}}\,\exp \left({\frac {-x^{2}}{2a^{2}}}\right)}

    Read more about it on `Wikipedia <https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution>`_

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 20, 200)
        for a in [1, 2, 5]:
            pdf = st.maxwell.pdf(x, scale=a)
            plt.plot(x, pdf, label=r'$a$ = {}'.format(a))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`2a \sqrt{\frac{2}{\pi}}`
    Variance  :math:`\frac{a^2(3 \pi - 8)}{\pi}`
    ========  =========================================================================

    Parameters
    ----------
    a : tensor_like of float
        Scale parameter (a > 0).

    """

    @staticmethod
    def maxwell_dist(a: TensorVariable, size: TensorVariable) -> TensorVariable:
        if rv_size_is_none(size):
            size = a.shape

        a = CheckParameterValue("a > 0")(a, pt.all(pt.gt(a, 0)))

        return Chi.dist(nu=3, size=size) * a

    def __new__(cls, name, a, **kwargs):
        return CustomDist(
            name,
            a,
            dist=cls.maxwell_dist,
            class_name="Maxwell",
            **kwargs,
        )

    @classmethod
    def dist(cls, a, **kwargs):
        return CustomDist.dist(
            a,
            dist=cls.maxwell_dist,
            class_name="Maxwell",
            **kwargs,
        )


# ===========================================================================
# Generalized Pareto family (GenPareto, ExtGenPareto)
# ===========================================================================
# These two distributions are built in three layers, top to bottom:
#
# 1. Pure PyTensor math kernels (``gen_pareto_logp`` / ``_logcdf`` / ``_icdf``
#    and the extended-family variants), depending only on ``pytensor``. These
#    math kernels are portable -- they can be dropped into another library such
#    as ``pytensor-distributions`` -- but they are only the density/CDF/quantile
#    expressions, not the full functional API (pdf, sf, isf, rvs, moments, ...).
# 2. ``SymbolicRandomVariable`` Ops that sample by inverse-CDF on a uniform
#    draw, so the random methods work on every backend (C, Numba, JAX) without
#    a SciPy object-mode fallback.
# 3. Thin ``Continuous`` distribution classes wiring the two together.
#
# The ``xi -> 0`` limit (where the heavy-tailed GPD collapses to the
# exponential) is handled with C1-smooth ``log1p(u)/u`` and ``expm1(u)/u``
# helpers rather than a ``switch(isclose(xi, 0), ...)`` branch: the naive branch
# leaves a gradient kink at ``xi = 0`` that drives NUTS divergences, while
# routing the whole family through ``m = log1p(xi z) / xi`` keeps both the value
# and its gradient continuous through the limit.


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


# Generalized Pareto core. Standardize to z = (value - mu) / sigma >= 0. With
# m = log1p(xi z) / xi the whole family is analytic in xi:
#     survival   exp(-m) = (1 + xi z) ** (-1/xi)
#     log-pdf    -log sigma - log1p(xi z) - m
#     log-cdf    log(1 - exp(-m)) = log1mexp(-m)
# All three reduce to the exponential law as xi -> 0 (m -> z).


def _safe_mul(a, b):
    """``a * b``, repairing ONLY the indeterminate ``0 * inf`` to ``0``.

    Only ``xi * z`` needs this: at ``xi = 0`` with an infinite observation the
    product is mathematically ``0`` (the exponential GPD carries no shape term),
    but ``0.0 * inf`` is ``nan`` under IEEE, and a ``nan`` in the unused branch of
    a ``switch`` can still leak (the backend may lower it to ``cond*a + ...``).

    The repair is restricted to the exact ``{0} x {+-inf}`` cases so that a
    genuine ``nan`` in ``a`` or ``b`` (e.g. ``xi = nan`` from bad input) still
    propagates instead of being silently turned into the ``xi = 0`` branch.
    """
    prod = a * b
    zero_times_inf = pt.or_(
        pt.and_(pt.eq(a, 0), pt.isinf(b)),
        pt.and_(pt.eq(b, 0), pt.isinf(a)),
    )
    return pt.switch(zero_times_inf, 0.0, prod)


def _gpd_log_h(z, sigma, xi):
    """GPD log-density, in-support expression (no support masking)."""
    t = _safe_mul(xi, z)
    return -pt.log(sigma) - pt.log1p(t) - z * _log1p_div(t)


def _gpd_log_H(z, xi):
    """GPD log-CDF, in-support expression (no support masking / no saturation)."""
    return pt.log1mexp(-(z * _log1p_div(_safe_mul(xi, z))))


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


def _in_gpd_support(z, xi):
    """Boolean mask of the GPD support: z >= 0 and (for xi < 0) z <= -1/xi."""
    return pt.and_(z >= 0, 1 + _safe_mul(xi, z) > 0)


# The ``gen_pareto_*`` / ``ext_gen_pareto_*`` builders below are pure PyTensor:
# they assemble the masked log-density / log-CDF / quantile graphs and call NO
# PyMC parameter check, which keeps them portable (the math can be reused
# elsewhere, e.g. in pytensor-distributions). Parameter validation lives only in
# the Continuous wrapper classes, which add ``check_parameters`` / ``check_icdf_*``.


def gen_pareto_logp(value, mu, sigma, xi):
    """Pure-PyTensor GPD log-density; out-of-support values map to ``-inf``."""
    z = (value - mu) / sigma
    logp = pt.switch(_in_gpd_support(z, xi), _gpd_log_h(z, sigma, xi), -np.inf)
    # The density vanishes at the +inf tail for every xi; for xi > 0 the
    # in-support branch would evaluate log1p(inf)/inf -> nan there, so pin
    # z = +inf to -inf explicitly.
    return pt.switch(pt.eq(z, np.inf), -np.inf, logp)


def gen_pareto_logcdf(value, mu, sigma, xi):
    """Pure-PyTensor GPD log-CDF."""
    z = (value - mu) / sigma
    # Three regions: below mu -> 0 (log -inf); for xi < 0 past the finite upper
    # endpoint mu - sigma/xi -> 1 (log 0); else log1mexp(-m).
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + _safe_mul(xi, z), 0))
    logcdf = pt.switch(above_upper, 0.0, _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    # CDF -> 1 (logcdf 0) at the +inf tail; for xi > 0 _gpd_log_H(inf) is nan.
    return pt.switch(pt.eq(z, np.inf), 0.0, logcdf)


def gen_pareto_icdf(value, mu, sigma, xi):
    """Pure-PyTensor GPD quantile function (assumes ``0 <= value <= 1``)."""
    value = pt.as_tensor_variable(value)
    excess = -pt.log1p(-value)  # = -log(1 - q) = m
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    # Explicit endpoints: q=1 -> finite upper bound (xi<0) or +inf, q=0 -> mu.
    # Without this, q=1 with xi<0 is ``inf * 0 = nan`` rather than ``mu - sigma/xi``.
    x = pt.switch(pt.eq(value, 1), _gpd_upper_bound(mu, sigma, xi), x)
    return pt.switch(pt.eq(value, 0), mu, x)


# Extended Generalized Pareto core. Naveau et al. (2016) extended GPD with
# carrier G(v) = v ** kappa: the CDF is F = H ** kappa with H the GPD CDF, so
# kappa > 0 reshapes the lower tail while xi keeps controlling the upper tail.
# kappa = 1 recovers the plain GPD.


def ext_gen_pareto_logp(value, mu, sigma, xi, kappa):
    """Pure-PyTensor extended-GPD log-density; out-of-support values map to ``-inf``."""
    z = (value - mu) / sigma
    # log g = log kappa + (kappa - 1) log H + log h. The carrier term vanishes
    # at kappa = 1; guarding it keeps the GPD reduction exact at the lower
    # endpoint (z = 0, log H = -inf), where (kappa - 1) * log H would be 0 * -inf.
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * _gpd_log_H(z, xi))
    logp = pt.log(kappa) + carrier + _gpd_log_h(z, sigma, xi)
    logp = pt.switch(_in_gpd_support(z, xi), logp, -np.inf)
    return pt.switch(pt.eq(z, np.inf), -np.inf, logp)


def ext_gen_pareto_logcdf(value, mu, sigma, xi, kappa):
    """Pure-PyTensor extended-GPD log-CDF."""
    z = (value - mu) / sigma
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + _safe_mul(xi, z), 0))
    logcdf = pt.switch(above_upper, 0.0, kappa * _gpd_log_H(z, xi))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    return pt.switch(pt.eq(z, np.inf), 0.0, logcdf)


def ext_gen_pareto_icdf(value, mu, sigma, xi, kappa):
    """Pure-PyTensor extended-GPD quantile function (assumes ``0 <= value <= 1``)."""
    value = pt.as_tensor_variable(value)
    # F = H ** kappa = q  ->  H = q ** (1/kappa); the GPD excess is -log(1 - H),
    # and 1 - H = -expm1(log(q)/kappa) (stable as q ** (1/kappa) -> 1).
    gpd_survival = -pt.expm1(pt.log(value) / kappa)
    excess = -pt.log(gpd_survival)
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    x = pt.switch(pt.eq(value, 1), _gpd_upper_bound(mu, sigma, xi), x)
    return pt.switch(pt.eq(value, 0), mu, x)


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

    Notes
    -----
    For :math:`\xi < 0` the support has a finite right endpoint
    :math:`x_F = \mu - \sigma/\xi`. This implementation uses the *open* support
    convention at that wall: ``logp`` returns :math:`-\infty` and ``logcdf``
    returns :math:`0` for :math:`x \geq x_F`. For :math:`-1 < \xi < 0` this is
    exact -- the density tends to :math:`0` as :math:`x \to x_F^-` -- but it
    differs from SciPy at the boundary itself for :math:`\xi \leq -1` (where
    SciPy reports a finite density at :math:`\xi = -1`, or a divergence for
    :math:`\xi < -1`). Those are measure-zero points that do not affect sampling
    or integration, and :math:`\xi \leq -1` corresponds to an extremely
    short-tailed regime that rarely arises in practice.

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
        return check_parameters(gen_pareto_logp(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def logcdf(value, mu, sigma, xi):
        return check_parameters(gen_pareto_logcdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def icdf(value, mu, sigma, xi):
        res = gen_pareto_icdf(value, mu, sigma, xi)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, msg="sigma > 0")

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

    Notes
    -----
    The upper tail is the GPD tail, so the same open-support convention applies:
    for :math:`\xi < 0`, ``logp`` is :math:`-\infty` and ``logcdf`` is :math:`0`
    at and beyond the finite right endpoint :math:`\mu - \sigma/\xi`. See the
    Notes on :class:`GenPareto` for the (measure-zero) boundary behaviour at
    :math:`\xi \leq -1`.

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
        return check_parameters(
            ext_gen_pareto_logp(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def logcdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            ext_gen_pareto_logcdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def icdf(value, mu, sigma, xi, kappa):
        res = ext_gen_pareto_icdf(value, mu, sigma, xi, kappa)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def support_point(rv, size, mu, sigma, xi, kappa):
        # Median solves H(m) ** kappa = 1/2, i.e. the ExtGPD quantile at 1/2.
        gpd_survival = -pt.expm1(np.log(0.5) / kappa)
        median = _gpd_quantile_from_excess(-pt.log(gpd_survival), mu, sigma, xi)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median
