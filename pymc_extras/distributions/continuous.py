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
from pymc.distributions.transforms import _default_transform
from pymc.logprob.transforms import Transform
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.basic import uniform
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.variable import TensorVariable
from scipy import stats

from pymc_extras.distributions._pytensor_extgenpareto import (
    _ext_gpd_excess_from_log_prob,
    ext_gen_pareto_icdf,
    ext_gen_pareto_logccdf,
    ext_gen_pareto_logcdf,
    ext_gen_pareto_logp,
)
from pymc_extras.distributions._pytensor_genpareto import (
    _gpd_quantile_from_excess,
    _log1p_div,
    gen_pareto_icdf,
    gen_pareto_logccdf,
    gen_pareto_logcdf,
    gen_pareto_logp,
)


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
        next_rng, v = uniform(size=size, rng=rng, return_next_rng=True)
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
        next_rng, u = uniform(size=size, rng=rng, return_next_rng=True)
        # Carrier draw u = F; excess = -log(1 - u ** (1/kappa)), via log1mexp so
        # small-kappa draws do not collapse to the lower endpoint (1 - u**.. -> 1).
        excess = _ext_gpd_excess_from_log_prob(pt.log(u), kappa)
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
              * :math:`\mu \leq x < \mu - \sigma/\xi`, when :math:`\xi < 0`
                (open at the right endpoint; see Notes)
    Mean      * :math:`\mu + \sigma / (1 - \xi)`, when :math:`\xi < 1`
              * :math:`\infty`, when :math:`\xi \geq 1`
    Variance  * :math:`\sigma^2 / ((1 - \xi)^2 (1 - 2\xi))`, when :math:`\xi < 1/2`
              * :math:`\infty`, when :math:`\xi \geq 1/2`
    Median    :math:`\mu + \sigma (2^{\xi} - 1) / \xi` (:math:`\mu + \sigma \ln 2`
              at :math:`\xi = 0`)
    Mode      :math:`\mu`, when :math:`\xi > -1` (the density is decreasing). For
              :math:`\xi \leq -1` the density is non-decreasing and its supremum
              sits at the right endpoint :math:`\mu - \sigma/\xi`.
    Entropy   :math:`\ln \sigma + \xi + 1`
    ========  =========================================================================

    The ``support_point`` (used to initialise sampling) is the median, which is
    finite for every :math:`\xi` -- unlike the mean, which diverges for
    :math:`\xi \geq 1`.

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

    **Numerical precision near the** :math:`\xi < 0` **wall.** The family is built
    on :math:`s = 1 + \xi (x - \mu)/\sigma`, which cancels toward :math:`0` as
    :math:`x \to x_F^-`. The log-density *value* keeps near-machine relative
    accuracy there (it is :math:`\sim |1 + 1/\xi|\,\log s`, large in magnitude),
    but its gradient with respect to :math:`\sigma` and :math:`\xi` contains
    :math:`\sim z/s` terms that inherit the lost low bits of :math:`s` and are
    accurate only to a *relative* :math:`\sim \varepsilon/s` (about 3 significant
    figures once :math:`x` is within :math:`\sim 10^{-13}` of the wall). This is a
    representability limit of the float64 *input* :math:`x` -- once it is within a
    few ULP of :math:`x_F` the low bits of :math:`s` are already gone and no
    rearrangement of ``logp(x, ...)`` recovers them. A model that can form the
    margin :math:`s` directly without computing :math:`1 + \xi z` -- e.g. a
    peaks-over-threshold model that knows the wall exactly -- avoids the loss, so
    gradient-critical callers working right at the boundary should parameterize in
    terms of that margin.

    Examples
    --------
    Fitting exceedances over a known threshold. The observations must lie inside
    the support :math:`[\mu,\ \mu - \sigma/\xi]`; for :math:`\xi < 0` the binding
    constraint is the upper wall, ``max(data) < mu - sigma/xi``, which rearranges to
    a *lower* bound on the shape, ``xi > -sigma / (max(data) - mu)`` (there is no
    upper bound -- :math:`\xi \geq 0` is an unbounded tail that always contains the
    data). The shape is *also* floored at ``xi > -1``: for :math:`\xi < -1` the
    density diverges at the upper endpoint, making the likelihood unbounded (an
    improper posterior NUTS runs away to) -- ``xi > -1`` is the usual regularity
    condition. Encoding both as the lower bound of an otherwise broad prior on
    :math:`\xi` lets NUTS sample the whole shape range -- bounded and unbounded
    tails alike -- without the support-wall divergences a free ``(sigma, xi)`` pair
    suffers: the transform stretches the wall to infinity in the unconstrained
    space. (Naming the GPD parameters ``pareto_*`` keeps them distinct from the
    ``mu`` / ``sigma`` *of the prior distributions*.)

    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import GenPareto

        xmin, xmax = data.min(), data.max()
        pareto_mu = threshold  # known peaks-over-threshold value (fixed)
        assert pareto_mu < xmin  # exceedances lie strictly above the threshold

        with pm.Model():
            # sigma carries the data's units; a broad prior on log(sigma) keeps the
            # shape xi inference robust across many orders of magnitude of data scale.
            pareto_sigma = pm.LogNormal("pareto_sigma", mu=0.0, sigma=10.0)
            # xi floored at the data-in-support bound -sigma/(xmax-mu) AND at -1
            # (xi < -1 diverges the density at the upper wall -> unbounded likelihood).
            pareto_xi = pm.TruncatedNormal(
                "pareto_xi",
                mu=0.0,
                sigma=2.0,
                lower=pm.math.maximum(-1.0, -pareto_sigma / (xmax - pareto_mu)),
            )
            GenPareto("obs", mu=pareto_mu, sigma=pareto_sigma, xi=pareto_xi, observed=data)
            idata = pm.sample()

    The lower bound depends on ``xmax``, so the prior support encodes the hard
    "data fall inside the support" requirement -- the standard way to handle a
    distribution whose support edge is an unknown the data constrain (cf.
    ``theta >= max(data)`` for ``Uniform(0, theta)``). This addresses *sampling
    stability*; it is distinct from the deeper float64 precision limit right at the
    wall noted above (reached only when the wall is pinned within ``~1e-13`` of the
    data).

    The threshold ``mu`` is held fixed -- the standard peaks-over-threshold setup,
    and load-bearing: a *free* ``mu`` makes the likelihood unbounded. For the GPD
    this is the classic three-parameter pathology -- ``mu`` slides onto
    ``min(data)`` while ``sigma -> 0``, which diverges once ``xi > n - 1`` (a heavy
    tail relative to the sample size, so usually only reachable for tiny samples).
    For :class:`ExtGenPareto` it is far easier to hit (``mu -> min(data)`` alone, for
    any ``kappa < 1``); see its Examples.
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

    def logccdf(value, mu, sigma, xi):
        return check_parameters(
            gen_pareto_logccdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0"
        )

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

    This is the Type 1 family of Naveau et al. (2016) [1]_, equivalently the EGP3
    model of Papastathopoulos and Tawn (2013) [2]_. The shape :math:`\xi` governs
    the upper tail exactly as in the GPD, while :math:`\kappa > 0` reshapes the
    lower tail (a single, threshold-free model for low, moderate and heavy values);
    because it models the body, the tail-index, scale and return-level estimates are
    stabler to the choice of threshold, allowing a lower one than a plain GPD fit.
    :math:`\kappa = 1` recovers the GPD.

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
              * :math:`\mu \leq x < \mu - \sigma/\xi`, when :math:`\xi < 0`
                (open at the right endpoint, as for :class:`GenPareto`)
    Mean      :math:`\mu + \frac{\sigma}{\xi}\left[\kappa B(\kappa, 1 - \xi) - 1\right]`,
              when :math:`\xi < 1`
    Variance  :math:`\left(\frac{\sigma}{\xi}\right)^2
              \left[\kappa B(\kappa, 1 - 2\xi) - 2\kappa B(\kappa, 1 - \xi) + 1\right]
              - (\text{Mean} - \mu)^2`, when :math:`\xi < 1/2`
    Median    the GPD quantile :math:`Q\!\left(2^{-1/\kappa}\right)` (see
              :class:`GenPareto`), i.e. the point at CDF probability
              :math:`H = 2^{-1/\kappa}`, equivalently survival
              :math:`1 - 2^{-1/\kappa}`
    Mode      * :math:`\mu + \frac{\sigma}{\xi}\left[(T^\star)^{-\xi} - 1\right]`
              (:math:`\mu - \sigma \ln T^\star` at :math:`\xi = 0`),
              :math:`T^\star = \frac{1 + \xi}{\kappa + \xi}`, when
              :math:`\kappa > 1` and :math:`\xi > -1`
              * :math:`\mu`, when :math:`\kappa \leq 1` (and :math:`\xi > -1`)
    ========  =========================================================================

    Here :math:`B(a, b) = \Gamma(a)\Gamma(b)/\Gamma(a + b)` is the Beta function;
    the :math:`r`-th central moment exists iff :math:`\xi < 1/r`, exactly as for
    the GPD. As :math:`\xi \to 0` the mean tends to
    :math:`\mu + \sigma(\psi(\kappa + 1) + \gamma)` (digamma :math:`\psi`,
    Euler--Mascheroni :math:`\gamma`) and the variance to
    :math:`\sigma^2(\pi^2/6 - \psi'(\kappa + 1))`. The mode formulae above assume
    :math:`\xi > -1`; for :math:`\xi \leq -1` the supremum sits at the right
    endpoint, as for :class:`GenPareto`. The ``support_point`` is the ExtGPD
    median when that is representable; for very small :math:`\kappa` the median
    rounds onto :math:`\mu`, so it falls back to the underlying GPD median
    (carrier :math:`H = 1/2`) to keep the initialization interior.

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
    :math:`\xi \leq -1`, and for the relative-precision limit of ``logp`` and its
    :math:`\sigma`/:math:`\xi` gradient as :math:`x` approaches that wall (the
    same :math:`s = 1 + \xi z` cancellation applies here). The :math:`\kappa`
    gradient is exempt -- its carrier term :math:`\log H` has no :math:`1/s`
    factor and stays accurate to machine precision at the wall.

    Examples
    --------
    The upper tail is the GPD tail, so fit it the same way: floor :math:`\xi` at
    both ``-sigma / (max(data) - mu)`` (data inside the support) and ``-1`` (below
    which the upper-endpoint density diverges into an unbounded likelihood) to
    sample without support-wall divergences or a runaway (see :class:`GenPareto`
    Examples for why). ``kappa`` only reshapes the lower tail; a ``Gamma(2, 1)`` prior (density
    ``-> 0`` at the origin, mode at 1) keeps it off the degenerate ``kappa -> 0``
    limit, where the distribution collapses toward a point mass and ``sigma`` can
    run away to compensate.

    .. code-block:: python

        import pymc as pm
        from pymc_extras.distributions import ExtGenPareto

        xmin, xmax = data.min(), data.max()
        pareto_mu = threshold
        assert pareto_mu < xmin  # strict: a kappa < 1 density diverges at x = mu

        with pm.Model():
            # sigma carries the data's units; a broad prior on log(sigma) keeps the
            # shape xi inference robust across many orders of magnitude of data scale.
            pareto_sigma = pm.LogNormal("pareto_sigma", mu=0.0, sigma=10.0)
            pareto_kappa = pm.Gamma("pareto_kappa", alpha=2, beta=1)
            # xi floored at the data-in-support bound -sigma/(xmax-mu) AND at -1
            # (xi < -1 diverges the density at the upper wall -> unbounded likelihood).
            pareto_xi = pm.TruncatedNormal(
                "pareto_xi",
                mu=0.0,
                sigma=2.0,
                lower=pm.math.maximum(-1.0, -pareto_sigma / (xmax - pareto_mu)),
            )
            ExtGenPareto(
                "obs",
                mu=pareto_mu,
                sigma=pareto_sigma,
                xi=pareto_xi,
                kappa=pareto_kappa,
                observed=data,
            )
            idata = pm.sample()

    Keep ``mu`` fixed. It is the lower endpoint of the support, where a
    ``kappa < 1`` density diverges; fixing the threshold keeps that singularity away
    from the data. A *free* ``mu`` slides up to ``min(data)`` to sit on the
    divergence -- an unbounded likelihood / improper posterior, the lower-endpoint
    mirror of the ``xi < -1`` upper wall (NUTS pins ``mu`` to ``min(data)`` with
    ``kappa < 1``). If the threshold must be estimated, floor ``kappa >= 1`` (no
    lower divergence) or put a prior on ``min(data) - mu`` that vanishes at 0.
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

    def logccdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            ext_gen_pareto_logccdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def icdf(value, mu, sigma, xi, kappa):
        res = ext_gen_pareto_icdf(value, mu, sigma, xi, kappa)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def support_point(rv, size, mu, sigma, xi, kappa):
        # The ExtGPD median solves H(m) ** kappa = 1/2 (carrier H = 0.5 ** (1/kappa)),
        # recovered with the shared log1mexp inverse. For small kappa that carrier is
        # so close to 0 that the median is sub-ULP from mu and rounds onto it, which
        # transforms to a -inf initial point. A support point only has to be a usable
        # initialization, so when the median collapses to mu fall back to the
        # underlying GPD median (carrier H = 1/2, excess = log 2) -- a higher ExtGPD
        # quantile (F = 0.5 ** kappa) at mu + O(sigma), interior (hence a finite
        # transformed logp) for any kappa at ordinary scales. It can still round
        # back to mu in the general representability limit where the support has no
        # distinct interior point (sigma far below ulp(mu), or a sub-ULP bounded
        # support). At kappa = 1 the two coincide, so ordinary kappa is unchanged.
        excess = _ext_gpd_excess_from_log_prob(np.log(0.5), kappa)
        median = _gpd_quantile_from_excess(excess, mu, sigma, xi)
        gpd_median = _gpd_quantile_from_excess(np.log(2.0), mu, sigma, xi)
        point = pt.switch(pt.le(median, mu), gpd_median, median)
        if not rv_size_is_none(size):
            point = pt.full(size, point)
        return point


class _GPDProbabilityIntegralTransform(Transform):
    """Default transform for the GPD family: ``y = logit(F(x))``.

    An unobserved (latent) GPD variable lives on a parameter-dependent support --
    ``[mu, inf)`` for ``xi >= 0`` and the bounded ``[mu, mu - sigma/xi)`` for
    ``xi < 0`` -- so it needs a transform to an unconstrained space for NUTS.

    Rather than an ``Interval`` (whose log-Jacobian is *discontinuous in xi at 0*:
    the bounded sigmoid map and the one-sided exp map do not connect as the upper
    endpoint ``mu - sigma/xi`` diverges, putting a ~1e12 gradient kink at xi = 0
    that triggers divergences when xi is itself random), this uses the
    probability-integral transform. ``u = F(x)`` is mapped to ``y = logit(u)`` on
    the whole real line. Because ``F`` is the family's own CDF, the transformed
    prior density is *exactly Logistic and free of mu/sigma/xi*, so it is C1 in
    every parameter (no kink anywhere), while the inverse enforces the correct
    support -- including the moving upper wall -- for all xi.

    The map is built in survival / log space so it never materialises a
    *saturated probability*: ``sigmoid(y)`` rounds to exactly ``1`` for
    ``y >= 37`` in float64, and feeding that into the quantile gives ``inf`` /
    ``nan`` on perfectly valid unconstrained values. ``backward`` instead builds
    the GPD excess ``m = -log(survival)`` directly from ``y`` (``softplus(y)``
    for the base family), and ``log_jac_det`` uses the analytic
    ``log F + log S - logp(x)`` rather than autodiffing the quantile graph.

    Bijectivity and the saturated readout. Where the quantile is representable this
    is an ordinary bijective PIT: ``forward`` and ``backward`` are inverse and
    ``log_jac_det`` is the true ``log|dx/dy|``. But the quantile is *not* always
    representable -- the ExtGPD median is sub-ULP from ``mu`` for small ``kappa``,
    and the heavy/bounded upper tail overflows -- so ``backward`` deliberately
    *saturates* ``x`` onto the support endpoint there (a ``pt.maximum`` floor at the
    bottom, overflow / the wall at the top). In that regime the map is no longer
    invertible: ``forward(backward(y)) != y``, and the saturated ``backward`` has
    derivative ``0`` (true ``log|dx/dy| = -inf``). ``log_jac_det`` is then *not* the
    Jacobian of the saturated readout; it is the density correction
    ``log F + log S - logp(x)`` of the *exact* PIT, so the transformed *density* the
    sampler targets stays the correct Logistic, while the recovered latent ``x`` is
    a quantized readout pinned at the boundary. This is the only consistent float64
    behaviour for a latent whose distribution is (numerically) a point mass at the
    boundary -- which is exactly what small ``kappa`` / the deep tail means -- but
    it does mean this is a density-correct reparameterization with a saturated
    readout there, not a bijection. (Observed data never uses the transform; this
    affects only a latent variable sampled in that degenerate regime.)

    Range of validity. ``log_jac_det`` computes the Jacobian as ``-softplus(y) -
    softplus(-y) - logp(backward(y))`` (the first two terms are ``log F + log S``,
    which the PIT makes the standard Logistic), so the framework's
    ``logp(backward) + log_jac_det`` cancels to *exactly* Logistic(y) -- finite and
    correct -- wherever ``x = backward(y)`` is representable with a finite
    ``logp(x)``. That holds across the whole sampler-reachable range at ordinary
    scales, including the small-``kappa`` lower tail (``kappa < ~|y| / 745``) where
    the carrier excess underflows and ``x`` collapses onto ``mu``: ``backward``
    floors ``x`` just inside the open support (``|mu| * 8 eps`` plus the dtype's
    smallest normal, ``finfo(dtype).tiny`` -- dtype-aware so it survives float32 and
    tiny ``sigma``) so the ``kappa < 1`` divergence at ``mu`` does not make
    ``logp(x) = +inf``.

    Outside that, the transformed logp is robustly ``-inf`` (never ``NaN``, and
    without relying on the optimizer cancelling the two ``logp`` terms): wherever
    ``|logp(x)| > 1 / sqrt(eps)`` the cancellation would lose the O(1) Logistic
    residue, so ``log_jac_det`` returns ``-inf`` directly. Two cases reach that, and
    *neither is the true Logistic value* -- they are representability limits, not
    correctness over all of ``R``:

    * The unreachable far upper tail -- the quantile overflows (``xi > 0``,
      ``y >~ 709 / xi``) or rounds onto the bounded wall, so ``logp(backward) =
      -inf``. A tail probability of ``~e^-(709/xi)``, never reached.
    * The degenerate ``sigma << ulp(mu)`` regime (a near-delta with ``mu != 0``):
      the floor lifts ``x`` to ``z ~ 1 / ulp(mu)``, so ``|logp|`` is enormous and
      the residue is unrecoverable. Sampling such a near-delta latent is not
      meaningful; it yields ``-inf`` (a clean reject) rather than a silent wrong
      value.

    Invalid parameters do not slip through as a valid latent. A non-finite shape
    (``xi`` or ``kappa``) makes ``logp(x)`` NaN; ``log_jac_det`` returns that NaN
    through a constant switch branch so the ``logistic - logp(x)`` cancellation
    cannot turn it back into a finite Logistic, and the transformed logp stays NaN
    (PyMC's ``check_start_vals`` then rejects it loudly) -- matching ``pm.Gamma`` /
    ``pm.StudentT`` for a non-finite shape. A non-finite scale (``sigma = inf``) is
    the degenerate infinite-scale limit and yields ``-inf`` (a clean reject), as
    ``pm.Normal`` does; ``sigma`` / ``kappa <= 0`` and ``sigma`` / ``kappa = nan``
    are caught by the wrappers' ``check_parameters``.

    The default starting point is interior and finite-logp: ``support_point``
    returns the median, or the underlying GPD median when the ExtGPD median rounds
    onto ``mu`` (the sole exception being the general representability limit where
    the support has no distinct interior point at all -- ``sigma`` far below
    ``ulp(mu)``, or a sub-ULP bounded width).

    Subclasses provide the family's ``_logp`` / ``_logcdf`` / ``_logccdf`` and the
    survival-space ``_excess_from_y``; ``inputs`` are the RV's owner inputs, so
    ``inputs[2:]`` are the distribution parameters.
    """

    name = "gpd_pit"
    ndim_supp = 0

    # Filled in by subclasses.
    _logp = _logcdf = _logccdf = staticmethod(lambda *a: None)

    @staticmethod
    def _excess_from_y(value, *params):
        """``m = -log(survival)`` as a function of the unconstrained ``y``."""
        raise NotImplementedError

    @staticmethod
    def _quantile_from_excess(excess, *params):
        raise NotImplementedError

    def forward(self, value, *inputs):
        params = inputs[2:]
        # logit(F) = log F - log(1 - F) = logcdf - logccdf, both stable.
        return self._logcdf(value, *params) - self._logccdf(value, *params)

    def backward(self, value, *inputs):
        params = inputs[2:]
        mu = params[0]
        x = self._quantile_from_excess(self._excess_from_y(value, *params), *params)
        # Floor x above mu so logp(x) cannot be +inf. Deep in the lower tail the
        # excess underflows to 0 and x rounds onto mu, where a kappa < 1 ExtGPD
        # density diverges (logp(mu) = +inf); the framework's logp(backward(y)) would
        # then be +inf and the transformed logp +inf - inf = NaN. The floor combines
        # a relative term (|mu| * 8 eps, to clear mu's ULP) and the dtype's smallest
        # normal (so it does not underflow -- a literal 1e-300 vanishes under float32
        # and the earlier sigma * 1e-300 under tiny sigma). It only binds once x has
        # already collapsed onto mu; the transformed logp is handled in log_jac_det,
        # so the floored value only needs to keep logp(x) finite.
        finfo = np.finfo(value.dtype)
        floor = pt.abs(mu) * (8.0 * finfo.eps) + finfo.tiny
        return pt.maximum(x, mu + floor)

    def log_jac_det(self, value, *inputs):
        params = inputs[2:]
        x = self.backward(value, *inputs)
        finfo = np.finfo(value.dtype)
        # The PIT transformed density is exactly Logistic(value): with F(x) =
        # sigmoid(value), log f_x(x) + log|dx/dy| = log F + log(1 - F) =
        # -softplus(value) - softplus(-value). Return that target minus logp(x); the
        # framework adds logp(backward(value)), so the two logp(x) cancel and the
        # transformed logp is exactly the Logistic log-density -- for all reachable y.
        # Where |logp(x)| is so large the cancellation would lose the O(1) Logistic
        # residue, return -inf instead. That happens only outside the representable
        # range: the unreachable upper saturation (quantile overflows / hits the wall,
        # logp = -inf), and the degenerate sigma << ulp(mu) regime where the floored x
        # sits at z ~ 1/ulp and |logp| ~ 1/(sigma ulp). The threshold 1/sqrt(eps) is
        # where the cancellation would start losing more than half the digits; no
        # reachable x at an ordinary scale comes near it (|logp| stays ~|y|).
        logistic = -pt.softplus(value) - pt.softplus(-value)
        logp_x = self._logp(x, *params)
        unrepresentable = pt.abs(logp_x) > 1.0 / np.sqrt(finfo.eps)
        jac = pt.switch(unrepresentable, -np.inf, logistic - logp_x)
        # A non-finite shape (xi or kappa) makes logp_x NaN. Returning logistic -
        # logp_x would let that NaN cancel against the framework's + logp(backward) and
        # resurface as a *finite* Logistic, silently accepting an invalid parameter as a
        # valid latent. Route it through a switch whose other branch is a bare constant
        # (no logp_x to cancel), so the transformed logp stays NaN -- matching pm.Gamma
        # / pm.StudentT, whose transformed latent logp is NaN (caught loudly at
        # initialization) for a non-finite shape rather than a spurious finite density.
        return pt.switch(pt.isnan(logp_x), np.nan, jac)


class _GenParetoPIT(_GPDProbabilityIntegralTransform):
    _logp = staticmethod(gen_pareto_logp)
    _logcdf = staticmethod(gen_pareto_logcdf)
    _logccdf = staticmethod(gen_pareto_logccdf)

    @staticmethod
    def _excess_from_y(value, mu, sigma, xi):
        # u = sigmoid(y); m = -log(1 - u) = -log(sigmoid(-y)) = softplus(y). Stable for all y.
        return pt.softplus(value)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi):
        return _gpd_quantile_from_excess(excess, mu, sigma, xi)


class _ExtGenParetoPIT(_GPDProbabilityIntegralTransform):
    _logp = staticmethod(ext_gen_pareto_logp)
    _logcdf = staticmethod(ext_gen_pareto_logcdf)
    _logccdf = staticmethod(ext_gen_pareto_logccdf)

    @staticmethod
    def _excess_from_y(value, mu, sigma, xi, kappa):
        # m = -log(1 - H), the GPD-survival exponent, from the carrier inverse
        # H = F_ext ** (1/kappa) = exp(-a), a = -log(F_ext)/kappa >= 0, and
        # y = logit(F_ext). So m = -log(1 - exp(-a)) = -log1mexp(-a).
        #
        # Bulk (value < cutoff): a from log F_ext = -softplus(-y); the log1mexp keeps a
        # tiny GPD survival from rounding to 0 in the small-kappa regime.
        #
        # Tail (value >= cutoff): -softplus(-y) underflows to 0, so carry log(a) instead
        # of a. With S_ext = exp(-t), t = softplus(y),
        #   -log(F_ext) = -log1p(-S_ext) = S_ext * _log1p_div(-S_ext),
        # so log_a = -t + log(_log1p_div(-S_ext)) - log(kappa), finite until t overflows.
        # m = -log1mexp(-a), with a -> 0 (m ~ -log_a) and a -> inf (m ~ 0) split out so a
        # itself never over/underflows. Both branches run on clamped inputs so the
        # discarded one (and its gradient) stays finite.
        #
        # cutoff = where exp(-y) underflows, dtype-aware (~700 float64, ~80 float32);
        # float64 is min(700, 708.4 - 8) = 700.
        cutoff = np.asarray(
            min(700.0, float(-np.log(np.finfo(value.dtype).tiny)) - 8.0), dtype=value.dtype
        )
        t = pt.softplus(value)
        log_F = -pt.softplus(-pt.minimum(value, cutoff))
        m_bulk = _ext_gpd_excess_from_log_prob(log_F, kappa)
        s = pt.exp(-pt.maximum(t, cutoff))  # S_ext, clamped so _log1p_div stays finite
        log_a = -t + pt.log(_log1p_div(-s)) - pt.log(kappa)
        a = pt.exp(pt.minimum(log_a, 700.0))
        m_tail = pt.switch(log_a < -36.0, -log_a, -pt.log1mexp(-a))
        return pt.switch(value < cutoff, m_bulk, m_tail)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi, kappa):
        return _gpd_quantile_from_excess(excess, mu, sigma, xi)


@_default_transform.register(GenPareto)
def _genpareto_default_transform(op, rv):
    return _GenParetoPIT()


@_default_transform.register(ExtGenPareto)
def _extgenpareto_default_transform(op, rv):
    return _ExtGenParetoPIT()
