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

import numpy as np
import pytensor.tensor as pt

from pymc.distributions.dist_math import (
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
)
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.basic import uniform
from pytensor.tensor.random.utils import normalize_size_param

from pymc_extras.distributions import _pytensor_extgenpareto as extgenpareto
from pymc_extras.distributions import _pytensor_genpareto as genpareto
from pymc_extras.distributions._pymc_genpareto import _GPDProbabilityIntegralTransform


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
        excess = extgenpareto._ext_gpd_excess_from_log_prob(pt.log(u), kappa)
        draws = genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi, kappa], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi, kappa
        )


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

    Keep ``mu`` fixed: it is the lower endpoint, where a ``kappa < 1`` density
    diverges. A free ``mu`` slides onto ``min(data)`` to sit on that divergence
    (unbounded likelihood) -- the lower-endpoint mirror of the ``xi < -1`` upper
    wall. If ``mu`` must be estimated, floor ``kappa >= 1`` or put a prior on
    ``min(data) - mu`` that vanishes at 0.
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
            extgenpareto.logpdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def logcdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            extgenpareto.logcdf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def logccdf(value, mu, sigma, xi, kappa):
        return check_parameters(
            extgenpareto.logsf(value, mu, sigma, xi, kappa),
            sigma > 0,
            kappa > 0,
            msg="sigma > 0, kappa > 0",
        )

    def icdf(value, mu, sigma, xi, kappa):
        res = extgenpareto.ppf(value, mu, sigma, xi, kappa)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, kappa > 0, msg="sigma > 0, kappa > 0")

    def support_point(rv, size, mu, sigma, xi, kappa):
        # ExtGPD median (carrier H = 0.5 ** (1/kappa)); for small kappa it rounds onto
        # mu (a -inf initial point), so fall back to the GPD median (excess = log 2),
        # interior for any kappa at ordinary scales. kappa = 1: the two coincide.
        excess = extgenpareto._ext_gpd_excess_from_log_prob(np.log(0.5), kappa)
        median = genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)
        gpd_median = genpareto._gpd_quantile_from_excess(np.log(2.0), mu, sigma, xi)
        point = pt.switch(pt.le(median, mu), gpd_median, median)
        if not rv_size_is_none(size):
            point = pt.full(size, point)
        return point


class _ExtGenParetoPIT(_GPDProbabilityIntegralTransform):
    """PIT transform for :class:`ExtGenPareto` (carrier inverse in ``_excess_from_y``)."""

    _logp = staticmethod(extgenpareto.logpdf)
    _logcdf = staticmethod(extgenpareto.logcdf)
    _logccdf = staticmethod(extgenpareto.logsf)

    @staticmethod
    def _excess_from_y(value, mu, sigma, xi, kappa):
        # GPD excess m = -log1mexp(-a) from the carrier inverse a = -log(F_ext)/kappa,
        # F_ext = sigmoid(y). Split at the cutoff where exp(-y) underflows (dtype-aware):
        # the tail carries log(a) rather than a (which would underflow), via
        # -log(F_ext) = S_ext * log1p_div(-S_ext). Both branches run on clamped inputs
        # so the discarded one stays finite.
        cutoff = np.asarray(
            min(700.0, float(-np.log(np.finfo(value.dtype).tiny)) - 8.0), dtype=value.dtype
        )
        t = pt.softplus(value)
        log_F = -pt.softplus(-pt.minimum(value, cutoff))
        m_bulk = extgenpareto._ext_gpd_excess_from_log_prob(log_F, kappa)
        s = pt.exp(-pt.maximum(t, cutoff))  # S_ext, clamped so genpareto._log1p_div stays finite
        log_a = -t + pt.log(genpareto._log1p_div(-s)) - pt.log(kappa)
        a = pt.exp(pt.minimum(log_a, 700.0))
        m_tail = pt.switch(log_a < -36.0, -log_a, -pt.log1mexp(-a))
        return pt.switch(value < cutoff, m_bulk, m_tail)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi, kappa):
        return genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)


@_default_transform.register(ExtGenPareto)
def _extgenpareto_default_transform(op, rv):
    return _ExtGenParetoPIT()
