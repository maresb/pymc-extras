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
from pymc.logprob.transforms import Transform
from pymc.pytensorf import floatX, normalize_rng_param
from pytensor.tensor.random.basic import uniform
from pytensor.tensor.random.utils import normalize_size_param

from pymc_extras.distributions import _pytensor_genpareto as genpareto


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
        draws = genpareto._gpd_quantile_from_excess(-pt.log(v), mu, sigma, xi)
        return cls(inputs=[rng, size, mu, sigma, xi], outputs=[next_rng, draws])(
            rng, size, mu, sigma, xi
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
    :math:`x \to x_F^-`. The log-density value stays accurate there, but its
    gradient w.r.t. :math:`\sigma` and :math:`\xi` carries :math:`\sim z/s` terms
    accurate only to :math:`\sim \varepsilon/s` (about 3 figures within
    :math:`\sim 10^{-13}` of the wall) -- a representability limit of the float64
    input :math:`x` that no rearrangement of ``logp`` recovers. A model that forms
    the margin :math:`s` directly avoids the loss.

    Examples
    --------
    Fitting exceedances over a known threshold. The data must lie in the support
    :math:`[\mu,\ \mu - \sigma/\xi]`; for :math:`\xi < 0` that is the upper wall
    ``max(data) < mu - sigma/xi``, i.e. a lower bound ``xi > -sigma/(max(data) - mu)``
    (there is no upper bound -- :math:`\xi \geq 0` always contains the data). Floor
    :math:`\xi` at ``-1`` as well: for :math:`\xi < -1` the density diverges at the
    upper endpoint (unbounded likelihood). Encoding both as the lower bound of a
    broad :math:`\xi` prior lets NUTS sample the whole shape range without the
    support-wall divergences a free ``(sigma, xi)`` pair suffers.

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

    Keep ``mu`` fixed (the standard peaks-over-threshold setup). A free ``mu``
    makes the likelihood unbounded: it slides onto ``min(data)`` as ``sigma -> 0``,
    which diverges once ``xi > n - 1`` (usually only for tiny samples).
    :class:`ExtGenPareto` hits this far more easily (``mu -> min(data)`` alone, for
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
        return check_parameters(genpareto.logpdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def logcdf(value, mu, sigma, xi):
        return check_parameters(genpareto.logcdf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def logccdf(value, mu, sigma, xi):
        return check_parameters(genpareto.logsf(value, mu, sigma, xi), sigma > 0, msg="sigma > 0")

    def icdf(value, mu, sigma, xi):
        res = genpareto.ppf(value, mu, sigma, xi)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, sigma > 0, msg="sigma > 0")

    def support_point(rv, size, mu, sigma, xi):
        # Median: mean is infinite for xi >= 1, so the median is the safe point.
        excess = np.log(2.0)  # -log(1 - 0.5)
        median = genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


class _GPDProbabilityIntegralTransform(Transform):
    """Default transform for a latent GPD-family variable: ``y = logit(F(x))``.

    Latent (Ext)GPD support depends on xi (``[mu, inf)`` for ``xi >= 0``, bounded
    ``[mu, mu - sigma/xi)`` for ``xi < 0``). An ``Interval`` transform's log-Jacobian
    is discontinuous in xi at 0 -- the upper endpoint ``mu - sigma/xi`` diverges --
    a kink that with random xi causes divergences (measured 163/1600, vs 3 for this
    PIT). The family's own CDF makes the transformed prior exactly Logistic,
    parameter-free and C1, so the kink is gone.

    A strict bijection (``forward(backward(y)) == y``) wherever the quantile is
    float64-representable: all of GenPareto, and ExtGenPareto down to ``kappa ~ 0.1``.
    For ExtGenPareto ``kappa << 1`` the median lies ``~0.5 ** (1 / kappa)`` below
    ``mu``, under ``ulp(mu)`` -- a numerical point mass at ``mu`` -- so ``backward``
    quantizes onto ``mu`` (the float64 limit any x-space transform shares).
    ``log_jac_det`` is the exact PIT correction, so the sampled density stays exactly
    Logistic; only the recovered ``x`` pins to ``mu``.

    Subclasses provide ``_logp`` / ``_logcdf`` / ``_logccdf`` and ``_excess_from_y``;
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
        # Floor x just above mu: a kappa < 1 density diverges at mu (logp = +inf),
        # where the deep lower tail rounds x. Dtype-aware for float32 / tiny sigma.
        finfo = np.finfo(value.dtype)
        floor = pt.abs(mu) * (8.0 * finfo.eps) + finfo.tiny
        return pt.maximum(x, mu + floor)

    def log_jac_det(self, value, *inputs):
        params = inputs[2:]
        x = self.backward(value, *inputs)
        finfo = np.finfo(value.dtype)
        # PIT density is exactly Logistic(value); since the framework adds logp(x)
        # back, return logistic - logp(x). Where |logp(x)| is too large for that to
        # keep the O(1) residue (deep-tail / near-delta saturation), reject with -inf.
        logistic = -pt.softplus(value) - pt.softplus(-value)
        logp_x = self._logp(x, *params)
        unrepresentable = pt.abs(logp_x) > 1.0 / np.sqrt(finfo.eps)
        return pt.switch(unrepresentable, -np.inf, logistic - logp_x)


class _GenParetoPIT(_GPDProbabilityIntegralTransform):
    """PIT transform for :class:`GenPareto`."""

    _logp = staticmethod(genpareto.logpdf)
    _logcdf = staticmethod(genpareto.logcdf)
    _logccdf = staticmethod(genpareto.logsf)

    @staticmethod
    def _excess_from_y(value, mu, sigma, xi):
        # u = sigmoid(y); m = -log(1 - u) = -log(sigmoid(-y)) = softplus(y). Stable for all y.
        return pt.softplus(value)

    @staticmethod
    def _quantile_from_excess(excess, mu, sigma, xi):
        return genpareto._gpd_quantile_from_excess(excess, mu, sigma, xi)


@_default_transform.register(GenPareto)
def _genpareto_default_transform(op, rv):
    return _GenParetoPIT()
