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
