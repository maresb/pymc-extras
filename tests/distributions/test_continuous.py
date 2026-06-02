#   Copyright 2020 The PyMC Developers
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
from decimal import Decimal, getcontext

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

# general imports
import pytest
import scipy.stats.distributions as sp


# test support imports from pymc
from pymc.distributions.distribution import support_point as _support_point
from pymc.logprob.utils import ParameterValueError
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    R,
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_icdf,
    check_logccdf,
    check_logcdf,
    check_logp,
    check_selfconsistency_icdf,
    seeded_scipy_distribution_builder,
    select_by_precision,
)
from scipy import stats

# the distributions to be tested
from pymc_extras.distributions import Chi, ExtGenPareto, GenExtreme, GenPareto, Maxwell
from pymc_extras.distributions.continuous import ext_gen_pareto_logp, gen_pareto_logp

pytestmark = pytest.mark.filterwarnings(
    "ignore:Numba will use object mode to run Generalized Extreme Value:UserWarning"
)

# The Generalized Pareto family legitimately evaluates to +/-inf at the
# (measure-zero) support boundary and at probabilities 0 / 1; NumPy flags those
# as divide-by-zero / invalid / overflow during ``.eval()``. They are the correct
# boundary values (the numerical comparisons still validate them). Scope the FPE
# silencing to the GPD test classes (via their ``pytestmark``) rather than muting
# the whole module, which also holds the GenExtreme / Chi / Maxwell tests.
_GPD_FPE_FILTERS = [
    pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
]


class TestGenExtremeClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.

    pm.logp(GenExtreme.dist(mu=0.,sigma=1.,xi=0.5),value=-0.01)
    """

    def test_logp(self):
        def ref_logp(value, mu, sigma, xi):
            if 1 + xi * (value - mu) / sigma > 0:
                return sp.genextreme.logpdf(value, c=-xi, loc=mu, scale=sigma)
            else:
                return -np.inf

        check_logp(
            GenExtreme,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-1, -0.99, -0.5, 0, 0.5, 0.99, 1]),
            },
            ref_logp,
        )

    def test_logcdf(self):
        def ref_logcdf(value, mu, sigma, xi):
            if 1 + xi * (value - mu) / sigma > 0:
                return sp.genextreme.logcdf(value, c=-xi, loc=mu, scale=sigma)
            else:
                return -np.inf

        check_logcdf(
            GenExtreme,
            R,
            {
                "mu": R,
                "sigma": Rplusbig,
                "xi": Domain([-1, -0.99, -0.5, 0, 0.5, 0.99, 1]),
            },
            ref_logcdf,
            decimal=select_by_precision(float64=6, float32=2),
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0, 1, 0, None, 0),
            (1, np.arange(1, 4), 0.1, None, 1 + np.arange(1, 4) * (1.1**-0.1 - 1) / 0.1),
            (np.arange(5), 1, 0.1, None, np.arange(5) + (1.1**-0.1 - 1) / 0.1),
            (
                0,
                1,
                np.linspace(-0.2, 0.2, 6),
                None,
                ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
                / np.linspace(-0.2, 0.2, 6),
            ),
            (1, 2, 0.1, 5, np.full(5, 1 + 2 * (1.1**-0.1 - 1) / 0.1)),
            (
                np.arange(6),
                np.arange(1, 7),
                np.linspace(-0.2, 0.2, 6),
                (3, 6),
                np.full(
                    (3, 6),
                    np.arange(6)
                    + np.arange(1, 7)
                    * ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
                    / np.linspace(-0.2, 0.2, 6),
                ),
            ),
        ],
    )
    def test_genextreme_support_point(self, mu, sigma, xi, size, expected):
        with pm.Model() as model:
            GenExtreme("x", mu=mu, sigma=sigma, xi=xi, size=size)
        assert_support_point_is_expected(model, expected)

    def test_gen_extreme_scipy_kwarg(self):
        dist = GenExtreme.dist(xi=1, scipy=False)
        assert dist.owner.inputs[-1].eval() == 1

        dist = GenExtreme.dist(xi=1, scipy=True)
        assert dist.owner.inputs[-1].eval() == -1


class TestGenExtreme(BaseTestDistributionRandom):
    pymc_dist = GenExtreme
    pymc_dist_params = {"mu": 0, "sigma": 1, "xi": -0.1}
    expected_rv_op_params = {"mu": 0, "sigma": 1, "xi": -0.1}
    # Notice, using different parametrization of xi sign to scipy
    reference_dist_params = {"loc": 0, "scale": 1, "c": 0.1}
    reference_dist = seeded_scipy_distribution_builder("genextreme")
    tests_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestChiClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            Chi,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi.logpdf(value, df=nu),
        )

    def test_logcdf(self):
        check_logcdf(
            Chi,
            Rplus,
            {"nu": Rplus},
            lambda value, nu: sp.chi.logcdf(value, df=nu),
        )


class TestMaxwell:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_logp(self):
        check_logp(
            Maxwell,
            Rplus,
            {"a": Rplus},
            lambda value, a: sp.maxwell.logpdf(value, scale=a),
        )

    def test_logcdf(self):
        check_logcdf(
            Maxwell,
            Rplus,
            {"a": Rplus},
            lambda value, a: sp.maxwell.logcdf(value, scale=a),
        )


# xi is unconstrained for the GPD family, so its domain carries explicit
# ``(None, None)`` edges: the harness then runs no "just-outside-the-edge"
# invalid-xi probe (there is no invalid xi) while still exercising every listed
# value -- the exponential limit xi = 0 and both tails. Two deliberate bounds on
# the range:
#   * strictly > -1: at xi <= -1 the GPD becomes (sub-)uniform with a *finite*
#     density at its closed upper endpoint, a measure-zero point where the
#     open-support convention here (-inf at the wall) legitimately differs from
#     SciPy. ``TestGenParetoBoundaries`` covers the xi < 0 wall directly.
#   * <= 1: a heavier tail (e.g. xi = 5) pushes the q = 0.99 quantile to ~1e10,
#     where ``check_icdf``'s *absolute* tolerance fails on a value that is in
#     fact correct to ~1e-15 relative -- false precision, not a real error.
XI_DOMAIN = Domain([-0.9, -0.5, -0.1, 0, 0.1, 0.5, 1], dtype="float64", edges=(None, None))
# kappa > 0: the trailing inf leaves the upper edge unbounded (no invalid probe
# above) while the leading 0 lets the harness probe kappa <= 0 (must raise). The
# inner values are the actual test points.
KAPPA_DOMAIN = Domain([0, 0.25, 0.5, 1, 2, 5, np.inf], dtype="float64")
# ``check_icdf`` compares absolute quantile values, so cap sigma to keep the
# heavy-tail quantiles within tolerance; the leading 0 / trailing inf still let
# the harness probe sigma <= 0 (must raise). sigma is a pure linear scale.
SIGMA_ICDF = Domain([0, 0.1, 0.5, 1.0, 2.0, np.inf], dtype="float64")


def ref_ext_logp(value, mu, sigma, xi, kappa):
    z = (value - mu) / sigma
    if z < 0 or (1 + xi * z) <= 0:
        return -np.inf
    log_H = sp.genpareto.logcdf(z, c=xi)
    log_h = sp.genpareto.logpdf(z, c=xi) - np.log(sigma)
    return np.log(kappa) + (kappa - 1) * log_H + log_h


def ref_ext_logcdf(value, mu, sigma, xi, kappa):
    z = (value - mu) / sigma
    if z < 0:
        return -np.inf
    if xi < 0 and (1 + xi * z) <= 0:
        return 0.0
    return kappa * sp.genpareto.logcdf(z, c=xi)


def ref_ext_logccdf(value, mu, sigma, xi, kappa):
    # S = 1 - H ** kappa, with H the GPD CDF. Compute via the GPD log-CDF so the
    # reference stays accurate deep in the tail (where H -> 1 and the naive
    # 1 - H**kappa underflows to 0) -- exactly the regime logccdf is for.
    z = (value - mu) / sigma
    if z < 0:
        return 0.0
    if xi < 0 and (1 + xi * z) <= 0:
        return -np.inf
    return np.log(-np.expm1(kappa * sp.genpareto.logcdf(z, c=xi)))


def ref_ext_icdf(q, mu, sigma, xi, kappa):
    # G^{-1}(q) = H^{-1}(q ** (1/kappa)).
    return sp.genpareto.ppf(q ** (1 / kappa), c=xi, loc=mu, scale=sigma)


def _gpd_ref_logp(value, mu, sigma, xi, kappa=None):
    """100-digit (Ext)GPD logp from the *exact* margin s = 1 + xi*z.

    ``mu/sigma/xi/kappa`` are ``Decimal``; ``value`` is the float64 input. Building
    ``s`` at 100 digits avoids the ``1 + xi*z -> 0`` cancellation that the float64
    path cannot, so this is the truth the float64 logp/gradient is measured against.
    """
    getcontext().prec = 100
    z = (Decimal(value) - mu) / sigma
    log_s = (1 + xi * z).ln()
    logp = -sigma.ln() - (1 + 1 / xi) * log_s  # = -log sigma - (1 + 1/xi) log s
    if kappa is not None:
        # carrier: (kappa - 1) log H, H = 1 - exp(-m) = 1 - s ** (-1/xi)
        log_H = (1 - ((Decimal(-1) / xi) * log_s).exp()).ln()
        logp += kappa.ln() + (kappa - 1) * log_H
    return logp


def _gpd_ref_grad(value, params, which):
    """High-precision central difference d logp / d ``which`` (params: Decimals).

    The reference function has ~1/s**k derivatives at the wall, so the step is tiny
    (1e-22 relative) and the arithmetic is 100-digit -- the finite-difference error
    stays far below the float64 gradient error being measured.
    """
    h = abs(params[which]) * Decimal("1e-22") or Decimal("1e-22")
    hi = dict(params, **{which: params[which] + h})
    lo = dict(params, **{which: params[which] - h})
    return (_gpd_ref_logp(value, **hi) - _gpd_ref_logp(value, **lo)) / (2 * h)


class TestGenParetoClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    pytestmark = _GPD_FPE_FILTERS

    def test_logp(self):
        check_logp(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logpdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logcdf(self):
        check_logcdf(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logcdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logccdf(self):
        # log survival function: exact and tail-stable (vs the lossy
        # log1mexp(logcdf) fallback). Compared against scipy genpareto.logsf.
        check_logccdf(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logsf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_icdf(self):
        check_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            lambda q, mu, sigma, xi: sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_icdf_selfconsistency(self):
        # cdf(icdf(q)) == q, no scipy reference needed.
        check_selfconsistency_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_logccdf_tail_is_stable(self):
        # Far in the heavy upper tail logcdf -> 0, so the generic
        # log1mexp(logcdf) survival fallback collapses; the direct logccdf stays
        # exact. Check it matches scipy out to x = 1e8.
        x = np.array([1e2, 1e4, 1e6, 1e8])
        for xi in (0.1, 0.3, 0.7):
            got = pm.logccdf(GenPareto.dist(mu=0.0, sigma=1.0, xi=xi), x).eval()
            ref = sp.genpareto.logsf(x, c=xi)
            np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0.0, 1.0, 0.0, None, np.log(2.0)),
            (2.0, 3.0, 0.5, None, 2.0 + 3.0 * (2.0**0.5 - 1) / 0.5),
            (0.0, 1.0, 0.0, (3,), np.full(3, np.log(2.0))),
        ],
    )
    def test_genpareto_support_point(self, mu, sigma, xi, size, expected):
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi, size=size)
        assert_support_point_is_expected(model, expected)

    def test_rng_matches_scipy(self):
        # Inverse-CDF sampling cannot match SciPy's draws element-wise (different
        # uniform stream), so compare distributionally via a KS test.
        for xi in (-0.3, 0.0, 0.4):
            draws = pm.draw(GenPareto.dist(mu=0.0, sigma=2.0, xi=xi, size=20_000), random_seed=11)
            ks = stats.kstest(draws, lambda v, xi=xi: sp.genpareto.cdf(v, c=xi, scale=2.0))
            assert ks.pvalue > 0.01


class TestGenPareto(BaseTestDistributionRandom):
    pymc_dist = GenPareto
    pymc_dist_params = {"mu": 0.0, "sigma": 2.0, "xi": 0.3}
    expected_rv_op_params = {"mu": 0.0, "sigma": 2.0, "xi": 0.3}
    tests_to_run = ["check_pymc_params_match_rv_op", "check_rv_size"]


class TestExtGenParetoClass:
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    pytestmark = _GPD_FPE_FILTERS

    def test_logp(self):
        check_logp(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logcdf(self):
        check_logcdf(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logcdf,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_logccdf(self):
        check_logccdf(
            ExtGenPareto,
            Rplus,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": Rplusbig,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_logccdf,
            decimal=select_by_precision(float64=6, float32=3),
        )

    def test_icdf(self):
        check_icdf(
            ExtGenPareto,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": SIGMA_ICDF,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            ref_ext_icdf,
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_icdf_selfconsistency(self):
        check_selfconsistency_icdf(
            ExtGenPareto,
            {
                "mu": Domain([0], edges=(None, None)),
                "sigma": SIGMA_ICDF,
                "xi": XI_DOMAIN,
                "kappa": KAPPA_DOMAIN,
            },
            decimal=select_by_precision(float64=5, float32=3),
        )

    def test_logccdf_reduces_to_gpd_at_kappa_one(self):
        value = np.linspace(0.05, 8.0, 40)
        for xi in (-0.3, 0.0, 0.4):
            ext = pm.logccdf(ExtGenPareto.dist(mu=0.0, sigma=1.5, xi=xi, kappa=1.0), value).eval()
            gpd = pm.logccdf(GenPareto.dist(mu=0.0, sigma=1.5, xi=xi), value).eval()
            np.testing.assert_allclose(ext, gpd, rtol=1e-12, atol=1e-12)

    def test_kappa_one_equals_gpd(self):
        # kappa = 1 collapses the carrier G(v) = v ** kappa to the identity.
        value = np.linspace(0.0, 8.0, 60)
        for xi in (-0.3, -1e-8, 0.0, 0.25, 0.8):
            ext = pm.logp(ExtGenPareto.dist(mu=0.0, sigma=1.5, xi=xi, kappa=1.0), value).eval()
            gpd = pm.logp(GenPareto.dist(mu=0.0, sigma=1.5, xi=xi), value).eval()
            np.testing.assert_allclose(ext, gpd, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "mu, sigma, xi, kappa, size",
        [
            (0.0, 1.0, 0.0, 2.0, None),
            (1.0, 2.0, 0.3, 0.5, None),
            (0.0, 1.0, -0.2, 3.0, (4,)),
        ],
    )
    def test_extgenpareto_support_point(self, mu, sigma, xi, kappa, size):
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa, size=size)
        expected = ref_ext_icdf(0.5, mu, sigma, xi, kappa)
        if size is not None:
            expected = np.full(size, expected)
        assert_support_point_is_expected(model, expected)

    @pytest.mark.parametrize("kappa", [0.5, 0.05, 0.01])
    def test_small_kappa_inverses_share_the_stable_excess(self, kappa):
        # icdf, the default transform's backward and support_point all invert the
        # carrier with the same log1mexp helper, so for small kappa they agree and
        # stay strictly above mu instead of collapsing onto it. A -log(-expm1(.))
        # form rounds the tiny GPD survival 1 - q ** (1/kappa) to 1, sending the
        # excess to 0 -> the lower endpoint -> a -inf initial logp. mu = 0 keeps
        # the (tiny) median representable; probed where ref_ext_icdf is itself exact.
        mu, sigma = 0.0, 1.0
        median = ref_ext_icdf(0.5, mu, sigma, 0.0, kappa)
        assert median > mu

        icdf_half = float(
            pm.icdf(ExtGenPareto.dist(mu=mu, sigma=sigma, xi=0.0, kappa=kappa), 0.5).eval()
        )
        np.testing.assert_allclose(icdf_half, median, rtol=1e-9)

        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=0.0, kappa=kappa)
        rv = model.free_RVs[0]
        tr = model.rvs_to_transforms[rv]
        backward0 = float(tr.backward(np.array(0.0), *rv.owner.inputs).eval())
        np.testing.assert_allclose(backward0, median, rtol=1e-9)  # y = logit(0.5) = 0

        assert_support_point_is_expected(model, np.array(median))
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_small_kappa_draws_do_not_collapse_to_mu(self):
        # The stable carrier inverse keeps small-kappa draws off the lower endpoint:
        # at kappa = 0.01 essentially none round to mu, whereas the unstable
        # -log(-expm1(.)) form sent the majority there (1 - u ** (1/kappa) -> 1).
        draws = pm.draw(
            ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.0, kappa=0.01, size=20_000),
            random_seed=7,
        )
        assert (draws >= 0.0).all()  # support is [mu, inf)
        assert np.mean(draws == 0.0) < 0.01

    @pytest.mark.parametrize("kappa", [1e-4, 1e-8, 1e-300])
    def test_support_point_falls_back_when_median_collapses(self, kappa):
        # When kappa is small enough that the ExtGPD median rounds onto mu (which
        # transforms to a -inf initial point), support_point falls back to the
        # underlying GPD median (excess = log 2) -- a higher quantile that is
        # representably interior for any kappa -- so the default initial logp is
        # finite over the whole kappa > 0 domain. mu = 2 makes even kappa = 1e-4
        # collapse (the tiny median excess is below ULP(mu)).
        mu, sigma = 2.0, 1.5
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=0.0, kappa=kappa)
        rv = model.free_RVs[0]
        sp = float(_support_point(rv).eval())
        expected_fallback = mu + sigma * np.log(2.0)  # GPD median, xi = 0
        np.testing.assert_allclose(sp, expected_fallback, rtol=1e-12)
        assert sp > mu
        # The headline fix: a finite default initial logp for every kappa > 0
        # (the forward map is logcdf - logccdf in log space, finite even when the
        # transformed point is deep, e.g. y ~ 691 for kappa = 1e-300).
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_rng_matches_distribution(self):
        # No SciPy equivalent: check that the empirical CDF (the model's own
        # logcdf) of the draws is Uniform(0, 1) via a KS test.
        for xi, kappa in ((-0.2, 0.5), (0.0, 2.0), (0.3, 3.0)):
            dist = ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=kappa, size=20_000)
            draws = pm.draw(dist, random_seed=7)
            u = np.exp(
                pm.logcdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=kappa), draws).eval()
            )
            assert stats.kstest(u, "uniform").pvalue > 0.01


class TestExtGenPareto(BaseTestDistributionRandom):
    pymc_dist = ExtGenPareto
    pymc_dist_params = {"mu": 0.0, "sigma": 1.5, "xi": 0.2, "kappa": 2.0}
    expected_rv_op_params = {"mu": 0.0, "sigma": 1.5, "xi": 0.2, "kappa": 2.0}
    tests_to_run = ["check_pymc_params_match_rv_op", "check_rv_size"]


class TestGenParetoBoundaries:
    """Explicit boundary / invalid-input behaviour for both GPD classes.

    These are exactly the cases an external review found regressing: ``x = inf``,
    ``q = 0``, ``q = 1`` (with ``xi < 0``), and ``sigma`` / ``kappa`` out of range.
    """

    pytestmark = _GPD_FPE_FILTERS

    def test_logp_logcdf_at_infinity(self):
        # density at +inf is 0 (logp -inf); CDF at +inf is 1 (logcdf 0). The
        # xi=0 path is the delicate one: xi*inf is nan without _safe_mul.
        for xi in (-0.5, 0.0, 0.5):
            assert pm.logp(GenPareto.dist(mu=0.0, sigma=1.0, xi=xi), np.inf).eval() == -np.inf
            assert pm.logcdf(GenPareto.dist(mu=0.0, sigma=1.0, xi=xi), np.inf).eval() == 0.0
            ext = ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=2.0)
            assert pm.logp(ext, np.inf).eval() == -np.inf
            assert pm.logcdf(ext, np.inf).eval() == 0.0

    def test_icdf_endpoints(self):
        # q=0 -> mu (lower endpoint); q=1 -> finite upper bound (xi<0) or +inf.
        for xi in (-0.5, 0.0, 0.5):
            expected_hi = 1.0 - 2.0 / xi if xi < 0 else np.inf
            assert pm.icdf(GenPareto.dist(mu=1.0, sigma=2.0, xi=xi), 0.0).eval() == 1.0
            np.testing.assert_allclose(
                pm.icdf(GenPareto.dist(mu=1.0, sigma=2.0, xi=xi), 1.0).eval(), expected_hi
            )
        # ExtGPD shares the same endpoints (carrier maps 0->0, 1->1).
        for xi, kappa in ((-0.5, 2.0), (0.0, 0.5), (0.5, 3.0)):
            expected_hi = 1.0 - 2.0 / xi if xi < 0 else np.inf
            ext = ExtGenPareto.dist(mu=1.0, sigma=2.0, xi=xi, kappa=kappa)
            assert pm.icdf(ext, 0.0).eval() == 1.0
            np.testing.assert_allclose(pm.icdf(ext, 1.0).eval(), expected_hi)

    def test_icdf_outside_unit_interval_is_nan(self):
        for q in (-0.1, 1.1):
            assert np.isnan(pm.icdf(GenPareto.dist(mu=0.0, sigma=1.0, xi=0.2), q).eval())
            ext = ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.2, kappa=2.0)
            assert np.isnan(pm.icdf(ext, q).eval())

    def test_invalid_sigma_raises(self):
        for sigma in (0.0, -1.0):
            with pytest.raises(ParameterValueError):
                pm.logp(GenPareto.dist(mu=0.0, sigma=sigma, xi=0.1), 1.0).eval()
            with pytest.raises(ParameterValueError):
                pm.logp(ExtGenPareto.dist(mu=0.0, sigma=sigma, xi=0.1, kappa=2.0), 1.0).eval()

    def test_invalid_kappa_raises(self):
        for kappa in (0.0, -1.0):
            with pytest.raises(ParameterValueError):
                pm.logp(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.1, kappa=kappa), 1.0).eval()

    @pytest.mark.parametrize("bad_xi", [np.nan, np.inf, -np.inf])
    def test_nonfinite_xi_propagates_consistently(self, bad_xi):
        # A non-finite ``xi`` must propagate as ``nan`` at *every* value -- not be
        # masked to the ``-inf`` / ``0`` of an out-of-support point ("valid
        # parameter, impossible value", which is a lie), and not be silently
        # turned into the xi = 0 exponential branch. This must hold even at
        # ``x = mu`` (where ``z = 0`` makes ``1 + xi z`` finite) and for
        # ``xi = +-inf``, so the guard keys on ``isfinite(xi)`` directly.
        # sanity: the xi = 0 branch is finite at x = 1, so a masked -inf would hide it
        assert np.isfinite(pm.logp(GenPareto.dist(mu=0.0, sigma=1.0, xi=0.0), 1.0).eval())
        for dist in (
            GenPareto.dist(mu=0.0, sigma=1.0, xi=bad_xi),
            ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=bad_xi, kappa=2.0),
        ):
            # in support (x=1), at the lower endpoint (x=mu=0), below (x=-1), above (+inf)
            for x in (1.0, 0.0, -1.0, np.inf):
                assert np.isnan(pm.logp(dist, x).eval())
                assert np.isnan(pm.logcdf(dist, x).eval())
                assert np.isnan(pm.logccdf(dist, x).eval())


class TestGenParetoHeavyTail:
    """Heavy-tail (xi > 1) coverage with *relative* tolerance.

    The generic ``check_icdf`` harness uses an absolute tolerance, so it cannot
    exercise xi > 1 -- the quantiles there are ~1e10 and dwarf any absolute
    bound. But xi > 1 is precisely the infinite-mean regime where the median
    ``support_point`` matters, so test it directly against SciPy with rtol.
    """

    pytestmark = _GPD_FPE_FILTERS

    @pytest.mark.parametrize("xi", [1.5, 3.0, 5.0])
    def test_logp_logcdf_icdf_match_scipy(self, xi):
        mu, sigma = 0.0, 1.3
        x = np.array([0.5, 2.0, 10.0, 1e4])
        np.testing.assert_allclose(
            pm.logp(GenPareto.dist(mu=mu, sigma=sigma, xi=xi), x).eval(),
            sp.genpareto.logpdf(x, c=xi, loc=mu, scale=sigma),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            pm.logcdf(GenPareto.dist(mu=mu, sigma=sigma, xi=xi), x).eval(),
            sp.genpareto.logcdf(x, c=xi, loc=mu, scale=sigma),
            rtol=1e-10,
        )
        q = np.array([0.1, 0.5, 0.9, 0.99, 0.999])
        np.testing.assert_allclose(
            pm.icdf(GenPareto.dist(mu=mu, sigma=sigma, xi=xi), q).eval(),
            sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
            rtol=1e-9,
        )

    @pytest.mark.parametrize("xi", [1.5, 3.0])
    def test_support_point_is_median_with_infinite_mean(self, xi):
        # mean is infinite for xi >= 1, so support_point must fall back to the
        # median (not the mean) -- check it equals the GPD median.
        mu, sigma = 1.0, 2.0
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi)
        expected = sp.genpareto.ppf(0.5, c=xi, loc=mu, scale=sigma)
        assert_support_point_is_expected(model, expected)

    def test_ext_support_point_median_infinite_mean(self):
        mu, sigma, xi, kappa = 0.0, 1.0, 2.0, 3.0
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa)
        expected = ref_ext_icdf(0.5, mu, sigma, xi, kappa)
        assert_support_point_is_expected(model, expected)

    def test_ext_logccdf_stable_in_far_tail(self):
        # ExtGPD survival = 1 - H**kappa. Routing through log H collapses to -inf
        # once H rounds to 1 in the far tail; the direct survival path stays
        # exact. For the exponential base (xi=0) the tail value is log(kappa) - x.
        for kappa in (0.5, 1.0, 2.5, 5.0):
            x = np.array([100.0, 300.0, 1000.0])
            got = pm.logccdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.0, kappa=kappa), x).eval()
            assert np.all(np.isfinite(got))
            np.testing.assert_allclose(got, np.log(kappa) - x, rtol=1e-9)

    @pytest.mark.parametrize("kappa", [10.0, 1e6, 1e20, 1e155, 1e300])
    def test_ext_logccdf_is_a_valid_log_probability_for_large_kappa(self, kappa):
        # A log survival probability is always <= 0. The tail branch must key on
        # kappa * S (not just the GPD survival), or large kappa makes
        # log(kappa) + (-x) positive. xi = 0 -> survival = 1 - (1 - e^-x)^kappa;
        # the exact tail value is log(kappa) - x when kappa * e^-x << 1. The huge
        # kappa cases (>= 1e155) guard the tail series: forming (kappa-1)(kappa-2)
        # overflows float64 (~1e310) and times an underflowed S^2 gives NaN, so the
        # series keeps only the first-order, single-kappa-factor term.
        x = np.array([40.0, 100.0, 1000.0])
        got = pm.logccdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.0, kappa=kappa), x).eval()
        assert np.all(got <= 0.0)
        # where kappa * e^-x is negligible, log survival == log(kappa) - x
        small = np.log(kappa) - x < -30.0
        np.testing.assert_allclose(got[small], (np.log(kappa) - x)[small], rtol=1e-9)

    @pytest.mark.parametrize("kappa", [1e-50, 1e-100, 1e-2])
    def test_ext_logccdf_small_kappa_in_the_body_matches_reference(self, kappa):
        # For small kappa, log(kappa) + a < -30 holds even where S_gpd ~ 1 (the body,
        # x = O(1)), so the tail branch must NOT trigger there: the leading behaviour
        # is log(1 - H**kappa) ~ log(kappa) + log(-log H), not log(kappa) + log(1 - H).
        # The gate keys on S_gpd being small too, so the body uses the exact generic
        # branch. xi = 0: 1 - H**kappa = -expm1(kappa * log1p(-exp(-x))).
        x = np.array([0.01, 0.1, 0.5, 2.0])
        got = pm.logccdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.0, kappa=kappa), x).eval()
        ref = np.log(-np.expm1(kappa * np.log1p(-np.exp(-x))))
        np.testing.assert_allclose(got, ref, rtol=1e-9)


class TestGenParetoBoundaryPrecision:
    """Precision of logp / gradient as ``value`` approaches the ``xi < 0`` wall.

    The family enters only through ``s = 1 + xi*z``; as ``value -> mu - sigma/xi``
    the float64 input loses the low bits of ``s`` (it cancels toward 0). The
    log-density *value* survives -- it is ``~|1 + 1/xi| * log s``, large magnitude,
    so the relative error stays near machine precision -- but gradient terms that
    scale like ``z/s`` inherit ``s``'s lost digits and are accurate only to
    ``~ulp/s``. The kappa gradient is exempt (its carrier term has no ``1/s``
    factor). These tests pin that behaviour against a 100-digit decimal reference
    built from the exact margin, so the limit is documented and regression-guarded.
    The fix that *recovers* the gradient is a margin-aware entry point (the caller
    supplies ``s`` without cancellation); see the class docstring note.
    """

    pytestmark = _GPD_FPE_FILTERS

    @pytest.mark.parametrize("xi", [-0.05, -0.3, -0.7])
    def test_boundary_logp_and_gradient_track_the_margin(self, xi):
        mu, sigma, kappa = 0.4, 1.3, 2.5
        eps = np.finfo(np.float64).eps
        margins = [1e-2, 1e-4, 1e-6, 1e-9, 1e-12]

        v, sig, xs, ks = (pt.dscalar(n) for n in ("v", "sig", "xs", "ks"))
        gpd_lp = gen_pareto_logp(v, mu, sig, xs)
        gpd_fn = pytensor.function(
            [v, sig, xs], [gpd_lp, pt.grad(gpd_lp, sig), pt.grad(gpd_lp, xs)]
        )
        ext_lp = ext_gen_pareto_logp(v, mu, sig, xs, ks)
        ext_fn = pytensor.function(
            [v, sig, xs, ks],
            [ext_lp, pt.grad(ext_lp, sig), pt.grad(ext_lp, xs), pt.grad(ext_lp, ks)],
        )

        cases = [
            ("GPD", gpd_fn, (), ["sigma", "xi"], None),
            ("ExtGPD", ext_fn, (kappa,), ["sigma", "xi", "kappa"], kappa),
        ]
        for label, fn, extra, names, kap in cases:
            params = {"mu": Decimal(mu), "sigma": Decimal(sigma), "xi": Decimal(xi)}
            if kap is not None:
                params["kappa"] = Decimal(kappa)

            prev_logp = np.inf  # margins go large s -> small s, so logp must decrease
            for s_target in margins:
                value = mu + sigma * ((s_target - 1) / xi)  # float64 input at margin s
                logp_f, *grads_f = (float(o) for o in fn(value, sigma, xi, *extra))
                logp_ref = _gpd_ref_logp(value, **params)

                assert np.isfinite(logp_f), (label, s_target)
                assert all(np.isfinite(g) for g in grads_f), (label, s_target)
                # density -> 0 at the wall for -1 < xi < 0, so logp is monotone in s
                assert logp_f < prev_logp, (label, s_target)
                prev_logp = logp_f
                # the *value* keeps near-machine relative accuracy at every margin
                assert abs((Decimal(logp_f) - logp_ref) / logp_ref) < 1e-4, (label, s_target)

                grad_bound = 100 * eps / s_target  # the achievable ~ulp/s limit
                for name, g in zip(names, grads_f):
                    g_ref = _gpd_ref_grad(value, params, name)
                    rel = abs((Decimal(g) - g_ref) / g_ref)
                    assert np.sign(g) == np.sign(float(g_ref)), (label, name, s_target)
                    if name == "kappa":
                        # no 1/s term -> stays exact even at the wall
                        assert rel < 1e-10, (label, name, s_target, float(rel))
                    else:
                        assert rel < grad_bound, (label, name, s_target, float(rel))

    def test_far_from_boundary_gradient_is_machine_accurate(self):
        # Contrast: away from the wall (s ~ 1) the gradient is exact to ~machine
        # precision -- the degradation above is margin-driven, not a generic defect.
        mu, sigma, xi = 0.4, 1.3, -0.3
        value = mu + sigma * ((0.5 - 1) / xi)  # s = 0.5, mid-support
        params = {"mu": Decimal(mu), "sigma": Decimal(sigma), "xi": Decimal(xi)}
        v, sig, xs = (pt.dscalar(n) for n in ("v", "sig", "xs"))
        lp = gen_pareto_logp(v, mu, sig, xs)
        fn = pytensor.function([v, sig, xs], [pt.grad(lp, sig), pt.grad(lp, xs)])
        g_sig, g_xi = (float(o) for o in fn(value, sigma, xi))
        for name, g in [("sigma", g_sig), ("xi", g_xi)]:
            g_ref = _gpd_ref_grad(value, params, name)
            assert abs((Decimal(g) - g_ref) / g_ref) < 1e-12


class TestGenParetoTransforms:
    """Both distributions register a default probability-integral transform.

    Without a transform, an unobserved (latent) GPD variable would be sampled on
    all of R, where every proposal below mu has -inf logp -- breaking NUTS. The
    transform maps to the (parameter-dependent) support, so sampling stays valid
    for both the heavy (xi >= 0) and bounded (xi < 0) regimes. A naive Interval
    transform would do this too, but its log-Jacobian is discontinuous in xi at
    0; the probability-integral transform (``y = logit(F(x))``) is C1 in every
    parameter, so it does not inject a gradient kink when xi is random.
    """

    pytestmark = _GPD_FPE_FILTERS

    def test_default_transform_is_registered(self):
        with pm.Model() as model:
            x = GenPareto("x", mu=5.0, sigma=1.0, xi=0.3)
            e = ExtGenPareto("e", mu=2.0, sigma=1.0, xi=-0.4, kappa=2.0)
        assert model.rvs_to_transforms[x] is not None
        assert model.rvs_to_transforms[e] is not None

    @pytest.mark.parametrize(
        "dist_kwargs, builder",
        [
            ({"mu": 0.0, "sigma": 1.5, "xi": -0.5}, GenPareto),
            ({"mu": 0.0, "sigma": 1.5, "xi": 0.0}, GenPareto),
            ({"mu": 0.0, "sigma": 1.5, "xi": 0.5}, GenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": -0.3, "kappa": 2.0}, ExtGenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3, "kappa": 2.0}, ExtGenPareto),
        ],
    )
    def test_transformed_density_integrates_to_one(self, dist_kwargs, builder):
        # The transform's log-Jacobian must be correct: exp(transformed logp)
        # integrates to 1 over the unconstrained line.
        from scipy.integrate import trapezoid

        with pm.Model() as model:
            builder("x", **dist_kwargs)
        y = model.value_vars[0]
        logp = pytensor.function([y], model.logp(sum=True))
        ys = np.linspace(-30, 30, 30001)
        density = np.exp(np.array([float(logp(yi)) for yi in ys]))
        np.testing.assert_allclose(trapezoid(density, ys), 1.0, atol=1e-3)

    @pytest.mark.parametrize(
        "dist_kwargs, builder",
        [
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3}, GenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": -0.5}, GenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3, "kappa": 2.0}, ExtGenPareto),
            ({"mu": 0.0, "sigma": 1.0, "xi": -0.3, "kappa": 0.5}, ExtGenPareto),
        ],
    )
    def test_transform_is_finite_and_logistic_in_the_tails(self, dist_kwargs, builder):
        # Regression for the saturation bug: sigmoid(y) rounds to exactly 1 for
        # y >= 37, so a naive backward via icdf(sigmoid(y)) returned inf/nan on
        # perfectly valid unconstrained values around y = 37. Working in survival
        # space (m = softplus(y)) keeps the transform finite far past that, equal
        # to the Logistic log-density (the PIT image of the GPD prior), with
        # forward(backward(y)) round-tripping. The probed range stays inside what
        # float64 can represent for *every* parametrization (the bounded xi < 0
        # case saturates at its wall only beyond |y| ~ 70 -- see
        # ``test_unbounded_transform_is_finite_arbitrarily_far`` for the xi >= 0
        # case, which has no wall and stays finite to |y| ~ 1000). For the
        # bounded xi < 0 case the round-trip CDF (``log1mexp`` near the wall)
        # loses accuracy beyond |y| ~ 48, so the probe stays at |y| <= 45 -- well
        # past the y = 37 saturation bug this guards.
        with pm.Model() as model:
            x = builder("x", **dist_kwargs)
        yv = model.value_vars[0]
        transformed_logp = pytensor.function([yv], model.logp(sum=True))
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))

        for y in (-45.0, -40.0, -37.0, 37.0, 40.0, 45.0):
            lp = float(transformed_logp(y))
            logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
            assert np.isfinite(lp)
            np.testing.assert_allclose(lp, logistic, atol=1e-6)
            np.testing.assert_allclose(float(roundtrip(y)), y, atol=1e-6)

    @pytest.mark.parametrize(
        "dist_kwargs, builder, y_max",
        [
            # Exponential tail (xi = 0, unbounded, no overflow): exact very far out.
            # The ExtGenPareto probe at y = 1000 reaches past y ~ 745, where the
            # naive log-F recovery of the excess underflowed (S_F -> 0, m -> inf);
            # recovering it from the survival side keeps it finite and exact.
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.0}, GenPareto, 1000.0),
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 2.0}, ExtGenPareto, 1000.0),
            # Mild heavy tail: the quantile ~ exp(xi * m) overflows only at
            # y ~ 709 / xi (~2363 here), so 1000 is well inside for both families.
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3}, GenPareto, 400.0),
            ({"mu": 0.0, "sigma": 1.0, "xi": 0.3, "kappa": 2.0}, ExtGenPareto, 1000.0),
        ],
    )
    def test_transform_exact_deep_into_the_tail(self, dist_kwargs, builder, y_max):
        # Where the quantile is representable, the transformed density is exactly
        # Logistic arbitrarily far out (xi >= 0 has no upper wall; the heavy-tail
        # cases stay below the y ~ 709 / xi overflow boundary).
        with pm.Model() as model:
            builder("x", **dist_kwargs)
        yv = model.value_vars[0]
        transformed_logp = pytensor.function([yv], model.logp(sum=True))
        for y in (100.0, y_max / 2, y_max):
            lp = float(transformed_logp(y))
            logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
            assert np.isfinite(lp)
            np.testing.assert_allclose(lp, logistic, atol=1e-6)

    @pytest.mark.parametrize("xi", [5.0, 3.0])
    def test_heavy_tail_transform_saturates_past_709_over_xi(self, xi):
        # Below the heavy-tail point y ~ 709/xi the transformed logp is exactly
        # Logistic(y). Past it the recovered quantile x = backward(y) ~ exp(xi*m)
        # overflows float64 to +inf, so logp(backward) = -inf; the transformed logp
        # then saturates to -inf (finite-graph robust -- it is *not* a NaN, since
        # log_jac_det floors that -inf -- but it is no longer the true Logistic
        # value, the honest representability limit). This y is utterly unreachable
        # (a tail probability of ~e^-(709/xi)).
        with pm.Model() as model:
            x = GenPareto("x", mu=0.0, sigma=1.0, xi=xi)
        yv = model.value_vars[0]
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        transformed_logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        boundary = 709.0 / xi
        y_inside = boundary * 0.8
        lp = float(transformed_logp(y_inside))
        assert np.isfinite(lp)
        np.testing.assert_allclose(
            lp, -np.logaddexp(0.0, y_inside) - np.logaddexp(0.0, -y_inside), atol=1e-6
        )
        assert np.isfinite(float(backward(y_inside)))
        # Past the float64 ceiling: quantile overflows, transformed logp -> -inf
        # (not NaN).
        assert not np.isfinite(float(backward(boundary * 1.5)))
        assert float(transformed_logp(boundary * 1.5)) == -np.inf

    def test_transformed_logp_robust_in_unoptimized_mode(self):
        # The transformed logp must not depend on the optimizer cancelling
        # logp(backward) against log_jac_det. In an unoptimized (fast_compile) graph
        # the actual numbers are evaluated; the construction must still avoid
        # +inf - inf = NaN. Over the whole sampler-reachable range, for bounded /
        # unbounded GPD and small-kappa ExtGPD (lower-tail quantile collapsing onto
        # mu), the transformed logp is finite and exactly Logistic -- no NaN.
        fast = pytensor.compile.mode.Mode(linker="py", optimizer="fast_compile")
        cases = [
            (GenPareto, {"mu": 2.0, "sigma": 1.5, "xi": -0.5}),
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.3}),
            (ExtGenPareto, {"mu": 2.0, "sigma": 1.0, "xi": 0.0, "kappa": 0.0067}),
            (ExtGenPareto, {"mu": 0.0, "sigma": 1e-50, "xi": 0.0, "kappa": 0.0067}),
        ]
        for builder, kw in cases:
            with pm.Model() as model:
                builder("x", **kw)
            fn = pytensor.function([model.value_vars[0]], model.logp(sum=True), mode=fast)
            for y in np.linspace(-25.0, 25.0, 51):
                lp = float(fn(y))
                assert not np.isnan(lp), (builder.__name__, kw, y)
                logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
                np.testing.assert_allclose(lp, logistic, atol=1e-3)

    @pytest.mark.parametrize(
        "kappa, ys",
        [
            (1e8, (-30.0, 0.0, 30.0)),
            (1e4, (-30.0, 0.0, 30.0)),
            (1.0, (-40.0, 0.0, 40.0)),
            (0.1, (-30.0, 0.0, 40.0)),
            # Upper tail with kappa < 1 is exactly where a y-only tail switch
            # returned a negative excess (x < mu); here it is exact.
            (1e-2, (-5.0, 0.0, 40.0)),
        ],
    )
    def test_extgenpareto_transform_is_exact_across_kappa(self, kappa, ys):
        # The ExtGPD inverse must depend on kappa, not just on y: the excess is
        # recovered via -log1mexp(log F_ext / kappa), whose log1p branch keeps a
        # tiny GPD survival from collapsing the excess to 0 (a -log(-expm1(.)) form
        # would). The map is exactly Logistic across ten orders of magnitude in
        # kappa at ordinary tail depths. Probe depths are kept inside each kappa's
        # exact domain (small kappa has essentially no representable *lower* tail --
        # the carrier H = F_ext ** (1/kappa) underflows once |y| / kappa >~ 745 --
        # which is the distribution concentrating toward mu, not a defect).
        with pm.Model() as model:
            ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.0, kappa=kappa)
        yv = model.value_vars[0]
        transformed_logp = pytensor.function([yv], model.logp(sum=True))
        tr = model.rvs_to_transforms[model.free_RVs[0]]
        inputs = model.free_RVs[0].owner.inputs
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
        for y in ys:
            lp = float(transformed_logp(y))
            logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
            assert np.isfinite(lp)
            np.testing.assert_allclose(lp, logistic, atol=1e-6)
            np.testing.assert_allclose(float(roundtrip(y)), y, atol=1e-6)

    @pytest.mark.parametrize("kappa", [1e-2, 1e-20, 1e-100, 1e-300])
    def test_extgenpareto_transform_finite_logistic_for_small_kappa(self, kappa):
        # Regression for the small-kappa transform NaN. For kappa < ~|y|/745 the
        # carrier excess underflows and the quantile collapses onto mu, where a
        # kappa < 1 density diverges (logp = +inf). The transform still returns the
        # exact Logistic(y) transformed logp -- finite -- rather than the +inf - inf
        # NaN a logcdf(x) + logccdf(x) route produced (kappa = 1e-2 at y = -10 was
        # already in this regime, not just absurdly small kappa). The recovered
        # quantile stays finite and in support (>= mu).
        mu = 2.0
        with pm.Model() as model:
            x = ExtGenPareto("x", mu=mu, sigma=1.0, xi=0.0, kappa=kappa)
        yv = model.value_vars[0]
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        for y in (-30.0, -10.0, 0.0, 30.0, 80.0):
            lp = float(logp(y))
            assert np.isfinite(lp)
            np.testing.assert_allclose(lp, -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y), atol=1e-6)
            xb = float(backward(y))
            assert np.isfinite(xb) and xb >= mu  # in support [mu, inf)

    def test_jacobian_gradient_is_continuous_through_xi_zero(self):
        # The headline reason for the probability-integral transform: with xi a
        # random variable, the transformed logp must be C1 in xi across 0. An
        # Interval transform fails this (its Jacobian jumps by ~1e12 at xi = 0).
        with pm.Model() as model:
            xi = pm.Normal("xi", 0.0, 1.0)
            GenPareto("x", mu=0.0, sigma=1.0, xi=xi)
        val_xi = next(v for v in model.value_vars if v.name == "xi")
        val_x = next(v for v in model.value_vars if v.name != "xi")
        logp = model.logp(sum=True)
        fn = pytensor.function([val_xi, val_x], pt.grad(logp, val_xi), on_unused_input="ignore")
        grad_minus = float(fn(-1e-6, 0.5))
        grad_plus = float(fn(1e-6, 0.5))
        assert abs(grad_minus - grad_plus) < 1e-3

    @pytest.mark.parametrize(
        "xi, mu, sigma",
        [(0.3, 5.0, 1.0), (0.0, 0.0, 2.0), (-0.5, 0.0, 1.0)],
    )
    def test_latent_sampling_stays_in_support(self, xi, mu, sigma):
        with pm.Model() as model:
            GenPareto("x", mu=mu, sigma=sigma, xi=xi)
            idata = pm.sample(
                200,
                tune=300,
                chains=2,
                progressbar=False,
                random_seed=1,
                compute_convergence_checks=False,
            )
        xs = idata.posterior["x"].values
        assert np.all(xs >= mu - 1e-9)
        if xi < 0:
            assert np.all(xs <= mu - sigma / xi + 1e-9)  # finite upper wall
        assert int(idata.sample_stats.diverging.values.sum()) == 0

    def test_observed_is_unaffected(self):
        # Observed data is fixed, so the transform must not change its logp.
        data = np.array([6.0, 7.0, 8.0])
        with pm.Model() as model:
            GenPareto("obs", mu=5.0, sigma=1.0, xi=0.2, observed=data)
        assert np.isfinite(model.compile_logp()({}))


class TestGenParetoSmoothShapeLimit:
    """The headline property: the logp is C1 in xi through the xi = 0 limit.

    A ``switch(isclose(xi, 0), ...)`` reparametrization is only C0 there -- the
    gradient jumps -- which drives NUTS divergences for data that pulls xi near
    zero. Routing the whole family through ``log1p(xi z) / xi`` keeps value and
    gradient continuous.
    """

    @pytest.mark.parametrize(
        "dist, params",
        [
            (GenPareto, {}),
            (ExtGenPareto, {"kappa": 2.0}),
        ],
    )
    def test_logp_gradient_is_continuous_through_xi_zero(self, dist, params):
        xi = pt.dscalar("xi")
        logp = pm.logp(dist.dist(mu=0.0, sigma=1.0, xi=xi, **params), pt.constant(2.3))
        fn = pytensor.function([xi], [logp, pt.grad(logp, xi)], on_unused_input="ignore")

        _, grad0 = fn(0.0)
        assert np.isfinite(grad0)
        # One-sided gradients straddling 0 agree with the value at 0 (no kink).
        _, grad_minus = fn(-1e-7)
        _, grad_plus = fn(1e-7)
        assert abs(grad_minus - grad0) < 1e-5
        assert abs(grad_plus - grad0) < 1e-5
        # And the gradient matches a central finite difference of the value.
        h = 1e-5
        fd = (fn(h)[0] - fn(-h)[0]) / (2 * h)
        assert abs(grad0 - fd) < 1e-5

    def test_value_continuous_through_xi_zero(self):
        # The reparametrization must also be accurate (not just smooth) for tiny
        # xi, where (1 + xi z) ** (-1/xi) loses all precision. Compare to the
        # exact exponential limit at xi = 0.
        value = np.linspace(0.05, 6.0, 50)
        lp0 = pm.logp(GenPareto.dist(mu=0.0, sigma=2.0, xi=0.0), value).eval()
        for xi in (1e-9, -1e-9, 1e-6):
            lp = pm.logp(GenPareto.dist(mu=0.0, sigma=2.0, xi=xi), value).eval()
            np.testing.assert_allclose(lp, lp0, atol=1e-5)
        np.testing.assert_allclose(lp0, sp.expon.logpdf(value, scale=2.0), atol=1e-12)
