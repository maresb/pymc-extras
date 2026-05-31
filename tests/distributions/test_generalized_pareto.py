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
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_icdf,
    check_logcdf,
    check_logp,
    select_by_precision,
)
from scipy import stats

from pymc_extras.distributions import ExtGenPareto, GenPareto

# The GPD family legitimately evaluates to +/-inf at the (measure-zero) support
# boundary and at probabilities 0/1 -- NumPy flags those as divide-by-zero /
# invalid / overflow. They are the correct boundary values (the numerical
# comparisons below still validate them), so silence the FPE warnings that the
# repo-wide ``filterwarnings = error`` would otherwise promote to test errors.
pytestmark = [
    pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning"),
]

# Shape grid spanning the exponential limit (xi = 0) and both tails. The first
# and last entries are treated as domain edges by the pymc test harness.
XI_DOMAIN = Domain([-1, -0.5, -0.1, 0, 0.1, 0.5, 1], dtype="float64")
KAPPA_DOMAIN = Domain([0.25, 0.5, 1, 2, 5], dtype="float64")
# ``check_icdf`` compares absolute values, so a bounded scale keeps the
# heavy-tail quantiles from blowing past the tolerance. sigma is a pure linear
# scale of the quantile, so this still exercises it fully.
SIGMA_ICDF = Domain([0.1, 0.5, 1.0, 2.0], dtype="float64")


def _ext_gpd_log_H(z, xi):
    """GPD log-CDF reference (NumPy), in support."""
    return sp.genpareto.logcdf(z, c=xi)


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


def ref_ext_icdf(q, mu, sigma, xi, kappa):
    # G^{-1}(q) = H^{-1}(q ** (1/kappa)).
    return sp.genpareto.ppf(q ** (1 / kappa), c=xi, loc=mu, scale=sigma)


class TestGenParetoClass:
    """Wrapper so the distribution tests can later be lifted into PyMC proper."""

    def test_logp(self):
        check_logp(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logpdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
            skip_paramdomain_outside_edge_test=True,
        )

    def test_logcdf(self):
        check_logcdf(
            GenPareto,
            Rplusbig,
            {"mu": Domain([0], edges=(None, None)), "sigma": Rplusbig, "xi": XI_DOMAIN},
            lambda value, mu, sigma, xi: sp.genpareto.logcdf(value, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=6, float32=3),
            skip_paramdomain_outside_edge_test=True,
        )

    def test_icdf(self):
        check_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            lambda q, mu, sigma, xi: sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=5, float32=3),
            skip_paramdomain_outside_edge_test=True,
        )

    @pytest.mark.parametrize(
        "mu, sigma, xi, size, expected",
        [
            (0.0, 1.0, 0.0, None, np.log(2.0)),
            (2.0, 3.0, 0.5, None, 2.0 + 3.0 * (2.0**0.5 - 1) / 0.5),
            (0.0, 1.0, 0.0, (3,), np.full(3, np.log(2.0))),
        ],
    )
    def test_support_point(self, mu, sigma, xi, size, expected):
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
            skip_paramdomain_outside_edge_test=True,
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
            skip_paramdomain_outside_edge_test=True,
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
            skip_paramdomain_outside_edge_test=True,
        )

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
    def test_support_point(self, mu, sigma, xi, kappa, size):
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=xi, kappa=kappa, size=size)
        expected = ref_ext_icdf(0.5, mu, sigma, xi, kappa)
        if size is not None:
            expected = np.full(size, expected)
        assert_support_point_is_expected(model, expected)

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


class TestSmoothShapeLimit:
    """The headline property: the logp is C1 in xi through the xi = 0 limit.

    A ``switch(isclose(xi, 0), ...)`` reparametrization is only C0 there -- the
    gradient jumps -- which drives NUTS divergences for data that pulls xi near
    zero. Routing through ``log1p(xi z) / xi`` makes value and gradient
    continuous.
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
