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
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

# general imports
import pytest
import scipy.stats.distributions as sp


# test support imports from pymc
from pymc.logprob.utils import ParameterValueError
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    R,
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_icdf,
    check_logcdf,
    check_logp,
    seeded_scipy_distribution_builder,
    select_by_precision,
)
from scipy import stats

# the distributions to be tested
from pymc_extras.distributions import Chi, ExtGenPareto, GenExtreme, GenPareto, Maxwell

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


def ref_ext_icdf(q, mu, sigma, xi, kappa):
    # G^{-1}(q) = H^{-1}(q ** (1/kappa)).
    return sp.genpareto.ppf(q ** (1 / kappa), c=xi, loc=mu, scale=sigma)


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

    def test_icdf(self):
        check_icdf(
            GenPareto,
            {"mu": Domain([0], edges=(None, None)), "sigma": SIGMA_ICDF, "xi": XI_DOMAIN},
            lambda q, mu, sigma, xi: sp.genpareto.ppf(q, c=xi, loc=mu, scale=sigma),
            decimal=select_by_precision(float64=5, float32=3),
        )

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
