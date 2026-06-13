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
    Rplus,
    Rplusbig,
    assert_support_point_is_expected,
    check_icdf,
    check_logccdf,
    check_logcdf,
    check_logp,
    check_selfconsistency_icdf,
    select_by_precision,
)
from scipy import stats

# the distributions to be tested
from pymc_extras.distributions import (
    ExtGenPareto,
    GenPareto,
    _pytensor_extgenpareto,
    _pytensor_genpareto,
)
from pymc_extras.distributions._pymc_extgenpareto import _ExtGenParetoPIT
from pymc_extras.distributions._pytensor_extgenpareto import logpdf as ext_gen_pareto_logp
from pymc_extras.distributions._pytensor_genpareto import logpdf as gen_pareto_logp

# xi is unconstrained, so (None, None) edges skip the harness's invalid-edge probe.
# Bounded to (-1, 1]: xi <= -1 diverges at the wall (see TestGenParetoBoundaries) and
# xi > 1 pushes the q=0.99 quantile past check_icdf's absolute tolerance.
XI_DOMAIN = Domain([-0.9, -0.5, -0.1, 0, 0.1, 0.5, 1], dtype="float64", edges=(None, None))
# Leading 0 lets the harness probe the invalid kappa <= 0; trailing inf is unbounded.
KAPPA_DOMAIN = Domain([0, 0.25, 0.5, 1, 2, 5, np.inf], dtype="float64")
# sigma capped so check_icdf's absolute tolerance holds on the heavy-tail quantiles.
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

    @pytest.mark.parametrize("kappa", [0.5, 0.01])
    def test_small_kappa_inverses_share_the_stable_excess(self, kappa):
        # icdf, the transform's backward, and support_point share one log1mexp carrier
        # inverse, so for small kappa they agree and stay strictly above mu instead of
        # collapsing onto it (a -log(-expm1(.)) form would round the tiny survival to 1,
        # sending the excess to 0 and the initial logp to -inf).
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

    @pytest.mark.parametrize("kappa", [1e-4, 1e-300])
    def test_support_point_falls_back_when_median_collapses(self, kappa):
        # When the ExtGPD median rounds onto mu (a -inf initial point), support_point
        # falls back to the GPD median (excess = log 2), interior for any kappa. mu = 2
        # makes even kappa = 1e-4 collapse (median excess below ULP(mu)).
        mu, sigma = 2.0, 1.5
        with pm.Model() as model:
            ExtGenPareto("x", mu=mu, sigma=sigma, xi=0.0, kappa=kappa)
        rv = model.free_RVs[0]
        sp = float(_support_point(rv).eval())
        expected_fallback = mu + sigma * np.log(2.0)  # GPD median, xi = 0
        np.testing.assert_allclose(sp, expected_fallback, rtol=1e-12)
        assert sp > mu
        # A finite default initial logp for every kappa > 0 (the forward map is
        # logcdf - logccdf in log space, finite even when the transformed point is
        # deep, e.g. y ~ 691 for kappa = 1e-300).
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

    Covers ``x = inf``, ``q = 0``, ``q = 1`` (with ``xi < 0``), and ``sigma`` /
    ``kappa`` out of range.
    """

    def test_logp_logcdf_at_infinity(self):
        # density at +inf is 0 (logp -inf); CDF at +inf is 1 (logcdf 0). The
        # xi=0 path is the delicate one: 0*inf is nan, so the +inf tail is pinned
        # explicitly. Batch the three xi into one dist so each method compiles once.
        xi = np.array([-0.5, 0.0, 0.5])
        for dist in (
            GenPareto.dist(mu=0.0, sigma=1.0, xi=xi),
            ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=xi, kappa=2.0),
        ):
            assert np.all(pm.logp(dist, np.inf).eval() == -np.inf)
            assert np.all(pm.logcdf(dist, np.inf).eval() == 0.0)

    def test_icdf_endpoints(self):
        # q=0 -> mu (lower endpoint); q=1 -> finite upper bound (xi<0) or +inf.
        # Batch the three xi into one dist so each endpoint compiles once, not per xi.
        xi = np.array([-0.5, 0.0, 0.5])
        with np.errstate(divide="ignore"):  # xi = 0 -> inf upper bound, not a warning
            expected_hi = np.where(xi < 0, 1.0 - 2.0 / xi, np.inf)
        # ExtGPD shares the same endpoints (carrier maps 0->0, 1->1).
        gpd = GenPareto.dist(mu=1.0, sigma=2.0, xi=xi)
        ext = ExtGenPareto.dist(mu=1.0, sigma=2.0, xi=xi, kappa=np.array([2.0, 0.5, 3.0]))
        for dist in (gpd, ext):
            assert np.all(pm.icdf(dist, 0.0).eval() == 1.0)
            np.testing.assert_allclose(pm.icdf(dist, 1.0).eval(), expected_hi)

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
        # kappa <= 0 and kappa = nan both fail the kappa > 0 check and raise.
        for kappa in (0.0, -1.0, np.nan):
            with pytest.raises(ParameterValueError):
                pm.logp(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.1, kappa=kappa), 1.0).eval()


class TestGenParetoHeavyTail:
    """Heavy-tail (xi > 1) coverage with *relative* tolerance.

    The generic ``check_icdf`` harness uses an absolute tolerance, so it cannot
    exercise xi > 1 -- the quantiles there are ~1e10 and dwarf any absolute
    bound. But xi > 1 is precisely the infinite-mean regime where the median
    ``support_point`` matters, so test it directly against SciPy with rtol.
    """

    @pytest.mark.parametrize("xi", [1.5, 5.0])
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

    @pytest.mark.parametrize("kappa", [10.0, 1e155, 1e300])
    def test_ext_logccdf_is_a_valid_log_probability_for_large_kappa(self, kappa):
        # A log survival probability is always <= 0. The tail branch must key on
        # kappa * S (not just the GPD survival), or large kappa makes
        # log(kappa) + (-x) positive. xi = 0 -> survival = 1 - (1 - e^-x)^kappa;
        # the exact tail value is log(kappa) - x when kappa * e^-x << 1. The huge
        # kappa cases (>= 1e155) guard the tail series: the kappa^k coefficients
        # (e.g. (kappa-1)(kappa-2)) would overflow float64 (~1e310) and multiply an
        # underflowed S^2 -> NaN, so the series is written in r = kappa*S and s = S.
        x = np.array([40.0, 100.0, 1000.0])
        got = pm.logccdf(ExtGenPareto.dist(mu=0.0, sigma=1.0, xi=0.0, kappa=kappa), x).eval()
        assert np.all(got <= 0.0)
        # where kappa * e^-x is negligible, log survival == log(kappa) - x
        small = np.log(kappa) - x < -30.0
        np.testing.assert_allclose(got[small], (np.log(kappa) - x)[small], rtol=1e-9)

    @pytest.mark.parametrize("kappa", [1e-2, 1e-100])
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
    built from the exact margin. (A caller that forms ``s = 1 + xi*z`` directly,
    without the cancellation, recovers the gradient.)
    """

    def test_boundary_logp_value_holds_but_gradient_tracks_the_margin(self):
        # One representative xi over a margin sweep is enough to pin the limit (the
        # s-cancellation mechanism is xi-independent): the largest margin (s = 1e-2,
        # bound ~ulp/1e-2 ~ 2e-12) is the machine-accurate anchor away from the wall;
        # the smallest (s = 1e-12) is the ~ulp/s limit at it.
        mu, sigma, xi, kappa = 0.4, 1.3, -0.3, 2.5
        eps = np.finfo(np.float64).eps
        margins = [1e-2, 1e-6, 1e-12]

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
        # The integrand is the smooth, exactly-Logistic transformed density, so 2001
        # points over +-30 integrate to ~1e-13 (tail truncation dominates).
        ys = np.linspace(-30, 30, 2001)
        density = np.exp(np.array([float(logp(yi)) for yi in ys]))
        np.testing.assert_allclose(trapezoid(density, ys), 1.0, atol=1e-3)

    @pytest.mark.parametrize(
        "builder, kwargs, ys, roundtrip",
        [
            # bounded xi<0: the y~37 sigmoid-saturation point; round-trips to ~45.
            (GenPareto, {"mu": 0.0, "sigma": 1.5, "xi": -0.5}, (-45.0, -37.0, 37.0, 45.0), True),
            # unbounded xi=0: exact arbitrarily far out.
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.0}, (-30.0, 100.0, 1000.0), False),
            # heavy xi>0: below the y~709/xi quantile overflow.
            (GenPareto, {"mu": 0.0, "sigma": 1.0, "xi": 0.3}, (37.0, 200.0, 400.0), False),
            # bounded ExtGPD: carrier on the moving xi<0 wall.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": -0.3, "kappa": 0.5},
                (-45.0, -37.0, 37.0, 45.0),
                True,
            ),
            # ExtGPD deep tail: the y~745 log-F underflow, recovered survival-side.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 2.0},
                (-30.0, 100.0, 1000.0),
                False,
            ),
            # ExtGPD large kappa: the inverse must depend on kappa.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e8},
                (-30.0, 0.0, 30.0),
                True,
            ),
            # ExtGPD kappa<1: the small-kappa inverse keeps a tiny survival off 0.
            (
                ExtGenPareto,
                {"mu": 0.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e-2},
                (-5.0, 0.0, 40.0),
                True,
            ),
            # ExtGPD collapse: quantile collapses onto mu, still exact finite Logistic.
            (
                ExtGenPareto,
                {"mu": 2.0, "sigma": 1.0, "xi": 0.0, "kappa": 1e-300},
                (-30.0, 0.0, 80.0),
                False,
            ),
        ],
        ids=[
            "bounded-gpd",
            "unbounded-gpd",
            "heavy-gpd",
            "bounded-ext",
            "unbounded-ext-deep",
            "ext-large-kappa",
            "ext-small-kappa",
            "ext-collapse",
        ],
    )
    def test_transformed_logp_is_logistic_where_representable(self, builder, kwargs, ys, roundtrip):
        # Where the quantile is representable the transformed logp equals Logistic(y),
        # x stays in support (x >= mu), and the map round-trips. Covers the sigmoid
        # saturation (y ~ 37), the deep tail, and ten orders of magnitude in kappa.
        mu = kwargs["mu"]
        with pm.Model() as model:
            x = builder("x", **kwargs)
        yv = model.value_vars[0]
        inputs = x.owner.inputs
        tr = model.rvs_to_transforms[x]
        logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        forward_backward = (
            pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
            if roundtrip
            else None
        )
        for y in ys:
            lp = float(logp(y))
            assert np.isfinite(lp), (kwargs, y)
            np.testing.assert_allclose(lp, -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y), atol=1e-6)
            xb = float(backward(y))
            assert np.isfinite(xb) and xb >= mu, (kwargs, y)  # support is [mu, ...)
            if forward_backward is not None:
                np.testing.assert_allclose(float(forward_backward(y)), y, atol=1e-6)

    def test_transformed_logp_robust_in_unoptimized_mode(self):
        # The transformed logp must not rely on the optimizer cancelling logp(backward)
        # against log_jac_det: in fast_compile the numbers are evaluated, and over the
        # whole reachable range (incl. the small-kappa collapse) it stays finite Logistic.
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

    def test_small_kappa_collapse_saturates_with_exact_density(self):
        # For kappa << 1 the ExtGPD median sits ~0.5 ** (1/kappa) below mu, under
        # ulp(mu), so the whole bulk is a numerical point mass: distinct y all map to
        # the same floored x. The map is therefore NOT injective here -- but the
        # sampled density stays exactly Logistic and the readout stays in support.
        mu = 2.0
        with pm.Model() as model:
            x = ExtGenPareto("x", mu=mu, sigma=1.0, xi=0.0, kappa=0.01)
        yv = model.value_vars[0]
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        logp = pytensor.function([yv], model.logp(sum=True))
        backward = pytensor.function([yv], tr.backward(yv, *inputs))
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
        ys = (-10.0, -5.0, 0.0)
        xs = [float(backward(y)) for y in ys]
        rts = [float(roundtrip(y)) for y in ys]
        # saturation: distinct y collapse onto one floored x, just above mu (in support)
        assert xs[0] == xs[1] == xs[2] > mu
        # so it is not injective here -- forward(backward(.)) is constant, not identity
        assert rts[0] == rts[1] == rts[2]
        # yet the sampled density is still exactly Logistic at each y
        for y in ys:
            np.testing.assert_allclose(
                float(logp(y)), -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y), atol=1e-2
            )

    def test_transform_finite_under_float32(self):
        # dtype-aware floor: the small-kappa lower-tail collapse must not NaN under
        # float32, where a literal 1e-300 floor underflows to 0 (leaving logp = +inf).
        fast = pytensor.compile.mode.Mode(linker="py", optimizer="fast_compile")
        with pytensor.config.change_flags(floatX="float32"):
            with pm.Model() as model:
                ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.0, kappa=1e-20)
            fn = pytensor.function([model.value_vars[0]], model.logp(sum=True), mode=fast)
            for y in (-10.0, 0.0, 10.0):
                lp = float(fn(np.float32(y)))
                assert not np.isnan(lp)
                logistic = -np.logaddexp(0.0, y) - np.logaddexp(0.0, -y)
                np.testing.assert_allclose(lp, logistic, atol=1e-2)

    def test_excess_from_y_upper_tail_is_finite_under_float32(self):
        # The bulk/tail crossover must track the dtype's exponent range: with a
        # hard-coded float64 cutoff (700), float32 y in ~[88, 700) routes through the
        # bulk branch where log F_ext has already underflowed, so the excess (and the
        # transformed logp) blow up to +inf well inside the reachable range.
        y = pt.scalar("y", dtype="float32")
        excess = _ExtGenParetoPIT._excess_from_y(
            y, np.float32(0.0), np.float32(1.0), np.float32(0.3), np.float32(2.0)
        )
        assert excess.dtype == "float32"
        fn = pytensor.function([y], excess)
        for yi in (90.0, 200.0, 700.0, 5000.0):
            m = float(fn(np.float32(yi)))
            assert np.isfinite(m)
            # m ~ y + log(kappa) far out in the tail (S_ext / kappa << 1 there).
            np.testing.assert_allclose(m, yi + np.log(2.0), rtol=1e-3)

    def test_excess_from_y_resolves_subnormal_kappa_tail(self):
        # For subnormal kappa, exp(-y)/kappa is O(1) in the tail, so the carrier
        # survival must be inverted exactly (m = -log(1 - F_ext ** (1/kappa))) -- a
        # S_ext/kappa << 1 asymptotic returns a negative "excess" and collapses
        # backward onto the floor. Reference m(y=710, kappa=4e-309, xi=0) = 0.3953903.
        y = pt.dscalar("y")
        excess = float(
            pytensor.function([y], _ExtGenParetoPIT._excess_from_y(y, 0.0, 1.0, 0.0, 4e-309))(710.0)
        )
        assert excess > 0.0
        np.testing.assert_allclose(excess, 0.395390331, rtol=1e-4)
        with pm.Model() as model:
            x = ExtGenPareto("x", mu=0.0, sigma=1.0, xi=0.0, kappa=4e-309)
        tr = model.rvs_to_transforms[x]
        inputs = x.owner.inputs
        yv = model.value_vars[0]
        roundtrip = pytensor.function([yv], tr.forward(tr.backward(yv, *inputs), *inputs))
        np.testing.assert_allclose(float(roundtrip(710.0)), 710.0, rtol=1e-5)

    def test_jacobian_gradient_is_continuous_through_xi_zero(self):
        # Why a PIT and not an Interval transform: with random xi the transformed
        # logp must be C1 in xi across 0; an Interval transform's Jacobian jumps by
        # ~1e12 at xi = 0, driving divergences.
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


class TestGenParetoSmoothShapeLimit:
    """The logp is C1 in xi through the xi = 0 limit.

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


class TestPyTensorFunctionalAPI:
    """The full functional API (cdf/pdf/sf/isf/rvs + ppf bounds) is self-consistent.

    pymc-extras itself uses only logpdf/logcdf/logsf/ppf; these guard the rest of the
    surface from typos.
    """

    @pytest.mark.parametrize(
        "module, params",
        [(_pytensor_genpareto, (0.0, 1.0, 0.2)), (_pytensor_extgenpareto, (0.0, 1.0, 0.2, 1.5))],
        ids=["genpareto", "extgenpareto"],
    )
    def test_functional_api_is_self_consistent(self, module, params):
        x = np.array([0.3, 1.0, 3.0])
        np.testing.assert_allclose(
            module.cdf(x, *params).eval(), np.exp(module.logcdf(x, *params).eval())
        )
        np.testing.assert_allclose(
            module.pdf(x, *params).eval(), np.exp(module.logpdf(x, *params).eval())
        )
        np.testing.assert_allclose(
            module.sf(x, *params).eval(), np.exp(module.logsf(x, *params).eval())
        )
        np.testing.assert_allclose(
            module.cdf(x, *params).eval() + module.sf(x, *params).eval(), 1.0, atol=1e-9
        )
        # isf agrees with ppf(1 - q) where both are accurate (away from the tails)
        q = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(module.isf(q, *params).eval(), module.ppf(1 - q, *params).eval())
        # ppf bounds: NaN outside [0, 1]; q = 0 -> mu (the lower support edge)
        assert np.all(np.isnan(module.ppf(np.array([-0.1, 1.1]), *params).eval()))
        np.testing.assert_allclose(float(module.ppf(0.0, *params).eval()), 0.0)

    def test_isf_keeps_the_upper_tail(self):
        # isf(x) for tiny x (upper tail) must not lose x to a 1 - x subtraction.
        # GPD at xi=0 has the closed form isf(x) = -log(x); ppf(1 - x) degrades there.
        x = np.array([1e-8, 1e-12, 1e-15])
        xi = pt.constant(0.0)  # tensor so the xi=0 upper-bound division stays symbolic
        isf_val = _pytensor_genpareto.isf(x, 0.0, 1.0, xi).eval()
        np.testing.assert_allclose(isf_val, -np.log(x), rtol=1e-12)
        naive = _pytensor_genpareto.ppf(1 - x, 0.0, 1.0, xi).eval()
        assert abs(isf_val[-1] - -np.log(x[-1])) < abs(naive[-1] - -np.log(x[-1]))

    @pytest.mark.parametrize(
        "module, params",
        [(_pytensor_genpareto, (0.0, 1.0, 0.2)), (_pytensor_extgenpareto, (0.0, 1.0, 0.2, 1.5))],
        ids=["genpareto", "extgenpareto"],
    )
    def test_rvs_shape_dtype_and_random_state(self, module, params):
        draws = module.rvs(
            *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
        )
        out = draws.eval()
        assert out.shape == (4, 3) and out.dtype == np.float64
        assert np.all(out >= 0.0)  # in support (mu = 0)
        # random_state controls the draws: same seed -> identical, different -> not
        same = module.rvs(
            *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(0))
        ).eval()
        diff = module.rvs(
            *params, size=(4, 3), random_state=pytensor.shared(np.random.default_rng(9))
        ).eval()
        np.testing.assert_array_equal(out, same)
        assert not np.array_equal(out, diff)
