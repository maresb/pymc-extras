import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable

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


def _gpd_tail(z, xi):
    """``(t, log_s)`` for ``t = xi * z`` and ``log_s = log(1 + xi * z)``.

    ``s = 1 + xi * z`` is the *only* place the observation enters the GPD family --
    the log-density, survival, CDF and support mask are all functions of it -- so
    every builder forms it here, exactly once per call, and reuses the pair. That
    keeps the one unavoidable precision loss from being re-derived inconsistently:
    as ``value`` approaches the ``xi < 0`` upper wall ``mu - sigma/xi``, ``t -> -1``
    and ``s = 1 + t`` cancels catastrophically. That loss is in the *inputs* (once
    ``value`` is within a few ULP of the wall the low bits of ``s`` are already
    gone and no rearrangement here recovers them); see the class precision note and
    the margin-aware entry points for boundary-critical callers.
    """
    t = _safe_mul(xi, z)
    return t, pt.log1p(t)


def _gpd_log_h(z, sigma, t, log_s):
    """GPD log-density, in-support expression (no support masking)."""
    return -pt.log(sigma) - log_s - z * _log1p_div(t)


def _gpd_log_S(z, t):
    """GPD log survival ``log(1 - H) = -m``, in-support expression.

    Computed directly from the survival exponent ``m = log1p(xi z) / xi`` rather
    than as ``log1mexp(log H)``, so it stays exact arbitrarily deep in the upper
    tail (where ``log H -> 0`` and any ``H``-based route underflows).
    """
    return -(z * _log1p_div(t))


def _gpd_log_H(z, t):
    """GPD log-CDF, in-support expression (no support masking / no saturation)."""
    return pt.log1mexp(_gpd_log_S(z, t))


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


def _in_gpd_support(z, t):
    """Boolean mask of the GPD support: z >= 0 and (for xi < 0) z <= -1/xi.

    ``t = xi * z``; the upper edge is ``s = 1 + t > 0`` (the same ``t`` the density
    is built from, so the mask and the value never disagree at the wall).
    """
    return pt.and_(z >= 0, 1 + t > 0)


def _propagate_nonfinite_shape(result, xi, kappa=None):
    """Map ``result`` to ``nan`` wherever a shape parameter is non-finite.

    The support / boundary ``switch`` masks below mask out-of-support values to
    ``-inf`` / ``0``; without this a non-finite shape would be silently turned
    into one of those ("valid parameter, impossible value", which is a lie).
    Keyed on ``isfinite(xi)`` directly -- not on ``1 + xi z`` -- so it fires for
    ``xi = nan`` and ``xi = +-inf`` at *every* value, including ``x = mu`` (where
    ``z = 0`` makes ``1 + xi z`` finite). ``nan``/``inf`` shape in -> ``nan`` out,
    consistently across logp / logcdf / logccdf. For the extended family ``kappa``
    is checked the same way (a non-finite ``kappa`` passes ``kappa > 0`` but would
    otherwise leak inconsistent ``nan`` / ``-inf`` / ``0`` across the three).
    """
    result = pt.switch(pt.isfinite(xi), result, np.nan)
    if kappa is not None:
        result = pt.switch(pt.isfinite(kappa), result, np.nan)
    return result


# The ``gen_pareto_*`` / ``ext_gen_pareto_*`` builders below are pure PyTensor:
# they assemble the masked log-density / log-CDF / quantile graphs and call NO
# PyMC parameter check, which keeps them portable (the math can be reused
# elsewhere, e.g. in pytensor-distributions). Parameter validation lives only in
# the Continuous wrapper classes, which add ``check_parameters`` / ``check_icdf_*``.


def gen_pareto_logp(value, mu, sigma, xi):
    """Pure-PyTensor GPD log-density; out-of-support values map to ``-inf``."""
    z = (value - mu) / sigma
    t, log_s = _gpd_tail(z, xi)
    logp = pt.switch(_in_gpd_support(z, t), _gpd_log_h(z, sigma, t, log_s), -np.inf)
    # The density vanishes at the +inf tail for every xi; for xi > 0 the
    # in-support branch would evaluate log1p(inf)/inf -> nan there, so pin
    # z = +inf to -inf explicitly.
    logp = pt.switch(pt.eq(z, np.inf), -np.inf, logp)
    return _propagate_nonfinite_shape(logp, xi)


def gen_pareto_logcdf(value, mu, sigma, xi):
    """Pure-PyTensor GPD log-CDF."""
    z = (value - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    # Three regions: below mu -> 0 (log -inf); for xi < 0 past the finite upper
    # endpoint mu - sigma/xi -> 1 (log 0); else log1mexp(-m).
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logcdf = pt.switch(above_upper, 0.0, _gpd_log_H(z, t))
    logcdf = pt.switch(z >= 0, logcdf, -np.inf)
    # CDF -> 1 (logcdf 0) at the +inf tail; for xi > 0 _gpd_log_H(inf) is nan.
    logcdf = pt.switch(pt.eq(z, np.inf), 0.0, logcdf)
    return _propagate_nonfinite_shape(logcdf, xi)


def gen_pareto_logccdf(value, mu, sigma, xi):
    """Pure-PyTensor GPD log complementary CDF (log survival function).

    The survival exponent ``m`` is computed directly, so this is exact and stable
    in the heavy upper tail -- the regime where the generic
    ``log1mexp(logcdf)`` fallback collapses (``logcdf -> 0`` there). This is the
    natural ``logsf`` primitive for a peaks-over-threshold model.
    """
    z = (value - mu) / sigma
    t, _ = _gpd_tail(z, xi)
    logsf = _gpd_log_S(z, t)  # log S = -m, exact in the tail
    # For xi < 0 past the finite upper endpoint, and at the +inf tail, S = 0.
    above_upper = pt.and_(pt.lt(xi, 0), pt.le(1 + t, 0))
    logsf = pt.switch(pt.or_(above_upper, pt.eq(z, np.inf)), -np.inf, logsf)
    # Below mu the survival is 1 (logsf 0).
    logsf = pt.switch(z < 0, 0.0, logsf)
    return _propagate_nonfinite_shape(logsf, xi)


def gen_pareto_icdf(value, mu, sigma, xi):
    """Pure-PyTensor GPD quantile function (assumes ``0 <= value <= 1``)."""
    value = pt.as_tensor_variable(value)
    excess = -pt.log1p(-value)  # = -log(1 - q) = m
    x = _gpd_quantile_from_excess(excess, mu, sigma, xi)
    # Explicit endpoints: q=1 -> finite upper bound (xi<0) or +inf, q=0 -> mu.
    # Without this, q=1 with xi<0 is ``inf * 0 = nan`` rather than ``mu - sigma/xi``.
    x = pt.switch(pt.eq(value, 1), _gpd_upper_bound(mu, sigma, xi), x)
    x = pt.switch(pt.eq(value, 0), mu, x)
    # The endpoint switches above would otherwise hand back mu / the upper bound even
    # for a non-finite xi; propagate NaN so q = 0 / 1 agree with the interior.
    return _propagate_nonfinite_shape(x, xi)
