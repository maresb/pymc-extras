# GPD / ExtGPD robustness near the ξ<0 support wall — findings & proposals

Branch: `gpd-margin-experiments` (off `add-gpd-extgpd-distributions` @ `dc8f483`, which
already has the B/C/D pass: `s` consolidation, boundary precision tests, docstring
note). Nothing here is pushed to the PR; it is the design exploration you asked for.

Goal you set: **work through standard PyMC, no hacks, with sampler-stable gradients.**

## TL;DR

There are **two different problems** near the bounded (ξ<0) wall, and they are
usually conflated. Separating them is the main result.

| | Problem 1 — sampling **geometry** | Problem 2 — float64 **precision** |
|---|---|---|
| Symptom | NUTS **divergences** | gradient loses digits (~ulp/s) |
| When | wall approaches `max(data)`; **routinely**, margin `s ≈ 1e-2…1e-4` | only at `s ≲ 1e-6`; **rarely reached** |
| Cause | support boundary depends on (σ,ξ) → stiff curved ridge / hard −∞ wall | `1 + ξz → 0` cancels in the float64 input |
| Fix | **reparameterize by the wall** — *standard PyMC, no hack* ✅ | supply the margin `s` directly (margin-aware logp) |
| Reaches the sampler? | yes, this is the practical pain | essentially never under the Problem-1 fix |

**The thing that actually bites in normal use is Problem 1, and it is fully solved
with a standard reparameterization — no new API, no hacks, stable gradients.** The
precision issue (Problem 2, what the original task targeted) is real but only matters
for a model that *pins* the wall within ~1e-6 of the data; an optional margin-aware
primitive covers that.

---

## Evidence

### Problem 1 is geometry, not precision

Standard Bayesian POT, `GenPareto("obs", mu=0, sigma=HalfNormal, xi=Uniform(-1,0),
observed=data)`, 300 pts (`03_nuts_divergences_vs_margin.py`):

```
 xi_true   divergences   s_min reached (over all draws)
  -0.1          64              5.7e-02
  -0.3         172              9.9e-03
  -0.5         334              5.5e-04
  -0.7         654              4.1e-05
```

Divergences climb steeply toward ξ=−1, but `s_min` only reaches ~1e-4 — where the
gradient is still good to ~`ulp/s ≈ 1e-12`. So the divergences are **not** the
float64 cancellation; they are the classic difficulty of a distribution whose
support edge `mu − σ/ξ` moves with the parameters (a stiff ridge where the wall
grazes `max(data)`, with a hard −∞ just beyond it).

### The standard-PyMC fix: reparameterize by the wall

Replace `σ` by the **wall margin** `δ = wall − max(data) > 0` (so `σ = (max(data)+δ)·(−ξ)`).
The support constraint becomes a plain positivity bound on `δ`, and the ridge
straightens out. *Same `GenPareto`, ordinary value-based logp — only the
parameterization changes* (`04_…`, `05_…`):

```
                                              divergences
  M1  standard (σ, ξ)          ta=0.8             924
  M2  standard (σ, ξ)          ta=0.99            412      (band-aid; still bad)
  M3  reparam (δ, ξ)  σ=-w·ξ   ta=0.9               0      ✅ standard GenPareto
  M4  margin-param + from-s    ta=0.9               0      (also 0; precision not needed)
```

Across ξ and for ExtGenPareto, recovering the SciPy MLE with ~0 divergences
(`05_…`, n=400):

```
  xi_true   MLE xi    M1 div   M3 div   M3 xi_post   s_min(M3)
   -0.3     -0.335      56        0       -0.321       1.7e-02
   -0.5     -0.533     216        0       -0.518       1.7e-03
   -0.7     -0.734     367        0       -0.719       3.0e-05
   -0.9     -0.937     749        2       -0.919       2.8e-05
  ExtGPD -0.4 / -0.7:  M3 div 0 / 1, xi & kappa recovered
```

M3 is plain modeling (`pm.Deterministic`, `σ = −w·ξ`) — no custom logp, no
Potential, no margin trickery. A one-line convenience
`genpareto_sigma_from_upper(mu, upper, xi) = (upper − mu)·(−ξ)` is enough to make it
a drop-in; the support edge is then a parameter you put a prior on.

### Problem 2 (precision) and the margin-aware primitive

When the wall is *pinned* within ~1e-6 of the data (δ tiny), the value-based logp's
σ/ξ gradient degrades as `ulp/s`. The margin-aware entry points take the margin `s`
(built cancellation-free from the model's own structure) and are **exact to full
float64 precision at every margin**. End-to-end test, gradient of logp w.r.t.
(σ, ξ, d) where `d` is the binding observation's distance to the wall
(`01_…`, vs a 100-digit `decimal` reference):

```
  d (=wall distance)   value-based d_sigma / d_d rel err     from-s rel err
       1e-6                 1.6e-10  /  2.9e-10                ~1e-16
       1e-9                 2.3e-07  /  2.7e-07                ~1e-16
       1e-12                5.7e-04  /  2.0e-04                ~1e-16
       1e-14                2.4e-02  /  5.5e-02   (2–5% !)     ~1e-16
```

Same story for the survival `logsf` (`02_…`). The primitives are four ~3-line pure
functions (`pymc_extras/distributions/gpd_margin.py`): GPD + ExtGPD × {logp, logsf}.

### Does Problem 2 ever reach the sampler? Barely — and it never *destabilizes* it

The open-support convention (`logp → −∞` at the wall) makes the boundary **repulsive**,
so the likelihood self-limits the margin. Even with a prior that actively favors a
tiny wall margin, and even with an aggressive external pin, NUTS stays divergence-free
(`06_pinned_wall_reaches_precision_regime.py`, ξ=−0.5):

```
  A) reparam, prior delta~HalfNormal(1e-4):     div=0,  s_min reached ≈ 1.6e-6
       -> the likelihood holds delta up; s barely touches the 1e-6 frontier
  B) external Potential pin (-1e3*delta), value-based: div=0,  delta driven to ~1.7e-11
  B') same pin, from-s (cancellation-free):           div=0,  delta ~1.2e-9
```

Two takeaways: (i) under normal priors the margin stays ≳1e-6, so Problem 2 is
essentially unreached; (ii) even when an artificial pin drives `s` to ~1e-11 — deep in
the cancellation regime — **NUTS still does not diverge**, value-based or from-s. So
the precision limit is an *accuracy* refinement for a deliberately-pinned corner, not a
sampler-stability issue. (The from-s and value-based posteriors agree at the median and
differ only in how deep the artificial pin wanders into the tail.)
The **caller must supply `s` as a correctly-differentiable tensor** (e.g. a POT model
parameterized by the wall margin, where `s_i = (δ + gap_i)/w` with `gap_i =
max(data) − x_i` precomputed): the value is right *and* the chain rule composes the
right gradient. This is why a generic `Distribution` cannot do it for you — fed fixed
`value` data + a wall parameter, it would re-form `wall − value` and re-introduce the
cancellation. It is fundamentally a *reparameterization*, not a kernel rewrite.

---

## Proposals

**P0 — already shipped (B/C/D, `dc8f483`).** `s` consolidated to one site; boundary
precision pinned by tests; docstring states the limit. Keep.

**P1 — RECOMMENDED: a native `upper=` parameterization (prototyped here).** The
real-world fix for divergent bounded-POT fits, entirely in standard PyMC. Prototyped
in this branch by adding an optional `upper=` to `GenPareto.dist` / `ExtGenPareto.dist`
(`sigma = (upper - mu) * (-xi)`, mutually exclusive with `sigma`, fully back-compatible
— existing 98 tests pass). Usage is a one-liner, no reparameterization math in the user
model (`07_upper_parameterization_oneliner.py`):

```python
delta = pm.HalfNormal("delta", 2.0)          # wall margin above max(data)
xi    = pm.Uniform("xi", -1, 0)
GenPareto("obs", mu=0.0, upper=data.max() + delta, xi=xi, observed=data)
#   div=0 across xi in {-0.3,-0.6,-0.9} for GPD and ExtGPD; recovers xi
```

This is the cleanest "standard PyMC, no hacks, stable gradients" answer. If a native
kwarg is too much surface, the same effect is available via a one-line helper
`genpareto_sigma_from_upper(mu, upper, xi)` (in `gpd_margin.py`) plus a docs example.
No numerical risk; additive.

**P2 — OPTIONAL (low priority): ship the margin-aware primitives** (`*_from_s`) for the
pinned-wall regime. Pure PyTensor, additive, exact at the boundary. Caveat from the
pinned-wall experiment: this buys *accuracy* in a deliberately-pinned corner, not
sampler stability (NUTS doesn't diverge there either), and normal priors don't even
reach it — so it is genuinely niche. Document that P1 is the first thing to reach for,
and that the caller must supply a cancellation-free, differentiable `s`. Defer to a
maintainer chat (non-standard surface); reasonable to skip until a concrete pinned-wall
model needs it.

**P3 — NOT recommended.** Baking an optional `s=` kwarg into `*_logp(value, …)`
(awkward `value`-or-`s` overload; dead args), a float128 path (platform-dependent,
only ~2 extra digits), or a "magic" boundary-stable `Distribution` over fixed `value`
data (impossible — the cancellation is in the inputs).

### Recommendation
Do **P1** (the standard-PyMC, no-hack, stable-gradient answer to the divergence
problem people actually hit) and keep **P0**. Offer **P2** as an advanced escape
hatch for pinned-wall models, pending a maintainer design discussion. The key
reframing to land: *divergences are geometry → reparameterize; the float64 margin
limit is a separate, rarely-reached concern with its own opt-in tool.*
