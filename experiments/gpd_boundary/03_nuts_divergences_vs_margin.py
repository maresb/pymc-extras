import warnings; warnings.filterwarnings("ignore")
import numpy as np, pymc as pm, pytensor.tensor as pt
import scipy.stats as st
from pymc_extras.distributions import GenPareto

def s_min_of(data, sigma, xi):  # smallest margin among data at given params
    return float(np.min(1 + xi*(np.asarray(data))/sigma))

print("=== Experiment 1: standard Bayesian POT fit, how small does s actually get? ===")
for xi_true in [-0.1, -0.3, -0.5, -0.7]:
    rng=np.random.default_rng(0)
    data = st.genpareto.rvs(c=xi_true, loc=0, scale=1.0, size=300, random_state=rng)
    wall = -1.0/xi_true
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma", 2.0)
        xi = pm.Uniform("xi", -0.999, -0.001)
        GenPareto("obs", mu=0.0, sigma=sigma, xi=xi, observed=data)
        idata = pm.sample(500, tune=500, chains=2, random_seed=1, progressbar=False,
                          init="jitter+adapt_diag")
    post = idata.posterior
    sig_s = post["sigma"].values.ravel(); xi_s = post["xi"].values.ravel()
    # smallest margin reached across all posterior draws (the binding observation)
    smins = [s_min_of(data, sg, x) for sg, x in zip(sig_s, xi_s)]
    div = int(idata.sample_stats["diverging"].sum())
    print(f"  xi_true={xi_true}: x_max={data.max():.4f} wall={wall:.3f} | "
          f"xi_post={xi_s.mean():.3f}+/-{xi_s.std():.3f}  divergences={div}  "
          f"s_min(min over draws)={min(smins):.2e}  s_min(median)={np.median(smins):.2e}")
