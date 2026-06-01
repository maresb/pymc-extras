import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("pymc").setLevel(logging.ERROR)
import numpy as np, pymc as pm, pytensor.tensor as pt
import scipy.stats as st
from pymc_extras.distributions import GenPareto
from pymc_extras.distributions.continuous import gen_pareto_logp

xi_true=-0.7
rng=np.random.default_rng(0)
data=st.genpareto.rvs(c=xi_true, loc=0, scale=1.0, size=300, random_state=rng)
xmax=data.max(); gaps = xmax - data  # precomputed, exact data structure (>=0)
print(f"data: n=300 x_max={xmax:.5f} wall_true={-1/xi_true:.4f}\n")

def report(name, idata, extra=""):
    div=int(idata.sample_stats["diverging"].sum())
    try: xi_s=idata.posterior["xi"].values.ravel()
    except: xi_s=idata.posterior["xi_"].values.ravel() if "xi_" in idata.posterior else None
    xs=f"xi={xi_s.mean():.3f}+/-{xi_s.std():.3f}" if xi_s is not None else ""
    print(f"  {name:<42} divergences={div:<4} {xs} {extra}")

# --- M1: standard (sigma, xi), default target_accept ---
with pm.Model() as m1:
    sigma=pm.HalfNormal("sigma",2.0); xi=pm.Uniform("xi",-0.999,-0.001)
    GenPareto("obs",mu=0.0,sigma=sigma,xi=xi,observed=data)
    report("M1 standard (sigma,xi) ta=0.8", pm.sample(600,tune=600,chains=2,random_seed=1,progressbar=False))

# --- M2: standard, target_accept=0.99 ---
with m1:
    report("M2 standard (sigma,xi) ta=0.99", pm.sample(600,tune=600,chains=2,random_seed=1,progressbar=False,target_accept=0.99))

# --- M3: reparameterize by wall w>xmax (sigma=-w*xi), still value-based logp ---
with pm.Model() as m3:
    delta=pm.HalfNormal("delta",2.0)       # wall margin above x_max
    w=pm.Deterministic("w", xmax+delta)
    xi=pm.Uniform("xi",-0.999,-0.001)
    sigma=pm.Deterministic("sigma", -w*xi) # sigma = -w*xi (mu=0 => wall=-sigma/xi=w)
    GenPareto("obs",mu=0.0,sigma=sigma,xi=xi,observed=data)
    report("M3 reparam (delta,xi) value-based ta=0.9", pm.sample(600,tune=600,chains=2,random_seed=1,progressbar=False,target_accept=0.9))

# --- M4: cancellation-free margin param: s_i=(delta+gaps)/w via Potential + margin primitive ---
with pm.Model() as m4:
    delta=pm.HalfNormal("delta",2.0); xi=pm.Uniform("xi",-0.999,-0.001)
    w=xmax+delta; sigma=-w*xi
    s_i = (delta + gaps)/w                 # cancellation-free margins (delta param + precomputed gaps)
    log_s=pt.log(s_i)
    logp = -pt.log(sigma) - log_s*(1+1/xi) # gen_pareto_logp_from_s, vectorized
    pm.Potential("lik", logp.sum())
    pm.Deterministic("xi_", xi)
    report("M4 margin-param (delta,xi) from-s ta=0.9", pm.sample(600,tune=600,chains=2,random_seed=1,progressbar=False,target_accept=0.9))
