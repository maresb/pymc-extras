import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("pymc").setLevel(logging.ERROR)
import numpy as np, pymc as pm, pytensor.tensor as pt
import scipy.stats as st
from pymc_extras.distributions import GenPareto, ExtGenPareto

def div(idata): return int(idata.sample_stats["diverging"].sum())

print("=== M3 wall-margin reparam vs M1 standard, across xi (GPD), + scipy MLE ===")
for xi_true in [-0.3,-0.5,-0.7,-0.9]:
    data=st.genpareto.rvs(c=xi_true,loc=0,scale=1.0,size=400,random_state=np.random.default_rng(7))
    xmax=data.max()
    mle=st.genpareto.fit(data, floc=0)  # (c, loc, scale) = (xi, 0, sigma)
    # M1 standard
    with pm.Model():
        s=pm.HalfNormal("sigma",2.0); x=pm.Uniform("xi",-0.999,-1e-3)
        GenPareto("obs",mu=0.0,sigma=s,xi=x,observed=data)
        i1=pm.sample(500,tune=500,chains=2,random_seed=1,progressbar=False)
    # M3 reparam
    with pm.Model():
        d=pm.HalfNormal("delta",2.0); x=pm.Uniform("xi",-0.999,-1e-3)
        w=xmax+d; sig=pm.Deterministic("sigma",-w*x)
        GenPareto("obs",mu=0.0,sigma=sig,xi=x,observed=data)
        i3=pm.sample(500,tune=500,chains=2,random_seed=1,progressbar=False)
        smin=float(np.min([np.min(1+xv*data/sv) for xv,sv in
                   zip(i3.posterior["xi"].values.ravel(), i3.posterior["sigma"].values.ravel())]))
    x1=i3.posterior["xi"].values.ravel()
    print(f"  xi_true={xi_true}: MLE xi={mle[0]:.3f} sigma={mle[2]:.3f} | "
          f"M1 div={div(i1):<4} | M3 div={div(i3):<3} xi={x1.mean():.3f} s_min={smin:.1e}")

print("\n=== ExtGenPareto: M3 reparam (kappa free) ===")
for xi_true in [-0.4,-0.7]:
    data=st.genpareto.rvs(c=xi_true,loc=0,scale=1.0,size=400,random_state=np.random.default_rng(3))**1  # approx
    xmax=data.max()
    with pm.Model():
        d=pm.HalfNormal("delta",2.0); x=pm.Uniform("xi",-0.999,-1e-3); k=pm.HalfNormal("kappa",2.0)
        w=xmax+d; sig=pm.Deterministic("sigma",-w*x)
        ExtGenPareto("obs",mu=0.0,sigma=sig,xi=x,kappa=k,observed=data)
        ie=pm.sample(500,tune=500,chains=2,random_seed=1,progressbar=False)
    print(f"  xi_true={xi_true}: ExtGPD M3 div={div(ie)} xi={ie.posterior['xi'].values.mean():.3f} kappa={ie.posterior['kappa'].values.mean():.2f}")
