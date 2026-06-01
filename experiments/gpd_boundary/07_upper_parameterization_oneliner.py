import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("pymc").setLevel(logging.ERROR)
import numpy as np, pymc as pm
import scipy.stats as st
from pymc_extras.distributions import GenPareto, ExtGenPareto
def div(i): return int(i.sample_stats["diverging"].sum())

# back-compat: sigma still works; upper+sigma errors
print("back-compat GenPareto.dist(mu,sigma,xi):", float(pm.logp(GenPareto.dist(0.,1.,-0.3), 0.5).eval()))
try: GenPareto.dist(mu=0, sigma=1, xi=-0.3, upper=3.0); print("ERROR: should have raised")
except ValueError as e: print("upper+sigma raises:", str(e)[:40])

print("\n=== one-liner `upper=` param: standard GenPareto, samples cleanly? ===")
for xi_true in [-0.3,-0.6,-0.9]:
    data=st.genpareto.rvs(c=xi_true,loc=0,scale=1.0,size=400,random_state=np.random.default_rng(5))
    xmax=data.max()
    with pm.Model():
        delta=pm.HalfNormal("delta",2.0); xi=pm.Uniform("xi",-0.999,-1e-3)
        wall=xmax+delta
        GenPareto("obs", mu=0.0, upper=wall, xi=xi, observed=data)   # <-- one line, no sigma math
        i=pm.sample(500,tune=500,chains=2,random_seed=1,progressbar=False)
    print(f"  GPD  xi_true={xi_true}: upper-param div={div(i)} xi_post={i.posterior['xi'].values.mean():.3f}")
data=st.genpareto.rvs(c=-0.5,loc=0,scale=1.0,size=400,random_state=np.random.default_rng(9)); xmax=data.max()
with pm.Model():
    delta=pm.HalfNormal("delta",2.0); xi=pm.Uniform("xi",-0.999,-1e-3); kappa=pm.HalfNormal("kappa",2.0)
    ExtGenPareto("obs", mu=0.0, upper=xmax+delta, xi=xi, kappa=kappa, observed=data)
    i=pm.sample(500,tune=500,chains=2,random_seed=1,progressbar=False)
print(f"  ExtGPD xi_true=-0.5: upper-param div={div(i)} xi={i.posterior['xi'].values.mean():.3f} kappa={i.posterior['kappa'].values.mean():.2f}")
