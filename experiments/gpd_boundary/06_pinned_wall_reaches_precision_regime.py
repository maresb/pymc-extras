import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("pymc").setLevel(logging.ERROR)
import numpy as np, pymc as pm, pytensor.tensor as pt
import scipy.stats as st
from pymc_extras.distributions import GenPareto
from pymc_extras.distributions.gpd_margin import gen_pareto_logp_from_s

xi_true=-0.5
data=st.genpareto.rvs(c=xi_true,loc=0,scale=1.0,size=300,random_state=np.random.default_rng(11))
xmax=data.max(); gaps=xmax-data
def div(i): return int(i.sample_stats["diverging"].sum())
def smin(i):
    return float(np.min([np.min(1+x*data/s) for x,s in
        zip(i.posterior["xi"].values.ravel(), i.posterior["sigma"].values.ravel())]))

print(f"data n=300 xi_true={xi_true} x_max={xmax:.4f} wall_true={-1/xi_true:.3f}\n")

# A) reparam, prior FAVORS tiny delta (HalfNormal scale 1e-4) -- tries to pin the wall
with pm.Model() as mA:
    delta=pm.HalfNormal("delta",1e-4); xi=pm.Uniform("xi",-0.999,-1e-3)
    w=xmax+delta; sigma=pm.Deterministic("sigma",-w*xi)
    GenPareto("obs",mu=0.0,sigma=sigma,xi=xi,observed=data)
    iA=pm.sample(800,tune=800,chains=2,random_seed=2,progressbar=False,target_accept=0.95)
post_d=iA.posterior["delta"].values.ravel()
print(f"A) prior delta~HalfNormal(1e-4): div={div(iA)} smin={smin(iA):.2e} "
      f"delta_post[min,med]=[{post_d.min():.2e},{np.median(post_d):.2e}]  (likelihood holds delta up?)")

# B) add a Potential that AGGRESSIVELY rewards small delta (external pin), value-based
with pm.Model() as mB:
    log_delta=pm.Normal("log_delta",-25,5); delta=pt.exp(log_delta); xi=pm.Uniform("xi",-0.999,-1e-3)
    w=xmax+delta; sigma=pm.Deterministic("sigma",-w*xi)
    GenPareto("obs",mu=0.0,sigma=sigma,xi=xi,observed=data)
    pm.Potential("pin", -1e3*delta)  # push wall onto the data
    iB=pm.sample(800,tune=800,chains=2,random_seed=2,progressbar=False,target_accept=0.95)
ld=iB.posterior["log_delta"].values.ravel()
print(f"B) external pin (Potential -1e3*delta), value-based: div={div(iB)} "
      f"log_delta_post[min,med]=[{ld.min():.1f},{np.median(ld):.1f}] => delta~[{np.exp(ld.min()):.1e},{np.exp(np.median(ld)):.1e}]")

# B') SAME aggressive pin but cancellation-free margins via from-s primitive
with pm.Model() as mBp:
    log_delta=pm.Normal("log_delta",-25,5); delta=pt.exp(log_delta); xi=pm.Uniform("xi",-0.999,-1e-3)
    w=xmax+delta; sigma=-w*xi
    s_i=(delta+gaps)/w                         # cancellation-free
    pm.Potential("lik", gen_pareto_logp_from_s(s_i,sigma,xi).sum())
    pm.Potential("pin", -1e3*delta); pm.Deterministic("xi_",xi)
    iBp=pm.sample(800,tune=800,chains=2,random_seed=2,progressbar=False,target_accept=0.95)
ldp=iBp.posterior["log_delta"].values.ravel()
print(f"B')same pin, from-s (cancellation-free):  div={div(iBp)} "
      f"log_delta_post[min,med]=[{ldp.min():.1f},{np.median(ldp):.1f}] => delta~[{np.exp(ldp.min()):.1e},{np.exp(np.median(ldp)):.1e}]")
