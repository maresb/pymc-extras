import warnings; warnings.filterwarnings("ignore")
import numpy as np, pytensor, pytensor.tensor as pt
from decimal import Decimal as D, getcontext
getcontext().prec = 100
from pymc_extras.distributions.continuous import gen_pareto_logp, ext_gen_pareto_logp

# ---- prototype margin-aware primitives (from the exact margin s = 1 + xi*z > 0) ----
def gen_pareto_logp_from_s(s, sigma, xi):
    log_s = pt.log(s)
    return -pt.log(sigma) - log_s * (1 + 1 / xi)          # -log sigma - log s - log s/xi

def ext_gen_pareto_logp_from_s(s, sigma, xi, kappa):
    log_s = pt.log(s); m = log_s / xi                      # m = -log survival >= 0
    log_H = pt.log1mexp(-m)                                # log(1 - exp(-m))
    carrier = pt.switch(pt.eq(kappa, 1.0), 0.0, (kappa - 1) * log_H)
    return pt.log(kappa) + carrier - pt.log(sigma) - log_s - m

# ---- 100-digit reference: logp as a function of (sigma, xi, d) via exact s = -xi*d/sigma ----
def ref_logp(sigma, xi, d, kappa=None):
    s = -xi * d / sigma                                    # exact margin, no cancellation
    log_s = s.ln(); logp = -sigma.ln() - (1 + 1/xi)*log_s
    if kappa is not None:
        log_H = (1 - ((D(-1)/xi)*log_s).exp()).ln()
        logp += kappa.ln() + (kappa-1)*log_H
    return logp
def ref_grad(sigma, xi, d, which, kappa=None):
    vals = {'sigma':sigma,'xi':xi,'d':d}
    h = abs(vals[which])*D('1e-25') or D('1e-25')
    hi=dict(vals); hi[which]+=h; lo=dict(vals); lo[which]-=h
    return (ref_logp(hi['sigma'],hi['xi'],hi['d'],kappa)-ref_logp(lo['sigma'],lo['xi'],lo['d'],kappa))/(2*h)

# ---- build both paths as functions of (sigma, xi, d) ----
mu = 0.4
sg, xs, dd, kp = pt.dscalar("sg"), pt.dscalar("xs"), pt.dscalar("dd"), pt.dscalar("kp")
xF = mu - sg/xs                       # the wall
value = xF - dd                       # reconstructs the float64 observation
s_clean = -xs*dd/sg                   # the caller's cancellation-free margin

def fns(kappa):
    if kappa is None:
        lp_val = gen_pareto_logp(value, mu, sg, xs)
        lp_s   = gen_pareto_logp_from_s(s_clean, sg, xs)
        ins=[sg,xs,dd]; names=['sigma','xi','d']; grads_in=[sg,xs,dd]
    else:
        lp_val = ext_gen_pareto_logp(value, mu, sg, xs, kp)
        lp_s   = ext_gen_pareto_logp_from_s(s_clean, sg, xs, kp)
        ins=[sg,xs,dd,kp]; names=['sigma','xi','d']; grads_in=[sg,xs,dd]
    f_val = pytensor.function(ins,[lp_val]+[pt.grad(lp_val,g) for g in grads_in])
    f_s   = pytensor.function(ins,[lp_s]  +[pt.grad(lp_s,g)   for g in grads_in])
    return f_val, f_s, names

sigma, xi = 1.3, -0.3
for kappa in [None, 2.5]:
    f_val, f_s, names = fns(kappa)
    print(f"\n===== {'GPD' if kappa is None else 'ExtGPD kappa=2.5'} =====")
    print(f"{'d(=dist to wall)':>16} {'path':>6} | {'logp_rel':>10}  "+"  ".join(f'd_{n}_rel' for n in names))
    for d in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-14]:
        args=[sigma,xi,d]+([kappa] if kappa else [])
        ov=[float(x) for x in f_val(*args)]; os=[float(x) for x in f_s(*args)]
        P=(D(sigma),D(xi),D(d)); K=D(kappa) if kappa else None
        lp_r=ref_logp(*P,K); gr=[ref_grad(*P,n,K) for n in names]
        def rel(a,b): return float(abs((D(a)-b)/b)) if b!=0 else 0.0
        # margin s for context
        s_ctx = -xi*d/sigma
        print(f"  d={d:>7.0e} (s={s_ctx:.0e}) value-based | {rel(ov[0],lp_r):10.1e}  "+"  ".join(f"{rel(ov[1+i],gr[i]):8.1e}" for i in range(len(names))))
        print(f"  {'':>20}     from-s | {rel(os[0],lp_r):10.1e}  "+"  ".join(f"{rel(os[1+i],gr[i]):8.1e}" for i in range(len(names))))
