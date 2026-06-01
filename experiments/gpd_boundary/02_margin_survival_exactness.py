import warnings; warnings.filterwarnings("ignore")
import numpy as np, pytensor, pytensor.tensor as pt
from decimal import Decimal as D, getcontext
getcontext().prec = 100
from pymc_extras.distributions.continuous import gen_pareto_logccdf, ext_gen_pareto_logccdf

# from-s survival primitives
def gen_pareto_logsf_from_s(s, sigma, xi):
    return -pt.log(s) / xi                                  # -m
def ext_gen_pareto_logsf_from_s(s, sigma, xi, kappa):
    m = pt.log(s) / xi; log_H = pt.log1mexp(-m)             # log H
    return pt.log1mexp(kappa * log_H)                       # log(1 - H**kappa)

def ref_logsf(sigma, xi, d, kappa=None):
    s=-xi*d/sigma; m = s.ln()/xi
    if kappa is None: return -m
    log_H=(1-(-m).exp()).ln()
    return (1-(kappa*log_H).exp()).ln()
def ref_grad(sigma, xi, d, which, kappa=None):
    vals={'sigma':sigma,'xi':xi,'d':d}; h=abs(vals[which])*D('1e-25') or D('1e-25')
    hi=dict(vals); hi[which]+=h; lo=dict(vals); lo[which]-=h
    return (ref_logsf(hi['sigma'],hi['xi'],hi['d'],kappa)-ref_logsf(lo['sigma'],lo['xi'],lo['d'],kappa))/(2*h)

mu=0.4; sg,xs,dd,kp=pt.dscalar(),pt.dscalar(),pt.dscalar(),pt.dscalar()
xF=mu-sg/xs; value=xF-dd; s_clean=-xs*dd/sg
sigma,xi=1.3,-0.3
for kappa in [None,2.5]:
    if kappa is None:
        lv=gen_pareto_logccdf(value,mu,sg,xs); ls=gen_pareto_logsf_from_s(s_clean,sg,xs); ins=[sg,xs,dd]
    else:
        lv=ext_gen_pareto_logccdf(value,mu,sg,xs,kp); ls=ext_gen_pareto_logsf_from_s(s_clean,sg,xs,kp); ins=[sg,xs,dd,kp]
    fv=pytensor.function(ins,[lv]+[pt.grad(lv,g) for g in [sg,xs,dd]])
    fs=pytensor.function(ins,[ls]+[pt.grad(ls,g) for g in [sg,xs,dd]])
    print(f"\n==== logsf {'GPD' if kappa is None else 'ExtGPD k=2.5'} ====  (d_sigma, d_xi, d_d rel err)")
    for d in [1e-3,1e-9,1e-12,1e-14]:
        a=[sigma,xi,d]+([kappa] if kappa else [])
        ov=[float(x) for x in fv(*a)]; os=[float(x) for x in fs(*a)]
        P=(D(sigma),D(xi),D(d)); K=D(kappa) if kappa else None
        gr=[ref_grad(*P,n,K) for n in ['sigma','xi','d']]
        def rel(a_,b): return float(abs((D(a_)-b)/b)) if b!=0 else 0.0
        print(f"  d={d:.0e} value | "+"  ".join(f"{rel(ov[1+i],gr[i]):8.1e}" for i in range(3)))
        print(f"  d={d:.0e} from-s| "+"  ".join(f"{rel(os[1+i],gr[i]):8.1e}" for i in range(3)))
