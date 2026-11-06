"""Exact FP64 bounded-WLS targets and the two fixed U1 paired mappings."""
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky,cho_solve,eigh

SQRT384=float(np.sqrt(384.))
LAMBDA=.1
METHODS=('paired_RRR_G1dose_v1','direction_amplitude_G1dose_v1')


def metric_grams(basis,view):
    if view['view']=='valid_shared384':
        gram=basis.T@basis/384.
        return dict(shared=gram)
    return dict(image=basis[:384].T@basis[:384],text=basis[384:].T@basis[384:])


def role_rms(coefficients,basis,view,fractions):
    v=coefficients@basis.T
    if view['view']=='valid_shared384':return np.sqrt(np.square(v).mean(-1))
    image=v[...,:384]/np.sqrt(view['f0']);text=v[...,384:]/np.sqrt(1-view['f0'])
    return np.sqrt(np.square(image).mean(-1)*fractions+np.square(text).mean(-1)*(1-fractions))


def project_targets(basis,delta,view,fractions,teacher_rms,D0):
    """Actual saved-B WLS; generalized eigenbasis amortizes all scene metrics.

    This never replaces B.T B by I. Both observed role Grams are retained.
    The return is an ideal FP64 target, not a matched FP32 Base addition.
    """
    assert basis.dtype==delta.dtype==np.float64 and basis.ndim==delta.ndim==2
    n=len(delta);r=basis.shape[1];g=metric_grams(basis,view)
    assert fractions.shape==teacher_rms.shape==(n,) and np.all((fractions>0)&(fractions<1))
    assert np.all(teacher_rms>0) and D0>0
    if 'shared' in g:
        rhs=delta@basis/384.
        a=cho_solve((cholesky(g['shared'],lower=True,check_finite=False),True),rhs.T,check_finite=False).T
        Ha=a@g['shared'];solver='Actual shared Gram Cholesky, common factor for all scenes'
        solver_arrays={}
    else:
        # E.T (Igram+Tgram) E=I, E.T Igram E=diag(lambda).
        # Every actual-mask H is diagonal in this congruence, without assuming
        # exact Euclidean orthogonality of the stored floating-point basis.
        eigen,E=eigh(g['image'],g['image']+g['text'],driver='gvd',check_finite=False)
        wi=fractions/view['f0'];wt=(1-fractions)/(1-view['f0'])
        diagonal=(wt[:,None]+(wi-wt)[:,None]*eigen[None])/384.
        assert np.all(diagonal>0)
        rhs=(wi[:,None]*(delta[:,:384]@basis[:384])+wt[:,None]*(delta[:,384:]@basis[384:]))/384.
        a=((rhs@E)/diagonal)@E.T
        Ha=(wi[:,None]*(a@g['image'])+wt[:,None]*(a@g['text']))/384.
        solver='Generalized symmetric eigensolve of the two actual saved-B role Grams; no B.T B=I substitution'
        solver_arrays=dict(metric_generalized_eigenvalues=eigen,metric_congruence=E)
    absolute=np.linalg.norm(Ha-rhs,axis=1);relative=absolute/np.maximum(np.linalg.norm(rhs,axis=1),1e-300)
    assert np.max(relative)<=1e-10,float(np.max(relative))
    rho=np.sqrt(np.maximum(np.sum(a*Ha,axis=1),0.))
    decoded_rho=role_rms(a,basis,view,fractions)
    assert np.max(np.abs(rho-decoded_rho))<1e-10
    epsilon=np.maximum(1e-8,1e-5*teacher_rms);zero=rho<=epsilon;dose=np.minimum(teacher_rms,D0)
    scale=np.zeros(n,np.float64);scale[~zero]=dose[~zero]/rho[~zero]
    target=a*scale[:,None];b=role_rms(target,basis,view,fractions)
    assert np.max(np.abs(b[~zero]-dose[~zero]),initial=0.)<1e-10
    U=np.zeros_like(target);L=np.zeros(n,np.float64)
    U[~zero]=target[~zero]/(SQRT384*b[~zero,None]);L[~zero]=np.log1p(b[~zero]/D0)
    assert np.max(L,initial=0.)<=np.log(2.)+1e-10
    values=dict(source_coefficients=a,source_rho=rho,source_decoded_rho=decoded_rho,
        teacher_fulltoken_rms=teacher_rms,target_dose=dose,zero_threshold=epsilon,projection_zero=zero,
        target_scale=scale,target_coefficients=target,target_ideal_RMS=b,
        RRR_targets=target/(D0*SQRT384),DA_direction_targets=U,DA_logamplitude_targets=L,
        normal_equation_absolute_error=absolute,normal_equation_relative_error=relative,wls_rhs=rhs,
        **{'metric_gram_'+k:v for k,v in g.items()},**solver_arrays)
    return values,dict(solver=solver,rank=r,scenes=n,zero_targets=int(zero.sum()),
        max_normal_equation_relative_error=float(relative.max()),
        max_source_rho_vs_decoded_RMS_error=float(np.max(np.abs(rho-decoded_rho))),
        CPU_ideal_FP64_mask_metric_only=True,actual_FP32_addition_matched=False)


def da_decode(u,l,basis,view,fractions,D0):
    m=D0*np.expm1(np.clip(l,0.,np.log(2.)))
    eta=SQRT384*role_rms(u,basis,view,fractions)
    q=SQRT384*m[...,None]*u/np.maximum(1.,eta)[...,None]
    q[(m==0)|(eta<=1e-12)]=0.
    return q,dict(amplitude=m,eta=eta)


def standardize_weighted(x,weights):
    assert x.dtype==weights.dtype==np.float64 and np.all(weights>0)
    assert abs(float(weights.sum())-1.)<1e-12
    mean=np.sum(weights[:,None]*x,axis=0)
    centered=x-mean;scale=np.maximum(np.sqrt(np.sum(weights[:,None]*np.square(centered),axis=0)),1e-5)
    return centered/scale,mean,scale


def gram(phi,weights):
    C=phi.T@(weights[:,None]*phi)+LAMBDA*np.eye(phi.shape[1])
    return C,cholesky(C,lower=True,check_finite=False)


def solve_targets(phi,weights,targets,C,L):
    mean=np.sum(weights[:,None]*targets,axis=0)
    rhs=phi.T@(weights[:,None]*(targets-mean))
    W=cho_solve((L,True),rhs,check_finite=False)
    error=float(np.linalg.norm(C@W-rhs)/max(np.linalg.norm(rhs),1e-300))
    assert np.isfinite(W).all() and error<=1e-10,error
    return W,mean,rhs,dict(normal_equation_relative_error=error,
        weighted_phi_mean_max_abs=float(np.max(np.abs(weights@phi))),
        lambda_average_loss=LAMBDA,bias_penalized=False,target_whitening=False)


def selftest():
    rng=np.random.default_rng(2026102496);p=4
    phi=np.r_[np.sqrt(p)*np.eye(p),-np.sqrt(p)*np.eye(p)].astype(np.float64)
    w=np.full(2*p,1/(2*p),np.float64);V=rng.normal(size=(p,3));mu=np.array([.1,-.2,.3])
    C,L=gram(phi,w);W,m,_,proof=solve_targets(phi,w,phi@V+mu,C,L)
    shrink=float(np.max(np.abs(W-V/1.1)));assert shrink<1e-12 and np.max(np.abs(m-mu))<1e-12
    zero,_,_,_=solve_targets(phi,w,np.zeros((len(phi),3)),C,L);assert np.count_nonzero(zero)==0
    basis=np.linalg.qr(rng.normal(size=(768,8)))[0];view=dict(view='role2_768',f0=.7)
    delta=rng.normal(size=(5,768));delta[0]=0.;fs=np.array([.1,.3,.5,.8,.95]);rs=np.ones(5);D0=.25
    target,wp=project_targets(basis,delta,view,fs,rs,D0)
    q,_=da_decode(target['DA_direction_targets'],target['DA_logamplitude_targets'],basis,view,fs,D0)
    perfect=float(np.max(np.abs(q-target['target_coefficients'])));assert perfect<1e-12
    small,_=da_decode(target['DA_direction_targets']*.1,target['DA_logamplitude_targets'],basis,view,fs,D0)
    assert np.max(np.abs(small-.1*target['target_coefficients']))<1e-12
    enlarged,_=da_decode(target['DA_direction_targets']*2,target['DA_logamplitude_targets'],basis,view,fs,D0)
    assert np.max(np.abs(enlarged-target['target_coefficients']))<1e-12
    assert not np.count_nonzero(q[0]) and not np.count_nonzero(target['target_coefficients'][0])
    qw=np.r_[np.full(3,.5/3),np.full(7,.5/7)].astype(np.float64)
    x=rng.normal(size=(10,4));xx,mean,scale=standardize_weighted(x,qw)
    assert abs(qw[:3].sum()-.5)<1e-15 and abs(qw[3:].sum()-.5)<1e-15
    assert np.max(np.abs(qw@xx))<1e-12
    return dict(status='PASS_U1_PAIRED_MAPPING_FIXED_MATH_SYNTHETIC_CONTROLS',passed=True,
        ridge_expected_1_over_1p1_max_abs_error=shrink,normal_equation_relative_error=proof['normal_equation_relative_error'],
        DA_perfect_target_reconstruction_max_abs_error=perfect,DA_small_direction_not_amplified=True,
        DA_large_direction_only_shrunk=True,zero_exact=True,Q_reference_half_weights=True,
        actual_regression_solves=0,synthetic_only=True,WLS=wp)
