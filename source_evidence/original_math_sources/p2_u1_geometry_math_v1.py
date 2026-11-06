"""Pure FP64 U1 geometry. No labels, model, scoring or candidate selection."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import numpy as np

KS=(2,4,8,16,24,32,48,64,96,128,192,256,384,512,768)
RS=(1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,768)
CLASS_RELATIVE_EIGENVALUE=1e-10
A_RESIDUAL_SINGULAR_VALUE=1e-6
B_RELATIVE_SINGULAR_VALUE=1e-10


def signs(u):
    """Largest absolute coordinate positive; argmax resolves ties by index."""
    u=np.array(u,dtype=np.float64,copy=True)
    if u.shape[1]:
        pivot=np.argmax(np.abs(u),axis=0)
        sign=np.where(u[pivot,np.arange(u.shape[1])]<0,-1.,1.)
        u*=sign[None]
    return u


def view_vectors(f,s,counts,view,training_rows):
    assert f.dtype==s.dtype==np.float64 and f.shape==s.shape and f.ndim==3 and f.shape[1:]==(2,384)
    assert counts.shape==f.shape[:2] and np.all(counts>0)
    fractions=counts[:,0]/counts.sum(1)
    f0=float(fractions[training_rows].mean())
    assert 0<f0<1
    if view=='valid_shared384':
        fv=np.einsum('ng,ngd->nd',counts/counts.sum(1)[:,None],f)
        sv=np.einsum('ng,ngd->nd',counts/counts.sum(1)[:,None],s)
        decode=np.ones(2,np.float64)
    else:
        assert view=='role2_768'
        w=np.sqrt(np.array([f0,1-f0],np.float64))
        fv=(f*w[None,:,None]).reshape(len(f),768)
        sv=(s*w[None,:,None]).reshape(len(f),768)
        decode=1/w
    assert np.isfinite(fv).all() and np.isfinite(sv).all()
    return fv,sv,dict(view=view,dimension=fv.shape[1],f0=f0,decode_role_multipliers=decode.tolist(),
        f0_fitted_from_training_rows_only=True,actual_mask_RMS_required_at_native_execution=True)


def class_svd(x):
    """C=mean(xx^T), x has rows representing scenes, equal scene weights."""
    assert x.ndim==2 and x.dtype==np.float64 and len(x)>0 and np.isfinite(x).all()
    _,sigma,vt=np.linalg.svd(x,full_matrices=False)
    eig=np.square(sigma)/len(x)
    rank=int(np.count_nonzero(eig>eig[0]*CLASS_RELATIVE_EIGENVALUE)) if len(eig) and eig[0]>0 else 0
    return dict(basis=signs(vt[:rank].T),eigenvalues=eig,singular_values=sigma,effective_rank=rank,
        relative_eigenvalue_cutoff=CLASS_RELATIVE_EIGENVALUE)


def paired_classes(f,s,centered):
    assert f.shape==s.shape and f.ndim==2
    center=(f.mean(0)+s.mean(0))/2 if centered else np.zeros(f.shape[1],np.float64)
    return dict(center=center,failure=class_svd(f-center),success=class_svd(s-center))


def residual(failed,successful,ks,kf,weighted):
    """Residualize FAILURE basis against success; never reverse this order."""
    if ks>successful['basis'].shape[0] or kf>failed['basis'].shape[0]:
        return dict(status='INVALID_DIMENSION',ks=ks,kf=kf)
    if ks>successful['effective_rank']:
        return dict(status='INVALID_SUCCESS_EFFECTIVE_RANK',ks=ks,kf=kf)
    if kf>failed['effective_rank']:
        return dict(status='INVALID_FAILURE_EFFECTIVE_RANK',ks=ks,kf=kf)
    us=successful['basis'][:,:ks];uf=failed['basis'][:,:kf]
    raw=uf-us@(us.T@uf)
    if weighted:
        eig=failed['eigenvalues']
        raw=raw*np.sqrt(eig[:kf]/eig[0])[None]
    u,sigma,_=np.linalg.svd(raw,full_matrices=False)
    rank=int(np.count_nonzero(sigma>A_RESIDUAL_SINGULAR_VALUE))
    return dict(status='VALID_RESIDUAL_DECOMPOSITION',ks=ks,kf=kf,basis=signs(u[:,:rank]),
        singular_values=sigma,effective_rank=rank,weighted=weighted,
        singular_value_cutoff=A_RESIDUAL_SINGULAR_VALUE)


def mgs_columns(candidates,cutoff=1e-10):
    accepted=[];skipped=[];remainders=[]
    for k,column in enumerate(candidates.T):
        v=column.copy()
        for _ in range(2):
            if accepted:
                q=np.stack(accepted,axis=1);v-=q@(q.T@v)
        norm=float(np.linalg.norm(v));remainders.append(norm)
        if norm<=cutoff:
            skipped.append(k)
        else:
            accepted.append(v/norm)
    d=candidates.shape[0]
    basis=signs(np.stack(accepted,axis=1)) if accepted else np.empty((d,0),np.float64)
    return basis,dict(skipped_candidate_columns=skipped,candidate_residual_norms=remainders,
        modified_Gram_Schmidt_passes=2,dependent_column_norm_cutoff=cutoff)


def delta_space(f,s,actual_rms,normalized,centered):
    delta=s-f
    assert actual_rms.shape==(len(delta),) and np.isfinite(actual_rms).all() and np.all(actual_rms>=0)
    floor=max(1e-12,1e-3*float(np.median(actual_rms)))
    x=delta/np.maximum(actual_rms,floor)[:,None] if normalized else delta
    mean=x.mean(0);row_rms=float(np.sqrt(np.square(x).sum(1).mean()))
    mean_zero_cutoff=1e-12*max(row_rms,1e-12)
    mean_nonzero=float(np.linalg.norm(mean))>mean_zero_cutoff
    z=x-mean if centered else x
    _,sigma,vt=np.linalg.svd(z,full_matrices=False)
    top=float(sigma[0]) if len(sigma) else 0.
    ratios=sigma/top if top else np.zeros_like(sigma)
    residual_rank=int(np.count_nonzero(ratios>B_RELATIVE_SINGULAR_VALUE))
    base=signs(vt[:residual_rank].T)
    mgs=dict(skipped_candidate_columns=[],candidate_residual_norms=[],modified_Gram_Schmidt_passes=0)
    if centered:
        columns=[mean[:,None]/np.linalg.norm(mean)] if mean_nonzero else []
        columns.append(base)
        base,mgs=mgs_columns(np.concatenate(columns,axis=1))
    return dict(status='VALID_DELTA_DECOMPOSITION',basis=base,effective_rank=base.shape[1],
        singular_values=sigma,relative_singular_values=ratios,residual_effective_rank=residual_rank,
        relative_singular_value_cutoff=B_RELATIVE_SINGULAR_VALUE,mean=mean,
        mean_nonzero=mean_nonzero,mean_zero_cutoff=mean_zero_cutoff,row_vector_RMS=row_rms,
        normalized=normalized,centered=centered,normalization_floor=floor,
        normalization_floor_scene_count=int(np.count_nonzero(actual_rms<floor)),
        mean_is_a_basis_direction_not_a_runtime_intercept=True,**mgs)


def oof_diagnostics(f,s,center,basis,ranks):
    """Per-scene energy identities only; no native or policy-gain claims."""
    assert ranks and max(ranks)<=basis.shape[1]
    fc=f-center;sc=s-center;delta=s-f
    p=fc@basis;q=sc@basis;t=delta@basis
    pp=np.cumsum(np.square(p),axis=1);qq=np.cumsum(np.square(q),axis=1)
    tt=np.cumsum(np.square(t),axis=1);pt=np.cumsum(p*t,axis=1)
    e=np.square(delta).sum(1)
    columns=np.asarray(ranks,np.int64)-1
    values=np.stack([pp[:,columns],qq[:,columns],tt[:,columns],
        e[:,None]+2*pt[:,columns]+pp[:,columns]],axis=2)
    assert np.isfinite(values).all()
    return values


def selftest():
    # Axis-aligned example detects the scientifically wrong reversed residual.
    failure=dict(basis=np.eye(4,dtype=np.float64)[:,[0,1]],effective_rank=2,eigenvalues=np.array([9.,1.]))
    success=dict(basis=np.eye(4,dtype=np.float64)[:,[0]],effective_rank=1,eigenvalues=np.array([1.]))
    a=residual(failure,success,1,2,False);w=residual(failure,success,1,2,True)
    assert a['effective_rank']==w['effective_rank']==1
    assert np.array_equal(a['basis'][:,0],np.array([0.,1.,0.,0.]))
    assert abs(w['singular_values'][0]-1/3)<1e-15
    identical=residual(failure,failure,2,2,False);assert identical['effective_rank']==0
    f=np.array([[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[2.,0.,0.]],np.float64)
    s=f+np.array([[0.,2.,1.],[0.,2.,-1.],[0.,2.,2.],[0.,2.,-2.]],np.float64)
    b=delta_space(f,s,np.ones(4),False,True)
    assert b['effective_rank']==2 and b['mean_nonzero']
    assert np.array_equal(b['basis'][:,0],np.array([0.,1.,0.]))
    assert np.linalg.norm(b['basis'].T@b['basis']-np.eye(2))<1e-14
    z=delta_space(f,f,np.zeros(4),True,True);assert z['effective_rank']==0 and not z['mean_nonzero']
    diagnostics=oof_diagnostics(f,s,np.zeros(3),b['basis'],[1,2])
    for j,r in enumerate((1,2)):
        projected=f-(f@b['basis'][:,:r])@b['basis'][:,:r].T
        assert np.max(np.abs(diagnostics[:,j,3]-np.square(s-projected).sum(1)))<1e-13
    pooled=np.stack([np.arange(384),-np.arange(384)],axis=0).astype(np.float64)[None]
    counts=np.array([[3,1],[1,3]],np.int64);p=np.repeat(pooled,2,axis=0)
    fv,sv,meta=view_vectors(p,p,counts,'role2_768',np.array([0]))
    decoded=fv.reshape(2,2,384)*np.array(meta['decode_role_multipliers'])[None,:,None]
    assert np.max(np.abs(decoded-p))<1e-13
    assert meta['f0']==.75
    return dict(status='PASS_U1_GEOMETRY_PURE_MATH_SELFTEST',failure_basis_residual_order=True,
        variance_sqrt_lambda_ratio=True,identical_full_subspaces_rank_zero=True,
        centered_delta_mean_preserved_zero_reachable=True,all_zero_delta_rank_zero=True,
        role2_training_only_f0_and_decode=True,OOF_energy_identity=True,
        real_scene_fits=0,GPU_or_model_or_scoring=False)
