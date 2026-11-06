"""Signed train-fold physical-metric directions and matched spherical experts.

Pure FP64 mathematics. No model, PDMS, routing, dose selection or file access.
"""
import hashlib
import numpy as np
from scipy.linalg import cholesky,cho_solve,eigh,solve_triangular
from scipy.optimize import linear_sum_assignment

MAIN_SEED=2026102491
RANDOM_SEED=2026102492
BOOTSTRAP_SEED=2026102493
MEAN_EPS=1e-8
MAX_ITER=300
REL_TOL=1e-6


def seed(*fields,base=MAIN_SEED):
    text='U1|'+str(base)+'|'+'|'.join(map(str,fields))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)


def token_order(tokens,salt):
    return np.array(sorted(range(len(tokens)),key=lambda j:(hashlib.sha256((str(salt)+'|'+str(tokens[j])).encode()).hexdigest(),j)),np.int64)


def grams(basis,view,fractions):
    if view['view']=='valid_shared384':
        g=basis.T@basis/384.
        return np.broadcast_to(g,(len(fractions),*g.shape)).copy()
    gi=basis[:384].T@basis[:384];gt=basis[384:].T@basis[384:]
    wi=fractions/view['f0'];wt=(1-fractions)/(1-view['f0'])
    return (wi[:,None,None]*gi[None]+wt[:,None,None]*gt[None])/384.


def project(basis,delta,view,fractions):
    """Actual saved-B WLS. No dose forcing; use the unscaled projection a."""
    assert basis.dtype==delta.dtype==np.float64 and fractions.shape==(len(delta),)
    assert np.all((fractions>0)&(fractions<1)) and np.isfinite(delta).all()
    if view['view']=='valid_shared384':
        H=basis.T@basis/384.;rhs=delta@basis/384.
        a=cho_solve((cholesky(H,lower=True),True),rhs.T).T;Ha=a@H
    else:
        assert view['view']=='role2_768'
        gi=basis[:384].T@basis[:384];gt=basis[384:].T@basis[384:]
        values,E=eigh(gi,gi+gt,driver='gvd')
        wi=fractions/view['f0'];wt=(1-fractions)/(1-view['f0'])
        diagonal=(wt[:,None]+(wi-wt)[:,None]*values[None])/384.
        assert np.all(diagonal>0)
        rhs=(wi[:,None]*(delta[:,:384]@basis[:384])+wt[:,None]*(delta[:,384:]@basis[384:]))/384.
        a=((rhs@E)/diagonal)@E.T
        Ha=(wi[:,None]*(a@gi)+wt[:,None]*(a@gt))/384.
    rho=np.sqrt(np.maximum(np.sum(a*Ha,axis=1),0.))
    error=np.linalg.norm(Ha-rhs,axis=1)/np.maximum(np.linalg.norm(rhs,axis=1),1e-300)
    assert np.isfinite(a).all() and error.max(initial=0.)<=1e-10
    return a,rho,rhs,error


def physical_transform(basis,view,all_fractions,training):
    # H_s is affine in the actual image fraction. Average its observed
    # coefficients, avoiding an N*r*r allocation for future high-rank spaces.
    fs=all_fractions[training]
    if view['view']=='valid_shared384':H=basis.T@basis/384.
    else:
        gi=basis[:384].T@basis[:384];gt=basis[384:].T@basis[384:]
        H=(float(np.mean(fs/view['f0']))*gi+float(np.mean((1-fs)/(1-view['f0'])))*gt)/384.
    L=cholesky(H,lower=True)
    proof=dict(Hbar_reconstruction_max_abs=float(np.max(np.abs(H-L@L.T))),
        Hbar_vs_I_over384_max_abs=float(np.max(np.abs(H-np.eye(len(H))/384.))),
        Hbar_all_training_rows_including_NO_DIRECTION=True,
        Hbar_mean_method='Exact affine H_s formula, FP64 averages of observed actual-mask role weights; no N*r*r materialization')
    return H,L,proof


def directions(a,rho,teacher_rms,L):
    epsilon=np.maximum(1e-8,1e-5*teacher_rms);no_direction=rho<=epsilon
    z=a@L;norm=np.linalg.norm(z,axis=1);u=np.zeros_like(z)
    assert np.all(norm[~no_direction]>0)
    u[~no_direction]=z[~no_direction]/norm[~no_direction,None]
    return dict(z=z,u=u,epsilon=epsilon,no_direction=no_direction,z_norm=norm)


def amplitude_scaler(u,rho,epsilon,D0):
    assert len(u)>0 and D0>0
    raw=np.log(np.maximum(rho,epsilon)/D0);median=float(np.median(raw))
    scale=max(1.4826*float(np.median(np.abs(raw-median))),.1)
    ell=np.clip((raw-median)/scale,-3.,3.)
    Vu=float(np.square(u-u.mean(0)).sum(1).mean());Vell=float(np.square(ell-ell.mean()).mean())
    if Vell<=1e-12:gamma=0.;status='AMPLITUDE_DEGENERATE_DIRECTION_ONLY'
    elif Vu<=1e-12:gamma=1.;status='DIRECTION_DEGENERATE_GAMMA_ONE'
    else:gamma=float(np.sqrt(Vu/Vell));status='FINITE_TRAIN_VARIANCE_MATCH'
    return ell,dict(median=median,scale=scale,gamma=gamma,Vu=Vu,Vell=Vell,status=status,
        amplitude_statistics_training_nonzero_only=True)


def transform_amplitude(rho,epsilon,D0,scaler):
    return np.clip((np.log(np.maximum(rho,epsilon)/D0)-scaler['median'])/scaler['scale'],-3.,3.)


def initial_rows(u,K,tokens,random_seed):
    """D² on unit directions = 2(1-cos); omit the irrelevant factor two."""
    unique=len(np.unique(u,axis=0))
    if unique<K:return None,dict(status='INVALID_UNIQUE_DIRECTION_COUNT',unique_directions=unique,K=K)
    rng=np.random.default_rng(random_seed);chosen=[int(rng.integers(len(u)))];fallbacks=[]
    while len(chosen)<K:
        distance=np.maximum(1.-(u@u[chosen].T).max(1),0.)
        distance[chosen]=0.;total=float(distance.sum())
        if total>0:
            j=int(rng.choice(len(u),p=distance/total))
            assert j not in chosen
        else:
            j=next(int(j) for j in token_order(tokens,random_seed) if j not in chosen)
            fallbacks.append(j)
        chosen.append(j)
    return np.array(chosen,np.int64),dict(status='VALID',seed=int(random_seed),rows=chosen,
        tokens=[str(tokens[j]) for j in chosen],all_zero_distance_fallback_rows=fallbacks,
        unique_directions=unique,initialization='Direction kmeans++ with probability proportional to 1-cos')


def costs(u,centers,ell,m,gamma):
    return 1.-u@centers.T+.5*gamma*gamma*np.square(ell[:,None]-m[None])


def finalize_membership(u,ell,centers,m,labels,gamma):
    """No-op conversion never reassigns the last actual training membership."""
    labels=np.array(labels,dtype=np.int64,copy=True);centers=np.array(centers,copy=True)
    before=float(costs(u,centers,ell,m,gamma)[np.arange(len(u)),labels].sum())
    no_op=np.zeros(len(centers),bool)
    for k in range(len(centers)):
        rows=np.flatnonzero(labels==k)
        norm=float(np.linalg.norm(u[rows].mean(0))) if len(rows) else 0.
        if not len(rows) or norm<=MEAN_EPS:centers[k]=0.;no_op[k]=True
    after=float(costs(u,centers,ell,m,gamma)[np.arange(len(u)),labels].sum())
    return centers,labels,no_op,dict(final_objective_before_noop=before,final_objective_after_noop=after,
        final_objective_after_explicit_noop=after,final_membership_retained_after_noop=True,
        final_noop_clusters=np.flatnonzero(no_op).tolist(),finalization_label_changes=0)


def convergence(prior,objective,scene_count,round_reset_count):
    if prior is None:return False,None,None
    improvement=prior-objective;relative=improvement/max(abs(prior),1e-12)
    # Bound accumulation roundoff in a sum of per-scene O(1) objectives. A
    # material increase is never treated as convergence, nor is a reset round.
    roundoff=64*np.finfo(np.float64).eps*max(1.,float(scene_count),abs(prior),abs(objective))
    stop=round_reset_count==0 and improvement>=-roundoff and relative<REL_TOL
    return stop,relative,float(roundoff)


def lloyd(u,ell,K,initial,tokens,gamma,max_iter=MAX_ITER):
    centers=u[initial].copy();m=ell[initial].copy();trace=[];resets=[];prior=None
    order=token_order(tokens,'empty-cluster-tie');tie=np.empty(len(order),np.int64);tie[order]=np.arange(len(order))
    last_labels=None
    for iteration in range(max_iter):
        resets_before=len(resets)
        loss=costs(u,centers,ell,m,gamma);labels=np.argmin(loss,axis=1);nearest=loss[np.arange(len(u)),labels]
        farthest=sorted(range(len(u)),key=lambda j:(-nearest[j],int(tie[j]),j));used=set()
        for k in range(K):
            rows=np.flatnonzero(labels==k)
            mean=u[rows].mean(0) if len(rows) else np.zeros(u.shape[1])
            norm=float(np.linalg.norm(mean))
            if not len(rows) or norm<=MEAN_EPS:
                j=next(j for j in farthest if j not in used);used.add(j)
                centers[k]=u[j];m[k]=ell[j]
                resets.append(dict(iteration=iteration,cluster=k,reason='EMPTY' if not len(rows) else 'MEAN_DIRECTION_DEGENERATE',
                    row=j,token=str(tokens[j]),mode_assignment_cost=float(nearest[j])))
            else:centers[k]=mean/norm;m[k]=float(ell[rows].mean())
        current=costs(u,centers,ell,m,gamma);final_labels=np.argmin(current,axis=1)
        objective=float(current[np.arange(len(u)),final_labels].sum())
        round_resets=len(resets)-resets_before
        stop,relative,roundoff=convergence(prior,objective,len(u),round_resets)
        trace.append(dict(iteration=iteration,objective=objective,relative_improvement=relative,
            round_reset_count=round_resets,absolute_roundoff_tolerance=roundoff,converged_this_round=stop,
            assignments_changed=None if last_labels is None else int(np.count_nonzero(final_labels!=last_labels))))
        prior=objective;last_labels=final_labels.copy()
        if stop:break
    # The last assignment is authoritative. A truly degenerate/empty final
    # group remains an explicit no-op center instead of changing K.
    centers,final_labels,no_op,final_proof=finalize_membership(u,ell,centers,m,final_labels,gamma)
    objective=final_proof['final_objective_after_noop']
    return dict(status='VALID',centers=centers,amplitude_centers=m,labels=final_labels,no_op=no_op,
        objective=objective,iterations=len(trace),trace=trace,resets=resets,
        **final_proof)


def cluster_pair(u,ell,K,tokens,identity,scaler,restarts=20,initial_base=MAIN_SEED):
    """Matched initialization rows; each mode selects its own train objective."""
    unique=len(np.unique(u,axis=0))
    if unique<K:return dict(status='INVALID_UNIQUE_DIRECTION_COUNT',unique_directions=unique,K=K,initializations=[],modes={})
    if K==1:
        mean=u.mean(0);norm=float(np.linalg.norm(mean));noop=norm<=MEAN_EPS
        center=np.zeros((1,u.shape[1])) if noop else mean[None]/norm
        labels=np.zeros(len(u),np.int64)
        b=dict(status='VALID',centers=center,amplitude_centers=np.array([float(ell.mean())]),labels=labels,
            no_op=np.array([noop]),objective=float((1.-u@center.T[:,0]).sum()),iterations=0,trace=[],resets=[],
            selected_restart=None,K1_all_training_unit_direction_mean=True)
        return dict(status='VALID',unique_directions=unique,K=1,initializations=[],modes={'direction':b},
            joint_mode_alias='direction',K1_fitted_once=True)
    inits=[];trials={'direction':[],'direction_logamp':[]}
    for restart in range(restarts):
        sd=seed(*identity,K,restart,base=initial_base);rows,proof=initial_rows(u,K,tokens,sd);assert rows is not None
        inits.append(proof|dict(restart=restart))
        for mode,gamma in [('direction',0.),('direction_logamp',scaler['gamma'])]:
            fit=lloyd(u,ell,K,rows,tokens,gamma)
            fit.update(restart=restart,initialization=proof,gamma=gamma)
            trials[mode].append(fit)
    modes={}
    for mode,all_trials in trials.items():
        winner=min(range(restarts),key=lambda j:(all_trials[j]['objective'],j))
        best=all_trials[winner]
        best['restart_summaries']=[dict(restart=t['restart'],objective=t['objective'],iterations=t['iterations'],
            no_op=t['no_op'].tolist(),reset_count=len(t['resets']),trace=t['trace'],resets=t['resets'],
            final_objective_before_noop=t['final_objective_before_noop'],
            final_objective_after_noop=t['final_objective_after_noop'],
            final_membership_retained_after_noop=t['final_membership_retained_after_noop']) for t in all_trials]
        best['selected_restart']=winner;modes[mode]=best
    return dict(status='VALID',unique_directions=unique,K=K,initializations=inits,modes=modes)


def center_coefficients(L,centers):
    return solve_triangular(L.T,centers.T,lower=False).T


def assign(u,ell,centers,m,gamma,no_direction):
    result=np.argmin(costs(u,centers,ell,m,gamma),axis=1).astype(np.int64)
    result[no_direction]=-1
    if not np.any(np.linalg.norm(centers,axis=1)>0):result[~no_direction]=-2
    return result


def random_groups(u,labels,tokens,identity,rep,K=None):
    K=int(labels.max())+1 if K is None else int(K)
    assert np.all((labels>=0)&(labels<K))
    size=np.bincount(labels,minlength=K)
    sd=seed(*identity,K,rep,base=RANDOM_SEED);rng=np.random.default_rng(sd);permutation=rng.permutation(len(u))
    random_labels=np.empty(len(u),np.int64);centers=np.zeros((K,u.shape[1]));no_op=np.zeros(K,bool);offset=0
    for k,n in enumerate(size):
        rows=permutation[offset:offset+n];random_labels[rows]=k;offset+=n
        mean=u[rows].mean(0) if len(rows) else np.zeros(u.shape[1]);norm=float(np.linalg.norm(mean))
        if norm>MEAN_EPS:centers[k]=mean/norm
        else:no_op[k]=True
    assert np.array_equal(np.bincount(random_labels,minlength=K),size)
    return dict(seed=sd,labels=random_labels,centers=centers,no_op=no_op,sizes=size,permutation=permutation,
        grouping='Size-preserving random permutation of training valid unit directions; noncompetitive control')


def adjusted_rand(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    _,ai=np.unique(a,return_inverse=True);_,bi=np.unique(b,return_inverse=True)
    C=np.zeros((ai.max()+1,bi.max()+1),np.int64);np.add.at(C,(ai,bi),1)
    choose=lambda x:np.sum(x*(x-1)/2)
    n=len(a);pairs=n*(n-1)/2
    if not pairs:return 1.
    x=choose(C);r=choose(C.sum(1));c=choose(C.sum(0));expected=r*c/pairs;den=.5*(r+c)-expected
    return float((x-expected)/den) if abs(den)>1e-15 else 1.


def selftest():
    rng=np.random.default_rng(2026102497);B=np.linalg.qr(rng.normal(size=(768,8)))[0]
    fs=np.array([.2,.4,.7,.8]);view=dict(view='role2_768',f0=float(fs[:3].mean()))
    delta=rng.normal(size=(4,768));a,rho,rhs,error=project(B,delta,view,fs)
    H,L,proof=physical_transform(B,view,fs,np.arange(3));Hs=grams(B,view,fs)
    independent=np.stack([np.linalg.solve(h,b) for h,b in zip(Hs,rhs)])
    assert np.max(np.abs(a-independent))<1e-11
    z=a@L;assert np.max(np.abs(np.square(z).sum(1)-np.einsum('ni,ij,nj->n',a,H,a)))<1e-12
    assert np.max(np.abs(center_coefficients(L,z)-a))<1e-12
    # Opposite signs remain different directions. K1 cancels to exact Base.
    u=np.array([[1.,0.],[-1.,0.],[1.,0.],[-1.,0.]])
    ell,scaler=amplitude_scaler(u,np.ones(4),np.full(4,1e-8),.5)
    assert scaler['gamma']==0.
    one=cluster_pair(u,ell,1,list('abcd'),(0,'synthetic',2),scaler)
    assert one['modes']['direction']['no_op'][0] and not np.count_nonzero(one['modes']['direction']['centers'])
    pair=cluster_pair(u,ell,2,list('abcd'),(0,'synthetic',2),scaler,restarts=3)
    d=pair['modes']['direction'];j=pair['modes']['direction_logamp']
    assert np.array_equal(d['centers'],j['centers']) and np.array_equal(d['labels'],j['labels'])
    assert d['objective']<1e-12 and float(d['centers'][0]@d['centers'][1])<-.999999
    invalid=cluster_pair(u,ell,3,list('abcd'),(0,'synthetic',2),scaler,restarts=3)
    assert invalid['status']=='INVALID_UNIQUE_DIRECTION_COUNT'
    random=random_groups(u,d['labels'],list('abcd'),(0,'synthetic',2),0)
    assert np.array_equal(np.bincount(random['labels']),np.bincount(d['labels']))
    assert adjusted_rand(np.array([0,0,1,1]),np.array([1,1,0,0]))==1.
    # Valid exact-distinct directions whose angular differences round to zero
    # force an empty/reset cluster on every round. V1 stopped after round two.
    near=np.array([[1.,0.],[1.,1e-12],[-1.,0.]])
    reset_fit=lloyd(near,np.zeros(3),3,np.arange(3),['a','b','c'],0.,max_iter=5)
    assert len(np.unique(near,axis=0))==3 and reset_fit['iterations']==5
    assert all(t['round_reset_count']>0 and not t['converged_this_round'] for t in reset_fit['trace'])
    assert not convergence(1.,1.01,100,0)[0]
    assert convergence(1.,1.+np.finfo(float).eps,100,0)[0]
    # Final centers can become no-ops while the last actual assignment must
    # survive. Re-running argmin here demonstrably moves the positive row.
    points=np.array([[1.,0.,0.],[-1.,0.,0.],[.5,np.sqrt(.75),0.]])
    cc=np.array([[.5,0.,np.sqrt(.75)],[.5,np.sqrt(.75),0.]])
    assigned=np.argmin(costs(points,cc,np.zeros(3),np.zeros(2),0.),axis=1)
    assert assigned.tolist()==[0,0,1]
    final,kept,noops,fp=finalize_membership(points,np.zeros(3),cc,np.zeros(2),assigned,0.)
    would_move=np.argmin(costs(points,final,np.zeros(3),np.zeros(2),0.),axis=1)
    assert noops.tolist()==[True,False] and np.array_equal(kept,assigned) and not np.array_equal(would_move,assigned)
    assert fp['final_objective_before_noop']==2. and fp['final_objective_after_noop']==2.
    empty_last=random_groups(points,np.array([0,0,1]),['a','b','c'],('empty-last',),0,K=3)
    assert empty_last['centers'].shape==(3,3) and empty_last['sizes'].tolist()==[2,1,0]
    assert empty_last['no_op'][2]
    return dict(status='PASS_CPU_DIRECTION_EXPERT_PHYSICAL_METRIC_AND_MATCHED_CLUSTER_SYNTHETIC',passed=True,
        actual_H_WLS_max_abs_error=float(np.max(np.abs(a-independent))),Hbar=proof,
        row_z_equals_LT_a=True,inverse_center_transform_exact=True,signed_antipodes_not_merged=True,
        K1_cancelled_mean_exact_noop=True,amplitude_degenerate_joint_matches_direction=True,
        matched_initial_scene_IDs=True,invalid_unique_directions_not_clamped=True,random_cluster_sizes_preserved=True,
        reset_rounds_never_converge=dict(iterations=reset_fit['iterations'],reset_counts=[t['round_reset_count'] for t in reset_fit['trace']]),
        significant_objective_increase_not_convergence=True,roundoff_sized_negative_improvement_permitted=True,
        random_control_retains_requested_K_with_empty_last_cluster=True,
        final_noop_membership_regression=dict(last_assignment=assigned.tolist(),retained_assignment=kept.tolist(),
            forbidden_reargmin_would_assign=would_move.tolist(),**fp),
        actual_data_clustering=False,models_scores_or_native_calls=0)
