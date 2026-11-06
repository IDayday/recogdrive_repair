"""Outer0 fresh train refit and membership-only slot transfer, CPU only.

No fit action is possible without the actual full-cal freeze and ROOT release.
Held teachers are never opened: pairs are read individually for train Q only.
"""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,hashlib,itertools,json,time,traceback
from collections import Counter,OrderedDict
import numpy as np
from scipy.optimize import linear_sum_assignment
from b5_common import read,atomic,stamp,sha
import ol_l1_pilot_fitter_v1 as inner
import ol_l2_cal_pack_v1 as cal
data,ops,geo=inner.data,inner.ops,inner.geo
ref,meta,checked=data.ref,data.meta,data.checked
ROOT=OL/'preparation/l3_outer_refit_v1'
FREEZE=OL/'analysis/l2_formal_selection_v2/FROZEN_LIBRARIES.json'
SELECTOR=OL/'code/ol_l2_formal_selection_v2.py'
PAIR_PACK=data.ROOT/'PACKS/PAIRS1850_V1/RESULT.json'
KNOWN=P2/'provenance/OFFLINE_LIBRARY_L1_MLP_KNOWN2000_ACTUAL_ROOT_ACCEPTANCE_V1.json'
FREEZE_STATUS='FROZEN_L2_ALL12_LIBRARIES_AFTER_FULL_INNERCAL1681_S012_V1'
PLAN_STATUS='READY_OUTER0_FRESH_REFIT_TRANSFER_AND_HELD1987_PLAN_NOT_EXECUTION_V1'
RELEASE_STATUS='PASS_ROOT_OFFLINE_LIBRARY_L3_OUTER0_CPU_REFIT_EXECUTION_RELEASE_V1'
BANK_STATUS='COMPLETE_OUTER0_FRESH_SELECTED_SLOT_REFIT_REQUIRES_ROOT_V1'
REQUEST_STATUS='COMPLETE_OUTER0_HELD1987_ORDINARY_REQUESTS_REQUIRES_NATIVE_VALIDATION_V1'
_CACHE=OrderedDict()


def sources():
    s=cal.sources()
    for p in (__file__,cal.__file__):s[str(Path(p).resolve())]=sha(p)
    return s
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(folder,payload,binding):return inner.saved(folder,payload,binding)
def result(path,value):atomic(path,value,exclusive=True);return ref(path)

def metadata():
    proof=read(cal.DATA_PROOF);assert proof['passed'] and proof['status']==inner.DATA_STATUS
    split=meta(proof['session_split']);fold=next(f for f in split['folds'] if f['fold']==0);inputs=meta(proof['inputs']);rows={r['index']:r for r in inputs['cases']}
    pairs=read(PAIR_PACK);assert ref(PAIR_PACK) in proof['source_results'] and pairs['passed'] and pairs['producer']==ref(data.__file__)
    assert pairs['full1850']==proof['full1850'] and len(pairs['indices'])==len(pairs['cases'])==1850
    train=fold['outer_train'];held=fold['outer_held'];ti=train['selected_indices'];hi=held['selected_indices']
    assert len(ti)==8055 and len(hi)==1987 and set(ti).isdisjoint(hi) and set(ti)|set(hi)==set(inputs['indices'])
    assert len(train['sessions'])==104 and len(held['sessions'])==26 and set(train['sessions']).isdisjoint(held['sessions'])
    for group in (train,held):
        assert Counter(rows[i]['audit_category'] for i in group['selected_indices'])==group['selected_counts']
        assert all(rows[i]['session'] in group['sessions'] for i in group['selected_indices'])
        for c,n in group['selected_counts'].items():
            assert np.isclose(group['conditional_natural_mean_weight_per_sample'][c],group['full_population_counts'][c]/group['full_population_scenes']/n,rtol=1e-14,atol=0)
    qualified=set(pairs['canonical_qualified_indices']);Q=[i for i in ti if i in qualified];normal=[i for i in ti if rows[i]['audit_category'] in ('mid','high')];high=[i for i in normal if rows[i]['audit_category']=='high']
    oldplan=read(inner.ROOT/'PLAN_FINAL.json');assert set(oldplan['qualified_fit_indices'])<=set(Q)
    return dict(proof=proof,fold=fold,rows=rows,train=ti,held=hi,Q=Q,normal=normal,high=high,
        case_sources={i:r for i,r in zip(pairs['indices'],pairs['cases'])},oldplan=oldplan)

def prepare(freeze_path,output,selector_source=SELECTOR):
    d=metadata();f=read(freeze_path);assert f['passed'] and f['status']==FREEZE_STATUS and f['producer']==ref(selector_source)
    reg=meta(f['source_registration']);assert reg['passed'] and reg['source']==f['producer'] and reg['calibration_plan']==f['calibration_plan'] and reg['actual_L2_behavior_scores_read']==0
    cp=meta(f['calibration_plan']);assert cp['producer']==ref(cal.__file__) and cp['source_files']==cal.sources()
    accepted=meta(f['source_native_acceptance']);q=meta(accepted['queue'])
    assert accepted['passed'] and accepted['status']=='PASS_CPU_L2_NATIVE_ALL_EXPERT_CROSS_APPLICATION_REQUIRES_ROOT_V1' and q['scope']=='main1681'
    assert q['noise_indices']==[0,1,2] and accepted['scenes']==1681 and q['source_files']==accepted['source_files']
    assert meta(q['preparation'])['calibration_plan']==f['calibration_plan']
    assert sum(len(j['positions'])*3 for j in q['jobs'])==cp['logical_expert_outputs']
    request_by_unit={}
    for job in q['jobs']:
        name=job['unit']
        if name in request_by_unit:assert request_by_unit[name]==job['request']
        else:request_by_unit[name]=job['request']
    units=[];libraries=f['libraries'];assert len(libraries)==len({r['unit'] for r in libraries})==12
    assert {r['unit'] for r in libraries}=={u['id'] for u in cp['units']}
    for library in libraries:
        bank=meta(library['inner_bank']);request=meta(library['inner_request']);spec=bank['spec'];slots=library['slots']
        assert library['spec']==spec and library['unit']==spec['id'] and request['bank']==library['inner_bank'] and bank['source_files']==inner.sources()
        assert library['inner_request']==request_by_unit[library['unit']] and library['inner_bank']==next(u['bank'] for u in cp['units'] if u['id']==library['unit'])
        assert request['source_files']==cal.sources() and request['plan']==f['calibration_plan'] and request['noise_indices']==[0,1,2]
        assert len(slots)==spec['K'] and [s['cluster'] for s in slots]==list(range(spec['K']))
        for s in slots:
            if s['Base']:assert (s['method'],s['alpha'],s['cap_D0'],s['selected_position'])==('Base',0,0,-1)
            else:
                pos=s['selected_position'];a=request['actions'][pos]
                assert 0<=pos<len(request['actions']) and s['source_action_id']==a['action_id']
                assert all(s[k]==a[k] for k in ('cluster','method','alpha','cap_D0')) and s['method'] in ops.MENU
        units.append(dict(id=library['unit'],spec=spec,inner_bank=library['inner_bank'],inner_request=library['inner_request'],slots=slots))
    by={u['id']:u for u in units}
    for u in units:
        parent=f"{u['spec']['route']}__P_to_C__K1";u['dependencies']=[]
        if u['spec']['K']==1 and u['spec']['order']=='C_to_P':
            other=by[parent];keys=('Base','method','alpha','cap_D0')
            if all(u['slots'][0][k]==other['slots'][0][k] for k in keys):u['dependencies']=[parent]
    known=read(KNOWN);assert known['passed'] and known['status']==inner.KNOWN_ROOT_STATUS and known['source_files']==inner.sources() and set(known['controls'])=={'mlp','mlp_replay'}
    for cr in known['controls'].values():
        c=meta(cr);assert c['passed'] and c['all_roles_absorption_ge90'] and c['endpoint']==2000 and c['replay1999_to2000_exact']
    p=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),formal_selection=ref(freeze_path),selector_source=f['producer'],selector_registration=f['source_registration'],
        data_proof=ref(cal.DATA_PROOF),session_split=d['proof']['session_split'],pair_pack=ref(PAIR_PACK),ordinary_inputs=d['proof']['inputs'],known_kernel_root=ref(KNOWN),
        units=units,outer_fold=0,train_indices=d['train'],Q_indices=d['Q'],normal_indices=d['normal'],high_indices=d['high'],held_indices=d['held'],
        train_sessions=d['fold']['outer_train']['sessions'],held_sessions=d['fold']['outer_held']['sessions'],held_group=d['fold']['outer_held'],
        inner_Q_indices=d['oldplan']['qualified_fit_indices'],train_pair_sources=[d['case_sources'][i] for i in d['Q']],output_root=str(ROOT/'ACTUAL'),
        only_selected_methods_refitted=True,all12_libraries_retained=True,no_held_teacher_payloads_allowed=True,
        matching='integer max-overlap; fixed old slot then lexicographic decoded physical-center SHA feasibility; exact hash tie new integer slot',
        held_logical_expert_outputs=1987*sum(u['spec']['K'] for u in units)*5)
    return result(output,p)

def context(plan,release,action,unit=None):
    p=read(plan);r=read(release)
    assert p['status']==PLAN_STATUS and p['source_files']==sources() and p['producer']==ref(__file__) and p['outer_fold']==0
    assert r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan) and r['source_files']==sources() and r['formal_selection']==p['formal_selection']
    assert action in r['allowed_actions'];meta(p['formal_selection']);assert ref(checked(p['selector_source']))==p['selector_source']
    if unit is not None:assert unit in r['allowed_units'] and unit in {u['id'] for u in p['units']}
    return p,Path(p['output_root'])

def train_data(plan,release):
    p,root=context(plan,release,'data');path=root/'DATA/RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    proof=meta(p['data_proof']);o=data.small_arrays(proof['ordinary_arrays']);pos={int(i):n for n,i in enumerate(o['indices_int64'])};by={}
    for i,record in zip(p['Q_indices'],p['train_pair_sources']):
        r=meta(record);assert r['index']==i and r['canonical_qualified_pair'] and r['full1850']==proof['full1850'];a=data.small_arrays(r['arrays']);assert bool(a['canonical_qualified_pair_bool'])
        for k in ('Base_pooled_float64','token_counts_int64','raw_aux24_float32'):assert a[k].tobytes()==o[k][pos[i]].tobytes()
        assert float(a['canonical_Base_S0_score_float64'])<.7 and float(a['canonical_teacher_S0_score_float64'])>.9;by[i]=a
    B=np.stack([by[i]['Base_pooled_float64'] for i in p['Q_indices']]);Y=np.stack([by[i]['teacher_pooled_float64'] for i in p['Q_indices']]);counts=np.stack([by[i]['token_counts_int64'] for i in p['Q_indices']]);rms=np.array([float(by[i]['actual_fulltoken_teacher_RMS_float64']) for i in p['Q_indices']],np.float64)
    assert np.isfinite(Y).all() and np.all(rms>0)
    ar=save(path.parent/'STATE',dict(Q_indices=np.array(p['Q_indices'],np.int64),Base=B,teacher=Y,counts=counts,rms=rms,
        aux=np.stack([by[i]['raw_aux24_float32'] for i in p['Q_indices']])),dict(plan=ref(plan),source_files=sources(),scope='outer_train_Q_only'))
    return result(path,dict(status='COMPLETE_OUTER_TRAIN_Q_ONLY_CPU_DATA_NOT_FIT',passed=True,plan=ref(plan),source_files=sources(),artifact=ar,
        pair_sources=p['train_pair_sources'],actual_train_pair_count=len(by),held_teacher_arrays_opened=0,aggregate_PAIRS1850_arrays_opened=0))

def space(plan,release,route):
    p,root=context(plan,release,'space');assert route in ('R','D');path=root/'SPACES'/route/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    dr=read(root/'DATA/RESULT.json');assert dr['plan']==ref(plan) and dr['source_files']==sources();d=inner.loaded(dr['artifact']);meta_rows=metadata()['rows']
    f,s,view=geo.old_geometry.view_vectors(d['Base'],d['teacher'],d['counts'],'valid_shared384',np.arange(len(d['Q_indices'])))
    fitted=geo.fit_space(f,s,d['rms'],route,'valid_shared384',48);D0=float(np.median(d['rms']))
    identity=digest(dict(data_proof=p['data_proof'],Q=p['Q_indices'],normal=p['normal_indices'],high=p['high_indices'],fit_sessions=p['train_sessions'],view='valid_shared384',outer_fold=0,scope='outer0_outer_train_refit'))
    state=dict(space=fitted,view=view,D0=D0,Q_indices=d['Q_indices'],fail_view=f,success_view=s,actual_fulltoken_RMS=d['rms'],raw_aux=d['aux'],
        recipe=dict(route=route,view='valid_shared384',rank=48),fit_identity=identity,tokens=[meta_rows[i]['token'] for i in p['Q_indices']],sessions=[meta_rows[i]['session'] for i in p['Q_indices']],
        shared_R_D_and_order_clustering_seed_identity=True)
    if fitted['status']=='VALID_SPACE':state['geometry']=geo.geometry(fitted['basis'],view,d['counts'][:,0]/d['counts'].sum(1),s-f,d['rms'])
    ar=save(path.parent/'STATE',state,dict(plan=ref(plan),source_files=sources(),route=route))
    decoder=data.arrays(path.parent/'DECODER.npz',dict(basis_float64=fitted['basis'],center_float64=fitted['center'],D0_float64=np.array(D0)))
    return result(path,dict(status='COMPLETE_FRESH_OUTER_TRAIN_SPACE_REQUIRES_ROOT',passed=True,plan=ref(plan),source_files=sources(),artifact=ar,decoder=decoder,view=view,
        space_status=fitted['status'],fit_identity=identity,Q=len(f),D0=D0,no_inner_basis_or_scale_reused=True))

def max_overlap(C):
    if not len(C):return 0
    rows,cols=linear_sum_assignment(C,maximize=True);return int(C[rows,cols].sum())

def exact_matching(C,hashes):
    """Integer primary optimum, then lexicographic feasible completion. No eps."""
    assert C.dtype==np.int64 and C.ndim==2 and C.shape[0]==C.shape[1]==len(hashes) and np.all(C>=0)
    target=max_overlap(C);remaining=target;columns=list(range(len(C)));chosen=[];trace=[]
    for row in range(len(C)):
        attempts=[]
        for j in sorted(columns,key=lambda j:(hashes[j],j)):
            other=[v for v in columns if v!=j];possible=int(C[row,j])+max_overlap(C[row+1:,other])
            attempts.append(dict(new_cluster=j,physical_hash=hashes[j],feasible=possible==remaining,maximum_remaining_overlap=possible))
            if possible==remaining:break
        else:raise AssertionError('No optimum-preserving lexicographic completion')
        chosen.append(j);columns.remove(j);remaining-=int(C[row,j]);trace.append(dict(old_slot=row,chosen_new_cluster=j,attempts=attempts))
    assert remaining==0 and sum(int(C[i,j]) for i,j in enumerate(chosen))==target
    return dict(old_to_new=chosen,new_to_old=np.argsort(chosen).tolist(),maximum_integer_overlap=target,trace=trace,
        tie_rule='old slot ascending; physical decoded centroid SHA ascending; identical SHA only -> new slot ascending',no_floating_epsilon=True)

def adjusted_rand(C):
    comb=lambda a:sum(int(v)*(int(v)-1)//2 for v in np.asarray(a).flat)
    n=int(C.sum());total=n*(n-1)//2
    if not total:return 1.
    a,b,c=comb(C),comb(C.sum(1)),comb(C.sum(0));den=.5*(b+c)*total-b*c
    return 1. if den==0 else float((a*total-b*c)/den)

def physical_hash(common,state):
    if common is None or common['centroid_noop']:direction=np.zeros((2,384),np.float64)
    else:
        direction=geo.decode_roles(common['centroid_coefficients']/state['D0'],state['space']['basis'],state['view']);length=np.linalg.norm(direction)
        assert length>0 and np.isfinite(length);direction=direction/length;direction[direction==0]=0.
    return geo.digest(direction),direction

def fit_selected_paired(v,aux,normal,normal_aux,target,D0,method):
    r=target['target_coefficients'].shape[1];x=np.r_[np.c_[v,aux],np.c_[normal,normal_aux]].astype(np.float64)
    width=r if method=='paired_RRR' else r+1;T=np.zeros((len(x),width),np.float64)
    T[:len(v)]=target['RRR_targets'] if method=='paired_RRR' else np.c_[target['DA_direction_targets'],target['DA_logamplitude_targets']]
    w=ops.normalized_weights(dict(Q=len(v),ordinary_reference_ge70=len(normal)))
    return dict(method=method,rank=r,D0=D0,learned=ops.ridge(x,T,w,False),role_total_weights=dict(Q=.5,ordinary_reference_ge70=.5),
        only_selected_method_RHS_solved=True,no_target_whitening=True)

def refit(plan,release,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,root=context(plan,release,'refit',unit);u=next(u for u in p['units'] if u['id']==unit);path=root/'BANKS'/unit/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    sp=read(root/'SPACES'/u['spec']['route']/'RESULT.json');assert sp['plan']==ref(plan);state=inner.loaded(sp['artifact']);K=u['spec']['K'];valid=state['space']['status']=='VALID_SPACE'
    g=state.get('geometry');cluster=geo.cluster(g,u['spec']['order'],K,state['tokens'],state['sessions'],state['fit_identity']) if valid else dict(status='INVALID_SPACE')
    valid=valid and cluster['status']=='VALID_DIRECTION_CLUSTER';binding=dict(plan=ref(plan),unit=unit,source_files=sources());common=[];centers=[];hashes=[]
    B=state['space']['basis'];center=state['space']['center']
    for k in range(K):
        rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64)
        c=ops.algebra(inner.coordinates(state['fail_view'][rr],center,B),inner.coordinates(state['success_view'][rr],center,B),g['coefficients'][rr],g['L'],state['D0'],g['orders']['P_to_C']['no_direction'][rr]) if len(rr) else None
        common.append(c);h,d=physical_hash(c,state);hashes.append(h);centers.append(d)
    old=meta(u['inner_bank']);old_labels=cal.fitted_fields(old['clustering'],[('labels',)])[0];old_q=cal.fitted_fields(meta(old['space'])['artifact'],[('Q_indices',)])[0]
    assert old_q.tolist()==p['inner_Q_indices'] and set(old_q)<=set(state['Q_indices'])
    lookup={int(i):n for n,i in enumerate(state['Q_indices'])};common_indices=np.array(old_q,np.int64);C=np.zeros((K,K),np.int64)
    if valid:
        nl=cluster['labels'][[lookup[int(i)] for i in common_indices]];np.add.at(C,(old_labels,nl),1);mapping=exact_matching(C,hashes)
        mapping.update(ARI=adjusted_rand(C),common_indices=common_indices,old_labels=old_labels,new_labels=nl,contingency=C,new_physical_centers=np.asarray(centers),new_physical_center_hashes=hashes)
    else:mapping=dict(old_to_new=list(range(K)),new_to_old=list(range(K)),contingency=C,status='INVALID_REFIT_CLUSTER_EXPLICIT_BASE_NO_MATCH_CLAIM',ARI=None)
    cr=save(path.parent/'CLUSTER',cluster,binding);mr=save(path.parent/'TRANSFER',mapping,binding);models=[];parent=None
    if u['dependencies']:
        parent=read(root/'BANKS'/u['dependencies'][0]/'RESULT.json');assert parent['plan']==ref(plan) and parent['space']==ref(root/'SPACES'/u['spec']['route']/'RESULT.json')
        if valid and parent['valid_cluster']:
            assert K==1 and parent['selected_slots'][0]['selection']['method']==u['slots'][0]['method']
    proof=meta(p['data_proof']);o=data.small_arrays(proof['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])}
    ordinary=inner.ordinary_view(o['Base_pooled_float64'],o['token_counts_int64'],state['view']) if valid else None
    for k in range(K):
        old_slot=mapping['new_to_old'][k];selection=u['slots'][old_slot];rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64)
        folder=path.parent/f'cluster{k:02d}';comm=save(folder/'COMMON',common[k],binding) if common[k] is not None else None
        support=ops.support(selection['method'],len(rr),len(p['high_indices']),len(p['normal_indices'])) if not selection['Base'] else dict(Base_fallback_required=True,status='FROZEN_INNER_SLOT_BASE')
        if not valid:support=dict(Base_fallback_required=True,status='INVALID_REFIT_SPACE_OR_CLUSTER')
        if support['Base_fallback_required']:entry=dict(status=support['status'],Base_fallback_required=True)
        elif parent is not None and parent['valid_cluster']:
            entry=parent['selected_slots'][0]['entry'];assert parent['selected_slots'][0]['Q_indices']==state['Q_indices'][rr].tolist()
        else:
            method=selection['method'];f=inner.coordinates(state['fail_view'][rr],center,B);s=inner.coordinates(state['success_view'][rr],center,B)
            if method.startswith('mlp'):
                high=inner.coordinates(ordinary[[where[i] for i in p['high_indices']]],center,B);ds=ops.paired_rows(f,s,high if method=='mlp_replay' else None)
                sd=inner.initialization(state,K,method);training=inner.mlp_run(ds,state['D0'],sd,folder/'MLP2000',dict(**binding,method=method,cluster=k))
                entry=dict(status='VALID',kind='mlp',state=training['endpoint'],training=ref(folder/'MLP2000/RESULT.json'),Base_fallback_required=False,
                    reused_numeric_engine=ref(inner.__file__),known_kernel_root=p['known_kernel_root'],initial_seed=int(sd))
            elif method in ('paired_RRR','direction_amplitude'):
                target,tp=ops.paired_targets(B,state['success_view'][rr]-state['fail_view'][rr],state['view'],g['fractions'][rr],state['actual_fulltoken_RMS'][rr],state['D0'])
                normal=[where[i] for i in p['normal_indices']];fit=fit_selected_paired(state['fail_view'][rr],state['raw_aux'][rr].astype(np.float64),ordinary[normal],o['raw_aux24_float32'][normal].astype(np.float64),target,state['D0'],method)
                ar=save(folder/'PAIRED_SELECTED',dict(model=fit,target=target,proof=tp),binding);entry=dict(status='VALID',kind='paired_single',state=ar,Base_fallback_required=False)
            else:
                high=inner.coordinates(ordinary[[where[i] for i in p['high_indices']]],center,B)
                fit=ops.fit_original(method,f,s,common[k],high);fit.pop('common',None);ar=save(folder/'MODEL',fit,binding);entry=dict(status='VALID',kind='analytic',state=ar,Base_fallback_required=False)
        model_uid=digest(dict(plan=ref(plan),new_space=ref(root/'SPACES'/u['spec']['route']/'RESULT.json'),clustering=cr,new_cluster=k,selection=selection,model=entry.get('state'),fit_members=state['Q_indices'][rr].tolist()))
        models.append(dict(new_cluster=k,old_slot=old_slot,selection=selection,Q_indices=state['Q_indices'][rr].tolist(),common=comm,entry=entry,physical_UID=model_uid))
    return result(path,dict(status=BANK_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),formal_selection=p['formal_selection'],spec=u['spec'],inner_bank=u['inner_bank'],
        space=ref(root/'SPACES'/u['spec']['route']/'RESULT.json'),clustering=cr,transfer=mr,selected_slots=models,valid_cluster=valid,
        K1_exact_reuse=None if parent is None or not valid or not parent['valid_cluster'] else ref(root/'BANKS'/u['dependencies'][0]/'RESULT.json'),
        old_model_parameters_or_responses_reused=False,held_GT_or_PDMS_read=False,all_selected_MLP_fixed2000_or_explicit_Base=True))

def paired_predict(entry,state,method):
    names=['input_mean','input_scale','target_mean','target_scale','weight','bias'];values=cal.fitted_fields(entry['state'],[('model','rank'),('model','D0'),*[('model','learned',n) for n in names]])
    rank,D0=values[:2];fit=dict(zip(names,values[2:]));B=state['space']['basis']
    def predict(v,aux,fraction):
        q=ops.predict_ridge(np.c_[v[None],aux[None]],fit)
        if method=='paired_RRR':return q[0,:rank]*(D0*np.sqrt(384.))
        return ops.paired.da_decode(q[:,:rank],q[:,-1],B,state['view'],np.array([fraction]),D0)[0][0]
    return predict

def pack_held(plan,release,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,root=context(plan,release,'pack-held',unit);folder=root/'HELD_REQUESTS'/unit;path=folder/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    bank=read(root/'BANKS'/unit/'RESULT.json');assert bank['plan']==ref(plan) and bank['source_files']==sources();sp=meta(bank['space']);decoder=data.small_arrays(sp['decoder'])
    state=dict(space=dict(basis=decoder['basis_float64'],center=decoder['center_float64']),D0=float(decoder['D0_float64']),view=sp['view'])
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);pos={int(i):n for n,i in enumerate(o['indices_int64'])};take=np.array([pos[i] for i in p['held_indices']]);roles=o['Base_pooled_float64'][take];counts=o['token_counts_int64'][take];aux=o['raw_aux24_float32'][take]
    v=inner.ordinary_view(roles,counts,state['view']);K=bank['spec']['K'];raw=np.zeros((1987,K,1,48),np.float64);actions=[];cache={}
    for m in sorted(bank['selected_slots'],key=lambda m:m['old_slot']):
        k=m['new_cluster'];entry=m['entry'];selection=m['selection'];valid=not entry['Base_fallback_required']
        if valid:
            common=None if m['common'] is None else inner.loaded(m['common'])
            predict=paired_predict(entry,state,selection['method']) if entry['kind']=='paired_single' else cal.predictor(entry,common,state,selection['method'],cache)
            for n in range(1987):raw[n,k,0]=predict(v[n],aux[n].astype(np.float64),float(counts[n,0]/counts[n].sum()))
        spec=dict(unit=unit,new_cluster=k,old_slot=m['old_slot'],method=selection['method'],alpha=selection['alpha'],cap_D0=selection['cap_D0'],physical_UID=m['physical_UID'],bank_SHA=ref(root/'BANKS'/unit/'RESULT.json')['sha256'])
        actions.append(dict(action_id=digest(spec),**spec,cluster=k,method_position=0,valid=valid,source_action_id=selection.get('source_action_id'),invalid_executes_literal_Base=not valid))
    rows=meta(p['ordinary_inputs'])['cases'];by={r['index']:r for r in rows};weights=np.array([p['held_group']['conditional_natural_mean_weight_per_sample'][by[i]['audit_category']] for i in p['held_indices']],np.float64);assert abs(weights.sum()-1)<1e-12
    ar=data.arrays(folder/'COMPACT.npz',dict(indices_int64=np.array(p['held_indices'],np.int64),raw_coefficients_float64=raw,ordinary_role_means_float64=roles,ordinary_view_float64=v,raw_aux24_float32=aux,token_counts_int64=counts,natural_mean_weights_float64=weights))
    assert np.isfinite(raw).all() and len(actions)==K
    return result(path,dict(status=REQUEST_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),bank=ref(root/'BANKS'/unit/'RESULT.json'),formal_selection=p['formal_selection'],
        compact=ar,decoder=sp['decoder'],view=state['view'],actions=actions,panel_indices=p['held_indices'],noise_indices=[0,1,2,3,4],
        input_scope='ordinary_Base_only_outer_session_held_prediction',logical_expert_outputs=1987*K*5,held_teacher_arrays_opened=0,
        raw_prediction_preserves_magnitude=True,only_downward_native_cap_required=True,physical_columns_not_same_number_inner_experts=True))

def materialize_case(request,index,positions=None):
    record=request if isinstance(request,dict) else ref(request);key=(record['path'],record['sha256'],digest(sources()))
    if key not in _CACHE:
        r=meta(record);assert r['passed'] and r['status']==REQUEST_STATUS and r['producer']==ref(__file__) and r['source_files']==sources()
        assert r['input_scope']=='ordinary_Base_only_outer_session_held_prediction' and r['held_teacher_arrays_opened']==0 and r['noise_indices']==[0,1,2,3,4]
        a=data.small_arrays(r['compact']);d=data.small_arrays(r['decoder']);assert a['indices_int64'].tolist()==r['panel_indices']
        _CACHE[key]=(r,a,d,{int(i):n for n,i in enumerate(a['indices_int64'])})
        if len(_CACHE)>2:_CACHE.popitem(last=False)
    _CACHE.move_to_end(key);r,a,d,where=_CACHE[key]
    return cal.case_arrays(a['raw_coefficients_float64'][where[index]],r['actions'],d,r['view'],positions)

def inventory(output):
    d=metadata();r=dict(status='CPU_METADATA_OUTER0_REFIT_AND_TRANSFER_NO_ACTUAL_FIT',ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),data_proof=ref(cal.DATA_PROOF),session_split=d['proof']['session_split'],pair_pack=ref(PAIR_PACK),
        train_N=len(d['train']),qualified_train_Q=len(d['Q']),normal_train_N=len(d['normal']),high_train_N=len(d['high']),held_N=len(d['held']),train_group=d['fold']['outer_train'],held_group=d['fold']['outer_held'],
        common_inner_Q=len(d['oldplan']['qualified_fit_indices']),no_held_teacher_payloads_read=True,formal_selection_read=False,new_fits_or_queue=0)
    return result(output,r)

def selftest(output):
    fixtures=[]
    for C,hashes in [(np.array([[3,3],[3,3]],np.int64),['z','a']),
                     (np.array([[100,99],[99,0]],np.int64),['a','z']),
                     (np.zeros((3,3),np.int64),['same','same','other'])]:
        answer=exact_matching(C,hashes);allp=list(itertools.permutations(range(len(C))));opt=max(sum(int(C[i,j]) for i,j in enumerate(p)) for p in allp)
        best=min((p for p in allp if sum(int(C[i,j]) for i,j in enumerate(p))==opt),key=lambda p:tuple((hashes[j],j) for j in p))
        assert tuple(answer['old_to_new'])==best and answer['maximum_integer_overlap']==opt;fixtures.append(answer)
    assert adjusted_rand(np.array([[3,0],[0,4]],np.int64))==1 and adjusted_rand(np.array([[0,3],[4,0]],np.int64))==1
    B=np.eye(384,48);state=dict(space=dict(basis=B),view=dict(view='valid_shared384'),D0=.2)
    h,v=physical_hash(dict(centroid_noop=False,centroid_coefficients=np.ones(48)*.2),state);assert np.isfinite(v).all() and abs(np.linalg.norm(v)-1)<1e-12
    assert physical_hash(None,state)[1].tobytes()==np.zeros((2,384)).tobytes()
    assert np.array_equal(exact_matching(np.array([[7]],np.int64),[h])['old_to_new'],[0])
    rng=np.random.default_rng(2026102539);v=rng.normal(size=(9,384));aux=rng.normal(size=(9,24));normal=rng.normal(size=(5,384));na=rng.normal(size=(5,24))
    target=dict(target_coefficients=np.zeros((9,4)),RRR_targets=rng.normal(size=(9,4)),DA_direction_targets=rng.normal(size=(9,4)),DA_logamplitude_targets=rng.normal(size=9))
    full=ops.fit_paired(v,aux,normal,na,target,.2);phi=np.c_[v[:2],aux[:2]];expected=ops.predict_ridge(phi,full['learned']);errors={};decode_errors={}
    for method in ('paired_RRR','direction_amplitude'):
        fitted=fit_selected_paired(v,aux,normal,na,target,.2,method);pred=ops.predict_ridge(phi,fitted['learned']);truth=expected[:,:4] if method=='paired_RRR' else expected[:,4:]
        error=float(np.max(np.abs(pred-truth)));assert np.allclose(pred,truth,atol=1e-12,rtol=1e-11);errors[method]=error
        mr=save(Path(output).parent/(Path(output).stem+'_NUMERIC')/method,dict(model=fitted),dict(synthetic_only=True,source_files=sources()))
        s=dict(space=dict(basis=np.eye(384,4),center=np.zeros(384)),D0=.2,view=dict(view='valid_shared384'))
        fn=paired_predict(dict(state=mr),s,method)
        for j in range(2):
            single=fn(v[j],aux[j],.4);original=ops.predict_paired(v[j],aux[j],full,method,s['space']['basis'],s['view'],np.array([.4]))[0]
            assert np.allclose(single,original,atol=1e-12,rtol=1e-11)
            decode_errors[method]=max(decode_errors.get(method,0.),float(np.max(np.abs(single-original))))
    raw=np.zeros((2,1,48),np.float64);raw[1,0,:]=np.arange(48)/100
    actions=[dict(action_id='0'*64,cluster=0,method_position=0,valid=False,alpha=0,cap_D0=0),dict(action_id='1'*64,cluster=1,method_position=0,valid=True,alpha=.5,cap_D0=2)]
    decoder=dict(basis_float64=B,D0_float64=np.array(.2));case=cal.case_arrays(raw,actions,decoder,dict(view='valid_shared384'))
    assert case['requested_shifts_float64'][0].tobytes()==np.zeros((2,384)).tobytes() and case['cap_RMS_float64'].tolist()==[0.,.4]
    expected_shift=geo.decode_roles(raw[1,0],B,dict(view='valid_shared384'))*.5
    assert case['requested_shifts_float64'][1].tobytes()==expected_shift.tobytes() and case['requested_shifts_float32'].tobytes()==case['requested_shifts_float64'].astype(np.float32).tobytes()
    binding=dict(synthetic_only=True,source_files=sources());ar=save(Path(output).parent/(Path(output).stem+'_NUMERIC')/'SUMMARY',dict(matching=fixtures,physical_direction=v[:2],selected_paired_ridge_errors=errors),binding)
    assert inner.loaded(ar)['selected_paired_ridge_errors']==errors
    r=dict(status='PASS_CPU_OUTER0_INTEGER_MATCHING_AND_PHYSICAL_HASH_NO_REAL_REFIT',passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),fixtures=fixtures,
        exhaustive_primary_and_lexicographic_optima_match=True,greedy_wrong_assignment_fixture_rejected=True,duplicate_hash_stable_integer_tie=True,
        permutation_invariant_ARI=True,signed_decoded_center_D0_removed=True,zero_physical_hash_verified=True,selected_only_paired_RHS_vs_frozen_combined_prediction_error=errors,
        selected_only_saved_model_decode_vs_frozen_combined_error=decode_errors,
        Base_slot_zero_alpha_downcap_direct_FP32_ABI=True,synthetic_serialization_exact=True,numeric_fixture=ar,new_real_space_cluster_model_fit_or_held_teacher_reads=0)
    return result(output,r)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inventory','selftest','prepare','data','space','refit','pack-held']);p.add_argument('--output',type=Path);p.add_argument('--selection',type=Path,default=FREEZE)
    p.add_argument('--selector-source',type=Path,default=SELECTOR);p.add_argument('--plan',type=Path);p.add_argument('--release',type=Path);p.add_argument('--route',choices=['R','D']);p.add_argument('--unit');p.add_argument('--threads',type=int,default=1);a=p.parse_args();assert 1<=a.threads<=8
    try:
        if a.action=='inventory':r=inventory(a.output or ROOT/'INVENTORY.json')
        elif a.action=='selftest':r=selftest(a.output or ROOT/'CPU_SELFTEST.json')
        elif a.action=='prepare':r=prepare(a.selection,a.output or ROOT/'PLAN.json',a.selector_source)
        elif a.action=='data':r=train_data(a.plan.resolve(),a.release.resolve())
        elif a.action=='space':r=space(a.plan.resolve(),a.release.resolve(),a.route)
        elif a.action=='refit':r=refit(a.plan.resolve(),a.release.resolve(),a.unit,a.threads)
        else:r=pack_held(a.plan.resolve(),a.release.resolve(),a.unit,a.threads)
    except BaseException:
        atomic(ROOT/'FAILURES'/(str(time.time_ns())+'.json'),dict(time=stamp(),action=a.action,unit=a.unit,traceback=traceback.format_exc()),exclusive=True);raise
    print(json.dumps(r),flush=True)
