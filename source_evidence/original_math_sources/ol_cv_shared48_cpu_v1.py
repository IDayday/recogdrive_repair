"""Strict nested shared384/r48 inner-fit factory for outer folds1..4.

CPU only. Fresh per-fold Q/space/cluster/operator fits require ROOT release.
Only metadata prepare/selftest may run without release. No native selection.
"""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,hashlib,json,time,traceback
from collections import Counter,OrderedDict
import numpy as np
from b5_common import read,atomic,stamp,sha
import ol_l1_pilot_fitter_v1 as inner
import ol_l2_cal_pack_v1 as adapter
data,ops,geo=inner.data,inner.ops,inner.geo
ref,meta,checked=data.ref,data.meta,data.checked
ROOT=OL/'preparation/cv_shared48_cpu_v1'
PROOF=OL/'preparation/l1_complete_small_assets_v1/ROOT_ACCEPTANCE.json'
PAIR_PACK=data.ROOT/'PACKS/PAIRS1850_V1/RESULT.json'
KNOWN=P2/'provenance/OFFLINE_LIBRARY_L1_MLP_KNOWN2000_ACTUAL_ROOT_ACCEPTANCE_V1.json'
PLAN_STATUS='READY_STRICT_NESTED_SHARED48_FOLDS1_TO4_CPU_FACTORY_NOT_EXECUTION_V1'
RELEASE_STATUS='PASS_ROOT_OFFLINE_LIBRARY_CV_SHARED48_CPU_EXECUTION_RELEASE_V1'
BANK_STATUS='COMPLETE_CV_SHARED48_INNER_FIT_ALL13_FIXED2000_REQUIRES_ROOT_V1'
REQUEST_STATUS='COMPLETE_CV_SHARED48_ORDINARY_REQUESTS_REQUIRES_NATIVE_VALIDATION_V1'
COARSE_STATUS='FROZEN_ROOT_CV_SHARED48_COARSE_PRUNING_FOR_FULL_INNERCAL_V1'
_CACHE=OrderedDict()


def sources():
    s=inner.sources()
    for p in (__file__,adapter.__file__,OL/'code/ol_l1_pilot_metadata_v1.py'):s[str(Path(p).resolve())]=sha(p)
    return s
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(folder,payload,binding):return inner.saved(folder,payload,binding)
def result(path,value):atomic(path,value,exclusive=True);return ref(path)
def hash_panel(fold,category,session,token=None):
    fields=['2026102531','L1_panel',str(fold),category,session]
    if token is not None:fields.append(token)
    return hashlib.sha256('|'.join(fields).encode()).hexdigest()


def pilot_panel(fold,group,rows):
    panel=[]
    for category,quota in [('failure',16),('mid',4),('high',12)]:
        eligible=[rows[i] for i in group['selected_indices'] if rows[i]['audit_category']==category];assert len(eligible)>=quota
        sessions=sorted({r['session'] for r in eligible},key=lambda s:(hash_panel(fold,category,s),s))
        buckets={s:sorted([r for r in eligible if r['session']==s],key=lambda r:(hash_panel(fold,category,s,r['token']),r['token'])) for s in sessions}
        chosen=[];position=0
        while len(chosen)<quota:
            for s in sessions:
                if len(buckets[s])>position:chosen.append(buckets[s][position])
                if len(chosen)==quota:break
            position+=1
        for r in chosen:panel.append(dict(index=r['index'],token=r['token'],session=r['session'],category=category,
            natural_mean_weight=group['full_population_counts'][category]/group['full_population_scenes']/quota))
    assert len(panel)==len({r['index'] for r in panel})==32 and abs(sum(r['natural_mean_weight'] for r in panel)-1)<1e-12
    return panel


def metadata():
    proof=read(PROOF);assert proof['passed'] and proof['status']==inner.DATA_STATUS
    inputs=meta(proof['inputs']);split=meta(proof['session_split']);pairs=read(PAIR_PACK)
    assert pairs['passed'] and pairs['producer']==ref(data.__file__) and ref(PAIR_PACK) in proof['source_results'] and pairs['full1850']==proof['full1850']
    assert len(pairs['indices'])==len(pairs['cases'])==1850 and len(set(pairs['canonical_qualified_indices']))==1807
    rows={r['index']:r for r in inputs['cases']};qualified=set(pairs['canonical_qualified_indices']);where=dict(zip(pairs['indices'],pairs['cases']))
    folds=[]
    for f in split['folds']:
        if f['fold'] not in (1,2,3,4):continue
        fit,cal,held=f['inner_fit'],f['inner_calibration'],f['outer_held']
        for group in (fit,cal,held):
            assert len(group['selected_indices'])==len(set(group['selected_indices'])) and Counter(rows[i]['audit_category'] for i in group['selected_indices'])==group['selected_counts']
            assert all(rows[i]['session'] in group['sessions'] for i in group['selected_indices'])
            for c,n in group['selected_counts'].items():assert np.isclose(group['conditional_natural_mean_weight_per_sample'][c],group['full_population_counts'][c]/group['full_population_scenes']/n,rtol=1e-14,atol=0)
        assert set(fit['sessions']).isdisjoint(cal['sessions']) and set(fit['sessions']+cal['sessions']).isdisjoint(held['sessions'])
        assert set(fit['selected_indices'])|set(cal['selected_indices'])==set(f['outer_train']['selected_indices'])
        ids=fit['selected_indices'];Q=[i for i in ids if i in qualified];normal=[i for i in ids if rows[i]['audit_category'] in ('mid','high')];high=[i for i in normal if rows[i]['audit_category']=='high']
        units=[dict(id=f"fold{f['fold']}__{route}__{order}__K{K}",fold=f['fold'],route=route,order=order,K=K,view='valid_shared384',rank=48,
            dependencies=[f"fold{f['fold']}__{route}__P_to_C__K1"] if K==1 and order=='C_to_P' else []) for route in ('R','D') for order in ('P_to_C','C_to_P') for K in range(1,9)]
        panel=pilot_panel(f['fold'],cal,rows)
        folds.append(dict(fold=f['fold'],fit_group=fit,calibration_group=cal,outer_train_group=f['outer_train'],held_group=held,fit_indices=ids,Q_indices=Q,
            normal_indices=normal,high_indices=high,train_pair_sources=[where[i] for i in Q],panel=panel,panel_indices=[r['index'] for r in panel],units=units,
            pilot_logical_expert_outputs=838656,output_root=str(ROOT/'ACTUAL'/f"fold{f['fold']}")))
    assert [f['fold'] for f in folds]==[1,2,3,4]
    return proof,folds,rows


def prepare(output):
    proof,folds,rows=metadata();known=read(KNOWN)
    assert known['passed'] and known['status']==inner.KNOWN_ROOT_STATUS and known['source_files']==inner.sources() and set(known['controls'])=={'mlp','mlp_replay'}
    for rr in known['controls'].values():
        r=meta(rr);assert r['passed'] and r['endpoint']==2000 and r['all_roles_absorption_ge90'] and r['replay1999_to2000_exact']
    doses=[dict(alpha=a,cap_D0=2.) for a in (.25,.5,.75,1.,1.5,2.)]+[dict(alpha=.5,cap_D0=1.)]
    value=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),data_proof=ref(PROOF),ordinary_inputs=proof['inputs'],session_split=proof['session_split'],pair_pack=ref(PAIR_PACK),
        known_kernel_root=ref(KNOWN),folds=folds,menu=list(ops.MENU),doses=doses,pilot_noises=[0,1],cal_noises=[0,1,2],
        units_total=128,pilot_logical_expert_outputs=4*838656,all_K1_to8_before_any_per_fold_coarse_pruning=True,fold0_retained_units_not_reused=True,
        panel_hash_rule='2026102531|L1_panel|fold|category|session[|token]; same prior fold0 session/token round-robin',
        per_fold_no_space_scaler_D0_cluster_operator_fit_from_innercal_or_outerheld=True,no_native_selection_or_queue=True,actual_new_fits=0)
    return result(output,value)


def context(plan,release,fold,action,unit=None,selection=None):
    p=read(plan);r=read(release);assert p['status']==PLAN_STATUS and p['producer']==ref(__file__) and p['source_files']==sources()
    assert r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan) and r['source_files']==sources() and fold in r['allowed_folds'] and action in r['allowed_actions']
    f=next(f for f in p['folds'] if f['fold']==fold)
    if unit is not None:assert unit in r['allowed_units'] and unit in {u['id'] for u in f['units']}
    if selection is not None:assert r['coarse_selection']==ref(selection)
    return p,f,Path(f['output_root'])


def data_stage(plan,release,fold):
    p,f,root=context(plan,release,fold,'data');path=root/'DATA/RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};parts=[]
    for i,rr in zip(f['Q_indices'],f['train_pair_sources']):
        r=meta(rr);assert r['index']==i and r['canonical_qualified_pair'];a=data.small_arrays(r['arrays']);assert bool(a['canonical_qualified_pair_bool'])
        for k in ('Base_pooled_float64','token_counts_int64','raw_aux24_float32'):assert a[k].tobytes()==o[k][where[i]].tobytes()
        assert float(a['canonical_Base_S0_score_float64'])<.7 and float(a['canonical_teacher_S0_score_float64'])>.9;parts.append(a)
    state=dict(Q_indices=np.array(f['Q_indices'],np.int64),Base=np.stack([a['Base_pooled_float64'] for a in parts]),teacher=np.stack([a['teacher_pooled_float64'] for a in parts]),
        counts=np.stack([a['token_counts_int64'] for a in parts]),aux=np.stack([a['raw_aux24_float32'] for a in parts]),rms=np.array([float(a['actual_fulltoken_teacher_RMS_float64']) for a in parts],np.float64))
    assert np.isfinite(state['teacher']).all() and np.all(state['rms']>0)
    ar=save(path.parent/'STATE',state,dict(plan=ref(plan),fold=fold,source_files=sources(),scope='this_fold_inner_fit_Q_only'))
    return result(path,dict(status='COMPLETE_CV_SHARED48_INNER_FIT_Q_DATA_NOT_FIT_V1',passed=True,plan=ref(plan),fold=fold,source_files=sources(),artifact=ar,
        actual_pair_rows=len(parts),innercal_or_outerheld_teacher_payloads_opened=0,aggregate_PAIRS1850_arrays_opened=0))


def space(plan,release,fold,route):
    p,f,root=context(plan,release,fold,'space');assert route in ('R','D');path=root/'SPACES'/route/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    dr=read(root/'DATA/RESULT.json');assert dr['plan']==ref(plan) and dr['fold']==fold and dr['source_files']==sources();d=inner.loaded(dr['artifact']);rows={r['index']:r for r in meta(p['ordinary_inputs'])['cases']}
    vf,vs,view=geo.old_geometry.view_vectors(d['Base'],d['teacher'],d['counts'],'valid_shared384',np.arange(len(d['Q_indices'])))
    fitted=geo.fit_space(vf,vs,d['rms'],route,'valid_shared384',48);D0=float(np.median(d['rms']))
    identity=digest(dict(data_proof=p['data_proof'],Q=f['Q_indices'],normal=f['normal_indices'],high=f['high_indices'],fit_sessions=f['fit_group']['sessions'],outer_fold=fold,view='valid_shared384',scope='inner_fit'))
    state=dict(space=fitted,view=view,D0=D0,Q_indices=d['Q_indices'],normal_indices=np.array(f['normal_indices'],np.int64),high_indices=np.array(f['high_indices'],np.int64),
        fail_view=vf,success_view=vs,actual_fulltoken_RMS=d['rms'],raw_aux=d['aux'],fit_identity=identity,recipe=dict(route=route,view='valid_shared384',rank=48),
        tokens=[rows[i]['token'] for i in f['Q_indices']],sessions=[rows[i]['session'] for i in f['Q_indices']],fold=fold)
    if fitted['status']=='VALID_SPACE':state['geometry']=geo.geometry(fitted['basis'],view,d['counts'][:,0]/d['counts'].sum(1),vs-vf,d['rms'])
    ar=save(path.parent/'STATE',state,dict(plan=ref(plan),fold=fold,route=route,source_files=sources()))
    dec=data.arrays(path.parent/'DECODER.npz',dict(basis_float64=fitted['basis'],center_float64=fitted['center'],D0_float64=np.array(D0)))
    return result(path,dict(status='COMPLETE_CV_SHARED48_FRESH_INNER_FOLD_SPACE_REQUIRES_ROOT_V1',passed=True,producer=ref(__file__),source_files=sources(),plan=ref(plan),fold=fold,route=route,
        artifact=ar,decoder=dec,view=view,space_status=fitted['status'],fit_identity=identity,D0=D0,Q=len(vf),no_fold0_or_other_fold_physical_basis_reused=True))


def fit(plan,release,fold,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,f,root=context(plan,release,fold,'fit',unit);spec=next(s for s in f['units'] if s['id']==unit);K=spec['K'];path=root/'BANKS'/unit/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    sr=read(root/'SPACES'/spec['route']/'RESULT.json');assert sr['plan']==ref(plan) and sr['fold']==fold and sr['source_files']==sources();state=inner.loaded(sr['artifact'])
    valid=state['space']['status']=='VALID_SPACE';cluster=geo.cluster(state['geometry'],spec['order'],K,state['tokens'],state['sessions'],state['fit_identity']) if valid else dict(status='INVALID_SPACE')
    valid=valid and cluster['status']=='VALID_DIRECTION_CLUSTER';binding=dict(plan=ref(plan),fold=fold,unit=unit,source_files=sources());models=[];reuse=None
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};v=inner.ordinary_view(o['Base_pooled_float64'],o['token_counts_int64'],state['view']) if valid else None
    B,center=state['space']['basis'],state['space']['center'];z=inner.coordinates(v,center,B) if valid else None
    if valid and spec['dependencies']:
        pr=read(root/'BANKS'/spec['dependencies'][0]/'RESULT.json');assert pr['plan']==ref(plan) and pr['fold']==fold and pr['space']==ref(root/'SPACES'/spec['route']/'RESULT.json')
        if pr['valid_cluster']:
            assert K==1 and (cluster['labels']==0).all();models=pr['clusters'];reuse=ref(root/'BANKS'/spec['dependencies'][0]/'RESULT.json')
    if reuse is None:
        for k in range(K):
            rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64);folder=path.parent/f'cluster{k:02d}';methods={};n=len(rr);common=None;cr=None
            if n:
                failed=inner.coordinates(state['fail_view'][rr],center,B);success=inner.coordinates(state['success_view'][rr],center,B)
                assert failed.tobytes()==z[[where[int(state['Q_indices'][j])] for j in rr]].tobytes()
                common=ops.algebra(failed,success,state['geometry']['coefficients'][rr],state['geometry']['L'],state['D0'],state['geometry']['orders']['P_to_C']['no_direction'][rr])
                cr=save(folder/'COMMON',common,binding);high=z[[where[i] for i in f['high_indices']]]
            paired_ref=None
            for method in ops.MENU:
                support=ops.support(method,n,len(f['high_indices']),len(f['normal_indices']))
                if not valid:support=dict(status='INVALID_SPACE_OR_CLUSTER',Base_fallback_required=True)
                if support['Base_fallback_required']:methods[method]=dict(status=support['status'],Base_fallback_required=True);continue
                out=folder/method/'RESULT.json'
                if out.exists():entry=read(out);assert entry['binding']==binding;methods[method]=entry;continue
                mb=dict(**binding,method=method,cluster=k)
                if method.startswith('mlp'):
                    ds=ops.paired_rows(failed,success,high if method=='mlp_replay' else None);sd=inner.initialization(state,K,method)
                    run=inner.mlp_run(ds,state['D0'],sd,folder/method/'TRAINING',mb)
                    entry=dict(status='VALID',kind='mlp',state=run['endpoint'],training=ref(folder/method/'TRAINING/RESULT.json'),initial_seed=int(sd),Base_fallback_required=False,
                        reused_numeric_engine=ref(inner.__file__),known_kernel_root=p['known_kernel_root'])
                elif method in ('paired_RRR','direction_amplitude'):
                    if paired_ref is None:
                        target,proof=ops.paired_targets(B,state['success_view'][rr]-state['fail_view'][rr],state['view'],state['geometry']['fractions'][rr],state['actual_fulltoken_RMS'][rr],state['D0'])
                        normal=[where[i] for i in f['normal_indices']];model=ops.fit_paired(state['fail_view'][rr],state['raw_aux'][rr].astype(np.float64),v[normal],o['raw_aux24_float32'][normal].astype(np.float64),target,state['D0'])
                        paired_ref=save(folder/'PAIRED_COMBINED',dict(model=model,target=target,proof=proof),binding)
                    entry=dict(status='VALID',kind='paired',state=paired_ref,Base_fallback_required=False)
                else:
                    model=ops.fit_original(method,failed,success,common,high);model.pop('common',None);ar=save(folder/method/'MODEL',model,mb)
                    entry=dict(status='VALID',kind='analytic',state=ar,Base_fallback_required=False)
                entry['binding']=binding;atomic(out,entry,exclusive=True);methods[method]=entry
            assert set(methods)==set(ops.MENU)
            models.append(dict(cluster=k,Q_indices=state['Q_indices'][rr].tolist(),common=cr,methods=methods,NO_DIRECTION_rows_retained=True,
                negative_controls_not_counted_as_repairs={m:dict(control=True,noncompetitive=True,coefficient_L2_only_not_actual_matched=True) for m in ops.CONTROLS}))
    ar=save(path.parent/'CLUSTER',cluster,binding)
    return result(path,dict(status=BANK_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),actual_release=ref(release),fold=fold,spec=spec,
        space=ref(root/'SPACES'/spec['route']/'RESULT.json'),clustering=ar,clusters=models,valid_cluster=valid,K1_exact_reuse=reuse,all13_menu_slots_present=True,
        weak_MLP_fit_not_PDMS_veto=True,no_innercal_or_held_teacher_fit=True,MLP_endpoint=2000,replay1999_to2000_from_frozen_engine=True))


def coarse_selection(path,selector_source,plan,f):
    s=read(path);assert s['passed'] and s['status']==COARSE_STATUS and s['fold']==f['fold'] and s['plan']==ref(plan) and s['producer']==ref(selector_source)
    reg=meta(s['source_registration']);assert reg['passed'] and reg['source']==s['producer'] and reg['native_scores_read_at_registration']==0
    accepted=meta(s['source_native_acceptance']);q=meta(accepted['queue']);assert accepted['passed'] and q['scope']=='pilot32' and q['noise_indices']==[0,1] and set(q['indices'])==set(f['panel_indices'])
    assert accepted['source_files']==q['source_files']
    assert sum(len(j['positions'])*2 for j in q['jobs'])==838656
    specs={u['id']:u for u in f['units']};rows={r['unit']:r for r in s['results']};assert len(s['results'])==len(rows)==32 and set(rows)==set(specs)
    assert {j['unit'] for j in q['jobs']}==set(specs)
    for job in q['jobs']:assert job['index'] in f['panel_indices'] and job['request']==rows[job['unit']]['request']
    keep=s['retained_units'];assert len(keep)==len(set(keep)) and set(keep)<=set(specs)
    for route in ('R','D'):
        Ks={specs[u]['K'] for u in keep if specs[u]['route']==route};assert 1 in Ks and len(Ks-{1})==2
        assert {u for u in keep if specs[u]['route']==route}=={u for u,v in specs.items() if v['route']==route and v['K'] in Ks}
    for uid,row in rows.items():
        assert row['spec']==specs[uid];rr=meta(row['request']);assert rr['producer']==ref(__file__) and rr['source_files']==sources() and rr['plan']==ref(plan) and rr['fold']==f['fold'] and rr['scope']=='pilot32'
        slots=row['retained_per_slot'];assert len(slots)==specs[uid]['K'] and {t['cluster'] for t in slots}==set(range(specs[uid]['K']))
        for slot in slots:
            menu=slot['retained_methods'];assert len(menu)==len(set(menu))==3 and menu[0]=='centroid_D0' and set(menu)<=set(ops.MENU) and slot['all_seven_doses_retained']
    return s,rows


def pack(plan,release,fold,unit,scope,selection=None,selector_source=None,threads=1):
    import torch
    torch.set_num_threads(threads);assert scope in ('pilot32','innercal');p,f,root=context(plan,release,fold,'pack-pilot' if scope=='pilot32' else 'pack-cal',unit,selection if scope=='innercal' else None)
    if scope=='innercal':
        assert selection is not None and selector_source is not None;s,selected=coarse_selection(selection,selector_source,plan,f);assert unit in s['retained_units']
        menus=[v['retained_methods'] for v in sorted(selected[unit]['retained_per_slot'],key=lambda v:v['cluster'])];panel=f['calibration_group']['selected_indices'];noise=[0,1,2]
    else:menus=None;panel=f['panel_indices'];noise=[0,1]
    folder=root/('PILOT_REQUESTS' if scope=='pilot32' else 'CAL_REQUESTS')/unit;path=folder/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources() and r['scope']==scope;return ref(path)
    bank=read(root/'BANKS'/unit/'RESULT.json');assert bank['status']==BANK_STATUS and bank['plan']==ref(plan) and bank['fold']==fold and bank['source_files']==sources()
    sp=meta(bank['space']);dec=data.small_arrays(sp['decoder']);state=dict(space=dict(basis=dec['basis_float64'],center=dec['center_float64']),D0=float(dec['D0_float64']),view=sp['view'])
    K=bank['spec']['K'];menus=[list(ops.MENU) for _ in range(K)] if menus is None else menus;M=len(menus[0]);assert all(len(m)==M for m in menus)
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};take=np.array([where[i] for i in panel]);roles=o['Base_pooled_float64'][take];counts=o['token_counts_int64'][take];aux=o['raw_aux24_float32'][take]
    v=inner.ordinary_view(roles,counts,state['view']);raw=np.zeros((len(panel),K,M,48),np.float64);valid=np.zeros((K,M),bool);cache={}
    if bank['valid_cluster']:
        for cluster in bank['clusters']:
            k=cluster['cluster'];common=None
            for j,method in enumerate(menus[k]):
                entry=cluster['methods'][method];valid[k,j]=not entry['Base_fallback_required']
                if not valid[k,j]:continue
                if common is None and entry['kind']=='analytic':common=inner.loaded(cluster['common'])
                predict=adapter.predictor(entry,common,state,method,cache)
                for n in range(len(panel)):raw[n,k,j]=predict(v[n],aux[n].astype(np.float64),float(counts[n,0]/counts[n].sum()))
    assert np.isfinite(raw).all();actions=[]
    for k in range(K):
        for j,method in enumerate(menus[k]):
            for dose in p['doses']:
                spec=dict(fold=fold,unit=unit,cluster=k,method=method,alpha=dose['alpha'],cap_D0=dose['cap_D0'],plan_SHA=ref(plan)['sha256'],bank_SHA=ref(root/'BANKS'/unit/'RESULT.json')['sha256'],scope=scope)
                actions.append(dict(action_id=digest(spec),**spec,method_position=j,valid=bool(valid[k,j]),invalid_executes_literal_Base=not bool(valid[k,j]),control=False))
    rows={r['index']:r for r in meta(p['ordinary_inputs'])['cases']}
    if scope=='pilot32':weights=np.array([r['natural_mean_weight'] for r in f['panel']],np.float64)
    else:weights=np.array([f['calibration_group']['conditional_natural_mean_weight_per_sample'][rows[i]['audit_category']] for i in panel],np.float64)
    assert abs(weights.sum()-1)<1e-12
    ar=data.arrays(folder/'COMPACT.npz',dict(indices_int64=np.array(panel,np.int64),raw_coefficients_float64=raw,valid_bool=valid,
        ordinary_role_means_float64=roles,ordinary_view_float64=v,raw_aux24_float32=aux,token_counts_int64=counts,natural_mean_weights_float64=weights))
    return result(path,dict(status=REQUEST_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),actual_release=ref(release),fold=fold,scope=scope,
        bank=ref(root/'BANKS'/unit/'RESULT.json'),space=bank['space'],compact=ar,decoder=sp['decoder'],view=state['view'],actions=actions,methods_per_slot=menus,panel_indices=panel,noise_indices=noise,
        source_coarse_selection=None if scope=='pilot32' else ref(selection),input_scope='ordinary_Base_only_this_fold_session_held_prediction',
        logical_expert_outputs=len(panel)*len(actions)*len(noise),population_counts=f['calibration_group']['full_population_counts'],population_total=f['calibration_group']['full_population_scenes'],
        raw_prediction_preserves_magnitude=True,only_downward_native_cap_required=True,innercal_or_outerheld_teacher_arrays_opened=0,online_response_or_quality_labels_used=False))


def materialize_case(request,index,positions=None):
    rr=request if isinstance(request,dict) else ref(request);key=(rr['path'],rr['sha256'],digest(sources()))
    if key not in _CACHE:
        r=meta(rr);assert r['passed'] and r['status']==REQUEST_STATUS and r['producer']==ref(__file__) and r['source_files']==sources()
        assert r['scope'] in ('pilot32','innercal') and r['input_scope']=='ordinary_Base_only_this_fold_session_held_prediction'
        a=data.small_arrays(r['compact']);d=data.small_arrays(r['decoder']);assert a['indices_int64'].tolist()==r['panel_indices'];_CACHE[key]=(r,a,d,{int(i):n for n,i in enumerate(a['indices_int64'])})
        if len(_CACHE)>2:_CACHE.popitem(last=False)
    _CACHE.move_to_end(key);r,a,d,where=_CACHE[key]
    return adapter.case_arrays(a['raw_coefficients_float64'][where[index]],r['actions'],d,r['view'],positions)


def selftest(output):
    proof,folds,rows=metadata();split=meta(proof['session_split']);zero=next(f for f in split['folds'] if f['fold']==0);same=pilot_panel(0,zero['inner_calibration'],rows)
    previous=read(inner.PILOT);assert [r['index'] for r in same]==previous['panel_indices'] and np.array([r['natural_mean_weight'] for r in same],np.float64).tobytes()==np.array([r['natural_mean_weight'] for r in previous['panel']],np.float64).tobytes()
    for f in folds:
        assert Counter(r['category'] for r in f['panel'])==dict(failure=16,mid=4,high=12) and set(f['panel_indices'])<=set(f['calibration_group']['selected_indices'])
        assert sum(u['K'] for u in f['units'])==144 and len(f['units'])==32 and {u['K'] for u in f['units']}==set(range(1,9))
        assert set(f['Q_indices'])<=set(f['fit_indices']) and set(f['Q_indices']).isdisjoint(f['calibration_group']['selected_indices']+f['held_group']['selected_indices'])
    B=np.eye(384,48);raw=np.zeros((2,1,48));raw[1,0]=np.arange(48)/10;acts=[dict(action_id='a'*64,cluster=0,method_position=0,valid=False,alpha=.25,cap_D0=2),dict(action_id='b'*64,cluster=1,method_position=0,valid=True,alpha=.5,cap_D0=2)]
    decoder=dict(basis_float64=B,D0_float64=np.array(.2));x=adapter.case_arrays(raw,acts,decoder,dict(view='valid_shared384'));assert not x['requested_shifts_float32'][0].any() and x['cap_RMS_float64'].tolist()==[.4,.4]
    assert x['requested_shifts_float64'][1].tobytes()==(geo.decode_roles(raw[1,0],B,dict(view='valid_shared384'))*.5).tobytes() and x['requested_shifts_float32'].tobytes()==x['requested_shifts_float64'].astype(np.float32).tobytes()
    r=dict(status='PASS_CPU_CV_SHARED48_METADATA_MENU_FOLD0_PARITY_AND_REQUEST_ABI_NO_REAL_FIT_V1',passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),
        fold0_panel_and_FP64_weight_byteexact_to_frozen=True,all_four_fold_sessions_and_Q_role_metadata_checked=True,complete_K1_to8_13by7_menu=True,
        invalid_literal_Base_and_raw_alpha_direct_FP32_cap_ABI=True,real_pair_payloads_opened=0,new_space_cluster_model_fits_native_or_queue=0,
        folds=[dict(fold=f['fold'],fit=len(f['fit_indices']),Q=len(f['Q_indices']),normal=len(f['normal_indices']),high=len(f['high_indices']),cal=len(f['calibration_group']['selected_indices']),held=len(f['held_group']['selected_indices']),pilot_indices=f['panel_indices']) for f in folds])
    return result(output,r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','selftest','data','space','fit','pack-pilot','pack-cal']);p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path,default=ROOT/'PLAN_FINAL.json');p.add_argument('--release',type=Path)
    p.add_argument('--fold',type=int,choices=[1,2,3,4]);p.add_argument('--route',choices=['R','D']);p.add_argument('--unit');p.add_argument('--selection',type=Path);p.add_argument('--selector-source',type=Path);p.add_argument('--threads',type=int,default=1);a=p.parse_args();assert 1<=a.threads<=8
    try:
        if a.action=='prepare':r=prepare(a.output or a.plan)
        elif a.action=='selftest':r=selftest(a.output or ROOT/'CPU_SELFTEST.json')
        elif a.action=='data':r=data_stage(a.plan.resolve(),a.release.resolve(),a.fold)
        elif a.action=='space':r=space(a.plan.resolve(),a.release.resolve(),a.fold,a.route)
        elif a.action=='fit':r=fit(a.plan.resolve(),a.release.resolve(),a.fold,a.unit,a.threads)
        else:r=pack(a.plan.resolve(),a.release.resolve(),a.fold,a.unit,'pilot32' if a.action=='pack-pilot' else 'innercal',a.selection,a.selector_source,a.threads)
    except BaseException:
        atomic(ROOT/'FAILURES'/(str(time.time_ns())+'.json'),dict(time=stamp(),action=a.action,fold=a.fold,unit=a.unit,traceback=traceback.format_exc()),exclusive=True);raise
    print(json.dumps(r),flush=True)
