"""Frozen full inner-calibration selection; all twelve libraries enter outer refit.

One method/dose per physical inner-fit cluster, never per-scene configuration
search. S012 mean precedes Base/K Oracle. This is development selection, not
outer-held confirmation or final top-six selection.
"""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from b5_common import read,atomic,stamp
import ol_l2_native_v1 as native
ref,meta,checked=native.ref,native.meta,native.data.checked
ROOT=OL/'analysis/l2_formal_selection_v2'
PLAN=OL/'preparation/l2_cal_pack_v1/PLAN.json'
STATUS='FROZEN_L2_ALL12_LIBRARIES_AFTER_FULL_INNERCAL1681_S012_V1'


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def objective(base,columns,weights):
    assert base.ndim==1 and columns.ndim==2 and len(base)==len(columns)==len(weights)
    return float(weights@np.maximum(base,columns.max(1))) if columns.shape[1] else float(weights@base)


def optimize(base,table,weights,actions,K):
    assert table.shape==(len(base),len(actions)) and len(actions)==K*21
    positions=[[j for j,a in enumerate(actions) if a['cluster']==k] for k in range(K)]
    assert all(len(p)==21 for p in positions)
    centroid=[next(j for j in p if actions[j]['method']=='centroid_D0' and actions[j]['alpha']==1 and actions[j]['cap_D0']==2) for p in positions]
    def value(selection):return objective(base,table[:,[p for p in selection if p>=0]],weights)
    def pick(selection,k):
        current=value(selection);options=[]
        for pos in [-1,*positions[k]]:
            trial=selection.copy();trial[k]=pos
            key=canonical(dict(Base=True,slot=k) if pos<0 else actions[pos])
            options.append((value(trial),key,trial))
        best=min(options,key=lambda x:(-x[0],x[1]))
        return best[2] if best[0]>current else selection
    starts=[]
    for name in ('all_centroid','Base_greedy'):
        selection=centroid.copy() if name=='all_centroid' else [-1]*K
        if name=='Base_greedy':
            for k in range(K):selection=pick(selection,k)
        trace=[dict(stage='start',J=value(selection),positions=selection.copy())]
        for sweep in range(2):
            for k in range(K):selection=pick(selection,k)
            trace.append(dict(stage=f'coordinate_pass{sweep+1}',J=value(selection),positions=selection.copy()))
        assert all(trace[j+1]['J']>=trace[j]['J'] for j in range(len(trace)-1))
        starts.append(dict(start=name,J=value(selection),positions=selection,trace=trace))
    winner=min(starts,key=lambda x:(-x['J'],canonical([actions[p] if p>=0 else dict(Base=True,slot=k) for k,p in enumerate(x['positions'])])))
    return dict(**winner,starts=starts)


def register():
    b=np.array([.5,.5]);w=np.array([.5,.5]);t=np.array([[1.,0.],[0.,1.]])
    assert objective(b,t,w)==1 and max(float(w@t[:,j]) for j in range(2))==.5
    assert objective(np.array([.5]),np.array([[[1.,0.],[0.,1.]]]).mean(-1),np.ones(1))==.5
    actions=[dict(cluster=0,method='centroid_D0' if j==0 else str(j),alpha=1,cap_D0=2) for j in range(21)]
    table=np.zeros((2,21));table[:,5]=.8
    o=optimize(b,table,w,actions,1);assert o['positions']==[5] and o['J']==.8
    table[:]=0;o=optimize(b,table,w,actions,1);assert o['J']==.5
    probe=np.zeros((5,42,3,7));assert probe[:,:,:,0][:,[1,22],:].shape==(5,2,3) and probe[:,:,:,0][:,[],:].shape==(5,0,3)
    p=read(PLAN);assert len(p['indices'])==1681 and len(p['sessions'])==21 and len(p['units'])==12 and p['noise_indices']==[0,1,2]
    atomic(ROOT/'SOURCE_REGISTRATION.json',dict(status='FROZEN_COMPLETE_CAL_SELECTION_SOURCE_BEFORE_L2_NATIVE_BEHAVIOR_READ_V1',passed=True,
        time=stamp(),source=ref(__file__),calibration_plan=ref(PLAN),
        pilot_registration=ref(OL/'analysis/l1_pilot_selection_v1/SELECTION_SOURCE_REGISTRATION.json'),
        root_native_release=ref(P2/'provenance/OFFLINE_LIBRARY_L2_CAL1681_NATIVE_ACTUAL_PIPELINE_RELEASE_V1.json'),
        selection='scene mean over S012, then maximum over Base and the K selected experts, then exact conditional population weights',
        weights='complete inner-cal population13676=294F+1491mid+11891high; not selected1681 ratios or global train population',
        optimizer='two starts(all-centroid, Base-sequential-greedy); two fixed-slot coordinate passes each; strict improvement only; canonical hash ties',
        methods_and_doses='three already-pruned methods per slot, all seven frozen dose/cap configurations; exactly one method/dose per slot, Base allowed',
        libraries_retained='all12; no new K/order/route pruning; no final top-six choice',
        normal_harm_and_safety_are_descriptive_not_selection_veto=True,
        actual_L2_behavior_scores_read=0,synthetic_mean_before_max_K1_and_monotonic_controls_passed=True),exclusive=True)
    print(json.dumps(ref(ROOT/'SOURCE_REGISTRATION.json')),flush=True)


def read_scores(record):
    v=meta(record)
    with np.load(checked(v['arrays']),allow_pickle=False) as z:
        a={k:np.array(z[k]) for k in ('scores_float64','Base_scores_float64','positions_int64','action_ids')}
    return v,a


def collect(acceptance,threads=4):
    registration=read(ROOT/'SOURCE_REGISTRATION.json');assert registration['passed'] and registration['source']==ref(__file__) and registration['calibration_plan']==ref(PLAN)
    p=read(PLAN);a=read(acceptance);assert a['passed'] and a['source_files']==native.sources()
    q=meta(a['queue']);assert q['scope']=='main1681' and q['noise_indices']==[0,1,2] and len(q['indices'])==1681
    assert a['jobs']==43706 and a['scenes']==1681 and a['actual_devices']==64 and a['all_experts_cross_applied_without_routing']
    admission=read(OL/'preparation/l2_native_v1/ADMISSION64_ACCEPTANCE.json')
    assert admission['passed'] and admission['serial_parallel_bytes_exact'] and admission['actual_devices']==16 and admission['actual_writers']==64
    ids=p['indices'];lookup={i:n for n,i in enumerate(ids)};weights=np.array([r['natural_mean_weight'] for r in p['rows']],np.float64)
    assert len(ids)==len(set(ids))==1681 and abs(weights.sum()-1)<1e-12
    requests={};matrices={};seen={};request_by_unit={};base=np.full((1681,3,7),np.nan)
    for j in q['jobs']:
        unit=j['unit']
        if unit in requests:assert request_by_unit[unit]==j['request'];continue
        r=meta(j['request']);requests[unit]=r;request_by_unit[unit]=j['request']
        assert r['plan']==ref(PLAN) and r['panel_indices']==ids and r['noise_indices']==[0,1,2]
        matrices[unit]=np.full((1681,len(r['actions']),3,7),np.nan);seen[unit]=np.zeros((1681,len(r['actions'])),bool)
    assert len(requests)==12 and set(requests)=={u['id'] for u in p['units']}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for count,(v,arrays) in enumerate(executor.map(read_scores,a['items']),1):
            j=v['job'];unit=j['unit'];n=lookup[v['index']];positions=j['positions'];r=requests[unit]
            assert v['passed'] and v['queue']==a['queue'] and arrays['positions_int64'].tolist()==positions
            assert arrays['action_ids'].tolist()==[r['actions'][pos]['action_id'] for pos in positions]
            assert arrays['scores_float64'].shape==(len(positions),3,7) and arrays['Base_scores_float64'].shape==(3,7)
            if np.isfinite(base[n]).all():assert base[n].tobytes()==arrays['Base_scores_float64'].tobytes()
            else:base[n]=arrays['Base_scores_float64']
            assert not seen[unit][n,positions].any();seen[unit][n,positions]=True;matrices[unit][n,positions]=arrays['scores_float64']
            if count%5000==0:print(json.dumps(dict(event='L2_SELECTION_COLLECT',items=count,total=len(a['items']))),flush=True)
    assert np.isfinite(base).all() and all(s.all() for s in seen.values()) and all(np.isfinite(t).all() for t in matrices.values())
    bmean=base[:,:,0].mean(1);baseline=float(weights@bmean);libraries=[]
    category=np.array([r['category'] for r in p['rows']])
    for unit in sorted(matrices):
        table=matrices[unit];r=requests[unit];spec=next(u['spec'] for u in p['units'] if u['id']==unit);actions=r['actions'];K=spec['K']
        selected=optimize(bmean,table[:,:,:,0].mean(2),weights,actions,K);slots=[]
        for k,pos in enumerate(selected['positions']):
            slots.append(dict(cluster=k,Base=True,method='Base',alpha=0,cap_D0=0,selected_position=-1,source_action_id=None) if pos<0 else
                dict(cluster=k,Base=False,method=actions[pos]['method'],alpha=actions[pos]['alpha'],cap_D0=actions[pos]['cap_D0'],selected_position=pos,source_action_id=actions[pos]['action_id']))
        columns=table[:,:,:,0][:,[pos for pos in selected['positions'] if pos>=0],:]
        mean_columns=columns.mean(2);oracle=np.maximum(bmean,mean_columns.max(1)) if len(mean_columns[0]) else bmean.copy()
        forced=[]
        for k,slot in enumerate(slots):
            if slot['Base']:continue
            s=table[:,slot['selected_position'],:,0].mean(1);bygroup={}
            for label in ('failure','mid','high'):
                mask=category==label;delta=100*(s[mask]-bmean[mask]);neg=delta<-1e-8
                bygroup[label]=dict(n=int(mask.sum()),Base_PDMS=float(100*bmean[mask].mean()),expert_PDMS=float(100*s[mask].mean()),gain_PDMS=float(delta.mean()),
                    damaged_count=int(neg.sum()),damaged_fraction=float(neg.mean()),mean_damage_among_damaged=float(-delta[neg].mean()) if neg.any() else 0.)
            forced.append(dict(cluster=k,selected_position=slot['selected_position'],natural_gain_PDMS=100*float(weights@(s-bmean)),groups=bygroup))
        matrix=native.data.arrays(ROOT/'matrices'/(unit+'.npz'),dict(indices_int64=np.array(ids,np.int64),scores_float64=table,Base_scores_float64=base,natural_weights_float64=weights))
        groups={label:dict(n=int((category==label).sum()),Base_PDMS=float(100*bmean[category==label].mean()),Oracle_PDMS=float(100*oracle[category==label].mean()),gain_PDMS=float(100*(oracle-bmean)[category==label].mean())) for label in ('failure','mid','high')}
        library=dict(unit=unit,spec=spec,inner_bank=r['bank'],inner_request=request_by_unit[unit],slots=slots,selection=selected,matrix=matrix,
            Base_PDMS=100*baseline,library_PDMS=100*selected['J'],gain_PDMS=100*(selected['J']-baseline),groups=groups,forced_selected_experts=forced,
            best_fixed_selected_expert_or_Base_PDMS=100*max([baseline,*[float(weights@mean_columns[:,j]) for j in range(mean_columns.shape[1])]]))
        libraries.append(library);print(json.dumps(dict(event='L2_LIBRARY_FROZEN',unit=unit,gain_PDMS=library['gain_PDMS'])),flush=True)
    out=dict(status=STATUS,passed=True,time=stamp(),producer=ref(__file__),source_registration=ref(ROOT/'SOURCE_REGISTRATION.json'),
        source_native_acceptance=ref(acceptance),calibration_plan=ref(PLAN),libraries=libraries,
        all12_libraries_enter_outer_train_refit=True,inner_cal_population_total=13676,selected_scenes=1681,sessions=21,noise_indices=[0,1,2],
        independent_outer_held_evaluation_done=False,final_six_selected=False,online_routing_studied=False,CONF_or_test_evaluated=False,
        matrices_preserve_all_experts_seven_score_fields=True,normal_harm_not_hidden_by_Base_Oracle=True)
    atomic(ROOT/'FROZEN_LIBRARIES.json',out,exclusive=True);print(json.dumps(ref(ROOT/'FROZEN_LIBRARIES.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','collect']);p.add_argument('--acceptance',type=Path);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    if args.action=='register':register()
    else:collect(args.acceptance,args.threads)
