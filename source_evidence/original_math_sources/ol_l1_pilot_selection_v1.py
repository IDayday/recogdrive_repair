"""Preregistered complete-menu coarse library selection, never final top-six.

Each cluster receives one shared method/dose. Noise-average precedes the
Base/K Oracle. Full calibration and OOF remain required after this pilot.
"""
from pathlib import Path
import sys,hashlib,json,argparse
sys.dont_write_bytecode=True
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import numpy as np
from b5_common import read,atomic,stamp
import ol_l1_native_v1 as native
ref,meta=native.ref,native.meta
ROOT=OL/'analysis/l1_pilot_selection_v1'

def canonical(spec):return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def objective(base,columns,weights):
    assert base.ndim==1 and columns.ndim==2 and columns.shape[0]==len(base)
    return float(weights@np.maximum(base,columns.max(1))) if columns.shape[1] else float(weights@base)

def optimize(base,table,weights,actions,K,centroid_index):
    """Two exact predefined starts, fixed slot order, two coordinate passes."""
    N,A=table.shape;assert A==K*91 and len(actions)==A
    def value(selection):return objective(base,table[:,[p for p in selection if p>=0]],weights)
    def slot_options(k):return [-1,*range(k*91,(k+1)*91)]
    def pick(selection,k):
        current=value(selection);best=list(selection);bestvalue=current
        candidates=[]
        for pos in slot_options(k):
            candidate=list(selection);candidate[k]=pos;j=value(candidate)
            key=canonical(dict(slot=k,Base=True) if pos<0 else actions[pos])
            candidates.append((j,key,candidate))
        winner=min(candidates,key=lambda x:(-x[0],x[1]))
        if winner[0]>current:bestvalue=winner[0];best=winner[2]
        return best,bestvalue
    starts=[]
    for name in ['all_centroid','Base_greedy']:
        selection=[k*91+centroid_index for k in range(K)] if name=='all_centroid' else [-1]*K
        if name=='Base_greedy':
            for k in range(K):selection,j=pick(selection,k)
        trace=[dict(stage='start',J=value(selection),positions=selection.copy())]
        for sweep in range(2):
            for k in range(K):selection,j=pick(selection,k)
            trace.append(dict(stage=f'coordinate_pass{sweep+1}',J=value(selection),positions=selection.copy()))
        starts.append(dict(start=name,J=value(selection),positions=selection,trace=trace))
    result=min(starts,key=lambda x:(-x['J'],canonical([actions[p] if p>=0 else dict(Base=True,slot=k) for k,p in enumerate(x['positions'])])))
    return dict(**result,starts=starts)

def marginal_methods(base,table,weights,actions,K,centroid_index):
    selected=[k*91+centroid_index for k in range(K)];methods=list(native.fitter.ops.MENU);out=[]
    for k in range(K):
        removed=selected.copy();removed[k]=-1
        reference=objective(base,table[:,[p for p in removed if p>=0]],weights);ranking=[]
        for method in methods:
            if method=='centroid_D0':continue
            candidates=[]
            for pos,a in enumerate(actions):
                if a['cluster']!=k or a['method']!=method:continue
                candidate=selected.copy();candidate[k]=pos
                j=objective(base,table[:,candidate],weights)
                candidates.append(dict(position=pos,J=j,hash=canonical(a)))
            best=min(candidates,key=lambda x:(-x['J'],x['hash']))
            ranking.append(dict(method=method,marginal_PDMS=100*(best['J']-reference),best=best,hash=canonical(dict(cluster=k,method=method))))
        ranking.sort(key=lambda x:(-x['marginal_PDMS'],x['hash']))
        out.append(dict(cluster=k,retained_methods=['centroid_D0',*[r['method'] for r in ranking[:2]]],ranking=ranking,
            all_seven_doses_retained=True,centroid_removal_reference_J=reference))
    return out

def selftest():
    base=np.array([.5,.5]);w=np.array([.5,.5]);table=np.array([[1.,0.],[0.,1.]])
    assert objective(base,table,w)==1 and max(float(w@table[:,k]) for k in range(2))==.5
    noise=np.array([[[1.,0.],[0.,1.]]]);mean=noise.mean(-1)
    assert objective(np.array([.5]),mean,np.array([1.]))==.5
    actions=[dict(cluster=0,method=str(j),alpha=1,cap_D0=2) for j in range(91)]
    t=np.zeros((2,91));t[:,5]=.8;o=optimize(base,t,w,actions,1,0)
    assert o['positions']==[5] and o['J']==.8
    atomic(ROOT/'SELECTION_SOURCE_REGISTRATION.json',dict(status='FROZEN_L1_COARSE_SELECTION_SOURCE_BEFORE_ANY_NATIVE_PILOT_SCORES_READ',passed=True,
        time=stamp(),source=ref(__file__),scientific_registration=ref(P2/'provenance/OFFLINE_LIBRARY_L1_PILOT_AND_SYMMETRIC_SCREEN_SCIENTIFIC_REGISTRATION_V1.json'),
        primary='scene noise mean, Base/K maximum, fixed natural weights; PDMS only',
        optimizer='all-centroid and Base-sequential-greedy starts; each two strict-improvement fixed-slot coordinate passes; ties canonical hash',
        pruning='two distinct K per R/D via order-best K union, both orders same directory; add K1; centroid plus marginal best2 methods per slot, all7 doses',
        final_top_six_selection=False,full_cal_and_complete_fivefold_OOF_required=True,
        synthetic_complementarity_and_mean_before_max_and_fixed_slot_selection=True,actual_candidate_scores_read=0),exclusive=True)
    print(ref(ROOT/'SELECTION_SOURCE_REGISTRATION.json'))

def collect(acceptance):
    reg=read(ROOT/'SELECTION_SOURCE_REGISTRATION.json');assert reg['passed'] and reg['source']==ref(__file__)
    accepted=read(acceptance);assert accepted['passed'] and accepted['source_files']==native.sources()
    q=meta(accepted['queue']);assert q['scope']=='main32'
    pilot=read(OL/'preparation/l1_pilot_metadata_v1/MANIFEST.json');indices=pilot['panel_indices'];positions={i:n for n,i in enumerate(indices)}
    weights=np.array([p['natural_mean_weight'] for p in pilot['panel']],np.float64);assert abs(weights.sum()-1)<1e-12
    base=np.full((32,2,7),np.nan);matrices={};seen={};request_by_unit={}
    for item in accepted['items']:
        v=meta(item);job=v['job'];r=meta(job['request']);unit=job['unit'];n=positions[v['index']];a=native.data.small_arrays(v['arrays'])
        if unit not in matrices:
            matrices[unit]=np.full((32,len(r['actions']),2,7),np.nan);seen[unit]=np.zeros((32,len(r['actions'])),bool);request_by_unit[unit]=job['request']
        if np.isfinite(base[n]).all():assert base[n].tobytes()==a['Base_scores_float64'].tobytes()
        else:base[n]=a['Base_scores_float64']
        pos=job['positions'];assert not seen[unit][n,pos].any();seen[unit][n,pos]=True;matrices[unit][n,pos]=a['scores_float64']
    assert len(matrices)==32 and np.isfinite(base).all() and all(s.all() for s in seen.values())
    bmean=base[:,:,0].mean(1);baseline=float(weights@bmean);results=[]
    for unit,table in sorted(matrices.items()):
        r=meta(request_by_unit[unit]);bank=meta(r['bank']);K=bank['spec']['K'];actions=r['actions'];mean=table[:,:,:,0].mean(2)
        centroid=next(j for j,a in enumerate(actions) if a['cluster']==0 and a['method']=='centroid_D0' and a['alpha']==1 and a['cap_D0']==2)
        chosen=optimize(bmean,mean,weights,actions,K,centroid);marginal=marginal_methods(bmean,mean,weights,actions,K,centroid)
        selections=chosen['positions'];columns=mean[:,[p for p in selections if p>=0]]
        stats=dict(Base_PDMS=100*baseline,library_PDMS=100*chosen['J'],gain_PDMS=100*(chosen['J']-baseline),
            best_fixed_expert_or_Base_PDMS=100*max([baseline,*[float(weights@columns[:,j]) for j in range(columns.shape[1])]]))
        array=native.data.arrays(ROOT/'matrices'/(unit+'.npz'),dict(indices_int64=np.array(indices,np.int64),scores_float64=table,Base_scores_float64=base,natural_weights_float64=weights))
        results.append(dict(unit=unit,spec=bank['spec'],request=request_by_unit[unit],matrix=array,selection=chosen,retained_per_slot=marginal,**stats))
    retained=[]
    for route in ['R','D']:
        available=[r for r in results if r['spec']['route']==route and r['spec']['K']>=2];chosenK=[]
        for order in ['P_to_C','C_to_P']:
            best=min([r for r in available if r['spec']['order']==order],key=lambda r:(-r['library_PDMS'],r['spec']['K'],canonical(r['spec'])))
            if best['spec']['K'] not in chosenK:chosenK.append(best['spec']['K'])
        if len(chosenK)<2:
            best=min([r for r in available if r['spec']['K'] not in chosenK],key=lambda r:(-r['library_PDMS'],r['spec']['K'],canonical(r['spec'])))
            chosenK.append(best['spec']['K'])
        retained.extend(r['unit'] for r in results if r['spec']['route']==route and (r['spec']['K'] in chosenK or r['spec']['K']==1))
    out=dict(status='COMPLETE_L1_FULL_MENU_COARSE_PILOT_FOR_FORMAL_CALIBRATION_NOT_FINAL_LIBRARIES',passed=True,time=stamp(),producer=ref(__file__),
        source_registration=ref(ROOT/'SELECTION_SOURCE_REGISTRATION.json'),native_acceptance=ref(acceptance),panel=ref(OL/'preparation/l1_pilot_metadata_v1/MANIFEST.json'),
        results=results,retained_units=retained,all32_cross_behavior_matrices_saved=True,Base_PDMS=100*baseline,
        no_test_or_full_OOF_claim=True,pruned_not_scientifically_invalid=True,online_routing_not_studied=True)
    atomic(ROOT/'SUMMARY.json',out,exclusive=True);print(ref(ROOT/'SUMMARY.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','collect']);p.add_argument('--acceptance',type=Path);a=p.parse_args()
    (selftest if a.action=='register' else lambda:collect(a.acceptance))()
