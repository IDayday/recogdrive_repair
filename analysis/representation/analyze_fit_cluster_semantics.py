import recogdrive_paths as paths
"""Join frozen fit-cluster labels to original-scene descriptive metadata."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('OPENBLAS_CORETYPE','Sandybridge')
import csv,json,hashlib
from pathlib import Path
import numpy as np

OUT=paths.analysis_root() / 'representation';ROOT=paths.native_root()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savecsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def association(label,values):
    vals,vi=np.unique(values.astype(str),return_inverse=True)
    a=np.zeros((8,len(vals)),np.int64);np.add.at(a,(label,vi),1)
    n=a.sum();p=a/n;pr=p.sum(1);pc=p.sum(0);expected=pr[:,None]*pc[None]
    nz=p>0;mi=float((p[nz]*np.log(p[nz]/expected[nz])).sum())
    hy=float(-(pr[pr>0]*np.log(pr[pr>0])).sum());hv=float(-(pc[pc>0]*np.log(pc[pc>0])).sum())
    chi=float(n*((p-expected)**2/np.maximum(expected,1e-100)).sum())
    return dict(n=int(n),MI_nats=mi,fraction_cluster_entropy_explained=mi/hy if hy else 0,
                symmetric_NMI=2*mi/(hy+hv)if hy+hv else 0,Cramers_V=float(np.sqrt(chi/n/max(1,min(a.shape)-1))),
                levels=vals.tolist(),contingency=a.tolist())
def eta(label,x):
    x=np.asarray(x,float);den=np.square(x-x.mean()).sum()
    return float(sum(np.sum(label==k)*np.square(x[label==k].mean()-x.mean()) for k in range(8))/den)if den else 0.

pair=load(OUT/'FIT_PAIR_GEOMETRY.npz')
metadata_path=OUT.parent/'scene_metadata/train_metadata.npz'
mfull=load(metadata_path);pos={t:i for i,t in enumerate(mfull['token'])}
idx=np.array([pos[t] for t in pair['tokens']]);m={k:v[idx] for k,v in mfull.items()}
assert np.array_equal(m['token'],pair['tokens'])
assert np.array_equal(m['log_id'],pair['sessions'])
label=pair['fit_labels']
selection_path=ROOT/'frozen_teacher/selections/train.npz';selection=load(selection_path)
sp={t:i for i,t in enumerate(selection['tokens'])};si=np.array([sp[t]for t in pair['tokens']]);q=selection['q_points'][si];choice=selection['choice'][si]
lib=json.loads((ROOT/'frozen_teacher/library.json').read_text())
expected_choice=np.array([{e['new_cluster']:e['slot']+1 for e in lib['experts']}[int(c)]for c in label])
chosen=q[np.arange(len(q)),choice];matched=q[np.arange(len(q)),expected_choice]
assert np.allclose(chosen,q.max(1),rtol=0,atol=1e-12)
m['speed_bin']=np.array(['<0.5','0.5-2','2-5','5-10','>=10'])[np.digitize(m['ego_speed_mps'],[.5,2,5,10])]
m['density_bin']=np.array(['0','1-3','4-8','>=9'])[np.digitize(m['vehicles20'],[1,4,9])]
m['Base_quality_bin']=np.array(['<70','70-90','90-99','99-100','100'])[np.digitize(q[:,0],[70,90,99,100])]
cats=['command','gt_motion','speed_bin','map_location','density_bin','red_light_observed','gt_start','gt_brake_to_stop','front_stopped_vehicle20','Base_quality_bin']
associations=[dict(variable=k,**association(label,m[k]))for k in cats]
nums=['ego_speed_mps','ego_accel_mps2','gt_path_m','gt_turn_deg','vehicles20','pedestrians20','bicycles20','objects20','front_vehicle_min_x_m']
continuous=[dict(variable=k,eta_squared=eta(label,m[k]))for k in nums]
profiles=[];enrichment=[];confusion=np.zeros((8,9),np.int64);np.add.at(confusion,(label,choice),1)
for c in range(8):
    mask=label==c
    row=dict(cluster=c,n=int(mask.sum()),matched_frozen_choice=int(expected_choice[mask][0]),
             distinct_logs=len(np.unique(m['log_id'][mask])),Base_mean_PDMS=float(q[mask,0].mean()),
             chosen_teacher_mean_PDMS=float(chosen[mask].mean()),matched_expert_mean_PDMS=float(matched[mask].mean()),
             matched_is_exact_choice_fraction=float((choice[mask]==expected_choice[mask]).mean()),
             matched_is_tied_best_fraction=float((matched[mask]>=chosen[mask]-1e-9).mean()),
             matched_regret_PDMS=float((chosen[mask]-matched[mask]).mean()))
    for k in nums:row[k+'_mean']=float(m[k][mask].mean());row[k+'_median']=float(np.median(m[k][mask]))
    for k in ['gt_start','gt_brake_to_stop','front_stopped_vehicle20','red_light_observed']:
        row[k+'_fraction']=float(m[k][mask].mean())
    for k in ['command','gt_motion','map_location']:
        for v in np.unique(m[k]):row[k+'__'+str(v)]=float((m[k][mask]==v).mean())
    for j in range(9):row[f'oracle_choice{j}_fraction']=float((choice[mask]==j).mean())
    profiles.append(row)
    for k in cats:
        for v in np.unique(m[k]):
            gm=float((m[k][mask]==v).mean());om=float((m[k][~mask]==v).mean())
            enrichment.append(dict(cluster=c,variable=k,level=str(v),n_group=int(mask.sum()),n_level_in_group=int(((m[k]==v)&mask).sum()),
                group_fraction=gm,rest_fraction=om,difference=gm-om,lift_vs_rest=gm/om if om else None))

summary=dict(schema='fit_cluster_scene_semantics_descriptive_v1',population='1654 successful Train paired-optimization cases only; not full Train or Test',
    sources={str(p):sha(p)for p in [OUT/'FIT_PAIR_GEOMETRY.npz',metadata_path,selection_path]},
    categorical_associations=associations,continuous_eta_squared=continuous,
    original_fit_cluster_vs_final_oracle=dict(n=len(choice),cluster_by_oracle_choice_confusion=confusion.tolist(),
        exact_matched_expert_choices=int((choice==expected_choice).sum()),exact_match_fraction=float((choice==expected_choice).mean()),
        matched_expert_is_tied_best_cases=int((matched>=chosen-1e-9).sum()),matched_expert_tied_best_fraction=float((matched>=chosen-1e-9).mean()),
        Base_final_choices=int((choice==0).sum()),Base_mean_PDMS=float(q[:,0].mean()),
        matched_expert_mean_PDMS=float(matched.mean()),oracle_mean_PDMS=float(chosen.mean()),
        mean_matched_expert_regret_PDMS=float((chosen-matched).mean())),
    limitations=['All descriptors are exploratory effect sizes, without p-value ranking or causal claims.',
        'GT motion uses future ground truth and is suitable for explanation, not available-at-deployment routing.',
        'Selection of low-Base/high-GT paired cases conditions this population; associations cannot be extrapolated to all scenes.',
        'No Test GT-optimized repair direction or fit-cluster membership exists; final Test choice is a different label.',
        'Adjacent scenes share logs; no independent-scene confidence or significance claim.'])
(OUT/'FIT_CLUSTER_SEMANTICS_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
savecsv(OUT/'FIT_CLUSTER_SCENE_PROFILES.csv',profiles)
savecsv(OUT/'FIT_CLUSTER_SEMANTIC_ASSOCIATIONS.csv',[{k:v for k,v in r.items()if k not in ['levels','contingency']}for r in associations])
savecsv(OUT/'FIT_CLUSTER_CONTINUOUS_ETA.csv',continuous)
savecsv(OUT/'FIT_CLUSTER_SEMANTIC_ENRICHMENT.csv',enrichment)
savecsv(OUT/'FIT_CLUSTER_VS_ORACLE_CONFUSION.csv',[dict(cluster=c,**{f'choice{j}':int(confusion[c,j])for j in range(9)})for c in range(8)])
print(json.dumps(summary['original_fit_cluster_vs_final_oracle'],indent=2))
print('ASSOCIATIONS',[(x['variable'],x['Cramers_V'],x['fraction_cluster_entropy_explained'])for x in associations])
print('ETA',continuous)
