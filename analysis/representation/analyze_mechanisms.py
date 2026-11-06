import recogdrive_paths as paths
"""Read-only CPU audit of frozen local D48/K8 teacher geometry."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('OPENBLAS_CORETYPE', 'Sandybridge')
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

OUT = paths.analysis_root() / 'representation'
ROOT = paths.native_root()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def summary(x):
    x = np.asarray(x, np.float64)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                q05=float(np.quantile(x, .05)), q25=float(np.quantile(x,.25)),
                q75=float(np.quantile(x,.75)), q95=float(np.quantile(x,.95)),
                min=float(x.min()), max=float(x.max()))

def cosine(x):
    x = np.asarray(x)
    y = x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-20)
    return y @ y.T

def table(path, rows):
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def adjusted_rand_score(a,b):
    _, a=np.unique(a,return_inverse=True);_,b=np.unique(b,return_inverse=True)
    m=np.zeros((a.max()+1,b.max()+1),np.int64)
    np.add.at(m,(a,b),1)
    choose=lambda x:np.sum(x*(x-1)/2)
    ab=choose(m);aa=choose(m.sum(1));bb=choose(m.sum(0));total=len(a)*(len(a)-1)/2
    expect=aa*bb/total;den=(aa+bb)/2-expect
    return float((ab-expect)/den) if den else 1.

def silhouette_score(x,label,metric):
    x=np.asarray(x,np.float64)
    if metric=='cosine':d=np.maximum(1-cosine(x),0)
    else:
        sq=np.square(x).sum(1)
        d=np.sqrt(np.maximum(sq[:,None]+sq[None]-2*x@x.T,0))
    np.fill_diagonal(d,0)
    a=np.empty(len(x));b=np.full(len(x),np.inf)
    for k in np.unique(label):
        ix=label==k;n=ix.sum()
        av=d[:,ix].sum(1)
        a[ix]=av[ix]/max(n-1,1)
        b[~ix]=np.minimum(b[~ix],av[~ix]/n)
    return float(np.mean((b-a)/np.maximum(np.maximum(a,b),1e-20)))

state_path = ROOT/'fitting/ALLTRAIN/SPACE_AND_CLUSTER.pt'
state = torch.load(state_path,map_location='cpu',weights_only=False)
inner = torch.load(ROOT/'fitting/INNER/SPACE_AND_CLUSTER.pt',map_location='cpu',weights_only=False)
transfer = torch.load(ROOT/'fitting/ALLTRAIN/CLUSTER_TRANSFER.pt',map_location='cpu',weights_only=False)
library = json.loads((ROOT/'frozen_teacher/library.json').read_text())
B = state['space']['basis']
delta = state['success_view']-state['fail_view']
scale = np.maximum(state['actual_fulltoken_RMS'],state['space']['normalization_floor'])
normalized = delta/scale[:,None]
coef = delta@B
projected = coef@B.T
label = state['cluster']['labels']
unit = state['geometry']['orders']['C_to_P']['unit_directions']
energy = np.square(state['space']['spectrum'])
cum = np.cumsum(energy)/energy.sum()
raw_ratio = np.square(projected).sum()/np.square(delta).sum()
individual_ratio = np.square(projected).sum(1)/np.square(delta).sum(1)
assert np.allclose(np.square(normalized).sum(),energy.sum(),rtol=1e-12)
assert np.max(np.abs(delta-state['geometry']['delta'])) < 1e-12
assert np.max(np.abs(B-state['geometry']['basis'])) < 1e-12
spectral_entropy_rank = float(np.exp(-np.sum((energy/energy.sum())*np.log(energy/energy.sum()))))
participation_rank = float(energy.sum()**2/np.square(energy).sum())

cluster_rows = []
for c in range(8):
    ix = np.flatnonzero(label==c)
    u = unit[ix]
    resultant = np.linalg.norm(u.mean(0))
    cv = u@state['cluster']['centers'][c]
    stats = state['cluster']['statistics'][c]
    cluster_rows.append(dict(cluster=c,n=len(ix),sessions=stats['sessions'],
        maximum_session_fraction=stats['maximum_session_fraction'],
        mean_cosine_to_center=float(cv.mean()),median_cosine_to_center=float(np.median(cv)),
        mean_angle_degrees=float(np.rad2deg(np.arccos(np.clip(cv,-1,1))).mean()),
        mean_pairwise_cosine=float((len(ix)*resultant**2-1)/(len(ix)-1)),
        mean_projection_energy_fraction=float(individual_ratio[ix].mean()),
        median_actual_fulltoken_teacher_RMS=float(np.median(state['actual_fulltoken_RMS'][ix])),
        weak_support=stats['support_weak']))

trials = state['cluster']['trials']
win = state['cluster']['selected_restart']
ari_to_best = [adjusted_rand_score(label,t['labels']) for t in trials]
stability = [adjusted_rand_score(a['labels'],b['labels']) for i,a in enumerate(trials) for b in trials[i+1:]]

slot_rows, constant_vectors, all_means = [], [], []
for slot in library['experts']:
    sid=slot['slot']; z=np.load(ROOT/f'frozen_teacher/slot{sid:02d}.npz',allow_pickle=False)
    common={k.replace('model.common.',''):z[k] for k in z.files if k.startswith('model.common.')}
    method=slot['method']
    if method=='centroid_D0': q=common['centroid_coefficients'];fixed=True
    elif method=='mean': q=common['mean_delta'];fixed=True
    else:q=common['mean_delta'];fixed=False
    # For dynamic operators this representative is their mean pair correction,
    # not a claim that their actual request is constant.
    representative=q@B.T
    requested=representative*slot['alpha']*library['raw_shift_factor']
    const_rms=float(np.sqrt(np.square(requested).mean()))
    cap=slot['cap_D0']*state['D0']*library['cap_factor']
    row=dict(slot=sid,choice=sid+1,method=method,new_cluster=slot['new_cluster'],
             fit_pairs=int(common['fit_pairs']),constant_pre_cap_request=fixed,
             representative_is_actual_constant_request=fixed,
             representative_requested_shared384_RMS=const_rms,native_fulltoken_cap_RMS=float(cap),
             representative_cosine_mean_vs_direction_centroid=float(common['mean_delta']@common['centroid_coefficients']/(np.linalg.norm(common['mean_delta'])*np.linalg.norm(common['centroid_coefficients']))))
    slot_rows.append(row);constant_vectors.append(requested);all_means.append(common['mean_delta']@B.T)
constant_vectors=np.array(constant_vectors)
cos=cosine(constant_vectors)

pair_rows=[]
oldslot={c:s['slot'] for s in library['experts'] for c in [s['new_cluster']]}
for i,token in enumerate(state['tokens']):
    pair_rows.append(dict(token=token,session=state['sessions'][i],train_index=int(state['Q_indices'][i]),
        fit_cluster=int(label[i]),matched_frozen_slot=oldslot[int(label[i])],
        matched_frozen_choice=oldslot[int(label[i])]+1,
        fulltoken_actual_teacher_RMS=float(state['actual_fulltoken_RMS'][i]),
        shared384_delta_RMS=float(np.sqrt(np.square(delta[i]).mean())),
        D48_projection_energy_fraction=float(individual_ratio[i]),
        direction_cosine_to_center=float(unit[i]@state['cluster']['centers'][label[i]])))

ordinary=np.load(ROOT/'fitting/ORDINARY_LOCAL.npz',allow_pickle=False)
index={str(t):i for i,t in enumerate(ordinary['tokens'])}
oi=np.array([index[t] for t in state['tokens']])
raw_state=ordinary['raw_aux24_float32'][oi].astype(np.float64)
std=(raw_state-raw_state.mean(0))/np.maximum(raw_state.std(0),1e-8)
silhouettes=dict(
    repair_direction_cosine=float(silhouette_score(unit,label,metric='cosine')),
    base_pooled_feature_cosine=float(silhouette_score(state['fail_view'],label,metric='cosine')),
    base_pooled_feature_euclidean=float(silhouette_score(state['fail_view'],label,metric='euclidean')),
    standardized_aux24_euclidean=float(silhouette_score(std,label,metric='euclidean')))

report=dict(schema='local_frozen_teacher_mechanism_audit_v1',scope='CPU only; no refitting, inference, scoring or training',
    source_files={str(p):digest(p) for p in [state_path,ROOT/'fitting/INNER/SPACE_AND_CLUSTER.pt',ROOT/'fitting/ALLTRAIN/CLUSTER_TRANSFER.pt',ROOT/'frozen_teacher/library.json',ROOT/'frozen_teacher/space.npz']},
    population=dict(qualified_pairs=len(label),candidate_count=1689,ordinary_selected_count=9881,total_Train=85813,pair_sessions=len(set(state['sessions']))),
    clustering=dict(input='unit-normalized full384 successful-minus-failed interface shifts',order='C_to_P',K=8,
        not_scene_semantic_clustering=True,not_final_oracle_selection_labels=True,
        selected_restart=int(win),restarts=len(trials),selected_objective=float(state['cluster']['objective']),
        objective_range=[float(min(t['objective'] for t in trials)),float(max(t['objective'] for t in trials))],
        all20_restart_pairwise_ARI=summary(stability),each_restart_ARI_to_best=ari_to_best,
        inner_to_alltrain_refit_ARI=float(transfer['ARI']),inner_to_alltrain_matching_contingency=transfer['contingency'].tolist(),
        inner_to_alltrain_comparison_shared_pairs=int(transfer['contingency'].sum()),
        inner_to_alltrain_maximum_matching_shared_pairs=int(transfer['maximum_integer_overlap']),
        mean_unit_direction_norm=float(np.linalg.norm(unit.mean(0))),clusters=cluster_rows,
        silhouette_by_fixed_direction_labels=silhouettes),
    subspace=dict(rank=48,dimension=384,normalized_deltas_centered=False,
        normalized_energy_fraction=float(cum[47]),raw_delta_energy_fraction=float(raw_ratio),
        per_pair_raw_projection_energy_fraction=summary(individual_ratio),
        spectrum_effective_rank=int(state['space']['effective_rank']),spectral_entropy_rank=spectral_entropy_rank,
        spectral_participation_rank=participation_rank,
        orthogonality_max_abs=float(np.max(np.abs(B.T@B-np.eye(48)))),
        normalized_energy_at_ranks={str(r):float(cum[r-1]) for r in [1,2,4,8,16,24,32,48,64,96,128,192,256,384]}),
    experts=dict(slots=slot_rows,representative_requested_shift_cosine=cos.tolist(),
        constant_request_choice_ids=[r['choice'] for r in slot_rows if r['constant_pre_cap_request']],
        input_dependent_choice_ids=[r['choice'] for r in slot_rows if not r['constant_pre_cap_request']],
        ordinary_aux24_unused_by_selected_analytic_operators=True,
        image_and_text_requested_shift_identical=True,padding_not_shifted=True,
        all_alphas=2.,raw_shift_factor=2.,cap_factor=2.,D0=float(state['D0']),cap_RMS=4*float(state['D0']),
        enhanced_raw_multiplicative_factor=4.,effective_rank_of_8_representative_vectors=int(np.linalg.matrix_rank(constant_vectors)),
        caveat='Representatives for dynamic contract/local are mean-delta vectors; actual FP32 add/cap can introduce small scene-specific changes even for constant pre-cap requests.'))
(OUT/'MECHANISM_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
table(OUT/'CLUSTER_GEOMETRY.csv',cluster_rows)
table(OUT/'EXPERT_OPERATORS.csv',slot_rows)
table(OUT/'FIT_PAIR_CLUSTERS.csv',pair_rows)
table(OUT/'BASIS_SPECTRUM.csv',[dict(component=i+1,singular_value=float(s),normalized_energy_fraction=float(e/energy.sum()),cumulative_energy_fraction=float(cum[i])) for i,(s,e) in enumerate(zip(state['space']['spectrum'],energy))])
table(OUT/'EXPERT_REPRESENTATIVE_COSINE.csv',[dict(choice_a=i+1,choice_b=j+1,cosine=float(cos[i,j]),both_input_independent=slot_rows[i]['constant_pre_cap_request'] and slot_rows[j]['constant_pre_cap_request']) for i in range(8) for j in range(i+1,8)])
np.savez_compressed(OUT/'FIT_PAIR_GEOMETRY.npz',tokens=np.array(state['tokens']),sessions=np.array(state['sessions']),fit_labels=label,coefficients=coef,full384_delta=delta,base_pooled=state['fail_view'])
print(json.dumps({k:report[k] for k in ['population','subspace','experts']},indent=2))
print('silhouette',silhouettes,'ARI',summary(stability))
