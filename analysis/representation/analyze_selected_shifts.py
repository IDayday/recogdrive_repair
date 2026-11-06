import recogdrive_paths as paths
"""Describe selected actual, not requested, role shifts across complete splits."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('OPENBLAS_CORETYPE','Sandybridge')
import csv,json,hashlib
from pathlib import Path
import numpy as np
OUT=paths.analysis_root() / 'representation'
ROOT=paths.native_root()
lib=json.loads((ROOT/'frozen_teacher/library.json').read_text())
B=np.load(ROOT/'frozen_teacher/space.npz')['basis']

def stats(x,prefix=''):
    x=np.asarray(x,np.float64)
    if not len(x):return {}
    return {prefix+k:float(v) for k,v in dict(mean=x.mean(),median=np.median(x),q05=np.quantile(x,.05),q95=np.quantile(x,.95),min=x.min(),max=x.max()).items()}
def eigstats(x,prefix):
    # x remains uncentered: one common direction should have energy rank one.
    G=x.T@x
    e=np.maximum(np.linalg.eigvalsh(G),0)[::-1]
    if not e.sum():return {prefix+'total_squared_energy':0.}
    p=e/e.sum();mask=p>0;c=np.cumsum(p)
    return {prefix+'total_squared_energy':float(e.sum()),prefix+'top1_energy_fraction':float(p[0]),
            prefix+'top2_energy_fraction':float(p[:2].sum()),prefix+'top8_energy_fraction':float(p[:8].sum()),
            prefix+'energy_rank99':int(np.searchsorted(c,.99)+1),prefix+'energy_rank999':int(np.searchsorted(c,.999)+1),
            prefix+'participation_rank':float(1/np.square(p).sum()),
            prefix+'entropy_rank':float(np.exp(-np.sum(p[mask]*np.log(p[mask]))))}
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

group_rows=[];vectors={};comparison=[];manifests={};full=[]
for pop in ['train','test']:
    path=ROOT/f'frozen_teacher/selections/{pop}.npz'
    data=read(path);manifests[pop]=dict(path=str(path),sha256=sha(path),n=len(data['tokens']))
    shift=data['selected_actual_role_shifts'].astype(np.float64)
    # native_cap returns scaled per-role addition vectors, not means of the
    # post-add floating point differences. Their average is an analysis view.
    shared=(shift[:,0]+shift[:,1])/2
    for choice in range(9):
        ix=np.flatnonzero(data['choice']==choice);x=shared[ix];n=len(ix)
        norms=np.linalg.norm(x,axis=1)
        nz=norms>1e-12
        u=x[nz]/norms[nz,None]
        mean=x.mean(0) if n else np.zeros(384)
        unitmean=u.mean(0) if len(u) else np.zeros(384)
        vectors[pop,choice]=dict(mean=mean,unitmean=unitmean,shift=x,unit=u)
        meth='Base' if choice==0 else lib['experts'][choice-1]['method']
        rows=dict(population=pop,choice=choice,method=meth,selected_scenes=n,fraction=n/len(data['tokens']),
                  shared_role_analysis_view='0.5*(scaled_image_addition_vector+scaled_text_addition_vector)',
                  zero_actual_shared_shifts=int((~nz).sum()),
                  distinct_shared_addition_vectors=int(len(np.unique(np.ascontiguousarray(x).view(np.dtype((np.void,x.dtype.itemsize*x.shape[1])))))),
                  image_text_max_absolute_difference=float(np.abs(shift[ix,0]-shift[ix,1]).max()) if n else None,
                  **stats(data['selected_actual_RMS'][ix],'native_valid_token_actual_RMS_'),
                  **stats(norms/np.sqrt(384),'actual_shared_RMS_'))
        if len(u):
            c=u@(unitmean/np.linalg.norm(unitmean))
            centered=x-mean
            rows.update(stats(c,'cosine_to_own_mean_direction_'))
            rows.update(eigstats(x,'uncentered_'))
            rows.update(eigstats(u,'unit_direction_'))
            rows.update(eigstats(centered,'centered_variation_'))
            rows['centered_variation_fraction_of_total_energy']=float(np.square(centered).sum()/np.square(x).sum())
            rows['D48_projection_energy_fraction']=float(np.square(x@B).sum()/np.square(x).sum())
        group_rows.append(rows)
    active=shared[np.linalg.norm(shared,axis=1)>1e-12]
    full.append(dict(population=pop,n=len(shared),nonzero_selected=len(active),**eigstats(active,'uncentered_'),
                     D48_projection_energy_fraction=float(np.square(active@B).sum()/np.square(active).sum())))

for choice in range(1,9):
    a,b=vectors['train',choice],vectors['test',choice]
    mean_a=a['mean']/np.linalg.norm(a['mean']);mean_b=b['mean']/np.linalg.norm(b['mean'])
    unit_a=a['unitmean']/np.linalg.norm(a['unitmean']);unit_b=b['unitmean']/np.linalg.norm(b['unitmean'])
    comparison.append(dict(choice=choice,method=lib['experts'][choice-1]['method'],
        train_selected_scenes=len(a['shift']),test_selected_scenes=len(b['shift']),
        cosine_raw_mean_shift_train_test=float(mean_a@mean_b),
        cosine_mean_unit_direction_train_test=float(unit_a@unit_b),
        native_shift_comparison_is_conditional_on_oracle_selection=True,
        **stats(b['unit']@unit_a,'test_cosine_to_train_mean_direction_'),
        **stats(a['unit']@unit_b,'train_cosine_to_test_mean_direction_')))

result=dict(schema='selected_actual_role_shift_descriptive_audit_v1',scope='Full Train85813 and Test12146; frozen selected teachers only; no inference or training',
    sources=manifests,analysis_shared384_view='arithmetic mean of native_cap scaled image and text addition vectors; not requested shifts and not mean post-add FP32 differences',
    native_cap_source=dict(path=str(paths.code_root()/'interface_teacher/native_cap.py'),sha256=sha(Path(str(paths.code_root()/'interface_teacher/native_cap.py')))),
    native_actual_RMS_from_original_selection=True,per_split=full,per_choice=group_rows,cross_split=comparison,
    native_actual_RMS_population='valid image|text tokens only; padding excluded from RMS denominator by native_cap.py:23',
    cautions=['Conditioning on oracle choice induces selection and cannot establish an inherent semantic category.',
        'The nearly fixed directions arise from shared teacher construction, not independent evidence of discovered scene semantics.',
        'FP32 add/subtract/mean rounding creates tiny full-dimensional residuals; spectral energy ranks are used instead of counting numerical nonzero eigenvalues.',
        'Train and Test selected subsets differ; similarities are observational, not matched-sample effects.'])
(OUT/'SELECTED_ACTUAL_SHIFT_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
writecsv(OUT/'SELECTED_ACTUAL_SHIFT_BY_CHOICE.csv',group_rows)
writecsv(OUT/'TRAIN_TEST_ACTUAL_SHIFT_SIMILARITY.csv',comparison)
np.savez_compressed(OUT/'SELECTED_ACTUAL_SHIFT_MEANS.npz',**{f'{p}_choice{c}_mean':v['mean'] for (p,c),v in vectors.items()})
print(json.dumps(dict(full=full,cross_split=comparison),indent=2))
