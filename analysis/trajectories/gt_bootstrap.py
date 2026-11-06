import recogdrive_paths as paths
from pathlib import Path
import csv,json
import numpy as np
from analyze_trajectories import csvout,js,sha
O=paths.analysis_root() / 'trajectories'

def main():
 rows=[];sources=[];rng=np.random.default_rng(20261104);draws=3000
 for pop in ['train','test']:
  p=O/f'{pop.upper()}_GT_ALIGNMENT_PER_SCENE.csv'
  with p.open() as f:r=list(csv.DictReader(f))
  meta=O.parent/'scene_metadata'/f'{pop}_metadata.npz'
  with np.load(meta,allow_pickle=False) as z:assert np.array_equal(z['token'],np.asarray([v['token'] for v in r]));log=z['log_id']
  _,inv=np.unique(log,return_inverse=True);nlogs=int(inv.max()+1);choice=np.asarray([int(v['choice']) for v in r])
  weights=rng.multinomial(nlogs,np.full(nlogs,1/nlogs),size=draws)
  for c in [-1,2,7,8]:
   m=choice>0 if c==-1 else choice==c;counts=np.bincount(inv,weights=m,minlength=nlogs);den=weights@counts;ok=den>0
   for k in ['Base_GT_ADE_m','teacher_GT_ADE_m','delta_GT_ADE_m','Base_GT_endpoint_alongtrack_error_m','teacher_GT_endpoint_alongtrack_error_m','delta_GT_abs_path_error_m']:
    val=np.asarray([float(v[k]) for v in r]);sums=np.bincount(inv,weights=val*m,minlength=nlogs);v=(weights[ok]@sums)/den[ok];lo,hi=np.quantile(v,[.025,.975])
    rows.append(dict(population=pop,choice=c,metric=k,n=int(m.sum()),support_logs=int((counts>0).sum()),mean=float(val[m].mean()),log_boot_ci95_low=float(lo),log_boot_ci95_high=float(hi)))
  sources.append(dict(path=str(p),sha256=sha(p),metadata_sha256=sha(meta),physical_logs=nlogs))
 csvout(O/'GT_ALIGNMENT_LOG_BOOTSTRAP.csv',rows);js(O/'GT_ALIGNMENT_LOG_BOOTSTRAP.json',dict(seed=20261104,draws=draws,sources=sources,rows=rows))
 for r in rows:
  if r['choice']==-1 and r['metric']=='delta_GT_ADE_m':print(json.dumps(r))
if __name__=='__main__':main()
