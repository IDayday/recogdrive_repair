import recogdrive_paths as paths
from pathlib import Path
import csv,json,hashlib
import numpy as np
O=paths.analysis_root() / 'trajectories'
N=paths.native_root()

def source_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=[];provenance=[];rng=np.random.default_rng(20261103);draws=3000
 for pop in ['train','test']:
  p=O/f'{pop.upper()}_SCENE_GEOMETRY.csv'
  with p.open() as f:r=list(csv.DictReader(f))
  inp=N/f'engineering/independent_full_teacher_effects_20261027/{pop.upper()}_SCENES.csv'
  with inp.open() as f:lm={v['token']:v['log_id'] for v in csv.DictReader(f)}
  logs=np.asarray([lm[v['token']] for v in r]);unique,inv=np.unique(logs,return_inverse=True)
  choice=np.asarray([int(v['choice']) for v in r]);nlogs=len(unique)
  weights=rng.multinomial(nlogs,np.full(nlogs,1/nlogs),size=draws)
  for c in [-1,*range(1,9)]:
   m=choice>0 if c==-1 else choice==c
   counts=np.bincount(inv,weights=m,minlength=nlogs);den=weights@counts;good=den>0
   for k in ['endpoint_dx','endpoint_dy','delta_path_m','delta_abs_final_heading_deg','ADE_m','gain_points']:
    val=np.asarray([float(v[k]) for v in r]);sums=np.bincount(inv,weights=val*m,minlength=nlogs)
    sims=(weights[good]@sums)/den[good];lo,hi=np.quantile(sims,[.025,.975])
    rows.append(dict(population=pop,choice=c,slot='all_repaired' if c==-1 else c-1,metric=k,n=int(m.sum()),support_logs=int((counts>0).sum()),mean=float(val[m].mean()),ci95_log_boot_lower=float(lo),ci95_log_boot_upper=float(hi),bootstrap_draws=draws))
  provenance.append(dict(population=pop,physical_log_ids=nlogs,source=str(inp),source_sha256=source_sha(inp)))
 with (O/'GEOMETRY_LOG_BOOTSTRAP.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 (O/'GEOMETRY_LOG_BOOTSTRAP.json').write_text(json.dumps(dict(seed=20261103,draws=draws,unit='log_id, resample same number logs with replacement and pool their scenes; scene-weighted mean; no independence of noises',sources=provenance,rows=rows),indent=2)+'\n')
 print(json.dumps(provenance))

if __name__=='__main__':main()
