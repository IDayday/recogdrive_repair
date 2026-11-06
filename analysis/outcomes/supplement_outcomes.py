import recogdrive_paths as paths
"""Analytic PDMS attribution and baseline-severity composition diagnostics.

Shapley decomposition is of an arithmetic score formula, not of causal model
abilities. No new rollout or metric evaluation is performed.
"""
from pathlib import Path
import json,csv,itertools,math,hashlib
import numpy as np
N=paths.native_root()
O=paths.analysis_root() / 'outcomes'
def calc(x):return x[...,0]*x[...,1]*(5*x[...,2]+5*x[...,3]+2*x[...,4])/12*100
rows=[];stats={}
for pop in ['train','test']:
 with np.load(N/f'requests/{pop}/MATRIX.npz') as z:s=z['scores']
 with np.load(N/f'frozen_teacher/selections/{pop}.npz') as z:choice=z['choice']
 n=len(choice);a=np.arange(n);base=s[:,0];teacher=s[a,choice]
 x0=base[:,:,1:6];x1=teacher[:,:,1:6]
 assert np.max(np.abs(calc(x0)-base[:,:,0]*100))<1e-10
 assert np.max(np.abs(calc(x1)-teacher[:,:,0]*100))<1e-10
 phi=np.zeros_like(x0)
 for mask in range(32):
  use=np.array([bool(mask&(1<<j)) for j in range(5)])
  k=int(use.sum());x=np.where(use,x1,x0);f0=calc(x)
  for j in range(5):
   if use[j]:continue
   y=x.copy();y[...,j]=x1[...,j]
   weight=math.factorial(k)*math.factorial(4-k)/math.factorial(5)
   phi[...,j]+=weight*(calc(y)-f0)
 gain=(teacher[:,:,0]-base[:,:,0])*100
 assert np.max(np.abs(phi.sum(-1)-gain))<1e-10
 for c in range(10):
  mask=np.ones(n,bool) if c==9 else choice==c
  for j,metric in enumerate(['NC','DAC','TTC','Progress','Comfort']):
   rows.append({'population':pop,'choice':'ALL' if c==9 else c,'scenes':int(mask.sum()),'metric':metric,
    'arithmetic_Shapley_PDMS_points':float(phi[mask,:,j].mean()),
    'fraction_of_PDMS_gain':float(phi[mask,:,j].sum()/gain[mask].sum()) if gain[mask].sum()!=0 else None})
 stat={'failed_metrics':{}}
 for k,name in [(1,'NC'),(2,'DAC'),(3,'TTC'),(5,'Comfort'),(6,'DDC')]:
  fail=base[:,:,k]<1;tf=teacher[:,:,k]<1;rec=fail&~tf;new=~fail&tf
  stat['failed_metrics'][name]={'base_failed_scene_noises':int(fail.sum()),'repaired_to_one':int(rec.sum()),
    'recovery_pct':float(rec.sum()/fail.sum()*100) if fail.sum() else None,'new_failed':int(new.sum()),
    'teacher_failed_scene_noises':int(tf.sum())}
 stats[pop]=stat
with (O/'PDMS_ARITHMETIC_SHAPLEY.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (O/'SAFETY_RECOVERY_TOTAL.json').open('w') as f:json.dump(stats,f,indent=2)
with (O/'BASE_SCORE_STRATA.csv').open() as f:rr=list(csv.DictReader(f))
tr=[r for r in rr if r['population']=='train'];te=[r for r in rr if r['population']=='test']
wt=np.array([float(r['scene_pct'])/100 for r in tr]);we=np.array([float(r['scene_pct'])/100 for r in te])
gt=np.array([float(r['gain']) for r in tr]);ge=np.array([float(r['gain']) for r in te])
composition=float(np.dot(we-wt,(ge+gt)/2));within=float(np.dot((we+wt)/2,ge-gt))
val={'strata':[r['stratum'] for r in tr],'Train_weights':wt.tolist(),'Test_weights':we.tolist(),'Train_within_stratum_gain':gt.tolist(),
 'Test_within_stratum_gain':ge.tolist(),'Train_gain':float(wt@gt),'Test_gain':float(we@ge),
 'Test_weights_Train_stratum_gain':float(we@gt),'Train_weights_Test_stratum_gain':float(wt@ge),
 'symmetric_Kitagawa_between_population_gain_difference':{'composition':composition,'within_strata':within,'total':composition+within,
 'composition_fraction':composition/(composition+within)},
 'interpretation':'Purely descriptive decomposition into the same five frozen baseline-PDMS strata. Within-stratum severity and all other covariates remain confounded; not causal attribution.'}
(O/'BASE_SEVERITY_COMPOSITION.json').write_text(json.dumps(val,indent=2)+'\n')
print(json.dumps({'severity':val,'recovery':stats,'Shapley_all':[r for r in rows if r['choice']=='ALL']},indent=2))
