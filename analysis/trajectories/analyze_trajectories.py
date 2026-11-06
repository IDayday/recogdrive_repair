"""CPU-only analysis of immutable native teacher trajectories; no model execution."""
from __future__ import annotations
import recogdrive_paths as paths
import concurrent.futures as cf
import csv,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
N=paths.native_root()
O=paths.analysis_root() / 'trajectories'
METHODS=['Base','contract','centroid_D0','centroid_D0','centroid_D0','centroid_D0','local','centroid_D0','mean']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 return x

def js(p,x):p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def describe(x):
 x=np.asarray(x,dtype=np.float64);x=x[np.isfinite(x)]
 return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,p10=float(np.quantile(x,.1)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None)
def corr(x,y):
 x=np.asarray(x);y=np.asarray(y);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None

def features(t):
 # Any leading dimensions, last two dimensions horizon8,state3. Native dt=.5sec.
 xy=t[...,:2].astype(np.float64);origin=np.zeros(xy.shape[:-2]+(1,2))
 d=np.diff(np.concatenate([origin,xy],axis=-2),axis=-2);ds=np.linalg.norm(d,axis=-1);speed=ds/.5
 hd=np.unwrap(np.concatenate([np.zeros(t.shape[:-2]+(1,)),t[...,2].astype(np.float64)],axis=-1),axis=-1)
 dh=np.diff(hd,axis=-1)
 curv=np.divide(np.abs(dh),ds,out=np.zeros_like(ds),where=ds>.25)
 return dict(endpoint_x=xy[...,-1,0],endpoint_y=xy[...,-1,1],endpoint_abs_y=np.abs(xy[...,-1,1]),path_m=ds.sum(-1),mean_speed_mps=ds.sum(-1)/4.,last_speed_mps=speed[...,-1],max_speed_mps=speed.max(-1),final_heading_deg=np.rad2deg(hd[...,-1]),abs_final_heading_deg=np.abs(np.rad2deg(hd[...,-1])),max_lateral_m=np.abs(xy[...,1]).max(-1),mean_abs_curvature=curv.sum(-1)/np.maximum((ds>.25).sum(-1),1),mean_acceleration_mps2=(speed[...,-1]-speed[...,0])/3.5)

def extract_train(sel):
 out=O/'TRAIN_SELECTED_TRAJECTORIES.npz'
 if out.exists():
  with np.load(out,allow_pickle=False) as z:
   assert np.array_equal(z['tokens'],sel['tokens'])
   return {k:z[k] for k in z.files}
 rows=[];index=N/'requests/train/REPAIRED_CACHE_INDEX.jsonl'
 with index.open() as f:
  for line in f:
   r=json.loads(line);rows.append((r['source_row_index'],r['token'],r['source_scores']['path'],r['source_scores']['sha256'],r['log_id']))
 rows.sort();assert len(rows)==len(sel['tokens'])
 for i,r in enumerate(rows):assert r[0]==i and r[1]==str(sel['tokens'][i])
 tr=np.empty((len(rows),2,5,8,3),np.float32);state=np.empty((len(rows),24),np.float32);logs=np.asarray([r[4] for r in rows])
 def worker(r):
  i,token,path,digest,_=r;b=Path(path).read_bytes();assert hashlib.sha256(b).hexdigest()==digest
  with np.load(io.BytesIO(b),allow_pickle=False) as z:
   assert np.array_equal(z['positions'],np.arange(9));a=z['trajectories'];c=int(sel['choice'][i])
   assert np.array_equal(z['base_trajectories'],a[0]);assert np.isfinite(a).all()
   return i,np.stack([a[0],a[c]]),z['state24']
 started=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=8) as pool:
  for i,t,s in pool.map(worker,rows,chunksize=16):tr[i]=t;state[i]=s
 data=dict(tokens=sel['tokens'],choice=sel['choice'],indices=sel['indices'],trajectories=tr,state24=state,logs=logs)
 np.savez_compressed(out,**data)
 js(O/'TRAIN_EXTRACTION.json',dict(status='COMPLETE',scenes=len(rows),source_count=len(rows),each_source_bytes_sha_verified=True,index_sha256=sha(index),elapsed_seconds=time.monotonic()-started,workers=8,output_sha256=sha(out)))
 return data

def analyse_pop(pop,data,selection,scores):
 t=data['trajectories'];base=t[:,0];teacher=t[:,1];choice=selection['choice'];n=len(choice)
 bf={k:v.mean(1) for k,v in features(base).items()};tf={k:v.mean(1) for k,v in features(teacher).items()}
 delta_xy=teacher[...,:2].astype(np.float64)-base[...,:2].astype(np.float64)
 dist=np.linalg.norm(delta_xy,axis=-1)
 geom=dict(ADE_m=dist.mean((1,2)),FDE_m=dist[:,:,-1].mean(1),endpoint_dx=delta_xy[:,:,-1,0].mean(1),endpoint_dy=delta_xy[:,:,-1,1].mean(1),endpoint_abs_dy=np.abs(delta_xy[:,:,-1,1]).mean(1),dx_positive_noise_fraction=(delta_xy[:,:,-1,0]>.1).mean(1),dx_negative_noise_fraction=(delta_xy[:,:,-1,0]<-.1).mean(1))
 q=scores[:,:,:,0].mean(2)*100;idx=np.arange(n);gain=q[idx,choice]-q[:,0]
 bnoise=np.linalg.norm(base[...,:2]-base[...,:2].mean(1,keepdims=True),axis=-1).mean((1,2))
 tnoise=np.linalg.norm(teacher[...,:2]-teacher[...,:2].mean(1,keepdims=True),axis=-1).mean((1,2))
 geom['Base_noise_spread_m']=bnoise;geom['teacher_noise_spread_m']=tnoise
 flags={'Base_stalled_path_lt2m':bf['path_m']<2,'Base_reverse_endpoint_x_lt_minus1m':bf['endpoint_x']<-1,'Base_left_final_heading_gt20deg':bf['final_heading_deg']>20,'Base_right_final_heading_lt_minus20deg':bf['final_heading_deg']<-20,'Base_straight_abs_final_heading_lt10deg':bf['abs_final_heading_deg']<10,'Base_slow_mean_speed_lt2mps':bf['mean_speed_mps']<2,'Base_fast_mean_speed_gt10mps':bf['mean_speed_mps']>10,'Base_low_progress_lt0p5':scores[:,0,:,4].mean(1)<.5,'Base_any_safety_failure':(scores[:,0,:,:][:,:,[1,2,3]]<1-1e-10).any((1,2))}
 records=[]
 for i in range(n):
  row=dict(population=pop,token=str(data['tokens'][i]),global_index=int(selection['indices'][i]),choice=int(choice[i]),slot=int(choice[i])-1,method=METHODS[int(choice[i])],Base_PDMS=q[i,0],teacher_PDMS=q[i,choice[i]],gain_points=gain[i])
  row.update({f'Base_{k}':float(v[i]) for k,v in bf.items()});row.update({f'teacher_{k}':float(v[i]) for k,v in tf.items()});row.update({f'delta_{k}':float(tf[k][i]-v[i]) for k,v in bf.items()});row.update({k:float(v[i]) for k,v in geom.items()});row.update({k:int(v[i]) for k,v in flags.items()})
  records.append(row)
 csvout(O/f'{pop.upper()}_SCENE_GEOMETRY.csv',records)
 group=[]
 for c in range(9):
  m=choice==c
  row=dict(population=pop,choice=c,slot=c-1,method=METHODS[c],n=int(m.sum()),share_pct=float(m.mean()*100))
  for k in bf:
   for prefix,a in [('Base',bf[k]),('teacher',tf[k]),('delta',tf[k]-bf[k])]:
    row[f'{prefix}_{k}']=float(a[m].mean()) if m.any() else None
  for k,a in geom.items():
   for kk,v in describe(a[m]).items():row[k+'_'+kk]=v
  row['gain_points_mean']=float(gain[m].mean()) if m.any() else None
  for k,a in flags.items():
   row[k+'_pct']=float(a[m].mean()*100) if m.any() else None
   row[k+'_enrichment']=float(a[m].mean()/a.mean()) if m.any() and a.mean()>0 else None
  group.append(row)
 csvout(O/f'{pop.upper()}_SELECTED_GROUPS.csv',group)
 result=dict(population=pop,n=n,selected_repair_n=int((choice>0).sum()),geometry={k:describe(a[choice>0]) for k,a in geom.items()},group_profiles=group,associations={k:dict(repaired_gain_pearson=corr(a[choice>0],gain[choice>0])) for k,a in geom.items()},selection_caveat='Profiles conditional on oracle choice, not independent causal/semantic clusters.')
 # Geometry-size dependence on actual outcome, repaired winners only.
 bins=[(0,.1),(.1,.5),(.5,1),(1,2),(2,5),(5,float('inf'))]
 size=[]
 for lo,hi in bins:
  m=(choice>0)&(geom['ADE_m']>=lo)&(geom['ADE_m']<hi)
  size.append(dict(population=pop,ADE_lower_m=lo,ADE_upper_m=None if not np.isfinite(hi) else hi,n=int(m.sum()),repaired_pct=float(m.sum()/max((choice>0).sum(),1)*100),gain_points_mean=float(gain[m].mean()) if m.any() else None,Base_PDMS=float(q[m,0].mean()) if m.any() else None,teacher_PDMS=float(q[np.flatnonzero(m),choice[m]].mean()) if m.any() else None))
 csvout(O/f'{pop.upper()}_GEOMETRY_GAIN_BINS.csv',size);result['geometry_gain_bins']=size
 return result,records,bf

def all_test_effects(data,scores,selection):
 t=data['trajectories'];f=features(t);bf={k:v[:,0] for k,v in f.items()};q=scores[:,:,:,0].mean(2)*100
 delta=t[...,:2].astype(np.float64)-t[:,0:1,...,:2].astype(np.float64);dist=np.linalg.norm(delta,axis=-1)
 rows=[];per_scene=[]
 for c in range(1,9):
  gain=q[:,c]-q[:,0]
  row=dict(choice=c,slot=c-1,method=METHODS[c],n=len(q),PDMS_mean=float(q[:,c].mean()),PDMS_delta_vs_Base=float(gain.mean()),gain_positive_pct=float((gain>1e-9).mean()*100),gain_negative_pct=float((gain< -1e-9).mean()*100),ADE_mean_m=float(dist[:,c].mean()),FDE_mean_m=float(dist[:,c,:,-1].mean()),endpoint_dx_mean=float(delta[:,c,:,-1,0].mean()),endpoint_dy_mean=float(delta[:,c,:,-1,1].mean()),endpoint_dx_positive_pct=float((delta[:,c,:,-1,0].mean(1)>.1).mean()*100),endpoint_dx_negative_pct=float((delta[:,c,:,-1,0].mean(1)<-.1).mean()*100))
  for k,v in f.items():row['delta_'+k]=float((v[:,c]-v[:,0]).mean())
  rows.append(row)
  for lo,hi,label in [(0,2,'stalled_path_lt2m'),(2,8,'path_2_to8m'),(8,20,'path_8_to20m'),(20,40,'path_20_to40m'),(40,float('inf'),'path_ge40m')]:
   m=(bf['path_m'].mean(1)>=lo)&(bf['path_m'].mean(1)<hi)
   if not m.any():continue
   per_scene.append(dict(choice=c,slot=c-1,method=METHODS[c],Base_plan_bucket=label,n=int(m.sum()),gain_points_mean=float(gain[m].mean()),endpoint_dx_mean=float(delta[m,c,:,-1,0].mean()),endpoint_dy_mean=float(delta[m,c,:,-1,1].mean()),delta_path_m=float((f['path_m'][m,c]-f['path_m'][m,0]).mean()),win_pct=float((selection['choice'][m]==c).mean()*100)))
 csvout(O/'TEST_ALL_SCENE_INTERVENTIONS.csv',rows);csvout(O/'TEST_ALL_SCENE_INTERVENTIONS_BY_BASE_PLAN.csv',per_scene)
 return rows

def plots(test,selection,scores):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 choice=selection['choice'];q=scores[:,:,:,0].mean(2)*100;gain=q[np.arange(len(q)),choice]-q[:,0]
 # Cases chosen by fixed interpretable order statistics; diagnostic examples only.
 examples=[]
 for c in range(1,9):
  m=np.flatnonzero(choice==c)
  if not len(m):continue
  gi=gain[m];target=np.median(gi);i=int(m[np.argmin(np.abs(gi-target))]);examples.append((f'choice{c}_median_selected_gain',i,c))
 # Largest harmful outcome per method applied unconditionally, for clear non-Oracle contrast.
 for c in [1,6,8]:
  dg=q[:,c]-q[:,0];i=int(np.argmin(dg));examples.append((f'choice{c}_max_harm_unconditional',i,c))
 cases=[];fig,axs=plt.subplots(3,4,figsize=(15,12));axs=axs.ravel()
 for ax,(label,i,c) in zip(axs,examples):
  for s in range(5):
   b=test['trajectories'][i,0,s];a=test['trajectories'][i,c,s]
   ax.plot(np.r_[0,b[:,1]],np.r_[0,b[:,0]],color='#43576a',alpha=.25,lw=1)
   ax.plot(np.r_[0,a[:,1]],np.r_[0,a[:,0]],color='#188e78',alpha=.25,lw=1)
  b=test['trajectories'][i,0,:,:2].mean(0);a=test['trajectories'][i,c,:,:2].mean(0)
  ax.plot(np.r_[0,b[:,1]],np.r_[0,b[:,0]],'o-',color='#43576a',lw=2,label='Base mean')
  ax.plot(np.r_[0,a[:,1]],np.r_[0,a[:,0]],'o-',color='#188e78',lw=2,label='repair mean')
  ax.set_title(f"slot {c-1} {METHODS[c]}\nBase {q[i,0]:.2f} -> repair {q[i,c]:.2f}",fontsize=10)
  ax.set_xlabel('lateral y [m]');ax.set_ylabel('forward x [m]');ax.invert_xaxis();ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
  cases.append(dict(rule=label,index=i,token=str(test['tokens'][i]),choice=c,selected_choice=int(choice[i]),Base_PDMS=q[i,0],repair_PDMS=q[i,c],trajectories=test['trajectories'][i,[0,c]].tolist(),scores=scores[i,[0,c]].tolist()))
 for ax in axs[len(examples):]:ax.axis('off')
 axs[0].legend(fontsize=7);fig.suptitle('Actual native trajectories: diagnostic order-statistic examples, S0-S4',fontsize=12);fig.tight_layout();fig.savefig(O/'REPRESENTATIVE_TRAJECTORIES.png',dpi=170);fig.savefig(O/'REPRESENTATIVE_TRAJECTORIES.pdf');plt.close(fig)
 js(O/'REPRESENTATIVE_TRAJECTORIES.json',dict(selection='Per-choice median selected gain + most harmful unconditional chosen method; not population random sample',cases=cases))

def main():
 started=time.monotonic();sources=[]
 def load(p):
  sources.append(dict(path=str(p),sha256=sha(p)))
  with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
 train_sel=load(N/'frozen_teacher/selections/train.npz');test_sel=load(N/'frozen_teacher/selections/test.npz')
 train_score=load(N/'requests/train/MATRIX.npz');test_score=load(N/'requests/test/MATRIX.npz')
 test=load(N/'engineering/candidate_pool_analysis_20261028/recogdrive_nine/CANDIDATES.npz')
 assert np.array_equal(test['tokens'],test_sel['tokens']);assert np.array_equal(test['scores'],test_score['scores'])
 train=extract_train(train_sel)
 small_test=dict(tokens=test['tokens'],trajectories=np.stack([test['trajectories'][:,0],test['trajectories'][np.arange(len(test_sel['choice'])),test_sel['choice']]],axis=1))
 assert np.array_equal(train['indices'],train_score['indices']);assert np.array_equal(test['indices'],test_score['indices'])
 tr,tr_rows,tr_b=analyse_pop('train',train,train_sel,train_score['scores'])
 te,te_rows,te_b=analyse_pop('test',small_test,test_sel,test_score['scores'])
 all_effects=all_test_effects(test,test_score['scores'],test_sel)
 js(O/'SUMMARY.json',dict(train=tr,test=te,test_all_scene_interventions=all_effects,noise_aggregation='paired each S0-S4 then equal-weight scene mean; five noises are not independent scenes',waypoint_geometry='raw native predicted waypoints, not closed-loop PDM simulated ego states',posthoc_choice='True-PDMS oracle fixed choice; profiles are descriptive not learned semantic abilities',dt_seconds=.5))
 plots(test,test_sel,test_score['scores'])
 js(O/'PROVENANCE.json',dict(status='COMPLETE',script_sha256=sha(__file__),sources=sources,elapsed_seconds=time.monotonic()-started,training_or_scoring_executed=False,full_train_scenes=len(train_sel['tokens']),full_test_scenes=len(test_sel['tokens'])))
 print(json.dumps(dict(status='COMPLETE',train=len(train_sel['tokens']),test=len(test_sel['tokens']),elapsed_seconds=time.monotonic()-started)))

if __name__=='__main__':main()
